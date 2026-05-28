"""
Federated Server with Multi-GPU Support and Strategy Pattern.
"""

import time
from collections import OrderedDict
from typing import Any, Dict, List, cast

import numpy as np
import torch
import torch.nn as nn
from threading import Thread
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from ..models.cnn_gru import CNN_GRU_Model
from ..clients.client import FederatedClient
from ..training.worker import train_clients_on_gpu
from ..strategies import get_strategy
from ..core import BaseTrainer, BaseAggregator


class FederatedServer:
    """
    Server FL cơ bản của project.

    Vai trò của class này:
    - giữ global model trên server
    - phát tham số global xuống các client ở đầu mỗi round
    - gọi worker để train client song song trên nhiều GPU
    - aggregate kết quả client thành global model mới
    - đánh giá global model trên test set
    """

    def __init__(self, clients: List[FederatedClient], test_data: Dict, config: Dict):
        """Khởi tạo server, model toàn cục, trainer và aggregator theo config."""
        self.clients = clients
        self.test_data = test_data
        self.config = config
        self.num_classes = config["num_classes"]

        # Detect GPUs
        self.num_gpus = config.get("num_gpus") or torch.cuda.device_count()
        if self.num_gpus == 0:
            self.num_gpus = 1
            self.primary_device = "cpu"
            self.use_cpu = True
        else:
            self.primary_device = "cuda:0"
            self.use_cpu = False

        device_info = "CPU" if self.use_cpu else f"{self.num_gpus} GPU(s)"
        print(f"\n🖥️  Detected {device_info}, primary device: {self.primary_device}")

        # Global model
        self.global_model = CNN_GRU_Model(
            config["input_shape"], config["num_classes"]
        ).to(self.primary_device)

        # Get strategy (trainer + aggregator)
        # Pass full config to ensure algorithm-specific params are passed
        self.trainer, self.aggregator = get_strategy(**config)
        print(
            f"📊 Strategy: {self.trainer.name} trainer + {self.aggregator.name} aggregator"
        )

        # History
        self.history = {
            "train_loss": [],
            "test_loss": [],
            "test_accuracy": [],
            "test_f1_macro": [],
            "test_f1_weighted": [],
            "test_precision_macro": [],
            "test_recall_macro": [],
        }

        # Paper Eq. 14: Local regularization info for CGoFed (passed to clients)
        self._local_reg_info: Dict = {}

    def update_clients(self, clients: List) -> None:
        """Update the client list for a new task. Preserves server state."""
        self.clients = clients

    def get_global_params(self) -> OrderedDict:
        """Lấy snapshot tham số của global model và đưa về CPU để phát cho client."""
        return OrderedDict(
            (k, v.cpu().clone()) for k, v in self.global_model.state_dict().items()
        )

    def set_global_params(self, params: OrderedDict):
        """Nạp bộ tham số mới vào global model sau khi aggregate xong."""
        self.global_model.load_state_dict(
            {k: v.to(self.primary_device) for k, v in params.items()}
        )

    def train_round(self, verbose: bool = True) -> Dict:
        """
        Chạy một federated round hoàn chỉnh.

        Luồng chính:
        - lấy `global_params`
        - chia client theo GPU
        - gọi worker để train local song song
        - thu kết quả từ client
        - aggregate thành global model mới
        """
        round_start = time.time()

        if verbose:
            device_info = "CPU" if self.use_cpu else f"{self.num_gpus} GPU(s)"
            print(
                f"\n→ {self.config['algorithm'].upper()}: Training {len(self.clients)} clients on {device_info}"
            )

        global_params = self.get_global_params()

        # Distribute clients across GPUs
        clients_per_gpu = [[] for _ in range(self.num_gpus)]
        for i, c in enumerate(self.clients):
            clients_per_gpu[i % self.num_gpus].append(c)

        if verbose:
            for gpu_id, clients in enumerate(clients_per_gpu):
                device_label = "CPU" if self.use_cpu else f"GPU {gpu_id}"
                print(f"   {device_label}: {len(clients)} clients")

        # Shared results dict
        results_dict = {}

        # Create threads for each GPU
        threads = []
        for gpu_id in range(self.num_gpus):
            if len(clients_per_gpu[gpu_id]) > 0:
                t = Thread(
                    target=train_clients_on_gpu,
                    args=(
                        gpu_id,
                        clients_per_gpu[gpu_id],
                        global_params,
                        self.config,
                        results_dict,
                        self.trainer,
                        self.use_cpu,
                        self._local_reg_info,  # Paper Eq. 14: Pass local regularization info
                    ),
                )
                threads.append(t)
                t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Collect results - use values() since client_ids may not be sequential
        results = list(results_dict.values())

        # Aggregate using strategy
        new_params = self.aggregator.aggregate(results, global_params)
        self.set_global_params(new_params)

        # Paper Eq. 14: Get local regularization info for CGoFed to pass to clients
        if hasattr(self.aggregator, "get_local_regularization_info"):
            self._local_reg_info = cast(
                Any, self.aggregator
            ).get_local_regularization_info()

        avg_loss = float(np.mean([r["loss"] for r in results]))
        round_time = time.time() - round_start

        if verbose:
            print(f"\n→ Train loss: {avg_loss:.4f}")
            print(f"→ Round time: {round_time:.2f}s")

        return {"train_loss": avg_loss, "round_time": round_time}

    def evaluate_global(self, batch_size: int = 8192) -> Dict:
        """Đánh giá global model trên toàn bộ test set và trả về các metric chính."""
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()

        X_test = self.test_data["X_test"]
        y_test = self.test_data["y_test"]
        n_test = len(y_test)

        all_preds = []
        all_targets = []
        total_loss = 0.0

        with torch.no_grad():
            for i in range(0, n_test, batch_size):
                X_batch = X_test[i : i + batch_size].to(self.primary_device)
                y_batch = y_test[i : i + batch_size].to(self.primary_device)

                out = self.global_model(X_batch)
                loss = criterion(out, y_batch)
                total_loss += loss.item() * len(y_batch)

                preds = out.argmax(dim=1)

                all_preds.extend(preds.detach().cpu().tolist())
                all_targets.extend(y_batch.detach().cpu().tolist())

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        zero_division: Any = 0

        metrics = {
            "loss": total_loss / n_test,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(
                y_true, y_pred, average="macro", zero_division=zero_division
            ),
            "recall_macro": recall_score(
                y_true, y_pred, average="macro", zero_division=zero_division
            ),
            "f1_macro": f1_score(
                y_true, y_pred, average="macro", zero_division=zero_division
            ),
            "f1_weighted": f1_score(
                y_true, y_pred, average="weighted", zero_division=zero_division
            ),
        }

        return metrics
