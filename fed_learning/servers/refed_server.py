"""
Re-Fed Server - Server for Retrieval-Enhanced Federated Incremental Learning.

Reference:
    Li et al., "Towards Efficient Replay in Federated Incremental Learning",
    CVPR 2024

Implements the server-side of Re-Fed:
- Standard FedAvg aggregation (modular)
- Distributes global model to clients
- Coordinates task transitions with PIM-based caching
- Tracks task state and seen classes
"""

import copy
import time
from collections import OrderedDict
from threading import Thread
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from .server import FederatedServer


class ReFedServer:
    """
    Server chuyên cho Re-Fed.

    Re-Fed's server is remarkably simple (paper's key advantage):
    - Standard FedAvg aggregation
    - No extra information transmitted beyond standard FL
    - The intelligence is in the client-side PIM caching

    The server:
    1. Manages global model and distributes it to clients
    2. Aggregates client updates (FedAvg)
    3. Coordinates task transitions
    4. Evaluates global model on test set
    """

    def __init__(self, clients, test_data: Dict, config: Dict):
        """Khởi tạo Re-Fed server, global model và các tham số điều phối PIM caching."""
        from ..models.cnn_gru import CNN_GRU_Model
        from ..strategies.fed_incremental.refed import ReFedTrainer, ReFedAggregator

        self.clients = clients
        self.test_data = test_data
        self.config = config
        self.num_classes = config["num_classes"]

        # Device setup
        self.num_gpus = config.get("num_gpus") or torch.cuda.device_count()
        if self.num_gpus == 0:
            self.num_gpus = 1
            self.primary_device = "cpu"
            self.use_cpu = True
        else:
            self.primary_device = "cuda:0"
            self.use_cpu = False

        device_info = "CPU" if self.use_cpu else f"{self.num_gpus} GPU(s)"
        print(f"\n  Re-Fed Server: {device_info}, primary: {self.primary_device}")

        # Global model
        self.global_model = CNN_GRU_Model(
            config["input_shape"], config["num_classes"]
        ).to(self.primary_device)

        # Re-Fed strategy
        self.trainer = ReFedTrainer(
            memory_size=config.get("refed_memory_size", 2000),
            lambda_pim=config.get("refed_lambda_pim", 0.5),
            pim_iterations=config.get("refed_pim_iterations", 5),
        )
        self.aggregator = ReFedAggregator()

        # Task tracking
        self.current_task: int = 0
        self.seen_classes: List[int] = []
        self.task_classes: Dict[int, list] = {}

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

        print(f"  Strategy: Re-Fed (FedAvg + PIM-based Sample Caching)")

    def get_global_params(self) -> OrderedDict:
        """Lấy snapshot tham số global model để gửi xuống client."""
        return OrderedDict(
            (k, v.cpu().clone()) for k, v in self.global_model.state_dict().items()
        )

    def set_global_params(self, params: OrderedDict):
        """Nạp tham số global mới sau bước aggregate."""
        self.global_model.load_state_dict(
            {k: v.to(self.primary_device) for k, v in params.items()}
        )

    def update_clients(self, clients) -> None:
        """Update the client list for a new task while preserving server state."""
        self.clients = clients

    def set_task(self, task_id: int, task_classes: list, seen_classes: list = None):
        """
        Đồng bộ server khi bắt đầu task mới và cập nhật các lớp đã thấy.
        """
        self.current_task = task_id
        self.task_classes[task_id] = task_classes

        if seen_classes is not None:
            self.seen_classes = list(seen_classes)
        else:
            self.seen_classes.extend(task_classes)

        print(f"\n  Task {task_id}: classes {task_classes}")
        print(f"   Total seen classes: {len(self.seen_classes)}")

    def coordinate_pim_caching(self, participating_clients=None, verbose: bool = True):
        """
        Điều phối bước PIM caching cho toàn bộ client trước khi train task mới.

        Đây là phần quan trọng nhất của Re-Fed ở phía server: yêu cầu từng client
        dùng global model hiện tại để chấm điểm và cache các mẫu replay quan trọng.
        """
        from ..models.cnn_gru import CNN_GRU_Model

        clients = participating_clients or self.clients

        if verbose:
            print(f"    Coordinating PIM caching for {len(clients)} clients...")

        global_params = self.get_global_params()

        for client in clients:
            if not hasattr(client, "update_cache_with_pim"):
                continue

            # Client needs a model on its device for PIM computation
            device = self.primary_device
            model = CNN_GRU_Model(
                self.config["input_shape"], self.config["num_classes"]
            ).to(device)
            model.load_state_dict({k: v.to(device) for k, v in global_params.items()})

            client.update_cache_with_pim(model, global_params, device)

            del model

        if not self.use_cpu:
            torch.cuda.empty_cache()

        if verbose:
            # Report cache statistics
            total_cached = 0
            for c in clients:
                if hasattr(c, "cached_y") and c.cached_y is not None:
                    total_cached += len(c.cached_y)
            avg_cached = total_cached / max(1, len(clients))
            print(f"    PIM caching complete. Avg cached per client: {avg_cached:.0f}")

    def train_round(self, participating_clients=None, verbose: bool = True) -> Dict:
        """
        Chạy một federated round của Re-Fed.

        Lưu ý: bước PIM caching không nằm trong hàm này mà cần gọi trước đó
        bằng `coordinate_pim_caching()` ở đầu task.
        """
        from ..training.refed_worker import train_refed_clients_on_gpu

        round_start = time.time()

        clients = participating_clients or self.clients

        if verbose:
            device_info = "CPU" if self.use_cpu else f"{self.num_gpus} GPU(s)"
            print(f"\n  Re-Fed: Training {len(clients)} clients on {device_info}")

        global_params = self.get_global_params()

        # Distribute clients across GPUs
        clients_per_gpu = [[] for _ in range(self.num_gpus)]
        for i, c in enumerate(clients):
            clients_per_gpu[i % self.num_gpus].append(c)

        # Train clients in parallel
        results_dict = {}
        threads = []

        for gpu_id in range(self.num_gpus):
            if len(clients_per_gpu[gpu_id]) > 0:
                t = Thread(
                    target=train_refed_clients_on_gpu,
                    args=(
                        gpu_id,
                        clients_per_gpu[gpu_id],
                        global_params,
                        self.config,
                        results_dict,
                        self.trainer,
                        self.use_cpu,
                    ),
                )
                threads.append(t)
                t.start()

        for t in threads:
            t.join()

        # Collect results
        results = list(results_dict.values())

        # Aggregate (FedAvg)
        new_params = self.aggregator.aggregate(results, global_params)
        self.set_global_params(new_params)

        avg_loss = float(np.mean([r["loss"] for r in results]))
        round_time = time.time() - round_start

        if verbose:
            print(f"\n  Train loss: {avg_loss:.4f}")
            print(f"  Round time: {round_time:.2f}s")

        return {"train_loss": avg_loss, "round_time": round_time}

    def evaluate_global(
        self,
        batch_size: int = 8192,
        compute_auc: bool = False,
        seen_classes_only: bool = True,
    ) -> Dict:
        """Đánh giá global model của Re-Fed trên test set hiện tại."""
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()

        X_test = self.test_data["X_test"]
        y_test = self.test_data["y_test"]

        # Filter to seen classes
        if seen_classes_only and self.seen_classes:
            seen_set = set(self.seen_classes)
            mask = torch.tensor([y.item() in seen_set for y in y_test])
            X_test = X_test[mask]
            y_test = y_test[mask]

        n_test = len(y_test)
        if n_test == 0:
            return {
                "loss": 0.0,
                "accuracy": 0.0,
                "f1_macro": 0.0,
                "f1_weighted": 0.0,
                "precision_macro": 0.0,
                "recall_macro": 0.0,
            }

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
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)

        metrics = {
            "loss": total_loss / n_test,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
        }

        return metrics
