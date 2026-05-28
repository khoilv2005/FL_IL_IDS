"""
FedLwF Server - Server for Federated Learning without Forgetting.
"""

from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from threading import Thread

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from .server import FederatedServer


class FedLwFServer:
    """
    Server chuyên cho FedLwF.

    Server này vẫn aggregate như FedAvg, nhưng có thêm trách nhiệm
    lưu snapshot global model sau mỗi task để các client dùng làm teacher.
    """

    def __init__(self, clients, test_data: Dict, config: Dict):
        """Khởi tạo global model, trainer LwF, aggregator và state theo task."""
        from ..models.cnn_gru import CNN_GRU_Model
        from ..strategies.fed_incremental.fedlwf import FedLwFTrainer, FedLwFAggregator

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
        print(f"\n🖥️  FedLwF Server: {device_info}, primary: {self.primary_device}")

        # Global model
        self.global_model = CNN_GRU_Model(
            config["input_shape"], config["num_classes"]
        ).to(self.primary_device)

        # FedLwF strategy
        self.trainer = FedLwFTrainer(
            lwf_alpha=config.get("lwf_alpha", 1.0),
            temperature=config.get("temperature", 2.0),
            distill_old_classes_only=config.get("distill_old_classes_only", False),
            temp_dir=config.get("temp_dir", "./temp_fedlwf_storage"),
        )
        self.aggregator = FedLwFAggregator()

        # Task tracking
        self.current_task: int = 0
        self.seen_classes = []
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
            "task_accuracies": [],
            "average_forgetting": [],
        }

        print(f"📊 Strategy: FedLwF (FedAvg + Knowledge Distillation)")
        print(f"   α={self.trainer.lwf_alpha}, T={self.trainer.temperature}")

    def get_global_params(self) -> OrderedDict:
        """Lấy tham số global hiện tại để phát xuống client ở đầu round."""
        return OrderedDict(
            (k, v.cpu().clone()) for k, v in self.global_model.state_dict().items()
        )

    def set_global_params(self, params: OrderedDict):
        """Cập nhật global model từ kết quả aggregate của round hiện tại."""
        self.global_model.load_state_dict(
            {k: v.to(self.primary_device) for k, v in params.items()}
        )

    def set_task(self, task_id: int, task_classes: list, seen_classes: list = None):
        """
        Đồng bộ task hiện tại và danh sách class đã thấy.
        """
        self.current_task = task_id
        self.task_classes[task_id] = task_classes

        # CRITICAL: Use explicit seen_classes list if provided (prevents reset bug
        # when server is recreated each task)
        if seen_classes is not None:
            self.seen_classes = list(seen_classes)
        else:
            self.seen_classes.extend(task_classes)

        print(f"\n📌 Task {task_id}: classes {task_classes}")
        print(f"   Total seen classes: {len(self.seen_classes)}")

    def update_clients(self, clients) -> None:
        """Update the client list for a new task. Preserves server state."""
        self.clients = clients

    def save_global_snapshot(self):
        """Lưu snapshot global model sau task để client dùng làm teacher ở task sau."""
        # Save in trainer (will be distributed via config)
        self.trainer.save_model_snapshot(self.global_model)

        # Also save snapshot to all clients
        global_state = self.get_global_params()
        for client in self.clients:
            client.old_model_state = OrderedDict(
                (k, v.clone()) for k, v in global_state.items()
            )
            client.old_model = None  # Clear cache, will reload

    def train_round(self, participating_clients=None, verbose: bool = True) -> Dict:
        """Chạy một round FedLwF: train local có distillation rồi aggregate như FedAvg."""
        import time
        from ..training.fedlwf_worker import train_fedlwf_clients_on_gpu

        round_start = time.time()

        clients = participating_clients or self.clients

        if verbose:
            device_info = "CPU" if self.use_cpu else f"{self.num_gpus} GPU(s)"
            print(f"\n→ FedLwF: Training {len(clients)} clients on {device_info}")

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
                    target=train_fedlwf_clients_on_gpu,
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
            print(f"\n→ Train loss: {avg_loss:.4f}")
            print(f"→ Round time: {round_time:.2f}s")

        return {"train_loss": avg_loss, "round_time": round_time}

    def evaluate_global(
        self,
        batch_size: int = 8192,
        compute_auc: bool = False,
        seen_classes_only: bool = True,
    ) -> Dict:
        """Đánh giá global model, có thể lọc theo `seen_classes` cho setting incremental."""
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()

        X_test = self.test_data["X_test"]
        y_test = self.test_data["y_test"]

        # Filter to seen classes if requested
        if seen_classes_only and self.seen_classes:
            seen_set = set(self.seen_classes)
            mask = torch.tensor([y.item() in seen_set for y in y_test])
            X_test = X_test[mask]
            y_test = y_test[mask]

        n_test = len(y_test)
        if n_test == 0:
            return {"accuracy": 0.0, "f1_macro": 0.0}

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
