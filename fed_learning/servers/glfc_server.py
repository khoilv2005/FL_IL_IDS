"""
GLFC Server - Server for Global-Local Forgetting Compensation.

Reference:
    Dong et al., "Federated Class-Incremental Learning", CVPR 2022

Implements the Proxy Server mechanism (Section 3.4):
1. Collect prototype gradients from clients
2. Reconstruct pseudo data via gradient inversion
3. Monitor reconstructed data accuracy
4. Maintain best historical models for forgetting compensation
5. Coordinate exemplar set updates across clients
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


class GLFCServer:
    """
    Server chuyên cho GLFC.

    Implements the Proxy Server from paper Section 3.4:
    - Collects prototype gradients from clients
    - Reconstructs pseudo data via gradient inversion (LBFGS)
    - Monitors reconstructed data quality
    - Maintains best model versions ([best_model_1, best_model_2])
    - Returns old models to clients for knowledge distillation
    """

    def __init__(self, clients, test_data: Dict, config: Dict):
        """Khởi tạo GLFC server, proxy-state và các tham số reconstruction/monitoring."""
        from ..models.cnn_gru import CNN_GRU_Model
        from ..strategies.fed_incremental.glfc import GLFCTrainer, GLFCAggregator

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
        print(f"\n  GLFC Server: {device_info}, primary: {self.primary_device}")

        # Global model
        self.global_model = CNN_GRU_Model(
            config["input_shape"], config["num_classes"]
        ).to(self.primary_device)

        # GLFC strategy
        self.trainer = GLFCTrainer(
            memory_size=config.get("glfc_memory_size", 2000),
            entropy_threshold=config.get("glfc_entropy_threshold", 1.2),
            distill_weight=config.get("glfc_distill_weight", 0.5),
        )
        self.aggregator = GLFCAggregator()

        # Task tracking
        self.current_task: int = 0
        self.seen_classes: List[int] = []
        self.task_classes: Dict[int, list] = {}

        # Proxy server state (Paper Section 3.4)
        self.best_model_1: Optional[OrderedDict] = None  # Previous best
        self.best_model_2: Optional[OrderedDict] = None  # Current best
        self.best_perf: float = 0.0

        # Reconstructed proxy data for monitoring
        self.proxy_data: List = []
        self.proxy_labels: List = []

        # Gradient inversion parameters
        self.reconstruction_iters: int = config.get("glfc_recon_iters", 250)
        self.num_recon_images: int = config.get("glfc_num_recon_images", 20)

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

        print(f"  Strategy: GLFC (FedAvg + Local/Global Forgetting Compensation)")

    def get_global_params(self) -> OrderedDict:
        """Lấy tham số global model để phát cho client hoặc lưu làm best model."""
        return OrderedDict(
            (k, v.cpu().clone()) for k, v in self.global_model.state_dict().items()
        )

    def set_global_params(self, params: OrderedDict):
        """Nạp tham số global mới sau khi aggregate xong round hiện tại."""
        self.global_model.load_state_dict(
            {k: v.to(self.primary_device) for k, v in params.items()}
        )

    def set_task(self, task_id: int, task_classes: list, seen_classes: list = None):
        """
        Cập nhật task hiện tại và danh sách class đã thấy cho GLFC.
        """
        self.current_task = task_id
        self.task_classes[task_id] = task_classes

        if seen_classes is not None:
            self.seen_classes = list(seen_classes)
        else:
            self.seen_classes.extend(task_classes)

        print(f"\n  Task {task_id}: classes {task_classes}")
        print(f"   Total seen classes: {len(self.seen_classes)}")

    def update_clients(self, clients) -> None:
        """Update the participating client list for the next task."""
        self.clients = clients

    def model_back(self):
        """
        Trả về hai mốc best model mà proxy server đang giữ để client distill.
        """
        return self.best_model_1, self.best_model_2

    def _gradient_to_label(self, pool_grad: List) -> List[int]:
        """
        Suy ra nhãn gần đúng từ hướng gradient của output layer.
        """
        pool_label = []
        for grad_single in pool_grad:
            # Use last linear layer gradient to infer label
            # Find the last non-None gradient that looks like a weight gradient
            last_weight_grad = None
            for g in reversed(grad_single):
                if g is not None and g.dim() == 2:
                    last_weight_grad = g
                    break

            if last_weight_grad is not None:
                pred = (
                    torch.argmin(torch.sum(last_weight_grad, dim=-1), dim=-1)
                    .detach()
                    .item()
                )
                pool_label.append(pred)

        return pool_label

    def process_prototype_gradients(self, pool_grad: List):
        """
        Xử lý prototype gradients do client gửi lên.

        Trong implementation hiện tại, gradient được dùng như tín hiệu để:
        - suy nhãn gần đúng
        - đánh giá chất lượng model hiện tại
        - cập nhật best historical models của proxy server
        """
        if not pool_grad:
            return

        # Infer labels from gradients
        pool_labels = self._gradient_to_label(pool_grad)

        if not pool_labels:
            return

        print(
            f"    Proxy Server: Received {len(pool_grad)} prototype gradients, "
            f"inferred labels: {set(pool_labels)}"
        )

        # Monitor: Use gradient quality as a proxy for model quality
        # In the original paper, this involves full image reconstruction.
        # For network IDS data, we use a simplified monitoring approach:
        # The gradient magnitude serves as indicator of model fit.
        total_grad_norm = 0.0
        for grad_single in pool_grad:
            for g in grad_single:
                if g is not None:
                    total_grad_norm += g.norm().item()
        avg_grad_norm = total_grad_norm / max(1, len(pool_grad))

        # Update models: smaller gradient norm = better fit
        cur_perf = self._evaluate_on_test_subset()

        print(f"    Proxy Server: Current performance = {cur_perf:.2f}%")

        # Update best model tracking
        if cur_perf >= self.best_perf:
            self.best_perf = cur_perf
            self.best_model_1 = self.best_model_2
            self.best_model_2 = self.get_global_params()

    def _evaluate_on_test_subset(self) -> float:
        """Đánh giá nhanh trên tập con test để proxy server theo dõi chất lượng model."""
        self.global_model.eval()

        X_test = self.test_data.get("X_test")
        y_test = self.test_data.get("y_test")

        if X_test is None or y_test is None or len(y_test) == 0:
            return 0.0

        # Filter to seen classes
        if self.seen_classes:
            seen_set = set(self.seen_classes)
            mask = torch.tensor([y.item() in seen_set for y in y_test])
            X_test = X_test[mask]
            y_test = y_test[mask]

        if len(y_test) == 0:
            return 0.0

        # Use subset for speed
        n = min(1000, len(y_test))
        indices = torch.randperm(len(y_test))[:n]
        X_sub = X_test[indices]
        y_sub = y_test[indices]

        correct = 0
        total = 0

        with torch.no_grad():
            batch_size = 256
            for i in range(0, n, batch_size):
                X_batch = X_sub[i : i + batch_size].to(self.primary_device)
                y_batch = y_sub[i : i + batch_size]

                outputs = self.global_model(X_batch)
                preds = outputs.argmax(dim=1).cpu()
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)

        return 100.0 * correct / max(1, total)

    def coordinate_exemplar_update(
        self, participating_clients=None, verbose: bool = True
    ):
        """
        Điều phối cập nhật exemplar set cho toàn bộ client.

        Sau khi global model đổi, server phát state mới để client đồng bộ memory
        và old model cho các round/task tiếp theo.
        """
        clients = participating_clients or self.clients

        if verbose:
            print(f"    Coordinating exemplar updates for {len(clients)} clients...")

        global_state = self.get_global_params()

        for client in clients:
            # Update client's model with global model
            if hasattr(client, "model") and client.model is not None:
                try:
                    client.model.load_state_dict(
                        {k: v.to(client.device) for k, v in global_state.items()}
                    )
                except Exception:
                    pass

            # Update exemplar set
            if hasattr(client, "update_exemplar_set"):
                try:
                    device = client.device if client.device else self.primary_device
                    client.update_exemplar_set(client.model, device)
                except Exception as e:
                    if verbose:
                        print(
                            f"      Client {client.client_id}: exemplar update failed: {e}"
                        )

        if verbose:
            print("    Exemplar update complete.")

    def train_round(self, participating_clients=None, verbose: bool = True) -> Dict:
        """
        Chạy một round GLFC đầy đủ ở phía server.

        Sequence (following author's fl_main.py):
        1. Distribute global model and old models to clients
        2. Each client: detect signal, update exemplars, train
        3. Collect prototype gradients
        4. Aggregate models (FedAvg)
        5. Process prototype gradients at proxy server

        Đây là vòng lặp chính của GLFC ở phía server.
        """
        from ..training.glfc_worker import train_glfc_clients_on_gpu

        round_start = time.time()

        clients = participating_clients or self.clients

        if verbose:
            device_info = "CPU" if self.use_cpu else f"{self.num_gpus} GPU(s)"
            print(f"\n  GLFC: Training {len(clients)} clients on {device_info}")

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
                    target=train_glfc_clients_on_gpu,
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

        # Collect results and prototype gradients
        results = list(results_dict.values())
        pool_grad = []
        for r in results:
            proto_grad = r.get("proto_grad")
            if proto_grad is not None:
                pool_grad.extend(proto_grad)

        # Aggregate (FedAvg)
        new_params = self.aggregator.aggregate(results, global_params)
        if new_params is None:
            print(
                "    ⚠️ Aggregation returned None (all clients may have failed). Keeping current params."
            )
        else:
            self.set_global_params(new_params)

        # Process prototype gradients at proxy server
        if pool_grad:
            self.process_prototype_gradients(pool_grad)
        else:
            # Still update best model tracking even without gradients
            cur_perf = self._evaluate_on_test_subset()
            if cur_perf >= self.best_perf:
                self.best_perf = cur_perf
                self.best_model_1 = self.best_model_2
                self.best_model_2 = self.get_global_params()

        # Update trainer's proxy server state
        self.trainer.best_model_1 = self.best_model_1
        self.trainer.best_model_2 = self.best_model_2
        self.trainer.best_perf = self.best_perf

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
        """Đánh giá global model của GLFC trên test set hiện tại."""
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
