"""
DER Server - Server for Dynamically Expandable Representation FCIL.

Reference:
    "DER: Dynamically Expandable Representation for Class Incremental Learning"
    Yan, Xie, He (CVPR 2021)

Inherits IncrementalServer:
    - evaluate_global(seen_classes_only=True)  → works because DERModel.forward() returns logits
    - set_task() base logic (task tracking, seen_classes)
    - get_global_params() / set_global_params()
    - Device setup (num_gpus, primary_device, use_cpu)

Overrides:
    - __init__(): Creates DERModel instead of CNN_GRU_Model
    - set_task(): Adds model expansion (add_task) + aggregator key update
    - train_round(): Two-stage support + uses der_worker
"""

import time
from collections import OrderedDict
from typing import Dict, List, Optional
from threading import Thread

from ..training.der_worker import _reconstruct_model_structure

import numpy as np
import torch

from .incremental_server import IncrementalServer


class DERServer(IncrementalServer):
    """
    Server chuyên cho DER với model mở rộng theo task và train hai giai đoạn.

    Inherits from IncrementalServer → FederatedServer:
        - evaluate_global(): Uses self.global_model(X_batch) which works
          with DERModel since DERModel.forward() returns logits [B, num_classes]
        - get_global_params() / set_global_params(): Works with any nn.Module
        - Device detection and GPU setup
        - seen_classes / task_classes tracking

    DER-specific additions:
        - Model expansion at each set_task() via DERModel.add_task()
        - Two-stage train_round(stage=1|2)
        - Exemplar buffer coordination after each task
        - Task classes history for worker model reconstruction
    """

    def __init__(self, clients, test_data: Dict, config: Dict):
        """
        Khởi tạo DER server.

        Bước quan trọng nhất ở đây là thay global model mặc định bằng `DERModel`
        để server có thể quản lý các extractor được thêm mới theo task.
        """
        # Parent handles: clients, test_data, config, num_classes,
        # GPU detection, history, seen_classes, task_classes
        super().__init__(clients, test_data, config)

        # Replace global_model with DERModel
        from ..models.der_model import DERModel

        del self.global_model
        self.global_model = DERModel(config["input_shape"], config["num_classes"]).to(
            self.primary_device
        )

        # Task classes history (needed by der_worker for model reconstruction)
        self._task_classes_history: Dict[int, List[int]] = {}

        print(f"📊 Strategy: DER (Dynamically Expandable Representation)")

    def set_global_params(self, params: OrderedDict):
        """
        Nạp state dict vào DERModel theo cách an toàn.

        Nếu số extractor hiện tại chưa khớp với state dict nhận được,
        server sẽ dựng lại đúng cấu trúc model rồi mới load tham số.
        """
        # Count how many extractors are in params
        extractor_indices = {
            int(k.split(".")[1])
            for k in params
            if k.startswith("extractors.") and k.split(".")[1].isdigit()
        }
        num_tasks_in_params = len(extractor_indices)

        # Reconstruct model structure if it doesn't match params
        if self.global_model.num_extractors != num_tasks_in_params:
            config_for_recon = {
                "input_shape": self.config["input_shape"],
                "num_classes": self.config["num_classes"],
                "s_max": self.config.get("s_max", 15.0),
                "task_classes_history": self._task_classes_history,
            }
            _reconstruct_model_structure(
                self.global_model, OrderedDict(params), config_for_recon
            )

        # Now safe to load
        self.global_model.load_state_dict(
            {k: v.to(self.primary_device) for k, v in params.items()}
        )

    def set_task(self, task_id: int, task_classes: list, seen_classes: list = None):
        """
        Chuẩn bị DER cho task mới.

        Ngoài phần tracking task/class của server cha, hàm này còn:
        - lưu lịch sử class để worker reconstruct model đúng cấu trúc
        - gọi `add_task()` để mở rộng DERModel
        - cập nhật trainable keys cho aggregator
        """
        # Parent handles: task tracking, seen_classes, task_classes dict, print
        super().set_task(task_id, task_classes, seen_classes)

        # Store for worker reconstruction
        self._task_classes_history[task_id] = list(task_classes)

        # Expand DERModel: add new extractor, expand classifier, create aux
        s_max = self.config.get("s_max", 15.0)
        self.global_model.add_task(task_classes, s_max=s_max)

        # Update aggregator with current trainable param keys
        if hasattr(self.aggregator, "set_trainable_keys"):
            trainable_keys = [
                k for k, p in self.global_model.named_parameters() if p.requires_grad
            ]
            self.aggregator.set_trainable_keys(trainable_keys)

    def train_round(
        self,
        participating_clients=None,
        stage: int = 1,
        verbose: bool = True,
    ) -> Dict:
        """
        Chạy một federated round của DER.

        DER cần biết đang ở stage nào để worker và client train đúng nhánh:
        - stage 1: học representation mới
        - stage 2: fine-tune classifier
        """
        from ..training.der_worker import train_der_clients_on_gpu

        round_start = time.time()

        clients = participating_clients or self.clients

        if verbose:
            stage_name = "Representation" if stage == 1 else "Classifier"
            device_info = "CPU" if self.use_cpu else f"{self.num_gpus} GPU(s)"
            print(
                f"\n→ DER Stage {stage} ({stage_name}): "
                f"Training {len(clients)} clients on {device_info}"
            )

        global_params = self.get_global_params()

        # Prepare config with task history for worker model reconstruction
        worker_config = {**self.config}
        worker_config["task_classes_history"] = self._task_classes_history

        # Distribute clients across GPUs
        clients_per_gpu = [[] for _ in range(self.num_gpus)]
        for i, c in enumerate(clients):
            clients_per_gpu[i % self.num_gpus].append(c)

        # Train clients in parallel threads
        results_dict = {}
        threads = []

        for gpu_id in range(self.num_gpus):
            if len(clients_per_gpu[gpu_id]) > 0:
                t = Thread(
                    target=train_der_clients_on_gpu,
                    args=(
                        gpu_id,
                        clients_per_gpu[gpu_id],
                        global_params,
                        worker_config,
                        results_dict,
                        self.trainer,
                        self.use_cpu,
                        stage,
                    ),
                )
                threads.append(t)
                t.start()

        for t in threads:
            t.join()

        # Collect and aggregate results
        results = list(results_dict.values())

        new_params = self.aggregator.aggregate(results, global_params)
        self.set_global_params(new_params)

        avg_loss = float(np.mean([r["loss"] for r in results]))
        round_time = time.time() - round_start

        if verbose:
            print(f"  → Stage {stage} loss: {avg_loss:.4f} ({round_time:.1f}s)")

            # Print replay stats if available
            total_replay = sum(r.get("replay_samples", 0) for r in results)
            if total_replay > 0:
                print(f"  → Replay samples: {total_replay}")

            # Print mask stats for Stage 1
            if stage == 1 and hasattr(self.global_model, "get_mask_stats"):
                stats = self.global_model.get_mask_stats()
                for k, v in stats.items():
                    print(f"  → {k}: {v:.2%}")

        return {"train_loss": avg_loss, "round_time": round_time}

    def coordinate_exemplar_update(
        self,
        participating_clients=None,
        verbose: bool = True,
    ):
        """
        Cập nhật exemplar buffer cho các client sau khi task kết thúc.

        Đây là bước DER dùng để chuẩn bị replay memory cho task tiếp theo.
        """
        clients = participating_clients or self.clients

        if verbose:
            print(f"\n📸 DER: Updating exemplar buffers for {len(clients)} clients")

        for idx, client in enumerate(clients):
            if not hasattr(client, "update_exemplars"):
                continue
            if client.num_samples == 0:
                continue

            client.update_exemplars(self.global_model)

        if verbose:
            total_buffer = sum(
                c.replay_buffer.total_samples
                for c in clients
                if hasattr(c, "replay_buffer")
            )
            print(f"   Exemplar update complete. Total buffer: {total_buffer}")

    def compute_average_forgetting(self) -> float:
        """Tính Average Forgetting của DER dựa trên accuracy từng task."""
        if self.current_task == 0:
            return 0.0
        current_accs = self.evaluate_per_task()
        if hasattr(self.trainer, "update_forgetting"):
            self.trainer.update_forgetting(current_accs)
            return self.trainer.last_af
        return 0.0

    def evaluate_per_task(self, batch_size: int = 8192) -> Dict[int, float]:
        """
        Đánh giá accuracy riêng cho từng task đã học.

        Hàm này hữu ích khi cần tính forgetting hoặc phân tích task-wise performance.
        """
        from sklearn.metrics import accuracy_score

        self.global_model.eval()
        task_accuracies = {}

        X_test = self.test_data["X_test"]
        y_test = self.test_data["y_test"]

        for task_id, task_classes in self.task_classes.items():
            if not task_classes:
                continue

            task_class_set = set(task_classes)
            mask = torch.tensor([y.item() in task_class_set for y in y_test])

            if not mask.any():
                task_accuracies[task_id] = 0.0
                continue

            X_task = X_test[mask]
            y_task = y_test[mask]

            all_preds = []
            all_targets = []

            with torch.no_grad():
                for i in range(0, len(y_task), batch_size):
                    X_batch = X_task[i : i + batch_size].to(self.primary_device)
                    y_batch = y_task[i : i + batch_size]

                    out = self.global_model(X_batch)
                    preds = out.argmax(dim=1)
                    all_preds.extend(preds.detach().cpu().tolist())
                    all_targets.extend(y_batch.detach().cpu().tolist())

            task_accuracies[task_id] = accuracy_score(all_targets, all_preds)

        return task_accuracies
