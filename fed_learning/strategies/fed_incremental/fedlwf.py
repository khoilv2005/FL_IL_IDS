"""
FedLwF Strategy - Federated Learning without Forgetting for Class-Incremental Learning.

Reference:
    Li & Hoiem, "Learning without Forgetting", ECCV 2016, IEEE TPAMI 2018

    Adapted for Federated Learning setting where:
    - Multiple clients learn incrementally
    - Knowledge distillation prevents forgetting across tasks
    - Server coordinates global model, clients use local old model snapshots

FedLwF = FedAvg + Knowledge Distillation (LwF)

Key Mechanism:
    L_total = L_CE(new_data) + α * T² * KL(σ(z_old/T) || σ(z_new/T))

Where:
    - L_CE: Cross-entropy loss on new task data
    - z_old: Logits from old (frozen) model
    - z_new: Logits from current model
    - T: Temperature for soft targets (default: 2.0)
    - α: Distillation weight (default: 1.0)
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple
import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import BaseTrainer, BaseAggregator


class FedLwFTrainer(BaseTrainer):
    """
    FedLwF Trainer - Federated Learning without Forgetting.

    Ý tưởng chính:
    - Sau mỗi task, lưu snapshot model cũ làm teacher.
    - Ở task mới, model hiện tại vừa học nhãn mới bằng CE,
      vừa bắt chước logit mềm của model cũ bằng distillation.
    - Nhờ đó giảm quên mà không cần replay dữ liệu cũ.

    Implements Knowledge Distillation to prevent catastrophic forgetting:
    1. Save model snapshot after each task (teacher)
    2. Train new model (student) to match teacher's soft outputs
    3. Combined loss: CE on new data + KD loss on all data

    Args:
        lwf_alpha: Weight for distillation loss (α), default 1.0
        temperature: Temperature for soft targets (T), default 2.0
        distill_old_classes_only: If True, restrict KD to old class logits only
                                  (more targeted: only preserves knowledge of past classes).
                                  If False (default), distill on all class logits.
        temp_dir: Directory to store old model snapshots
    """

    def __init__(
        self,
        lwf_alpha: float = 1.0,
        temperature: float = 2.0,
        distill_old_classes_only: bool = False,
        temp_dir: str = "./temp_fedlwf_storage",
        **kwargs,
    ):
        self.lwf_alpha = lwf_alpha
        self.temperature = temperature
        self.distill_old_classes_only = distill_old_classes_only
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)

        # Storage for old model per task
        # {task_id: model_state_dict (CPU)}
        self.old_model_states: Dict[int, OrderedDict] = {}

        # Cached old model for current training session
        self._cached_old_model: Optional[nn.Module] = None
        self._cached_device: Optional[str] = None

        # Task tracking
        self.current_task: int = 0
        self.seen_classes: Set[int] = set()
        self.old_classes: List[int] = []
        self.new_classes: List[int] = []

        # For compatibility with training scripts
        self.mu_coefficient: float = 1.0
        self.best_acc_per_task: Dict[int, float] = {}
        self.current_acc_per_task: Dict[int, float] = {}
        self.last_af: float = 0.0

    def set_task(self, task_id: int, new_classes: List[int]):
        """
        Được gọi khi bắt đầu task mới.

        Hàm này cập nhật:
        - danh sách class cũ và class mới
        - task hiện tại
        - cache teacher model để task mới load lại đúng snapshot
        """
        # Store old classes before updating
        self.old_classes = list(self.seen_classes)
        self.new_classes = new_classes

        self.current_task = task_id
        self.seen_classes.update(new_classes)

        # Invalidate cached model
        self._cached_old_model = None

        print(
            f"  FedLwF Task {task_id}: old_classes={len(self.old_classes)}, "
            f"new_classes={len(new_classes)}, α={self.lwf_alpha}, T={self.temperature}"
        )

    def save_model_snapshot(self, model: nn.Module):
        """
        Lưu snapshot model sau khi hoàn thành một task.

        Snapshot này sẽ đóng vai trò teacher cho các task sau,
        phục vụ tính distillation loss.
        """
        # Save state dict to CPU (memory efficient)
        state_dict = OrderedDict(
            (k, v.cpu().clone()) for k, v in model.state_dict().items()
        )
        self.old_model_states[self.current_task] = state_dict

        # Also save to disk for persistence
        path = os.path.join(self.temp_dir, f"task_{self.current_task}_model.pt")
        torch.save(state_dict, path)

        print(f"  📸 Saved model snapshot for Task {self.current_task}")

    def load_old_model(
        self, model_template: nn.Module, device: str
    ) -> Optional[nn.Module]:
        """
        Load model cũ để làm teacher cho knowledge distillation.

        Ưu tiên dùng cache trong RAM; nếu chưa có thì đọc snapshot của
        task trước từ bộ nhớ hoặc từ disk. Task đầu tiên sẽ trả về None.
        """
        if self.current_task == 0:
            return None

        prev_task = self.current_task - 1

        # Check cache
        if self._cached_old_model is not None and self._cached_device == device:
            return self._cached_old_model

        # Get state dict
        if prev_task in self.old_model_states:
            state_dict = self.old_model_states[prev_task]
        else:
            # Try loading from disk
            path = os.path.join(self.temp_dir, f"task_{prev_task}_model.pt")
            if os.path.exists(path):
                state_dict = torch.load(path, map_location="cpu")
            else:
                print(f"  ⚠️ No old model found for task {prev_task}")
                return None

        # Create old model from template
        try:
            old_model = copy.deepcopy(model_template)
            old_model.load_state_dict({k: v.to(device) for k, v in state_dict.items()})
            old_model.eval()

            # Freeze parameters
            for param in old_model.parameters():
                param.requires_grad = False

            # Cache
            self._cached_old_model = old_model
            self._cached_device = device

            return old_model

        except Exception as e:
            print(f"  ⚠️ Failed to load old model: {e}")
            return None

    def compute_distillation_loss(
        self,
        old_logits: torch.Tensor,
        new_logits: torch.Tensor,
        old_class_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Tính knowledge distillation loss.

        L_KD = T² * KL(σ(z_old/T) || σ(z_new/T))

        Nếu truyền `old_class_indices`, chỉ distill trên các logit của lớp cũ.
        """
        T = self.temperature

        if old_class_indices is not None and len(old_class_indices) > 0:
            # Only distill on old classes (more targeted)
            old_indices = torch.tensor(old_class_indices, device=old_logits.device)
            old_logits = old_logits[:, old_indices]
            new_logits = new_logits[:, old_indices]

        # Soft targets from old model
        old_probs = F.softmax(old_logits / T, dim=1)

        # Log-softmax from new model
        new_log_probs = F.log_softmax(new_logits / T, dim=1)

        # KL divergence: KL(P || Q) = Σ P(x) * log(P(x)/Q(x))
        # = Σ P(x) * (log P(x) - log Q(x))
        kd_loss = F.kl_div(new_log_probs, old_probs, reduction="batchmean")

        # Scale by T² (as per Hinton et al.)
        return (T**2) * kd_loss

    def compute_loss(
        self,
        model: nn.Module,
        output: torch.Tensor,
        target: torch.Tensor,
        global_params: Optional[OrderedDict] = None,
        inputs: Optional[torch.Tensor] = None,
        old_model: Optional[nn.Module] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Tính loss của FedLwF = CE + α * KD.

        Luồng xử lý:
        - Task đầu: chỉ dùng cross-entropy.
        - Từ task sau: lấy old model, sinh old logits, tính KD loss,
          rồi cộng với CE loss theo hệ số `lwf_alpha`.
        """
        # Cross-entropy loss on new task data
        ce_loss = F.cross_entropy(output, target)

        # No distillation for first task or if no inputs
        if self.current_task == 0 or inputs is None:
            return ce_loss

        # Get old model
        if old_model is None:
            device = next(model.parameters()).device
            old_model = self.load_old_model(model, device)

        if old_model is None:
            return ce_loss

        # Get old model outputs
        with torch.no_grad():
            old_logits = old_model(inputs)

        # Compute distillation loss
        # distill_old_classes_only=True: restrict KD to old class logits only
        # distill_old_classes_only=False (default): distill on all class logits
        if self.distill_old_classes_only:
            kd_loss = self.compute_distillation_loss(
                old_logits,
                output,
                old_class_indices=self.old_classes if self.old_classes else None,
            )
        else:
            kd_loss = self.compute_distillation_loss(old_logits, output)

        return ce_loss + self.lwf_alpha * kd_loss

    def update_forgetting(self, task_accuracies: Dict[int, float]):
        """Cập nhật thống kê accuracy để tính Average Forgetting."""
        self.current_acc_per_task = task_accuracies.copy()

        for task_id, acc in task_accuracies.items():
            if task_id not in self.best_acc_per_task:
                self.best_acc_per_task[task_id] = acc
            else:
                self.best_acc_per_task[task_id] = max(
                    self.best_acc_per_task[task_id], acc
                )

        # Compute Average Forgetting
        if len(self.best_acc_per_task) > 1:
            forgetting_sum = 0.0
            count = 0
            for task_id in self.best_acc_per_task:
                if (
                    task_id != self.current_task
                    and task_id in self.current_acc_per_task
                ):
                    forgetting = (
                        self.best_acc_per_task[task_id]
                        - self.current_acc_per_task[task_id]
                    )
                    forgetting_sum += max(0, forgetting)
                    count += 1

            self.last_af = forgetting_sum / max(1, count)

    def cleanup(self):
        """Dọn cache model cũ và xóa thư mục tạm của FedLwF."""
        self._cached_old_model = None
        self.old_model_states.clear()

        # Optionally clean temp directory
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


class FedLwFAggregator(BaseAggregator):
    """
    FedLwF Aggregation - Standard FedAvg weighted average.

    FedLwF không đổi cách aggregate phía server.
    Điểm khác biệt chỉ nằm ở local loss của client.
    """

    def aggregate(
        self, results: List[Dict], global_params: Optional[OrderedDict] = None, **kwargs
    ) -> OrderedDict:
        """Aggregate theo weighted average giống FedAvg."""
        return self._weighted_average(results)


class FedLwFWithProximalTrainer(FedLwFTrainer):
    """
    FedLwF + Proximal regularization.

    Đây là biến thể kết hợp:
    - distillation của LwF
    - proximal term của FedProx
    để vừa chống quên vừa giảm lệch client trên dữ liệu non-IID.
    """

    def __init__(
        self,
        lwf_alpha: float = 1.0,
        temperature: float = 2.0,
        mu: float = 0.01,
        **kwargs,
    ):
        super().__init__(lwf_alpha=lwf_alpha, temperature=temperature, **kwargs)
        self.mu = mu

    def compute_loss(
        self,
        model: nn.Module,
        output: torch.Tensor,
        target: torch.Tensor,
        global_params: Optional[OrderedDict] = None,
        inputs: Optional[torch.Tensor] = None,
        old_model: Optional[nn.Module] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Tính loss của FedLwF + Proximal = LwF loss + proximal term."""
        # Get FedLwF loss (CE + KD)
        lwf_loss = super().compute_loss(
            model, output, target, global_params, inputs, old_model, **kwargs
        )

        # Add proximal term if global params provided
        if global_params is None:
            return lwf_loss

        prox_term = 0.0
        for name, param in model.named_parameters():
            if name in global_params:
                global_param = global_params[name].to(param.device)
                prox_term += ((param - global_param) ** 2).sum()

        return lwf_loss + (self.mu / 2) * prox_term
