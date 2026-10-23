"""
DER Strategy - Dynamically Expandable Representation for FCIL.

Reference:
    "DER: Dynamically Expandable Representation for Class Incremental Learning"
    Yan, Xie, He (CVPR 2021)

Paper Equations Implemented:
    Eq.3:  L_{H_t} = CE loss on unified classifier
    Eq.8:  Annealing schedule s = 1/s_max + (s_max - 1/s_max) * (b-1)/(B-1)
    Eq.10: Sparsity loss L_S
    Eq.11: Full loss L_DER = L_{H_t} + λ_a * L_{H_t^a} + λ_s * L_S

Stage 1 (Representation Learning):
    - Train new extractor F_t + masks + classifier H_t + auxiliary H_t^a
    - Loss: L_DER (Eq.11)
    - Annealing schedule resets each epoch (Eq.8: b from 1 to B per epoch)

Stage 2 (Classifier Finetuning):
    - Freeze all extractors, train classifier H_t only
    - Loss: CE with temperature scaling: CE(Softmax(H_t(u)/δ), y)
    - Data: class-balanced subset from D_t ∪ M_t
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import BaseTrainer, BaseAggregator


class DERTrainer(BaseTrainer):
    """
    DER local training strategy with two-stage training and mask annealing.

    Cách đọc nhanh:
    - Stage 1: học extractor mới + mask + auxiliary classifier.
    - Stage 2: cố định extractor, chỉ tinh chỉnh classifier chung.
    - Mục tiêu là thêm khả năng biểu diễn mới mà vẫn hạn chế quên kiến thức cũ.

    Inherits BaseTrainer:
        - compute_loss() default CE → overridden for DER losses
        - pre_step() → overridden for gradient compensation (Eq.9)
        - post_step(), pre_train(), post_train() → inherited (no-op)
        - get_optimizer_class() → inherited (Adam)

    Attributes:
        lambda_aux: Auxiliary loss weight (λ_a, paper default 1.0)
        lambda_sparsity: Sparsity loss weight (λ_s, paper default 0.5)
        s_max: Max annealing parameter (paper default 15)
        temperature: Temperature for Stage 2 classifier (δ, paper: 5 for CIFAR)
        training_stage: 1=representation, 2=classifier
    """

    def __init__(
        self,
        lambda_aux: float = 1.0,
        lambda_sparsity: float = 0.5,
        s_max: float = 15.0,
        temperature: float = 2.0,
        buffer_size: int = 500,
    ):
        self.lambda_aux = lambda_aux
        self.lambda_sparsity = lambda_sparsity
        self.s_max = s_max
        self.temperature = temperature
        self.buffer_size = buffer_size

        # Task tracking
        self.current_task: int = 0
        self.seen_classes: Set[int] = set()
        self.old_classes: List[int] = []
        self.new_classes: List[int] = []

        # Training stage (1=representation, 2=classifier)
        self.training_stage: int = 1

        # Annealing state (Eq.8)
        # Paper: s resets each epoch (b goes from 1 to B within each epoch)
        # In federated setting: we track across all batches within a client's training
        self.current_batch: int = 0
        self.total_batches: int = 1

        # Forgetting tracking (same interface as FedLwF/FedCBDR)
        self.best_acc_per_task: Dict[int, float] = {}
        self.current_acc_per_task: Dict[int, float] = {}
        self.last_af: float = 0.0

    def set_task(self, task_id: int, new_classes: List[int]):
        """
        Được gọi khi bắt đầu task mới để cập nhật class cũ/mới,
        reset batch counter và đồng bộ trạng thái trainer.
        """
        new_set = set(int(c) for c in new_classes)
        self.old_classes = sorted(int(c) for c in self.seen_classes if int(c) not in new_set)
        self.new_classes = list(new_classes)
        self.current_task = task_id
        self.seen_classes = set(self.old_classes) | new_set
        self.current_batch = 0

        print(
            f"  DERTrainer Task {task_id}: "
            f"old={len(self.old_classes)}, new={len(new_classes)}"
        )

    def set_stage(self, stage: int):
        """
        Chuyển giữa hai giai đoạn train của DER.

        - `stage=1`: học representation mới.
        - `stage=2`: fine-tune classifier.
        """
        self.training_stage = stage
        self.current_batch = 0

    def compute_annealing_s(self) -> float:
        """
        Tính tham số annealing `s` cho mask ở batch hiện tại.

        Giá trị này giúp mask chuyển dần từ mềm sang cứng trong quá trình train.
        """
        if self.total_batches <= 1:
            return self.s_max

        # Paper uses 1-indexed b, we use 0-indexed current_batch
        ratio = min(1.0, self.current_batch / max(1, self.total_batches - 1))
        return 1.0 / self.s_max + (self.s_max - 1.0 / self.s_max) * ratio

    def compute_loss(
        self,
        model: nn.Module,
        output: torch.Tensor,
        target: torch.Tensor,
        global_params: Optional[OrderedDict] = None,
        inputs: Optional[torch.Tensor] = None,
        s: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Tính loss theo giai đoạn training hiện tại của DER.

        Hàm này chỉ làm dispatcher:
        - Stage 1 -> `_stage1_loss()`
        - Stage 2 -> `_stage2_loss()`
        """
        if self.training_stage == 1:
            return self._stage1_loss(model, output, target, inputs, s)
        else:
            return self._stage2_loss(output, target)

    def _stage1_loss(
        self,
        model: nn.Module,
        output: torch.Tensor,
        target: torch.Tensor,
        inputs: Optional[torch.Tensor],
        s: Optional[float],
    ) -> torch.Tensor:
        """
        Tính loss Stage 1 của DER.

        Bao gồm 3 thành phần:
        - CE loss của classifier chính
        - auxiliary loss để phân biệt lớp mới với nhóm "other"
        - sparsity loss để khuyến khích pruning/gating phù hợp
        """
        device = output.device

        # 1. Main classifier CE loss (Eq.3)
        ce_loss = F.cross_entropy(output, target)

        # 2. Auxiliary classifier loss
        # Paper: λ_a = 0 for first step (no old classes to distinguish)
        aux_loss = torch.tensor(0.0, device=device)
        if (
            self.current_task > 0
            and self.lambda_aux > 0
            and inputs is not None
            and hasattr(model, "forward_aux")
        ):
            aux_output = model.forward_aux(inputs, s=s)
            aux_target = self._remap_aux_targets(target, device)
            aux_loss = F.cross_entropy(aux_output, aux_target)

        # 3. Sparsity loss (Eq.10)
        sparsity_loss = torch.tensor(0.0, device=device)
        if (
            self.lambda_sparsity > 0
            and s is not None
            and hasattr(model, "get_sparsity_loss")
        ):
            sparsity_loss = model.get_sparsity_loss(s)

        self.current_batch += 1

        return (
            ce_loss + self.lambda_aux * aux_loss + self.lambda_sparsity * sparsity_loss
        )

    def _stage2_loss(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Tính loss Stage 2: cross-entropy có chia temperature.

        Mục tiêu của stage này là làm classifier ổn định hơn sau khi
        representation mới đã được học ở Stage 1.
        """
        return F.cross_entropy(output / self.temperature, target)

    def _remap_aux_targets(
        self, target: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """
        Remap nhãn cho auxiliary classifier.

        Tất cả lớp cũ được gộp vào một nhãn `other`, còn lớp mới được
        đánh lại chỉ số từ 0..|Y_t|-1.
        """
        new_cls_map = {c: i for i, c in enumerate(self.new_classes)}
        other_idx = len(self.new_classes)  # "other" class index
        remapped = torch.full_like(target, other_idx)
        for c, i in new_cls_map.items():
            remapped[target == c] = i
        return remapped

    def update_forgetting(self, task_accuracies: Dict[int, float]):
        """
        Theo dõi accuracy từng task và tính Average Forgetting.

        Hàm này chủ yếu phục vụ báo cáo/evaluation loop, không tác động
        trực tiếp tới gradient của DER.
        """
        self.current_acc_per_task = task_accuracies.copy()

        for tid, acc in task_accuracies.items():
            if tid not in self.best_acc_per_task:
                self.best_acc_per_task[tid] = acc
            else:
                self.best_acc_per_task[tid] = max(self.best_acc_per_task[tid], acc)

        # Compute AF
        if len(self.best_acc_per_task) > 1:
            forgetting_sum = 0.0
            count = 0
            for tid in self.best_acc_per_task:
                if tid != self.current_task and tid in self.current_acc_per_task:
                    f = self.best_acc_per_task[tid] - self.current_acc_per_task[tid]
                    forgetting_sum += max(0, f)
                    count += 1
            self.last_af = forgetting_sum / max(1, count)
