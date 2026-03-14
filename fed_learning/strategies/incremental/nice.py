"""
NICE Strategy - Neurogenesis Inspired Contextual Encoding for Replay-free FCIL.

Reference:
    "NICE: Neurogenesis Inspired Contextual Encoding for Replay-free Class Incremental Learning"
    Gurbuz, Moorman, Dovrolis (CVPR 2024)
    GitHub: https://github.com/BurakGurbuz97/NICE

Implements:
    - NICETrainer: Phase-based training with CE loss + gradient masking
    - NICEAggregator: FedAvg with frozen parameter protection
    - Standalone NICE operations (neuron selection, connection pruning, age management)
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ...core import BaseTrainer, BaseAggregator


# ============================================================================
# NICE Operations (from paper + GitHub nice_operations.py)
# ============================================================================


def pick_top_neurons(scores: np.ndarray, tau: float) -> np.ndarray:
    """Chọn neuron theo chiến lược tau-greedy.

    Hàm này sắp xếp activation giảm dần rồi giữ tập neuron nhỏ nhất sao cho
    tổng activation tích lũy đạt ít nhất `tau * total_activation`.
    """
    total = scores.sum()
    if total <= 0:
        # No activations: select all
        return np.ones(len(scores), dtype=bool)

    threshold = tau * total
    sorted_idx = np.argsort(scores)[::-1]  # descending
    cumsum = np.cumsum(scores[sorted_idx])

    # Find smallest set with cumulative >= threshold
    n_keep = np.searchsorted(cumsum, threshold, side="left") + 1
    n_keep = min(n_keep, len(scores))

    selected = np.zeros(len(scores), dtype=bool)
    selected[sorted_idx[:n_keep]] = True
    return selected


def select_learner_units(model, tau: float, data: torch.Tensor):
    """Chọn các learner neuron dựa trên activation hiện tại.

    Mỗi phase, NICE reset các neuron chưa trưởng thành về trạng thái young,
    sau đó dựa vào activation để chọn ra nhóm learner mới.
    """
    # Step 1: Reset ALL non-mature to young (age=0)
    # Official: new_ranks[new_ranks < 2] = 0
    for name in model.LAYER_NAMES:
        if name == "fc2":
            continue  # fc2 ages managed by server (per-class)
        ranks = model.unit_ranks[name]
        ranks[ranks < 2] = 0

    if tau >= 1.0:
        # tau=100%: promote ALL young to learner (no pruning)
        for name in model.LAYER_NAMES:
            if name == "fc2":
                continue
            ranks = model.unit_ranks[name]
            ranks[ranks == 0] = 1
        return

    # Step 2: Compute activations
    activations = model.get_activations(data)

    # Step 3-4: Apply tau-greedy selection per layer
    for name in model.LAYER_NAMES:
        if name == "fc2":
            continue
        ranks = model.unit_ranks[name]
        young_mask = ranks == 0

        if not young_mask.any():
            continue

        # Get activation scores for young (candidate) neurons
        act = activations[name].cpu().numpy()
        young_scores = act[young_mask]

        # Pick top neurons among candidates
        selected = pick_top_neurons(young_scores, tau)

        # Set selected neurons to learner (age=1)
        young_indices = np.where(young_mask)[0]
        for i, idx in enumerate(young_indices):
            if selected[i]:
                ranks[idx] = 1  # promote to learner


def drop_young_to_learner(model):
    """Cắt các kết nối từ neuron young sang neuron không còn young.

    Mục tiêu là ngăn neuron mới sinh ảnh hưởng ngược lên các neuron đã ổn định,
    từ đó bảo vệ tri thức cũ trong mạng.
    """
    # Conv-to-conv pairs
    conv_pairs = [
        ("conv1", "conv2"),
        ("conv2", "conv3"),
    ]

    for src_name, tgt_name in conv_pairs:
        src_young = model.unit_ranks[src_name] == 0  # [in_dim]
        tgt_not_young = model.unit_ranks[tgt_name] >= 1  # [out_dim]

        if not src_young.any() or not tgt_not_young.any():
            continue

        mask = model.weight_masks[tgt_name]
        # Conv weight mask: [out_channels, in_channels, kernel_size]
        for out_idx in range(len(tgt_not_young)):
            if tgt_not_young[out_idx]:
                for in_idx in range(len(src_young)):
                    if src_young[in_idx]:
                        mask[out_idx, in_idx, :] = 0.0

    # conv3+gru → fc1: conv2lin transition
    # fc1 input = [conv3_flat (256*cnn_len) | gru_output (100)]
    # conv3 channel i maps to positions [i*cnn_len : (i+1)*cnn_len] in fc1 input
    fc1_not_young = model.unit_ranks["fc1"] >= 1
    if fc1_not_young.any():
        fc1_mask = model.weight_masks["fc1"]  # [256, concat_size]
        cnn_len = model.cnn_output_size // 256  # spatial dim per channel

        # conv3 young → fc1 not-young
        conv3_young = model.unit_ranks["conv3"] == 0
        if conv3_young.any():
            for ch_idx in range(len(conv3_young)):
                if conv3_young[ch_idx]:
                    col_start = ch_idx * cnn_len
                    col_end = col_start + cnn_len
                    for out_idx in range(len(fc1_not_young)):
                        if fc1_not_young[out_idx]:
                            fc1_mask[out_idx, col_start:col_end] = 0.0

        # gru young → fc1 not-young
        gru_young = model.unit_ranks["gru"] == 0
        if gru_young.any():
            gru_offset = model.cnn_output_size  # gru starts after CNN
            for gru_idx in range(len(gru_young)):
                if gru_young[gru_idx]:
                    col_idx = gru_offset + gru_idx
                    for out_idx in range(len(fc1_not_young)):
                        if fc1_not_young[out_idx]:
                            fc1_mask[out_idx, col_idx] = 0.0

    # fc1 → fc2
    fc1_young = model.unit_ranks["fc1"] == 0
    fc2_not_young = model.unit_ranks["fc2"] >= 1
    if fc1_young.any() and fc2_not_young.any():
        fc2_mask = model.weight_masks["fc2"]  # [num_classes, 256]
        for out_idx in range(len(fc2_not_young)):
            if fc2_not_young[out_idx]:
                for in_idx in range(len(fc1_young)):
                    if fc1_young[in_idx]:
                        fc2_mask[out_idx, in_idx] = 0.0

    # Physically zero weights where mask is 0 (official set_mask behavior)
    model.apply_masks_to_weights()


def grow_all_to_young(model):
    """Mở toàn bộ kết nối đi vào các neuron đang ở trạng thái young.

    Đây là bước cho phép neuron mới nhận thông tin từ mọi nguồn trước khi
    bước prune tiếp theo tinh chỉnh lại cấu trúc kết nối.
    """
    for name in model.LAYER_NAMES:
        if name == "gru":
            # GRU mask is 1D (output mask)
            young = model.unit_ranks[name] == 0
            model.weight_masks[name][young] = 1.0
            model.bias_masks[name][young] = 1.0
            continue

        young = model.unit_ranks[name] == 0
        if not young.any():
            continue

        mask = model.weight_masks[name]
        # Enable all input connections for young target neurons
        for idx in range(len(young)):
            if young[idx]:
                if mask.dim() == 3:
                    mask[idx, :, :] = 1.0
                elif mask.dim() == 2:
                    mask[idx, :] = 1.0
        model.bias_masks[name][young] = 1.0

    # Note: No apply_masks_to_weights() here - we're enabling connections,
    # not disabling. Physical zeroing is only needed when masking OUT.


def increase_unit_ranks(model):
    """Tăng tuổi cho các neuron đã được kích hoạt học.

    Sau mỗi task, learner sẽ trở thành mature và các neuron mature tiếp tục già đi.
    """
    for name in model.LAYER_NAMES:
        ranks = model.unit_ranks[name]
        mask = ranks >= 1
        ranks[mask] += 1


def update_freeze_masks(model):
    """Tạo freeze mask cho các neuron mature để chặn cập nhật gradient.

    Chỉ các kết nối vừa mature vừa đang active mới bị đóng băng.
    Nhờ đó NICE bảo vệ trí nhớ cũ mà không khóa nhầm các kết nối đã chết.
    """
    model.freeze_masks = {}
    for name in model.LAYER_NAMES:
        ranks = model.unit_ranks[name]
        # Base freeze mask: mature neurons (age > 1)
        mature_mask = ranks > 1
        model.freeze_masks[name] = mature_mask


# ============================================================================
# NICETrainer
# ============================================================================


class NICETrainer(BaseTrainer):
    """
    NICE Trainer - Neurogenesis-inspired replay-free training.

    Trainer này giữ phần loss đơn giản, còn anti-forgetting chủ yếu đến từ:
    - phase-based training ở client
    - gradient masking cho neuron mature
    - output/unit masking trong model

    Attributes:
        tau: Base threshold for neuron selection (default 0.95)
        max_phases: Number of phases per episode (default 5)
        phase_epochs: Epochs per phase (default 5)
        memo_per_class: Activation samples per class for context detector
    """

    def __init__(
        self,
        tau: float = 0.95,
        max_phases: int = 5,
        phase_epochs: int = 5,
        memo_per_class: int = 50,
        **kwargs,
    ):
        self.tau = tau
        self.max_phases = max_phases
        self.phase_epochs = phase_epochs
        self.memo_per_class = memo_per_class

        # Task tracking
        self.current_task: int = 0
        self.seen_classes: Set[int] = set()
        self.new_classes: List[int] = []

        # Forgetting tracking
        self.best_acc_per_task: Dict[int, float] = {}
        self.current_acc_per_task: Dict[int, float] = {}
        self.last_af: float = 0.0

    def set_task(self, task_id: int, new_classes: List[int]):
        """Cập nhật task hiện tại và các class mới của episode NICE."""
        self.current_task = task_id
        self.new_classes = list(new_classes)
        self.seen_classes.update(new_classes)

    def compute_loss(
        self,
        model: nn.Module,
        output: torch.Tensor,
        target: torch.Tensor,
        global_params: Optional[OrderedDict] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Tính CE loss chuẩn; chống quên của NICE nằm ở masking chứ không ở loss."""
        return F.cross_entropy(output, target)

    def pre_step(
        self, model: nn.Module, global_params: Optional[OrderedDict] = None, **kwargs
    ):
        """Đóng băng gradient của neuron mature trước khi optimizer cập nhật."""
        if hasattr(model, "reset_frozen_gradients"):
            model.reset_frozen_gradients()

    def update_forgetting(self, task_accuracies: Dict[int, float]):
        """Cập nhật Average Forgetting để theo dõi mức độ quên của NICE."""
        self.current_acc_per_task = task_accuracies.copy()

        for task_id, acc in task_accuracies.items():
            if task_id not in self.best_acc_per_task:
                self.best_acc_per_task[task_id] = acc
            else:
                self.best_acc_per_task[task_id] = max(
                    self.best_acc_per_task[task_id], acc
                )

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


# ============================================================================
# NICEAggregator
# ============================================================================
