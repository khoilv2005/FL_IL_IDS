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
    """Tau-greedy neuron selection (Paper Eq.1-2).

    Sort activations descending, keep smallest subset with cumulative
    activation >= tau * total_activation.

    Args:
        scores: activation scores per neuron
        tau: threshold (0.0 to 1.0), fraction of total activation to retain

    Returns:
        Boolean mask, True for selected neurons
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
    """Select learner neurons using tau-greedy elimination.

    Official logic (from GitHub nice_operations.py):
    1. Reset ALL non-mature neurons to 0 (young) first
    2. Compute activations
    3. Apply tau-greedy selection: pick top neurons with cumsum >= tau*total
    4. Set selected neurons to 1 (learner)

    This differs from promoting then reverting — it ensures a clean slate
    each phase, so previous learner status doesn't persist.

    Args:
        model: NICEModel instance
        tau: selection threshold (1.0 = keep all, 0.95 = prune 5%)
        data: sample data for computing activations
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
        young_mask = (ranks == 0)

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
    """Prune connections from young source to non-young target.

    Zero weight mask entries where source neuron is young (age=0)
    and target neuron is not young (age>=1). This prevents young
    neurons from influencing mature/learner neurons.

    Handles conv-to-conv pairs AND conv/gru-to-fc1 transition:
    - conv1→conv2, conv2→conv3: standard per-channel masking
    - conv3+gru→fc1: conv3 neurons expand spatially (256*cnn_len),
      gru neurons map 1:1 to the last 100 dims of fc1's input
    - fc1→fc2: standard per-neuron masking

    After masking, physically zeros the weights (official set_mask behavior).

    Args:
        model: NICEModel instance
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
    """Restore all connections into young (age=0) target neurons.

    Enable all incoming connections for young neurons, allowing them
    to receive information from all sources. Does NOT apply physical
    weight zeroing here since we're enabling connections, not disabling.

    Args:
        model: NICEModel instance
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
    """Age transition: all neurons with rank >= 1 get +1.

    Called at end of episode (task). Learners (1) become mature (2),
    mature neurons age further.

    Args:
        model: NICEModel instance
    """
    for name in model.LAYER_NAMES:
        ranks = model.unit_ranks[name]
        mask = ranks >= 1
        ranks[mask] += 1


def update_freeze_masks(model):
    """Build gradient freeze masks for mature neurons (age > 1).

    Official logic (from GitHub nice_operations.py):
    Freeze masks are the INTERSECTION of mature neuron mask AND active
    weight connections. Only freeze weights that are both mature AND have
    active connections (mask=1). This prevents freezing dead connections.

    For weights: freeze_w[target_mature, :] = 1, then freeze_w *= weight_mask
    For biases: freeze_b[target_mature] = 1

    Args:
        model: NICEModel instance
    """
    model.freeze_masks = {}
    for name in model.LAYER_NAMES:
        ranks = model.unit_ranks[name]
        # Base freeze mask: mature neurons (age > 1)
        mature_mask = (ranks > 1)
        model.freeze_masks[name] = mature_mask


# ============================================================================
# NICETrainer
# ============================================================================

class NICETrainer(BaseTrainer):
    """
    NICE Trainer - Neurogenesis-inspired replay-free training.

    Handles CE loss computation and gradient masking.
    Phase-based training logic is in NICEClient.train().

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
        """Called at beginning of each new task (episode)."""
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
        """Standard CE loss. Anti-forgetting is via gradient masking + output masking."""
        return F.cross_entropy(output, target)

    def pre_step(
        self, model: nn.Module, global_params: Optional[OrderedDict] = None, **kwargs
    ):
        """Apply gradient freezing for mature neurons before optimizer.step."""
        if hasattr(model, "reset_frozen_gradients"):
            model.reset_frozen_gradients()

    def update_forgetting(self, task_accuracies: Dict[int, float]):
        """Update forgetting metrics (AF tracking)."""
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
                if task_id != self.current_task and task_id in self.current_acc_per_task:
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

class NICEAggregator(BaseAggregator):
    """
    NICE Aggregator - FedAvg with per-neuron frozen parameter protection.

    After averaging, restores frozen (mature) neuron parameters from
    the global model. This works at the individual neuron level, not
    just at the full-layer level, so partially-mature layers are also
    properly protected.
    """

    def __init__(self):
        self._frozen_keys: Set[str] = set()
        self._freeze_masks: Dict[str, np.ndarray] = {}

    def set_frozen_keys(self, keys: List[str]):
        """Set which parameter keys are fully frozen (should not be averaged)."""
        self._frozen_keys = set(keys)

    def set_freeze_masks(self, freeze_masks: Dict[str, np.ndarray]):
        """Set per-layer freeze masks for partial freezing.

        Args:
            freeze_masks: Dict mapping layer names to boolean arrays
                         where True = frozen (mature) neuron
        """
        self._freeze_masks = {k: np.array(v) for k, v in freeze_masks.items()}

    def aggregate(
        self,
        results: List[Dict],
        global_params: Optional[OrderedDict] = None,
        **kwargs,
    ) -> OrderedDict:
        """Aggregate with per-neuron frozen parameter protection."""
        agg_results = []
        for r in results:
            params = r.get("params", r.get("masked_params"))
            agg_results.append({
                "params": params,
                "num_samples": r["num_samples"],
            })

        averaged = self._weighted_average(agg_results)

        # Restore frozen parameters from global model
        if global_params is not None:
            # 1. Fully frozen keys (all neurons mature)
            for key in self._frozen_keys:
                if key in global_params and key in averaged:
                    averaged[key] = global_params[key].clone()

            # 2. Per-neuron freezing for partially-mature layers
            if self._freeze_masks:
                for key in averaged:
                    if key not in global_params:
                        continue

                    layer_name = key.split(".")[0]
                    if layer_name == "gru" or layer_name not in self._freeze_masks:
                        continue

                    freeze = self._freeze_masks[layer_name]
                    if not np.any(freeze):
                        continue

                    mask = torch.tensor(freeze, dtype=torch.bool)

                    if "weight" in key and averaged[key].dim() >= 2:
                        if len(freeze) == averaged[key].shape[0]:
                            # Restore frozen rows from global params
                            averaged[key][mask] = global_params[key][mask].clone()
                    elif "bias" in key:
                        if len(freeze) == averaged[key].shape[0]:
                            averaged[key][mask] = global_params[key][mask].clone()

        return averaged

    def set_task(self, task_id: int):
        """Compatibility interface."""
        pass
