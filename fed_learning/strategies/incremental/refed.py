"""
Re-Fed Strategy - Retrieval-Enhanced Federated Incremental Learning.

Reference:
    Li et al., "Towards Efficient Replay in Federated Incremental Learning",
    CVPR 2024

Paper Algorithm Summary:
========================
1. Personalized Informative Model (PIM) (Section 3.2, Eq. 3):
   - Each client maintains a PIM v_k that blends local and global knowledge
   - PIM update: v_{k,s} = v_{k,s-1} - η * (Σ ∇l(f_v(x̃), ỹ) + q(λ)(v_{k,s-1} - w^{t-1}))
   - where q(λ) = (1-λ)/(2λ), λ ∈ (0, 1)
   - λ close to 0 → PIM aligns with global model
   - λ close to 1 → PIM focuses on local training

2. Sample Importance (Eq. 4-5):
   - Gradient norm: G_p(x̃) = ||∇l(f_v(x̃), ỹ)||_2
   - Importance score with early-emphasis: I(x̃) = Σ_{p=1}^{s} (1/p) * G_p(x̃)
   - Samples with higher importance are cached for replay

3. Local Training (Eq. 6):
   - Standard CE loss on cached + new task data
   - w_{k,p} = w_{k,p-1} - η * Σ ∇l(f_w(x̃), ỹ)

4. Federated Aggregation:
   - Standard FedAvg (modular - any aggregation works)

Key Properties:
   - Modular: pluggable into any FL algorithm (FedAvg, FedProx, etc.)
   - Privacy-preserving: no extra information transmitted beyond standard FL
   - Resource-efficient: no distillation data or generated data needed
"""

import copy
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import BaseTrainer, BaseAggregator


class ReFedTrainer(BaseTrainer):
    """
    Re-Fed Trainer - Standard CE training for local model.

    The PIM update and importance scoring are handled in the client.
    The trainer handles the local model training on cached + new data.

    Args:
        memory_size: Maximum number of cached samples per client
        lambda_pim: Balance between local and global info in PIM (0, 1)
        pim_iterations: Number of PIM update iterations (s in paper)
        temp_dir: Directory for temporary storage
    """

    def __init__(
        self,
        memory_size: int = 2000,
        lambda_pim: float = 0.5,
        pim_iterations: int = 5,
        temp_dir: str = "./temp_refed_storage",
        **kwargs,
    ):
        self.memory_size = memory_size
        self.lambda_pim = lambda_pim
        self.pim_iterations = pim_iterations
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)

        # Task tracking
        self.current_task: int = 0
        self.seen_classes: Set[int] = set()
        self.old_classes: List[int] = []
        self.new_classes: List[int] = []

    def set_task(self, task_id: int, task_classes: List[int]):
        """Update task tracking."""
        self.old_classes = list(self.seen_classes)
        self.new_classes = list(task_classes)
        self.seen_classes.update(task_classes)
        self.current_task = task_id

    def compute_loss(
        self,
        model: nn.Module,
        output: torch.Tensor,
        target: torch.Tensor,
        global_params: Optional[OrderedDict] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Standard Cross-Entropy loss for local training (Eq. 6).

        Re-Fed uses standard CE on combined cached + new data.
        The key mechanism is in sample selection (PIM), not loss modification.
        """
        return nn.CrossEntropyLoss()(output, target)


class ReFedAggregator(BaseAggregator):
    """
    Re-Fed Aggregator - Standard FedAvg aggregation.

    Paper Section 3.2: Re-Fed is modular and works with any
    aggregation method. Default is FedAvg (weighted average).
    """

    def aggregate(
        self,
        results: List[Dict],
        global_params: Optional[OrderedDict] = None,
        **kwargs,
    ) -> OrderedDict:
        """Aggregate using weighted average (FedAvg)."""
        return self._weighted_average(results)
