"""
NICE Client - Client for Neurogenesis Inspired Contextual Encoding.

Reference:
    "NICE: Neurogenesis Inspired Contextual Encoding for Replay-free Class Incremental Learning"
    Gurbuz, Moorman, Dovrolis (CVPR 2024)
    GitHub: https://github.com/BurakGurbuz97/NICE

Key Features:
    - Phase-based training: max_phases x phase_epochs per episode
    - Tau-greedy neuron selection per phase
    - Connection pruning (drop_young_to_learner, grow_all_to_young)
    - Output masking via model.forward_output()
    - Gradient freezing via model.reset_frozen_gradients()
    - No replay buffer (replay-free)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Set, Any
from collections import OrderedDict

try:
    from torch.amp import autocast as torch_autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast as torch_autocast, GradScaler

from .client import FederatedClient
from ..core import BaseTrainer
from ..models.nice_model import NICEModel
from ..strategies.fed_incremental.nice import (
    select_learner_units,
    drop_young_to_learner,
    grow_all_to_young,
)


class NICEClient(FederatedClient):
    """
    Client for NICE algorithm with phase-based training.

    Training loop per episode (task):
        For each phase in max_phases:
            1. select_learner_units(model, tau, sample_data)
               - Phase 1: tau=1.0 (keep all), Phase 2+: tau=0.95
            2. drop_young_to_learner(model) - prune connections
            3. grow_all_to_young(model) - restore young connections
            4. Train phase_epochs with:
               - model.forward_output(x) for loss (output masking)
               - model.reset_frozen_gradients() after backward
            5. Store activations for context detector

    No replay buffer needed (replay-free algorithm).
    """

    def __init__(
        self,
        client_id: int,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        max_phases: int = 5,
        phase_epochs: int = 5,
        tau: float = 0.95,
    ):
        super().__init__(client_id, X_train, y_train)

        # NICE hyperparameters
        self.max_phases = max_phases
        self.phase_epochs = phase_epochs
        self.tau = tau

        # Task tracking
        self.current_task: int = 0
        self.current_classes: Set[int] = set()
        self.seen_classes: Set[int] = set()

    def set_task_data(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        task_id: int,
        task_classes: List[int],
    ):
        """Set data for current task."""
        self.X_train = X_train
        self.y_train = y_train
        self.num_samples = len(y_train)
        self.current_task = task_id
        self.current_classes = set(task_classes)
        self.seen_classes.update(task_classes)

    def train(
        self,
        trainer: BaseTrainer,
        epochs: int,
        batch_size: int,
        lr: float,
        global_params: Optional[OrderedDict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Phase-based NICE training.

        The `epochs` parameter is ignored in favor of max_phases x phase_epochs.
        This matches the paper's training protocol.

        Official behavior (from GitHub learner.py):
        - New optimizer created each phase (fresh momentum/state)
        - Phase 1: tau=100% (keep all), Phase 2+: tau=95%
        - Last episode: all phases use tau=100% (keep all neurons)

        Args:
            trainer: NICETrainer instance
            epochs: Ignored (uses max_phases x phase_epochs)
            batch_size: Batch size
            lr: Learning rate
            global_params: Global model parameters
            **kwargs: Additional arguments (neuron_ages, masks, is_last_task, etc.)

        Returns:
            Dict with client_id, num_samples, loss, params, neuron_ages
        """
        model = self.model
        model.train()

        # Get NICE-specific config from kwargs or trainer
        max_phases = getattr(trainer, "max_phases", self.max_phases)
        phase_epochs = getattr(trainer, "phase_epochs", self.phase_epochs)
        tau = getattr(trainer, "tau", self.tau)
        is_last_task = kwargs.get("is_last_task", False)
        phase_offset = int(kwargs.get("phase_offset", 0))
        max_phases_override = kwargs.get("max_phases_override")
        if max_phases_override is not None:
            max_phases = max(1, int(max_phases_override))

        # Sample data for neuron selection (subsample for efficiency)
        n_sample = min(500, self.num_samples)
        sample_idx = torch.randperm(self.num_samples)[:n_sample]
        sample_data = self.X_train[sample_idx].to(self.device)

        total_loss = 0.0
        total_batches = 0

        # Freeze BN for layers with all-mature neurons
        model.freeze_bn_for_mature()

        # ================================================================
        # Phase loop
        # ================================================================
        for phase in range(max_phases):
            global_phase = phase_offset + phase
            # Official: Phase 1 uses tau=100% (keep all), Phase 2+ uses tau=config
            # Official: Last episode uses tau=100% for ALL phases (keep all neurons)
            if is_last_task:
                phase_tau = 1.0  # Last episode: keep all neurons
            elif global_phase == 0:
                phase_tau = 1.0  # First phase: keep all candidates
            else:
                phase_tau = tau  # Other phases: prune with tau

            # 1. Select learner neurons via tau-greedy
            model.eval()
            select_learner_units(model, phase_tau, sample_data)
            model.train()
            model.freeze_bn_for_mature()

            # 2. Connection pruning (also physically zeros weights)
            drop_young_to_learner(model)

            # 3. Restore connections into young neurons
            grow_all_to_young(model)

            # Move masks to device
            model._move_masks_to_device(self.device)

            # 4. Create NEW optimizer each phase (official behavior)
            # Fresh optimizer state prevents momentum from previous phase
            # from interfering with newly selected neurons
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scaler = GradScaler(enabled=self.use_amp)

            # 5. Train phase_epochs
            for ep in range(phase_epochs):
                for X_batch, y_batch in self._create_batches(batch_size):
                    optimizer.zero_grad()

                    if self.use_amp:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            output = model.forward_output(X_batch)
                            loss = trainer.compute_loss(
                                model, output, y_batch, global_params
                            )

                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        # Freeze mature gradients
                        model.reset_frozen_gradients()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        output = model.forward_output(X_batch)
                        loss = trainer.compute_loss(
                            model, output, y_batch, global_params
                        )

                        loss.backward()
                        # Freeze mature gradients
                        model.reset_frozen_gradients()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                    total_loss += loss.item()
                    total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)

        return {
            "client_id": self.client_id,
            "num_samples": self.num_samples,
            "loss": avg_loss,
            "params": OrderedDict(
                (k, v.cpu().clone()) for k, v in model.state_dict().items()
            ),
            "neuron_ages": model.get_neuron_ages_state(),
        }

    def get_buffer_stats(self) -> Dict:
        """NICE does not use buffer - return empty stats."""
        return {
            "buffer_type": "none",
            "has_replay": False,
            "note": "NICE is replay-free",
        }
