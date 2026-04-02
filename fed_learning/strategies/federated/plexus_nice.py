"""
PlexusNICE Strategy - Decentralized NICE using Plexus mechanisms.

Combines:
- Plexus: Hash-based peer sampling + bandwidth-based aggregator selection + PopulationView
- NICE: Neuron age system + gradient freezing + phase-based training (replay-free)

Reference:
    NICE: Gurbuz, Moorman, Dovrolis (CVPR 2024)
    Plexus: Dhasade et al. (EuroMLSys 2025)
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import BaseTrainer, BaseAggregator
from .plexus import SampleManager, PopulationView
from ..incremental.nice import (
    pick_top_neurons,
    select_learner_units,
    drop_young_to_learner,
    grow_all_to_young,
    increase_unit_ranks,
    update_freeze_masks,
)


# ============================================================================
# DistributedContextDetector
# ============================================================================


class DistributedContextDetector:
    """
    Decentralized context detection for PlexusNICE.

    Replaces NICEServer's ContextDetector with a simplified distributed version:
    - Uses round-counting as proxy for task context
    - No centralized LR classifier training

    In decentralized setting, we use:
        task_id = current_round // rounds_per_task

    This is a simplification - full distributed context detection would require
    a consensus protocol or gossip-based voting.
    """

    def __init__(self, rounds_per_task: int = 5):
        self.rounds_per_task = rounds_per_task
        self.episode_classes: Dict[int, List[int]] = {}

    def set_episode_classes(self, episode: int, classes: List[int]):
        """Store classes for an episode."""
        self.episode_classes[episode] = list(classes)

    def get_context_for_round(self, round_num: int) -> int:
        """
        Get the likely episode/task context for a given round.

        Uses round-counting as proxy:
        - Round 0 to R-1: Episode 0
        - Round R to 2R-1: Episode 1
        - etc.
        """
        return round_num // self.rounds_per_task

    def predict_episode(self, binary_activations: np.ndarray) -> int:
        """
        Predict episode - falls back to round-counting.

        In centralized NICE, this uses chained LR classifiers on activation patterns.
        In decentralized PlexusNICE, we use round-counting as a proxy.
        """
        # This is called during inference but we use round-based context instead
        return 0  # Will be overridden by server's round-based context

    def train_models(self, current_episode: int):
        """No-op in decentralized setting - no centralized training."""
        pass

    def push_activations(self, model, data: torch.Tensor, episode: int):
        """No-op in decentralized setting."""
        pass


# ============================================================================
# PlexusNICETrainer
# ============================================================================


class PlexusNICETrainer(BaseTrainer):
    """
    NICE trainer adapted for Plexus decentralized setting.

    Inherits NICETrainer's phase-based training but:
    - Tracks task locally (no server notification)
    - Uses distributed context detector (round-counting proxy)

    Anti-forgetting mechanisms:
    - Neuron age system (young -> learner -> mature)
    - Gradient freezing on mature neurons
    - Output masking via LetLearner
    """

    def __init__(
        self,
        tau: float = 0.95,
        max_phases: int = 5,
        phase_epochs: int = 5,
        memo_per_class: int = 50,
        rounds_per_task: int = 5,
    ):
        self.tau = tau
        self.max_phases = max_phases
        self.phase_epochs = phase_epochs
        self.memo_per_class = memo_per_class
        self.rounds_per_task = rounds_per_task

        # Task tracking
        self.current_task: int = 0
        self.seen_classes: Set[int] = set()
        self.new_classes: List[int] = []

        # Forgetting tracking
        self.best_acc_per_task: Dict[int, float] = {}
        self.current_acc_per_task: Dict[int, float] = {}
        self.last_af: float = 0.0

    def set_task(self, task_id: int, new_classes: List[int]):
        """Update task state when starting a new episode."""
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
        """Standard CE loss - NICE anti-forgetting is in masking."""
        return F.cross_entropy(output, target)

    def pre_step(
        self,
        model: nn.Module,
        global_params: Optional[OrderedDict] = None,
        **kwargs,
    ):
        """Freeze mature neuron gradients before optimizer step."""
        if hasattr(model, "reset_frozen_gradients"):
            model.reset_frozen_gradients()

    def update_forgetting(self, task_accuracies: Dict[int, float]):
        """Track forgetting metrics."""
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
# PlexusNICEAggregator
# ============================================================================


class PlexusNICEAggregator(BaseAggregator):
    """
    NICE aggregator for Plexus decentralized setting.

    Extends PlexusAggregator with:
    - NICE's frozen-parameter restoration (restore mature neuron params)
    - Neuron age synchronization via piggybacked state
    - Freeze mask propagation via piggybacked state

    Key difference from NICEAggregator:
    - No reliance on server to set frozen_keys / freeze_masks
    - State is piggybacked on model transfers, not broadcast
    """

    def __init__(
        self,
        sample_size: int = 13,
        num_aggregators: int = 1,
        success_fraction: float = 0.8,
        inactivity_threshold: int = 50,
        client_bandwidths: Optional[Dict[int, float]] = None,
    ):
        self.sample_size = sample_size
        self.num_aggregators = num_aggregators
        self.success_fraction = success_fraction
        self.inactivity_threshold = inactivity_threshold
        self.client_bandwidths: Dict[int, float] = client_bandwidths or {}

        # Plexus sampling infrastructure
        self.sample_manager = SampleManager(sample_size, num_aggregators)
        self.population_view = PopulationView()
        self.current_round: int = 0

        # NICE-specific: frozen parameter state
        self._frozen_keys: Set[str] = set()
        self._freeze_masks: Dict[str, np.ndarray] = {}

        # NICE-specific: neuron ages state (piggybacked)
        self._neuron_ages_state: Optional[Dict[str, np.ndarray]] = None
        self._masks_state: Optional[Dict] = None

    def set_neuron_ages(self, ages: Dict[str, np.ndarray]):
        """Set neuron ages received via piggyback."""
        self._neuron_ages_state = ages

    def set_freeze_masks(self, masks: Dict[str, np.ndarray]):
        """Set freeze masks received via piggyback."""
        self._freeze_masks = {k: np.array(v) for k, v in masks.items()}
        # NOTE: frozen_keys should be set separately via set_frozen_keys()
        # using actual parameter keys from the model's named_parameters()

    def set_frozen_keys(self, keys: List[str]):
        """Set fully-frozen parameter keys (actual parameter names like 'conv1.weight')."""
        self._frozen_keys = set(keys)

    def aggregate(
        self,
        results: List[Dict],
        global_params: Optional[OrderedDict] = None,
        **kwargs,
    ) -> OrderedDict:
        """
        Aggregate using FedAvg with NICE's mature-neuron restoration.

        Flow:
        1. Weighted average of all received params (server already filtered by success_fraction)
        2. Restore fully-frozen keys from global model
        3. Restore per-neuron frozen params from global model
        """
        if not results:
            return global_params

        # Update population view
        for r in results:
            cid = r.get("client_id", -1)
            self.population_view.update(cid, self.current_round, is_online=True)

        # Step 1: Weighted average
        agg_results = []
        for r in results:
            params = r.get("params", r.get("masked_params"))
            agg_results.append({
                "params": params,
                "num_samples": r["num_samples"],
            })

        averaged = self._weighted_average(agg_results)

        if averaged is None:
            return global_params if global_params is not None else OrderedDict()

        # Step 2: Restore fully frozen keys
        if global_params is not None:
            for key in self._frozen_keys:
                if key in global_params and key in averaged:
                    averaged[key] = global_params[key].clone()

            # Step 3: Per-neuron freezing for partially-mature layers
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

                    mask_tensor = torch.tensor(freeze, dtype=torch.bool)

                    if "weight" in key and averaged[key].dim() >= 2:
                        if len(freeze) == averaged[key].shape[0]:
                            averaged[key][mask_tensor] = global_params[key][mask_tensor].clone()
                    elif "bias" in key:
                        if len(freeze) == averaged[key].shape[0]:
                            averaged[key][mask_tensor] = global_params[key][mask_tensor].clone()

        return averaged

    def get_round_aggregators(self, round_num: int, all_client_ids: List[int]) -> List[int]:
        """Bandwidth-based aggregator selection."""
        active = self.population_view.get_active_peers(round_num, self.inactivity_threshold)
        if active:
            # Intersect active peers with the server-provided candidate set
            candidates = [pid for pid in active if pid in all_client_ids]
            if not candidates:
                candidates = all_client_ids
        else:
            candidates = all_client_ids
        return self.sample_manager.get_aggregators(round_num, candidates, self.client_bandwidths)

    def get_round_sample(self, round_num: int, all_client_ids: List[int]) -> List[int]:
        """Hash-ordered sample selection."""
        active = self.population_view.get_active_peers(round_num, self.inactivity_threshold)
        if active:
            # Intersect active peers with the server-provided candidate set
            candidates = [pid for pid in active if pid in all_client_ids]
            if not candidates:
                candidates = all_client_ids
        else:
            candidates = all_client_ids
        sample = self.sample_manager.get_sample(round_num, candidates, self.client_bandwidths)
        if len(sample) < 3:
            return candidates
        return sample
