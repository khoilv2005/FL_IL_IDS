"""
PlexusDER Strategy - Decentralized DER using Plexus mechanisms.

Combines:
- Plexus: Hash-based peer sampling + bandwidth-based aggregator selection + PopulationView
- DER: Two-stage training with dynamic model expansion + HAT masks + exemplar replay

Reference:
    DER: Yan, Xie, He (CVPR 2021)
    Plexus: Dhasade et al. (EuroMLSys 2025)
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import BaseTrainer, BaseAggregator
from .plexus import SampleManager, PopulationView


# ============================================================================
# PlexusDERTrainer
# ============================================================================


class PlexusDERTrainer(BaseTrainer):
    """
    DER trainer adapted for Plexus decentralized setting.

    Inherits the two-stage training logic from DERTrainer but removes
    server-side task tracking (done via protocol in decentralized setting).

    Stage 1: Train current extractor + masks + classifier + aux_classifier
    Stage 2: Freeze extractors, fine-tune classifier only
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
        self.current_batch: int = 0
        self.total_batches: int = 1

        # Forgetting tracking
        self.best_acc_per_task: Dict[int, float] = {}
        self.current_acc_per_task: Dict[int, float] = {}
        self.last_af: float = 0.0

    def set_task(self, task_id: int, new_classes: List[int]):
        """Update task state when starting a new task."""
        self.old_classes = list(self.seen_classes)
        self.new_classes = list(new_classes)
        self.current_task = task_id
        self.seen_classes.update(new_classes)
        self.current_batch = 0

    def set_stage(self, stage: int):
        """Switch between training stages."""
        self.training_stage = stage
        self.current_batch = 0

    def compute_annealing_s(self) -> float:
        """Compute annealing parameter s for HAT masks (Paper Eq.8)."""
        if self.total_batches <= 1:
            return self.s_max
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
        """Compute loss based on training stage."""
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
        """Stage 1 loss: CE + auxiliary loss + sparsity loss (Paper Eq.11)."""
        device = output.device

        # Main classifier CE loss (Eq.3)
        ce_loss = F.cross_entropy(output, target)

        # Auxiliary classifier loss
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

        # Sparsity loss (Eq.10)
        sparsity_loss = torch.tensor(0.0, device=device)
        if (
            self.lambda_sparsity > 0
            and s is not None
            and hasattr(model, "get_sparsity_loss")
        ):
            sparsity_loss = model.get_sparsity_loss(s)

        self.current_batch += 1
        return ce_loss + self.lambda_aux * aux_loss + self.lambda_sparsity * sparsity_loss

    def _stage2_loss(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Stage 2 loss: CE with temperature scaling."""
        return F.cross_entropy(output / self.temperature, target)

    def _remap_aux_targets(self, target: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Remap labels for auxiliary classifier (old classes -> 'other')."""
        new_cls_map = {c: i for i, c in enumerate(self.new_classes)}
        other_idx = len(self.new_classes)
        remapped = torch.full_like(target, other_idx)
        for c, i in new_cls_map.items():
            remapped[target == c] = i
        return remapped

    def update_forgetting(self, task_accuracies: Dict[int, float]):
        """Track forgetting metrics."""
        self.current_acc_per_task = task_accuracies.copy()
        for tid, acc in task_accuracies.items():
            if tid not in self.best_acc_per_task:
                self.best_acc_per_task[tid] = acc
            else:
                self.best_acc_per_task[tid] = max(self.best_acc_per_task[tid], acc)

        if len(self.best_acc_per_task) > 1:
            forgetting_sum = 0.0
            count = 0
            for tid in self.best_acc_per_task:
                if tid != self.current_task and tid in self.current_acc_per_task:
                    f = self.best_acc_per_task[tid] - self.current_acc_per_task[tid]
                    forgetting_sum += max(0, f)
                    count += 1
            self.last_af = forgetting_sum / max(1, count)


# ============================================================================
# PlexusDERAggregator
# ============================================================================


class PlexusDERAggregator(BaseAggregator):
    """
    DER aggregator for Plexus decentralized setting.

    Extends PlexusAggregator with:
    - DER's frozen-parameter restoration (restore non-trainable params from global)
    - Task-boundary protocol for model expansion tracking
    - Piggybacked state exchange (task_classes_history, num_extractors)

    Key difference from DERAggregator:
    - Does NOT rely on server to set trainable_keys
    - Derives trainable_keys from piggybacked model structure info
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

        # DER-specific: trainable keys derived from model structure
        self.trainable_keys: Set[str] = set()

        # DER-specific: model structure tracking (piggybacked)
        self._num_extractors: int = 0
        self._task_classes_history: Dict[int, List[int]] = {}

    def set_model_structure(self, num_extractors: int, task_classes_history: Dict[int, List[int]]):
        """Set model structure info received via piggyback or local computation."""
        self._num_extractors = num_extractors
        self._task_classes_history = task_classes_history

    def derive_trainable_keys(self, model_state_dict: OrderedDict):
        """
        Derive trainable keys from model structure.

        In decentralized setting, each peer can derive this locally
        since model structure is synchronized via protocol.
        """
        trainable = set()
        current_task = self._num_extractors - 1 if self._num_extractors > 0 else 0

        for key in model_state_dict.keys():
            # Current extractor parameters
            if f"extractors.{current_task}." in key:
                trainable.add(key)
            # Classifier (always trainable in DER)
            elif "classifier" in key:
                trainable.add(key)
            # Aux classifier (always trainable)
            elif "aux_classifier" in key:
                trainable.add(key)
            # Mask embeds for current task
            elif f"extractors.{current_task}.mask_embeds" in key:
                trainable.add(key)

        self.trainable_keys = trainable
        return trainable

    def aggregate(
        self,
        results: List[Dict],
        global_params: Optional[OrderedDict] = None,
        **kwargs,
    ) -> OrderedDict:
        """
        Aggregate using FedAvg with DER's frozen-parameter restoration.

        Flow:
        1. Weighted average of all received params (server already filtered by success_fraction)
        2. Restore frozen (non-trainable) params from global model
        """
        if not results:
            return global_params

        # Update population view
        for r in results:
            cid = r.get("client_id", -1)
            self.population_view.update(cid, self.current_round, is_online=True)

        # Step 1: Weighted average
        agg = self._weighted_average(results)

        if agg is None:
            return global_params if global_params is not None else OrderedDict()

        # Step 2: Restore frozen params from global model
        if global_params is not None and self.trainable_keys:
            for k in agg:
                if k not in self.trainable_keys:
                    agg[k] = global_params[k].clone()

        return agg

    def get_round_aggregators(self, round_num: int, all_client_ids: List[int]) -> List[int]:
        """Bandwidth-based aggregator selection (from Plexus)."""
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
        """Hash-ordered sample selection (from Plexus)."""
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
