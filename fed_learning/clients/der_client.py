"""
DER Client - Client for Dynamically Expandable Representation.

Reference:
    "DER: Dynamically Expandable Representation for Class Incremental Learning"
    Yan, Xie, He (CVPR 2021)

Extends FederatedClient with:
1. Two-stage training (representation learning → classifier finetuning)
2. Exemplar replay buffer (reuses ReplayBuffer from FedCBDR)
3. Herding-based exemplar selection after each task
"""

import contextlib
import gc
from collections import OrderedDict
from typing import Optional, Dict, Any, List, Tuple, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.amp import autocast as torch_autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast as torch_autocast, GradScaler

from .client import FederatedClient
from ..core import BaseTrainer
from ..strategies.incremental.fedcbdr import ReplayBuffer


class DERClient(FederatedClient):
    """
    Client for DER algorithm with two-stage training and exemplar replay.

    Inherits from FederatedClient:
        - setup_for_gpu(model, device)
        - _create_batches(batch_size) — for task 0 (no replay)
        - _amp_ctx() — AMP context manager

    Reuses from FedCBDR:
        - ReplayBuffer — class-balanced storage with importance sampling

    Two-stage training per task:
        Stage 1 (Representation): Train new extractor + masks + classifiers
                                  Data = current task + replay exemplars
        Stage 2 (Classifier):     Train unified classifier only (balanced)
                                  Data = balanced sample from current + replay

    Attributes:
        replay_buffer: ReplayBuffer for exemplar storage
        current_task: Current task ID
        current_classes: Classes in current task
        seen_classes: All classes seen so far
    """

    def __init__(
        self,
        client_id: int,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        buffer_size: int = 500,
    ):
        super().__init__(client_id, X_train, y_train)

        # Exemplar buffer (reuse FedCBDR's ReplayBuffer)
        self.replay_buffer = ReplayBuffer(max_size=buffer_size)
        self.buffer_size = buffer_size

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
        """
        Set data for current task.

        Args:
            X_train: Features for current task
            y_train: Labels for current task
            task_id: Task identifier
            task_classes: List of class IDs in this task
        """
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
        stage: int = 1,
        replay_ratio: float = 0.5,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Two-stage training for DER.

        Stage 1 (Representation Learning):
            - Trainable: current extractor + masks + classifier + aux_classifier
            - Data: current task data + replay exemplars
            - Loss: L_CE + λ_a * L_aux + λ_s * L_sparsity

        Stage 2 (Classifier Finetuning):
            - Trainable: classifier only (all extractors frozen)
            - Data: class-balanced sample from current + replay
            - Loss: CE with temperature scaling

        Args:
            trainer: DERTrainer instance
            epochs: Number of local epochs
            batch_size: Batch size
            lr: Learning rate
            global_params: Global model parameters
            stage: Training stage (1 or 2)
            replay_ratio: Ratio of replay samples in combined batches
            **kwargs: Additional arguments

        Returns:
            Dict with client_id, num_samples, loss, params
        """
        self.model.train()

        # Select parameters based on training stage
        if stage == 2 and hasattr(self.model, 'get_classifier_params'):
            # Stage 2: only classifier
            self.model.freeze_all_extractors()
            opt_params = self.model.get_classifier_params()
        elif hasattr(self.model, 'get_trainable_params'):
            # Stage 1: current extractor + classifiers + masks
            opt_params = self.model.get_trainable_params()
        else:
            opt_params = list(self.model.parameters())

        if not opt_params:
            # Nothing to train
            return {
                "client_id": self.client_id,
                "num_samples": self.num_samples,
                "loss": 0.0,
                "params": OrderedDict(
                    (k, v.cpu().clone()) for k, v in self.model.state_dict().items()
                ),
            }

        optimizer_cls = trainer.get_optimizer_class()
        optimizer = optimizer_cls(opt_params, lr=lr)
        scaler = GradScaler(enabled=self.use_amp)

        # Notify trainer of stage
        if hasattr(trainer, 'set_stage'):
            trainer.set_stage(stage)

        # Pre-train hook
        trainer.pre_train(self.model, global_params, lr=lr, **kwargs)

        total_loss = 0.0
        total_samples = 0

        # Determine batch generator
        has_replay = self.replay_buffer.total_samples > 0 and self.current_task > 0

        for ep in range(epochs):
            # Paper Eq.8: annealing schedule resets at the beginning of EACH epoch.
            # b goes from 1 to B within each epoch (not monotonically across epochs).
            if stage == 1 and hasattr(trainer, 'current_batch'):
                trainer.current_batch = 0

            if stage == 2:
                # Stage 2: class-balanced batches
                batch_gen = self._create_balanced_batches(batch_size)
            elif has_replay:
                # Stage 1 with replay: combined current + exemplar batches
                batch_gen = self._create_combined_batches(batch_size, replay_ratio)
            else:
                # Stage 1 task 0: no replay, use base class method
                batch_gen = self._create_batches(batch_size)

            for X_batch, y_batch in batch_gen:
                optimizer.zero_grad()

                with self._amp_ctx():
                    # DERModel.forward() accepts s parameter for mask annealing
                    # Trainer computes s internally via compute_annealing_s()
                    s = None
                    if stage == 1 and hasattr(trainer, 'compute_annealing_s'):
                        s = trainer.compute_annealing_s()

                    out = self.model(X_batch, s=s)
                    loss = trainer.compute_loss(
                        self.model, out, y_batch, global_params,
                        inputs=X_batch, s=s, **kwargs
                    )

                if self.use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    # Eq.9: gradient compensation BEFORE clip (on raw unscaled grads)
                    if stage == 1 and s is not None and hasattr(self.model, 'compensate_mask_gradients'):
                        self.model.compensate_mask_gradients(s)
                    torch.nn.utils.clip_grad_norm_(opt_params, max_norm=1.0)
                    trainer.pre_step(self.model, global_params, **kwargs)
                    scaler.step(optimizer)
                    scaler.update()
                    trainer.post_step(self.model, global_params, **kwargs)
                else:
                    loss.backward()
                    # Eq.9: gradient compensation BEFORE clip (on raw grads)
                    if stage == 1 and s is not None and hasattr(self.model, 'compensate_mask_gradients'):
                        self.model.compensate_mask_gradients(s)
                    torch.nn.utils.clip_grad_norm_(opt_params, max_norm=1.0)
                    trainer.pre_step(self.model, global_params, **kwargs)
                    optimizer.step()
                    trainer.post_step(self.model, global_params, **kwargs)

                bs = len(y_batch)
                total_loss += loss.item() * bs
                total_samples += bs

        # Post-train hook
        trainer.post_train(self.model, global_params, **kwargs)

        # Restore extractor if we froze it for Stage 2
        if stage == 2 and hasattr(self.model, 'unfreeze_current_extractor'):
            self.model.unfreeze_current_extractor()

        return {
            "client_id": self.client_id,
            "num_samples": self.num_samples,
            "loss": total_loss / max(1, total_samples),
            "params": OrderedDict(
                (k, v.cpu().clone()) for k, v in self.model.state_dict().items()
            ),
            "replay_samples": self.replay_buffer.total_samples,
            "replay_classes": self.replay_buffer.num_classes,
        }

    # =========================================================================
    # Batch generators
    # =========================================================================

    def _create_combined_batches(self, batch_size: int, replay_ratio: float = 0.5):
        """
        Create batches combining current task data and replay exemplars.

        Same pattern as FedCBDRClient._create_combined_batches.

        Args:
            batch_size: Total batch size
            replay_ratio: Ratio of replay samples in each batch

        Yields:
            (X_batch, y_batch) combined from current + replay
        """
        X_replay, y_replay = self.replay_buffer.get_all_samples()
        has_replay = X_replay is not None and len(y_replay) > 0

        if not has_replay:
            yield from self._create_batches(batch_size)
            return

        replay_batch_size = int(batch_size * replay_ratio)
        current_batch_size = batch_size - replay_batch_size

        current_indices = torch.randperm(self.num_samples)
        replay_indices = torch.randperm(len(y_replay))

        n_batches = max(1, self.num_samples // max(1, current_batch_size))

        for b in range(n_batches):
            # Current task samples
            start_c = b * current_batch_size
            end_c = min(start_c + current_batch_size, self.num_samples)
            idx_c = current_indices[start_c:end_c]

            X_current = self.X_train[idx_c].to(self.device, non_blocking=True)
            y_current = self.y_train[idx_c].to(self.device, non_blocking=True)

            # Replay samples (cycle if needed)
            start_r = (b * replay_batch_size) % len(y_replay)
            end_r = start_r + replay_batch_size

            if end_r <= len(y_replay):
                idx_r = replay_indices[start_r:end_r]
            else:
                idx_r = torch.cat([
                    replay_indices[start_r:],
                    replay_indices[:end_r - len(y_replay)]
                ])

            X_rep = X_replay[idx_r].to(self.device, non_blocking=True)
            y_rep = y_replay[idx_r].to(self.device, non_blocking=True)

            # Combine and shuffle
            X_batch = torch.cat([X_current, X_rep], dim=0)
            y_batch = torch.cat([y_current, y_rep], dim=0)

            perm = torch.randperm(len(y_batch))
            yield X_batch[perm], y_batch[perm]

    def _create_balanced_batches(self, batch_size: int):
        """
        Create class-balanced batches for Stage 2 classifier finetuning.

        Paper: Stage 2 uses balanced subset from D_tilde_t = D_t ∪ M_t.
        Equal samples per class from current task data + replay buffer.

        Args:
            batch_size: Total batch size

        Yields:
            (X_batch, y_batch) with balanced class representation
        """
        # Collect all available data: current task + replay
        all_X = [self.X_train]
        all_y = [self.y_train]

        X_replay, y_replay = self.replay_buffer.get_all_samples()
        if X_replay is not None and len(y_replay) > 0:
            all_X.append(X_replay)
            all_y.append(y_replay)

        combined_X = torch.cat(all_X, dim=0)
        combined_y = torch.cat(all_y, dim=0)

        # Group indices by class
        class_indices = {}
        for cls in self.seen_classes:
            mask = (combined_y == cls)
            if mask.any():
                class_indices[cls] = torch.where(mask)[0]

        if not class_indices:
            yield from self._create_batches(batch_size)
            return

        # Samples per class per batch
        n_classes = len(class_indices)
        samples_per_class = max(1, batch_size // n_classes)

        # Total samples to generate (approximately one pass through data)
        total_available = sum(len(idx) for idx in class_indices.values())
        n_batches = max(1, total_available // batch_size)

        # PRE-SHUFFLE indices ONCE per epoch (not per batch).
        # Previous implementation called torch.randperm(N_class) for every class
        # for every batch → e.g. 300k-element randperm × 10 classes × 1000 batches
        # = 10M randperm calls, causing 17x slowdown vs _create_batches.
        # Now: 10 randperm calls total, then use contiguous slices per batch.
        shuffled_class_idx = {
            cls: indices[torch.randperm(len(indices))]
            for cls, indices in class_indices.items()
        }

        for b in range(n_batches):
            batch_X = []
            batch_y = []

            for cls, shuf_idx in shuffled_class_idx.items():
                n_sel = min(samples_per_class, len(shuf_idx))
                # Cycle through pre-shuffled indices with contiguous slices
                start = (b * n_sel) % len(shuf_idx)
                end = start + n_sel
                if end <= len(shuf_idx):
                    sel = shuf_idx[start:end]  # contiguous view — fast
                else:
                    # Wrap around end of shuffled array
                    sel = torch.cat([shuf_idx[start:], shuf_idx[:end - len(shuf_idx)]])
                batch_X.append(combined_X[sel])
                batch_y.append(combined_y[sel])

            X_batch = torch.cat(batch_X, dim=0).to(self.device, non_blocking=True)
            y_batch = torch.cat(batch_y, dim=0).to(self.device, non_blocking=True)

            perm = torch.randperm(len(y_batch))
            yield X_batch[perm], y_batch[perm]

    # =========================================================================
    # Exemplar management
    # =========================================================================

    def update_exemplars(self, model: nn.Module, batch_size: int = 128):
        """
        Update exemplar buffer after task completion using herding selection.

        Paper: Fixed memory budget, herding selects samples closest to class mean
        in feature space. Balanced across all seen classes.

        Memory-efficient: processes ONE class at a time (instead of extracting
        features for all samples at once, which caused OOM for large datasets).

        Args:
            model: Trained global model for feature extraction
            batch_size: Batch size for feature extraction
        """
        if self.num_samples == 0:
            return

        device = next(model.parameters()).device
        model.eval()

        # Step 1: Determine how many exemplars to keep per class
        samples_per_class = self.buffer_size // max(1, len(self.seen_classes))
        samples_per_class = max(1, samples_per_class)

        # Step 2: Herding selection CLASS BY CLASS (memory efficient)
        # Old approach: extract features for ALL samples → OOM for large datasets
        # (e.g., 3M samples × 1124 features × 4 bytes = 13+ GB CPU tensor)
        # New approach: process one class at a time, max 2000 candidates per class
        # → max 2000 × 1124 × 4 = ~9 MB at any moment
        MAX_CANDIDATES_PER_CLASS = 2000

        selected_X = []
        selected_y = []

        for cls in self.current_classes:
            mask = (self.y_train == cls)
            if not mask.any():
                continue

            cls_X = self.X_train[mask]
            cls_y = self.y_train[mask]
            n_select = min(samples_per_class, len(cls_y))

            if n_select >= len(cls_y):
                # Take all samples for this class (not enough to require herding)
                selected_X.append(cls_X)
                selected_y.append(cls_y)
                continue

            # Limit candidates for feature extraction (saves memory)
            n_cand = min(len(cls_y), MAX_CANDIDATES_PER_CLASS)
            if n_cand < len(cls_y):
                cand_idx = torch.randperm(len(cls_y))[:n_cand]
                cls_X_cand = cls_X[cand_idx]
            else:
                cls_X_cand = cls_X

            # Extract features for this class's candidates only
            cls_features = []
            with torch.no_grad():
                for i in range(0, len(cls_X_cand), batch_size):
                    X_batch = cls_X_cand[i:i + batch_size].to(device)
                    if hasattr(model, 'extractors') and model.current_task >= 0:
                        # Use s_max for binarized masks during feature extraction
                        s = getattr(model, '_s_max', 15.0)
                        feat = model.extractors[model.current_task](X_batch, s=s)
                    elif hasattr(model, 'get_fused_representation'):
                        feat = model.get_fused_representation(X_batch)
                    else:
                        feat = model(X_batch)
                    cls_features.append(feat.cpu())
                    del X_batch, feat

            cls_features = torch.cat(cls_features, dim=0)  # [n_cand, feat_dim]

            # Herding: iteratively select samples closest to running mean
            class_mean = cls_features.mean(dim=0)
            herding_indices = []
            current_sum = torch.zeros_like(class_mean)
            selected_mask = torch.zeros(len(cls_features), dtype=torch.bool)

            for _ in range(n_select):
                target = (len(herding_indices) + 1) * class_mean
                distances = ((cls_features + current_sum - target) ** 2).sum(dim=1)
                distances[selected_mask] = float('inf')  # exclude already selected
                best_idx = distances.argmin().item()
                herding_indices.append(best_idx)
                current_sum = current_sum + cls_features[best_idx]
                selected_mask[best_idx] = True

            indices = torch.tensor(herding_indices)
            selected_X.append(cls_X_cand[indices])
            selected_y.append(cls_y[cand_idx][indices] if n_cand < len(cls_y) else cls_y[indices])

            del cls_features
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Step 3: Add to replay buffer
        if selected_X:
            X_sel = torch.cat(selected_X, dim=0)
            y_sel = torch.cat(selected_y, dim=0)
            # Uniform importance (herding already selects representative samples)
            imp_sel = torch.ones(len(y_sel)) / len(y_sel)

            self.replay_buffer.add_samples(
                X_sel, y_sel, imp_sel,
                class_ids=list(self.current_classes)
            )

            print(f"    Client {self.client_id}: Exemplars updated (herding), "
                  f"total={self.replay_buffer.total_samples} samples, "
                  f"classes={self.replay_buffer.num_classes}")

    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get exemplar buffer statistics."""
        return {
            "total_samples": self.replay_buffer.total_samples,
            "num_classes": self.replay_buffer.num_classes,
            "class_distribution": self.replay_buffer.get_class_distribution(),
            "buffer_size": self.buffer_size,
        }
