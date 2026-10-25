"""
Re-Fed Client - Client with Personalized Informative Model for sample caching.

Reference:
    Li et al., "Towards Efficient Replay in Federated Incremental Learning",
    CVPR 2024

Extends FederatedClient with:
1. Personalized Informative Model (PIM) for importance-aware sample selection
2. Gradient-norm-based sample importance scoring with early-emphasis
3. Memory buffer for cached samples (replay)
4. Training on combined cached + new task data
"""

import contextlib
import copy
from collections import OrderedDict
from typing import Optional, Dict, Any, List, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    from torch.amp import autocast as torch_autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast as torch_autocast, GradScaler

from .client import FederatedClient
from ..core import BaseTrainer


class ReFedClient(FederatedClient):
    """
    Client for Re-Fed algorithm.

    Implements per-client sample caching with PIM:
    - Maintains a Personalized Informative Model (PIM) that blends
      local and global knowledge (Paper Eq. 3)
    - Computes sample importance via gradient norms during PIM update
      with early-emphasis weighting (Paper Eq. 5)
    - Caches high-importance samples for replay
    - Trains local model on cached + new task data (Paper Eq. 6)
    """

    def __init__(
        self,
        client_id: int,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        memory_size: int = 2000,
        lambda_pim: float = 0.5,
        pim_iterations: int = 5,
    ):
        super().__init__(client_id, X_train, y_train)

        # Memory buffer: cached samples for replay
        self.memory_size = memory_size
        self.cached_X: Optional[torch.Tensor] = None
        self.cached_y: Optional[torch.Tensor] = None

        # PIM parameters (Paper Eq. 3)
        self.lambda_pim = lambda_pim  # Balance local vs global info
        self.pim_iterations = pim_iterations  # Number of PIM update iterations (s)
        self.q_lambda = (1 - lambda_pim) / (2 * lambda_pim)  # q(λ) = (1-λ)/(2λ)

        # Task tracking
        self.current_task: int = 0
        self.seen_classes: Set[int] = set()
        self.old_classes: List[int] = []
        self.new_classes: List[int] = []

        # Original data storage (before mixing with cached)
        self._original_X: Optional[torch.Tensor] = None
        self._original_y: Optional[torch.Tensor] = None

        # Previous task local samples for PIM importance scoring
        self._prev_task_X: Optional[torch.Tensor] = None
        self._prev_task_y: Optional[torch.Tensor] = None

    def set_task_data(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        task_id: int,
        task_classes: List[int],
    ):
        """
        Set data for a new task.

        When a new task arrives:
        1. Save previous task's FULL local data for PIM importance scoring
        2. Update task tracking
        3. Store original new data before mixing with cache
        """
        # Save previous task's FULL local data (cached + new) for PIM scoring
        # Paper: T^{t-1}_{k,local} = T^{t-2}_{k,cached} + T^{t-1}_k (Eq. 2)
        # At this point, self.X_train still contains the mixed data from
        # the previous task (cached old + task data), which is exactly
        # what PIM needs to score for re-caching decisions.
        if self._original_X is not None:
            self._prev_task_X = self.X_train.clone()
            self._prev_task_y = self.y_train.clone()

        # Store original new data
        self._original_X = X_train.clone()
        self._original_y = y_train.clone()

        # Update task tracking
        self.old_classes = list(self.seen_classes)
        self.new_classes = list(task_classes)
        self.current_task = task_id
        self.seen_classes.update(task_classes)

        # Set training data (will be mixed with cache later)
        self.X_train = X_train
        self.y_train = y_train
        self.num_samples = len(y_train)

    def update_cache_with_pim(
        self,
        model: nn.Module,
        global_params: OrderedDict,
        device: str,
    ):
        """
        Update sample cache using PIM importance scoring.

        Paper Algorithm 1, Steps 5-9:
        1. Initialize PIM from current local model
        2. Update PIM on previous samples with momentum toward global model (Eq. 3)
        3. During PIM update, accumulate gradient norms per sample (Eq. 4-5)
        4. Cache samples with highest importance scores

        Args:
            model: Current local model
            global_params: Global model parameters (w^{t-1})
            device: Device to compute on
        """
        if self._prev_task_X is None or len(self._prev_task_X) == 0:
            # First task: no previous samples to score
            # Cache a random subset if data is larger than memory
            self._cache_random_subset()
            self._mix_data_with_cache()
            return

        # Initialize PIM as copy of current local model (Paper: starts from local model)
        pim = copy.deepcopy(model)
        pim.to(device)
        pim.train()

        # Global model params on device for momentum term
        global_params_device = {k: v.to(device) for k, v in global_params.items()}

        # Previous samples to evaluate
        prev_X = self._prev_task_X
        prev_y = self._prev_task_y
        n_prev = len(prev_y)

        # Initialize importance scores
        importance_scores = torch.zeros(n_prev, dtype=torch.float32)

        # Learning rate for PIM update
        lr = 0.001
        batch_size = 128

        # PIM update for s iterations (Paper Eq. 3)
        for p in range(1, self.pim_iterations + 1):
            # Process all previous samples in batches
            indices = torch.randperm(n_prev)

            for start in range(0, n_prev, batch_size):
                batch_idx = indices[start:start + batch_size]
                X_batch = prev_X[batch_idx].to(device, non_blocking=True)
                y_batch = prev_y[batch_idx].to(device, non_blocking=True)

                # Compute per-sample gradient norms (Paper Eq. 4)
                sample_grad_norms = self._compute_sample_gradient_norms(
                    pim, X_batch, y_batch
                )

                # Accumulate with early-emphasis weighting: 1/p (Paper Eq. 5)
                importance_scores[batch_idx] += (1.0 / p) * sample_grad_norms.cpu()

                # PIM update (Paper Eq. 3):
                # v_{k,s} = v_{k,s-1} - η * (∇l(f_v(x̃), ỹ) + q(λ)(v - w))
                pim.zero_grad()
                output = pim(X_batch)
                loss = F.cross_entropy(output, y_batch)
                loss.backward()

                # Apply momentum-like update with global model
                with torch.no_grad():
                    for name, param in pim.named_parameters():
                        if param.grad is not None:
                            # Standard gradient term
                            grad = param.grad

                            # Momentum term toward global model: q(λ)(v - w)
                            if name in global_params_device:
                                momentum = self.q_lambda * (param - global_params_device[name])
                                grad = grad + momentum

                            param -= lr * grad

        # Select top-M samples with highest importance
        self._select_and_cache(prev_X, prev_y, importance_scores)

        # Mix cached samples with new task data
        self._mix_data_with_cache()

        # Cleanup PIM
        del pim
        if "cuda" in device:
            torch.cuda.empty_cache()

    def _compute_sample_gradient_norms(
        self,
        model: nn.Module,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-sample gradient norms (Paper Eq. 4).

        G_p(x̃) = ||∇l(f_v(x̃), ỹ)||_2

        Uses per-sample gradient approximation: compute loss per sample,
        accumulate gradient norms from individual backward passes.
        For efficiency, use batch gradient and approximate per-sample norms.
        """
        batch_size = len(y_batch)
        grad_norms = torch.zeros(batch_size, device=X_batch.device)

        # Per-sample approach: compute individual losses
        model.eval()  # Temporarily eval to avoid dropout variation
        for i in range(batch_size):
            model.zero_grad()
            x_i = X_batch[i:i+1]
            y_i = y_batch[i:i+1]
            out_i = model(x_i)
            loss_i = F.cross_entropy(out_i, y_i)
            loss_i.backward()

            # Compute gradient norm across all parameters
            total_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    total_norm += param.grad.norm(2).item() ** 2
            grad_norms[i] = total_norm ** 0.5

        model.train()
        return grad_norms

    def _select_and_cache(
        self,
        X_data: torch.Tensor,
        y_data: torch.Tensor,
        importance_scores: torch.Tensor,
    ):
        """
        Select top-M important samples and cache them.

        Paper Section 3.2: Cache samples with higher importance scores
        based on local storage M.

        Maintains class balance: allocate memory equally across all seen classes,
        then select top-importance samples within each class.
        """
        unique_classes = y_data.unique().tolist()
        n_classes = max(1, len(unique_classes))
        samples_per_class = max(1, self.memory_size // n_classes)

        cached_X_list = []
        cached_y_list = []

        for cls in unique_classes:
            mask = y_data == cls
            cls_X = X_data[mask]
            cls_scores = importance_scores[mask]
            n_select = min(samples_per_class, len(cls_X))

            if n_select > 0:
                # Sort by importance (descending) and take top-k
                _, top_indices = cls_scores.topk(n_select)
                cached_X_list.append(cls_X[top_indices])
                cached_y_list.append(torch.full((n_select,), cls, dtype=torch.long))

        if cached_X_list:
            # Merge with existing cache from even older tasks
            if self.cached_X is not None and len(self.cached_y) > 0:
                # Rebalance: total budget = memory_size
                # old cache classes + new scored classes
                old_cache_classes = self.cached_y.unique().tolist()
                all_classes = list(set(old_cache_classes + unique_classes))
                new_per_class = max(1, self.memory_size // max(1, len(all_classes)))

                # Trim old cache to new per-class budget
                trimmed_X = []
                trimmed_y = []
                for cls in old_cache_classes:
                    if cls not in unique_classes:
                        # Keep old classes (already cached, not re-scored)
                        cls_mask = self.cached_y == cls
                        cls_data = self.cached_X[cls_mask]
                        n_keep = min(new_per_class, len(cls_data))
                        trimmed_X.append(cls_data[:n_keep])
                        trimmed_y.append(torch.full((n_keep,), cls, dtype=torch.long))

                # Re-scored classes: use new selection (trim to new budget)
                for cls_X, cls_y in zip(cached_X_list, cached_y_list):
                    n_keep = min(new_per_class, len(cls_y))
                    trimmed_X.append(cls_X[:n_keep])
                    trimmed_y.append(cls_y[:n_keep])

                if trimmed_X:
                    self.cached_X = torch.cat(trimmed_X, dim=0)
                    self.cached_y = torch.cat(trimmed_y, dim=0)
            else:
                self.cached_X = torch.cat(cached_X_list, dim=0)
                self.cached_y = torch.cat(cached_y_list, dim=0)
        # If no samples selected, keep existing cache
        elif self.cached_X is None:
            self.cached_X = torch.tensor([])
            self.cached_y = torch.tensor([], dtype=torch.long)

    def _cache_random_subset(self):
        """
        Cache a random subset of current data (for first task).

        When there are no previous samples to score, use random selection
        with class-balanced allocation.
        """
        if self._original_X is None:
            return

        X_data = self._original_X
        y_data = self._original_y
        unique_classes = y_data.unique().tolist()
        n_classes = max(1, len(unique_classes))
        samples_per_class = max(1, self.memory_size // n_classes)

        cached_X_list = []
        cached_y_list = []

        for cls in unique_classes:
            mask = y_data == cls
            cls_X = X_data[mask]
            n_select = min(samples_per_class, len(cls_X))

            if n_select > 0:
                # Random selection
                perm = torch.randperm(len(cls_X))[:n_select]
                cached_X_list.append(cls_X[perm])
                cached_y_list.append(torch.full((n_select,), cls, dtype=torch.long))

        if cached_X_list:
            if self.cached_X is not None and len(self.cached_y) > 0:
                cached_X_list.insert(0, self.cached_X)
                cached_y_list.insert(0, self.cached_y)
            self.cached_X = torch.cat(cached_X_list, dim=0)
            self.cached_y = torch.cat(cached_y_list, dim=0)

            # Trim to budget if needed
            if len(self.cached_y) > self.memory_size:
                perm = torch.randperm(len(self.cached_y))[:self.memory_size]
                self.cached_X = self.cached_X[perm]
                self.cached_y = self.cached_y[perm]

    def _mix_data_with_cache(self):
        """
        Mix new task data with cached replay data.

        Paper Eq. 6: Train on both cached samples and new task samples.
        """
        if self.cached_X is None or len(self.cached_y) == 0:
            return

        if self._original_X is None:
            return

        self.X_train = torch.cat([self._original_X, self.cached_X], dim=0)
        self.y_train = torch.cat([self._original_y, self.cached_y], dim=0)
        self.num_samples = len(self.y_train)

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
        Train local model on combined data (cached + new).

        Paper Eq. 6: Standard local training on combined data.
        Uses the base FederatedClient.train() logic since Re-Fed's
        training loss is standard CE.
        """
        return super().train(
            trainer=trainer,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            global_params=global_params,
            **kwargs,
        )
