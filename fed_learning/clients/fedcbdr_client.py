"""
FedCBDR Client - Client with Replay Buffer for Class Incremental Learning.

Reference:
    "Class-wise Balancing Data Replay for Federated Class-Incremental Learning"
    Zhuang Qi et al., arXiv:2507.07712, 2025

This client extends FederatedClient with:
1. Memory buffer for experience replay
2. Feature extraction for leverage score computation
3. Combined training on new data + replay buffer
"""

import contextlib
from collections import OrderedDict
from typing import Optional, Dict, Any, List, Tuple, Set

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

try:
    from torch.amp import autocast as torch_autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast as torch_autocast, GradScaler

from .client import FederatedClient
from ..core import BaseTrainer
from ..strategies.incremental.fedcbdr import ReplayBuffer, LeverageScoreCalculator


class FedCBDRClient(FederatedClient):
    """
    Client for FedCBDR algorithm with replay buffer.
    
    Extends FederatedClient with:
    - Memory buffer for storing old task samples
    - Feature extraction for GDR (Global-perspective Data Replay)
    - Combined training on current task + replay data
    
    Attributes:
        replay_buffer: ReplayBuffer for old task samples
        leverage_calculator: LeverageScoreCalculator for importance sampling
        current_task: Current task ID
        current_classes: Classes in current task
    """
    
    def __init__(
        self,
        client_id: int,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        buffer_size: int = 500,
        leverage_rank: int = 50
    ):
        """
        Initialize FedCBDR client.
        
        Args:
            client_id: Client identifier
            X_train: Training features (CPU tensor)
            y_train: Training labels (CPU tensor)
            buffer_size: Maximum replay buffer size
            leverage_rank: Rank for leverage score SVD
        """
        super().__init__(client_id, X_train, y_train)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(max_size=buffer_size)
        self.buffer_size = buffer_size
        
        # Leverage score calculator
        self.leverage_calculator = LeverageScoreCalculator(rank=leverage_rank)
        
        # Task tracking
        self.current_task: int = 0
        self.current_classes: Set[int] = set()
        self.seen_classes: Set[int] = set()
        
        # Store original data reference (before task filtering)
        self.X_original = X_train
        self.y_original = y_train
    
    def set_task_data(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        task_id: int,
        task_classes: List[int]
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
    
    def update_replay_buffer_simple(self):
        """
        Simple replay buffer update without feature extraction (fallback for OOM).
        
        Randomly samples from current task data.
        """
        if self.num_samples == 0:
            return
        
        samples_per_class = self.buffer_size // max(1, len(self.seen_classes))
        
        X_selected = []
        y_selected = []
        
        for cls in self.current_classes:
            mask = (self.y_train == cls)
            if not mask.any():
                continue
            
            cls_X = self.X_train[mask]
            cls_y = self.y_train[mask]
            
            n_select = min(samples_per_class, len(cls_y))
            indices = torch.randperm(len(cls_y))[:n_select]
            
            X_selected.append(cls_X[indices].cpu())
            y_selected.append(cls_y[indices].cpu())
        
        if X_selected:
            X_selected = torch.cat(X_selected, dim=0)
            y_selected = torch.cat(y_selected, dim=0)
            imp_selected = torch.ones(len(y_selected)) / len(y_selected)
            
            self.replay_buffer.add_samples(
                X_selected, y_selected, imp_selected,
                class_ids=list(self.current_classes)
            )
            
            print(f"    Client {self.client_id}: Buffer updated (simple), "
                  f"total={self.replay_buffer.total_samples} samples")
    
    def update_replay_buffer(
        self,
        model: nn.Module,
        selected_indices: Optional[torch.Tensor] = None,
        use_herding: bool = True
    ):
        """
        Update replay buffer after task completion.
        
        Uses importance-based sampling if indices not provided,
        or herding selection for representative samples.
        
        Args:
            model: Trained model for feature extraction
            selected_indices: Pre-selected indices (from GDR server coordination)
            use_herding: Whether to use herding for sample selection
        """
        import gc
        
        if self.num_samples == 0:
            return
        
        # If we have pre-selected indices, skip feature extraction entirely
        # This saves significant memory on Kaggle
        if selected_indices is not None:
            X_selected = self.X_train[selected_indices].cpu()
            y_selected = self.y_train[selected_indices].cpu()
            # Use uniform importance for pre-selected samples
            imp_selected = torch.ones(len(selected_indices)) / len(selected_indices)
            
            # Add to replay buffer
            self.replay_buffer.add_samples(
                X_selected, y_selected, imp_selected,
                class_ids=list(self.current_classes)
            )
            
            print(f"    Client {self.client_id}: Buffer updated (GDR), "
                  f"total={self.replay_buffer.total_samples} samples")
            return
        
        # Only extract features if no pre-selected indices
        device = next(model.parameters()).device
        
        # Compute importance scores for current task data
        with torch.no_grad():
            model.eval()
            
            # Process in batches to avoid OOM
            batch_size = 128  # Smaller batch for memory efficiency
            all_features = []
            
            for i in range(0, self.num_samples, batch_size):
                X_batch = self.X_train[i:i+batch_size].to(device)
                
                # Get features (before classifier)
                if hasattr(model, 'get_fused_representation'):
                    features = model.get_fused_representation(X_batch)
                else:
                    # Fallback: use penultimate layer output
                    features = self._extract_features(model, X_batch)
                
                all_features.append(features.cpu())
                
                # Clean up GPU memory
                del X_batch, features
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            features = torch.cat(all_features, dim=0)
            del all_features
            gc.collect()
        
        # Compute leverage scores
        importance_scores = self.leverage_calculator.compute_scores(
            features, normalize=True
        )
        
        if use_herding:
            # Use herding selection
            X_selected, y_selected, imp_selected = self._herding_selection(
                features, importance_scores
            )
        else:
            # Use importance-based random sampling
            X_selected, y_selected, imp_selected = self._importance_sampling(
                importance_scores
            )
        
        # Clean up features
        del features, importance_scores
        gc.collect()
        
        # Add to replay buffer
        self.replay_buffer.add_samples(
            X_selected, y_selected, imp_selected,
            class_ids=list(self.current_classes)
        )
        
        print(f"    Client {self.client_id}: Buffer updated, "
              f"total={self.replay_buffer.total_samples} samples, "
              f"classes={self.replay_buffer.num_classes}")
    
    def _extract_features(
        self,
        model: nn.Module,
        X: torch.Tensor
    ) -> torch.Tensor:
        """Extract features from model (fallback if get_fused_representation not available)."""
        # Try to get features from model's forward hooks or intermediate layers
        features = []
        
        def hook_fn(module, input, output):
            features.append(output)
        
        # Register hook on the layer before classifier
        # This is model-specific, adjust as needed
        handle = None
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, nn.Linear) and 'output' not in name.lower():
                # Found a linear layer that's not the output
                handle = module.register_forward_hook(hook_fn)
                break
        
        if handle is None:
            # Fallback: just return flattened output
            output = model(X)
            return output.detach()
        
        # Forward pass
        _ = model(X)
        handle.remove()
        
        if features:
            return features[0].detach()
        else:
            return model(X).detach()
    
    def _herding_selection(
        self,
        features: torch.Tensor,
        importance_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select samples using herding (closest to class mean).
        
        Herding selects samples that best approximate the class mean
        in the feature space.
        """
        # Calculate samples per class for this task
        samples_per_class = self.buffer_size // max(1, len(self.seen_classes))
        samples_per_class = max(1, samples_per_class)
        
        selected_X = []
        selected_y = []
        selected_imp = []
        
        for cls in self.current_classes:
            mask = (self.y_train == cls)
            if not mask.any():
                continue
            
            cls_features = features[mask]
            cls_X = self.X_train[mask]
            cls_y = self.y_train[mask]
            cls_imp = importance_scores[mask]
            
            n_select = min(samples_per_class, len(cls_y))
            
            if n_select >= len(cls_y):
                # Take all samples
                selected_X.append(cls_X)
                selected_y.append(cls_y)
                selected_imp.append(cls_imp)
            else:
                # Herding: iteratively select samples closest to running mean
                class_mean = cls_features.mean(dim=0)
                selected_indices = []
                current_sum = torch.zeros_like(class_mean)
                
                for _ in range(n_select):
                    # Find sample that, when added, brings sum closest to (i+1)*mean
                    target = (len(selected_indices) + 1) * class_mean
                    distances = ((cls_features + current_sum - target) ** 2).sum(dim=1)
                    
                    # Mask already selected
                    for idx in selected_indices:
                        distances[idx] = float('inf')
                    
                    best_idx = distances.argmin().item()
                    selected_indices.append(best_idx)
                    current_sum = current_sum + cls_features[best_idx]
                
                indices = torch.tensor(selected_indices)
                selected_X.append(cls_X[indices])
                selected_y.append(cls_y[indices])
                selected_imp.append(cls_imp[indices])
        
        if not selected_X:
            return self.X_train[:1], self.y_train[:1], importance_scores[:1]
        
        return (
            torch.cat(selected_X, dim=0),
            torch.cat(selected_y, dim=0),
            torch.cat(selected_imp, dim=0)
        )
    
    def _importance_sampling(
        self,
        importance_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select samples using importance-based sampling.
        
        Higher importance score = higher probability of selection.
        """
        samples_per_class = self.buffer_size // max(1, len(self.seen_classes))
        
        selected_X = []
        selected_y = []
        selected_imp = []
        
        for cls in self.current_classes:
            mask = (self.y_train == cls)
            if not mask.any():
                continue
            
            cls_X = self.X_train[mask]
            cls_y = self.y_train[mask]
            cls_imp = importance_scores[mask]
            
            n_select = min(samples_per_class, len(cls_y))
            
            # Sample based on importance
            probs = cls_imp / (cls_imp.sum() + 1e-8)
            indices = torch.multinomial(probs, n_select, replacement=False)
            
            selected_X.append(cls_X[indices])
            selected_y.append(cls_y[indices])
            selected_imp.append(cls_imp[indices])
        
        if not selected_X:
            return self.X_train[:1], self.y_train[:1], importance_scores[:1]
        
        return (
            torch.cat(selected_X, dim=0),
            torch.cat(selected_y, dim=0),
            torch.cat(selected_imp, dim=0)
        )
    
    def extract_features_for_gdr(
        self,
        model: nn.Module,
        batch_size: int = 128
    ) -> torch.Tensor:
        """
        Extract features for Global-perspective Data Replay (GDR).
        
        These features are sent to server for global SVD and
        leverage score computation.
        
        MEMORY OPTIMIZED: Uses smaller batches and explicit cleanup.
        
        Args:
            model: Global model for feature extraction
            batch_size: Batch size for processing (default 128 for memory efficiency)
            
        Returns:
            Feature matrix [N, D] for current task data (on CPU)
        """
        import gc
        
        if self.num_samples == 0:
            return torch.empty(0)
        
        device = next(model.parameters()).device
        
        with torch.no_grad():
            model.eval()
            all_features = []
            
            for i in range(0, self.num_samples, batch_size):
                X_batch = self.X_train[i:i+batch_size].to(device)
                
                if hasattr(model, 'get_fused_representation'):
                    features = model.get_fused_representation(X_batch)
                else:
                    features = self._extract_features(model, X_batch)
                
                # Move to CPU immediately
                all_features.append(features.cpu())
                
                # Clean up GPU memory after each batch
                del X_batch, features
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Concatenate on CPU
            result = torch.cat(all_features, dim=0)
            del all_features
            gc.collect()
        
        return result
    
    def _create_combined_batches(self, batch_size: int, replay_ratio: float = 0.5):
        """
        Create batches combining current task data and replay buffer.
        
        Args:
            batch_size: Total batch size
            replay_ratio: Ratio of replay samples in each batch
            
        Yields:
            X_batch, y_batch combined from current + replay data
        """
        # Get replay data
        X_replay, y_replay = self.replay_buffer.get_all_samples()
        has_replay = X_replay is not None and len(y_replay) > 0
        
        if not has_replay:
            # No replay data, just use current task data
            yield from self._create_batches(batch_size)
            return
        
        # Calculate batch composition
        replay_batch_size = int(batch_size * replay_ratio)
        current_batch_size = batch_size - replay_batch_size
        
        # Shuffle indices
        current_indices = torch.randperm(self.num_samples)
        replay_indices = torch.randperm(len(y_replay))
        
        # Create batches
        n_batches = max(1, self.num_samples // current_batch_size)
        
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
                # Wrap around
                idx_r = torch.cat([
                    replay_indices[start_r:],
                    replay_indices[:end_r - len(y_replay)]
                ])
            
            X_rep = X_replay[idx_r].to(self.device, non_blocking=True)
            y_rep = y_replay[idx_r].to(self.device, non_blocking=True)
            
            # Combine
            X_batch = torch.cat([X_current, X_rep], dim=0)
            y_batch = torch.cat([y_current, y_rep], dim=0)
            
            # Shuffle combined batch
            perm = torch.randperm(len(y_batch))
            yield X_batch[perm], y_batch[perm]
    
    def train(
        self,
        trainer: BaseTrainer,
        epochs: int,
        batch_size: int,
        lr: float,
        global_params: Optional[OrderedDict] = None,
        use_replay: bool = True,
        replay_ratio: float = 0.5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train with replay buffer.
        
        For task 0: Standard training (no replay)
        For task > 0: Combined training with replay buffer
        
        Args:
            trainer: Training strategy (FedCBDRTrainer)
            epochs: Local epochs
            batch_size: Batch size
            lr: Learning rate
            global_params: Global model parameters
            use_replay: Whether to use replay buffer
            replay_ratio: Ratio of replay samples in batch
            **kwargs: Additional trainer arguments
            
        Returns:
            Dict with training results
        """
        self.model.train()
        
        # Get optimizer
        optimizer_cls = trainer.get_optimizer_class()
        optimizer = optimizer_cls(self.model.parameters(), lr=lr)
        scaler = GradScaler(enabled=self.use_amp)
        
        # Pre-train hook
        trainer.pre_train(self.model, global_params, lr=lr, **kwargs)
        
        total_loss = 0.0
        total_samples = 0
        
        # Determine if we should use replay
        has_replay = self.replay_buffer.total_samples > 0 and self.current_task > 0
        should_replay = use_replay and has_replay
        
        for ep in range(epochs):
            if should_replay:
                batch_gen = self._create_combined_batches(batch_size, replay_ratio)
            else:
                batch_gen = self._create_batches(batch_size)
            
            for X_batch, y_batch in batch_gen:
                optimizer.zero_grad()
                
                with self._amp_ctx():
                    out = self.model(X_batch)
                    loss = trainer.compute_loss(
                        self.model, out, y_batch, global_params, **kwargs
                    )
                
                if self.use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    trainer.pre_step(self.model, global_params, **kwargs)
                    scaler.step(optimizer)
                    scaler.update()
                    trainer.post_step(self.model, global_params, **kwargs)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    trainer.pre_step(self.model, global_params, **kwargs)
                    optimizer.step()
                    trainer.post_step(self.model, global_params, **kwargs)
                
                bs = len(y_batch)
                total_loss += loss.item() * bs
                total_samples += bs
        
        # Post-train hook
        trainer.post_train(self.model, global_params, **kwargs)
        
        return {
            "client_id": self.client_id,
            "num_samples": self.num_samples,
            "loss": total_loss / max(1, total_samples),
            "params": OrderedDict(
                (k, v.cpu().clone()) for k, v in self.model.state_dict().items()
            ),
            "replay_samples": self.replay_buffer.total_samples,
            "replay_classes": self.replay_buffer.num_classes
        }
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get replay buffer statistics."""
        return {
            "total_samples": self.replay_buffer.total_samples,
            "num_classes": self.replay_buffer.num_classes,
            "class_distribution": self.replay_buffer.get_class_distribution(),
            "buffer_size": self.buffer_size
        }
