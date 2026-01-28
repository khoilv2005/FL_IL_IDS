"""
FedCBDR Strategy - Class-wise Balancing Data Replay for Federated Class-Incremental Learning.

Reference:
    "Class-wise Balancing Data Replay for Federated Class-Incremental Learning"
    Zhuang Qi, Ying-Peng Tang, Lei Meng, Han Yu, Xiaoxiao Li, Xiangxu Meng
    arXiv:2507.07712, 2025

Paper Algorithm Summary:
========================
1. Global-perspective Data Replay (GDR):
   - Extract features using global model
   - Privacy-preserving encryption via ISVD (Inverse SVD)
   - Server computes leverage scores for importance-based sampling
   - Class-balanced replay buffer selection

2. Task-aware Temperature Scaling (TTS):
   - Separate temperature for old vs new task logits
   - Separate weights for old vs new task samples
   - Helps maintain balance between old and new knowledge

3. Memory Buffer Management:
   - Fixed-size replay buffer per client
   - Importance-based sampling using leverage scores
   - Herding selection for representative samples
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple
import copy
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import BaseTrainer, BaseAggregator


class FedCBDRTrainer(BaseTrainer):
    """
    FedCBDR local training with Task-aware Temperature Scaling (TTS).
    
    Implements the TTS loss function from paper Section 4.2:
    - Separate temperature scaling for old vs new task logits
    - Separate sample weights for old vs new task samples
    
    TTS Loss (Paper Eq. 7):
        L_TTS = (1/N_old) * Σ ω_old * CE(y, Softmax(Concat(z_old/τ_old, z_new/τ_new)))
              + (1/N_new) * Σ ω_new * CE(y, Softmax(Concat(z_old/τ_old, z_new/τ_new)))
    
    Args:
        tau_old: Temperature for old task logits (lower = sharper, preserves old)
        tau_new: Temperature for new task logits (higher = smoother)
        omega_old: Weight for old task samples (higher = more emphasis on old)
        omega_new: Weight for new task samples
        num_old_classes: Number of classes from previous tasks
        num_new_classes: Number of classes in current task
    """
    
    def __init__(
        self,
        tau_old: float = 0.9,
        tau_new: float = 1.1,
        omega_old: float = 1.1,
        omega_new: float = 0.9,
        **kwargs
    ):
        self.tau_old = tau_old
        self.tau_new = tau_new
        self.omega_old = omega_old
        self.omega_new = omega_new
        
        # Task tracking
        self.current_task: int = 0
        self.seen_classes: Set[int] = set()
        self.old_classes: List[int] = []
        self.new_classes: List[int] = []
        
        # For compatibility with training scripts
        self.mu_coefficient: float = 1.0
        self.best_acc_per_task: Dict[int, float] = {}
        self.current_acc_per_task: Dict[int, float] = {}
        self.last_af: float = 0.0
    
    def set_task(self, task_id: int, new_classes: List[int]):
        """
        Called at the beginning of each new task.
        
        Updates the class tracking for TTS loss computation.
        """
        # Store old classes before updating
        self.old_classes = list(self.seen_classes)
        self.new_classes = new_classes
        
        self.current_task = task_id
        self.seen_classes.update(new_classes)
        
        print(f"  FedCBDR Task {task_id}: old_classes={len(self.old_classes)}, new_classes={len(new_classes)}")
    
    def compute_loss(
        self,
        model: nn.Module,
        output: torch.Tensor,
        target: torch.Tensor,
        global_params: Optional[OrderedDict] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute TTS loss with temperature scaling.
        
        For task 0 (no old classes): standard CrossEntropy
        For task > 0: TTS loss with separate temperatures
        """
        # Task 0: Standard cross-entropy (no old classes)
        if self.current_task == 0 or len(self.old_classes) == 0:
            return F.cross_entropy(output, target)
        
        # Task > 0: Apply TTS loss
        return self._compute_tts_loss(output, target)
    
    def _compute_tts_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Task-aware Temperature Scaling loss.
        
        Paper Eq. 7: Apply different temperatures to old vs new class logits.
        """
        device = logits.device
        batch_size = logits.size(0)
        num_classes = logits.size(1)
        
        # Determine class indices
        old_class_indices = sorted(self.old_classes)
        new_class_indices = sorted(self.new_classes)
        
        # Ensure indices are valid
        max_old = max(old_class_indices) if old_class_indices else -1
        max_new = max(new_class_indices) if new_class_indices else -1
        
        if max_old >= num_classes or max_new >= num_classes:
            # Fallback to standard CE if class indices exceed logits dimension
            return F.cross_entropy(logits, target)
        
        # Create temperature-scaled logits
        scaled_logits = logits.clone()
        
        # Apply temperature scaling
        if old_class_indices:
            old_indices = torch.tensor(old_class_indices, device=device)
            scaled_logits[:, old_indices] = logits[:, old_indices] / self.tau_old
        
        if new_class_indices:
            new_indices = torch.tensor(new_class_indices, device=device)
            scaled_logits[:, new_indices] = logits[:, new_indices] / self.tau_new
        
        # Determine which samples are old vs new based on target
        old_class_set = set(old_class_indices)
        new_class_set = set(new_class_indices)
        
        target_np = target.cpu().numpy()
        old_mask = torch.tensor([t in old_class_set for t in target_np], device=device)
        new_mask = torch.tensor([t in new_class_set for t in target_np], device=device)
        
        # Compute weighted loss
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Loss for old class samples
        if old_mask.any():
            old_logits = scaled_logits[old_mask]
            old_targets = target[old_mask]
            old_loss = F.cross_entropy(old_logits, old_targets, reduction='mean')
            loss = loss + self.omega_old * old_loss
        
        # Loss for new class samples
        if new_mask.any():
            new_logits = scaled_logits[new_mask]
            new_targets = target[new_mask]
            new_loss = F.cross_entropy(new_logits, new_targets, reduction='mean')
            loss = loss + self.omega_new * new_loss
        
        # Handle edge case where no samples match
        if not old_mask.any() and not new_mask.any():
            loss = F.cross_entropy(scaled_logits, target)
        
        return loss
    
    def update_forgetting(self, task_accuracies: Dict[int, float]):
        """Update accuracy tracking for metrics (compatibility)."""
        self.current_acc_per_task = task_accuracies.copy()
        
        for task_id, acc in task_accuracies.items():
            if task_id not in self.best_acc_per_task:
                self.best_acc_per_task[task_id] = acc
            else:
                self.best_acc_per_task[task_id] = max(
                    self.best_acc_per_task[task_id], acc
                )
        
        # Compute Average Forgetting
        if len(self.best_acc_per_task) > 1:
            forgetting_sum = 0.0
            count = 0
            for task_id in self.best_acc_per_task:
                if task_id != self.current_task and task_id in self.current_acc_per_task:
                    forgetting = self.best_acc_per_task[task_id] - self.current_acc_per_task[task_id]
                    forgetting_sum += max(0, forgetting)
                    count += 1
            
            self.last_af = forgetting_sum / max(1, count)


class FedCBDRAggregator(BaseAggregator):
    """
    FedCBDR aggregation - standard FedAvg weighted average.
    
    FedCBDR uses standard FedAvg for model aggregation.
    The innovation is in the data replay (GDR) and loss function (TTS),
    not in the aggregation method.
    """
    
    def aggregate(
        self,
        results: List[Dict],
        global_params: Optional[OrderedDict] = None,
        **kwargs
    ) -> OrderedDict:
        """Standard FedAvg aggregation."""
        return self._weighted_average(results)


class ReplayBuffer:
    """
    Memory buffer for experience replay in FedCBDR.
    
    Implements class-balanced storage with importance-based sampling.
    Each client maintains a fixed-size buffer across tasks.
    
    Attributes:
        max_size: Maximum number of samples in buffer
        samples_per_class: Target samples per class for balance
    """
    
    def __init__(self, max_size: int = 500, device: str = "cpu"):
        self.max_size = max_size
        self.device = device
        
        # Storage: {class_id: {"X": tensor, "y": tensor, "importance": tensor}}
        self.class_buffers: Dict[int, Dict[str, torch.Tensor]] = {}
        
        # Track total samples
        self.total_samples = 0
    
    @property
    def num_classes(self) -> int:
        """Number of classes in buffer."""
        return len(self.class_buffers)
    
    @property
    def samples_per_class(self) -> int:
        """Target samples per class for balanced buffer."""
        if self.num_classes == 0:
            return self.max_size
        return max(1, self.max_size // self.num_classes)
    
    def add_samples(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        importance_scores: Optional[torch.Tensor] = None,
        class_ids: Optional[List[int]] = None
    ):
        """
        Add samples to buffer with importance-based selection.
        
        Args:
            X: Feature tensor [N, ...]
            y: Label tensor [N]
            importance_scores: Optional importance scores [N] for prioritization
            class_ids: Optional list of new class IDs to add
        """
        if importance_scores is None:
            importance_scores = torch.ones(len(y))
        
        # Group by class
        unique_classes = y.unique().tolist()
        
        for cls in unique_classes:
            mask = (y == cls)
            X_cls = X[mask].cpu()
            y_cls = y[mask].cpu()
            imp_cls = importance_scores[mask].cpu()
            
            if cls not in self.class_buffers:
                # New class: initialize buffer
                self.class_buffers[cls] = {
                    "X": X_cls,
                    "y": y_cls,
                    "importance": imp_cls
                }
            else:
                # Existing class: merge and keep most important
                existing = self.class_buffers[cls]
                combined_X = torch.cat([existing["X"], X_cls], dim=0)
                combined_y = torch.cat([existing["y"], y_cls], dim=0)
                combined_imp = torch.cat([existing["importance"], imp_cls], dim=0)
                
                # Keep top-k by importance
                target_size = self.samples_per_class
                if len(combined_imp) > target_size:
                    _, top_indices = combined_imp.topk(target_size)
                    self.class_buffers[cls] = {
                        "X": combined_X[top_indices],
                        "y": combined_y[top_indices],
                        "importance": combined_imp[top_indices]
                    }
                else:
                    self.class_buffers[cls] = {
                        "X": combined_X,
                        "y": combined_y,
                        "importance": combined_imp
                    }
        
        # Rebalance buffer to stay within max_size
        self._rebalance()
        self._update_count()
    
    def _rebalance(self):
        """Rebalance buffer to ensure class balance and max size."""
        if self.num_classes == 0:
            return
        
        target_per_class = self.samples_per_class
        
        for cls in self.class_buffers:
            buf = self.class_buffers[cls]
            current_size = len(buf["y"])
            
            if current_size > target_per_class:
                # Keep most important samples
                _, top_indices = buf["importance"].topk(target_per_class)
                self.class_buffers[cls] = {
                    "X": buf["X"][top_indices],
                    "y": buf["y"][top_indices],
                    "importance": buf["importance"][top_indices]
                }
    
    def _update_count(self):
        """Update total sample count."""
        self.total_samples = sum(
            len(buf["y"]) for buf in self.class_buffers.values()
        )
    
    def get_all_samples(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get all samples from buffer.
        
        Returns:
            X: Combined features tensor
            y: Combined labels tensor
        """
        if self.total_samples == 0:
            return None, None
        
        X_list = []
        y_list = []
        
        for cls in sorted(self.class_buffers.keys()):
            X_list.append(self.class_buffers[cls]["X"])
            y_list.append(self.class_buffers[cls]["y"])
        
        return torch.cat(X_list, dim=0), torch.cat(y_list, dim=0)
    
    def get_balanced_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a class-balanced batch from buffer.
        
        Args:
            batch_size: Total batch size
            
        Returns:
            X_batch, y_batch tensors with balanced class representation
        """
        if self.total_samples == 0:
            return None, None
        
        samples_per_class = max(1, batch_size // self.num_classes)
        
        X_list = []
        y_list = []
        
        for cls in self.class_buffers:
            buf = self.class_buffers[cls]
            n_available = len(buf["y"])
            n_sample = min(samples_per_class, n_available)
            
            # Random sampling
            indices = torch.randperm(n_available)[:n_sample]
            X_list.append(buf["X"][indices])
            y_list.append(buf["y"][indices])
        
        return torch.cat(X_list, dim=0), torch.cat(y_list, dim=0)
    
    def update_importance(self, class_id: int, new_importance: torch.Tensor):
        """Update importance scores for a class."""
        if class_id in self.class_buffers:
            n_samples = len(self.class_buffers[class_id]["y"])
            if len(new_importance) == n_samples:
                self.class_buffers[class_id]["importance"] = new_importance.cpu()
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get number of samples per class."""
        return {
            cls: len(buf["y"]) 
            for cls, buf in self.class_buffers.items()
        }
    
    def clear(self):
        """Clear all samples from buffer."""
        self.class_buffers.clear()
        self.total_samples = 0


class LeverageScoreCalculator:
    """
    Compute leverage scores for importance-based sampling.
    
    Paper Section 4.1: Global-perspective Data Replay (GDR)
    
    Leverage score for sample j:
        τ_j = ||e_j^T U||_2^2
    
    Where U is from SVD: X = U Σ V^T
    Higher leverage score = more representative sample in latent space.
    """
    
    def __init__(self, rank: int = 50):
        """
        Args:
            rank: Number of singular vectors to use (k in paper)
        """
        self.rank = rank
    
    def compute_scores(
        self,
        features: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Compute leverage scores for features.
        
        Args:
            features: Feature matrix [N, D]
            normalize: Whether to normalize scores to sum to 1
            
        Returns:
            Leverage scores [N]
        """
        N, D = features.shape
        device = features.device
        
        # Use appropriate rank
        k = min(self.rank, min(N, D))
        
        try:
            # SVD: X = U Σ V^T
            U, S, Vt = torch.linalg.svd(features.float(), full_matrices=False)
            
            # Take top-k left singular vectors
            U_k = U[:, :k]  # [N, k]
            
            # Leverage score: τ_j = ||row_j(U_k)||_2^2
            scores = (U_k ** 2).sum(dim=1)  # [N]
            
        except Exception as e:
            # Fallback to uniform scores if SVD fails
            print(f"  Warning: SVD failed ({e}), using uniform scores")
            scores = torch.ones(N, device=device)
        
        if normalize:
            scores = scores / (scores.sum() + 1e-8)
        
        return scores
    
    def compute_scores_encrypted(
        self,
        features: torch.Tensor,
        P: Optional[torch.Tensor] = None,
        Q: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute encrypted features for privacy-preserving GDR.
        
        Paper Eq. 4: X' = P · X · Q (ISVD encryption)
        
        Args:
            features: Original feature matrix [N, D]
            P: Random orthogonal matrix [N, N] (client-specific)
            Q: Random orthogonal matrix [D, D] (shared across clients)
            
        Returns:
            encrypted_features: X' = P X Q
            P: The client-specific matrix (for later decryption)
        """
        N, D = features.shape
        device = features.device
        
        # Generate random orthogonal matrices if not provided
        if P is None:
            P = self._random_orthogonal(N, device)
        if Q is None:
            Q = self._random_orthogonal(D, device)
        
        # Encrypt: X' = P X Q
        encrypted = P @ features.float() @ Q
        
        return encrypted, P
    
    def _random_orthogonal(self, n: int, device: str) -> torch.Tensor:
        """Generate random orthogonal matrix via QR decomposition."""
        random_matrix = torch.randn(n, n, device=device)
        Q, _ = torch.linalg.qr(random_matrix)
        return Q
