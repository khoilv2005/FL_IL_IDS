"""
CGoFed Strategy - Constrained Gradient Optimization for Federated Class Incremental Learning.

Reference:
    "CGoFed: Constrained Gradient Optimization Strategy for Federated Class 
    Incremental Learning", IEEE TKDE, 2025

Key mechanism:
    - Train ONLY on NEW classes (no replay)
    - Use SVD to build representation space of old tasks
    - Project gradient orthogonally to preserve old knowledge
    - Relax constraint with adaptive α coefficient
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import BaseTrainer, BaseAggregator


class CGoFedTrainer(BaseTrainer):
    """
    CGoFed local training with SVD-based gradient constraint.
    
    Key: Train only on NEW classes, but project gradient to preserve old knowledge.
    
    Args:
        lambda_decay: Decay rate for α relaxation coefficient
        theta_threshold: AF threshold to reset α
        energy_threshold: SVD energy threshold for selecting basis vectors
        num_samples_rep: Number of samples for building representation space
    """
    
    def __init__(
        self,
        mu: float = 0.01,
        lambda_decay: float = 0.1,
        theta_threshold: float = 0.1,
        energy_threshold: float = 0.95,
        num_samples_rep: int = 100,
    ):
        self.mu = mu
        self.lambda_decay = lambda_decay
        self.theta_threshold = theta_threshold
        self.energy_threshold = energy_threshold
        self.num_samples_rep = num_samples_rep
        
        # Task tracking
        self.current_task: int = 0
        self.seen_classes: Set[int] = set()
        
        # Representation space of old tasks (basis vectors from SVD)
        # Dict: {param_name: U matrix (orthogonal basis)}
        self.old_space: Dict[str, torch.Tensor] = {}
        
        # Importance weights for basis vectors (from sigmoid of singular values)
        self.importance_weights: Dict[str, torch.Tensor] = {}
        
        # α relaxation
        self.alpha: float = 1.0
        self.t_reset: int = 0  # Last task where AF exceeded threshold
        
        # Accuracies for computing AF
        self.best_acc_per_task: Dict[int, float] = {}
        self.current_acc_per_task: Dict[int, float] = {}
    
    def set_task(self, task_id: int, new_classes: List[int]):
        """Called at the beginning of each new task."""
        self.current_task = task_id
        self.seen_classes.update(new_classes)
        
        # Update α using exponential decay
        if task_id > 0:
            self.alpha = math.exp(-self.lambda_decay * (task_id - self.t_reset))
    
    def update_forgetting(self, task_accuracies: Dict[int, float]):
        """
        Update AF (Average Forgetting) and reset α if needed.
        
        Args:
            task_accuracies: {task_id: current_accuracy}
        """
        self.current_acc_per_task = task_accuracies.copy()
        
        # Update best accuracies
        for tid, acc in task_accuracies.items():
            if tid not in self.best_acc_per_task:
                self.best_acc_per_task[tid] = acc
            else:
                self.best_acc_per_task[tid] = max(self.best_acc_per_task[tid], acc)
        
        # Compute AF
        if len(self.best_acc_per_task) > 1:
            forgetting = []
            for tid in range(self.current_task):
                if tid in self.best_acc_per_task and tid in self.current_acc_per_task:
                    f = self.best_acc_per_task[tid] - self.current_acc_per_task[tid]
                    forgetting.append(max(0, f))
            
            if forgetting:
                avg_forgetting = sum(forgetting) / len(forgetting)
                
                # Reset α if AF exceeds threshold
                if avg_forgetting > self.theta_threshold:
                    self.t_reset = self.current_task
                    self.alpha = 1.0  # Reset to strong constraint
                    print(f"⚠️ AF={avg_forgetting:.4f} > θ={self.theta_threshold}, reset α")
    
    def build_representation_space(
        self,
        model: nn.Module,
        data_loader,  # Iterable of (X, y) batches
        device: str = "cuda"
    ):
        """
        Build representation space using SVD after training on current task.
        Call this AFTER training current task completes.
        
        This stores the gradient space (U matrix from SVD) for future projection.
        """
        model.eval()
        
        # Collect gradients from forward passes on samples
        all_grads = {name: [] for name, _ in model.named_parameters() if _.requires_grad}
        
        sample_count = 0
        for X, y in data_loader:
            if sample_count >= self.num_samples_rep:
                break
                
            X, y = X.to(device), y.to(device)
            
            # Forward pass
            model.zero_grad()
            output = model(X)
            loss = F.cross_entropy(output, y)
            loss.backward()
            
            # Collect gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    all_grads[name].append(param.grad.view(-1).cpu().clone())
            
            sample_count += len(X)
        
        # Build representation matrix and SVD for each parameter
        for name in all_grads:
            if len(all_grads[name]) == 0:
                continue
                
            # Stack gradients: [num_samples, param_dim]
            R = torch.stack(all_grads[name], dim=0)
            
            # SVD: R = U @ S @ V^T
            try:
                U, S, Vh = torch.linalg.svd(R, full_matrices=False)
                
                # Select top-k singular values based on energy threshold
                total_energy = (S ** 2).sum()
                cumsum = torch.cumsum(S ** 2, dim=0)
                k = (cumsum < self.energy_threshold * total_energy).sum() + 1
                k = min(k.item(), len(S))
                
                # Store basis vectors (columns of V, which is Vh.T)
                # We use Vh[:k] which gives us the k most important directions
                basis = Vh[:k].T  # [param_dim, k]
                
                # Importance weights using sigmoid
                weights = torch.sigmoid(S[:k])
                
                # Merge with old space if exists
                if name in self.old_space:
                    old_basis = self.old_space[name]
                    old_weights = self.importance_weights[name]
                    
                    # Simple merge: concatenate and select top-k
                    merged_basis = torch.cat([old_basis, basis], dim=1)
                    merged_weights = torch.cat([old_weights, weights])
                    
                    # Keep top-k based on weights
                    max_k = min(50, merged_basis.shape[1])  # Limit memory
                    if merged_weights.shape[0] > max_k:
                        _, topk_idx = torch.topk(merged_weights, max_k)
                        merged_basis = merged_basis[:, topk_idx]
                        merged_weights = merged_weights[topk_idx]
                    
                    self.old_space[name] = merged_basis
                    self.importance_weights[name] = merged_weights
                else:
                    self.old_space[name] = basis
                    self.importance_weights[name] = weights
                    
            except Exception as e:
                print(f"SVD failed for {name}: {e}")
                continue
        
        print(f"✓ Built representation space for task {self.current_task}")
    
    def compute_loss(
        self, 
        model: nn.Module,
        output: torch.Tensor, 
        target: torch.Tensor,
        global_params: Optional[OrderedDict] = None,
        **kwargs
    ) -> torch.Tensor:
        """Standard cross-entropy loss (gradient constraint is in post_step)."""
        return F.cross_entropy(output, target)
    
    def post_step(
        self,
        model: nn.Module,
        global_params: Optional[OrderedDict] = None,
        **kwargs
    ) -> None:
        """
        Apply gradient projection BEFORE optimizer.step().
        
        Formula: g' = g - α * P * g
        where P = U @ U^T is projection onto old space
        """
        if self.current_task == 0 or len(self.old_space) == 0:
            return
        
        device = next(model.parameters()).device
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is None or name not in self.old_space:
                    continue
                
                g = param.grad.view(-1)  # Flatten gradient
                
                # Get basis vectors for old space
                U = self.old_space[name].to(device)  # [param_dim, k]
                weights = self.importance_weights[name].to(device)  # [k]
                
                # Project gradient onto old space: P @ g = U @ (U^T @ g)
                # Weight by importance
                proj_coeff = U.T @ g  # [k]
                weighted_coeff = proj_coeff * weights
                projection = U @ weighted_coeff  # [param_dim]
                
                # Remove projection (with relaxation α)
                # g' = g - α * projection
                g_new = g - self.alpha * projection
                
                # Reshape back
                param.grad = g_new.view(param.shape)
    
    def get_optimizer_class(self) -> type:
        """CGoFed uses SGD."""
        return torch.optim.SGD


class CGoFedAggregator(BaseAggregator):
    """
    CGoFed aggregation with cross-task gradient regularization.
    
    Uses similarity between representation spaces for personalized aggregation.
    """
    
    def __init__(self, cross_task_weight: float = 0.5):
        self.cross_task_weight = cross_task_weight
        
        # Store representation matrices from clients: {client_id: {task_id: R}}
        self.client_representations: Dict[int, Dict[int, torch.Tensor]] = {}
        
        # Historical global models: {task_id: params}
        self.task_global_models: Dict[int, OrderedDict] = {}
        
        self.current_task: int = 0
    
    def set_task(self, task_id: int):
        self.current_task = task_id
    
    def update_client_representation(
        self, 
        client_id: int, 
        task_id: int, 
        representation: torch.Tensor
    ):
        """Store client's representation matrix for similarity computation."""
        if client_id not in self.client_representations:
            self.client_representations[client_id] = {}
        self.client_representations[client_id][task_id] = representation
    
    def _compute_similarity(self, R1: torch.Tensor, R2: torch.Tensor) -> float:
        """Compute similarity between two representation matrices (L2 norm)."""
        # Normalize and compute distance
        R1_norm = R1 / (torch.norm(R1) + 1e-8)
        R2_norm = R2 / (torch.norm(R2) + 1e-8)
        return -torch.norm(R1_norm - R2_norm).item()  # Negative distance = similarity
    
    def aggregate(
        self, 
        results: List[Dict], 
        global_params: Optional[OrderedDict] = None,
        **kwargs
    ) -> OrderedDict:
        """Aggregate with optional cross-task regularization."""
        # Standard weighted average
        agg_params = self._weighted_average(results)
        
        # Save for future cross-task reference
        self.task_global_models[self.current_task] = OrderedDict(
            (k, v.cpu().clone()) for k, v in agg_params.items()
        )
        
        # Cross-task regularization (if we have history)
        if self.current_task > 0 and len(self.task_global_models) > 1:
            agg_params = self._apply_cross_task_reg(agg_params)
        
        return agg_params
    
    def _apply_cross_task_reg(self, agg_params: OrderedDict) -> OrderedDict:
        """Blend with historical models for stability."""
        if len(self.task_global_models) <= 1:
            return agg_params
        
        # Average of all previous task models
        prev_tasks = [t for t in self.task_global_models if t < self.current_task]
        if not prev_tasks:
            return agg_params
        
        # Simple average of historical models
        avg_old = None
        for tid in prev_tasks:
            hist = self.task_global_models[tid]
            if avg_old is None:
                avg_old = OrderedDict((k, v.clone().float()) for k, v in hist.items())
            else:
                for k in avg_old:
                    if avg_old[k].dtype.is_floating_point:
                        avg_old[k] += hist[k].float()
        
        for k in avg_old:
            if avg_old[k].dtype.is_floating_point:
                avg_old[k] /= len(prev_tasks)
        
        # Blend
        w = self.cross_task_weight
        result = OrderedDict()
        for k in agg_params:
            if agg_params[k].dtype.is_floating_point:
                old_v = avg_old[k].to(agg_params[k].device)
                result[k] = (1 - w) * agg_params[k] + w * old_v
            else:
                result[k] = agg_params[k]
        
        return result
