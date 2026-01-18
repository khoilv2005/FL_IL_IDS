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
    - Cross-task regularization with similarity-based TOP-K selection
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
        
        # Representation space of old tasks (basis vectors from SVD) - stored on CPU
        self.old_space: Dict[str, torch.Tensor] = {}
        self.importance_weights: Dict[str, torch.Tensor] = {}
        
        # [FIX MULTI-GPU SPEED] Per-device GPU cache to avoid repeated CPU->GPU copies
        # Structure: { 'cuda:0': {param_name: (U, weights)}, 'cuda:1': ... }
        self.gpu_cache: Dict[str, Dict[str, tuple]] = {}
        
        # α relaxation
        self.alpha: float = 1.0
        self.t_reset: int = 0
        
        # Accuracies for computing AF
        self.best_acc_per_task: Dict[int, float] = {}
        self.current_acc_per_task: Dict[int, float] = {}
        self.last_af: float = 0.0
    
    def set_task(self, task_id: int, new_classes: List[int]):
        """Called at the beginning of each new task."""
        self.current_task = task_id
        self.seen_classes.update(new_classes)
        
        if task_id > 0:
            self.alpha = math.exp(-self.lambda_decay * (task_id - self.t_reset))
    
    def update_forgetting(self, task_accuracies: Dict[int, float]):
        """Update AF and reset α if needed."""
        self.current_acc_per_task = task_accuracies.copy()
        
        for tid, acc in task_accuracies.items():
            if tid not in self.best_acc_per_task:
                self.best_acc_per_task[tid] = acc
            else:
                self.best_acc_per_task[tid] = max(self.best_acc_per_task[tid], acc)
        
        # Calculate AF (Average Forgetting) - Eq.16 in paper
        self.last_af = 0.0
        if len(self.best_acc_per_task) > 1:
            forgetting = []
            for tid in range(self.current_task):
                if tid in self.best_acc_per_task and tid in self.current_acc_per_task:
                    f = self.best_acc_per_task[tid] - self.current_acc_per_task[tid]
                    forgetting.append(max(0, f))
            
            if forgetting:
                self.last_af = sum(forgetting) / len(forgetting)
                if self.last_af > self.theta_threshold:
                    self.t_reset = self.current_task
                    self.alpha = 1.0
                    print(f"⚠️ AF={self.last_af:.4f} > θ={self.theta_threshold}, reset α to 1.0")
    
    def get_current_af(self) -> float:
        """Get the last computed Average Forgetting value."""
        return getattr(self, 'last_af', 0.0)
    
    def build_representation_space(
        self,
        model: nn.Module,
        data_loader,
        device: str = "cuda"
    ):
        """Build representation space using SVD after training."""
        # Keep in train mode for GRU backward compatibility (cudnn requirement)
        model.train()
        
        all_grads = {name: [] for name, _ in model.named_parameters() if _.requires_grad}
        
        sample_count = 0
        for X, y in data_loader:
            if sample_count >= self.num_samples_rep:
                break
                
            X, y = X.to(device), y.to(device)
            
            model.zero_grad()
            output = model(X)
            loss = F.cross_entropy(output, y)
            loss.backward()
            
            # Handle None grads (from BatchNorm when batch_size=1) by replacing with zeros
            for name, param in model.named_parameters():
                if param.grad is not None:
                    all_grads[name].append(param.grad.view(-1).cpu().clone())
                else:
                    # BatchNorm skipped when batch_size=1, fill with zeros
                    all_grads[name].append(torch.zeros(param.numel()))
            
            sample_count += len(X)
        
        for name in all_grads:
            if len(all_grads[name]) == 0:
                continue
                
            R = torch.stack(all_grads[name], dim=0)
            
            try:
                U, S, Vh = torch.linalg.svd(R, full_matrices=False)
                
                total_energy = (S ** 2).sum()
                cumsum = torch.cumsum(S ** 2, dim=0)
                k = (cumsum < self.energy_threshold * total_energy).sum() + 1
                k = min(k.item(), len(S))
                
                basis = Vh[:k].T
                weights = torch.sigmoid(S[:k])
                
                if name in self.old_space:
                    old_basis = self.old_space[name]
                    old_weights = self.importance_weights[name]
                    
                    merged_basis = torch.cat([old_basis, basis], dim=1)
                    merged_weights = torch.cat([old_weights, weights])
                    
                    max_k = min(50, merged_basis.shape[1])
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
        """Standard cross-entropy loss."""
        return F.cross_entropy(output, target)
    
    def post_step(
        self,
        model: nn.Module,
        global_params: Optional[OrderedDict] = None,
        **kwargs
    ) -> None:
        """Ultra-optimized gradient projection: skip bias, cache per-GPU, minimize GIL contention"""
        if self.current_task == 0 or len(self.old_space) == 0:
            return
        
        try:
            device_obj = next(model.parameters()).device
            device_key = str(device_obj)
        except StopIteration:
            return
        
        if device_key not in self.gpu_cache:
            self.gpu_cache[device_key] = {}
        
        # Get param dict (create fresh to avoid stale references in multi-threaded)
        param_dict = dict(model.named_parameters())
        
        with torch.no_grad():
            # Only iterate over old_space keys (faster than model.named_parameters())
            for name in self.old_space.keys():
                if name not in param_dict:
                    continue
                    
                param = param_dict[name]
                
                # Double check grad exists (can change in multi-threaded)
                if param.grad is None:
                    continue
                
                # Skip bias and small params (reduces GIL contention)
                if 'bias' in name or param.numel() < 100:
                    continue
                
                # Per-device GPU cache
                if name not in self.gpu_cache[device_key]:
                    self.gpu_cache[device_key][name] = (
                        self.old_space[name].to(device_obj, non_blocking=True),
                        self.importance_weights[name].to(device_obj, non_blocking=True)
                    )
                
                U, weights = self.gpu_cache[device_key][name]
                
                # Compute projection delta
                g = param.grad.view(-1)
                proj_coeff = U.T @ g
                delta = self.alpha * (U @ (proj_coeff * weights))
                
                # [SAFE] Use sub_ (in-place subtract) instead of copy_
                # This is safer in multi-threaded environment
                param.grad.sub_(delta.view(param.shape))
    
    def get_optimizer_class(self) -> type:
        return torch.optim.SGD


class CGoFedAggregator(BaseAggregator):
    """
    CGoFed aggregation with similarity-based cross-task regularization.
    
    Uses similarity between client representations to select TOP-K 
    relevant historical models for weighted aggregation.
    
    Args:
        cross_task_weight: Weight λ for blending with historical models
        top_k: Number of most similar historical models to select (paper: K=2)
    """
    
    def __init__(self, cross_task_weight: float = 0.3, top_k: int = 2):
        self.cross_task_weight = cross_task_weight
        self.top_k = top_k
        
        # Store mean gradient vectors from clients: {client_id: {task_id: R_vector}}
        self.client_representations: Dict[int, Dict[int, torch.Tensor]] = {}
        
        # Historical global models: {task_id: params}
        self.task_global_models: Dict[int, OrderedDict] = {}
        
        # Mean representation per task (aggregated from clients)
        self.task_representations: Dict[int, torch.Tensor] = {}
        
        self.current_task: int = 0
    
    def set_task(self, task_id: int):
        self.current_task = task_id
    
    def _store_client_representations(self, results: List[Dict]):
        """Extract and store representations from client results."""
        task_reps = []
        
        for r in results:
            if "representation" in r and r["representation"] is not None:
                client_id = r["client_id"]
                rep = r["representation"]
                
                # Store per-client
                if client_id not in self.client_representations:
                    self.client_representations[client_id] = {}
                self.client_representations[client_id][self.current_task] = rep
                
                task_reps.append(rep)
        
        # Compute mean representation for this task
        if task_reps:
            self.task_representations[self.current_task] = torch.stack(task_reps).mean(dim=0)
    
    def _compute_similarity(self, R1: torch.Tensor, R2: torch.Tensor) -> float:
        """
        Compute similarity between two representation vectors.
        Returns negative L2 distance (higher = more similar).
        """
        R1_norm = R1 / (torch.norm(R1) + 1e-8)
        R2_norm = R2 / (torch.norm(R2) + 1e-8)
        return -torch.norm(R1_norm - R2_norm).item()
    
    def _select_top_k_similar(self) -> List[Dict]:
        """
        Select TOP-K most similar historical task models.
        
        Returns:
            List of {task_id, similarity, params}
        """
        if self.current_task == 0 or len(self.task_representations) <= 1:
            return []
        
        current_rep = self.task_representations.get(self.current_task)
        if current_rep is None:
            return []
        
        # Compute similarity with all previous tasks
        similarities = []
        for tid in range(self.current_task):
            if tid in self.task_representations and tid in self.task_global_models:
                hist_rep = self.task_representations[tid]
                sim = self._compute_similarity(current_rep, hist_rep)
                similarities.append({
                    "task_id": tid,
                    "similarity": sim,
                    "params": self.task_global_models[tid]
                })
        
        # Sort by similarity (descending) and select TOP-K
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:self.top_k]
    
    def _weighted_aggregate_with_history(
        self, 
        current_params: OrderedDict,
        selected_models: List[Dict]
    ) -> OrderedDict:
        """
        Weighted aggregation with historical models based on similarity.
        
        Formula: Final = (1 - λ) * Current + λ * Σ(w_i * Hist_i)
        where w_i = softmax(similarity_i)
        """
        if not selected_models:
            return current_params
        
        # Compute softmax weights from similarities
        sim_scores = torch.tensor([s["similarity"] for s in selected_models])
        weights = F.softmax(sim_scores, dim=0)
        
        # Aggregate historical models
        hist_agg = None
        for i, model_info in enumerate(selected_models):
            hist_params = model_info["params"]
            w = weights[i].item()
            
            if hist_agg is None:
                hist_agg = OrderedDict(
                    (k, w * v.float()) for k, v in hist_params.items()
                    if v.dtype.is_floating_point
                )
                # Copy non-float parameters
                for k, v in hist_params.items():
                    if not v.dtype.is_floating_point:
                        hist_agg[k] = v.clone()
            else:
                for k in hist_agg:
                    if hist_agg[k].dtype.is_floating_point:
                        hist_agg[k] += w * hist_params[k].float()
        
        # Blend with current: (1-λ) * current + λ * history
        λ = self.cross_task_weight
        result = OrderedDict()
        
        for k in current_params:
            if current_params[k].dtype.is_floating_point:
                hist_v = hist_agg[k].to(current_params[k].device)
                result[k] = (1 - λ) * current_params[k] + λ * hist_v
            else:
                result[k] = current_params[k]
        
        return result
    
    def aggregate(
        self, 
        results: List[Dict], 
        global_params: Optional[OrderedDict] = None,
        **kwargs
    ) -> OrderedDict:
        """
        Aggregate with similarity-based cross-task regularization.
        
        1. Standard weighted average of client updates
        2. Store client representations
        3. Select TOP-K similar historical models
        4. Weighted blend with history
        """
        # 1. Standard weighted average
        agg_params = self._weighted_average(results)
        
        # 2. Store representations from this round
        self._store_client_representations(results)
        
        # 3. Save current model for future reference
        self.task_global_models[self.current_task] = OrderedDict(
            (k, v.cpu().clone()) for k, v in agg_params.items()
        )
        
        # 4. Cross-task regularization (if we have history)
        if self.current_task > 0:
            selected = self._select_top_k_similar()
            if selected:
                print(f"📊 Selected TOP-{len(selected)} similar tasks: {[s['task_id'] for s in selected]}")
                agg_params = self._weighted_aggregate_with_history(agg_params, selected)
        
        return agg_params
