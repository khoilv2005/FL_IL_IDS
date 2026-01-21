"""
CGoFed Strategy - Constrained Gradient Optimization for Federated Class Incremental Learning.

Reference:
    "CGoFed: Constrained Gradient Optimization Strategy for Federated Class 
    Incremental Learning", IEEE TKDE, Vol. 37, No. 5, May 2025
    Authors: Jiyuan Feng, Xu Yang, Liwen Liang, Weihong Han, Binxing Fang, Qing Liao

Paper Algorithm Summary:
========================
1. Representation Space (Eq. 2-4):
   - R^t = gradient vectors from samples (NOT activations)
   - SVD: R = U @ S @ V^T
   - Select k basis vectors based on energy threshold
   - Use V^T[:k, :] as basis (right singular vectors for gradient space)

2. Importance Weights (Eq. 5):
   - w_i = sigmoid(s_i) where s_i are singular values

3. Relaxation Coefficient α (Eq. 6-7):
   - α = exp(-λ * (t - t_reset))
   - Reset when AF > θ (Average Forgetting threshold)

4. Gradient Projection (Eq. 8):
   - g' = g - α * Σ(w_i * (v_i^T @ g) * v_i)
   - Project onto orthogonal complement of old task space

5. Cross-Task Regularization (Eq. 9-11):
   - Compute similarity between current and historical tasks
   - Select TOP-K most similar historical models
   - Weighted aggregation with history
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple
import copy
import math
import os
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import BaseTrainer, BaseAggregator


class CGoFedTrainer(BaseTrainer):
    """
    CGoFed local training with SVD-based gradient constraint.
    
    Implements the Relax-Constrained Gradient Update from paper Section 5.1.
    
    Key mechanism:
    - Build representation space using GRADIENT vectors (paper eq. 2)
    - SVD decomposition to find principal gradient directions
    - Project new gradients orthogonally to old task space
    - Adaptive α relaxation to balance stability-plasticity
    
    Args:
        mu: Proximal regularization weight (optional, default 0.01)
        lambda_decay: Decay rate for α relaxation (paper: λ)
        theta_threshold: AF threshold to reset α (paper: θ)
        energy_threshold: SVD energy threshold for rank selection
        num_samples_rep: Number of samples for building representation
        temp_dir: Directory for storing SVD basis matrices
    """
    
    def __init__(
        self,
        mu: float = 0.01,
        lambda_decay: float = 0.1,
        theta_threshold: float = 0.1,
        energy_threshold: float = 0.95,
        num_samples_rep: int = 100,
        temp_dir: str = "./temp_svd_storage",
    ):
        self.mu = mu
        self.lambda_decay = lambda_decay
        self.theta_threshold = theta_threshold
        self.energy_threshold = energy_threshold
        self.num_samples_rep = num_samples_rep
        
        # Temp directory for SVD matrices (lazy loading)
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Task tracking
        self.current_task: int = 0
        self.seen_classes: Set[int] = set()
        
        # Representation space: stores FILE PATHS for lazy loading
        # Key: task_key -> path to basis matrix
        self.old_space: Dict[str, str] = {}
        self.importance_weights: Dict[str, str] = {}
        
        # Gradient dimension (set when first building representation)
        self.gradient_dim: Optional[int] = None
        
        # α relaxation (paper eq. 6-7)
        self.alpha: float = 1.0
        self.t_reset: int = 0
        
        # Accuracies for computing AF (Average Forgetting)
        self.best_acc_per_task: Dict[int, float] = {}
        self.current_acc_per_task: Dict[int, float] = {}
        self.last_af: float = 0.0
    
    def set_task(self, task_id: int, new_classes: List[int]):
        """
        Called at the beginning of each new task.
        
        Updates α according to paper eq. 6:
        α^t = exp(-λ * (t - t_reset))
        """
        self.current_task = task_id
        self.seen_classes.update(new_classes)
        
        # Update α with exponential decay (paper eq. 6)
        if task_id > 0:
            self.alpha = math.exp(-self.lambda_decay * (task_id - self.t_reset))
            print(f"  α = exp(-{self.lambda_decay} * ({task_id} - {self.t_reset})) = {self.alpha:.4f}")
    
    def update_forgetting(self, task_accuracies: Dict[int, float]):
        """
        Update Average Forgetting (AF) and reset α if needed.
        
        Paper eq. 16:
        AF = (1/t) * Σ max(0, best_acc[j] - current_acc[j]) for j < t
        
        Paper eq. 7:
        If AF > θ, reset t_reset = t and α = 1.0
        """
        self.current_acc_per_task = task_accuracies.copy()
        
        # Update best accuracies
        for tid, acc in task_accuracies.items():
            if tid not in self.best_acc_per_task:
                self.best_acc_per_task[tid] = acc
            else:
                self.best_acc_per_task[tid] = max(self.best_acc_per_task[tid], acc)
        
        # Calculate AF (paper eq. 16)
        self.last_af = 0.0
        if len(self.best_acc_per_task) > 1:
            forgetting = []
            for tid in range(self.current_task):
                if tid in self.best_acc_per_task and tid in self.current_acc_per_task:
                    f = self.best_acc_per_task[tid] - self.current_acc_per_task[tid]
                    forgetting.append(max(0, f))
            
            if forgetting:
                self.last_af = sum(forgetting) / len(forgetting)
                
                # Check if need to reset α (paper eq. 7)
                if self.last_af > self.theta_threshold:
                    self.t_reset = self.current_task
                    self.alpha = 1.0
                    print(f"⚠️ AF={self.last_af:.4f} > θ={self.theta_threshold}, reset α to 1.0")
    
    def get_current_af(self) -> float:
        """Get the last computed Average Forgetting value."""
        return self.last_af
    
    def _flatten_gradients(self, model: nn.Module) -> torch.Tensor:
        """
        Flatten all model gradients into a single vector.
        
        Returns:
            Tensor of shape [d] where d = total number of parameters
        """
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1))
            else:
                # For params without grad (e.g., frozen), use zeros
                grads.append(torch.zeros(p.numel(), device=p.device))
        return torch.cat(grads)
    
    def _unflatten_gradients(self, flat_grad: torch.Tensor, model: nn.Module):
        """
        Unflatten gradient vector and assign to model parameters.
        
        Args:
            flat_grad: Flattened gradient tensor [d]
            model: Model to assign gradients to
        """
        offset = 0
        for p in model.parameters():
            numel = p.numel()
            if p.grad is not None:
                p.grad.copy_(flat_grad[offset:offset + numel].view_as(p.grad))
            offset += numel
    
    def build_representation_space(
        self,
        model: nn.Module,
        data_loader,
        device: str = "cuda"
    ):
        """
        Build representation space using GRADIENT vectors (paper eq. 2-4).
        
        Paper CGoFed eq. 2:
        R^t = [g_1, g_2, ..., g_n]^T where g_i is gradient from sample i
        
        Paper eq. 3-4:
        SVD: R = U @ S @ V^T
        Select k vectors based on energy threshold:
        k = argmin_k' { Σ_{i=1}^{k'} s_i^2 >= ε * Σ s_i^2 }
        
        Paper uses V^T[:k, :] as basis vectors in gradient space.
        """
        model.train()  # Need gradients
        criterion = nn.CrossEntropyLoss()
        
        gradient_vectors = []
        sample_count = 0
        
        print(f"  → Building gradient-based representation space...")
        
        for X, y in data_loader:
            if sample_count >= self.num_samples_rep:
                break
            
            X = X.to(device)
            y = y.to(device)
            
            # Process each sample individually for per-sample gradients
            for i in range(len(X)):
                if sample_count >= self.num_samples_rep:
                    break
                
                model.zero_grad()
                
                # Forward pass for single sample
                x_i = X[i:i+1]
                y_i = y[i:i+1]
                
                output = model(x_i)
                loss = criterion(output, y_i)
                loss.backward()
                
                # Flatten gradients into vector g_i (paper eq. 2)
                g_i = self._flatten_gradients(model).cpu()
                gradient_vectors.append(g_i)
                sample_count += 1
        
        if not gradient_vectors:
            print("⚠️ No gradient vectors collected")
            return
        
        # Stack into matrix R: [n_samples, d] (paper eq. 2)
        R = torch.stack(gradient_vectors, dim=0)
        self.gradient_dim = R.shape[1]
        print(f"  → Gradient matrix R: {R.shape} (n_samples={R.shape[0]}, d={R.shape[1]})")
        
        try:
            # SVD: R = U @ S @ V^T (paper eq. 3)
            # U: [n, min(n,d)], S: [min(n,d)], Vh: [min(n,d), d]
            U, S, Vh = torch.linalg.svd(R, full_matrices=False)
            
            # Energy-based rank selection (paper eq. 4)
            # Find k such that cumulative energy >= threshold
            energy = S ** 2
            cum_energy = torch.cumsum(energy, dim=0)
            total_energy = cum_energy[-1]
            
            # k = smallest k' where cum_energy[k'] >= ε * total_energy
            ratio = cum_energy / total_energy
            k = (ratio < self.energy_threshold).sum().item() + 1
            k = min(k, len(S), 100)  # Cap at 100 for memory
            
            # Use RIGHT singular vectors V^T[:k, :] as basis (paper-compliant)
            # Vh has shape [min(n,d), d], so Vh[:k, :] gives [k, d]
            # Each row is a basis vector in gradient space
            basis = Vh[:k, :]  # [k, d] - k basis vectors of dimension d
            
            # Importance weights using sigmoid (paper eq. 5)
            # w_i = sigmoid(s_i)
            weights = torch.sigmoid(S[:k])
            
            explained_var = (cum_energy[k-1] / total_energy * 100).item()
            print(f"  → SVD: k={k} vectors, explained variance={explained_var:.1f}%")
            print(f"  → Basis shape: {basis.shape}, Weights shape: {weights.shape}")
            
            # Save per-task (paper accumulates all old task bases)
            task_key = f"task_{self.current_task}"
            basis_path = os.path.join(self.temp_dir, f"{task_key}_basis.pt")
            weights_path = os.path.join(self.temp_dir, f"{task_key}_weights.pt")
            
            torch.save(basis.clone(), basis_path)
            torch.save(weights.clone(), weights_path)
            
            self.old_space[task_key] = basis_path
            self.importance_weights[task_key] = weights_path
            
            print(f"  → Saved basis to {basis_path}")
            
        except Exception as e:
            print(f"⚠️ SVD failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Cleanup
        del gradient_vectors, R
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"✓ Built representation space for task {self.current_task}")
    
    def compute_loss(
        self, 
        model: nn.Module,
        output: torch.Tensor, 
        target: torch.Tensor,
        global_params: Optional[OrderedDict] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute loss with optional proximal regularization.
        
        Loss = CE(output, target) + (μ/2) * ||θ - θ_global||^2
        """
        ce_loss = F.cross_entropy(output, target)
        
        # Optional proximal term (like FedProx)
        if self.mu > 0 and global_params is not None:
            prox_term = 0.0
            for name, param in model.named_parameters():
                if name in global_params:
                    global_param = global_params[name].to(param.device)
                    prox_term += torch.sum((param - global_param) ** 2)
            return ce_loss + (self.mu / 2) * prox_term
        
        return ce_loss
    
    def post_step(
        self,
        model: nn.Module,
        global_params: Optional[OrderedDict] = None,
        **kwargs
    ) -> None:
        """
        Gradient projection after optimizer step (paper eq. 8).
        
        Paper CGoFed eq. 8:
        g' = g - α * Σ_j Σ_i (w_{j,i} * (v_{j,i}^T @ g) * v_{j,i})
        
        Where:
        - g: current gradient vector
        - α: relaxation coefficient
        - v_{j,i}: i-th basis vector from task j
        - w_{j,i}: importance weight for v_{j,i}
        
        This projects the gradient onto the orthogonal complement of
        all old task representation spaces.
        """
        # Skip for first task (no old space to project against)
        if self.current_task == 0 or not self.old_space:
            return
        
        with torch.no_grad():
            # Get device from model
            device = next(model.parameters()).device
            
            # Flatten current gradients: g ∈ R^d
            g = self._flatten_gradients(model)
            
            # Compute total projection onto all old task spaces
            total_projection = torch.zeros_like(g)
            
            for task_key, basis_path in self.old_space.items():
                try:
                    # Load basis V^T[:k, :] and weights from disk
                    basis = torch.load(basis_path, map_location=device)  # [k, d]
                    weights_path = self.importance_weights[task_key]
                    weights = torch.load(weights_path, map_location=device)  # [k]
                    
                    # Check dimension match
                    if basis.shape[1] != g.shape[0]:
                        print(f"⚠️ Dimension mismatch for {task_key}: "
                              f"basis={basis.shape[1]}, grad={g.shape[0]}")
                        continue
                    
                    # Project gradient onto each basis vector (paper eq. 8)
                    # For each basis vector v_i: projection = w_i * (v_i^T @ g) * v_i
                    for i in range(basis.shape[0]):
                        v_i = basis[i, :]  # [d] - i-th basis vector
                        w_i = weights[i]   # scalar weight
                        
                        # Compute projection coefficient: v_i^T @ g
                        coef = torch.dot(v_i, g)
                        
                        # Weighted projection: w_i * coef * v_i
                        proj_i = w_i * coef * v_i
                        total_projection += proj_i
                    
                    del basis, weights
                    
                except Exception as e:
                    print(f"⚠️ Projection failed for {task_key}: {e}")
                    continue
            
            # Apply constrained update: g' = g - α * projection (paper eq. 8)
            g_new = g - self.alpha * total_projection
            
            # Unflatten and update model gradients
            self._unflatten_gradients(g_new, model)
    
    def get_optimizer_class(self) -> type:
        """Use SGD as specified in paper experiments."""
        return torch.optim.SGD


class CGoFedAggregator(BaseAggregator):
    """
    CGoFed aggregation with Cross-Task Gradient Regularization.
    
    Implements paper Section 5.2: Cross-task Gradient Regularization.
    
    Key mechanism:
    - Compute representation similarity between tasks (paper eq. 9)
    - Select TOP-K most similar historical models (paper eq. 10)
    - Personalized aggregation with history (paper eq. 11)
    
    Args:
        cross_task_weight: Weight λ for blending with history (paper: λ)
        top_k: Number of similar models to select (paper: K, default 2)
    """
    
    def __init__(self, cross_task_weight: float = 0.3, top_k: int = 2):
        self.cross_task_weight = cross_task_weight
        self.top_k = top_k
        
        # Store gradient representations from clients
        # {client_id: {task_id: gradient_vector}}
        self.client_representations: Dict[int, Dict[int, torch.Tensor]] = {}
        
        # Historical global models: {task_id: OrderedDict params}
        self.task_global_models: Dict[int, OrderedDict] = {}
        
        # Mean representation per task (aggregated from clients)
        self.task_representations: Dict[int, torch.Tensor] = {}
        
        self.current_task: int = 0
    
    def set_task(self, task_id: int):
        """Set current task ID."""
        self.current_task = task_id
    
    def _store_client_representations(self, results: List[Dict]):
        """
        Extract and store gradient representations from client results.
        
        These are used for computing cross-task similarity (paper eq. 9).
        """
        task_reps = []
        
        for r in results:
            if "representation" in r and r["representation"] is not None:
                client_id = r["client_id"]
                rep = r["representation"]
                
                # Store per-client per-task
                if client_id not in self.client_representations:
                    self.client_representations[client_id] = {}
                self.client_representations[client_id][self.current_task] = rep
                
                task_reps.append(rep)
        
        # Compute mean representation for this task
        if task_reps:
            stacked = torch.stack(task_reps, dim=0)
            self.task_representations[self.current_task] = stacked.mean(dim=0)
            print(f"  → Stored {len(task_reps)} representations for task {self.current_task}")
    
    def _compute_similarity(self, R1: torch.Tensor, R2: torch.Tensor) -> float:
        """
        Compute similarity between two representation vectors.
        
        Paper eq. 9: Uses negative L2 distance of normalized vectors.
        Higher value = more similar.
        
        sim(R1, R2) = -||R1/||R1|| - R2/||R2||||_2
        """
        # Normalize
        R1_norm = R1 / (torch.norm(R1) + 1e-8)
        R2_norm = R2 / (torch.norm(R2) + 1e-8)
        
        # Negative L2 distance (so higher = more similar)
        dist = torch.norm(R1_norm - R2_norm).item()
        return -dist
    
    def _select_top_k_similar(self) -> List[Dict]:
        """
        Select TOP-K most similar historical task models.
        
        Paper eq. 10: Select K historical models with highest similarity
        to current task's representation.
        
        Returns:
            List of {task_id, similarity, params}
        """
        if self.current_task == 0:
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
        selected = similarities[:self.top_k]
        
        return selected
    
    def _weighted_aggregate_with_history(
        self, 
        current_params: OrderedDict,
        selected_models: List[Dict]
    ) -> OrderedDict:
        """
        Personalized aggregation with historical models (paper eq. 11).
        
        Paper eq. 11:
        θ_final = (1 - λ) * θ_current + λ * Σ_i (w_i * θ_hist_i)
        
        Where w_i = softmax(similarity_i) for selected models.
        """
        if not selected_models:
            return current_params
        
        # Compute softmax weights from similarities
        sim_scores = torch.tensor([s["similarity"] for s in selected_models])
        weights = F.softmax(sim_scores, dim=0)
        
        print(f"  → Cross-task weights: {[f'{w:.3f}' for w in weights.tolist()]}")
        
        # Weighted aggregation of historical models
        hist_agg = None
        for i, model_info in enumerate(selected_models):
            hist_params = model_info["params"]
            w = weights[i].item()
            
            if hist_agg is None:
                hist_agg = OrderedDict()
                for k, v in hist_params.items():
                    if v.dtype.is_floating_point:
                        hist_agg[k] = w * v.float()
                    else:
                        hist_agg[k] = v.clone()
            else:
                for k in hist_agg:
                    if hist_agg[k].dtype.is_floating_point:
                        hist_agg[k] += w * hist_params[k].float()
        
        # Blend: (1-λ) * current + λ * history
        λ = self.cross_task_weight
        result = OrderedDict()
        
        for k in current_params:
            if current_params[k].dtype.is_floating_point:
                hist_v = hist_agg[k].to(current_params[k].device)
                result[k] = (1 - λ) * current_params[k].float() + λ * hist_v
            else:
                result[k] = current_params[k].clone()
        
        return result
    
    def aggregate(
        self, 
        results: List[Dict], 
        global_params: Optional[OrderedDict] = None,
        **kwargs
    ) -> OrderedDict:
        """
        Aggregate with cross-task regularization (paper Section 5.2).
        
        Steps:
        1. Standard weighted average of client updates (FedAvg)
        2. Store client representations for similarity
        3. Save current model for future reference
        4. Select TOP-K similar historical models
        5. Weighted blend with history
        """
        # 1. Standard weighted average (FedAvg)
        agg_params = self._weighted_average(results)
        
        # 2. Store representations from clients
        self._store_client_representations(results)
        
        # 3. Save current aggregated model for future cross-task reference
        self.task_global_models[self.current_task] = OrderedDict(
            (k, v.cpu().clone()) for k, v in agg_params.items()
        )
        
        # 4 & 5. Cross-task regularization (if we have history)
        if self.current_task > 0:
            selected = self._select_top_k_similar()
            if selected:
                task_ids = [s['task_id'] for s in selected]
                print(f"📊 Cross-task: Selected TOP-{len(selected)} similar tasks: {task_ids}")
                agg_params = self._weighted_aggregate_with_history(agg_params, selected)
        
        return agg_params
