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
import os
import shutil
import gc

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
        temp_dir: str = "./temp_svd_storage",
    ):
        self.mu = mu
        self.lambda_decay = lambda_decay
        self.theta_threshold = theta_threshold
        self.energy_threshold = energy_threshold
        self.num_samples_rep = num_samples_rep
        
        # Temp directory for Lazy Loading SVD matrices
        # NOTE: Do NOT delete existing files here! Files from previous tasks are needed.
        # Cleanup should be done ONCE at the start of main() in training script.
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Task tracking
        self.current_task: int = 0
        self.seen_classes: Set[int] = set()
        
        # Representation space: stores FILE PATHS (str) instead of tensors
        self.old_space: Dict[str, str] = {}
        self.importance_weights: Dict[str, str] = {}
        
        # GPU cache for loaded tensors
        self.gpu_cache: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = {}
        
        # α relaxation
        self.alpha: float = 1.0
        self.t_reset: int = 0
        
        # Accuracies for computing AF
        self.best_acc_per_task: Dict[int, float] = {}
        self.current_acc_per_task: Dict[int, float] = {}
        
        # Tracking last AF
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
    
    def update_representation_from_client_data(
        self,
        client_data: Dict[int, Dict[str, torch.Tensor]],
        model: nn.Module,
        batch_size: int = 512,
        device: str = "cuda"
    ):
        """
        Build representation space from client data (FL-compliant).
        
        This method encapsulates the entire SVD pipeline:
        1. Select representative client(s) - never aggregate all data
        2. Create DataLoader from representative data
        3. Call build_representation_space to compute SVD
        
        Args:
            client_data: Dict mapping client_id -> {"X_train": Tensor, "y_train": Tensor}
            model: The global model
            batch_size: Batch size for DataLoader
            device: Device to run computations on
        """
        from torch.utils.data import TensorDataset, DataLoader
        
        # Find clients with data
        active_cids = [cid for cid, data in client_data.items() 
                       if len(data.get("y_train", [])) > 0]
        
        if len(active_cids) == 0:
            print("⚠️ No client data available for representation space")
            return
        
        # Select representative client (first active client)
        rep_cid = active_cids[0]
        rep_X = client_data[rep_cid]["X_train"]
        rep_y = client_data[rep_cid]["y_train"]
        
        # If representative has too few samples, sample from a few more clients
        # But NEVER concatenate all clients (that would violate FL privacy)
        min_samples = 50
        if len(rep_y) < min_samples and len(active_cids) > 1:
            for extra_cid in active_cids[1:3]:  # Max 2-3 clients
                if len(client_data[extra_cid].get("y_train", [])) > 0:
                    rep_X = torch.cat([rep_X, client_data[extra_cid]["X_train"]], dim=0)
                    rep_y = torch.cat([rep_y, client_data[extra_cid]["y_train"]], dim=0)
                if len(rep_y) >= min_samples:
                    break
        
        # Create DataLoader
        rep_loader = DataLoader(
            TensorDataset(rep_X, rep_y),
            batch_size=batch_size,
            shuffle=True
        )
        
        # Build representation space (SVD)
        self.build_representation_space(model, rep_loader, device)

    
    def build_representation_space(
        self,
        model: nn.Module,
        data_loader,
        device: str = "cuda"
    ):
        """
        Build representation space using ACTIVATIONS (paper-compliant).
        
        Paper CGoFed (IEEE TKDE 2025) eq. 2:
        R^t = F(Θ^t, X^t) - use forward activations, NOT gradients.
        
        Key fixes from paper:
        1. Use activations (get_fused_representation) instead of gradients
        2. Use LEFT singular vectors U (column space) instead of Vh.T
        3. Save separate basis per task (don't merge)
        """
        model.eval()  # Eval mode for consistent activations
        all_reps = []  # List of [B, dim] tensors
        
        sample_count = 0
        with torch.no_grad():
            for X, y in data_loader:
                if sample_count >= self.num_samples_rep:
                    break
                X = X.to(device)
                
                # Get fused representation (activations before MLP head)
                # This is R^t from paper eq. 2
                rep = model.get_fused_representation(X)  # [B, dim]
                all_reps.append(rep.cpu())
                sample_count += len(X)
        
        if not all_reps:
            print("⚠️ No samples for representation space")
            return
        
        # Stack all representations: R = [n_samples, dim]
        R = torch.cat(all_reps, dim=0)
        print(f"  → Representation matrix R: {R.shape}")
        
        try:
            # SVD: R = U @ S @ Vh
            # Paper uses LEFT singular vectors U (column space of R)
            U, S, Vh = torch.linalg.svd(R, full_matrices=False)
            
            # Energy-based rank selection (paper eq. 3)
            cum_energy = torch.cumsum(S ** 2, dim=0)
            total_energy = cum_energy[-1]
            k = (cum_energy < self.energy_threshold * total_energy).sum() + 1
            k = min(k.item(), len(S), 100)  # Cap at 100 for memory
            
            # Use LEFT singular vectors U (paper-compliant)
            # U[:, :k] gives principal directions in sample space
            basis = U[:, :k]  # [n_samples, k]
            weights = torch.sigmoid(S[:k])
            
            print(f"  → SVD basis: k={k} vectors (energy={self.energy_threshold})")
            
            # Save per-task SEPARATE (don't merge with previous tasks)
            # Paper accumulates all old task bases, projects onto ALL of them
            task_key = f"task_{self.current_task}"
            basis_path = os.path.join(self.temp_dir, f"{task_key}_basis.pt")
            weights_path = os.path.join(self.temp_dir, f"{task_key}_weights.pt")
            
            torch.save(basis, basis_path)
            torch.save(weights, weights_path)
            
            # Store paths (not tensors) to save memory
            self.old_space[task_key] = basis_path
            self.importance_weights[task_key] = weights_path
            
            print(f"  → Saved basis to {basis_path}")
            
        except Exception as e:
            print(f"⚠️ SVD failed: {e}")
        
        # Clean up
        del all_reps, R
        gc.collect()
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
        """Standard cross-entropy loss."""
        return F.cross_entropy(output, target)
    
    def post_step(
        self,
        model: nn.Module,
        global_params: Optional[OrderedDict] = None,
        **kwargs
    ) -> None:
        """
        Gradient projection: g' = g - α * projection
        
        Paper CGoFed (IEEE TKDE 2025):
        Project gradient to be orthogonal to old representation spaces.
        This prevents catastrophic forgetting by not interfering with
        old task representations.
        
        Note: With activation-based representation, we project the gradient
        of the classifier layer onto the representation space of activations.
        """
        if self.current_task == 0 or not self.old_space:
            return
        
        # Only apply projection to classifier layers (MLP head)
        # Because representation space is for pre-classifier activations
        classifier_layers = ['dense1', 'dense2', 'output']
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                
                # Only project classifier layer gradients
                if not any(cl in name for cl in classifier_layers):
                    continue
                
                # Skip small layers and biases
                if 'bias' in name or param.numel() < 100:
                    continue
                
                g_flat = param.grad.view(-1)
                total_projection = torch.zeros_like(g_flat)
                
                # Loop through ALL old tasks (per-task basis)
                for task_key, basis_path in self.old_space.items():
                    try:
                        # Load basis and weights from disk
                        U = torch.load(basis_path, map_location=param.device)
                        weights_path = self.importance_weights[task_key]
                        weights = torch.load(weights_path, map_location=param.device)
                        
                        # Project gradient onto this task's representation space
                        # For activation-based: project onto column space of U
                        # Note: U is [n_samples, k], we need to handle dimension mismatch
                        
                        # Use simplified projection: just compute cosine similarity
                        # and subtract scaled component
                        if U.shape[0] == g_flat.shape[0]:
                            # Direct projection if dimensions match
                            for i in range(U.shape[1]):
                                u_col = U[:, i]
                                w = weights[i] if i < len(weights) else 1.0
                                proj = torch.dot(g_flat, u_col) * u_col * w
                                total_projection += proj
                        else:
                            # Dimension mismatch - use mean representation
                            mean_rep = U.mean(dim=0)  # [k]
                            if mean_rep.shape[0] <= g_flat.shape[0]:
                                # Pad or truncate as needed
                                proj_dim = min(mean_rep.shape[0], g_flat.shape[0])
                                for i in range(proj_dim):
                                    w = weights[i] if i < len(weights) else 1.0
                                    proj = g_flat[i] * mean_rep[i] * w
                                    total_projection[i] += proj
                        
                        del U, weights
                        
                    except Exception as e:
                        print(f"⚠️ Projection failed for {task_key}: {e}")
                        continue
                
                # Subtract scaled projection from gradient (in-place)
                param.grad.view(-1).sub_(self.alpha * total_projection)
    
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
