"""
CGoFed Strategy - Constrained Gradient Optimization for Federated Class Incremental Learning.

Reference:
    "CGoFed: Constrained Gradient Optimization Strategy for Federated Class 
    Incremental Learning", IEEE TKDE, Vol. 37, No. 5, May 2025
    Authors: Jiyuan Feng, Xu Yang, Liwen Liang, Weihong Han, Binxing Fang, Qing Liao

Paper Algorithm Summary:
========================
1. Representation Space (Eq. 2-4):
   - R^t = F(Θ^t, X^t) = [z_1, ..., z_n] - representations from FORWARD propagation
   - SVD: R^t = U^t Σ^t (V^t)^T
   - Select κ basis vectors based on energy threshold
   - M^t = [u_1, ..., u_κ] - basis from LEFT singular vectors (U)

2. Importance Weights (Eq. 5):
   - Λ^t = sigmoid(Σ^t) = 1 / (1 + exp(-σ_i))
   - Weighted basis: M^t = [λ_1 u_1, ..., λ_κ u_κ]

3. Relaxation Coefficient (Eq. 7-8):
   - f(α, t) = α^t (power decay, NOT exponential)
   - μ_t = μ_init * α^(t - t_τ) when AF >= τ

4. Gradient Projection (Eq. 9):
   - ∇L ← ∇L - μ_t * (∇L) @ M^t @ (M^t)^T
   - Remove components in old task representation space

5. Cross-Task Regularization (Eq. 10-14):
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
    CGoFed local training with ACTIVATION-based SVD gradient constraint.
    
    Implements the Relax-Constrained Gradient Update from paper Section 5.1.
    
    Key mechanism (Paper Eq. 2-9):
    - R^t = F(Θ^t, X^t): Collect ACTIVATIONS via forward propagation
    - SVD: R^t = U^t Σ^t (V^t)^T
    - Basis M^t = [u_1, ..., u_κ]: LEFT singular vectors (U)
    - Importance Λ^t = sigmoid(Σ^t)
    - Projection: ∇L ← ∇L - μ_t * (∇L @ M^t @ M^t^T)
    - Adaptive μ_t relaxation: μ_t = α^(t - t_τ)
    
    Args:
        mu: Proximal regularization weight (optional, default 0.01)
        lambda_decay: Decay rate α for relaxation (paper Eq. 7: f(α,t) = α^t)
        theta_threshold: AF threshold τ to reset μ (paper Eq. 8)
        energy_threshold: SVD energy threshold for rank κ selection
        beta: Scaling factor for sigmoid importance weights
        num_samples_rep: Number of samples n_s for building representation
        temp_dir: Directory for storing SVD basis matrices
    """
    
    def __init__(
        self,
        mu: float = 0.01,
        lambda_decay: float = 0.1,
        theta_threshold: float = 0.1,
        energy_threshold: float = 0.95,
        beta: float = 1.0,  # Scaling for sigmoid importance
        num_samples_rep: int = 100,
        temp_dir: str = "./temp_svd_storage",
    ):
        self.mu = mu
        self.lambda_decay = lambda_decay
        self.theta_threshold = theta_threshold
        self.energy_threshold = energy_threshold
        self.beta = beta
        self.num_samples_rep = num_samples_rep
        
        # Temp directory for SVD matrices (lazy loading)
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Task tracking
        self.current_task: int = 0
        self.seen_classes: Set[int] = set()
        
        # Per-layer representation space
        # Key: task_key -> {layer_idx: {"basis": path, "importance": path}}
        self.layer_bases: Dict[str, Dict[int, Dict[str, str]]] = {}
        
        # Legacy attributes for compatibility
        self.old_space: Dict[str, str] = {}
        self.importance_weights: Dict[str, str] = {}
        
        # Gradient dimension (set when first building representation)
        self.gradient_dim: Optional[int] = None
        
        # μ_t relaxation coefficient (paper Eq. 7-8)
        # μ_t = μ_init * α^(t - t_reset) where α = lambda_decay
        self.mu_coefficient: float = 1.0  # Starts at 1.0, decays each task
        self.t_reset: int = 0  # Reset point when AF > θ
        
        # Accuracies for computing AF (Average Forgetting)
        self.best_acc_per_task: Dict[int, float] = {}
        self.current_acc_per_task: Dict[int, float] = {}
        self.last_af: float = 0.0
        
        # Cache for per-layer projection matrices (loaded once, used many times)
        # {layer_idx: projection_matrix}
        self._cached_proj_matrices: Optional[Dict[int, torch.Tensor]] = None
    
    def set_task(self, task_id: int, new_classes: List[int]):
        """
        Called at the beginning of each new task.
        
        Updates μ_t according to paper Eq. 7-8:
        - Eq. 7: f(α, t) = α^t
        - Eq. 8: μ_t = μ_init * f(α, t - t_τ) if AF >= τ
        
        Where α is the decay rate (lambda_decay in our code).
        """
        self.current_task = task_id
        self.seen_classes.update(new_classes)
        
        # Invalidate cache when task changes (new basis may be added)
        self._cached_proj_matrices = None
        
        # Update μ with power decay (paper Eq. 7-8)
        # μ_t = μ_init * α^(t - t_reset)
        if task_id > 0:
            # Paper: f(α, t) = α^t, so μ_t = μ_init * α^(t - t_reset)
            self.mu_coefficient = self.lambda_decay ** (task_id - self.t_reset)
            print(f"  μ_t = {self.lambda_decay}^({task_id} - {self.t_reset}) = {self.mu_coefficient:.4f}")
    
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
                
                # Check if need to reset μ (paper Eq. 8)
                if self.last_af > self.theta_threshold:
                    self.t_reset = self.current_task
                    self.mu_coefficient = 1.0
                    print(f"⚠️ AF={self.last_af:.4f} > θ={self.theta_threshold}, reset μ to 1.0")
    
    def get_current_af(self) -> float:
        """Get the last computed Average Forgetting value."""
        return self.last_af
    
    def _get_weight_modules(self, model: nn.Module) -> List[Tuple[str, nn.Module]]:
        """
        Get weight modules (Conv, Linear) for activation hooks.
        
        Paper Eq. 2: R^t = F(Θ^t, X^t)
        We collect input activations to these layers during forward pass.
        
        Returns:
            List of (module_name, module) tuples for Conv/Linear layers
        """
        weight_modules = []
        for name, module in model.named_modules():
            # Only Conv and Linear layers
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                # Skip batch norm (they don't have 'weight' in typical naming)
                if 'bn' in name or 'batch' in name.lower():
                    continue
                weight_modules.append((name, module))
        return weight_modules
    
    def _collect_activations(
        self,
        model: nn.Module,
        data_loader,
        device: str,
        num_samples: int
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Collect per-layer INPUT ACTIVATIONS during forward pass.
        
        Paper Eq. 2: R^t = F(Θ^t, X^t)
        This is the representation obtained through forward propagation.
        
        For each weight layer (Conv/Linear), we register a forward hook
        to capture the INPUT tensor to that layer.
        
        Args:
            model: Model to collect activations from
            data_loader: Data loader for samples
            device: Device to use
            num_samples: Number of samples to use
            
        Returns:
            Dict mapping module_name -> list of input activation tensors
        """
        weight_modules = self._get_weight_modules(model)
        layer_activations: Dict[str, List[torch.Tensor]] = {name: [] for name, _ in weight_modules}
        
        # Storage for captured activations
        captured: Dict[str, torch.Tensor] = {}
        handles = []
        
        def make_hook(layer_name: str):
            def hook_fn(module, inp, out):
                # inp is a tuple, take first element
                if isinstance(inp, tuple) and len(inp) > 0:
                    activation = inp[0]
                else:
                    activation = inp
                # Flatten: [batch, ...] -> [batch, d]
                activation = activation.detach().view(activation.size(0), -1)
                captured[layer_name] = activation.cpu()
            return hook_fn
        
        # Register hooks
        for name, module in weight_modules:
            handle = module.register_forward_hook(make_hook(name))
            handles.append(handle)
        
        # Collect activations via forward pass
        model.eval()
        sample_count = 0
        
        with torch.no_grad():
            for X, y in data_loader:
                if sample_count >= num_samples:
                    break
                
                X = X.to(device)
                batch_size = X.size(0)
                
                # Forward pass triggers hooks
                _ = model(X)
                
                # Store captured activations for each layer
                for name, _ in weight_modules:
                    if name in captured:
                        # Append each sample's activation
                        layer_activations[name].append(captured[name])
                
                sample_count += batch_size
                captured.clear()
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        return layer_activations
    
    def build_space_from_client_data(
        self,
        model: nn.Module,
        client_data: Dict,
        config: Dict,
        device: str = "cuda"
    ):
        """
        Encapsulated method to build representation space from client data.
        
        This method handles the sampling logic internally, keeping the runner
        file clean and algorithm-agnostic.
        
        Paper CGoFed Section 5.1:
        - Sample uniformly from available clients for robust gradient space
        - Uses gradient vectors from samples (not activations)
        - SVD decomposition to find principal gradient directions
        
        Args:
            model: Global model to compute gradients on
            client_data: Dict of {client_id: {"X_train": tensor, "y_train": tensor}}
            config: Configuration dict with "num_samples_rep"
            device: Device to run computation on
        """
        print("\n🔐 Building gradient-based representation space (paper Section 5.1)...")
        
        # Find clients with data
        active_cids = [cid for cid, data in client_data.items() 
                       if len(data.get("y_train", [])) > 0]
        
        if len(active_cids) == 0:
            print("⚠️ No client data available for representation space")
            return
        
        # Sample uniformly from available clients for robust gradient space
        # This ensures diversity in the representation
        all_X, all_y = [], []
        num_samples = config.get("num_samples_rep", self.num_samples_rep)
        samples_per_client = max(10, num_samples // len(active_cids) + 1)
        
        for cid in active_cids:
            X = client_data[cid]["X_train"]
            y = client_data[cid]["y_train"]
            
            # Random sample from this client
            if len(y) > samples_per_client:
                indices = torch.randperm(len(y))[:samples_per_client]
                X, y = X[indices], y[indices]
            
            all_X.append(X)
            all_y.append(y)
        
        # Concatenate all samples
        rep_X = torch.cat(all_X, dim=0)
        rep_y = torch.cat(all_y, dim=0)
        
        # Limit to num_samples_rep
        if len(rep_y) > num_samples:
            indices = torch.randperm(len(rep_y))[:num_samples]
            rep_X, rep_y = rep_X[indices], rep_y[indices]
        
        print(f"   Using {len(rep_y)} samples from {len(active_cids)} clients for SVD.")
        
        # Create DataLoader
        from torch.utils.data import TensorDataset, DataLoader
        rep_loader = DataLoader(
            TensorDataset(rep_X, rep_y),
            batch_size=32,  # Small batch for per-sample gradient computation
            shuffle=False
        )
        
        # Delegate to the core build method
        self.build_representation_space(
            model=model,
            data_loader=rep_loader,
            device=device
        )
    
    def build_representation_space(
        self,
        model: nn.Module,
        data_loader,
        device: str = "cuda"
    ):
        """
        Build PER-LAYER representation space using ACTIVATIONS (Paper Eq. 2-5).
        
        Paper Eq. 2: R^t = F(Θ^t, X^t)
        - R^t is the representation from forward propagation
        
        Paper Eq. 3: SVD of R^t = U Σ V^T
        - Basis M^t = [u_1, ..., u_κ] from left singular vectors U
        
        Paper Eq. 5: Importance Λ = sigmoid(Σ)
        - λ_i = 1 / (1 + exp(-σ_i))
        
        For each layer, we:
        1. Collect input activations during forward pass
        2. SVD on activation matrix
        3. Store U[:, :k] as basis and sigmoid(S[:k]) as importance
        """
        was_training = model.training
        
        print(f"  → Building ACTIVATION-based representation space (Paper Eq. 2-5)...")
        
        # Collect per-layer activations via forward hooks
        layer_activations = self._collect_activations(
            model, data_loader, device, self.num_samples_rep
        )
        
        if not layer_activations:
            print("⚠️ No activations collected")
            return
        
        # Get weight modules
        weight_modules = self._get_weight_modules(model)
        print(f"  → Found {len(weight_modules)} layers for projection")
        
        task_key = f"task_{self.current_task}"
        self.layer_bases[task_key] = {}
        
        # Process each layer with SVD
        for layer_name, module in weight_modules:
            if layer_name not in layer_activations or not layer_activations[layer_name]:
                continue
            
            try:
                # Stack activations: list of [batch, d] -> [N, d]
                R = torch.cat(layer_activations[layer_name], dim=0)  # [N, d]
                n_samples = R.shape[0]
                d = R.shape[1]
                
                # Paper Eq. 3: SVD of R^T (or R, depending on convention)
                # We use R^T so U columns span the feature space
                A = R.T  # [d, N]
                U, S, Vh = torch.linalg.svd(A, full_matrices=False)
                
                # Energy-based rank selection (Paper Eq. 4)
                energy = S ** 2
                cum_energy = torch.cumsum(energy, dim=0)
                total_energy = cum_energy[-1] + 1e-10
                ratio = cum_energy / total_energy
                
                k = (ratio < self.energy_threshold).sum().item() + 1
                k = min(k, len(S), 50)  # Cap per layer
                
                # Paper: M^t = [u_1, ..., u_κ] - left singular vectors
                basis = U[:, :k]  # [d, k]
                
                # Paper Eq. 5: Λ = sigmoid(Σ)
                importance = torch.sigmoid(self.beta * S[:k])
                
                # Save per-layer basis
                basis_path = os.path.join(self.temp_dir, f"{task_key}_{layer_name}_basis.pt")
                importance_path = os.path.join(self.temp_dir, f"{task_key}_{layer_name}_importance.pt")
                
                torch.save(basis.clone(), basis_path)
                torch.save(importance.clone(), importance_path)
                
                self.layer_bases[task_key][layer_name] = {
                    "basis": basis_path,
                    "importance": importance_path,
                    "shape": (d, k)
                }
                
                print(f"    Layer {layer_name}: R=[{n_samples}, {d}], k={k}")
                
            except Exception as e:
                print(f"  ⚠️ SVD failed for layer {layer_name}: {e}")
                continue
        
        # Print summary
        total_bases = sum(1 for task in self.layer_bases.values() for _ in task)
        print(f"  → Built {len(self.layer_bases[task_key])} layer bases for task {self.current_task}")
        print(f"  → Total bases across all tasks: {total_bases}")
        
        # Cleanup
        del layer_activations
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if was_training:
            model.train()
        
        print(f"✓ Built activation-based representation space for task {self.current_task}")
    
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
    
    def pre_step(
        self,
        model: nn.Module,
        global_params: Optional[OrderedDict] = None,
        **kwargs
    ) -> None:
        """
        Gradient projection BEFORE optimizer step (paper eq. 8).
        
        This is called between backward() and optimizer.step() to modify
        gradients before they are applied to weights.
        
        Paper CGoFed eq. 8:
        g' = g - α * Σ_j Σ_i (w_{j,i} * (v_{j,i}^T @ g) * v_{j,i})
        
        OPTIMIZED VERSION with NaN prevention:
        - Cache basis matrices in memory (no disk I/O per step)
        - Vectorized projection (no Python loop)
        - Matrix operation: proj = V^T @ diag(w) @ V @ g
        - Comprehensive NaN/Inf checks at every step
        """
        # Skip for first task (no old space to project against)
        if self.current_task == 0 or not self.layer_bases:
            return
        
        with torch.no_grad():
            device = next(model.parameters()).device
            
            # Lazy load and cache projection matrices
            if self._cached_proj_matrices is None:
                self._cache_projection_matrices(device)
            
            # Skip if no cached projections
            if not self._cached_proj_matrices:
                return
            
            # Get module name to param mapping
            module_params = {}
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                    if hasattr(module, 'weight') and module.weight is not None:
                        module_params[name] = module.weight
            
            # Apply projection per layer (Paper Eq. 9)
            for layer_name, Uf in self._cached_proj_matrices.items():
                if layer_name not in module_params:
                    continue
                
                param = module_params[layer_name]
                if param.grad is None:
                    continue
                
                # Move to device if needed
                if Uf.device != device:
                    Uf = Uf.to(device)
                    self._cached_proj_matrices[layer_name] = Uf
                
                # Flatten gradient for projection
                grad_flat = param.grad.view(-1)  # [out * in]
                
                # Check dimension compatibility
                # Uf shape: [d, d] where d = input activation dim
                # For Conv: d = C_in * kernel_size, For Linear: d = in_features
                if grad_flat.shape[0] != Uf.shape[0]:
                    # Uf dimension matches input activations, need to reshape grad
                    # grad: [out, in] or [out, in, k, k] -> we project on 'in' dimension
                    sz = param.grad.size(0)  # out_dim
                    in_dim = param.grad.numel() // sz
                    
                    if in_dim != Uf.shape[0]:
                        continue  # Dimension mismatch, skip
                    
                    grad_2d = param.grad.view(sz, in_dim)  # [out, in]
                    
                    # Apply projection: grad = grad - μ * (grad @ Uf)
                    projected = torch.mm(grad_2d, Uf)  # [out, in]
                    
                    if torch.isnan(projected).any() or torch.isinf(projected).any():
                        continue
                    
                    # Paper Eq. 9: g = g - μ_t * g @ M @ M^T
                    grad_new = grad_2d - self.mu_coefficient * projected
                    param.grad.copy_(grad_new.view_as(param.grad))
                else:
                    # Direct projection (unlikely for most architectures)
                    projected = torch.mv(Uf, grad_flat)
                    if torch.isnan(projected).any() or torch.isinf(projected).any():
                        continue
                    grad_new = grad_flat - self.mu_coefficient * projected
                    param.grad.copy_(grad_new.view_as(param.grad))
    
    def _cache_projection_matrices(self, device: str):
        """
        Build and cache projection matrices for all layers from all old tasks.
        
        Paper Eq. 9: Projection matrix M @ diag(Λ) @ M^T
        
        Where M = U[:, :k] is left singular vectors from Paper Eq. 3
        and Λ = sigmoid(Σ) from Paper Eq. 5
        """
        self._cached_proj_matrices = {}
        
        for task_key, layer_dict in self.layer_bases.items():
            for layer_name, info in layer_dict.items():
                try:
                    # Load basis and importance
                    basis = torch.load(info["basis"], map_location=device)
                    importance = torch.load(info["importance"], map_location=device)
                    
                    # Validate
                    if torch.isnan(basis).any() or torch.isinf(basis).any():
                        continue
                    if torch.isnan(importance).any() or torch.isinf(importance).any():
                        continue
                    
                    # Build projection matrix: Uf = M @ diag(Λ) @ M^T (Paper Eq. 9)
                    # basis: [d, k], importance: [k]
                    Uf = torch.mm(basis * importance, basis.T)  # [d, d]
                    
                    # Accumulate projections across tasks
                    if layer_name in self._cached_proj_matrices:
                        self._cached_proj_matrices[layer_name] += Uf
                    else:
                        self._cached_proj_matrices[layer_name] = Uf
                        
                except Exception as e:
                    print(f"⚠️ Failed to cache layer {layer_name}: {e}")
        
        if self._cached_proj_matrices:
            print(f"  📦 Cached {len(self._cached_proj_matrices)} layer projection matrices")
    
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
                
                # Skip NaN/Inf representations
                if torch.isnan(rep).any() or torch.isinf(rep).any():
                    print(f"  ⚠️ Skipping NaN representation from client {client_id}")
                    continue
                
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
        # Check for NaN/Inf
        if torch.isnan(R1).any() or torch.isinf(R1).any():
            return 0.0
        if torch.isnan(R2).any() or torch.isinf(R2).any():
            return 0.0
            
        # Normalize
        norm1 = torch.norm(R1)
        norm2 = torch.norm(R2)
        
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
            
        R1_norm = R1 / (norm1 + 1e-8)
        R2_norm = R2 / (norm2 + 1e-8)
        
        # Negative L2 distance (so higher = more similar)
        dist = torch.norm(R1_norm - R2_norm).item()
        
        # Clamp to prevent extreme values
        return max(-10.0, min(0.0, -dist))
    
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
        # Filter out NaN/Inf similarities first
        valid_models = [(s, i) for i, s in enumerate(selected_models) 
                        if not (math.isnan(s["similarity"]) or math.isinf(s["similarity"]))]
        
        if not valid_models:
            print("  ⚠️ All cross-task similarities are NaN/Inf, skipping historical blend")
            return current_params
            
        selected_models = [s for s, _ in valid_models]
        sim_scores = torch.tensor([s["similarity"] for s in selected_models])
        
        # Prevent softmax numerical issues
        sim_scores = torch.clamp(sim_scores, min=-10.0, max=10.0)
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
        
        # Final sanity check - if result has NaN, return current params
        for k, v in result.items():
            if v.dtype.is_floating_point and (torch.isnan(v).any() or torch.isinf(v).any()):
                print(f"  ⚠️ Historical blend produced NaN in {k}, using current params only")
                return current_params
        
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
