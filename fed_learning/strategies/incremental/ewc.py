"""
EWC (Elastic Weight Consolidation) Strategy for Federated Incremental Learning.

Reference:
    Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks",
    PNAS 2017

EWC prevents catastrophic forgetting by adding a regularization term that
penalizes changes to parameters that were important for previous tasks.

Usage:
    # Combine with any Federated method using Mixin pattern:
    class FedAvgEWCTrainer(EWCMixin, FedAvgTrainer): pass
    class FedProxEWCTrainer(EWCMixin, FedProxTrainer): pass
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Set
import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import BaseTrainer


class EWCMixin:
    """
    EWC Mixin - adds Elastic Weight Consolidation to any trainer.
    
    EWC Loss (Kirkpatrick et al., 2017):
        L = L_base + (λ/2) * Σ_i F_i * (θ_i - θ_i*)²
    
    Where:
        - F_i: Fisher Information for parameter i (importance)
        - θ_i*: Optimal parameters after previous task
        - λ: EWC regularization strength
    
    This mixin should be placed BEFORE the base trainer in MRO:
        class FedAvgEWCTrainer(EWCMixin, FedAvgTrainer): pass
    """
    
    def __init__(
        self,
        ewc_lambda: float = 1000.0,
        fisher_samples: int = 200,
        online_ewc: bool = False,
        gamma: float = 0.9,
        temp_dir: str = "./temp_ewc_storage",
    ):
        """
        Args:
            ewc_lambda: Regularization strength (λ in paper)
            fisher_samples: Number of samples for Fisher computation
            online_ewc: If True, use Online EWC (running average of Fisher)
            gamma: Decay factor for Online EWC
            temp_dir: Directory to store Fisher matrices
        """
        self.ewc_lambda = ewc_lambda
        self.fisher_samples = fisher_samples
        self.online_ewc = online_ewc
        self.gamma = gamma
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)
        
        # Storage for Fisher matrices and optimal params per task
        # {task_id: {"fisher": path, "params": path}}
        self.ewc_data: Dict[int, Dict[str, str]] = {}
        
        # Task tracking
        self.current_task: int = 0
        self.seen_classes: Set[int] = set()
        
        # For AF calculation (compatibility with training script)
        self.mu_coefficient: float = 1.0
        self.best_acc_per_task: Dict[int, float] = {}
        self.current_acc_per_task: Dict[int, float] = {}
        self.last_af: float = 0.0
    
    def set_task(self, task_id: int, new_classes: List[int]):
        """Called at the beginning of each new task."""
        self.current_task = task_id
        self.seen_classes.update(new_classes)
    
    def compute_fisher_information(
        self,
        model: nn.Module,
        data_loader,
        device: str = "cuda"
    ):
        """
        Compute Fisher Information Matrix (diagonal approximation).
        
        F_i = E[(∂L/∂θ_i)²]
        
        Approximated using empirical samples.
        """
        # Switch to train mode to allow RNN backward pass (required by cuDNN)
        # But we want to freeze Batch Norm stats and disable Dropout for deterministic Fisher
        model.train()
        
        # Optional: Disable dropout manually for stable Fisher estimation
        for m in model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.training = False
        fisher = {name: torch.zeros_like(param) for name, param in model.named_parameters()
                  if param.requires_grad}
        
        sample_count = 0
        criterion = nn.CrossEntropyLoss()
        
        for X, y in data_loader:
            if sample_count >= self.fisher_samples:
                break
            
            X, y = X.to(device), y.to(device)
            
            # Compute gradients for each sample
            for i in range(len(X)):
                if sample_count >= self.fisher_samples:
                    break
                
                model.zero_grad()
                output = model(X[i:i+1])
                
                # Use log-likelihood (negative loss)
                loss = criterion(output, y[i:i+1])
                loss.backward()
                
                # Accumulate squared gradients
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        fisher[name] += param.grad.detach() ** 2
                
                sample_count += 1
        
        # Average
        for name in fisher:
            fisher[name] /= sample_count
        
        return fisher
    
    def consolidate(
        self,
        model: nn.Module,
        data_loader,
        device: str = "cuda"
    ):
        """
        Consolidate knowledge after completing a task.
        
        Computes Fisher Information and stores optimal parameters.
        """
        print(f"  🔐 Computing Fisher Information for EWC (task {self.current_task})...")
        
        # Compute Fisher
        fisher = self.compute_fisher_information(model, data_loader, device)
        
        # Store optimal parameters
        optimal_params = {name: param.detach().cpu().clone() 
                         for name, param in model.named_parameters()
                         if param.requires_grad}
        
        # Online EWC: running average of Fisher
        if self.online_ewc and self.current_task > 0:
            prev_task = self.current_task - 1
            if prev_task in self.ewc_data:
                prev_fisher = torch.load(self.ewc_data[prev_task]["fisher"])
                for name in fisher:
                    if name in prev_fisher:
                        fisher[name] = self.gamma * prev_fisher[name].to(device) + fisher[name]
        
        # Save to disk
        task_key = f"task_{self.current_task}"
        fisher_path = os.path.join(self.temp_dir, f"{task_key}_fisher.pt")
        params_path = os.path.join(self.temp_dir, f"{task_key}_params.pt")
        
        # Move Fisher to CPU for storage
        fisher_cpu = {name: f.cpu() for name, f in fisher.items()}
        torch.save(fisher_cpu, fisher_path)
        torch.save(optimal_params, params_path)
        
        self.ewc_data[self.current_task] = {
            "fisher": fisher_path,
            "params": params_path
        }
        
        print(f"  ✓ Stored Fisher matrix and optimal params for task {self.current_task}")
    
    def compute_loss(
        self, 
        model: nn.Module,
        output: torch.Tensor, 
        target: torch.Tensor,
        global_params: Optional[OrderedDict] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute loss with EWC regularization.
        
        L = L_base + (λ/2) * Σ_i F_i * (θ_i - θ_i*)²
        """
        # Get base loss from parent (FedAvg, FedProx, etc.)
        base_loss = super().compute_loss(model, output, target, global_params, **kwargs)
        
        # No EWC penalty for first task
        if not self.ewc_data:
            return base_loss
        
        device = next(model.parameters()).device
        ewc_penalty = torch.tensor(0.0, device=device)
        
        # Sum over all previous tasks (or just latest for Online EWC)
        tasks_to_consider = [max(self.ewc_data.keys())] if self.online_ewc else list(self.ewc_data.keys())
        
        for task_id in tasks_to_consider:
            if task_id not in self.ewc_data:
                continue
            
            # Load Fisher and optimal params
            fisher = torch.load(self.ewc_data[task_id]["fisher"], map_location=device)
            optimal_params = torch.load(self.ewc_data[task_id]["params"], map_location=device)
            
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if name not in fisher or name not in optimal_params:
                    continue
                
                # EWC penalty: F_i * (θ_i - θ_i*)²
                diff = param - optimal_params[name].to(device)
                ewc_penalty += (fisher[name].to(device) * diff ** 2).sum()
        
        return base_loss + (self.ewc_lambda / 2.0) * ewc_penalty
    
    def update_forgetting(self, task_accuracies: Dict[int, float]):
        """Update Average Forgetting metrics (for compatibility)."""
        self.current_acc_per_task = task_accuracies.copy()
        
        for tid, acc in task_accuracies.items():
            if tid not in self.best_acc_per_task:
                self.best_acc_per_task[tid] = acc
            else:
                self.best_acc_per_task[tid] = max(self.best_acc_per_task[tid], acc)
        
        # Calculate AF
        self.last_af = 0.0
        if len(self.best_acc_per_task) > 1:
            forgetting = []
            for tid in range(self.current_task):
                if tid in self.best_acc_per_task and tid in self.current_acc_per_task:
                    f = self.best_acc_per_task[tid] - self.current_acc_per_task[tid]
                    forgetting.append(max(0, f))
            if forgetting:
                self.last_af = sum(forgetting) / len(forgetting)
    
    def get_current_af(self) -> float:
        """Get the last computed Average Forgetting value."""
        return self.last_af


# Pre-built combined trainers for convenience
from ..federated.fedavg import FedAvgTrainer
from ..federated.fedprox import FedProxTrainer

class FedAvgEWCTrainer(EWCMixin, FedAvgTrainer):
    """FedAvg + EWC: Standard averaging with EWC regularization."""
    def __init__(self, **kwargs):
        ewc_args = {
            'ewc_lambda': kwargs.pop('ewc_lambda', 1000.0),
            'fisher_samples': kwargs.pop('fisher_samples', 200),
            'online_ewc': kwargs.pop('online_ewc', False),
            'gamma': kwargs.pop('gamma', 0.9),
            'temp_dir': kwargs.pop('temp_dir', "./temp_ewc_storage")
        }
        EWCMixin.__init__(self, **ewc_args)
        FedAvgTrainer.__init__(self)

class FedProxEWCTrainer(EWCMixin, FedProxTrainer):
    """FedProx + EWC: Proximal regularization with EWC."""
    def __init__(self, **kwargs):
        ewc_args = {
            'ewc_lambda': kwargs.pop('ewc_lambda', 1000.0),
            'fisher_samples': kwargs.pop('fisher_samples', 200),
            'online_ewc': kwargs.pop('online_ewc', False),
            'gamma': kwargs.pop('gamma', 0.9),
            'temp_dir': kwargs.pop('temp_dir', "./temp_ewc_storage")
        }
        mu = kwargs.pop('mu', 0.01)
        EWCMixin.__init__(self, **ewc_args)
        FedProxTrainer.__init__(self, mu=mu)
