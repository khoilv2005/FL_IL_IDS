"""
LwF (Learning without Forgetting) Strategy for Federated Incremental Learning.

Reference:
    Li & Hoiem, "Learning without Forgetting", ECCV 2016, IEEE TPAMI 2018

LwF prevents catastrophic forgetting by using Knowledge Distillation:
the new model is trained to match the soft outputs of the old model
on new task data.

Usage:
    # Combine with any Federated method using Mixin pattern:
    class FedAvgLwFTrainer(LwFMixin, FedAvgTrainer): pass
    class FedProxLwFTrainer(LwFMixin, FedProxTrainer): pass
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Set
import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import BaseTrainer


class LwFMixin:
    """
    LwF Mixin - adds Learning without Forgetting to any trainer.
    
    LwF Loss (Li & Hoiem, 2016):
        L = L_CE + α * T² * KL(σ(z_old/T) || σ(z_new/T))
    
    Where:
        - z_old: Logits from old model (frozen)
        - z_new: Logits from current model on new data
        - T: Temperature for soft targets
        - α: Distillation weight
    
    This mixin should be placed BEFORE the base trainer in MRO:
        class FedAvgLwFTrainer(LwFMixin, FedAvgTrainer): pass
    """
    
    def __init__(
        self,
        lwf_alpha: float = 1.0,
        temperature: float = 2.0,
        temp_dir: str = "./temp_lwf_storage",
        **kwargs
    ):
        """
        Args:
            lwf_alpha: Weight for distillation loss (α)
            temperature: Temperature for soft targets (T)
            temp_dir: Directory to store old models
        """
        # Call parent __init__ (for Mixin pattern)
        super().__init__(**kwargs)
        
        self.lwf_alpha = lwf_alpha
        self.temperature = temperature
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)
        
        # Storage for old model per task
        # {task_id: model_path}
        self.old_models: Dict[int, str] = {}
        
        # Cached old model for current training (avoid reloading)
        self._cached_old_model: Optional[nn.Module] = None
        self._cached_old_task: int = -1
        
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
        
        # Invalidate cached old model when task changes
        self._cached_old_model = None
        self._cached_old_task = -1
    
    def consolidate(
        self,
        model: nn.Module,
        data_loader=None,  # Not needed for LwF
        device: str = "cuda"
    ):
        """
        Consolidate knowledge after completing a task.
        
        For LwF, we save the current model state to use as teacher
        for future tasks.
        """
        print(f"  🔐 Saving old model for LwF (task {self.current_task})...")
        
        # Save model state
        task_key = f"task_{self.current_task}"
        model_path = os.path.join(self.temp_dir, f"{task_key}_model.pt")
        
        # Save only state dict (not full model)
        torch.save(model.state_dict(), model_path)
        
        self.old_models[self.current_task] = model_path
        
        print(f"  ✓ Stored model snapshot for task {self.current_task}")
    
    def _get_old_model(self, model: nn.Module, device: str) -> Optional[nn.Module]:
        """
        Get the old model for distillation.
        
        Uses caching to avoid reloading every batch.
        """
        if self.current_task == 0:
            return None
        
        # Use the model from the previous task
        prev_task = self.current_task - 1
        
        if prev_task not in self.old_models:
            return None
        
        # Check cache
        if self._cached_old_model is not None and self._cached_old_task == prev_task:
            return self._cached_old_model
        
        # Load old model
        try:
            old_state = torch.load(self.old_models[prev_task], map_location=device)
            
            # Create a copy of the current model architecture
            old_model = copy.deepcopy(model)
            old_model.load_state_dict(old_state)
            old_model.to(device)
            old_model.eval()
            
            # Freeze old model
            for param in old_model.parameters():
                param.requires_grad = False
            
            # Cache it
            self._cached_old_model = old_model
            self._cached_old_task = prev_task
            
            return old_model
            
        except Exception as e:
            print(f"⚠️ Failed to load old model: {e}")
            return None
    
    def _distillation_loss(
        self,
        old_logits: torch.Tensor,
        new_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Knowledge Distillation loss.
        
        L_KD = T² * KL(σ(z_old/T) || σ(z_new/T))
        """
        T = self.temperature
        
        # Soft targets from old model
        old_probs = F.softmax(old_logits / T, dim=1)
        
        # Log-softmax from new model
        new_log_probs = F.log_softmax(new_logits / T, dim=1)
        
        # KL divergence
        kd_loss = F.kl_div(new_log_probs, old_probs, reduction='batchmean')
        
        # Scale by T²
        return (T ** 2) * kd_loss
    
    def compute_loss(
        self, 
        model: nn.Module,
        output: torch.Tensor, 
        target: torch.Tensor,
        global_params: Optional[OrderedDict] = None,
        inputs: Optional[torch.Tensor] = None,  # Need input for old model
        **kwargs
    ) -> torch.Tensor:
        """
        Compute loss with LwF distillation.
        
        L = L_base + α * L_KD
        """
        # Get base loss from parent (FedAvg, FedProx, etc.)
        base_loss = super().compute_loss(model, output, target, global_params, **kwargs)
        
        # No distillation for first task
        if self.current_task == 0 or inputs is None:
            return base_loss
        
        device = next(model.parameters()).device
        
        # Get old model
        old_model = self._get_old_model(model, device)
        if old_model is None:
            return base_loss
        
        # Get old model outputs
        with torch.no_grad():
            old_logits = old_model(inputs)
        
        # Compute distillation loss
        kd_loss = self._distillation_loss(old_logits, output)
        
        return base_loss + self.lwf_alpha * kd_loss
    
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

class FedAvgLwFTrainer(LwFMixin, FedAvgTrainer):
    """FedAvg + LwF: Standard averaging with Knowledge Distillation."""
    pass

class FedProxLwFTrainer(LwFMixin, FedProxTrainer):
    """FedProx + LwF: Proximal regularization with Knowledge Distillation."""
    pass
