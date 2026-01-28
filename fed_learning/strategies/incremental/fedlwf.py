"""
FedLwF Strategy - Federated Learning without Forgetting for Class-Incremental Learning.

Reference:
    Li & Hoiem, "Learning without Forgetting", ECCV 2016, IEEE TPAMI 2018
    
    Adapted for Federated Learning setting where:
    - Multiple clients learn incrementally
    - Knowledge distillation prevents forgetting across tasks
    - Server coordinates global model, clients use local old model snapshots

FedLwF = FedAvg + Knowledge Distillation (LwF)

Key Mechanism:
    L_total = L_CE(new_data) + α * T² * KL(σ(z_old/T) || σ(z_new/T))

Where:
    - L_CE: Cross-entropy loss on new task data
    - z_old: Logits from old (frozen) model
    - z_new: Logits from current model
    - T: Temperature for soft targets (default: 2.0)
    - α: Distillation weight (default: 1.0)
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple
import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import BaseTrainer, BaseAggregator


class FedLwFTrainer(BaseTrainer):
    """
    FedLwF Trainer - Federated Learning without Forgetting.
    
    Implements Knowledge Distillation to prevent catastrophic forgetting:
    1. Save model snapshot after each task (teacher)
    2. Train new model (student) to match teacher's soft outputs
    3. Combined loss: CE on new data + KD loss on all data
    
    Args:
        lwf_alpha: Weight for distillation loss (α), default 1.0
        temperature: Temperature for soft targets (T), default 2.0
        distill_on_new_only: If True, only distill on new task data
                            If False, distill on all data including replay
        temp_dir: Directory to store old model snapshots
    """
    
    def __init__(
        self,
        lwf_alpha: float = 1.0,
        temperature: float = 2.0,
        distill_on_new_only: bool = False,
        temp_dir: str = "./temp_fedlwf_storage",
        **kwargs
    ):
        self.lwf_alpha = lwf_alpha
        self.temperature = temperature
        self.distill_on_new_only = distill_on_new_only
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)
        
        # Storage for old model per task
        # {task_id: model_state_dict (CPU)}
        self.old_model_states: Dict[int, OrderedDict] = {}
        
        # Cached old model for current training session
        self._cached_old_model: Optional[nn.Module] = None
        self._cached_device: Optional[str] = None
        
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
        
        Updates class tracking and prepares for distillation.
        """
        # Store old classes before updating
        self.old_classes = list(self.seen_classes)
        self.new_classes = new_classes
        
        self.current_task = task_id
        self.seen_classes.update(new_classes)
        
        # Invalidate cached model
        self._cached_old_model = None
        
        print(f"  FedLwF Task {task_id}: old_classes={len(self.old_classes)}, "
              f"new_classes={len(new_classes)}, α={self.lwf_alpha}, T={self.temperature}")
    
    def save_model_snapshot(self, model: nn.Module):
        """
        Save model snapshot after completing a task.
        
        This snapshot serves as the "teacher" for future tasks.
        """
        # Save state dict to CPU (memory efficient)
        state_dict = OrderedDict(
            (k, v.cpu().clone()) for k, v in model.state_dict().items()
        )
        self.old_model_states[self.current_task] = state_dict
        
        # Also save to disk for persistence
        path = os.path.join(self.temp_dir, f"task_{self.current_task}_model.pt")
        torch.save(state_dict, path)
        
        print(f"  📸 Saved model snapshot for Task {self.current_task}")
    
    def load_old_model(
        self,
        model_template: nn.Module,
        device: str
    ) -> Optional[nn.Module]:
        """
        Load the old model for knowledge distillation.
        
        Uses the model from the previous task as teacher.
        Returns None if no old model exists (first task).
        """
        if self.current_task == 0:
            return None
        
        prev_task = self.current_task - 1
        
        # Check cache
        if (self._cached_old_model is not None and 
            self._cached_device == device):
            return self._cached_old_model
        
        # Get state dict
        if prev_task in self.old_model_states:
            state_dict = self.old_model_states[prev_task]
        else:
            # Try loading from disk
            path = os.path.join(self.temp_dir, f"task_{prev_task}_model.pt")
            if os.path.exists(path):
                state_dict = torch.load(path, map_location='cpu')
            else:
                print(f"  ⚠️ No old model found for task {prev_task}")
                return None
        
        # Create old model from template
        try:
            old_model = copy.deepcopy(model_template)
            old_model.load_state_dict({
                k: v.to(device) for k, v in state_dict.items()
            })
            old_model.eval()
            
            # Freeze parameters
            for param in old_model.parameters():
                param.requires_grad = False
            
            # Cache
            self._cached_old_model = old_model
            self._cached_device = device
            
            return old_model
            
        except Exception as e:
            print(f"  ⚠️ Failed to load old model: {e}")
            return None
    
    def compute_distillation_loss(
        self,
        old_logits: torch.Tensor,
        new_logits: torch.Tensor,
        old_class_indices: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Compute Knowledge Distillation loss.
        
        L_KD = T² * KL(σ(z_old/T) || σ(z_new/T))
        
        If old_class_indices provided, only distill on those classes.
        """
        T = self.temperature
        
        if old_class_indices is not None and len(old_class_indices) > 0:
            # Only distill on old classes (more targeted)
            old_indices = torch.tensor(old_class_indices, device=old_logits.device)
            old_logits = old_logits[:, old_indices]
            new_logits = new_logits[:, old_indices]
        
        # Soft targets from old model
        old_probs = F.softmax(old_logits / T, dim=1)
        
        # Log-softmax from new model
        new_log_probs = F.log_softmax(new_logits / T, dim=1)
        
        # KL divergence: KL(P || Q) = Σ P(x) * log(P(x)/Q(x))
        # = Σ P(x) * (log P(x) - log Q(x))
        kd_loss = F.kl_div(new_log_probs, old_probs, reduction='batchmean')
        
        # Scale by T² (as per Hinton et al.)
        return (T ** 2) * kd_loss
    
    def compute_loss(
        self,
        model: nn.Module,
        output: torch.Tensor,
        target: torch.Tensor,
        global_params: Optional[OrderedDict] = None,
        inputs: Optional[torch.Tensor] = None,
        old_model: Optional[nn.Module] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute FedLwF loss = CE + α * KD.
        
        Args:
            model: Current model being trained
            output: Model predictions (logits)
            target: Ground truth labels
            global_params: Not used for FedLwF
            inputs: Input features (needed for old model forward)
            old_model: Pre-loaded old model (optional, for efficiency)
            
        Returns:
            Combined loss tensor
        """
        # Cross-entropy loss on new task data
        ce_loss = F.cross_entropy(output, target)
        
        # No distillation for first task or if no inputs
        if self.current_task == 0 or inputs is None:
            return ce_loss
        
        # Get old model
        if old_model is None:
            device = next(model.parameters()).device
            old_model = self.load_old_model(model, device)
        
        if old_model is None:
            return ce_loss
        
        # Get old model outputs
        with torch.no_grad():
            old_logits = old_model(inputs)
        
        # Compute distillation loss
        # Option: only distill on old classes for more targeted preservation
        if self.distill_on_new_only:
            kd_loss = self.compute_distillation_loss(
                old_logits, output, 
                old_class_indices=self.old_classes if self.old_classes else None
            )
        else:
            kd_loss = self.compute_distillation_loss(old_logits, output)
        
        return ce_loss + self.lwf_alpha * kd_loss
    
    def update_forgetting(self, task_accuracies: Dict[int, float]):
        """Update accuracy tracking for forgetting metrics."""
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
                    forgetting = (self.best_acc_per_task[task_id] - 
                                 self.current_acc_per_task[task_id])
                    forgetting_sum += max(0, forgetting)
                    count += 1
            
            self.last_af = forgetting_sum / max(1, count)
    
    def cleanup(self):
        """Clean up cached models and temporary files."""
        self._cached_old_model = None
        self.old_model_states.clear()
        
        # Optionally clean temp directory
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


class FedLwFAggregator(BaseAggregator):
    """
    FedLwF Aggregation - Standard FedAvg weighted average.
    
    FedLwF uses standard FedAvg for aggregation.
    The regularization (knowledge distillation) happens at the local training level.
    """
    
    def aggregate(
        self,
        results: List[Dict],
        global_params: Optional[OrderedDict] = None,
        **kwargs
    ) -> OrderedDict:
        """Standard FedAvg aggregation."""
        return self._weighted_average(results)


class FedLwFWithProximalTrainer(FedLwFTrainer):
    """
    FedLwF + Proximal regularization (combines LwF with FedProx).
    
    Loss = CE + α * KD + (μ/2) * ||w - w_global||²
    
    This variant adds FedProx's proximal term for better handling of
    heterogeneous data distributions.
    """
    
    def __init__(
        self,
        lwf_alpha: float = 1.0,
        temperature: float = 2.0,
        mu: float = 0.01,
        **kwargs
    ):
        super().__init__(lwf_alpha=lwf_alpha, temperature=temperature, **kwargs)
        self.mu = mu
    
    def compute_loss(
        self,
        model: nn.Module,
        output: torch.Tensor,
        target: torch.Tensor,
        global_params: Optional[OrderedDict] = None,
        inputs: Optional[torch.Tensor] = None,
        old_model: Optional[nn.Module] = None,
        **kwargs
    ) -> torch.Tensor:
        """Compute FedLwF + Proximal loss."""
        # Get FedLwF loss (CE + KD)
        lwf_loss = super().compute_loss(
            model, output, target, global_params, inputs, old_model, **kwargs
        )
        
        # Add proximal term if global params provided
        if global_params is None:
            return lwf_loss
        
        prox_term = 0.0
        for name, param in model.named_parameters():
            if name in global_params:
                global_param = global_params[name].to(param.device)
                prox_term += ((param - global_param) ** 2).sum()
        
        return lwf_loss + (self.mu / 2) * prox_term
