"""
FedLwF Client - Client for Federated Learning without Forgetting.

Extends FederatedClient with:
1. Old model caching for knowledge distillation
2. Combined loss computation (CE + KD)
3. Model snapshot saving after each task
"""

import contextlib
from collections import OrderedDict
from typing import Optional, Dict, Any, List, Set
import copy

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
from ..strategies.incremental.fedlwf import FedLwFTrainer


class FedLwFClient(FederatedClient):
    """
    Client for FedLwF algorithm with Knowledge Distillation.
    
    Key features:
    - Maintains old model snapshot for KD loss computation
    - Efficient caching to avoid repeated model loading
    - Task-aware training with distillation
    
    Attributes:
        old_model: Cached old model for distillation (None for task 0)
        current_task: Current task ID
        seen_classes: Set of all seen class IDs
    """
    
    def __init__(
        self,
        client_id: int,
        X_train: torch.Tensor,
        y_train: torch.Tensor
    ):
        """
        Initialize FedLwF client.
        
        Args:
            client_id: Client identifier
            X_train: Training features (CPU tensor)
            y_train: Training labels (CPU tensor)
        """
        super().__init__(client_id, X_train, y_train)
        
        # Old model for KD (cached per task)
        self.old_model: Optional[nn.Module] = None
        self.old_model_state: Optional[OrderedDict] = None
        
        # Task tracking
        self.current_task: int = 0
        self.seen_classes: Set[int] = set()
        self.old_classes: List[int] = []
        self.new_classes: List[int] = []
    
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
        
        # Update task tracking
        self.old_classes = list(self.seen_classes)
        self.new_classes = task_classes
        self.current_task = task_id
        self.seen_classes.update(task_classes)
    
    def save_model_snapshot(self, model: nn.Module):
        """
        Save model snapshot after task completion.
        
        This snapshot will be used as teacher for next task.
        """
        # Save state dict (CPU) for memory efficiency
        self.old_model_state = OrderedDict(
            (k, v.cpu().clone()) for k, v in model.state_dict().items()
        )
        
        # Clear cached old model (will reload with new state)
        self.old_model = None
    
    def _load_old_model(self, model_template: nn.Module, device: str):
        """
        Load old model from saved state for KD.
        
        Uses caching to avoid repeated loading.
        """
        if self.old_model_state is None:
            self.old_model = None
            return
        
        if self.old_model is not None:
            # Already loaded
            return
        
        try:
            # Create copy and load state
            self.old_model = copy.deepcopy(model_template)
            self.old_model.load_state_dict({
                k: v.to(device) for k, v in self.old_model_state.items()
            })
            self.old_model.eval()
            
            # Freeze
            for param in self.old_model.parameters():
                param.requires_grad = False
                
        except Exception as e:
            print(f"  Client {self.client_id}: Failed to load old model: {e}")
            self.old_model = None
    
    def train(
        self,
        trainer: BaseTrainer,
        epochs: int,
        batch_size: int,
        lr: float,
        global_params: Optional[OrderedDict] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train with Knowledge Distillation loss.
        
        Loss = CE(output, target) + α * KD(old_output, output)
        
        Args:
            trainer: FedLwFTrainer instance
            epochs: Local epochs
            batch_size: Batch size
            lr: Learning rate
            global_params: Global model parameters
            
        Returns:
            Dict with training results
        """
        self.model.train()
        
        # Get optimizer
        optimizer_cls = trainer.get_optimizer_class()
        optimizer = optimizer_cls(self.model.parameters(), lr=lr)
        scaler = GradScaler(enabled=self.use_amp)
        
        # Load old model for KD (if not first task)
        if self.current_task > 0 and self.old_model_state is not None:
            self._load_old_model(self.model, self.device)
        
        # Pre-train hook
        trainer.pre_train(self.model, global_params, lr=lr, **kwargs)
        
        total_loss = 0.0
        total_samples = 0
        
        for ep in range(epochs):
            for X_batch, y_batch in self._create_batches(batch_size):
                optimizer.zero_grad()
                
                with self._amp_ctx():
                    out = self.model(X_batch)
                    
                    # Compute loss with KD
                    if isinstance(trainer, FedLwFTrainer):
                        loss = trainer.compute_loss(
                            self.model, out, y_batch,
                            global_params=global_params,
                            inputs=X_batch,
                            old_model=self.old_model,
                            **kwargs
                        )
                    else:
                        # Fallback to standard loss
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
            "task_id": self.current_task,
        }
    
    def cleanup_old_model(self):
        """Clean up old model to free memory."""
        self.old_model = None
        # Keep old_model_state for potential future use
