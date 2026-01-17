"""
Federated Client with Multi-GPU Support and Strategy Pattern.
"""

import contextlib
from collections import OrderedDict
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from torch.amp import autocast as torch_autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast as torch_autocast, GradScaler

from ..core import BaseTrainer


class FederatedClient:
    """
    Client hỗ trợ Multi-GPU và Strategy Pattern:
    - Data giữ trên CPU
    - Khi train, load batch lên GPU được chỉ định
    - Training logic delegated to Trainer strategy
    """
    
    def __init__(self, client_id: int, X_train: torch.Tensor, y_train: torch.Tensor):
        self.client_id = client_id
        self.X_train = X_train  # CPU tensor
        self.y_train = y_train  # CPU tensor
        self.num_samples = len(y_train)
        
        # Model sẽ được set sau khi assign GPU
        self.model: Optional[nn.Module] = None
        self.device: Optional[str] = None
        self.use_amp: bool = False
    
    def setup_for_gpu(self, model: nn.Module, device: str):
        """Setup client để train trên GPU cụ thể."""
        self.model = model
        self.device = device
        self.use_amp = ("cuda" in device)
    
    def _amp_ctx(self):
        return (
            torch_autocast(device_type="cuda", dtype=torch.float16)
            if self.use_amp else contextlib.nullcontext()
        )
    
    def _create_batches(self, batch_size: int):
        """Tạo batches và move lên GPU khi cần."""
        indices = torch.randperm(self.num_samples)
        for i in range(0, self.num_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = self.X_train[batch_idx].to(self.device, non_blocking=True)
            y_batch = self.y_train[batch_idx].to(self.device, non_blocking=True)
            yield X_batch, y_batch
    
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
        Train using the provided trainer strategy.
        
        Args:
            trainer: Training strategy (FedAvgTrainer, FedProxTrainer, etc.)
            epochs: Number of local epochs
            batch_size: Batch size
            lr: Learning rate
            global_params: Global model parameters (for regularization)
            **kwargs: Additional trainer-specific parameters
            
        Returns:
            Dict with client_id, num_samples, loss, and params
        """
        self.model.train()
        
        # Get optimizer from trainer
        optimizer_cls = trainer.get_optimizer_class()
        optimizer = optimizer_cls(self.model.parameters(), lr=lr)
        scaler = GradScaler(enabled=self.use_amp)
        
        # Pre-train hook
        trainer.pre_train(self.model, global_params, lr=lr, **kwargs)
        
        total_loss = 0.0
        total_samples = 0
        
        for ep in range(epochs):
            for X_batch, y_batch in self._create_batches(batch_size):
                optimizer.zero_grad()
                
                with self._amp_ctx():
                    out = self.model(X_batch)
                    loss = trainer.compute_loss(
                        self.model, out, y_batch, global_params, **kwargs
                    )
                
                if self.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                # Post-step hook (e.g., Fed+ correction)
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
            )
        }
