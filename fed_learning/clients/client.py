"""
Federated Client with Multi-GPU Support
"""

import contextlib
from collections import OrderedDict
from typing import Optional, Callable, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from torch.amp import autocast as torch_autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast as torch_autocast, GradScaler


class FederatedClientMultiGPU:
    """
    Client hỗ trợ Multi-GPU:
    - Data giữ trên CPU
    - Khi train, load batch lên GPU được chỉ định
    """
    
    def __init__(self, client_id: int, X_train: torch.Tensor, y_train: torch.Tensor):
        self.client_id = client_id
        self.X_train = X_train  # CPU tensor
        self.y_train = y_train  # CPU tensor
        self.num_samples = len(y_train)
        
        # Model sẽ được set sau khi assign GPU
        self.model = None
        self.device = None
    
    def setup_for_gpu(self, model: nn.Module, device: str):
        """Setup client để train trên GPU cụ thể"""
        self.model = model
        self.device = device
        self.use_amp = ("cuda" in device)
    
    def _amp_ctx(self):
        return (
            torch_autocast(device_type="cuda", dtype=torch.float16)
            if self.use_amp else contextlib.nullcontext()
        )
    
    def _create_batches(self, batch_size: int):
        """Tạo batches và move lên GPU khi cần"""
        indices = torch.randperm(self.num_samples)
        for i in range(0, self.num_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            # Load batch lên GPU
            X_batch = self.X_train[batch_idx].to(self.device, non_blocking=True)
            y_batch = self.y_train[batch_idx].to(self.device, non_blocking=True)
            yield X_batch, y_batch
    
    def _train_base(
        self,
        epochs: int,
        batch_size: int,
        lr: float,
        optimizer_cls: type = optim.Adam,
        loss_modifier: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        post_step_hook: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """
        Base training loop - tái sử dụng cho tất cả algorithms.
        
        Args:
            epochs: Số epoch training
            batch_size: Batch size
            lr: Learning rate
            optimizer_cls: Class optimizer (Adam, SGD, etc.)
            loss_modifier: Hàm modify loss (VD: thêm proximal term cho FedProx)
            post_step_hook: Hàm chạy sau mỗi optimizer step (VD: Fed+ correction)
        
        Returns:
            Dict với client_id, num_samples, loss, params
        """
        self.model.train()
        optimizer = optimizer_cls(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler(enabled=self.use_amp)
        
        total_loss = 0.0
        total_samples = 0
        
        for ep in range(epochs):
            for X_batch, y_batch in self._create_batches(batch_size):
                optimizer.zero_grad()
                
                with self._amp_ctx():
                    out = self.model(X_batch)
                    loss = criterion(out, y_batch)
                    
                    # Apply loss modifier (e.g., proximal term for FedProx)
                    if loss_modifier is not None:
                        loss = loss_modifier(loss)
                
                if self.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                # Apply post-step hook (e.g., Fed+ correction)
                if post_step_hook is not None:
                    post_step_hook()
                
                bs = len(y_batch)
                total_loss += loss.item() * bs
                total_samples += bs
        
        return {
            "client_id": self.client_id,
            "num_samples": self.num_samples,
            "loss": total_loss / max(1, total_samples),
            "params": OrderedDict((k, v.cpu().clone()) for k, v in self.model.state_dict().items())
        }
    
    def train_fedavg(self, epochs: int, batch_size: int, lr: float) -> Dict[str, Any]:
        """Train với FedAvg - standard local training"""
        return self._train_base(
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            optimizer_cls=optim.Adam,
        )
    
    def train_fedprox(
        self, 
        epochs: int, 
        batch_size: int, 
        global_params: OrderedDict,
        mu: float, 
        lr: float
    ) -> Dict[str, Any]:
        """Train với FedProx - thêm proximal term"""
        # Pre-move global params to device
        global_params_device = {k: v.to(self.device) for k, v in global_params.items()}
        
        def loss_modifier(ce_loss: torch.Tensor) -> torch.Tensor:
            """Thêm proximal term: mu/2 * ||w - w_global||^2"""
            prox = 0.0
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in global_params_device:
                    prox += torch.sum((param - global_params_device[name])**2)
            return ce_loss + (mu / 2.0) * prox
        
        return self._train_base(
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            optimizer_cls=optim.Adam,
            loss_modifier=loss_modifier,
        )
    
    def train_fedplus(
        self, 
        epochs: int, 
        batch_size: int, 
        global_params: OrderedDict,
        mu: float, 
        lr: float
    ) -> Dict[str, Any]:
        """Train với Fed+ - thêm correction step sau mỗi update"""
        # Pre-move global params
        global_params_device = {k: v.to(self.device) for k, v in global_params.items()}
        
        # Theta for Fed+
        theta = 1.0 / (1.0 + lr * mu)
        
        def post_step_hook():
            """Fed+ correction: w = theta * w + (1-theta) * w_global"""
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in global_params_device:
                        param.data = theta * param.data + (1.0 - theta) * global_params_device[name]
        
        return self._train_base(
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            optimizer_cls=optim.SGD,  # Fed+ uses SGD
            post_step_hook=post_step_hook,
        )
