"""
CGoFed Client - Specialized client for Class Incremental Learning.

Reference:
    "CGoFed: Constrained Gradient Optimization Strategy for Federated Class 
    Incremental Learning", IEEE TKDE, 2025

Extends FederatedClient with gradient-based representation computation
for cross-task similarity (paper Section 5.2, eq. 9).
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Optional, Dict, Any

from .client import FederatedClient
from ..core import BaseTrainer


class CGoFedClient(FederatedClient):
    """
    Client for CGoFed algorithm with gradient representation computation.
    
    Inherits all standard FL functionality from FederatedClient,
    adds compute_gradient_representation() for cross-task similarity.
    
    Paper Reference:
    - Representation R is computed as mean gradient vector (paper eq. 9)
    - Used by server to compute similarity between tasks
    - Enables personalized aggregation with historical models
    """
    
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
        Train and compute gradient representation for CGoFed.
        
        Extends parent train() by computing representation after training.
        This representation is sent to server for cross-task similarity.
        """
        # Standard training with gradient projection (via trainer.post_step)
        result = super().train(trainer, epochs, batch_size, lr, global_params, **kwargs)
        
        # Compute gradient representation for cross-task similarity (paper eq. 9)
        result["representation"] = self.compute_gradient_representation(
            model=self.model,
            num_samples=100,
            batch_size=32
        )
        
        return result
    
    def compute_gradient_representation(
        self,
        model: nn.Module,
        num_samples: int = 100,
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Compute mean gradient vector for cross-task similarity (paper eq. 9).
        
        Paper CGoFed Section 5.2:
        Each client computes a representation R (mean gradient vector) and 
        sends it to server. Server uses R to compute similarity between 
        current task and historical tasks.
        
        Args:
            model: The trained model
            num_samples: Number of samples to use for gradient computation
            batch_size: Batch size for gradient computation
            
        Returns:
            Mean gradient vector [d] where d = total model parameters
        """
        # Train mode required for GRU backward (cudnn requirement)
        model.train()
        device = next(model.parameters()).device
        
        all_grads = []
        sample_count = 0
        
        # Sample indices
        n_available = min(num_samples, self.num_samples)
        indices = torch.randperm(self.num_samples)[:n_available]
        
        # Collect per-sample gradients
        for i in range(0, len(indices), batch_size):
            if sample_count >= num_samples:
                break
            
            batch_idx = indices[i:i+batch_size]
            X_batch = self.X_train[batch_idx].to(device)
            y_batch = self.y_train[batch_idx].to(device)
            
            # Compute gradient for this batch
            model.zero_grad()
            output = model(X_batch)
            loss = torch.nn.functional.cross_entropy(output, y_batch)
            loss.backward()
            
            # Flatten all gradients into single vector
            grads = []
            for p in model.parameters():
                if p.grad is not None:
                    grads.append(p.grad.view(-1))
                else:
                    # For frozen params or BatchNorm edge cases
                    grads.append(torch.zeros(p.numel(), device=device))
            
            grad_vector = torch.cat(grads)
            all_grads.append(grad_vector.cpu())
            sample_count += len(batch_idx)
        
        # Return mean gradient vector (Reduced representation)
        if all_grads:
            mean_grad = torch.stack(all_grads).mean(dim=0)
            return mean_grad
        else:
            # Return zero vector if no gradients computed
            total_params = sum(p.numel() for p in model.parameters())
            return torch.zeros(total_params)
