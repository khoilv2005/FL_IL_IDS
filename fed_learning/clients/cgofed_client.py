"""
CGoFed Client - Specialized client for Class Incremental Learning.
Extends FederatedClient with representation matrix computation.
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Optional, Dict, Any

from .client import FederatedClient
from ..core import BaseTrainer


class CGoFedClient(FederatedClient):
    """
    Client for CGoFed algorithm with representation matrix computation.
    
    Inherits all standard FL functionality from FederatedClient,
    adds compute_representation_matrix() for cross-task similarity.
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
        Train and compute representation matrix for CGoFed.
        
        Extends parent train() by computing representation after training.
        """
        # Standard training
        result = super().train(trainer, epochs, batch_size, lr, global_params, **kwargs)
        
        # Compute representation matrix for cross-task similarity
        result["representation"] = self.compute_representation_matrix(
            model=self.model,
            num_samples=100,
            batch_size=32
        )
        
        return result
    
    def compute_representation_matrix(
        self,
        model: nn.Module,
        num_samples: int = 100,
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Compute mean gradient vector for CGoFed similarity computation.
        
        This computes the representation R (mean gradient) that will be sent 
        to server for cross-task similarity computation.
        
        Args:
            model: The trained model
            num_samples: Number of samples to use for gradient computation
            batch_size: Batch size for gradient computation
            
        Returns:
            Mean gradient vector [d] where d = total model parameters
        """
        model.eval()
        device = next(model.parameters()).device
        
        all_grads = []
        sample_count = 0
        
        # Collect gradients from forward passes
        indices = torch.randperm(min(num_samples, self.num_samples))
        for i in range(0, len(indices), batch_size):
            if sample_count >= num_samples:
                break
                
            batch_idx = indices[i:i+batch_size]
            X_batch = self.X_train[batch_idx].to(device)
            y_batch = self.y_train[batch_idx].to(device)
            
            model.zero_grad()
            output = model(X_batch)
            loss = torch.nn.functional.cross_entropy(output, y_batch)
            loss.backward()
            
            # Flatten and collect gradients
            grad_vector = torch.cat([
                p.grad.view(-1) for p in model.parameters() if p.grad is not None
            ])
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
