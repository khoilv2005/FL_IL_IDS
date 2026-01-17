"""
FedProx Strategy - Federated Proximal algorithm.

Reference:
    Li et al., "Federated Optimization in Heterogeneous Networks", 
    MLSys 2020
"""

from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ...core import BaseTrainer, BaseAggregator


class FedProxTrainer(BaseTrainer):
    """
    FedProx local training - adds proximal term to loss.
    
    Loss:
        L = L_ce + (μ/2) * ||w - w_global||²
    
    Note: This trainer is thread-safe for multi-GPU training.
    """
    
    def __init__(self, mu: float = 0.01):
        self.mu = mu
    
    def compute_loss(
        self, 
        model: nn.Module,
        output: torch.Tensor, 
        target: torch.Tensor,
        global_params: Optional[OrderedDict] = None,
        **kwargs
    ) -> torch.Tensor:
        # Base cross-entropy loss
        ce_loss = nn.CrossEntropyLoss()(output, target)
        
        # Proximal term - move global params to device on-demand
        if global_params is not None:
            device = next(model.parameters()).device
            prox = 0.0
            for name, param in model.named_parameters():
                if param.requires_grad and name in global_params:
                    # Move to same device as model
                    global_p = global_params[name].to(device)
                    prox += torch.sum((param - global_p) ** 2)
            return ce_loss + (self.mu / 2.0) * prox
        
        return ce_loss


class FedProxAggregator(BaseAggregator):
    """
    FedProx aggregation - same as FedAvg (weighted average).
    
    The proximal term is client-side, aggregation is standard.
    """
    
    def __init__(self, mu: float = 0.01):
        self.mu = mu  # Stored for reference
    
    def aggregate(
        self, 
        results: List[Dict], 
        global_params: Optional[OrderedDict] = None,
        **kwargs
    ) -> OrderedDict:
        return self._weighted_average(results)
