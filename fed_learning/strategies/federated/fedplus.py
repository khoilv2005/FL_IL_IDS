"""
Fed+ Strategy - Federated Learning with Dynamic Regularization.

Reference:
    Yu et al., "Fed+: A Unified Approach to Federated Learning with 
    Heterogeneous Labels", 2021
"""

from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ...core import BaseTrainer, BaseAggregator


class FedPlusTrainer(BaseTrainer):
    """
    Fed+ local training - applies correction step after each update.
    
    Correction:
        w = θ * w_local + (1 - θ) * w_global
        where θ = 1 / (1 + η * μ)
    
    Note: This trainer is thread-safe for multi-GPU training.
    """
    
    def __init__(self, mu: float = 0.01):
        self.mu = mu
        self._lr: float = 0.001
    
    def pre_train(
        self,
        model: nn.Module,
        global_params: Optional[OrderedDict] = None,
        lr: float = 0.001,
        **kwargs
    ) -> None:
        """Store learning rate for theta computation."""
        self._lr = lr
    
    def post_step(
        self,
        model: nn.Module,
        global_params: Optional[OrderedDict] = None,
        **kwargs
    ) -> None:
        """
        Apply Fed+ correction after optimizer step.
        
        Thread-safe: moves global_params to the correct device on-demand.
        """
        if global_params is None:
            return
            
        # Compute theta
        theta = 1.0 / (1.0 + self._lr * self.mu)
        
        # Get device from model
        device = next(model.parameters()).device
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in global_params:
                    # Move global param to same device as model param
                    global_p = global_params[name].to(device)
                    param.data = theta * param.data + (1.0 - theta) * global_p
    
    def get_optimizer_class(self) -> type:
        """Fed+ uses SGD instead of Adam."""
        return torch.optim.SGD


class FedPlusAggregator(BaseAggregator):
    """
    Fed+ aggregation - same as FedAvg (weighted average).
    
    The correction is client-side, aggregation is standard.
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
