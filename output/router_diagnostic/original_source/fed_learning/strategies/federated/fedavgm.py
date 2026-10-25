"""
FedAvgM Strategy - FedAvg with Server Momentum.

Reference:
    Hsu et al., "Measuring the Effects of Non-Identical Data Distribution 
    for Federated Visual Classification", 2019
"""

from collections import OrderedDict
from typing import Dict, List, Optional

import torch

from ...core import BaseTrainer, BaseAggregator


class FedAvgMTrainer(BaseTrainer):
    """
    FedAvgM local training - same as FedAvg.
    
    The momentum is applied server-side, not client-side.
    """
    pass


class FedAvgMAggregator(BaseAggregator):
    """
    FedAvgM aggregation - weighted average + server momentum.
    
    Formula:
        delta_t = w_avg - w_t
        v_{t+1} = β * v_t + delta_t
        w_{t+1} = w_t + η * v_{t+1}
    """
    
    def __init__(self, momentum: float = 0.9, server_lr: float = 1.0):
        self.momentum = momentum
        self.server_lr = server_lr
        self.velocity: Optional[OrderedDict] = None
    
    def aggregate(
        self, 
        results: List[Dict], 
        global_params: Optional[OrderedDict] = None,
        **kwargs
    ) -> OrderedDict:
        if global_params is None:
            raise ValueError("FedAvgM requires global_params for momentum")
        
        # Step 1: Weighted average
        w_avg = self._weighted_average(results)
        
        # Step 2: Initialize velocity
        if self.velocity is None:
            self.velocity = OrderedDict(
                (k, torch.zeros_like(v)) for k, v in global_params.items()
            )
        
        # Step 3: Apply momentum
        new_params = OrderedDict()
        for k in global_params.keys():
            if global_params[k].dtype.is_floating_point:
                delta = w_avg[k] - global_params[k]
                new_v = self.momentum * self.velocity[k] + delta
                self.velocity[k] = new_v
                new_params[k] = global_params[k] + self.server_lr * new_v
            else:
                new_params[k] = w_avg[k]
        
        return new_params
    
    def reset(self) -> None:
        self.velocity = None
