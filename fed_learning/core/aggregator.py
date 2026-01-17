"""
Base Aggregator - Abstract base class for model aggregation strategies.
"""

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, List, Optional

import torch


class BaseAggregator(ABC):
    """
    Abstract base class for model aggregation strategies.
    
    Different aggregation algorithms (FedAvg, FedAvgM, etc.) extend this
    class to implement their specific aggregation logic.
    """
    
    @property
    def name(self) -> str:
        """Return the name of this aggregator."""
        return self.__class__.__name__.replace("Aggregator", "")
    
    @abstractmethod
    def aggregate(
        self, 
        results: List[Dict], 
        global_params: Optional[OrderedDict] = None,
        **kwargs
    ) -> OrderedDict:
        """
        Aggregate client model updates into new global model parameters.
        
        Args:
            results: List of dicts from clients, each containing:
                - 'params': OrderedDict of model parameters
                - 'num_samples': Number of training samples
                - 'loss': Training loss (optional)
            global_params: Current global model parameters
            **kwargs: Additional algorithm-specific parameters
            
        Returns:
            OrderedDict: Aggregated model parameters
        """
        pass
    
    def reset(self) -> None:
        """
        Reset any stateful components (e.g., momentum buffers).
        
        Override if the aggregator maintains state between rounds.
        """
        pass
    
    def _weighted_average(self, results: List[Dict]) -> OrderedDict:
        """
        Compute weighted average of model parameters.
        
        Weight is proportional to number of samples per client.
        This is the core computation used by FedAvg and related algorithms.
        
        Args:
            results: List of client results with 'params' and 'num_samples'
            
        Returns:
            Weighted average of parameters
        """
        total_samples = sum(r["num_samples"] for r in results)
        
        agg = None
        for r in results:
            w_i = r["num_samples"] / max(1, total_samples)
            params = r["params"]
            
            if agg is None:
                agg = OrderedDict(
                    (k, w_i * v.float()) for k, v in params.items()
                )
            else:
                for k in agg.keys():
                    if agg[k].dtype.is_floating_point:
                        agg[k] = agg[k] + w_i * params[k].float()
                    else:
                        # Non-floating point params (e.g., batch norm stats)
                        agg[k] = params[k]
        
        return agg
