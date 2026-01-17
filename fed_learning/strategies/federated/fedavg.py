"""
FedAvg Strategy - Federated Averaging algorithm.

Reference:
    McMahan et al., "Communication-Efficient Learning of Deep Networks 
    from Decentralized Data", AISTATS 2017
"""

from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ...core import BaseTrainer, BaseAggregator


class FedAvgTrainer(BaseTrainer):
    """
    FedAvg local training - standard training without modifications.
    
    Uses CrossEntropyLoss and Adam optimizer by default.
    """
    
    # All methods use default implementations from BaseTrainer
    pass


class FedAvgAggregator(BaseAggregator):
    """
    FedAvg aggregation - weighted average by sample count.
    
    Formula:
        w_global = Σ (n_i / N) * w_i
    """
    
    def aggregate(
        self, 
        results: List[Dict], 
        global_params: Optional[OrderedDict] = None,
        **kwargs
    ) -> OrderedDict:
        return self._weighted_average(results)
