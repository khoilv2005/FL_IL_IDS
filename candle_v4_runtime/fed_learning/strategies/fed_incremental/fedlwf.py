"""
Federated wrappers for LwF-based incremental learning.
"""

from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ...core import BaseAggregator
from ..federated.fedprox import FedProxTrainer
from ..incremental.lwf import LwFTrainer


class FedLwFTrainer(LwFTrainer):
    """Compatibility wrapper giữ tên trainer cho federated LwF."""

    pass


class FedLwFAggregator(BaseAggregator):
    """
    FedLwF Aggregation - Standard FedAvg weighted average.

    FedLwF không đổi cách aggregate phía server.
    Điểm khác biệt chỉ nằm ở local loss của client.
    """

    def aggregate(
        self, results: List[Dict], global_params: Optional[OrderedDict] = None, **kwargs
    ) -> OrderedDict:
        """Aggregate theo weighted average giống FedAvg."""
        return self._weighted_average(results)


class FedLwFWithProximalTrainer(FedLwFTrainer):
    """
    FedLwF + Proximal regularization.

    Đây là biến thể kết hợp:
    - distillation của LwF
    - proximal term của FedProx
    để vừa chống quên vừa giảm lệch client trên dữ liệu non-IID.
    """

    def __init__(
        self,
        lwf_alpha: float = 1.0,
        temperature: float = 2.0,
        mu: float = 0.01,
        **kwargs,
    ):
        super().__init__(lwf_alpha=lwf_alpha, temperature=temperature, **kwargs)
        self.mu = mu

    def compute_loss(
        self,
        model: nn.Module,
        output: torch.Tensor,
        target: torch.Tensor,
        global_params: Optional[OrderedDict] = None,
        inputs: Optional[torch.Tensor] = None,
        old_model: Optional[nn.Module] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Tính loss của FedLwF + Proximal = LwF loss + proximal term."""
        # Get FedLwF loss (CE + KD)
        lwf_loss = super().compute_loss(
            model, output, target, global_params, inputs, old_model, **kwargs
        )

        # Add proximal term if global params provided
        if global_params is None:
            return lwf_loss

        prox_term = 0.0
        for name, param in model.named_parameters():
            if name in global_params:
                global_param = global_params[name].to(param.device)
                prox_term += ((param - global_param) ** 2).sum()

        return lwf_loss + (self.mu / 2) * prox_term
