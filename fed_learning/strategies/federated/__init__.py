"""
Federated Learning Strategies.

Available strategies:
- FedAvg: Federated Averaging
- FedAvgM: FedAvg with Server Momentum
- FedProx: Federated Proximal
- Fed+: Fed+ with Dynamic Regularization
"""

from .fedavg import FedAvgTrainer, FedAvgAggregator
from .fedavgm import FedAvgMTrainer, FedAvgMAggregator
from .fedprox import FedProxTrainer, FedProxAggregator
from .fedplus import FedPlusTrainer, FedPlusAggregator

__all__ = [
    "FedAvgTrainer", "FedAvgAggregator",
    "FedAvgMTrainer", "FedAvgMAggregator",
    "FedProxTrainer", "FedProxAggregator",
    "FedPlusTrainer", "FedPlusAggregator",
]
