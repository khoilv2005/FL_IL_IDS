"""
Federated Learning Strategies.

Available strategies:
- FedAvg: Federated Averaging
- FedAvgM: FedAvg with Server Momentum
- FedProx: Federated Proximal
- Fed+: Fed+ with Dynamic Regularization
- PlexusDER: Decentralized DER (with server)
- PlexusNICE: Decentralized NICE (with server)
"""

from .fedavg import FedAvgTrainer, FedAvgAggregator
from .fedavgm import FedAvgMTrainer, FedAvgMAggregator
from .fedprox import FedProxTrainer, FedProxAggregator
from .fedplus import FedPlusTrainer, FedPlusAggregator
from .plexus_der import PlexusDERTrainer, PlexusDERAggregator
from .plexus_nice import PlexusNICETrainer, PlexusNICEAggregator
from .dfca import DFCATrainer, DFCAAggregator

__all__ = [
    "FedAvgTrainer", "FedAvgAggregator",
    "FedAvgMTrainer", "FedAvgMAggregator",
    "FedProxTrainer", "FedProxAggregator",
    "FedPlusTrainer", "FedPlusAggregator",
    "PlexusDERTrainer", "PlexusDERAggregator",
    "PlexusNICETrainer", "PlexusNICEAggregator",
    "DFCATrainer", "DFCAAggregator",
]
