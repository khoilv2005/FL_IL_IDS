"""
Federated Learning Strategies.

Available strategies:
- FedAvg: Federated Averaging
- FedAvgM: FedAvg with Server Momentum
- FedProx: Federated Proximal
- Fed+: Fed+ with Dynamic Regularization
- Plexus: Decentralized FL (EuroMLSys 2025)
- PlexusDER: Decentralized DER
- PlexusNICE: Decentralized NICE
"""

from .fedavg import FedAvgTrainer, FedAvgAggregator
from .fedavgm import FedAvgMTrainer, FedAvgMAggregator
from .fedprox import FedProxTrainer, FedProxAggregator
from .fedplus import FedPlusTrainer, FedPlusAggregator
from .plexus import PlexusTrainer, PlexusAggregator
from .plexus_der import PlexusDERTrainer, PlexusDERAggregator
from .plexus_nice import PlexusNICETrainer, PlexusNICEAggregator

__all__ = [
    "FedAvgTrainer", "FedAvgAggregator",
    "FedAvgMTrainer", "FedAvgMAggregator",
    "FedProxTrainer", "FedProxAggregator",
    "FedPlusTrainer", "FedPlusAggregator",
    "PlexusTrainer", "PlexusAggregator",
    "PlexusDERTrainer", "PlexusDERAggregator",
    "PlexusNICETrainer", "PlexusNICEAggregator",
]
