"""
Incremental Learning Strategies.

Available strategies:
- CGoFed: Constrained Gradient Optimization for Federated Class Incremental Learning
- FedCBDR: Class-wise Balancing Data Replay for FCIL
- DER: Dynamically Expandable Representation for FCIL
- NICE: Neurogenesis Inspired Contextual Encoding (Replay-free)
"""

from .cgofed import CGoFedTrainer, CGoFedAggregator
from .fedcbdr import FedCBDRTrainer, FedCBDRAggregator
from .der import DERTrainer, DERAggregator
from .nice import NICETrainer, NICEAggregator

__all__ = [
    "CGoFedTrainer",
    "CGoFedAggregator",
    "FedCBDRTrainer",
    "FedCBDRAggregator",
    "DERTrainer",
    "DERAggregator",
    "NICETrainer",
    "NICEAggregator",
]
