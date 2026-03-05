"""
Incremental Learning Strategies.

Available strategies:
- CGoFed: Constrained Gradient Optimization for Federated Class Incremental Learning
- FedCBDR: Class-wise Balancing Data Replay for FCIL
- DER: Dynamically Expandable Representation for FCIL
- NICE: Neurogenesis Inspired Contextual Encoding (Replay-free)
- GLFC: Global-Local Forgetting Compensation for FCIL
- Re-Fed: Retrieval-Enhanced Federated Incremental Learning
"""

from .cgofed import CGoFedTrainer, CGoFedAggregator
from .fedcbdr import FedCBDRTrainer, FedCBDRAggregator
from .der import DERTrainer, DERAggregator
from .nice import NICETrainer, NICEAggregator
from .glfc import GLFCTrainer, GLFCAggregator
from .refed import ReFedTrainer, ReFedAggregator

__all__ = [
    "CGoFedTrainer",
    "CGoFedAggregator",
    "FedCBDRTrainer",
    "FedCBDRAggregator",
    "DERTrainer",
    "DERAggregator",
    "NICETrainer",
    "NICEAggregator",
    "GLFCTrainer",
    "GLFCAggregator",
    "ReFedTrainer",
    "ReFedAggregator",
]
