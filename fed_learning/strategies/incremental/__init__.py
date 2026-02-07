"""
Incremental Learning Strategies.

Available strategies:
- CGoFed: Constrained Gradient Optimization for Federated Class Incremental Learning
- FedCBDR: Class-wise Balancing Data Replay for FCIL
"""

from .cgofed import CGoFedTrainer, CGoFedAggregator
from .fedcbdr import FedCBDRTrainer, FedCBDRAggregator

__all__ = [
    "CGoFedTrainer",
    "CGoFedAggregator",
    "FedCBDRTrainer",
    "FedCBDRAggregator",
]
