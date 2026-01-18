"""
Incremental Learning Strategies.

Available strategies:
- CGoFed: Constrained Gradient Optimization for Federated Class Incremental Learning
"""

from .cgofed import CGoFedTrainer, CGoFedAggregator

__all__ = [
    "CGoFedTrainer",
    "CGoFedAggregator",
]
