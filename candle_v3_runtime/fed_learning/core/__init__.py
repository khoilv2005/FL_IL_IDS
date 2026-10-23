"""
Core Module - Base abstractions for learning strategies.

This module provides the foundation for implementing different learning 
algorithms (Federated, Incremental, Decentralized).

Classes:
    BaseTrainer: Abstract base for local training strategies
    BaseAggregator: Abstract base for model aggregation strategies
"""

from .trainer import BaseTrainer
from .aggregator import BaseAggregator

__all__ = [
    "BaseTrainer",
    "BaseAggregator",
]
