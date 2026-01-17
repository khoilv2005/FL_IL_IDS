"""
Strategies Module - Learning algorithm implementations.

This module provides pluggable learning strategies for different paradigms:
- Federated Learning (FedAvg, FedProx, FedAvgM, Fed+)
- Incremental Learning (future: EWC, LwF)
- Decentralized Learning (future: Gossip, DPSGD)

Usage:
    from strategies import get_strategy
    
    trainer, aggregator = get_strategy("fedprox", mu=0.01)
"""

from typing import Tuple, Dict, Any

from ..core import BaseTrainer, BaseAggregator

# Import federated strategies
from .federated import (
    FedAvgTrainer, FedAvgAggregator,
    FedAvgMTrainer, FedAvgMAggregator,
    FedProxTrainer, FedProxAggregator,
    FedPlusTrainer, FedPlusAggregator,
)

# Registry of available strategies
STRATEGIES: Dict[str, Dict[str, type]] = {
    "fedavg": {
        "trainer": FedAvgTrainer,
        "aggregator": FedAvgAggregator,
    },
    "fedavgm": {
        "trainer": FedAvgMTrainer,
        "aggregator": FedAvgMAggregator,
    },
    "fedprox": {
        "trainer": FedProxTrainer,
        "aggregator": FedProxAggregator,
    },
    "fedplus": {
        "trainer": FedPlusTrainer,
        "aggregator": FedPlusAggregator,
    },
}


def get_strategy(
    algorithm: str, 
    **config
) -> Tuple[BaseTrainer, BaseAggregator]:
    """
    Factory function to get trainer and aggregator for an algorithm.
    
    Args:
        algorithm: Algorithm name (case-insensitive)
            - "fedavg": Federated Averaging
            - "fedavgm": FedAvg with Server Momentum
            - "fedprox": Federated Proximal
            - "fedplus": Fed+ with Dynamic Regularization
        **config: Algorithm-specific configuration:
            - mu: For FedProx/Fed+ (default: 0.01)
            - server_momentum: For FedAvgM (default: 0.9)
            - server_lr: For FedAvgM (default: 1.0)
    
    Returns:
        Tuple of (trainer, aggregator) instances
        
    Raises:
        ValueError: If algorithm is not recognized
        
    Example:
        >>> trainer, aggregator = get_strategy("fedprox", mu=0.01)
        >>> loss = trainer.compute_loss(model, output, target, global_params)
        >>> new_params = aggregator.aggregate(results, global_params)
    """
    algo_lower = algorithm.lower()
    
    if algo_lower not in STRATEGIES:
        available = ", ".join(STRATEGIES.keys())
        raise ValueError(
            f"Unknown algorithm: '{algorithm}'. Available: {available}"
        )
    
    strategy = STRATEGIES[algo_lower]
    
    # Create trainer
    if algo_lower in ("fedprox", "fedplus"):
        trainer = strategy["trainer"](mu=config.get("mu", 0.01))
    else:
        trainer = strategy["trainer"]()
    
    # Create aggregator
    if algo_lower == "fedavgm":
        aggregator = strategy["aggregator"](
            momentum=config.get("server_momentum", 0.9),
            server_lr=config.get("server_lr", 1.0)
        )
    elif algo_lower in ("fedprox", "fedplus"):
        aggregator = strategy["aggregator"](mu=config.get("mu", 0.01))
    else:
        aggregator = strategy["aggregator"]()
    
    return trainer, aggregator


def get_trainer(algorithm: str, **config) -> BaseTrainer:
    """Get only the trainer for an algorithm."""
    trainer, _ = get_strategy(algorithm, **config)
    return trainer


def get_aggregator(algorithm: str, **config) -> BaseAggregator:
    """Get only the aggregator for an algorithm."""
    _, aggregator = get_strategy(algorithm, **config)
    return aggregator


def list_strategies() -> Dict[str, str]:
    """List all available strategies with descriptions."""
    return {
        "fedavg": "Federated Averaging - weighted average by sample count",
        "fedavgm": "FedAvg + Server Momentum - accelerated convergence",
        "fedprox": "Federated Proximal - handles heterogeneity with proximal term",
        "fedplus": "Fed+ - dynamic regularization for heterogeneous data",
    }


__all__ = [
    "get_strategy",
    "get_trainer",
    "get_aggregator",
    "list_strategies",
    "STRATEGIES",
    "BaseTrainer",
    "BaseAggregator",
]
