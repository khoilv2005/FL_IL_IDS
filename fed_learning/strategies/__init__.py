"""
Strategies Module - Learning algorithm implementations.

This module provides pluggable learning strategies for different paradigms:
- Federated Learning (FedAvg, FedProx, FedAvgM, Fed+)
- Incremental Learning (CGoFed)
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

# Import incremental learning strategies
from .incremental import (
    CGoFedTrainer, CGoFedAggregator,
)
from .incremental.ewc import (
    EWCMixin, FedAvgEWCTrainer, FedProxEWCTrainer,
)
from .incremental.fedlwf import (
    FedLwFTrainer, FedLwFAggregator, FedLwFWithProximalTrainer
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
    # Incremental Learning
    "cgofed": {
        "trainer": CGoFedTrainer,
        "aggregator": CGoFedAggregator,
    },
    # EWC-based (FedAvg + EWC, FedProx + EWC)
    "fedavg_ewc": {
        "trainer": FedAvgEWCTrainer,
        "aggregator": FedAvgAggregator,
    },
    "fedprox_ewc": {
        "trainer": FedProxEWCTrainer,
        "aggregator": FedProxAggregator,
    },
    # LwF-based (FedAvg + LwF, FedProx + LwF)
    "fedavg_lwf": {
        "trainer": FedLwFTrainer,
        "aggregator": FedLwFAggregator,
    },
    "fedprox_lwf": {
        "trainer": FedLwFWithProximalTrainer,
        "aggregator": FedLwFAggregator,  # Uses same aggregator as FedAvg/LwF
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
            - "cgofed": Constrained Gradient for Class Incremental Learning
            - "fedavg_ewc": FedAvg + EWC
            - "fedprox_ewc": FedProx + EWC
            - "fedavg_lwf": FedAvg + LwF
            - "fedprox_lwf": FedProx + LwF
        **config: Algorithm-specific configuration
    
    Returns:
        Tuple of (trainer, aggregator) instances
        
    Raises:
        ValueError: If algorithm is not recognized
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
    elif algo_lower == "cgofed":
        trainer = strategy["trainer"](
            mu=config.get("mu", 0.01),
            lambda_decay=config.get("lambda_decay", 0.1),
            theta_threshold=config.get("theta_threshold", 0.1),
            energy_threshold=config.get("energy_threshold", 0.95),
            num_samples_rep=config.get("num_samples_rep", 100),
        )
    elif algo_lower in ("fedavg_ewc", "fedprox_ewc"):
        trainer = strategy["trainer"](
            ewc_lambda=config.get("ewc_lambda", 1000.0),
            fisher_samples=config.get("fisher_samples", 200),
            online_ewc=config.get("online_ewc", False),
            mu=config.get("mu", 0.01),  # For FedProx base
        )
    elif algo_lower in ("fedavg_lwf", "fedprox_lwf"):
        trainer = strategy["trainer"](
            lwf_alpha=config.get("lwf_alpha", 1.0),
            temperature=config.get("temperature", 2.0),
            distill_on_new_only=config.get("distill_on_new_only", False),
            mu=config.get("mu", 0.01),  # For FedProx base
        )
    else:
        trainer = strategy["trainer"]()
    
    # Create aggregator
    if algo_lower == "fedavgm":
        aggregator = strategy["aggregator"](
            momentum=config.get("server_momentum", 0.9),
            server_lr=config.get("server_lr", 1.0)
        )
    elif algo_lower in ("fedprox", "fedplus", "fedprox_ewc", "fedprox_lwf"):
        aggregator = strategy["aggregator"](mu=config.get("mu", 0.01))
    elif algo_lower == "cgofed":
        aggregator = strategy["aggregator"](
            cross_task_weight=config.get("cross_task_weight", 0.3),
            top_k=config.get("top_k", 2)
        )
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
        "cgofed": "CGoFed - Constrained Gradient for Class Incremental Learning",
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
