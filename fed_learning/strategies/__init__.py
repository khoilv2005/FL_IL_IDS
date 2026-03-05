"""
Strategies Module - Learning algorithm implementations.

This module provides pluggable learning strategies for different paradigms:
- Federated Learning (FedAvg, FedProx, FedAvgM, Fed+)
- Incremental Learning (CGoFed, FedCBDR, EWC variants, LwF variants)

Usage:
    from strategies import get_strategy

    trainer, aggregator = get_strategy("fedprox", mu_fedprox=0.5)
"""

from typing import Tuple, Dict, Any

from ..core import BaseTrainer, BaseAggregator

# Import federated strategies
from .federated import (
    FedAvgTrainer,
    FedAvgAggregator,
    FedAvgMTrainer,
    FedAvgMAggregator,
    FedProxTrainer,
    FedProxAggregator,
    FedPlusTrainer,
    FedPlusAggregator,
)

# Import incremental learning strategies
from .incremental import (
    CGoFedTrainer,
    CGoFedAggregator,
    FedCBDRTrainer,
    FedCBDRAggregator,
    DERTrainer,
    DERAggregator,
    NICETrainer,
    NICEAggregator,
    GLFCTrainer,
    GLFCAggregator,
    ReFedTrainer,
    ReFedAggregator,
)
from .incremental.ewc import (
    EWCMixin,
    FedAvgEWCTrainer,
    FedProxEWCTrainer,
)
from .incremental.fedlwf import (
    FedLwFTrainer,
    FedLwFAggregator,
    FedLwFWithProximalTrainer,
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
    "fedcbdr": {
        "trainer": FedCBDRTrainer,
        "aggregator": FedCBDRAggregator,
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
    # DER (Dynamically Expandable Representation)
    "der": {
        "trainer": DERTrainer,
        "aggregator": DERAggregator,
    },
    # NICE (Neurogenesis Inspired Contextual Encoding) - Replay-free
    "nice": {
        "trainer": NICETrainer,
        "aggregator": NICEAggregator,
    },
    # GLFC (Global-Local Forgetting Compensation) - CVPR 2022
    "glfc": {
        "trainer": GLFCTrainer,
        "aggregator": GLFCAggregator,
    },
    # Re-Fed (Retrieval-Enhanced Federated Incremental Learning) - CVPR 2024
    "refed": {
        "trainer": ReFedTrainer,
        "aggregator": ReFedAggregator,
    },
}


def get_strategy(algorithm: str, **config) -> Tuple[BaseTrainer, BaseAggregator]:
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
            - "der": DER - Dynamically Expandable Representation
            - "nice": NICE - Neurogenesis Inspired Contextual Encoding (Replay-free)
            - "glfc": GLFC - Global-Local Forgetting Compensation (CVPR 2022)
            - "refed": Re-Fed - Retrieval-Enhanced Federated Incremental Learning (CVPR 2024)
        **config: Algorithm-specific configuration

    Returns:
        Tuple of (trainer, aggregator) instances

    Raises:
        ValueError: If algorithm is not recognized
    """
    algo_lower = algorithm.lower()

    if algo_lower not in STRATEGIES:
        available = ", ".join(STRATEGIES.keys())
        raise ValueError(f"Unknown algorithm: '{algorithm}'. Available: {available}")

    strategy = STRATEGIES[algo_lower]

    # Create trainer
    if algo_lower in ("fedprox", "fedplus"):
        trainer = strategy["trainer"](
            mu=config.get("mu_fedprox", config.get("mu", 0.01))
        )
    elif algo_lower == "cgofed":
        trainer = strategy["trainer"](
            mu=0.0,  # Paper Eq. 14: NO proximal term in CGoFed
            mu_projection=config.get("mu_cgofed", 1.0),
            lambda_decay=config.get("lambda_decay", 0.8),
            theta_threshold=config.get("theta_threshold", 0.35),
            energy_threshold=config.get("energy_threshold", 0.99),
            num_samples_rep=config.get("num_samples_rep", 2000),
            lambda_cross_task=config.get(
                "lambda_cross_task", config.get("cross_task_weight", 0.08)
            ),
        )
    elif algo_lower in ("fedavg_ewc", "fedprox_ewc"):
        trainer = strategy["trainer"](
            ewc_lambda=config.get("ewc_lambda", 10.0),
            fisher_samples=config.get("fisher_samples", 200),
            online_ewc=config.get("online_ewc", False),
            mu=config.get("mu_fedprox", config.get("mu", 0.01)),  # For FedProx base
        )
    elif algo_lower in ("fedavg_lwf", "fedprox_lwf"):
        trainer = strategy["trainer"](
            lwf_alpha=config.get("lwf_alpha", 1.0),
            temperature=config.get("temperature", 2.0),
            distill_old_classes_only=config.get("distill_old_classes_only", False),
            mu=config.get("mu_fedprox", config.get("mu", 0.01)),  # For FedProx base
        )
    elif algo_lower == "fedcbdr":
        trainer = strategy["trainer"](
            tau_old=config.get("tau_old", 0.9),
            tau_new=config.get("tau_new", 1.1),
            omega_old=config.get("omega_old", 1.1),
            omega_new=config.get("omega_new", 0.9),
        )
    elif algo_lower == "der":
        trainer = strategy["trainer"](
            lambda_aux=config.get("lambda_aux", 1.0),
            lambda_sparsity=config.get("lambda_sparsity", 0.5),
            s_max=config.get("s_max", 15.0),
            temperature=config.get("der_temperature", 2.0),
            buffer_size=config.get("buffer_size", 500),
        )
    elif algo_lower == "nice":
        trainer = strategy["trainer"](
            tau=config.get("tau", 0.95),
            max_phases=config.get("nice_max_phases", 5),
            phase_epochs=config.get("nice_phase_epochs", 5),
            memo_per_class=config.get("memo_per_class", 50),
        )
    elif algo_lower == "glfc":
        trainer = strategy["trainer"](
            memory_size=config.get("glfc_memory_size", 2000),
            entropy_threshold=config.get("glfc_entropy_threshold", 1.2),
            distill_weight=config.get("glfc_distill_weight", 0.5),
        )
    elif algo_lower == "refed":
        trainer = strategy["trainer"](
            memory_size=config.get("refed_memory_size", 2000),
            lambda_pim=config.get("refed_lambda_pim", 0.5),
            pim_iterations=config.get("refed_pim_iterations", 5),
        )
    else:
        trainer = strategy["trainer"]()

    # Create aggregator
    if algo_lower == "fedavgm":
        aggregator = strategy["aggregator"](
            momentum=config.get("server_momentum", 0.9),
            server_lr=config.get("server_lr", 1.0),
        )
    elif algo_lower in ("fedprox", "fedplus", "fedprox_ewc"):
        aggregator = strategy["aggregator"](
            mu=config.get("mu_fedprox", config.get("mu", 0.01))
        )
    elif algo_lower == "cgofed":
        aggregator = strategy["aggregator"](
            cross_task_weight=config.get("cross_task_weight", 0.3),
            top_k=config.get("top_k", 2),
            rounds_per_task=config.get("rounds_per_task", 5),
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
        "fedcbdr": "FedCBDR - Class-Balancing Data Replay with temperature scaling",
        "fedavg_ewc": "FedAvg + EWC - Elastic Weight Consolidation on FedAvg",
        "fedprox_ewc": "FedProx + EWC - Elastic Weight Consolidation on FedProx",
        "fedavg_lwf": "FedAvg + LwF - Learning without Forgetting on FedAvg",
        "fedprox_lwf": "FedProx + LwF - Learning without Forgetting on FedProx",
        "der": "DER - Dynamically Expandable Representation for FCIL",
        "nice": "NICE - Neurogenesis Inspired Contextual Encoding (Replay-free)",
        "glfc": "GLFC - Global-Local Forgetting Compensation (CVPR 2022)",
        "refed": "Re-Fed - Retrieval-Enhanced Federated Incremental Learning (CVPR 2024)",
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
