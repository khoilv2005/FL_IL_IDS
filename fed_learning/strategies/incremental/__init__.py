"""
Standalone incremental learning strategies.

Package này tách riêng các thuật toán incremental để dùng trong bối cảnh
single-model / non-federated training.

Khác với `fed_learning.strategies.fed_incremental`:
- chỉ export trainer/local-logic
- không export aggregator
- không mang ý nghĩa server/client federated trong API import
"""

from .ewc import EWCTrainer
from .lwf import LwFTrainer
from .der import DERTrainer
from .nice import NICETrainer
from .glfc import GLFCTrainer
from .refed import ReFedTrainer
from .cgofed import CGoFedTrainer
from .fedcbdr import CBDRTrainer


INCREMENTAL_STRATEGIES = {
    "ewc": EWCTrainer,
    "lwf": LwFTrainer,
    "der": DERTrainer,
    "nice": NICETrainer,
    "glfc": GLFCTrainer,
    "refed": ReFedTrainer,
    "cgofed": CGoFedTrainer,
    "cbdr": CBDRTrainer,
}


def get_incremental_strategy(algorithm: str, **config):
    """Tạo trainer cho chế độ incremental learning local."""
    algo = algorithm.lower()
    if algo not in INCREMENTAL_STRATEGIES:
        available = ", ".join(INCREMENTAL_STRATEGIES.keys())
        raise ValueError(
            f"Unknown standalone incremental algorithm: '{algorithm}'. Available: {available}"
        )

    trainer_cls = INCREMENTAL_STRATEGIES[algo]

    if algo == "ewc":
        return trainer_cls(
            ewc_lambda=config.get("ewc_lambda", 10.0),
            fisher_samples=config.get("fisher_samples", 200),
            online_ewc=config.get("online_ewc", False),
        )
    if algo == "lwf":
        return trainer_cls(
            lwf_alpha=config.get("lwf_alpha", 1.0),
            temperature=config.get("temperature", 2.0),
            distill_old_classes_only=config.get("distill_old_classes_only", False),
        )
    if algo == "cbdr":
        return trainer_cls(
            tau_old=config.get("tau_old", 0.9),
            tau_new=config.get("tau_new", 1.1),
            omega_old=config.get("omega_old", 1.1),
            omega_new=config.get("omega_new", 0.9),
        )
    if algo == "der":
        return trainer_cls(
            lambda_aux=config.get("lambda_aux", 1.0),
            lambda_sparsity=config.get("lambda_sparsity", 0.5),
            s_max=config.get("s_max", 15.0),
            temperature=config.get("der_temperature", 2.0),
            buffer_size=config.get("buffer_size", 500),
        )
    if algo == "nice":
        return trainer_cls(
            tau=config.get("tau", 0.95),
            max_phases=config.get("nice_max_phases", 5),
            phase_epochs=config.get("nice_phase_epochs", 5),
            memo_per_class=config.get("memo_per_class", 50),
        )
    if algo == "glfc":
        return trainer_cls(
            memory_size=config.get("glfc_memory_size", 2000),
            entropy_threshold=config.get("glfc_entropy_threshold", 1.2),
            distill_weight=config.get("glfc_distill_weight", 0.5),
        )
    if algo == "refed":
        return trainer_cls(
            memory_size=config.get("refed_memory_size", 2000),
            lambda_pim=config.get("refed_lambda_pim", 0.5),
            pim_iterations=config.get("refed_pim_iterations", 5),
        )
    if algo == "cgofed":
        return trainer_cls(
            mu=0.0,
            mu_projection=config.get("mu_cgofed", 1.0),
            lambda_decay=config.get("lambda_decay", 0.8),
            theta_threshold=config.get("theta_threshold", 0.35),
            energy_threshold=config.get("energy_threshold", 0.99),
            num_samples_rep=config.get("num_samples_rep", 2000),
            lambda_cross_task=config.get(
                "lambda_cross_task", config.get("cross_task_weight", 0.08)
            ),
        )

    return trainer_cls(memory_size=config.get("refed_memory_size", 2000))


def list_incremental_strategies():
    """Danh sách thuật toán incremental local đang hỗ trợ."""
    return {
        "ewc": "Elastic Weight Consolidation (local IL)",
        "lwf": "Learning without Forgetting (local IL)",
        "der": "Dynamically Expandable Representation (local IL)",
        "nice": "NICE replay-free incremental learning",
        "glfc": "Global-Local Forgetting Compensation (local IL)",
        "refed": "Retrieval-Enhanced replay incremental learning",
        "cgofed": "Constrained Gradient Optimization (local IL)",
        "cbdr": "Class-balanced data replay (local IL)",
    }


__all__ = [
    "EWCTrainer",
    "LwFTrainer",
    "DERTrainer",
    "NICETrainer",
    "GLFCTrainer",
    "ReFedTrainer",
    "CGoFedTrainer",
    "CBDRTrainer",
    "INCREMENTAL_STRATEGIES",
    "get_incremental_strategy",
    "list_incremental_strategies",
]
