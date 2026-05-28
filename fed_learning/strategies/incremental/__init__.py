"""
Core incremental learning strategies.

Package này là phần cốt lõi của các thuật toán IL.
Các wrapper dành cho federated learning nằm ở
`fed_learning.strategies.fed_incremental` và sẽ tái sử dụng logic từ đây.
"""

from .ewc import EWCTrainer
from .lwf import LwFTrainer
from .der import DERTrainer
from .rne import RNETrainer
from .nice import NICETrainer


INCREMENTAL_STRATEGIES = {
    "ewc": EWCTrainer,
    "lwf": LwFTrainer,
    "der": DERTrainer,
    "rne": RNETrainer,
    "rne_compress": RNETrainer,
    "nice": NICETrainer,
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
            ewc_lambda=config.get("ewc_lambda", 1000.0),
            fisher_samples=config.get("fisher_samples", 200),
            online_ewc=config.get("online_ewc", False),
            debug_logging=config.get("debug_logging", False),
        )
    if algo == "lwf":
        return trainer_cls(
            lwf_alpha=config.get("lwf_alpha", 1.0),
            temperature=config.get("temperature", 2.0),
            distill_old_classes_only=config.get("distill_old_classes_only", False),
        )
    if algo in ("der", "rne", "rne_compress"):
        return trainer_cls(
            lambda_aux=config.get("lambda_aux", 1.0),
            lambda_sparsity=config.get(
                "lambda_sparsity",
                0.0 if algo in ("rne", "rne_compress") else 0.5,
            ),
            s_max=config.get("s_max", 15.0),
            temperature=config.get("der_temperature", config.get("rne_temperature", 2.0)),
            buffer_size=config.get("buffer_size", 500),
            **(
                {
                    "old_head_lr_scale": config.get("rne_old_head_lr_scale", 1.0),
                    "kd_weight": config.get(
                        "rne_kd_weight",
                        2.0 if algo == "rne_compress" else 1.0,
                    ),
                }
                if algo in ("rne", "rne_compress")
                else {}
            ),
        )
    if algo == "nice":
        return trainer_cls(
            tau=config.get("tau", 0.95),
            max_phases=config.get("nice_max_phases", 5),
            phase_epochs=config.get("nice_phase_epochs", 5),
            memo_per_class=config.get("memo_per_class", 50),
        )
    return trainer_cls()


def list_incremental_strategies():
    """Danh sách thuật toán incremental local đang hỗ trợ."""
    return {
        "ewc": "Elastic Weight Consolidation (local IL)",
        "lwf": "Learning without Forgetting (local IL)",
        "der": "Dynamically Expandable Representation (local IL)",
        "rne": "Recurrent Network Expansion (local IL)",
        "rne_compress": "RNE-compress recurrent network expansion (local IL)",
        "nice": "NICE replay-free incremental learning",
    }


__all__ = [
    "EWCTrainer",
    "LwFTrainer",
    "DERTrainer",
    "RNETrainer",
    "NICETrainer",
    "INCREMENTAL_STRATEGIES",
    "get_incremental_strategy",
    "list_incremental_strategies",
]
