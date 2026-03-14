"""Standalone GLFC trainer for non-federated incremental learning."""

from ..fed_incremental.glfc import GLFCTrainer, get_one_hot, compute_entropy


__all__ = ["GLFCTrainer", "get_one_hot", "compute_entropy"]
