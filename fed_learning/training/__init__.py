"""Training module - utilities for running federated learning."""
from .runner import train_federated_multigpu
from .worker import train_clients_on_gpu
from .cgofed_worker import train_cgofed_clients_on_gpu

__all__ = ["train_federated_multigpu", "train_clients_on_gpu", "train_cgofed_clients_on_gpu"]
