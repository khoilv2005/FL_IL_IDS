"""Training module - utilities for running federated learning."""
from .runner import train_federated_multigpu
from .worker import train_clients_on_gpu

__all__ = ["train_federated_multigpu", "train_clients_on_gpu"]
