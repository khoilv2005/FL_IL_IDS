"""Training module"""
from .trainer import train_federated_multigpu
from .gpu_trainer import train_clients_on_gpu

__all__ = ["train_federated_multigpu", "train_clients_on_gpu"]
