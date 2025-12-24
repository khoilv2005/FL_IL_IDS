"""
Federated Learning with Multi-GPU Support
==========================================
Modular package for federated learning training.
"""

from .models.cnn_gru import CNN_GRU_Model
from .data.loader import load_all_client_data_to_ram
from .clients.client import FederatedClientMultiGPU
from .servers.server import FederatedServerMultiGPU
from .training.trainer import train_federated_multigpu
from .training.gpu_trainer import train_clients_on_gpu

__all__ = [
    "CNN_GRU_Model",
    "load_all_client_data_to_ram",
    "FederatedClientMultiGPU",
    "FederatedServerMultiGPU",
    "train_federated_multigpu",
    "train_clients_on_gpu",
]
