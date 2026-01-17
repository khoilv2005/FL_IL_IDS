"""
Federated Learning with Multi-GPU Support
==========================================
Modular package for federated learning training.

Structure:
    core/           - Base abstractions (BaseTrainer, BaseAggregator)
    strategies/     - Learning algorithms (FedAvg, FedProx, etc.)
    clients/        - Federated client
    servers/        - Federated server
    models/         - Model definitions
    data/           - Data loading utilities
    training/       - Training utilities
"""

from .models.cnn_gru import CNN_GRU_Model
from .data.loader import load_all_client_data_to_ram
from .clients.client import FederatedClient
from .servers.server import FederatedServer
from .training.runner import train_federated_multigpu
from .training.worker import train_clients_on_gpu
from .strategies import get_strategy, get_trainer, get_aggregator, list_strategies
from .core import BaseTrainer, BaseAggregator

__all__ = [
    # Models
    "CNN_GRU_Model",
    # Data
    "load_all_client_data_to_ram",
    # Client/Server
    "FederatedClient",
    "FederatedServer",
    # Training
    "train_federated_multigpu",
    "train_clients_on_gpu",
    # Strategies
    "get_strategy",
    "get_trainer",
    "get_aggregator",
    "list_strategies",
    # Core
    "BaseTrainer",
    "BaseAggregator",
]
