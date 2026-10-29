"""
Federated Learning with Multi-GPU Support
==========================================
Modular package for federated learning training.

Package Structure:
    core/           - Base abstractions (BaseTrainer, BaseAggregator)
    strategies/     - Learning algorithms (FedAvg, FedProx, CGoFed, DER, etc.)
        federated/      - Standard FL strategies
        incremental/    - Class-incremental learning strategies
    clients/        - Federated client implementations
    servers/        - Federated server implementations
    models/         - Neural network model definitions
    data/           - Data loading utilities
    training/       - GPU workers and training orchestration
    factories/      - Client and server creation factories
    visualization/  - Plotting and metrics (IEEE style)
    utils/          - Common utilities (seed, cleanup)
    plexus/         - Plexus decentralized FL (no server)

Note: To avoid circular imports, some modules must be imported directly:
    from fed_learning.training.task_loop import run_incremental_training
    from fed_learning.factories import create_clients, create_server
"""

from .models.cnn_gru import CNN_GRU_Model
from .data.loader import load_all_client_data_to_ram
from .clients.client import FederatedClient
from .servers.server import FederatedServer
from .training.runner import train_federated_multigpu
from .training.worker import train_clients_on_gpu
from .strategies import get_strategy, get_trainer, get_aggregator, list_strategies
from .core import BaseTrainer, BaseAggregator
from .utils import set_seed, cleanup_temp_folders
from .plexus import (
    PlexusSampler,
    PlexusNode,
    PlexusOrchestrator,
    PlexusAggregator,
    NodeWrapper,
    run_plexus_training,
)

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
    # Utils
    "set_seed",
    "cleanup_temp_folders",
    # Plexus (Decentralized FL)
    "PlexusSampler",
    "PlexusNode",
    "PlexusOrchestrator",
    "PlexusAggregator",
    "NodeWrapper",
    "run_plexus_training",
]
