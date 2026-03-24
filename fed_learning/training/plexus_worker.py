"""
Plexus Worker - Multi-GPU training worker for Plexus clients.

Reference:
    Dhasade et al., "Practical Federated Learning without a Server",
    EuroMLSys 2025

Standard BaseGPUWorker — Plexus does not modify local training, so this
worker is identical to the StandardWorker but typed for PlexusClient.
"""

from collections import OrderedDict
from typing import Dict, List

from ..clients.plexus_client import PlexusClient
from ..strategies.federated.plexus import PlexusTrainer
from .base_worker import BaseGPUWorker


class PlexusWorker(BaseGPUWorker):
    """Worker for Plexus algorithm (standard local training)."""
    pass


def train_plexus_clients_on_gpu(
    gpu_id: int,
    clients: List[PlexusClient],
    global_params: OrderedDict,
    config: Dict,
    results_dict: Dict,
    trainer: PlexusTrainer,
    use_cpu: bool = False,
):
    """
    Train Plexus clients on a specific GPU.

    Each client follows the standard training protocol (same as FedAvg):
    1. Load global model.
    2. Train with CrossEntropyLoss for ``local_epochs``.
    3. Return updated parameters.

    The Plexus-specific sampling and aggregation logic is handled by
    ``PlexusServer``, not here.

    Args:
        gpu_id: GPU ID (0, 1, 2, ...).
        clients: List of Plexus clients to train.
        global_params: Global model parameters.
        config: Training configuration.
        results_dict: Shared dict to store results.
        trainer: PlexusTrainer instance.
        use_cpu: Whether to use CPU instead of GPU.
    """
    worker = PlexusWorker(
        gpu_id, clients, global_params, config, results_dict, trainer, use_cpu,
    )
    worker.run()
