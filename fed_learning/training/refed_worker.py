"""
Re-Fed Worker - Multi-GPU training worker for Re-Fed clients.

Handles parallel training of Re-Fed clients with:
- Standard local training on cached + new data
- Each client's cache was prepared via PIM before training rounds

Inherits from BaseGPUWorker with no additional overrides needed
(standard training loop with PIM caching handled by server).
"""

from collections import OrderedDict
from typing import Dict, List

from ..clients.refed_client import ReFedClient
from ..strategies.incremental.refed import ReFedTrainer
from .base_worker import BaseGPUWorker


class ReFedWorker(BaseGPUWorker):
    """Worker for Re-Fed - uses standard training loop."""
    pass


def train_refed_clients_on_gpu(
    gpu_id: int,
    clients: List[ReFedClient],
    global_params: OrderedDict,
    config: Dict,
    results_dict: Dict,
    trainer: ReFedTrainer,
    use_cpu: bool = False,
):
    """
    Train Re-Fed clients on a specific GPU.

    Each client trains on combined data (cached + new task),
    following Paper Eq. 6.

    Args:
        gpu_id: GPU ID (0, 1, 2, ...)
        clients: List of Re-Fed clients to train
        global_params: Global model parameters
        config: Training configuration
        results_dict: Shared dict to store results
        trainer: ReFedTrainer instance
        use_cpu: Whether to use CPU instead of GPU
    """
    worker = ReFedWorker(
        gpu_id, clients, global_params, config, results_dict, trainer, use_cpu,
    )
    worker.run()
