"""
FedLwF Worker - Multi-GPU training worker for FedLwF clients.

Handles parallel training of FedLwF clients with knowledge distillation.

Inherits from BaseGPUWorker with no additional overrides needed
(standard training loop with distillation handled by trainer/client).
"""

from collections import OrderedDict
from typing import Dict, List

from ..clients.fedlwf_client import FedLwFClient
from ..strategies.incremental.fedlwf import FedLwFTrainer
from .base_worker import BaseGPUWorker


class FedLwFWorker(BaseGPUWorker):
    """Worker for FedLwF - uses standard training loop."""
    pass


def train_fedlwf_clients_on_gpu(
    gpu_id: int,
    clients: List[FedLwFClient],
    global_params: OrderedDict,
    config: Dict,
    results_dict: Dict,
    trainer: FedLwFTrainer,
    use_cpu: bool = False,
):
    """
    Train FedLwF clients on a specific GPU.

    Args:
        gpu_id: GPU ID (0, 1, 2, ...)
        clients: List of FedLwF clients to train
        global_params: Global model parameters
        config: Training configuration
        results_dict: Shared dict to store results
        trainer: FedLwFTrainer instance
        use_cpu: Whether to use CPU instead of GPU
    """
    worker = FedLwFWorker(
        gpu_id, clients, global_params, config, results_dict, trainer, use_cpu,
    )
    worker.run()
