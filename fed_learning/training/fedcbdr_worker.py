"""
FedCBDR Worker - Multi-GPU training worker for FedCBDR clients.

Handles parallel training of FedCBDR clients on GPU with replay buffer.

Inherits from BaseGPUWorker, overriding:
- get_train_kwargs(): passes replay configuration
"""

from collections import OrderedDict
from typing import Dict, List

from ..clients.fedcbdr_client import FedCBDRClient
from ..strategies.incremental.fedcbdr import FedCBDRTrainer
from .base_worker import BaseGPUWorker


class FedCBDRWorker(BaseGPUWorker):
    """Worker for FedCBDR algorithm with replay buffer support."""

    def __init__(
        self,
        gpu_id: int,
        clients: List[FedCBDRClient],
        global_params: OrderedDict,
        config: Dict,
        results_dict: Dict,
        trainer: FedCBDRTrainer,
        use_cpu: bool = False,
    ):
        super().__init__(gpu_id, clients, global_params, config, results_dict, trainer, use_cpu)
        self.use_replay = config.get("use_replay", True)
        self.replay_ratio = config.get("replay_ratio", 0.5)

    def get_train_kwargs(self, client, idx: int) -> Dict:
        return {
            "global_params": self.global_params,
            "use_replay": self.use_replay,
            "replay_ratio": self.replay_ratio,
        }


def train_fedcbdr_clients_on_gpu(
    gpu_id: int,
    clients: List[FedCBDRClient],
    global_params: OrderedDict,
    config: Dict,
    results_dict: Dict,
    trainer: FedCBDRTrainer,
    use_cpu: bool = False,
):
    """
    Train FedCBDR clients on a specific GPU.

    Args:
        gpu_id: GPU ID (0, 1, 2, ...)
        clients: List of FedCBDR clients to train
        global_params: Global model parameters
        config: Training configuration
        results_dict: Shared dict to store results
        trainer: FedCBDRTrainer instance
        use_cpu: Whether to use CPU instead of GPU
    """
    worker = FedCBDRWorker(
        gpu_id, clients, global_params, config, results_dict, trainer, use_cpu,
    )
    worker.run()
