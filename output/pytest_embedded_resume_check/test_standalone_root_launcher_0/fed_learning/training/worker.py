"""
GPU Trainer - Train clients on specific GPU using strategy pattern.

Uses BaseGPUWorker for the common training loop.
"""

from collections import OrderedDict
from typing import Dict, List

from ..core import BaseTrainer
from ..clients.client import FederatedClient
from .base_worker import BaseGPUWorker


class StandardWorker(BaseGPUWorker):
    """Standard worker for FedAvg/FedProx/FedPlus/EWC."""

    def __init__(
        self,
        gpu_id: int,
        clients: List[FederatedClient],
        global_params: OrderedDict,
        config: Dict,
        results_dict: Dict,
        trainer: BaseTrainer,
        use_cpu: bool = False,
        local_reg_info: Dict = None,
    ):
        super().__init__(gpu_id, clients, global_params, config, results_dict, trainer, use_cpu)
        self.local_reg_info = local_reg_info or {}

    def get_train_kwargs(self, client, idx: int) -> Dict:
        kwargs = {"global_params": self.global_params}
        if self.local_reg_info:
            kwargs.update(self.local_reg_info)
        return kwargs


def train_clients_on_gpu(
    gpu_id: int,
    clients: List[FederatedClient],
    global_params: OrderedDict,
    config: Dict,
    results_dict: Dict,
    trainer: BaseTrainer,
    use_cpu: bool = False,
    local_reg_info: Dict = None,
):
    """
    Train a group of clients on a specific GPU (or CPU).

    This function runs in a separate thread for multi-GPU parallelism.

    Args:
        gpu_id: GPU index (0, 1, 2, ...)
        clients: List of clients to train
        global_params: Global model parameters
        config: Training configuration
        results_dict: Shared dict to store results (thread-safe by GIL)
        trainer: Training strategy (FedAvgTrainer, FedProxTrainer, etc.)
        use_cpu: If True, use CPU instead of GPU
        local_reg_info: Local regularization info (CGoFed Eq. 14)
    """
    worker = StandardWorker(
        gpu_id, clients, global_params, config, results_dict,
        trainer, use_cpu, local_reg_info,
    )
    worker.run()
