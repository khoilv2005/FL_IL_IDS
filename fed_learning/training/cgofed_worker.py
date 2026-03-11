"""
CGoFed GPU Worker - Specialized worker for Class Incremental Learning.
Trains CGoFed clients on specific GPU.

Inherits from BaseGPUWorker, overriding:
- get_init_params(): for Eq.12 personalized initialization
- get_train_kwargs(): for per-client regularization info
"""

from collections import OrderedDict
from typing import Dict, List, Optional

from ..clients.cgofed_client import CGoFedClient
from ..core import BaseTrainer
from .base_worker import BaseGPUWorker


class CGoFedWorker(BaseGPUWorker):
    """Worker for CGoFed algorithm with per-client regularization."""

    def __init__(
        self,
        gpu_id: int,
        clients: List[CGoFedClient],
        global_params: OrderedDict,
        config: Dict,
        results_dict: Dict,
        trainer: BaseTrainer,
        use_cpu: bool = False,
        client_reg_info: Dict = None,
        client_init_models: Dict = None,
    ):
        super().__init__(gpu_id, clients, global_params, config, results_dict, trainer, use_cpu)
        self.client_reg_info = client_reg_info or {}
        self.client_init_models = client_init_models or {}

    def get_init_params(self, client) -> Optional[OrderedDict]:
        """Eq.12: initialize from per-client personalized model when available."""
        if client.client_id in self.client_init_models:
            client_init = self.client_init_models[client.client_id]
            if client_init is not None:
                return client_init
        return self.global_params

    def get_train_kwargs(self, client, idx: int) -> Dict:
        """Paper Eq.14: Pass per-client regularization info."""
        init_params = self.get_init_params(client)
        kwargs = {"global_params": init_params}
        if self.client_reg_info and client.client_id in self.client_reg_info:
            kwargs.update(self.client_reg_info[client.client_id])
        return kwargs


def train_cgofed_clients_on_gpu(
    gpu_id: int,
    clients: List[CGoFedClient],
    global_params: OrderedDict,
    config: Dict,
    results_dict: Dict,
    trainer: BaseTrainer,
    use_cpu: bool = False,
    client_reg_info: Dict = None,
    client_init_models: Dict = None,
):
    """
    Train CGoFed clients on a specific GPU.

    Similar to standard train_clients_on_gpu but uses CGoFedClient
    which automatically computes representation matrix after training.

    Args:
        gpu_id: GPU index (0, 1, 2, ...)
        clients: List of CGoFedClient instances
        global_params: Global model parameters
        config: Training configuration
        results_dict: Shared dict to store results
        trainer: CGoFedTrainer instance
        use_cpu: If True, use CPU instead of GPU
        client_reg_info: Per-client regularization info (Eq.14)
        client_init_models: Per-client personalized init models (Eq.12)
    """
    worker = CGoFedWorker(
        gpu_id, clients, global_params, config, results_dict,
        trainer, use_cpu, client_reg_info, client_init_models,
    )
    worker.run()
