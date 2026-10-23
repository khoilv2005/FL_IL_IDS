"""
GLFC Worker - Multi-GPU training worker for GLFC clients.

Handles parallel training of GLFC clients with:
- Entropy-based signal detection
- Exemplar set update
- Knowledge distillation
- Prototype gradient computation

Inherits from BaseGPUWorker, overriding:
- prepare_client(): calls update_exemplar_set before training
"""

from collections import OrderedDict
from typing import Dict, List

from ..clients.glfc_client import GLFCClient
from ..strategies.fed_incremental.glfc import GLFCTrainer
from .base_worker import BaseGPUWorker


class GLFCWorker(BaseGPUWorker):
    """Worker for GLFC algorithm with exemplar management."""

    def prepare_client(self, client, model, idx: int):
        """Step 1: Detect entropy signal and update exemplar set."""
        if hasattr(client, "update_exemplar_set"):
            client.update_exemplar_set(model, self.device)


def train_glfc_clients_on_gpu(
    gpu_id: int,
    clients: List[GLFCClient],
    global_params: OrderedDict,
    config: Dict,
    results_dict: Dict,
    trainer: GLFCTrainer,
    use_cpu: bool = False,
):
    """
    Train GLFC clients on a specific GPU.

    Each client follows the GLFC training protocol:
    1. Load global model
    2. Detect entropy signal (global forgetting compensation)
    3. Update exemplar set if signal detected
    4. Train with combined loss (CE + distillation)
    5. Compute prototype gradients for proxy server

    Args:
        gpu_id: GPU ID (0, 1, 2, ...)
        clients: List of GLFC clients to train
        global_params: Global model parameters
        config: Training configuration
        results_dict: Shared dict to store results
        trainer: GLFCTrainer instance
        use_cpu: Whether to use CPU instead of GPU
    """
    worker = GLFCWorker(
        gpu_id, clients, global_params, config, results_dict, trainer, use_cpu,
    )
    worker.run()
