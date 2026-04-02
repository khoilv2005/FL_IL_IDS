"""
PlexusDER Worker - Multi-GPU training worker for PlexusDER clients.

Combines:
- PlexusWorker pattern (sample/aggregator selection handled by server)
- DERWorker pattern (two-stage training, DERModel, exemplar replay)

Inherits from BaseGPUWorker, overriding:
- create_model(): uses DERModel
- prepare_model(): reconstructs task structure
- get_train_kwargs(): passes stage and replay_ratio
- prepare_client(): sets up annealing schedule
"""

from collections import OrderedDict
from typing import Dict, List

import torch

from ..models.der_model import DERModel
from .base_worker import BaseGPUWorker
from .der_worker import _reconstruct_model_structure


class PlexusDERWorker(BaseGPUWorker):
    """Worker for PlexusDER algorithm with two-stage training and DERModel."""

    def __init__(
        self,
        gpu_id: int,
        clients: list,
        global_params: OrderedDict,
        config: Dict,
        results_dict: Dict,
        trainer,
        use_cpu: bool = False,
        stage: int = 1,
    ):
        super().__init__(gpu_id, clients, global_params, config, results_dict, trainer, use_cpu)
        self.stage = stage
        self.replay_ratio = config.get("replay_ratio", 0.5)

    def create_model(self):
        """Create DERModel instead of CNN_GRU_Model."""
        return DERModel(
            self.config["input_shape"], self.config["num_classes"]
        ).to(self.device)

    def prepare_model(self, model):
        """Reconstruct DERModel task structure to match global_params."""
        _reconstruct_model_structure(model, self.global_params, self.config)
        model.to(self.device)

    def prepare_client(self, client, model, idx: int):
        """Set up DER-specific annealing schedule per client."""
        # Compute total batches PER EPOCH for annealing schedule (Eq.8)
        if hasattr(self.trainer, 'total_batches'):
            self.trainer.total_batches = max(1, client.num_samples // self.batch_size)
        # Reset batch counter per client
        if hasattr(self.trainer, 'current_batch'):
            self.trainer.current_batch = 0

    def get_train_kwargs(self, client, idx: int) -> Dict:
        """Pass stage and replay_ratio to client.train()."""
        return {
            "global_params": self.global_params,
            "stage": self.stage,
            "replay_ratio": self.replay_ratio,
        }

    def run(self):
        """Override run to customize logging message."""
        import time
        gpu_start = time.time()

        model = self.create_model()
        self.prepare_model(model)

        for idx, client in enumerate(self.clients):
            init_params = self.get_init_params(client)
            self.load_params(model, init_params)

            client.setup_for_gpu(model, self.device)
            self.prepare_client(client, model, idx)

            train_kwargs = self.get_train_kwargs(client, idx)
            result = client.train(
                trainer=self.trainer,
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                **train_kwargs,
            )

            self.results_dict[client.client_id] = result

        gpu_time = time.time() - gpu_start
        stage_name = "Representation" if self.stage == 1 else "Classifier"
        print(f"      [{self.device_name}] PlexusDER Stage {self.stage} ({stage_name}): "
              f"{len(self.clients)} clients done in {gpu_time:.1f}s")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def train_plexus_der_clients_on_gpu(
    gpu_id: int,
    clients: list,
    global_params: OrderedDict,
    config: Dict,
    results_dict: Dict,
    trainer,
    use_cpu: bool = False,
    stage: int = 1,
):
    """
    Train PlexusDER clients on a specific GPU.

    Args:
        gpu_id: GPU ID (0, 1, 2, ...)
        clients: List of PlexusDERClient instances
        global_params: Global DERModel parameters
        config: Training configuration
        results_dict: Shared dict to store results (thread-safe by GIL)
        trainer: PlexusDERTrainer instance
        use_cpu: Whether to use CPU instead of GPU
        stage: Training stage (1=representation, 2=classifier)
    """
    worker = PlexusDERWorker(
        gpu_id, clients, global_params, config, results_dict,
        trainer, use_cpu, stage,
    )
    worker.run()
