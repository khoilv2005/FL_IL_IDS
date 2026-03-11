"""
NICE Worker - Multi-GPU training worker for NICE clients.

Handles parallel training of NICE clients on GPU.
Follows der_worker.py pattern: create model, load params, transfer
neuron ages/masks, setup client, train, collect results.

Inherits from BaseGPUWorker, overriding:
- create_model(): uses NICEModel
- prepare_client(): transfers neuron ages, masks, freeze masks
- get_train_kwargs(): passes is_last_task flag
"""

from collections import OrderedDict
from typing import Dict, List

import numpy as np
import torch

from ..models.nice_model import NICEModel
from .base_worker import BaseGPUWorker


class NICEWorker(BaseGPUWorker):
    """Worker for NICE algorithm with NICEModel and neuron age/mask transfer."""

    def __init__(
        self,
        gpu_id: int,
        clients: list,
        global_params: OrderedDict,
        config: Dict,
        results_dict: Dict,
        trainer,
        use_cpu: bool = False,
    ):
        super().__init__(gpu_id, clients, global_params, config, results_dict, trainer, use_cpu)
        # Extract neuron ages and masks from config
        self.neuron_ages = config.get("neuron_ages", None)
        self.masks_state = config.get("masks", None)
        self.freeze_masks_raw = config.get("freeze_masks", {})
        self.is_last_task = config.get("is_last_task", False)

    def create_model(self):
        """Create NICEModel instead of CNN_GRU_Model."""
        return NICEModel(
            self.config["input_shape"], self.config["num_classes"]
        ).to(self.device)

    def prepare_client(self, client, model, idx: int):
        """Transfer neuron ages, weight/bias masks, and freeze masks."""
        # Transfer neuron ages from server
        if self.neuron_ages is not None:
            model.set_neuron_ages_state(self.neuron_ages)

        # Transfer weight/bias masks from server
        if self.masks_state is not None:
            model.set_masks_state(self.masks_state)

        # Transfer freeze masks
        model.freeze_masks = {}
        for k, v in self.freeze_masks_raw.items():
            if isinstance(v, list):
                model.freeze_masks[k] = np.array(v, dtype=bool)
            elif isinstance(v, np.ndarray):
                model.freeze_masks[k] = v.copy()

        # Move masks to device
        model._move_masks_to_device(self.device)

    def get_train_kwargs(self, client, idx: int) -> Dict:
        return {
            "global_params": self.global_params,
            "is_last_task": self.is_last_task,
        }

    def run(self):
        """Override run for NICE-specific logging."""
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
        print(f"      [{self.device_name}] NICE: "
              f"{len(self.clients)} clients done in {gpu_time:.1f}s")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def train_nice_clients_on_gpu(
    gpu_id: int,
    clients: list,
    global_params: OrderedDict,
    config: Dict,
    results_dict: Dict,
    trainer,
    use_cpu: bool = False,
):
    """
    Train NICE clients on a specific GPU.

    Args:
        gpu_id: GPU ID (0, 1, 2, ...)
        clients: List of NICEClient instances
        global_params: Global NICEModel parameters
        config: Training configuration (includes neuron_ages, masks, freeze_masks)
        results_dict: Shared dict to store results (thread-safe by GIL)
        trainer: NICETrainer instance
        use_cpu: Whether to use CPU instead of GPU
    """
    worker = NICEWorker(
        gpu_id, clients, global_params, config, results_dict, trainer, use_cpu,
    )
    worker.run()
