"""
NICE Worker - Multi-GPU training worker for NICE clients.

Handles parallel training of NICE clients on GPU.
Follows der_worker.py pattern: create model, load params, transfer
neuron ages/masks, setup client, train, collect results.
"""

import time
from collections import OrderedDict
from typing import Dict, List

import numpy as np
import torch

from ..models.nice_model import NICEModel


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

    This function runs in a separate thread for multi-GPU parallelism.

    Args:
        gpu_id: GPU ID (0, 1, 2, ...)
        clients: List of NICEClient instances
        global_params: Global NICEModel parameters
        config: Training configuration (includes neuron_ages, masks, freeze_masks)
        results_dict: Shared dict to store results (thread-safe by GIL)
        trainer: NICETrainer instance
        use_cpu: Whether to use CPU instead of GPU
    """
    if use_cpu:
        device = "cpu"
        device_name = "CPU"
    else:
        device = f"cuda:{gpu_id}"
        device_name = f"GPU {gpu_id}"

    gpu_start = time.time()

    # Create NICEModel for this device
    model = NICEModel(config["input_shape"], config["num_classes"]).to(device)

    # Training hyperparameters
    epochs = config.get("local_epochs", 3)
    batch_size = config.get("batch_size", 128)
    lr = config.get("learning_rate", 0.001)

    # Extract neuron ages and masks from config
    neuron_ages = config.get("neuron_ages", None)
    masks_state = config.get("masks", None)
    freeze_masks_raw = config.get("freeze_masks", {})

    for idx, client in enumerate(clients):
        # Load global params into model
        model.load_state_dict(
            {k: v.to(device) for k, v in global_params.items()},
            strict=True,
        )

        # Transfer neuron ages from server
        if neuron_ages is not None:
            model.set_neuron_ages_state(neuron_ages)

        # Transfer weight/bias masks from server
        if masks_state is not None:
            model.set_masks_state(masks_state)

        # Transfer freeze masks
        model.freeze_masks = {}
        for k, v in freeze_masks_raw.items():
            if isinstance(v, list):
                model.freeze_masks[k] = np.array(v, dtype=bool)
            elif isinstance(v, np.ndarray):
                model.freeze_masks[k] = v.copy()

        # Move masks to device
        model._move_masks_to_device(device)

        # Setup client for this GPU
        client.setup_for_gpu(model, device)

        # Train
        result = client.train(
            trainer=trainer,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            global_params=global_params,
            is_last_task=config.get("is_last_task", False),
        )

        results_dict[client.client_id] = result

    gpu_time = time.time() - gpu_start
    print(f"      [{device_name}] NICE: "
          f"{len(clients)} clients done in {gpu_time:.1f}s")

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
