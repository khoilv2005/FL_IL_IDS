"""
GLFC Worker - Multi-GPU training worker for GLFC clients.

Handles parallel training of GLFC clients with:
- Entropy-based signal detection
- Exemplar set update
- Knowledge distillation
- Prototype gradient computation
"""

import time
from collections import OrderedDict
from typing import Dict, List

import torch

from ..models.cnn_gru import CNN_GRU_Model
from ..clients.glfc_client import GLFCClient
from ..strategies.incremental.glfc import GLFCTrainer


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

    This function is designed to run in a separate thread.
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
    device = "cpu" if use_cpu else f"cuda:{gpu_id}"
    device_name = "CPU" if use_cpu else f"GPU {gpu_id}"

    gpu_start = time.time()

    # Create model for this GPU
    model = CNN_GRU_Model(config["input_shape"], config["num_classes"]).to(device)

    # Training hyperparameters
    epochs = config.get("local_epochs", 3)
    batch_size = config.get("batch_size", 128)
    lr = config.get("learning_rate", 0.001)

    print(f"      [{device_name}] Starting {len(clients)} GLFC clients...")

    for idx, client in enumerate(clients):
        # Load global params
        model.load_state_dict({k: v.to(device) for k, v in global_params.items()})

        # Setup client for this GPU
        client.setup_for_gpu(model, device)

        # Step 1: Detect entropy signal and update exemplar set
        if hasattr(client, "update_exemplar_set"):
            client.update_exemplar_set(model, device)

        # Step 2: Train with GLFC loss
        result = client.train(
            trainer=trainer,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            global_params=global_params,
        )

        # Log progress
        if (idx + 1) % 50 == 0 or idx == len(clients) - 1:
            print(
                f"      [{device_name}] Progress: {idx + 1}/{len(clients)} clients done"
            )

        results_dict[client.client_id] = result

    gpu_time = time.time() - gpu_start
    print(f"      [{device_name}] All {len(clients)} GLFC clients done in {gpu_time:.2f}s")

    # Cleanup
    del model
    if not use_cpu:
        torch.cuda.empty_cache()
