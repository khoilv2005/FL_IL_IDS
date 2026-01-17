"""
GPU Trainer - Train clients on specific GPU using strategy pattern.
"""

import time
from collections import OrderedDict
from typing import Dict, List

import torch

from ..models.cnn_gru import CNN_GRU_Model
from ..clients.client import FederatedClient
from ..core import BaseTrainer


def train_clients_on_gpu(
    gpu_id: int,
    clients: List[FederatedClient],
    global_params: OrderedDict,
    config: Dict,
    results_dict: Dict,
    trainer: BaseTrainer,
    use_cpu: bool = False
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
    """
    if use_cpu:
        device = "cpu"
        device_name = "CPU"
    else:
        device = f"cuda:{gpu_id}"
        device_name = f"GPU {gpu_id}"
    
    gpu_start = time.time()
    
    # Create model for this device
    model = CNN_GRU_Model(config["input_shape"], config["num_classes"]).to(device)
    
    print(f"      [{device_name}] Starting {len(clients)} clients...")
    
    for idx, client in enumerate(clients):
        # Load global params into model
        model.load_state_dict({k: v.to(device) for k, v in global_params.items()})
        
        # Setup client for this GPU
        client.setup_for_gpu(model, device)
        
        # Train using strategy
        result = client.train(
            trainer=trainer,
            epochs=config["local_epochs"],
            batch_size=config["batch_size"],
            lr=config["learning_rate"],
            global_params=global_params,
        )
        
        # Log progress
        if (idx + 1) % 50 == 0 or idx == len(clients) - 1:
            print(f"      [{device_name}] Progress: {idx+1}/{len(clients)} clients done")
        
        results_dict[client.client_id] = result
    
    gpu_time = time.time() - gpu_start
    print(f"      [{device_name}] ✓ All {len(clients)} clients done in {gpu_time:.2f}s")
    
    # Clear GPU memory
    del model
    if not use_cpu:
        torch.cuda.empty_cache()
