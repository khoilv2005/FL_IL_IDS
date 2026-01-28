"""
FedCBDR Worker - Multi-GPU training worker for FedCBDR clients.

Handles parallel training of FedCBDR clients on GPU with replay buffer.
"""

from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..models.cnn_gru import CNN_GRU_Model
from ..clients.fedcbdr_client import FedCBDRClient
from ..strategies.incremental.fedcbdr import FedCBDRTrainer


def train_fedcbdr_clients_on_gpu(
    gpu_id: int,
    clients: List[FedCBDRClient],
    global_params: OrderedDict,
    config: Dict,
    results_dict: Dict,
    trainer: FedCBDRTrainer,
    use_cpu: bool = False
):
    """
    Train FedCBDR clients on a specific GPU.
    
    This function is designed to run in a separate thread.
    
    Args:
        gpu_id: GPU ID (0, 1, 2, ...)
        clients: List of FedCBDR clients to train
        global_params: Global model parameters
        config: Training configuration
        results_dict: Shared dict to store results
        trainer: FedCBDRTrainer instance
        use_cpu: Whether to use CPU instead of GPU
    """
    device = "cpu" if use_cpu else f"cuda:{gpu_id}"
    
    # Create model for this GPU
    model = CNN_GRU_Model(
        config["input_shape"], config["num_classes"]
    ).to(device)
    
    # Training hyperparameters
    epochs = config.get("local_epochs", 3)
    batch_size = config.get("batch_size", 128)
    lr = config.get("learning_rate", 0.001)
    use_replay = config.get("use_replay", True)
    replay_ratio = config.get("replay_ratio", 0.5)
    
    for client in clients:
        # Load global params
        model.load_state_dict({
            k: v.to(device) for k, v in global_params.items()
        })
        
        # Setup client for this GPU
        client.setup_for_gpu(model, device)
        
        # Train with replay
        result = client.train(
            trainer=trainer,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            global_params=global_params,
            use_replay=use_replay,
            replay_ratio=replay_ratio,
        )
        
        results_dict[client.client_id] = result
    
    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
