"""
GPU Trainer - Train clients on specific GPU
"""

import time
from collections import OrderedDict
from typing import Dict, List

import torch

from ..models.cnn_gru import CNN_GRU_Model
from ..clients.client import FederatedClientMultiGPU


def train_clients_on_gpu(gpu_id: int, clients: List[FederatedClientMultiGPU],
                          global_params: OrderedDict, config: Dict,
                          results_dict: Dict, algorithm: str,
                          use_cpu: bool = False):
    """
    Train một nhóm clients trên 1 GPU cụ thể (hoặc CPU).
    Hàm này chạy trong 1 thread riêng.
    
    Args:
        gpu_id: GPU index (0, 1, 2, ...)
        clients: List of clients to train
        global_params: Global model parameters
        config: Training configuration
        results_dict: Shared dict to store results
        algorithm: Federated learning algorithm
        use_cpu: Nếu True, dùng CPU thay vì GPU
    """
    if use_cpu:
        device = "cpu"
        device_name = "CPU"
    else:
        device = f"cuda:{gpu_id}"
        device_name = f"GPU {gpu_id}"
    
    gpu_start = time.time()
    
    # Tạo model cho device này
    model = CNN_GRU_Model(config["input_shape"], config["num_classes"]).to(device)
    
    print(f"      [{device_name}] Starting {len(clients)} clients...")
    
    for idx, client in enumerate(clients):
        client_start = time.time()
        
        # Load global params vào model
        model.load_state_dict({k: v.to(device) for k, v in global_params.items()})
        
        # Setup client
        client.setup_for_gpu(model, device)
        
        # Train theo algorithm
        if algorithm == "fedavg":
            result = client.train_fedavg(
                config["local_epochs"], config["batch_size"], config["learning_rate"]
            )
        elif algorithm == "fedprox":
            result = client.train_fedprox(
                config["local_epochs"], config["batch_size"], 
                global_params, config["mu"], config["learning_rate"]
            )
        elif algorithm == "fedavgm":
            # FedAvgM uses same client training as FedAvg
            result = client.train_fedavg(
                config["local_epochs"], config["batch_size"], config["learning_rate"]
            )
        elif algorithm == "fedplus":
            result = client.train_fedplus(
                config["local_epochs"], config["batch_size"],
                global_params, config["mu"], config["learning_rate"]
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        client_time = time.time() - client_start
        
        # Log mỗi 50 clients thay vì mỗi client
        if (idx + 1) % 50 == 0 or idx == len(clients) - 1:
            print(f"      [{device_name}] Progress: {idx+1}/{len(clients)} clients done")
        
        results_dict[client.client_id] = result
    
    gpu_time = time.time() - gpu_start
    print(f"      [{device_name}] ✓ All {len(clients)} clients done in {gpu_time:.2f}s")
    
    # Clear GPU memory (only if using GPU)
    del model
    if not use_cpu:
        torch.cuda.empty_cache()
