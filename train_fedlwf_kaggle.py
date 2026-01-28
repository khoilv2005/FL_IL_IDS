"""
FedLwF Kaggle Training Script

Federated Learning with Learning without Forgetting (LwF) for Class-Incremental Learning.
Optimized for Kaggle notebook environment with multi-GPU support.

Key Concept:
    FedLwF combines FedAvg with Knowledge Distillation to prevent catastrophic forgetting
    when learning new classes incrementally in a federated setting.

Loss Function:
    L = L_CE(y, ŷ) + α * T² * KL(σ(z_old/T) || σ(z_new/T))
    
    Where:
    - L_CE: Cross-entropy loss for new task
    - KL: KL divergence for knowledge distillation
    - α: Distillation weight (increases with tasks)
    - T: Temperature for softening probabilities
    - z_old, z_new: Logits from old and new models

Usage:
    In Kaggle notebook:
    >>> from train_fedlwf_kaggle import main
    >>> main()
"""

import os
import sys
import copy
import json
import time
import gc
import warnings
import threading
from datetime import datetime
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

# Suppress warnings
warnings.filterwarnings('ignore')

# Check if running on Kaggle
IS_KAGGLE = '/kaggle' in os.getcwd() or os.path.exists('/kaggle/working')

# Set paths
if IS_KAGGLE:
    sys.path.insert(0, '/kaggle/working/Code/FL_IL_IDS')
    BASE_PATH = "/kaggle/input/iot-dataset-2023"
else:
    BASE_PATH = os.path.join(os.path.dirname(__file__), "data_chunks")

# Import project modules
from fed_learning.models.cnn_gru import CNN_GRU_Model
from fed_learning.data.incremental_loader import IncrementalDataLoader
from fed_learning.clients.fedlwf_client import FedLwFClient
from fed_learning.servers.fedlwf_server import FedLwFServer
from fed_learning.strategies.incremental.fedlwf import (
    FedLwFTrainer, 
    FedLwFAggregator,
)


# ============================================================================
#                           CONFIGURATION
# ============================================================================

def get_config() -> Dict:
    """Get training configuration for FedLwF."""
    config = {
        # Dataset settings
        "base_path": BASE_PATH,
        "num_clients": 3,
        
        # Model settings
        "input_shape": (1, 78),  # Network traffic features
        "initial_classes": 10,   # Base task classes
        "classes_per_task": 6,   # New classes per incremental task
        "total_classes": 34,     # Total attack types
        
        # Training settings
        "num_rounds": 15,
        "local_epochs": 3,
        "batch_size": 128,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        
        # FedLwF specific settings
        "distillation_alpha": 1.0,       # Base KD weight
        "distillation_alpha_scale": 1.5, # Scale factor per task
        "temperature": 2.0,              # Temperature for soft targets
        "alpha_decay": 0.95,             # Optional decay per round
        
        # FedProx settings (optional, set proximal_mu=0 to disable)
        "proximal_mu": 0.0,              # Set >0 to enable FedProx
        
        # Evaluation settings
        "eval_frequency": 5,  # Evaluate every N rounds
        "save_snapshots": True,
        
        # System settings
        "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "use_multiprocessing": True,
    }
    
    # Calculate number of tasks
    remaining = config["total_classes"] - config["initial_classes"]
    config["num_tasks"] = 1 + (remaining // config["classes_per_task"])
    
    return config


# ============================================================================
#                           GPU UTILITIES
# ============================================================================

def get_device_info() -> Dict:
    """Get GPU/device information."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": []
    }
    
    if info["cuda_available"]:
        for i in range(info["num_gpus"]):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append({
                "id": i,
                "name": props.name,
                "memory_gb": props.total_memory / (1024**3),
            })
    
    return info


def setup_environment():
    """Setup training environment."""
    # Set random seeds
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Optimize CUDA
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # Print environment info
    print("=" * 60)
    print("FedLwF Training Environment Setup")
    print("=" * 60)
    
    device_info = get_device_info()
    if device_info["cuda_available"]:
        print(f"CUDA: Available ({device_info['num_gpus']} GPU(s))")
        for dev in device_info["devices"]:
            print(f"  GPU {dev['id']}: {dev['name']} ({dev['memory_gb']:.1f} GB)")
    else:
        print("CUDA: Not available, using CPU")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"Running on: {'Kaggle' if IS_KAGGLE else 'Local'}")
    print("=" * 60)
    
    return device_info


# ============================================================================
#                           DATA LOADING
# ============================================================================

def prepare_task_data(
    data_loader: IncrementalDataLoader,
    task_id: int,
    config: Dict
) -> Dict[str, Dict]:
    """
    Prepare data for a specific task.
    
    Args:
        data_loader: Incremental data loader
        task_id: Task ID (0-based)
        config: Configuration dict
        
    Returns:
        Dictionary mapping client_id to their data
    """
    print(f"\n[Task {task_id}] Preparing data...")
    
    # Calculate class range
    if task_id == 0:
        start_class = 0
        end_class = config["initial_classes"]
    else:
        start_class = config["initial_classes"] + (task_id - 1) * config["classes_per_task"]
        end_class = start_class + config["classes_per_task"]
    
    print(f"[Task {task_id}] Classes: {start_class} to {end_class - 1}")
    
    # Load data for each client
    client_data = {}
    
    for client_id in range(1, config["num_clients"] + 1):
        # Load current task data
        train_data = data_loader.load_client_data(
            client_id=client_id,
            task_id=task_id,
            split="train"
        )
        
        test_data = data_loader.load_client_data(
            client_id=client_id,
            task_id=task_id,
            split="test"
        )
        
        if train_data is not None:
            client_data[client_id] = {
                "train": train_data,
                "test": test_data,
                "classes": list(range(start_class, end_class)),
                "num_samples": len(train_data[0]) if train_data else 0
            }
            print(f"  Client {client_id}: {client_data[client_id]['num_samples']} samples")
    
    return client_data


def load_test_data_all_tasks(
    data_loader: IncrementalDataLoader,
    up_to_task: int,
    config: Dict
) -> List[Tuple]:
    """
    Load test data for all tasks up to current task.
    
    Args:
        data_loader: Data loader
        up_to_task: Load tasks 0 to up_to_task (inclusive)
        config: Configuration
        
    Returns:
        List of (X_test, y_test) tuples for each task
    """
    test_data_list = []
    
    for task_id in range(up_to_task + 1):
        # Aggregate test data from all clients for this task
        X_test_all = []
        y_test_all = []
        
        for client_id in range(1, config["num_clients"] + 1):
            test_data = data_loader.load_client_data(
                client_id=client_id,
                task_id=task_id,
                split="test"
            )
            
            if test_data is not None:
                X, y = test_data
                X_test_all.append(X)
                y_test_all.append(y)
        
        if X_test_all:
            X_combined = np.concatenate(X_test_all, axis=0)
            y_combined = np.concatenate(y_test_all, axis=0)
            test_data_list.append((X_combined, y_combined))
    
    return test_data_list


# ============================================================================
#                           MULTI-GPU TRAINING
# ============================================================================

def train_clients_parallel(
    clients: List[FedLwFClient],
    global_params: OrderedDict,
    config: Dict,
    trainer: FedLwFTrainer,
) -> Dict[int, Dict]:
    """
    Train clients in parallel on multiple GPUs.
    
    Args:
        clients: List of FedLwF clients
        global_params: Global model parameters
        config: Training configuration
        trainer: FedLwFTrainer instance
        
    Returns:
        Dictionary mapping client_id to training results
    """
    from fed_learning.training.fedlwf_worker import train_fedlwf_clients_on_gpu
    
    num_gpus = config["num_gpus"]
    results = {}
    
    if num_gpus == 0:
        # CPU training
        print("  Training on CPU...")
        train_fedlwf_clients_on_gpu(
            gpu_id=0,
            clients=clients,
            global_params=global_params,
            config=config,
            results_dict=results,
            trainer=trainer,
            use_cpu=True
        )
    elif num_gpus == 1:
        # Single GPU
        print("  Training on single GPU...")
        train_fedlwf_clients_on_gpu(
            gpu_id=0,
            clients=clients,
            global_params=global_params,
            config=config,
            results_dict=results,
            trainer=trainer,
            use_cpu=False
        )
    else:
        # Multi-GPU parallel training
        print(f"  Training on {num_gpus} GPUs...")
        
        # Distribute clients across GPUs
        clients_per_gpu = [[] for _ in range(num_gpus)]
        for i, client in enumerate(clients):
            gpu_idx = i % num_gpus
            clients_per_gpu[gpu_idx].append(client)
        
        # Start threads for each GPU
        threads = []
        for gpu_id in range(num_gpus):
            if clients_per_gpu[gpu_id]:
                thread = threading.Thread(
                    target=train_fedlwf_clients_on_gpu,
                    kwargs={
                        "gpu_id": gpu_id,
                        "clients": clients_per_gpu[gpu_id],
                        "global_params": global_params,
                        "config": config,
                        "results_dict": results,
                        "trainer": trainer,
                        "use_cpu": False
                    }
                )
                threads.append(thread)
                thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
    
    return results


# ============================================================================
#                           MAIN TRAINING LOOP
# ============================================================================

def train_task(
    task_id: int,
    server: FedLwFServer,
    clients: List[FedLwFClient],
    client_data: Dict,
    test_data_list: List[Tuple],
    config: Dict,
    trainer: FedLwFTrainer,
    aggregator: FedLwFAggregator,
) -> Dict:
    """
    Train a single task using FedLwF.
    
    Args:
        task_id: Current task ID
        server: FedLwF server
        clients: List of clients
        client_data: Data for current task
        test_data_list: Test data for all tasks seen so far
        config: Configuration
        trainer: FedLwF trainer
        aggregator: FedLwF aggregator
        
    Returns:
        Task results dictionary
    """
    print(f"\n{'='*60}")
    print(f"TASK {task_id} TRAINING")
    print(f"{'='*60}")
    
    # Calculate number of classes seen so far
    if task_id == 0:
        num_classes = config["initial_classes"]
    else:
        num_classes = config["initial_classes"] + task_id * config["classes_per_task"]
    
    print(f"Total classes seen: {num_classes}")
    
    # Update server for new task
    server.prepare_task(task_id, num_classes)
    
    # Adjust distillation alpha based on task
    if task_id > 0:
        current_alpha = config["distillation_alpha"] * (config["distillation_alpha_scale"] ** (task_id - 1))
        trainer.alpha = current_alpha
        print(f"Distillation alpha: {current_alpha:.3f}")
    
    # Setup clients with new task data
    active_clients = []
    for client in clients:
        cid = client.client_id
        if cid in client_data:
            client.set_task_data(
                task_id=task_id,
                train_data=client_data[cid]["train"],
                test_data=client_data[cid]["test"],
                classes=client_data[cid]["classes"]
            )
            active_clients.append(client)
    
    print(f"Active clients: {len(active_clients)}")
    
    # Training rounds
    task_metrics = {
        "round_losses": [],
        "round_accs": [],
        "task_accs": [],
        "forgetting": [],
    }
    
    num_rounds = config["num_rounds"]
    
    for round_num in range(num_rounds):
        print(f"\n[Task {task_id}] Round {round_num + 1}/{num_rounds}")
        
        # Get global params
        global_params = server.get_global_params()
        
        # Train clients
        results = train_clients_parallel(
            clients=active_clients,
            global_params=global_params,
            config=config,
            trainer=trainer,
        )
        
        # Aggregate results
        client_updates = []
        total_loss = 0.0
        total_samples = 0
        
        for client_id, result in results.items():
            if result is not None:
                client_updates.append({
                    "params": result["params"],
                    "num_samples": result["num_samples"],
                    "loss": result["loss"],
                })
                total_loss += result["loss"] * result["num_samples"]
                total_samples += result["num_samples"]
        
        # Aggregate
        if client_updates:
            aggregated_params = aggregator.aggregate(
                client_updates=client_updates,
                global_params=global_params
            )
            server.update_global_params(aggregated_params)
        
        # Calculate average loss
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        task_metrics["round_losses"].append(avg_loss)
        
        print(f"  Average loss: {avg_loss:.4f}")
        
        # Evaluation
        if (round_num + 1) % config["eval_frequency"] == 0 or round_num == num_rounds - 1:
            print(f"\n  [Evaluation]")
            
            # Evaluate on all seen tasks
            task_accuracies = []
            for eval_task_id, (X_test, y_test) in enumerate(test_data_list):
                acc = server.evaluate(X_test, y_test)
                task_accuracies.append(acc)
                print(f"    Task {eval_task_id}: {acc:.2f}%")
            
            task_metrics["task_accs"].append(task_accuracies)
            
            # Calculate average accuracy
            avg_acc = np.mean(task_accuracies)
            task_metrics["round_accs"].append(avg_acc)
            print(f"    Average: {avg_acc:.2f}%")
            
            # Calculate forgetting (if task > 0)
            if task_id > 0 and len(task_accuracies) > 1:
                forgetting = server.compute_average_forgetting(task_accuracies)
                task_metrics["forgetting"].append(forgetting)
                print(f"    Forgetting: {forgetting:.2f}%")
    
    # Save model snapshot at end of task
    if config.get("save_snapshots", True):
        server.save_global_snapshot(task_id)
    
    # Save client snapshots for future KD
    for client in active_clients:
        client.save_model_snapshot(task_id)
    
    return task_metrics


def main():
    """Main training function for FedLwF."""
    print("\n" + "=" * 70)
    print("FedLwF: Federated Learning without Forgetting")
    print("Class-Incremental Learning with Knowledge Distillation")
    print("=" * 70)
    
    # Setup
    start_time = time.time()
    device_info = setup_environment()
    config = get_config()
    
    # Print configuration
    print("\n[Configuration]")
    print(f"  Tasks: {config['num_tasks']} (Base: {config['initial_classes']} classes, "
          f"Incremental: {config['classes_per_task']} per task)")
    print(f"  Clients: {config['num_clients']}")
    print(f"  Rounds per task: {config['num_rounds']}")
    print(f"  Local epochs: {config['local_epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Distillation alpha: {config['distillation_alpha']}")
    print(f"  Temperature: {config['temperature']}")
    
    # Initialize data loader
    print("\n[Initializing Data Loader]")
    data_loader = IncrementalDataLoader(
        base_path=config["base_path"],
        num_clients=config["num_clients"]
    )
    
    # Create trainer and aggregator
    print("\n[Creating FedLwF Components]")
    trainer = FedLwFTrainer(
        alpha=config["distillation_alpha"],
        temperature=config["temperature"],
    )
    aggregator = FedLwFAggregator()
    
    # Create server
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    server = FedLwFServer(
        model=CNN_GRU_Model(config["input_shape"], config["total_classes"]),
        device=device,
        config=config,
    )
    
    # Create clients
    clients = []
    for client_id in range(1, config["num_clients"] + 1):
        client = FedLwFClient(
            client_id=client_id,
            model=None,  # Will be set during training
            device=device,
            alpha=config["distillation_alpha"],
            temperature=config["temperature"],
        )
        clients.append(client)
    
    print(f"  Created {len(clients)} FedLwF clients")
    
    # Results storage
    all_results = {
        "config": config,
        "tasks": {},
        "final_metrics": {},
    }
    
    # Train each task incrementally
    for task_id in range(config["num_tasks"]):
        # Prepare data for this task
        client_data = prepare_task_data(data_loader, task_id, config)
        
        # Load test data for all seen tasks
        test_data_list = load_test_data_all_tasks(data_loader, task_id, config)
        
        # Train task
        task_metrics = train_task(
            task_id=task_id,
            server=server,
            clients=clients,
            client_data=client_data,
            test_data_list=test_data_list,
            config=config,
            trainer=trainer,
            aggregator=aggregator,
        )
        
        all_results["tasks"][task_id] = task_metrics
        
        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    # Evaluate on all tasks
    final_test_data = load_test_data_all_tasks(
        data_loader, config["num_tasks"] - 1, config
    )
    
    final_accuracies = []
    for task_id, (X_test, y_test) in enumerate(final_test_data):
        acc = server.evaluate(X_test, y_test)
        final_accuracies.append(acc)
        print(f"  Task {task_id}: {acc:.2f}%")
    
    avg_acc = np.mean(final_accuracies)
    print(f"\n  Average Accuracy: {avg_acc:.2f}%")
    
    # Calculate final forgetting
    if config["num_tasks"] > 1:
        forgetting = server.compute_average_forgetting(final_accuracies)
        print(f"  Average Forgetting: {forgetting:.2f}%")
    
    all_results["final_metrics"] = {
        "task_accuracies": final_accuracies,
        "average_accuracy": avg_acc,
        "forgetting": forgetting if config["num_tasks"] > 1 else 0.0,
    }
    
    # Save results
    results_path = os.path.join(
        "/kaggle/working" if IS_KAGGLE else ".",
        f"fedlwf_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n[Results saved to {results_path}]")
    
    # Training time
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time / 60:.1f} minutes")
    print("\n" + "=" * 70)
    print("FedLwF Training Complete!")
    print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    main()
