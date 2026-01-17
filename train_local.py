"""
Federated Learning - Local Training Script
==========================================
Main entry point for training on local machine.

Usage:
    python train_local.py
"""

import os
import json
from datetime import datetime

import torch

from fed_learning import (
    load_all_client_data_to_ram,
    FederatedClient,
    FederatedServer,
    train_federated_multigpu,
)
from fed_learning.visualization.plots import save_training_plots


# =============================================================================
# CONFIG - Sửa các tham số ở đây trước khi train
# =============================================================================
CONFIG = {
    # ===================
    # DATA PATHS
    # ===================
    "data_dir": "./prepare_data/data/federated_splits/100-clients",
    "output_dir": "./results",
    "checkpoint_dir": "./checkpoints",
    
    # ===================
    # FEDERATED LEARNING
    # ===================
    "num_clients": 100,
    "algorithm": "fedplus",  # fedavg, fedprox, fedavgm, fedplus
    
    # FedAvgM
    "server_momentum": 0.9,
    "server_lr": 1.0,
    
    # FedProx / Fed+
    "mu": 0.01,
    
    # ===================
    # MODEL (auto-detect)
    # ===================
    "input_shape": None,
    "num_classes": None,
    
    # ===================
    # TRAINING
    # ===================
    "num_rounds": 3,
    "local_epochs": 3,
    "learning_rate": 1e-3,
    "batch_size": 1024,
    
    # ===================
    # MULTI-GPU
    # ===================
    "num_gpus": None,  # None = auto-detect
    
    # ===================
    # EVAL & SAVE
    # ===================
    "eval_every": 1,
    "save_checkpoint_every": 1,
}


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "="*80)
    print("🚀 FEDERATED LEARNING - LOCAL TRAINING")
    print("="*80)
    
    # Load data into RAM
    client_data, test_data, input_shape, num_classes = load_all_client_data_to_ram(
        CONFIG["data_dir"], CONFIG["num_clients"]
    )
    
    CONFIG["input_shape"] = input_shape
    CONFIG["num_classes"] = num_classes
    
    print(f"\nConfig: {json.dumps(CONFIG, indent=2, default=str)}")
    
    # Create clients
    clients = []
    for cid in range(CONFIG["num_clients"]):
        c = FederatedClient(
            client_id=cid,
            X_train=client_data[cid]['X_train'],
            y_train=client_data[cid]['y_train']
        )
        clients.append(c)
    
    print(f"\n✓ Created {len(clients)} clients")
    
    # Create server
    server = FederatedServer(clients, test_data, CONFIG)
    
    # Train
    start_time = datetime.now()
    history = train_federated_multigpu(server, CONFIG)
    end_time = datetime.now()
    
    duration = (end_time - start_time).total_seconds()
    print(f"\n✓ Training complete! Duration: {duration:.2f}s ({duration/60:.2f} min)")
    
    # Save model and history
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    algo = CONFIG["algorithm"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_path = os.path.join(CONFIG["output_dir"], f"{algo}_model_{ts}.pth")
    torch.save(server.global_model.state_dict(), model_path)
    print(f"💾 Saved model: {model_path}")
    
    hist_path = os.path.join(CONFIG["output_dir"], f"{algo}_history_{ts}.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"💾 Saved history: {hist_path}")
    
    # Save visualization plots
    save_training_plots(history, CONFIG["output_dir"], prefix=f"{algo}_")
    
    print("\n" + "="*80)
    print("✅ ALL DONE!")
    print("="*80)


if __name__ == "__main__":
    main()
