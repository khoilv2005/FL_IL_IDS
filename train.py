"""
Federated Learning with Multi-GPU Support
==========================================
Main entry point for training.

Usage:
    python main.py
"""

import os
import json
from datetime import datetime

import torch

# Import config từ file cùng thư mục
from config import CONFIG

# Import từ fed_learning package
from fed_learning.data.loader import load_all_client_data_to_ram
from fed_learning.clients.client import FederatedClientMultiGPU
from fed_learning.servers.server import FederatedServerMultiGPU
from fed_learning.training.trainer import train_federated_multigpu
from fed_learning.visualization.plots import save_training_plots


def main():
    print("\n" + "="*80)
    print("🚀 FEDERATED LEARNING WITH MULTI-GPU SUPPORT")
    print("="*80)
    
    # Load data vào RAM
    client_data, test_data, input_shape, num_classes = load_all_client_data_to_ram(
        CONFIG["data_dir"], CONFIG["num_clients"]
    )
    
    CONFIG["input_shape"] = input_shape
    CONFIG["num_classes"] = num_classes
    
    print(f"\nConfig: {json.dumps(CONFIG, indent=2, default=str)}")
    
    # Tạo clients
    clients = []
    for cid in range(CONFIG["num_clients"]):
        c = FederatedClientMultiGPU(
            client_id=cid,
            X_train=client_data[cid]['X_train'],
            y_train=client_data[cid]['y_train']
        )
        clients.append(c)
    
    print(f"\n✓ Created {len(clients)} clients")
    
    # Tạo server
    server = FederatedServerMultiGPU(clients, test_data, CONFIG)
    
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
