"""
Federated Learning - Kaggle Training Script
============================================
Main entry point for training on Kaggle.

Usage:
    Upload fed_learning folder to Kaggle dataset, then run this script.
"""

import os
import sys
import json
import types
import importlib.util
from datetime import datetime

import torch

# =============================================================================
# KAGGLE SETUP - ADVANCED IMPORT FIX
# =============================================================================
MODULE_PATH = "/kaggle/input/ai4fids-fedlearning-modules"

def setup_imports():
    """
    Setup imports handling both nested and flattened dataset structures.
    """
    if not os.path.exists(MODULE_PATH):
        print(f"⚠️ Warning: Module path {MODULE_PATH} not found!")
        return

    # Case 1: Standard structure (fed_learning folder exists inside)
    # /kaggle/input/.../fed_learning/__init__.py
    pkg_path = os.path.join(MODULE_PATH, "fed_learning")
    if os.path.exists(pkg_path):
        print(f"📦 Found standard package structure at {pkg_path}")
        if MODULE_PATH not in sys.path:
            sys.path.insert(0, MODULE_PATH)
        return

    # Case 2: Flattened structure (dataset IS the package)
    # /kaggle/input/.../__init__.py (this is fed_learning's init)
    init_path = os.path.join(MODULE_PATH, "__init__.py")
    if os.path.exists(init_path):
        print(f"📦 Found flattened package structure at {MODULE_PATH}")
        
        # Manually create the package 'fed_learning'
        # pointing to the flattened directory
        spec = importlib.util.spec_from_file_location("fed_learning", init_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules["fed_learning"] = module
            
            # Add MODULE_PATH to sys.path so submodules can be found relatively?
            # No, relative imports rely on __package__.
            # We need to ensure 'fed_learning' is treated as a package.
            
            # Execute the module
            spec.loader.exec_module(module)
            
            # Hack: Add submodule paths to sys.modules manually if needed?
            # Better trick: Add MODULE_PATH to sys.path, import contents?
            # No, correct way for flattened structure is difficult with relative imports.
            
            # BEST FIX for flattened structure: 
            # Create a symlink 'fed_learning' -> MODULE_PATH inside /tmp
            try:
                tmp_dir = "/tmp/fed_pkg_fix"
                os.makedirs(tmp_dir, exist_ok=True)
                symlink_path = os.path.join(tmp_dir, "fed_learning")
                
                if os.path.exists(symlink_path):
                    os.remove(symlink_path)
                    
                os.symlink(MODULE_PATH, symlink_path)
                
                if tmp_dir not in sys.path:
                    sys.path.insert(0, tmp_dir)
                    
                print(f"🔗 Created symlink {symlink_path} -> {MODULE_PATH}")
            except Exception as e:
                print(f"⚠️ Failed to create symlink: {e}")
                
        return

    print("❌ Could not detect valid package structure!")

# Run setup
setup_imports()

# Import
try:
    from fed_learning import (
        load_all_client_data_to_ram,
        FederatedClient,
        FederatedServer,
        train_federated_multigpu,
    )
    from fed_learning.visualization.plots import save_training_plots
    print("✓ Imports ready!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


# =============================================================================
# CONFIG - Sửa các tham số ở đây trước khi train
# =============================================================================
CONFIG = {
    # ===================
    # DATA PATHS
    # ===================
    "data_dir": "/kaggle/input/data-100clients",
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
    "num_rounds": 15,
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
    print("🚀 FEDERATED LEARNING - KAGGLE TRAINING")
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
