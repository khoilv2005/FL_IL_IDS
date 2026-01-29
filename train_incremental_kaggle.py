"""
Federated Class Incremental Learning - Unified Training Script
==============================================================
Unified entry point for CGoFed, EWC, FedLwF, and other strategies.

Usage:
    Switch algorithm in CONFIG["algorithm"]:
    - "cgofed": Constrained Gradient Optimization
    - "fedavg_ewc" / "fedprox_ewc": Elastic Weight Consolidation
    - "fedavg_lwf" / "fedprox_lwf": Learning without Forgetting
    - "fedcbdr": Class-Balancing Data Replay

    Upload fed_learning folder to Kaggle dataset, then run this script.
"""

import os
import sys
import gc
import json
import shutil
from datetime import datetime
from typing import Dict, List

import torch
from torch.utils.data import TensorDataset, DataLoader


# =============================================================================
# KAGGLE SETUP
# =============================================================================
MODULE_PATH = "/kaggle/input/ai4fids-fedlearning-modules"

def setup_imports():
    """Setup imports for both nested and flattened dataset structures."""
    if not os.path.exists(MODULE_PATH):
        print(f"⚠️ Warning: Module path {MODULE_PATH} not found!")
        return

    # Case 1: Standard structure
    pkg_path = os.path.join(MODULE_PATH, "fed_learning")
    if os.path.exists(pkg_path):
        print(f"📦 Found standard package structure at {pkg_path}")
        if MODULE_PATH not in sys.path:
            sys.path.insert(0, MODULE_PATH)
        return

    # Case 2: Flattened structure - create symlink
    init_path = os.path.join(MODULE_PATH, "__init__.py")
    if os.path.exists(init_path):
        print(f"📦 Found flattened package structure at {MODULE_PATH}")
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

setup_imports()

# Import fed_learning modules
try:
    from fed_learning import train_federated_multigpu
    from fed_learning.servers import IncrementalServer
    from fed_learning.clients import CGoFedClient
    from fed_learning.data.incremental_loader import IncrementalDataLoader
    from fed_learning.strategies import get_strategy
    # Import specific implementation classes for client creation and checks
    from fed_learning.clients.fedcbdr_client import FedCBDRClient
    from fed_learning.clients.fedlwf_client import FedLwFClient
    
    # Updated imports for Servers
    from fed_learning.servers import (
        IncrementalServer, 
        FedCBDRServer, 
        FedLwFServer
    )
    
    from fed_learning.strategies.incremental.fedlwf import FedLwFTrainer
    from fed_learning.strategies.incremental.ewc import EWCMixin
    print("✓ Imports ready!")
except ImportError as e:
    print(f"❌ Import failed (some optional modules might be missing): {e}")


# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # Data
    "data_dir": "/kaggle/input/data-10clients",
    
    # Algorithm Selection
    # Options: "cgofed", "fedavg_ewc", "fedprox_ewc", "fedavg_lwf", "fedprox_lwf", "fedcbdr"
    "algorithm": "fedprox_ewc",
    
    # Output
    "output_dir": "./results_incremental",
    
    # Incremental Learning - 5 Tasks Distribution
    "num_clients": 10,
    "total_classes": 34,
    "base_classes": 10,
    "classes_per_task": 6,
    
    # Common Parameters
    "mu": 1.5,                    # Proximal term (FedProx/CGoFed)
    "rounds_per_task": 5,
    "local_epochs": 5,
    "learning_rate": 0.001,
    "batch_size": 1024,
    "eval_every": 1,
    
    # --- Algorithm Specific Params ---
    
    # CGoFed
    "lambda_decay": 0.05,
    "theta_threshold": 0.01,
    "cross_task_weight": 0.35,
    "energy_threshold": 0.97,
    "num_samples_rep": 1536,
    "top_k": 2,
    
    # EWC
    "ewc_lambda": 1000.0,
    "fisher_samples": 200,
    "online_ewc": False,
    
    # LwF (FedLwF)
    "lwf_alpha": 1.0,
    "temperature": 2.0,
    "lwf_alpha_scale": 1.0, # Decay/Growth factor for alpha
    "distill_on_new_only": False,
    
    # FedCBDR
    "tau_old": 0.9,
    "tau_new": 1.1,
    "omega_old": 1.1,
    "omega_new": 0.9,
    "buffer_size": 500,
    "replay_ratio": 0.5,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def cleanup_temp_folders():
    """Clean up temporary folders."""
    folders = [
        "./temp_svd_storage", 
        "./temp_ewc_storage", 
        "./temp_fedlwf_storage", 
        "./temp_test_data"
    ]
    for folder in folders:
        if os.path.exists(folder):
            print(f"🧹 Cleaning {folder}...")
            shutil.rmtree(folder)

def create_clients(client_data, config, task_id, new_classes):
    """Factory to create clients based on algorithm."""
    algo = config["algorithm"].lower()
    clients = []
    
    client_ids = sorted(client_data.keys())
    
    for cid in client_ids:
        data = client_data[cid]
        X, y = data["X_train"], data["y_train"]
        
        if algo == "fedcbdr":
            # FedCBDR Client - Needs persistence for Replay Buffer
            # Check if client already exists in a global map (simulated here via kwargs or global var)
            # In this script, we'll create a new client but initialize it. 
            # ideally, the main loop should maintain `all_clients` map.
            # Here we just instantiate. The main loop must handle persistence.
            client = FedCBDRClient(
                cid, X, y, 
                buffer_size=config.get("buffer_size", 500),
                leverage_rank=config.get("leverage_rank", 50)
            )
            clients.append(client)
            
        elif algo in ["fedavg_lwf", "fedprox_lwf"]:
             clients.append(FedLwFClient(cid, X, y))
             
        else:
            # Standard/CGoFed Client
            clients.append(CGoFedClient(cid, X, y))
            
    return clients

def get_algorithm_specific_components(config, clients, test_data, task_config):
    """Factory for Server and any logic hooks."""
    algo = config["algorithm"].lower()
    
    if algo == "fedcbdr":
        return FedCBDRServer(clients, test_data, task_config)
    elif algo in ["fedavg_lwf", "fedprox_lwf"]:
        return FedLwFServer(clients, test_data, task_config)
    else:
        return IncrementalServer(clients, test_data, task_config)

def post_task_processing(trainer, server, client_data, config, participating_clients=None):
    """Handle post-task logic (Fisher, Snapshot, SVD, Buffer Update)."""
    algo = config["algorithm"].lower()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. CGoFed: Build Representation Space
    if algo == "cgofed":
        if hasattr(trainer, 'build_space_from_client_data'):
            trainer.build_space_from_client_data(
                model=server.global_model,
                client_data=client_data,
                config=config,
                device=device
            )
            
    # 2. EWC: Consolidate (Compute Fisher)
    elif "ewc" in algo: # fedavg_ewc, fedprox_ewc
        if hasattr(trainer, 'consolidate'):
            print(f"\n🔐 Computing Fisher Information for EWC...")
            # Aggregate data for Fisher
            all_X, all_y = [], []
            for data in client_data.values():
                if len(data.get("y_train", [])) > 0:
                    all_X.append(data["X_train"])
                    all_y.append(data["y_train"])
            
            if all_X:
                X = torch.cat(all_X)
                y = torch.cat(all_y)
                # Limit samples
                if len(y) > config.get("fisher_samples", 200):
                    idx = torch.randperm(len(y))[:config["fisher_samples"]]
                    X, y = X[idx], y[idx]
                
                loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
                trainer.consolidate(server.global_model, loader, device)

    # 3. LwF: Save Snapshot - Check for FedLwFTrainer explicitly
    elif isinstance(trainer, FedLwFTrainer) or "lwf" in algo:
        print(f"\n📸 Saving model snapshot for LwF...")
        if hasattr(trainer, 'save_model_snapshot'):
            trainer.save_model_snapshot(server.global_model)
        elif hasattr(server, 'save_global_snapshot'):
            server.save_global_snapshot()

    # 4. FedCBDR: GDR Update
    if algo == "fedcbdr" and hasattr(server, 'coordinate_gdr'):
        print(f"\n🔄 Updating Replay Buffers (GDR)...")
        server.coordinate_gdr(participating_clients, verbose=True)


# =============================================================================
# MAIN
# =============================================================================
def main():
    """Main entry point - routes to appropriate algorithm."""
    algo = CONFIG["algorithm"].lower()
    
    print("\n" + "="*80)
    print(f"🎯 Selected Algorithm: {algo.upper()}")
    print("="*80)
    
    if algo == "fedcbdr":
        main_fedcbdr()
    elif algo in ["fedavg_lwf", "fedprox_lwf"]:
        main_fedlwf()
    else:
        main_cgofed()  # Default: CGoFed or EWC


def main_cgofed():
    """Main training for CGoFed/EWC algorithms."""
    # Setup Output
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{CONFIG['output_dir']}_{CONFIG['algorithm']}_{ts}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Config
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(CONFIG, f, indent=2, default=str)

    print("\n" + "="*80)
    print(f"🚀 FEDERATED CLASS INCREMENTAL LEARNING - {CONFIG['algorithm'].upper()}")
    print("="*80)
    
    # Cleanup temp folders
    cleanup_temp_folders()
    os.makedirs("./temp_test_data", exist_ok=True)
    
    # Load Data
    data_loader = IncrementalDataLoader(data_dir=CONFIG["data_dir"])
    print(f"\n{data_loader}")
    
    # Update Config
    CONFIG["input_shape"] = data_loader.input_shape
    CONFIG["num_classes"] = CONFIG["total_classes"]
    
    # Get Strategy
    trainer, aggregator = get_strategy(**CONFIG)
    print(f"✓ Trainer: {trainer.__class__.__name__}")
    print(f"✓ Aggregator: {aggregator.__class__.__name__}")
    
    # State Variables
    all_history = {"task_accuracies": [], "task_forgetting": []}
    all_test_data = {}
    best_acc_per_task = {}
    server = None
    
    # Task Loop
    for task_id in range(data_loader.get_num_tasks()):
        print(f"\n{'='*80}\n📚 TASK {task_id}/{data_loader.get_num_tasks()}\n{'='*80}")
        
        # Get Data
        new_classes = data_loader.get_task_classes(task_id)
        seen_classes = []
        for t in range(task_id + 1):
            seen_classes.extend(data_loader.get_task_classes(t))
        
        # Client Data
        client_data = {}
        for cid in data_loader.get_all_client_ids():
            X, y = data_loader.get_client_data(cid, task_id)
            if len(y) > 0:
                client_data[cid] = {"X_train": X, "y_train": y}
        
        print(f"  Clients with data: {len(client_data)}")
        if not client_data:
            continue
        
        # Test Data
        test_X, test_y = data_loader.get_test_data(task_id, cumulative=True)
        test_data = {"X_test": test_X, "y_test": test_y}
        test_data_path = f"./temp_test_data/test_task_{task_id}.pt"
        torch.save(test_data, test_data_path)
        all_test_data[task_id] = test_data_path
        
        # Create Clients
        clients = create_clients(client_data, CONFIG, task_id, new_classes)
        
        # Initialize Server
        if server is None:
            task_config = {
                **CONFIG,
                "num_rounds": CONFIG["rounds_per_task"],
            }
            server = get_algorithm_specific_components(CONFIG, clients, test_data, task_config)
        
        server.set_task(task_id, new_classes)
        
        # Training Rounds
        print(f"\n🏋️ Training for {CONFIG['rounds_per_task']} rounds...")
        for round_idx in range(CONFIG["rounds_per_task"]):
            print(f"--- Round {round_idx + 1}/{CONFIG['rounds_per_task']} ---")
            server.train_round(participating_clients=clients, verbose=True)
            
            if (round_idx + 1) % CONFIG["eval_every"] == 0:
                metrics = server.evaluate_global(seen_classes_only=True)
                print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
        
        # Post-task Processing
        post_task_processing(trainer, server, client_data, CONFIG, clients)
        
        # Forgetting Calculation
        task_accuracies = {}
        for prev_tid, path in all_test_data.items():
            loaded_test = torch.load(path)
            server.test_data = loaded_test
            m = server.evaluate_global(seen_classes_only=False)
            task_accuracies[prev_tid] = m['accuracy']
            best_acc_per_task[prev_tid] = max(best_acc_per_task.get(prev_tid, 0), m['accuracy'])
        
        # Average Forgetting
        af = 0.0
        if task_id > 0:
            diffs = [max(0, best_acc_per_task[t] - task_accuracies[t]) for t in range(task_id)]
            af = sum(diffs) / len(diffs) if diffs else 0.0
        
        # Record
        final_metrics = server.evaluate_global(seen_classes_only=True)
        all_history["task_accuracies"].append({
            "task": task_id,
            "seen_classes": len(seen_classes),
            "accuracy": final_metrics["accuracy"],
            "avg_forgetting": af,
        })
        
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Summary
    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETE")
    print("="*80)
    for h in all_history["task_accuracies"]:
        af_str = f", AF: {h['avg_forgetting']*100:.2f}%" if h['avg_forgetting'] > 0 else ""
        print(f"  Task {h['task']}: {h['seen_classes']} classes → Acc: {h['accuracy']*100:.2f}%{af_str}")
    
    if all_history["task_accuracies"]:
        final = all_history["task_accuracies"][-1]
        print(f"\n  🎯 Final Accuracy: {final['accuracy']*100:.2f}%")
        print(f"  🎯 Final Average Forgetting: {final['avg_forgetting']*100:.2f}%")
    
    # Save Results
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(all_history, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
    
    print(f"\n💾 Results saved to: {output_dir}")
    print("✅ DONE!")


# =============================================================================
# FEDCBDR TRAINING FUNCTIONS
# =============================================================================

CONFIG_FEDCBDR = {
    # Data
    "data_dir": "/kaggle/input/data-10clients",
    "output_dir": "./results_fedcbdr",
    
    # Incremental Learning - 5 Tasks Distribution
    "num_clients": 10,
    "total_classes": 34,
    "base_classes": 10,         # Task 0: 10 classes
    "classes_per_task": 6,      # Task 1-4: +6 classes per task (10+6*4=34)
    
    # FedCBDR Algorithm
    "algorithm": "fedcbdr",
    
    # Task-aware Temperature Scaling (TTS) - Paper recommended
    "tau_old": 0.9,             # Temperature for old task logits
    "tau_new": 1.1,             # Temperature for new task logits
    "omega_old": 1.1,           # Weight for old task samples
    "omega_new": 0.9,           # Weight for new task samples
    
    # Replay Buffer
    "buffer_size": 500,         # Max samples per client in replay buffer
    "leverage_rank": 50,        # Rank for SVD in leverage score computation
    "use_replay": True,
    "replay_ratio": 0.5,
    "use_herding": True,
    
    # Training
    "rounds_per_task": 5,
    "local_epochs": 3,
    "learning_rate": 0.001,
    "batch_size": 128,
    
    # Eval
    "eval_every": 1,
}


def create_fedcbdr_clients(client_data, config):
    """Create FedCBDR clients from client data."""
    from fed_learning.clients.fedcbdr_client import FedCBDRClient
    
    clients = []
    
    for cid, data in client_data.items():
        if len(data.get("y_train", [])) > 0:
            client = FedCBDRClient(
                client_id=cid,
                X_train=data["X_train"],
                y_train=data["y_train"],
                buffer_size=config.get("buffer_size", 500),
            )
            clients.append(client)
    
    return clients


def compute_per_task_accuracy_fedcbdr(server, all_test_data):
    """
    Compute accuracy for each previous task (for AF calculation) - FedCBDR version.
    
    Uses server's evaluate_per_task() method which handles task_classes internally.
    For more control, can load test data from disk like CGoFed version.
    """
    task_accuracies = {}
    
    for prev_tid, prev_test_path in all_test_data.items():
        try:
            loaded_test_data = torch.load(prev_test_path)
        except Exception as e:
            print(f"⚠️ Failed to load test data for task {prev_tid}: {e}")
            continue
        
        # Temporarily set test data
        original_test_data = server.test_data
        server.test_data = loaded_test_data
        
        # Evaluate
        metrics = server.evaluate_global(seen_classes_only=False)
        task_accuracies[prev_tid] = metrics['accuracy']
        print(f"    Task {prev_tid} Acc: {metrics['accuracy']*100:.2f}%")
        
        # Restore and cleanup
        server.test_data = original_test_data
        del loaded_test_data
        gc.collect()
    
    return task_accuracies


def compute_average_forgetting_fedcbdr(server, task_accuracies, best_acc_per_task, task_id):
    """
    Compute Average Forgetting (AF) metric for FedCBDR.
    
    AF = (1/T-1) * Σ_{t=0}^{T-2} max_{t'≤T-1} (a_{t',t} - a_{T-1,t})
    """
    current_af = 0.0
    
    # Update best accuracies
    for tid, acc in task_accuracies.items():
        if tid not in best_acc_per_task:
            best_acc_per_task[tid] = acc
        else:
            best_acc_per_task[tid] = max(best_acc_per_task[tid], acc)
    
    # Calculate forgetting for previous tasks
    if task_id > 0:
        forgetting = []
        for tid in range(task_id):
            if tid in best_acc_per_task and tid in task_accuracies:
                f = best_acc_per_task[tid] - task_accuracies[tid]
                forgetting.append(max(0, f))
        if forgetting:
            current_af = sum(forgetting) / len(forgetting)
    
    return current_af


def save_checkpoint_fedcbdr(server, config, task_id, seen_classes, output_dir):
    """Save training checkpoint for FedCBDR after each task."""
    checkpoint_path = os.path.join(output_dir, f"checkpoint_task_{task_id}.pt")
    
    checkpoint = {
        'task_id': task_id,
        'model_state_dict': server.global_model.state_dict(),
        'seen_classes': list(seen_classes),
        'task_classes': server.task_classes,
        'config': config,
        'history': server.history,
        'trainer_state': {
            'tau_old': server.trainer.tau_old,
            'tau_new': server.trainer.tau_new,
            'omega_old': server.trainer.omega_old,
            'omega_new': server.trainer.omega_new,
        }
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 FedCBDR Checkpoint saved: {checkpoint_path}")


def print_replay_buffer_stats(clients):
    """Print replay buffer statistics for FedCBDR clients."""
    total_buffer = 0
    classes_in_buffer = set()
    
    for client in clients:
        if hasattr(client, 'replay_buffer'):
            total_buffer += client.replay_buffer.total_samples
            classes_in_buffer.update(client.replay_buffer.class_buffers.keys())
    
    print(f"   📦 Replay Buffer Statistics:")
    print(f"      Total samples: {total_buffer}")
    print(f"      Classes covered: {len(classes_in_buffer)}")
    return total_buffer, len(classes_in_buffer)


def main_fedcbdr():
    """Main training function for FedCBDR."""
    from fed_learning.servers.incremental_server import FedCBDRServer
    from fed_learning.clients.fedcbdr_client import FedCBDRClient
    
    print("\n" + "="*80)
    print("🚀 FEDCBDR - Class-wise Balancing Data Replay for FCIL")
    print("="*80)
    print(f"   τ_old={CONFIG_FEDCBDR['tau_old']}, τ_new={CONFIG_FEDCBDR['tau_new']}")
    print(f"   ω_old={CONFIG_FEDCBDR['omega_old']}, ω_new={CONFIG_FEDCBDR['omega_new']}")
    print(f"   Buffer size: {CONFIG_FEDCBDR['buffer_size']}")
    print(f"   Replay ratio: {CONFIG_FEDCBDR['replay_ratio']}")
    
    # Cleanup
    cleanup_temp_folders()
    os.makedirs("./temp_test_data", exist_ok=True)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{CONFIG_FEDCBDR['output_dir']}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(CONFIG_FEDCBDR, f, indent=2, default=str)

    # Load Data
    data_loader = IncrementalDataLoader(data_dir=CONFIG_FEDCBDR["data_dir"])
    print(f"\n{data_loader}")
    
    # Update Config with Data Params
    CONFIG_FEDCBDR["input_shape"] = data_loader.input_shape
    CONFIG_FEDCBDR["num_classes"] = CONFIG_FEDCBDR["total_classes"]

    # Note: FedCBDRServer creates its own FedCBDRTrainer internally,
    # so we don't need to call get_strategy() here.
    print(f"✓ Using FedCBDRServer with built-in FedCBDRTrainer")

    # State Variables
    server = None  # Initialize server as None
    input_shape = data_loader.input_shape  # Get input shape from data loader
    all_history = {"task_accuracies": [], "task_forgetting": []}
    all_test_data = {}
    best_acc_per_task = {}
    
    # Persistent Clients (needed for FedCBDR/LwF state)
    all_clients: Dict[int, object] = {}  # Renamed from persistent_clients

    # --- Task Loop ---
    for task_id in range(data_loader.get_num_tasks()):
        print(f"\n{'='*80}\n📚 TASK {task_id}/{data_loader.get_num_tasks()}\n{'='*80}")
        
        # 1. Prepare Data
        new_classes = data_loader.get_task_classes(task_id)
        seen_classes = []
        for t in range(task_id + 1):
            seen_classes.extend(data_loader.get_task_classes(t))
            
        # Get Client Data
        client_data_map = {}
        for cid in data_loader.get_all_client_ids():
            X, y = data_loader.get_client_data(cid, task_id)
            if len(y) > 0:
                client_data_map[cid] = {"X_train": X, "y_train": y}
        
        print(f"  Clients with data: {len(client_data_map)}")
        if not client_data_map:
            print("  ⚠️ No data for this task, skipping.")
            continue

        # 2. Prepare Test Data
        test_X, test_y = data_loader.get_test_data(task_id, cumulative=True)
        test_data = {"X_test": test_X, "y_test": test_y}
        test_data_path = os.path.join("./temp_test_data", f"test_task_{task_id}.pt")
        torch.save(test_data, test_data_path)
        all_test_data[task_id] = test_data_path
        
        # Skip if no training data
        if all(len(cd.get("y_train", [])) == 0 for cd in client_data_map.values()):
            print(f"⚠️ No training data for task {task_id}, skipping...")
            continue
        
        # Create/update clients with task info
        for cid, data in client_data_map.items():
            if cid not in all_clients:
                all_clients[cid] = FedCBDRClient(
                    client_id=cid,
                    X_train=data["X_train"],
                    y_train=data["y_train"],
                    buffer_size=CONFIG_FEDCBDR.get("buffer_size", 500),
                )
                # Set task info for new client
                all_clients[cid].set_task_data(
                    X_train=data["X_train"],
                    y_train=data["y_train"],
                    task_id=task_id,
                    task_classes=new_classes,
                )
            else:
                # Update existing client with new task data
                all_clients[cid].set_task_data(
                    X_train=data["X_train"],
                    y_train=data["y_train"],
                    task_id=task_id,
                    task_classes=new_classes,
                )
        
        participating_clients = [
            all_clients[cid] for cid in client_data_map.keys()
        ]
        
        # Initialize or update server
        if server is None:
            task_config = {
                **CONFIG_FEDCBDR,
                "num_rounds": CONFIG_FEDCBDR["rounds_per_task"],
                "input_shape": input_shape,
                "num_classes": CONFIG_FEDCBDR["total_classes"],
            }
            server = FedCBDRServer(participating_clients, test_data, task_config)
        
        # Set task for server
        server.set_task(task_id, new_classes)
        
        # ==================================================================
        # TRAINING ROUNDS
        # ==================================================================
        print(f"\n🏋️ Training Task {task_id} for {CONFIG_FEDCBDR['rounds_per_task']} rounds...")
        
        for round_idx in range(CONFIG_FEDCBDR["rounds_per_task"]):
            print(f"\n--- Round {round_idx + 1}/{CONFIG_FEDCBDR['rounds_per_task']} ---")
            
            # Train round
            round_result = server.train_round(
                participating_clients=participating_clients,
                verbose=True
            )
            
            # Evaluate
            if (round_idx + 1) % CONFIG_FEDCBDR["eval_every"] == 0:
                metrics = server.evaluate_global(seen_classes_only=True)
                print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
                print(f"  F1 (macro): {metrics['f1_macro']*100:.2f}%")
        
        # ==================================================================
        # POST-TASK: Update Replay Buffers via GDR
        # ==================================================================
        print(f"\n🔄 Updating replay buffers for Task {task_id}...")
        server.coordinate_gdr(
            participating_clients=participating_clients,
            verbose=True
        )
        
        # ==================================================================
        # TASK EVALUATION
        # ==================================================================
        print(f"\n📊 Final evaluation for Task {task_id}...")
        
        # Compute per-task accuracy using helper function
        print("\n🔍 Computing Per-Task Accuracy for AF...")
        task_accs = compute_per_task_accuracy_fedcbdr(server, all_test_data)
        
        # Compute Average Forgetting using helper function
        current_af = compute_average_forgetting_fedcbdr(
            server, task_accs, best_acc_per_task, task_id
        )
        
        if task_id > 0:
            all_history["task_forgetting"].append(current_af)
            print(f"   Average Forgetting (AF): {current_af*100:.2f}%")
        
        # Overall metrics
        final_metrics = server.evaluate_global(
            seen_classes_only=True,
            compute_auc=True
        )
        print(f"\n   📈 Overall (seen classes):")
        print(f"      Accuracy: {final_metrics['accuracy']*100:.2f}%")
        print(f"      F1 (macro): {final_metrics['f1_macro']*100:.2f}%")
        
        # Print replay buffer stats using helper function
        print_replay_buffer_stats(list(all_clients.values()))
        
        # Save checkpoint using helper function
        save_checkpoint_fedcbdr(server, CONFIG_FEDCBDR, task_id, seen_classes, output_dir)
        
        # Record history
        all_history["task_accuracies"].append({
            "task": task_id,
            "seen_classes": len(seen_classes),
            "accuracy": final_metrics["accuracy"],
            "f1_macro": final_metrics["f1_macro"],
            "per_task_acc": task_accs,
            "avg_forgetting": current_af,
        })
        
        # Cleanup
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    print("\n" + "="*80)
    print("🎉 FEDCBDR TRAINING COMPLETE")
    print("="*80)
    
    # Final evaluation
    print("\n📊 Final Evaluation (all classes):")
    final_metrics = server.evaluate_global(
        seen_classes_only=False,
        compute_auc=True
    )
    print(f"   Accuracy:    {final_metrics['accuracy']*100:.2f}%")
    print(f"   F1 (macro):  {final_metrics['f1_macro']*100:.2f}%")
    print(f"   F1 (weight): {final_metrics['f1_weighted']*100:.2f}%")
    
    # Forgetting summary
    if all_history["task_forgetting"]:
        print(f"\n📉 Forgetting Analysis:")
        print(f"   Final AF: {all_history['task_forgetting'][-1]*100:.2f}%")
    
    # Save results
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(all_history, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
    
    print(f"\n💾 Results saved to: {output_dir}")
    print("="*80)


# =============================================================================
# FEDLWF TRAINING FUNCTIONS
# =============================================================================

CONFIG_FEDLWF = {
    # Data
    "data_dir": "/kaggle/input/data-10clients",
    "output_dir": "./results_fedlwf",
    
    # Incremental Learning - 5 Tasks Distribution
    "num_clients": 10,
    "total_classes": 34,
    "base_classes": 10,
    "classes_per_task": 6,
    
    # FedLwF Algorithm
    "algorithm": "fedlwf",
    
    # Knowledge Distillation settings
    "lwf_alpha": 1.0,           # Base KD weight
    "lwf_alpha_scale": 1.5,     # Scale factor per task
    "temperature": 2.0,         # Temperature for soft targets
    "distill_on_new_only": False,
    
    # Training
    "rounds_per_task": 5,
    "local_epochs": 3,
    "learning_rate": 0.001,
    "batch_size": 128,
    
    # Eval
    "eval_every": 1,
}


def create_fedlwf_clients(client_data, config):
    """Create FedLwF clients from client data."""
    from fed_learning.clients.fedlwf_client import FedLwFClient
    
    clients = []
    
    for cid, data in client_data.items():
        if len(data.get("y_train", [])) > 0:
            client = FedLwFClient(
                client_id=cid,
                X_train=data["X_train"],
                y_train=data["y_train"],
            )
            clients.append(client)
    
    return clients


def compute_per_task_accuracy_fedlwf(server, all_test_data):
    """
    Compute accuracy for each previous task (for AF calculation) - FedLwF version.
    
    Uses the same approach as FedCBDR for consistency.
    """
    task_accuracies = {}
    
    for prev_tid, prev_test_path in all_test_data.items():
        try:
            loaded_test_data = torch.load(prev_test_path)
        except Exception as e:
            print(f"⚠️ Failed to load test data for task {prev_tid}: {e}")
            continue
        
        # Temporarily set test data
        original_test_data = server.test_data
        server.test_data = loaded_test_data
        
        # Evaluate
        metrics = server.evaluate_global(seen_classes_only=False)
        task_accuracies[prev_tid] = metrics['accuracy']
        print(f"    Task {prev_tid} Acc: {metrics['accuracy']*100:.2f}%")
        
        # Restore and cleanup
        server.test_data = original_test_data
        del loaded_test_data
        gc.collect()
    
    return task_accuracies


def compute_average_forgetting_fedlwf(server, task_accuracies, best_acc_per_task, task_id):
    """
    Compute Average Forgetting (AF) metric for FedLwF.
    
    AF = (1/T-1) * Σ_{t=0}^{T-2} (best_acc[t] - current_acc[t])
    """
    current_af = 0.0
    
    # Update best accuracies
    for tid, acc in task_accuracies.items():
        if tid not in best_acc_per_task:
            best_acc_per_task[tid] = acc
        else:
            best_acc_per_task[tid] = max(best_acc_per_task[tid], acc)
    
    # Calculate forgetting for previous tasks
    if task_id > 0:
        forgetting = []
        for tid in range(task_id):
            if tid in best_acc_per_task and tid in task_accuracies:
                f = best_acc_per_task[tid] - task_accuracies[tid]
                forgetting.append(max(0, f))
        if forgetting:
            current_af = sum(forgetting) / len(forgetting)
    
    return current_af


def save_checkpoint_fedlwf(server, config, task_id, seen_classes, output_dir, current_alpha):
    """Save training checkpoint for FedLwF after each task."""
    checkpoint_path = os.path.join(output_dir, f"checkpoint_task_{task_id}.pt")
    
    checkpoint = {
        'task_id': task_id,
        'model_state_dict': server.global_model.state_dict(),
        'seen_classes': list(seen_classes),
        'task_classes': server.task_classes,
        'config': config,
        'history': server.history,
        'trainer_state': {
            'lwf_alpha': server.trainer.lwf_alpha,
            'temperature': server.trainer.temperature,
            'current_alpha': current_alpha,
        }
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 FedLwF Checkpoint saved: {checkpoint_path}")


def get_kd_alpha_for_task(config, task_id):
    """
    Calculate Knowledge Distillation alpha for a given task.
    
    α_t = α_base * (α_scale)^(t-1) for t > 0
    α_0 = α_base (no KD for first task)
    """
    if task_id == 0:
        return config["lwf_alpha"]
    
    return config["lwf_alpha"] * (config["lwf_alpha_scale"] ** (task_id - 1))


def print_kd_info(task_id, current_alpha, temperature):
    """Print Knowledge Distillation information for FedLwF."""
    print(f"   🎓 Knowledge Distillation Settings:")
    print(f"      α (KD weight): {current_alpha:.3f}")
    print(f"      Temperature: {temperature}")
    if task_id == 0:
        print(f"      Note: First task - no KD applied (no old knowledge)")
    else:
        print(f"      KD active: Preserving knowledge from {task_id} previous task(s)")


def main_fedlwf():
    """Main training function for FedLwF."""
    from fed_learning.servers.incremental_server import FedLwFServer
    from fed_learning.clients.fedlwf_client import FedLwFClient
    
    print("\n" + "="*80)
    print("🚀 FEDLWF - Federated Learning without Forgetting")
    print("="*80)
    print(f"   α (KD weight): {CONFIG_FEDLWF['lwf_alpha']}")
    print(f"   Temperature: {CONFIG_FEDLWF['temperature']}")
    print(f"   α scale per task: {CONFIG_FEDLWF['lwf_alpha_scale']}")
    
    # Cleanup
    cleanup_temp_folders()
    os.makedirs("./temp_test_data", exist_ok=True)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{CONFIG_FEDLWF['output_dir']}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(CONFIG_FEDLWF, f, indent=2, default=str)
    
    # Initialize data loader
    print(f"\n📂 Loading data from: {CONFIG_FEDLWF['data_dir']}")
    data_loader = IncrementalDataLoader(
        data_dir=CONFIG_FEDLWF["data_dir"]
    )
    
    print(f"\n{data_loader}")
    
    # Get input shape
    input_shape = data_loader.input_shape
    CONFIG_FEDLWF["input_shape"] = input_shape
    CONFIG_FEDLWF["num_classes"] = CONFIG_FEDLWF["total_classes"]
    
    # Initialize tracking variables
    all_history = {"task_accuracies": [], "task_forgetting": []}
    all_clients = {}
    all_test_data = {}
    best_acc_per_task = {}  # Track best accuracy per task for AF calculation
    server = None
    current_alpha = CONFIG_FEDLWF["lwf_alpha"]
    
    # ==========================================================================
    # TASK LOOP
    # ==========================================================================
    for task_id in range(data_loader.get_num_tasks()):
        print(f"\n{'='*80}")
        print(f"📚 TASK {task_id}/{data_loader.get_num_tasks()}")
        print(f"{'='*80}")
        
        # Adjust alpha for current task using helper function
        current_alpha = get_kd_alpha_for_task(CONFIG_FEDLWF, task_id)
        
        # Print KD info using helper function
        print_kd_info(task_id, current_alpha, CONFIG_FEDLWF["temperature"])
        
        # Get data for this task
        new_classes = data_loader.get_task_classes(task_id)
        
        # Derive seen classes
        seen_classes = []
        for t in range(task_id + 1):
            seen_classes.extend(data_loader.get_task_classes(t))
        
        # Load client data for this task
        client_data = {}
        client_ids = data_loader.get_all_client_ids()
        print(f"  Loading data for {len(client_ids)} clients...")
        
        for cid in client_ids:
            X, y = data_loader.get_client_data(cid, task_id)
            if len(y) > 0:
                client_data[cid] = {"X_train": X, "y_train": y}
        
        print(f"  Clients with data for task {task_id}: {len(client_data)}")
        
        # Load global test data (cumulative)
        test_X, test_y = data_loader.get_test_data(task_id, cumulative=True)
        test_data = {"X_test": test_X, "y_test": test_y}
        
        # Save test data to disk
        test_data_path = os.path.join("./temp_test_data", f"test_task_{task_id}.pt")
        torch.save(test_data, test_data_path)
        all_test_data[task_id] = test_data_path
        
        # Skip if no training data
        if all(len(cd.get("y_train", [])) == 0 for cd in client_data.values()):
            print(f"⚠️ No training data for task {task_id}, skipping...")
            continue
        
        # Create/update clients with task info
        for cid, data in client_data.items():
            if cid not in all_clients:
                all_clients[cid] = FedLwFClient(
                    client_id=cid,
                    X_train=data["X_train"],
                    y_train=data["y_train"],
                )
                # Set task info for new client
                all_clients[cid].set_task_data(
                    X_train=data["X_train"],
                    y_train=data["y_train"],
                    task_id=task_id,
                    task_classes=new_classes,
                )
            else:
                # Update existing client with new task data
                all_clients[cid].set_task_data(
                    X_train=data["X_train"],
                    y_train=data["y_train"],
                    task_id=task_id,
                    task_classes=new_classes,
                )
        
        participating_clients = [
            all_clients[cid] for cid in client_data.keys()
        ]
        
        # Initialize or update server
        if server is None:
            task_config = {
                **CONFIG_FEDLWF,
                "num_rounds": CONFIG_FEDLWF["rounds_per_task"],
                "input_shape": input_shape,
                "num_classes": CONFIG_FEDLWF["total_classes"],
                "lwf_alpha": current_alpha,
            }
            server = FedLwFServer(participating_clients, test_data, task_config)
        else:
            # Update alpha in trainer
            server.trainer.lwf_alpha = current_alpha
        
        # Set task for server
        server.set_task(task_id, new_classes)
        
        # ==================================================================
        # TRAINING ROUNDS
        # ==================================================================
        print(f"\n🏋️ Training Task {task_id} for {CONFIG_FEDLWF['rounds_per_task']} rounds...")
        
        for round_idx in range(CONFIG_FEDLWF["rounds_per_task"]):
            print(f"\n--- Round {round_idx + 1}/{CONFIG_FEDLWF['rounds_per_task']} ---")
            
            # Train round
            round_result = server.train_round(
                participating_clients=participating_clients,
                verbose=True
            )
            
            # Evaluate
            if (round_idx + 1) % CONFIG_FEDLWF["eval_every"] == 0:
                metrics = server.evaluate_global(seen_classes_only=True)
                print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
                print(f"  F1 (macro): {metrics['f1_macro']*100:.2f}%")
        
        # Forgetting Calculation
        print("  🔍 Computing Forgetting...")
        del test_data; gc.collect()
        
        # Compute accuracy per previous task
        current_task_accuracies = {}
        for prev_tid, path in all_test_data.items():
            loaded_test = torch.load(path)
            server.test_data = loaded_test
            tm = server.evaluate_global(seen_classes_only=False) # Evaluate on specific task set
            current_task_accuracies[prev_tid] = tm['accuracy']
            # Track best
            best_acc_per_task[prev_tid] = max(best_acc_per_task.get(prev_tid, 0), tm['accuracy'])
        
        # Calculate AF
        af = 0.0
        if task_id > 0:
            diffs = [max(0, best_acc_per_task[t] - current_task_accuracies[t]) for t in range(task_id)]
            af = sum(diffs) / len(diffs) if diffs else 0.0
        print(f"  Avg Forgetting: {af*100:.2f}%")

        # Save History
        all_history["task_accuracies"].append({
            "task": task_id,
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "avg_forgetting": af,
            "per_task_acc": current_task_accuracies
        })
        
        # 9. Checkpoint
        ckpt_path = os.path.join(output_dir, f"checkpoint_task_{task_id}.pt")
        torch.save({
            'task_id': task_id,
            'model_state_dict': server.global_model.state_dict(),
            'config': CONFIG_FEDLWF,
            'seen_classes': list(seen_classes)
        }, ckpt_path)
        print(f"💾 Checkpoint saved: {ckpt_path}")
        
        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()

    # Final Summary
    print("\n" + "="*80)
    print("🏁 TRAINING COMPLETE")
    print("="*80)
    final = all_history["task_accuracies"][-1]
    print(f"Final Accuracy: {final['accuracy']*100:.2f}%")
    print(f"Final Forgetting: {final['avg_forgetting']*100:.2f}%")
    
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(all_history, f, indent=2)

if __name__ == "__main__":
    main()