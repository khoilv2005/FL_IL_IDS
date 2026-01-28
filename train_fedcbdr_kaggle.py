"""
FedCBDR - Federated Class-Incremental Learning with Class-wise Balancing Data Replay
====================================================================================
Training script for Kaggle environment.

Reference:
    "Class-wise Balancing Data Replay for Federated Class-Incremental Learning"
    Zhuang Qi, Ying-Peng Tang, Lei Meng, Han Yu, Xiaoxiao Li, Xiangxu Meng
    arXiv:2507.07712, 2025

Key Features:
1. Global-perspective Data Replay (GDR) - Privacy-preserving replay buffer coordination
2. Task-aware Temperature Scaling (TTS) - Balanced loss for old vs new classes
3. Class-balanced memory buffer with herding/leverage score selection

Usage on Kaggle:
    Upload fed_learning folder to Kaggle dataset, then run this script.
"""

import os
import sys
import gc
import json
import shutil
from datetime import datetime

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


# =============================================================================
# KAGGLE SETUP
# =============================================================================
MODULE_PATH = "/kaggle/input/ai4fids-fedlearning-modules"


def setup_imports():
    """Setup imports for both nested and flattened dataset structures."""
    if not os.path.exists(MODULE_PATH):
        print(f"⚠️ Warning: Module path {MODULE_PATH} not found!")
        print("   Trying local imports...")
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
    from fed_learning.servers.fedcbdr_server import FedCBDRServer
    from fed_learning.clients.fedcbdr_client import FedCBDRClient
    from fed_learning.data.incremental_loader import IncrementalDataLoader
    from fed_learning.strategies.incremental.fedcbdr import (
        FedCBDRTrainer, FedCBDRAggregator, ReplayBuffer
    )
    print("✓ FedCBDR imports ready!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("   Traceback:")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
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
    "tau_old": 0.9,             # Temperature for old task logits (lower = sharper)
    "tau_new": 1.1,             # Temperature for new task logits (higher = smoother)
    "omega_old": 1.1,           # Weight for old task samples
    "omega_new": 0.9,           # Weight for new task samples
    
    # Replay Buffer
    "buffer_size": 500,         # Max samples per client in replay buffer
    "leverage_rank": 50,        # Rank for SVD in leverage score computation
    "use_replay": True,         # Whether to use replay buffer
    "replay_ratio": 0.5,        # Ratio of replay samples in each batch
    "use_herding": True,        # Use herding for sample selection
    
    # Training
    "rounds_per_task": 5,       # Communication rounds per task
    "local_epochs": 3,          # Local epochs per round
    "learning_rate": 0.001,
    "batch_size": 128,
    
    # Eval
    "eval_every": 1,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def cleanup_temp_folders():
    """Clean up temporary folders from previous runs."""
    for folder in ["./temp_fedcbdr"]:
        if os.path.exists(folder):
            print(f"🧹 Cleaning {folder}...")
            shutil.rmtree(folder)


def get_task_structure(total_classes: int, base_classes: int, classes_per_task: int):
    """
    Generate task structure for incremental learning.
    
    Returns:
        Dict mapping task_id to list of class indices
    """
    task_classes = {}
    
    # Task 0: base classes
    task_classes[0] = list(range(base_classes))
    
    # Subsequent tasks
    current_class = base_classes
    task_id = 1
    
    while current_class < total_classes:
        end_class = min(current_class + classes_per_task, total_classes)
        task_classes[task_id] = list(range(current_class, end_class))
        current_class = end_class
        task_id += 1
    
    return task_classes


def load_task_data(data_loader, task_classes, client_ids=None):
    """
    Load and filter data for specific task classes.
    
    Args:
        data_loader: IncrementalDataLoader instance
        task_classes: List of class IDs for this task
        client_ids: Optional list of client IDs to load
        
    Returns:
        client_data: Dict[client_id, Dict[X_train, y_train]]
    """
    client_data = {}
    task_class_set = set(task_classes)
    
    # Determine which clients to load
    if client_ids is None:
        # Find all available client files
        client_files = list(data_loader.data_dir.glob("client_*_train.npz"))
        client_ids = []
        for f in client_files:
            try:
                cid = int(f.stem.split("_")[1])
                client_ids.append(cid)
            except:
                pass
    
    for cid in sorted(client_ids):
        client_file = data_loader.data_dir / f"client_{cid}_train.npz"
        
        if not client_file.exists():
            continue
        
        try:
            data = np.load(client_file)
            X = data['X_train'].astype(np.float32)
            y = data['y_train'].astype(np.int64)
            
            # Filter to task classes
            mask = np.isin(y, list(task_class_set))
            
            if mask.sum() > 0:
                client_data[cid] = {
                    "X_train": torch.from_numpy(X[mask]),
                    "y_train": torch.from_numpy(y[mask]),
                }
        except Exception as e:
            print(f"   Warning: Failed to load client {cid}: {e}")
            continue
    
    return client_data


def create_fedcbdr_clients(client_data, config):
    """Create FedCBDR clients from client data."""
    clients = []
    
    for cid, data in client_data.items():
        if len(data.get("y_train", [])) > 0:
            client = FedCBDRClient(
                client_id=cid,
                X_train=data["X_train"],
                y_train=data["y_train"],
                buffer_size=config["buffer_size"],
                leverage_rank=config["leverage_rank"],
            )
            clients.append(client)
    
    return clients


def print_task_summary(task_id, task_classes, num_clients, total_samples):
    """Print summary of current task."""
    print(f"\n{'='*70}")
    print(f"📌 TASK {task_id}")
    print(f"{'='*70}")
    print(f"   Classes: {task_classes}")
    print(f"   Participating clients: {num_clients}")
    print(f"   Total samples: {total_samples:,}")


def print_metrics(metrics, task_id, round_idx=None):
    """Print evaluation metrics."""
    header = f"Task {task_id}"
    if round_idx is not None:
        header += f" Round {round_idx+1}"
    
    print(f"\n📊 {header} Metrics:")
    print(f"   Accuracy:    {metrics['accuracy']*100:.2f}%")
    print(f"   F1 (macro):  {metrics['f1_macro']*100:.2f}%")
    print(f"   F1 (weight): {metrics['f1_weighted']*100:.2f}%")
    print(f"   Precision:   {metrics['precision_macro']*100:.2f}%")
    print(f"   Recall:      {metrics['recall_macro']*100:.2f}%")


# =============================================================================
# MAIN TRAINING
# =============================================================================
def main():
    print("\n" + "="*80)
    print("🚀 FEDCBDR - Class-wise Balancing Data Replay for FCIL")
    print("="*80)
    print(f"   τ_old={CONFIG['tau_old']}, τ_new={CONFIG['tau_new']}")
    print(f"   ω_old={CONFIG['omega_old']}, ω_new={CONFIG['omega_new']}")
    print(f"   Buffer size: {CONFIG['buffer_size']}")
    print(f"   Replay ratio: {CONFIG['replay_ratio']}")
    
    # Cleanup
    cleanup_temp_folders()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{CONFIG['output_dir']}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(CONFIG, f, indent=2, default=str)
    
    # Initialize data loader
    print(f"\n📂 Loading data from: {CONFIG['data_dir']}")
    data_loader = IncrementalDataLoader(CONFIG["data_dir"])
    
    # Get input shape and task structure
    input_shape = data_loader.input_shape
    CONFIG["input_shape"] = input_shape
    CONFIG["num_classes"] = CONFIG["total_classes"]
    
    print(f"   Input shape: {input_shape}")
    print(f"   Total classes: {CONFIG['total_classes']}")
    
    # Generate task structure
    task_structure = get_task_structure(
        CONFIG["total_classes"],
        CONFIG["base_classes"],
        CONFIG["classes_per_task"]
    )
    num_tasks = len(task_structure)
    print(f"   Number of tasks: {num_tasks}")
    
    # Load test data
    print("\n📥 Loading test data...")
    test_path = data_loader.data_dir / "global_test_data.npz"
    test_npz = np.load(test_path)
    test_data = {
        "X_test": torch.from_numpy(test_npz["X_test"].astype(np.float32)),
        "y_test": torch.from_numpy(test_npz["y_test"].astype(np.int64)),
    }
    print(f"   Test samples: {len(test_data['y_test']):,}")
    
    # Initialize clients (will be populated with task data)
    all_clients = {}  # client_id -> FedCBDRClient
    
    # Initialize server
    server = None
    
    # History tracking
    history = {
        "task_metrics": {},
        "round_metrics": [],
        "forgetting": [],
    }
    
    # ==========================================================================
    # INCREMENTAL LEARNING LOOP
    # ==========================================================================
    seen_classes = []
    
    for task_id in range(num_tasks):
        task_classes = task_structure[task_id]
        seen_classes.extend(task_classes)
        
        # Load task data
        print(f"\n📥 Loading data for Task {task_id}...")
        task_client_data = load_task_data(data_loader, task_classes)
        
        if not task_client_data:
            print(f"   ⚠️ No data for Task {task_id}, skipping...")
            continue
        
        total_samples = sum(len(d["y_train"]) for d in task_client_data.values())
        print_task_summary(task_id, task_classes, len(task_client_data), total_samples)
        
        # Create/update clients
        for cid, data in task_client_data.items():
            if cid not in all_clients:
                # New client
                client = FedCBDRClient(
                    client_id=cid,
                    X_train=data["X_train"],
                    y_train=data["y_train"],
                    buffer_size=CONFIG["buffer_size"],
                    leverage_rank=CONFIG["leverage_rank"],
                )
                all_clients[cid] = client
            else:
                # Existing client: update with new task data
                all_clients[cid].set_task_data(
                    X_train=data["X_train"],
                    y_train=data["y_train"],
                    task_id=task_id,
                    task_classes=task_classes,
                )
        
        participating_clients = [
            all_clients[cid] for cid in task_client_data.keys()
        ]
        
        # Initialize or update server
        if server is None:
            server = FedCBDRServer(
                clients=list(all_clients.values()),
                test_data=test_data,
                config=CONFIG,
            )
        
        # Set task for server and trainer
        server.set_task(task_id, task_classes)
        
        # Update all clients with task info
        for client in participating_clients:
            client.current_task = task_id
            client.current_classes = set(task_classes)
            client.seen_classes.update(task_classes)
        
        # ==================================================================
        # TRAINING ROUNDS
        # ==================================================================
        print(f"\n🏋️ Training Task {task_id} for {CONFIG['rounds_per_task']} rounds...")
        
        for round_idx in range(CONFIG["rounds_per_task"]):
            print(f"\n--- Round {round_idx+1}/{CONFIG['rounds_per_task']} ---")
            
            # Train
            round_result = server.train_round(
                participating_clients=participating_clients,
                verbose=True
            )
            
            # Record
            history["round_metrics"].append({
                "task_id": task_id,
                "round": round_idx,
                "train_loss": round_result["train_loss"],
            })
            
            # Evaluate
            if (round_idx + 1) % CONFIG["eval_every"] == 0:
                metrics = server.evaluate_global(
                    seen_classes_only=True,
                    compute_auc=(round_idx == CONFIG["rounds_per_task"] - 1)
                )
                print_metrics(metrics, task_id, round_idx)
                
                # Update server history
                server.history["train_loss"].append(round_result["train_loss"])
                server.history["test_loss"].append(metrics["loss"])
                server.history["test_accuracy"].append(metrics["accuracy"])
                server.history["test_f1_macro"].append(metrics["f1_macro"])
                server.history["test_f1_weighted"].append(metrics["f1_weighted"])
                server.history["test_precision_macro"].append(metrics["precision_macro"])
                server.history["test_recall_macro"].append(metrics["recall_macro"])
        
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
        
        # Per-task accuracy
        task_accs = server.evaluate_per_task()
        history["task_metrics"][task_id] = task_accs
        
        print(f"   Per-task accuracies:")
        for tid, acc in sorted(task_accs.items()):
            marker = "→" if tid == task_id else " "
            print(f"   {marker} Task {tid}: {acc*100:.2f}%")
        
        # Average forgetting
        if task_id > 0:
            af = server.compute_average_forgetting()
            history["forgetting"].append(af)
            print(f"   Average Forgetting: {af*100:.2f}%")
        
        # Overall metrics
        final_metrics = server.evaluate_global(
            seen_classes_only=True,
            compute_auc=True
        )
        print(f"\n   📈 Overall (seen classes):")
        print(f"      Accuracy: {final_metrics['accuracy']*100:.2f}%")
        print(f"      F1 (macro): {final_metrics['f1_macro']*100:.2f}%")
        
        # Save checkpoint
        checkpoint_path = f"{output_dir}/checkpoint_task{task_id}.pt"
        server.save_checkpoint(checkpoint_path, task_id)
        
        # Buffer statistics
        print(f"\n   📦 Replay Buffer Statistics:")
        total_buffer = sum(c.replay_buffer.total_samples for c in all_clients.values())
        total_classes_in_buffer = len(set().union(*[
            set(c.replay_buffer.class_buffers.keys()) for c in all_clients.values()
        ]))
        print(f"      Total samples: {total_buffer}")
        print(f"      Classes covered: {total_classes_in_buffer}")
        
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETE")
    print("="*80)
    
    # Final evaluation on all classes
    print("\n📊 Final Evaluation (all classes):")
    final_metrics = server.evaluate_global(
        seen_classes_only=False,
        compute_auc=True
    )
    print(f"   Accuracy:    {final_metrics['accuracy']*100:.2f}%")
    print(f"   F1 (macro):  {final_metrics['f1_macro']*100:.2f}%")
    print(f"   F1 (weight): {final_metrics['f1_weighted']*100:.2f}%")
    print(f"   Precision:   {final_metrics['precision_macro']*100:.2f}%")
    print(f"   Recall:      {final_metrics['recall_macro']*100:.2f}%")
    if final_metrics.get("auc_macro_ovr"):
        print(f"   AUC:         {final_metrics['auc_macro_ovr']*100:.2f}%")
    
    # Forgetting summary
    if history["forgetting"]:
        print(f"\n📉 Forgetting Analysis:")
        print(f"   Final AF: {history['forgetting'][-1]*100:.2f}%")
        print(f"   Max AF:   {max(history['forgetting'])*100:.2f}%")
    
    # Save final results
    results = {
        "config": CONFIG,
        "final_metrics": final_metrics,
        "task_metrics": history["task_metrics"],
        "forgetting": history["forgetting"],
        "server_history": server.history,
    }
    
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
    
    print(f"\n💾 Results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
