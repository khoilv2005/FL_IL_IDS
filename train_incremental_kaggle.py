"""
Federated Class Incremental Learning - Kaggle Training Script
==============================================================
Train with CGoFed or FedAvg for Class Incremental Learning on Kaggle.

Reference:
    "CGoFed: Constrained Gradient Optimization Strategy for Federated Class 
    Incremental Learning", IEEE TKDE, Vol. 37, No. 5, May 2025

Usage:
    Upload fed_learning folder to Kaggle dataset, then run this script.
"""

import os
import sys
import gc
import json
import shutil
from datetime import datetime

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
    print("✓ Imports ready!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


# =============================================================================
# CONFIG - CGoFed Optimized for Better Retention
# =============================================================================
CONFIG = {
    # Data
    "data_dir": "/kaggle/input/data-10clients",
    "output_dir": "./results_incremental",
    
    # Incremental Learning - 5 Tasks Distribution
    "num_clients": 10,
    "total_classes": 34,
    "base_classes": 10,         # Task 0: 10 classes
    "classes_per_task": 6,      # Task 1-4: +6 classes per task (10+6*4=34)
    
    # Algorithm - CGoFed Optimized
    "algorithm": "cgofed",
    "mu": 1.5,                    # Stronger constraint to prevent forgetting
    
    # α Relaxation - Faster decay for plasticity
    "lambda_decay": 0.05,         # Faster α decay → better plasticity
    
    # AF Threshold - Early detection
    "theta_threshold": 0.01,      # Reset α when AF > 1% (early detection)
    
    "cross_task_weight": 0.35,    # Cross-task regularization
    "energy_threshold": 0.97,     # High SVD basis retention
    "num_samples_rep": 1536,      # Stable SVD with more samples
    "top_k": 2,                   # Paper: K=2
    
    # Training - Optimized
    "rounds_per_task": 5,         # Converges by round 4-5
    "local_epochs": 5,            # OK
    "learning_rate": 0.001,       # Balanced learning rate
    "batch_size": 1024,             # Smaller batch → better gradient projection
    
    # Eval
    "eval_every": 1,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def cleanup_temp_folders():
    """Clean up temporary folders from previous runs."""
    for folder in ["./temp_svd_storage", "./temp_test_data"]:
        if os.path.exists(folder):
            print(f"🧹 Cleaning {folder}...")
            shutil.rmtree(folder)


def create_clients(client_data, config):
    """Create federated clients from client data."""
    clients = []
    for cid in range(config["num_clients"]):
        if cid in client_data and len(client_data[cid].get("y_train", [])) > 0:
            c = CGoFedClient(
                client_id=cid,
                X_train=client_data[cid]["X_train"],
                y_train=client_data[cid]["y_train"],
            )
            clients.append(c)
    return clients


def build_representation_space(trainer, client_data, server, config):
    """
    Build representation space for CGoFed (paper eq. 2-4).
    
    Delegates to trainer.build_space_from_client_data() for encapsulation.
    This keeps the runner file clean and algorithm-agnostic.
    """
    # Check if trainer supports this (CGoFed only)
    if hasattr(trainer, 'build_space_from_client_data'):
        # New encapsulated method in CGoFedTrainer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        trainer.build_space_from_client_data(
            model=server.global_model,
            client_data=client_data,
            config=config,
            device=device
        )
    elif hasattr(trainer, 'build_representation_space'):
        # Fallback: old direct method (backward compatibility)
        print("\n🔐 Building gradient-based representation space (paper Section 5.1)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Find clients with data
        active_cids = [cid for cid, data in client_data.items() 
                       if len(data.get("y_train", [])) > 0]
        
        if len(active_cids) == 0:
            print("⚠️ No client data available for representation space")
            return
        
        # Sample uniformly from available clients for robust gradient space
        all_X, all_y = [], []
        samples_per_client = max(10, config["num_samples_rep"] // len(active_cids) + 1)
        
        for cid in active_cids:
            X = client_data[cid]["X_train"]
            y = client_data[cid]["y_train"]
            
            if len(y) > samples_per_client:
                indices = torch.randperm(len(y))[:samples_per_client]
                X, y = X[indices], y[indices]
            
            all_X.append(X)
            all_y.append(y)
        
        rep_X = torch.cat(all_X, dim=0)
        rep_y = torch.cat(all_y, dim=0)
        
        # Limit to num_samples_rep
        if len(rep_y) > config["num_samples_rep"]:
            indices = torch.randperm(len(rep_y))[:config["num_samples_rep"]]
            rep_X, rep_y = rep_X[indices], rep_y[indices]
        
        print(f"   Using {len(rep_y)} samples from {len(active_cids)} clients for SVD.")
        
        rep_loader = DataLoader(
            TensorDataset(rep_X, rep_y),
            batch_size=32,
            shuffle=False
        )
        
        trainer.build_representation_space(
            model=server.global_model,
            data_loader=rep_loader,
            device=device
        )


def compute_per_task_accuracy(server, all_test_data):
    """Compute accuracy for each previous task (for AF calculation)."""
    task_accuracies = {}
    
    for prev_tid, prev_test_path in all_test_data.items():
        try:
            loaded_test_data = torch.load(prev_test_path)
        except Exception as e:
            print(f"⚠️ Failed to load test data for task {prev_tid}: {e}")
            continue
        
        server.test_data = loaded_test_data
        t_metrics = server.evaluate_global()
        task_accuracies[prev_tid] = t_metrics['accuracy']
        print(f"    Task {prev_tid} Acc: {t_metrics['accuracy']*100:.2f}%")
        
        # Cleanup immediately to save memory
        del loaded_test_data
        server.test_data = None
        gc.collect()
    
    return task_accuracies


def compute_average_forgetting(trainer, task_accuracies, best_acc_per_task, task_id):
    """Compute Average Forgetting (AF) metric."""
    current_af = 0.0
    current_alpha = 1.0
    
    # Update best accuracies
    for tid, acc in task_accuracies.items():
        if tid not in best_acc_per_task:
            best_acc_per_task[tid] = acc
        else:
            best_acc_per_task[tid] = max(best_acc_per_task[tid], acc)
    
    if hasattr(trainer, 'update_forgetting'):
        # CGoFed: Use built-in method
        trainer.update_forgetting(task_accuracies)
        current_af = trainer.get_current_af()
        current_alpha = trainer.mu_coefficient
    else:
        # FedAvg: Calculate manually (AF = Avg(Best - Current))
        if len(best_acc_per_task) > 1:
            forgetting = []
            for tid in range(task_id):
                if tid in best_acc_per_task and tid in task_accuracies:
                    f = best_acc_per_task[tid] - task_accuracies[tid]
                    forgetting.append(max(0, f))
            if forgetting:
                current_af = sum(forgetting) / len(forgetting)
    
    return current_af, current_alpha


def save_checkpoint(server, trainer, config, task_id, seen_classes):
    """Save training checkpoint after each task."""
    os.makedirs(config["output_dir"], exist_ok=True)
    checkpoint_path = os.path.join(config["output_dir"], f"checkpoint_task_{task_id}.pt")
    
    torch.save({
        'task_id': task_id,
        'model_state_dict': server.global_model.state_dict(),
        'trainer_old_space': getattr(trainer, 'old_space', {}),
        'trainer_importance_weights': getattr(trainer, 'importance_weights', {}),
        'config': config,
        'seen_classes': list(seen_classes),
    }, checkpoint_path)
    
    print(f"💾 Checkpoint saved: {checkpoint_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "="*80)
    print("🚀 FEDERATED CLASS INCREMENTAL LEARNING (CGoFed) - KAGGLE")
    print("   Paper: IEEE TKDE 2025 - Constrained Gradient Optimization")
    print("="*80)
    
    # Cleanup temp folders from previous runs
    cleanup_temp_folders()
    os.makedirs("./temp_test_data", exist_ok=True)
    
    # Initialize data loader
    data_loader = IncrementalDataLoader(
        data_dir=CONFIG["data_dir"]
    )
    
    print(f"\n{data_loader}")
    print(f"Total tasks: {data_loader.get_num_tasks()}")
    
    # Get training strategy
    trainer, aggregator = get_strategy(
        CONFIG["algorithm"],
        mu=CONFIG["mu"],
        lambda_decay=CONFIG["lambda_decay"],
        theta_threshold=CONFIG["theta_threshold"],
        cross_task_weight=CONFIG["cross_task_weight"],
        energy_threshold=CONFIG["energy_threshold"],
        num_samples_rep=CONFIG["num_samples_rep"],
        top_k=CONFIG.get("top_k", 2),
    )
    
    # Initialize tracking variables
    all_history = {"task_accuracies": [], "task_forgetting": []}
    global_model = None
    all_test_data = {}      # task_id -> file path (disk-based caching)
    best_acc_per_task = {}  # task_id -> best accuracy
    
    # ==========================================================================
    # TASK LOOP
    # ==========================================================================
    for task_id in range(data_loader.get_num_tasks()):
        print(f"\n{'='*80}")
        print(f"📚 TASK {task_id}/{data_loader.get_num_tasks()}")
        print(f"{'='*80}")
        
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
                 # Check if server expects (X, y) or DataLoader. 
                 # Existing code uses dict {"X_train": X, "y_train": y}
                 client_data[cid] = {"X_train": X, "y_train": y}
        
        print(f"  Clients with data for task {task_id}: {len(client_data)}")

        # Load global test data (cumulative)
        test_X, test_y = data_loader.get_test_data(task_id, cumulative=True)
        test_data = {"X_test": test_X, "y_test": test_y}  # Server expects dict format
        
        # Save test data to disk (prevents OOM)
        test_data_path = os.path.join("./temp_test_data", f"test_task_{task_id}.pt")
        torch.save(test_data, test_data_path)
        all_test_data[task_id] = test_data_path
        
        # Skip if no training data
        if all(len(cd.get("y_train", [])) == 0 for cd in client_data.values()):
            print(f"⚠️ No training data for task {task_id}, skipping...")
            continue
        
        # Set task (CGoFed only)
        if hasattr(trainer, 'set_task'):
            trainer.set_task(task_id, new_classes)
        if hasattr(aggregator, 'set_task'):
            aggregator.set_task(task_id)
        
        # Create clients
        clients = create_clients(client_data, CONFIG)
        print(f"Active clients: {len(clients)}")
        
        if len(clients) == 0:
            continue
        
        # Setup server
        task_config = {
            **CONFIG,
            "num_rounds": CONFIG["rounds_per_task"],
            "input_shape": data_loader.input_shape,
            "num_classes": CONFIG["total_classes"],
        }
        
        server = IncrementalServer(clients, test_data, task_config)
        if global_model is not None:
            server.set_global_params(global_model)
        server.trainer = trainer
        server.aggregator = aggregator
        
        # Train
        print(f"\n🎯 Training on {len(new_classes)} new classes: {new_classes}")
        history = train_federated_multigpu(server, task_config)
        
        # Build representation space (CGoFed only)
        build_representation_space(trainer, client_data, server, CONFIG)
        
        # Save global model for next task
        global_model = server.get_global_params()
        
        # Evaluate on all seen classes
        is_last_task = (task_id == data_loader.num_tasks - 1)
        print(f"\n📊 Evaluating on all {len(seen_classes)} seen classes...")
        metrics = server.evaluate_global(compute_auc=is_last_task)
        
        print(f"  Global Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"  F1 (macro): {metrics['f1_macro']*100:.2f}%")
        if is_last_task and metrics.get('auc_macro_ovr') is not None:
            print(f"  AUC (macro OvR): {metrics['auc_macro_ovr']*100:.2f}%")
        
        # Compute per-task accuracy for AF
        print("\n🔍 Computing Per-Task Accuracy for AF...")
        del test_data  # Free memory before loading previous test data
        gc.collect()
        
        task_accuracies = compute_per_task_accuracy(server, all_test_data)
        
        # Restore test data for server
        server.test_data = torch.load(all_test_data[task_id])
        
        # Compute Average Forgetting
        current_af, current_alpha = compute_average_forgetting(
            trainer, task_accuracies, best_acc_per_task, task_id
        )
        
        print(f"  Average Forgetting (AF): {current_af*100:.2f}%")
        if hasattr(trainer, 'alpha'):
            print(f"  Current α: {current_alpha:.4f}")
        
        # Record history
        all_history["task_accuracies"].append({
            "task": task_id,
            "seen_classes": len(seen_classes),
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "per_task_acc": task_accuracies,
            "avg_forgetting": current_af,
            "alpha": current_alpha,
        })
        
        # Save checkpoint
        save_checkpoint(server, trainer, CONFIG, task_id, seen_classes)
        
        # Clean memory
        del client_data, clients
        torch.cuda.empty_cache()
        gc.collect()
        print(f"🧹 Memory cleaned after Task {task_id}")
    
    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    hist_path = os.path.join(CONFIG["output_dir"], f"incremental_{CONFIG['algorithm']}_{ts}.json")
    with open(hist_path, "w") as f:
        json.dump(all_history, f, indent=2)
    print(f"\n💾 Saved: {hist_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("📊 INCREMENTAL LEARNING SUMMARY (CGoFed)")
    print("="*80)
    for h in all_history["task_accuracies"]:
        af_str = f", AF: {h['avg_forgetting']*100:.2f}%" if h['avg_forgetting'] > 0 else ""
        alpha_str = f", α={h['alpha']:.3f}" if h.get('alpha') else ""
        print(f"  Task {h['task']}: {h['seen_classes']} classes → "
              f"Acc: {h['accuracy']*100:.2f}%{af_str}{alpha_str}")
    
    # Final metrics
    if all_history["task_accuracies"]:
        final = all_history["task_accuracies"][-1]
        print(f"\n  🎯 Final Accuracy (all classes): {final['accuracy']*100:.2f}%")
        print(f"  🎯 Final Average Forgetting: {final['avg_forgetting']*100:.2f}%")
    
    print("\n✅ DONE!")


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
    
    # Initialize data loader
    print(f"\n📂 Loading data from: {CONFIG_FEDCBDR['data_dir']}")
    data_loader = IncrementalDataLoader(
        data_dir=CONFIG_FEDCBDR["data_dir"]
    )
    
    print(f"\n{data_loader}")
    
    # Get input shape
    input_shape = data_loader.input_shape
    CONFIG_FEDCBDR["input_shape"] = input_shape
    CONFIG_FEDCBDR["num_classes"] = CONFIG_FEDCBDR["total_classes"]
    
    # Initialize tracking variables
    all_history = {"task_accuracies": [], "task_forgetting": []}
    all_clients = {}
    all_test_data = {}
    best_acc_per_task = {}
    server = None
    
    # ==========================================================================
    # TASK LOOP
    # ==========================================================================
    for task_id in range(data_loader.get_num_tasks()):
        print(f"\n{'='*80}")
        print(f"📚 TASK {task_id}/{data_loader.get_num_tasks()}")
        print(f"{'='*80}")
        
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
        
        # Create/update clients
        for cid, data in client_data.items():
            if cid not in all_clients:
                all_clients[cid] = FedCBDRClient(
                    client_id=cid,
                    X_train=data["X_train"],
                    y_train=data["y_train"],
                    buffer_size=CONFIG_FEDCBDR.get("buffer_size", 500),
                )
            else:
                all_clients[cid].set_task_data(
                    X_train=data["X_train"],
                    y_train=data["y_train"],
                )
        
        participating_clients = [
            all_clients[cid] for cid in client_data.keys()
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
        
        # Update all clients with task info
        for client in participating_clients:
            client.set_task(task_id, new_classes)
        
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
        
        # Per-task accuracy
        task_accs = server.evaluate_per_task()
        
        print(f"   Per-task accuracies:")
        for tid, acc in sorted(task_accs.items()):
            status = "✓" if acc > 0.7 else "⚠️"
            print(f"    {status} Task {tid}: {acc*100:.2f}%")
        
        # Average forgetting
        if task_id > 0:
            af = server.compute_average_forgetting()
            all_history["task_forgetting"].append(af)
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
        
        # Record history
        all_history["task_accuracies"].append({
            "task": task_id,
            "seen_classes": len(seen_classes),
            "accuracy": final_metrics["accuracy"],
            "f1_macro": final_metrics["f1_macro"],
            "per_task_acc": task_accs,
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
    server = None
    current_alpha = CONFIG_FEDLWF["lwf_alpha"]
    
    # ==========================================================================
    # TASK LOOP
    # ==========================================================================
    for task_id in range(data_loader.get_num_tasks()):
        print(f"\n{'='*80}")
        print(f"📚 TASK {task_id}/{data_loader.get_num_tasks()}")
        print(f"{'='*80}")
        
        # Adjust alpha for current task
        if task_id > 0:
            current_alpha = CONFIG_FEDLWF["lwf_alpha"] * (CONFIG_FEDLWF["lwf_alpha_scale"] ** (task_id - 1))
            print(f"   Current KD α: {current_alpha:.3f}")
        
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
        
        # Create/update clients
        for cid, data in client_data.items():
            if cid not in all_clients:
                all_clients[cid] = FedLwFClient(
                    client_id=cid,
                    X_train=data["X_train"],
                    y_train=data["y_train"],
                )
            else:
                all_clients[cid].set_task_data(
                    X_train=data["X_train"],
                    y_train=data["y_train"],
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
        
        # Update all clients with task info
        for client in participating_clients:
            client.set_task(task_id, new_classes)
        
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
        
        # ==================================================================
        # POST-TASK: Save model snapshot for KD
        # ==================================================================
        print(f"\n📸 Saving model snapshot for Task {task_id}...")
        server.save_global_snapshot()
        
        # ==================================================================
        # TASK EVALUATION
        # ==================================================================
        print(f"\n📊 Final evaluation for Task {task_id}...")
        
        # Per-task accuracy
        task_accs = server.evaluate_per_task()
        
        print(f"   Per-task accuracies:")
        for tid, acc in sorted(task_accs.items()):
            status = "✓" if acc > 0.7 else "⚠️"
            print(f"    {status} Task {tid}: {acc*100:.2f}%")
        
        # Average forgetting
        if task_id > 0:
            af = server.compute_average_forgetting()
            all_history["task_forgetting"].append(af)
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
        
        # Record history
        all_history["task_accuracies"].append({
            "task": task_id,
            "seen_classes": len(seen_classes),
            "accuracy": final_metrics["accuracy"],
            "f1_macro": final_metrics["f1_macro"],
            "per_task_acc": task_accs,
            "kd_alpha": current_alpha,
        })
        
        # Cleanup
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    print("\n" + "="*80)
    print("🎉 FEDLWF TRAINING COMPLETE")
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


if __name__ == "__main__":
    main()