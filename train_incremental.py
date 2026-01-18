"""
Federated Class Incremental Learning - Training Script
======================================================
Train with CGoFed for Class Incremental Learning.

Usage:
    python train_incremental.py
"""

import os
import json
from datetime import datetime

import torch
from torch.utils.data import TensorDataset, DataLoader

from fed_learning import train_federated_multigpu
from fed_learning.servers import IncrementalServer
from fed_learning.clients import CGoFedClient
from fed_learning.data.incremental_loader import IncrementalDataLoader
from fed_learning.strategies import get_strategy


# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    # Data
    "data_dir": "./prepare_data/data/federated_splits/100-clients",
    "output_dir": "./results_incremental",
    
    # Incremental Learning
    "num_clients": 100,
    "total_classes": 34,
    "base_classes": 5,        # Task 1: 5 classes
    "classes_per_task": 4,    # Task 2+: +4 classes each
    
    # Algorithm
    "algorithm": "cgofed",
    "mu": 0.01,
    "lambda_decay": 0.1,         # α decay rate (slower = more stability)
    "theta_threshold": 0.01,     # AF threshold (paper: θ=0.01, very sensitive)
    "cross_task_weight": 0.2,    # 20% history blend
    
    # Training per task (aligned with paper)
    "rounds_per_task": 10,       # Paper: 20 rounds, we use 10 for efficiency
    "local_epochs": 5,           # Paper: 5 epochs
    "learning_rate": 2e-4,       # Slower learning for gradient projection to work
    "batch_size": 1024,
    
    # Eval
    "eval_every": 1,
}


def main():
    print("\n" + "="*80)
    print("🚀 FEDERATED CLASS INCREMENTAL LEARNING")
    print("="*80)
    
    # Initialize incremental data loader
    data_loader = IncrementalDataLoader(
        data_dir=CONFIG["data_dir"],
        num_clients=CONFIG["num_clients"],
        base_classes=CONFIG["base_classes"],
        classes_per_task=CONFIG["classes_per_task"],
        total_classes=CONFIG["total_classes"],
    )
    
    print(f"\n{data_loader}")
    print(f"Total tasks: {data_loader.num_tasks}")
    
    # Get strategy
    trainer, aggregator = get_strategy(
        CONFIG["algorithm"],
        mu=CONFIG["mu"],
        lambda_decay=CONFIG["lambda_decay"],
        theta_threshold=CONFIG["theta_threshold"],
        cross_task_weight=CONFIG["cross_task_weight"],
    )
    
    # History
    all_history = {
        "task_accuracies": [],
        "task_forgetting": [],
    }
    
    global_model = None
    all_test_data = {}  # Store test data for each task to compute AF
    
    for task_id in range(data_loader.num_tasks):
        print(f"\n{'='*80}")
        print(f"📚 TASK {task_id + 1}/{data_loader.num_tasks}")
        print(f"{'='*80}")
        
        # Get data for this task
        client_data, test_data, new_classes = data_loader.get_task_data(task_id)
        seen_classes = data_loader.get_seen_classes(task_id)
        
        # Store test data for AF calculation later
        all_test_data[task_id] = test_data
        
        # Skip if no data
        if all(len(cd.get("y_train", [])) == 0 for cd in client_data.values()):
            print(f"⚠️ No training data for task {task_id}, skipping...")
            continue
        
        # Set task
        trainer.set_task(task_id, new_classes)
        aggregator.set_task(task_id)
        
        # Create CGoFed clients (with representation computation)
        clients = []
        for cid in range(CONFIG["num_clients"]):
            if len(client_data[cid]["y_train"]) > 0:
                c = CGoFedClient(
                    client_id=cid,
                    X_train=client_data[cid]["X_train"],
                    y_train=client_data[cid]["y_train"],
                )
                clients.append(c)
        
        print(f"Active clients: {len(clients)}")
        
        if len(clients) == 0:
            continue
        
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
        
        print(f"\n🎯 Training on {len(new_classes)} new classes: {new_classes}")
        history = train_federated_multigpu(server, task_config)
        
        # Build representation space for gradient projection (SVD)
        # FL-Compliant: Use REPRESENTATIVE CLIENT instead of aggregating all data
        print("\n🔐 Building representation space (FL-compliant: representative client)...")
        
        # Select representative client (first active client with enough data)
        rep_client = clients[0]
        rep_cid = rep_client.client_id
        rep_X = client_data[rep_cid]["X_train"]
        rep_y = client_data[rep_cid]["y_train"]
        
        # If representative has too few samples, sample from a few more clients
        min_samples = 50
        if len(rep_y) < min_samples and len(clients) > 1:
            for extra_client in clients[1:3]:
                extra_cid = extra_client.client_id
                if len(client_data[extra_cid]["y_train"]) > 0:
                    rep_X = torch.cat([rep_X, client_data[extra_cid]["X_train"]], dim=0)
                    rep_y = torch.cat([rep_y, client_data[extra_cid]["y_train"]], dim=0)
                if len(rep_y) >= min_samples:
                    break
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rep_loader = DataLoader(
            TensorDataset(rep_X, rep_y),
            batch_size=CONFIG["batch_size"],
            shuffle=True
        )
        trainer.build_representation_space(
            model=server.global_model,
            data_loader=rep_loader,
            device=device
        )
        
        global_model = server.get_global_params()
        
        # Check if this is the last task (for AUC calculation)
        is_last_task = (task_id == data_loader.num_tasks - 1)
        
        print(f"\n📊 Evaluating on all {len(seen_classes)} seen classes...")
        metrics = server.evaluate_global(compute_auc=is_last_task)
        
        print(f"  Global Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"  F1 (macro): {metrics['f1_macro']*100:.2f}%")
        if is_last_task and metrics.get('auc_macro_ovr') is not None:
            print(f"  AUC (macro OvR): {metrics['auc_macro_ovr']*100:.2f}%")
        
        # [FIX] Compute per-task accuracy for accurate AF calculation
        print("\n🔍 Computing Per-Task Accuracy for AF...")
        task_accuracies = {}
        original_test_data = server.test_data  # Save original
        
        for prev_tid, prev_test_data in all_test_data.items():
            # Temporarily set server test data to this task's data
            server.test_data = prev_test_data
            t_metrics = server.evaluate_global()
            task_accuracies[prev_tid] = t_metrics['accuracy']
            print(f"    Task {prev_tid} Acc: {t_metrics['accuracy']*100:.2f}%")
        
        # Restore original test data
        server.test_data = original_test_data
        
        # Update forgetting with ALL task accuracies (triggers α reset if AF > θ)
        trainer.update_forgetting(task_accuracies)
        
        # Get and log Average Forgetting
        current_af = trainer.get_current_af()
        print(f"  Average Forgetting (AF): {current_af*100:.2f}%")
        print(f"  Current α: {trainer.alpha:.4f}")
        
        all_history["task_accuracies"].append({
            "task": task_id,
            "seen_classes": len(seen_classes),
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "per_task_acc": task_accuracies,
            "avg_forgetting": current_af,
            "alpha": trainer.alpha,
        })
    
    # Save results
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    hist_path = os.path.join(CONFIG["output_dir"], f"cgofed_incremental_{ts}.json")
    with open(hist_path, "w") as f:
        json.dump(all_history, f, indent=2)
    print(f"\n💾 Saved: {hist_path}")
    
    # Summary
    print("\n" + "="*80)
    print("📊 INCREMENTAL LEARNING SUMMARY")
    print("="*80)
    for h in all_history["task_accuracies"]:
        print(f"  Task {h['task']}: {h['seen_classes']} classes → Acc: {h['accuracy']*100:.2f}%")
    
    print("\n✅ DONE!")


if __name__ == "__main__":
    main()
