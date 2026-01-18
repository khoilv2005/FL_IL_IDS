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

from fed_learning import FederatedClient, FederatedServer, train_federated_multigpu
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
    "lambda_decay": 0.1,       # α decay rate
    "theta_threshold": 0.1,    # AF threshold to reset α
    "cross_task_weight": 0.3,
    
    # Training per task
    "rounds_per_task": 5,
    "local_epochs": 3,
    "learning_rate": 1e-3,
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
    
    for task_id in range(data_loader.num_tasks):
        print(f"\n{'='*80}")
        print(f"📚 TASK {task_id + 1}/{data_loader.num_tasks}")
        print(f"{'='*80}")
        
        # Get data for this task
        client_data, test_data, new_classes = data_loader.get_task_data(task_id)
        seen_classes = data_loader.get_seen_classes(task_id)
        
        # Skip if no data
        if all(len(cd.get("y_train", [])) == 0 for cd in client_data.values()):
            print(f"⚠️ No training data for task {task_id}, skipping...")
            continue
        
        # Set task
        trainer.set_task(task_id, new_classes)
        aggregator.set_task(task_id)
        
        # Create clients
        clients = []
        for cid in range(CONFIG["num_clients"]):
            if len(client_data[cid]["y_train"]) > 0:
                c = FederatedClient(
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
        
        server = FederatedServer(clients, test_data, task_config)
        
        if global_model is not None:
            server.set_global_params(global_model)
        
        server.trainer = trainer
        server.aggregator = aggregator
        
        print(f"\n🎯 Training on {len(new_classes)} new classes: {new_classes}")
        history = train_federated_multigpu(server, task_config)
        
        # Build representation space for gradient projection (SVD)
        all_X = torch.cat([client_data[cid]["X_train"] for cid in client_data 
                          if len(client_data[cid]["y_train"]) > 0], dim=0)
        all_y = torch.cat([client_data[cid]["y_train"] for cid in client_data 
                          if len(client_data[cid]["y_train"]) > 0], dim=0)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rep_loader = DataLoader(
            TensorDataset(all_X, all_y),
            batch_size=CONFIG["batch_size"],
            shuffle=True
        )
        trainer.build_representation_space(
            model=server.global_model,
            data_loader=rep_loader,
            device=device
        )
        
        global_model = server.get_global_params()
        
        print(f"\n📊 Evaluating on all {len(seen_classes)} seen classes...")
        metrics = server.evaluate_global()
        
        print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"  F1 (macro): {metrics['f1_macro']*100:.2f}%")
        
        all_history["task_accuracies"].append({
            "task": task_id,
            "seen_classes": len(seen_classes),
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
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
