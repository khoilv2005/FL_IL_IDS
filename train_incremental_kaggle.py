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
from sklearn.metrics import confusion_matrix

# Visualization modules
try:
    from fed_learning.visualization.metrics import (
        plot_confusion_matrix,
        plot_per_class_metrics,
    )
    from fed_learning.visualization.plots import save_training_plots
    from fed_learning.visualization.fcil_plots import (
        generate_fcil_report,
        plot_incremental_accuracy_curve,
        plot_task_accuracy_heatmap,
        plot_forgetting_comparison,
        compute_average_incremental_accuracy,
        compute_forgetting_measure,
    )

    VISUALIZATION_AVAILABLE = True
    FCIL_VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    FCIL_VISUALIZATION_AVAILABLE = False
    print("⚠️ Visualization modules not found. Plotting disabled.")


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
    from fed_learning.servers import IncrementalServer, FedCBDRServer, FedLwFServer

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
    # Reproducibility
    "random_seed": 42,  # Set to None for random behavior
    # Algorithm Selection
    # Options: "cgofed", "fedavg_ewc", "fedprox_ewc", "fedavg_lwf", "fedprox_lwf", "fedcbdr"
    "algorithm": "cgofed",
    # Output
    "output_dir": "./results_incremental",
    # Incremental Learning - 5 Tasks Distribution
    "num_clients": 10,
    "total_classes": 34,
    "base_classes": 10,
    "classes_per_task": 6,
    # Common Parameters
    # IoT CIC 2023: non-IID Dirichlet α=5.0 (moderate heterogeneity)
    # 0.5 đủ kéo model về global mà không giảm convergence speed
    "mu_fedprox": 0.5,  # Proximal term ||θ - θ_global||² (FedProx/Fed+/EWC/LwF)
    "rounds_per_task": 5,  # Tăng rounds, giảm local epochs -> communicate more often
    "local_epochs": 3,  # Ít local epochs giảm client drift trong FL
    # Large batch (1024) cần LR cao hơn theo Linear Scaling Rule
    # Nhưng với CGoFed projection, không scale full 16x mà chỉ 5x để ổn định
    "learning_rate": 0.002,  # Scaled up từ 0.001 (batch=64) cho batch=1024
    "batch_size": 1024,  # Large batch: gradient ổn định, tốc độ nhanh, cần GPU memory
    "eval_every": 1,
    # --- Algorithm Specific Params ---
    # CGoFed - Tuned cho IoT CIC 2023 (34 classes, 46 features, 5 tasks)
    # Projection: 3.0 cân bằng stability-plasticity, tránh chặn gradient hữu ích
    # giữa các class tương tự (DDoS/DoS variants share gradient directions)
    "mu_cgofed": 3.0,  # Gradient projection coefficient (paper Eq. 9)
    # Relaxation: 0.5 cho decay vừa phải qua 5 tasks (0.5 -> 0.25 -> 0.125 -> clamped 0.1)
    # FIXED: Was 0.3, too aggressive - mu decayed to near-zero by task 3
    "lambda_decay": 0.5,  # Power decay α^(t-t_reset), task 1: 0.5, task 2: 0.25
    "theta_threshold": 0.05,  # AF > 5% mới reset mu, tránh trigger quá nhạy
    # Cross-task: 0.1 blend vừa phải, tận dụng class groups tương tự (DDoS, DoS, Recon)
    "cross_task_weight": 0.1,
    "energy_threshold": 0.95,  # 95% SVD energy, đủ cho 46-dim input
    # Tăng num_samples_rep để SVD đủ diversity với batch_size lớn
    "num_samples_rep": 2000,  # 2x batch_size, đủ cho representation space robust
    "top_k": 2,
    # EWC
    "ewc_lambda": 10.0,
    "fisher_samples": 200,
    "online_ewc": False,
    # LwF (FedLwF)
    "lwf_alpha": 1.0,
    "temperature": 2.0,
    "lwf_alpha_scale": 1.0,  # Decay/Growth factor for alpha
    "distill_on_new_only": False,
    # FedCBDR
    "tau_old": 0.9,
    "tau_new": 1.1,
    "omega_old": 1.1,
    "omega_new": 0.9,
    "buffer_size": 500,
    "replay_ratio": 0.5,
    "seed": 42,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def set_seed(seed=42):
    """Set random seed for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Seed set to {seed}")


def cleanup_temp_folders():
    """Clean up temporary folders."""
    folders = [
        "./temp_svd_storage",
        "./temp_ewc_storage",
        "./temp_fedlwf_storage",
        "./temp_test_data",
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
                cid,
                X,
                y,
                buffer_size=config.get("buffer_size", 500),
                leverage_rank=config.get("leverage_rank", 50),
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


def post_task_processing(
    trainer, server, client_data, config, participating_clients=None
):
    """Handle post-task logic (Fisher, Snapshot, SVD, Buffer Update)."""
    algo = config["algorithm"].lower()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. CGoFed: Build Representation Space
    if algo == "cgofed":
        if hasattr(trainer, "build_space_from_client_data"):
            trainer.build_space_from_client_data(
                model=server.global_model,
                client_data=client_data,
                config=config,
                device=device,
            )

    # 2. EWC: Consolidate (Compute Fisher)
    elif "ewc" in algo:  # fedavg_ewc, fedprox_ewc
        if hasattr(trainer, "consolidate"):
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
                    idx = torch.randperm(len(y))[: config["fisher_samples"]]
                    X, y = X[idx], y[idx]

                loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
                trainer.consolidate(server.global_model, loader, device)

    # 3. LwF: Save Snapshot - Check for FedLwFTrainer explicitly
    elif isinstance(trainer, FedLwFTrainer) or "lwf" in algo:
        print(f"\n📸 Saving model snapshot for LwF...")
        if hasattr(trainer, "save_model_snapshot"):
            trainer.save_model_snapshot(server.global_model)
        elif hasattr(server, "save_global_snapshot"):
            server.save_global_snapshot()

    # 4. FedCBDR: GDR Update
    if algo == "fedcbdr" and hasattr(server, "coordinate_gdr"):
        print(f"\n🔄 Updating Replay Buffers (GDR)...")
        server.coordinate_gdr(participating_clients, verbose=True)


# =============================================================================
# MAIN
# =============================================================================
def main():
    # Set random seed for reproducibility
    set_seed(CONFIG.get("random_seed", 42))

    # Setup Output
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{CONFIG['output_dir']}_{CONFIG['algorithm']}_{ts}"
    os.makedirs(output_dir, exist_ok=True)

    # Save Config
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(CONFIG, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print(f"🚀 FEDERATED CLASS INCREMENTAL LEARNING - {CONFIG['algorithm'].upper()}")
    print("=" * 80)

    cleanup_temp_folders()
    os.makedirs("./temp_test_data", exist_ok=True)

    # Load Data
    data_loader = IncrementalDataLoader(data_dir=CONFIG["data_dir"])
    print(f"\n{data_loader}")

    # Update Config with Data Params
    CONFIG["input_shape"] = data_loader.input_shape
    CONFIG["num_classes"] = CONFIG["total_classes"]

    # Get Strategy (Trainer & Aggregator)
    # CONFIG includes "algorithm" key, so we pass **CONFIG directly
    trainer, aggregator = get_strategy(**CONFIG)
    print(f"✓ Trainer: {trainer.__class__.__name__}")
    print(f"✓ Aggregator: {aggregator.__class__.__name__}")

    # State Variables
    global_model = None
    all_history = {"task_accuracies": [], "task_forgetting": []}
    all_test_data = {}
    best_acc_per_task = {}

    # Persistent Clients (needed for FedCBDR/LwF state)
    persistent_clients: Dict[int, object] = {}

    # --- Task Loop ---
    for task_id in range(data_loader.get_num_tasks()):
        print(
            f"\n{'=' * 80}\n📚 TASK {task_id}/{data_loader.get_num_tasks()}\n{'=' * 80}"
        )

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

        # 3. Manage Clients (Persistent State)
        participating_clients = []

        # Initialize persistence map if first task
        if task_id == 0:
            persistent_clients = {}

        for cid, data in client_data_map.items():
            if cid not in persistent_clients:
                # Instantiate new client
                if CONFIG["algorithm"] == "fedcbdr":
                    persistent_clients[cid] = FedCBDRClient(
                        cid,
                        data["X_train"],
                        data["y_train"],
                        buffer_size=CONFIG.get("buffer_size", 500),
                        leverage_rank=CONFIG.get("leverage_rank", 50),
                    )
                elif "lwf" in CONFIG["algorithm"]:
                    persistent_clients[cid] = FedLwFClient(
                        cid, data["X_train"], data["y_train"]
                    )
                else:
                    # Standard/CGoFed (Stateless between tasks usually, but we keep for consistency)
                    persistent_clients[cid] = CGoFedClient(
                        cid, data["X_train"], data["y_train"]
                    )

            # Retrieve client
            client = persistent_clients[cid]

            # Update client with new task data
            if hasattr(client, "set_task_data"):  # FedCBDR/LwF Clients
                client.set_task_data(
                    data["X_train"], data["y_train"], task_id, new_classes
                )
            else:
                # Stateless Clients: Just replace data
                client.X_train = data["X_train"]
                client.y_train = data["y_train"]
                client.num_samples = len(data["y_train"])

            participating_clients.append(client)

        # 4. Prepare Server
        task_config = {
            **CONFIG,
            "num_classes": CONFIG["total_classes"],
            "num_rounds": CONFIG["rounds_per_task"],
        }

        # Adjust dynamic params (e.g. LwF alpha decay)
        if "lwf" in CONFIG["algorithm"]:
            current_alpha = (
                CONFIG["lwf_alpha"]
                * (CONFIG.get("lwf_alpha_scale", 1.0) ** max(0, task_id - 1))
                if task_id > 0
                else CONFIG["lwf_alpha"]
            )
            task_config["lwf_alpha"] = current_alpha
            if hasattr(trainer, "lwf_alpha"):
                trainer.lwf_alpha = current_alpha
            print(f"   LwF Alpha: {current_alpha:.4f}")

        server = get_algorithm_specific_components(
            CONFIG, participating_clients, test_data, task_config
        )

        if global_model is not None:
            server.set_global_params(global_model)

        server.trainer = trainer
        server.aggregator = aggregator

        if hasattr(server, "set_task"):
            # CRITICAL FIX: Pass the full seen_classes list, not just new_classes
            # seen_classes was computed at line 353-355, containing ALL classes seen up to now
            server.set_task(task_id, new_classes, seen_classes)
        if hasattr(trainer, "set_task"):
            trainer.set_task(task_id, new_classes)
        if hasattr(aggregator, "set_task"):
            aggregator.set_task(task_id)

        # 5. Train
        print(f"\n🎯 Training on {len(new_classes)} new classes...")
        train_federated_multigpu(server, task_config)

        # 6. Post-Task Processing (Logic Hook)
        post_task_processing(
            trainer, server, client_data_map, CONFIG, participating_clients
        )

        # 7. Update Global Model
        global_model = server.get_global_params()

        # 8. Evaluate & Metrics
        print(f"\n📊 Evaluation:")
        metrics = server.evaluate_global(
            compute_auc=(task_id == data_loader.num_tasks - 1)
        )
        print(
            f"  Accuracy: {metrics['accuracy'] * 100:.2f}% | F1: {metrics['f1_macro'] * 100:.2f}%"
        )

        # 8a. Visualization & Debugging
        print(f"\n🎨 Visualization & Debugging:")
        try:
            # 1. Prediction for Confusion Matrix
            server.global_model.eval()
            X_test_cm = server.test_data["X_test"]
            y_test_cm = server.test_data["y_test"]

            # Subsample for speed if needed (e.g. max 10k samples)
            if len(y_test_cm) > 10000:
                indices = torch.randperm(len(y_test_cm))[:10000]
                X_test_cm = X_test_cm[indices]
                y_test_cm = y_test_cm[indices]

            with torch.no_grad():
                out_cm = server.global_model(X_test_cm.to(server.primary_device))
                preds_cm = out_cm.argmax(dim=1).cpu().numpy()
                y_true_cm = y_test_cm.numpy()

            # 2. Check for collapse (predicting only 1 class)
            unique_preds = set(preds_cm)
            print(f"  🔍 Unique predicted classes: {sorted(list(unique_preds))}")
            if len(unique_preds) == 1:
                print(
                    f"  ⚠️ WARNING: Model is predicting ONLY class {list(unique_preds)[0]}!"
                )

            # 3. Use Visualization Module
            if VISUALIZATION_AVAILABLE:
                task_plot_dir = os.path.join(output_dir, "plots", f"task_{task_id}")
                os.makedirs(task_plot_dir, exist_ok=True)

                # Plot CM
                cm_path = os.path.join(task_plot_dir, "confusion_matrix.png")
                plot_confusion_matrix(
                    y_true_cm,
                    preds_cm,
                    save_path=cm_path,
                    title=f"Confusion Matrix (Task {task_id})",
                )

                # Plot Per-Class
                metrics_path = os.path.join(task_plot_dir, "per_class_metrics.png")
                plot_per_class_metrics(y_true_cm, preds_cm, save_path=metrics_path)
                print(f"  📸 Saved plots to {task_plot_dir}")

        except Exception as e:
            print(f"  ⚠️ Visualization/Debug failed: {e}")

        # Forgetting Calculation
        print("  🔍 Computing Forgetting...")
        del test_data
        gc.collect()

        # Compute accuracy per previous task
        current_task_accuracies = {}
        for prev_tid, path in all_test_data.items():
            loaded_test = torch.load(path)
            server.test_data = loaded_test
            tm = server.evaluate_global(
                seen_classes_only=False
            )  # Evaluate on specific task set
            current_task_accuracies[prev_tid] = tm["accuracy"]
            # Track best
            best_acc_per_task[prev_tid] = max(
                best_acc_per_task.get(prev_tid, 0), tm["accuracy"]
            )

        # Calculate AF
        af = 0.0
        if task_id > 0:
            diffs = [
                max(0, best_acc_per_task[t] - current_task_accuracies[t])
                for t in range(task_id)
            ]
            af = sum(diffs) / len(diffs) if diffs else 0.0
        print(f"  Avg Forgetting: {af * 100:.2f}%")

        # CRITICAL FIX: Feed AF back to trainer for μ reset mechanism (paper Eq. 8)
        # Without this, trainer never knows AF > θ and cannot reset mu_coefficient
        if hasattr(trainer, "update_forgetting"):
            trainer.update_forgetting(current_task_accuracies)

        # Save History
        all_history["task_accuracies"].append(
            {
                "task": task_id,
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "avg_forgetting": af,
                "per_task_acc": current_task_accuracies,
            }
        )

        # 9. Checkpoint
        ckpt_path = os.path.join(output_dir, f"checkpoint_task_{task_id}.pt")
        torch.save(
            {
                "task_id": task_id,
                "model_state_dict": global_model,
                "config": CONFIG,
                "seen_classes": list(seen_classes),
            },
            ckpt_path,
        )
        print(f"💾 Checkpoint saved: {ckpt_path}")

        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()

    # Final Summary
    print("\n" + "=" * 80)
    print("🏁 TRAINING COMPLETE")
    print("=" * 80)
    final = all_history["task_accuracies"][-1]
    print(f"Final Accuracy: {final['accuracy'] * 100:.2f}%")
    print(f"Final Forgetting: {final['avg_forgetting'] * 100:.2f}%")

    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(all_history, f, indent=2)

    # ==========================================================================
    # FCIL VISUALIZATION REPORT
    # ==========================================================================
    if FCIL_VISUALIZATION_AVAILABLE:
        print("\n📊 Generating FCIL Visualization Report...")
        try:
            # Convert history to FCIL format
            # Format: {task_id: [acc_on_task_0, acc_on_task_1, ..., acc_on_task_t]}
            fcil_results = {}
            strategy_name = CONFIG.get("algorithm", "unknown")
            fcil_results[strategy_name] = {}

            for entry in all_history["task_accuracies"]:
                task_id = entry["task"]
                per_task_acc = entry.get("per_task_acc", {})

                # Build accuracy list [acc_task_0, acc_task_1, ...]
                acc_list = []
                for t in range(task_id + 1):
                    acc_list.append(per_task_acc.get(t, entry["accuracy"]))

                fcil_results[strategy_name][task_id] = acc_list

            # Generate FCIL report
            fcil_output_dir = os.path.join(output_dir, "fcil_plots")
            os.makedirs(fcil_output_dir, exist_ok=True)

            # 1. Task Accuracy Heatmap
            plot_task_accuracy_heatmap(
                fcil_results[strategy_name],
                strategy_name=strategy_name,
                save_path=os.path.join(fcil_output_dir, "task_accuracy_heatmap.png"),
            )

            # 2. Incremental Accuracy Curve (single strategy)
            plot_incremental_accuracy_curve(
                fcil_results,
                save_path=os.path.join(fcil_output_dir, "incremental_accuracy.png"),
                title=f"Incremental Learning - {strategy_name}",
            )

            # 3. Compute and print FCIL metrics
            aia = compute_average_incremental_accuracy(fcil_results[strategy_name])
            forgetting = compute_forgetting_measure(fcil_results[strategy_name])

            print(f"\n📈 FCIL Metrics:")
            print(f"  • Final AIA: {aia[-1] * 100:.2f}%")
            print(
                f"  • Avg Forgetting: {sum(forgetting) / len(forgetting) * 100:.2f}%"
                if forgetting
                else "  • Avg Forgetting: N/A"
            )

            # Save FCIL metrics
            fcil_metrics = {
                "strategy": strategy_name,
                "final_aia": aia[-1] if aia else 0,
                "aia_per_task": aia,
                "forgetting_per_task": forgetting,
                "avg_forgetting": sum(forgetting) / len(forgetting)
                if forgetting
                else 0,
            }
            with open(os.path.join(fcil_output_dir, "fcil_metrics.json"), "w") as f:
                json.dump(fcil_metrics, f, indent=2)

            print(f"✅ FCIL report saved to: {fcil_output_dir}")

        except Exception as e:
            print(f"⚠️ FCIL visualization failed: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
