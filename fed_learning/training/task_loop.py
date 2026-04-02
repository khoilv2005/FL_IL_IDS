"""
Task Loop - Main orchestration for Federated Class Incremental Learning.

This module contains the complete training pipeline extracted from the
main training script, so that the entry point only needs CONFIG.

Usage:
    from fed_learning.training.task_loop import run_incremental_training
    run_incremental_training(CONFIG)
"""

import os
import gc
import json
from datetime import datetime
from typing import Dict, Any

import torch
import numpy as np

from fed_learning.data.incremental_loader import IncrementalDataLoader
from fed_learning.strategies import get_strategy
from fed_learning.training.runner import train_federated_multigpu
from fed_learning.utils.seed import set_seed
from fed_learning.utils.cleanup import cleanup_temp_folders
from fed_learning.factories.client_factory import (
    get_or_create_persistent_client,
    update_client_data,
)
from fed_learning.factories.server_factory import create_server
from fed_learning.training.post_task import post_task_processing
from fed_learning.decentralized.runner import run_decentralized_incremental_training

# Visualization imports (optional)
try:
    from fed_learning.visualization.metrics import (
        plot_confusion_matrix,
        plot_per_class_metrics,
    )

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

try:
    from fed_learning.visualization.fcil_plots import (
        plot_incremental_accuracy_curve,
        plot_task_accuracy_heatmap,
        compute_average_incremental_accuracy,
        compute_forgetting_measure,
    )

    FCIL_VISUALIZATION_AVAILABLE = True
except ImportError:
    FCIL_VISUALIZATION_AVAILABLE = False


# =============================================================================
# ALGORITHM-SPECIFIC TRAINING DISPATCHERS
# =============================================================================


def _train_nice(server, participating_clients, config, data_loader, task_id):
    """NICE: Phase-based training with single federated round."""
    nice_rounds = 1  # NICE uses internal phases, only 1 federated round needed
    is_last_task = task_id == data_loader.get_num_tasks() - 1

    print(
        f"\n  === NICE Training ({nice_rounds} rounds) ==="
        f"{' [LAST EPISODE: tau=100%]' if is_last_task else ''}"
    )

    for r in range(nice_rounds):
        server.train_round(
            participating_clients=participating_clients,
            verbose=True,
        )
        if (r + 1) % config.get("eval_every", 1) == 0:
            eval_metrics = server.evaluate_global()
            print(
                f"    Round {r + 1}/{nice_rounds} -> "
                f"Acc: {eval_metrics['accuracy'] * 100:.2f}%"
            )


def _train_der(server, participating_clients, config, trainer):
    """DER: Two-stage federated training (representation + classifier)."""
    stage1_rounds = config.get("der_stage1_rounds", config["rounds_per_task"])
    stage2_rounds = config.get("der_stage2_rounds", 3)

    # Stage 1: Representation learning
    if hasattr(trainer, "set_stage"):
        trainer.set_stage(1)

    print(f"\n  === DER Stage 1: Representation Learning ({stage1_rounds} rounds) ===")
    for r in range(stage1_rounds):
        server.train_round(
            participating_clients=participating_clients,
            stage=1,
            verbose=True,
        )
        if (r + 1) % config.get("eval_every", 1) == 0:
            eval_metrics = server.evaluate_global()
            print(
                f"    Round {r + 1}/{stage1_rounds} → "
                f"Acc: {eval_metrics['accuracy'] * 100:.2f}%"
            )

    # Stage 2: Classifier learning
    if hasattr(trainer, "set_stage"):
        trainer.set_stage(2)

    # Paper Section 3.2: Re-initialize classifier once before Stage 2
    if hasattr(server.global_model, "reset_classifier"):
        server.global_model.reset_classifier()
        print("  → Classifier H_t re-initialized (paper Section 3.2)")

    print(f"\n  === DER Stage 2: Classifier Learning ({stage2_rounds} rounds) ===")
    for r in range(stage2_rounds):
        server.train_round(
            participating_clients=participating_clients,
            stage=2,
            verbose=True,
        )
        if (r + 1) % config.get("eval_every", 1) == 0:
            eval_metrics = server.evaluate_global()
            print(
                f"    Round {r + 1}/{stage2_rounds} → "
                f"Acc: {eval_metrics['accuracy'] * 100:.2f}%"
            )

    return stage1_rounds  # Return for weight alignment check


def _train_glfc(server, participating_clients, config):
    """GLFC: Standard round-based training with exemplar management."""
    glfc_rounds = config.get("rounds_per_task", 5)
    print(f"\n  === GLFC Training ({glfc_rounds} rounds) ===")

    for r in range(glfc_rounds):
        server.train_round(
            participating_clients=participating_clients,
            verbose=True,
        )
        if (r + 1) % config.get("eval_every", 1) == 0:
            eval_metrics = server.evaluate_global()
            print(
                f"    Round {r + 1}/{glfc_rounds} -> "
                f"Acc: {eval_metrics['accuracy'] * 100:.2f}%"
            )

    # After all rounds: coordinate exemplar update
    print("  Updating exemplar sets for all participants...")
    server.coordinate_exemplar_update(participating_clients, verbose=True)


def _train_refed(server, participating_clients, config, task_id):
    """Re-Fed: PIM-based caching + standard FedAvg training."""
    refed_rounds = config.get("rounds_per_task", 5)

    # Step 1: Coordinate PIM caching BEFORE training rounds
    if task_id > 0:
        print(f"\n  === Re-Fed: PIM Caching (Task {task_id}) ===")
    else:
        print(f"\n  === Re-Fed: Initial Caching (Task 0) ===")
    server.coordinate_pim_caching(participating_clients, verbose=True)

    # Step 2: Standard federated training on cached + new data
    print(f"\n  === Re-Fed Training ({refed_rounds} rounds) ===")
    for r in range(refed_rounds):
        server.train_round(
            participating_clients=participating_clients,
            verbose=True,
        )
        if (r + 1) % config.get("eval_every", 1) == 0:
            eval_metrics = server.evaluate_global()
            print(
                f"    Round {r + 1}/{refed_rounds} -> "
                f"Acc: {eval_metrics['accuracy'] * 100:.2f}%"
            )


def _train_plexus(server, participating_clients, config):
    """Plexus: Decentralized FL with rotating aggregator and hash-based sampling."""
    plexus_rounds = config.get("rounds_per_task", 5)
    print(f"\n  === Plexus Decentralized Training ({plexus_rounds} rounds) ===")

    for r in range(plexus_rounds):
        server.train_round(
            participating_clients=participating_clients,
            verbose=True,
        )
        if (r + 1) % config.get("eval_every", 1) == 0:
            eval_metrics = server.evaluate_global()
            print(
                f"    Round {r + 1}/{plexus_rounds} -> "
                f"Acc: {eval_metrics['accuracy'] * 100:.2f}%"
            )


# =============================================================================
# EVALUATION & VISUALIZATION
# =============================================================================


def _evaluate_and_visualize(server, task_id, output_dir, config):
    """Run evaluation, generate confusion matrix, and debug diagnostics."""
    print(f"\n🎨 Visualization & Debugging:")
    try:
        server.global_model.eval()
        X_test_cm = server.test_data["X_test"]
        y_test_cm = server.test_data["y_test"]

        # Subsample for speed (max 10k)
        if len(y_test_cm) > 10000:
            indices = torch.randperm(len(y_test_cm))[:10000]
            X_test_cm = X_test_cm[indices]
            y_test_cm = y_test_cm[indices]

        with torch.no_grad():
            out_cm = server.global_model(X_test_cm.to(server.primary_device))
            preds_cm = out_cm.argmax(dim=1).cpu().numpy()
            y_true_cm = y_test_cm.numpy()

        # Check for collapse
        unique_preds = set(preds_cm)
        print(f"  🔍 Unique predicted classes: {sorted(list(unique_preds))}")

        # DEBUG: Output layer weights
        with torch.no_grad():
            if hasattr(server.global_model, "fc2"):
                fc2_weight = server.global_model.fc2.weight
            elif hasattr(server.global_model, "classifier"):
                fc2_weight = server.global_model.classifier.weight
            else:
                fc2_weight = None
            if fc2_weight is not None:
                print(f"  DEBUG[2]: Output layer weight norms per class:")
                for c in sorted(set(y_true_cm)):
                    if c < fc2_weight.shape[0]:
                        print(f"    Class {c}: {fc2_weight[c].norm().item():.4f}")

        # DEBUG: Mean prediction probability per class
        with torch.no_grad():
            probs_cm = torch.softmax(out_cm, dim=1)
            print(f"  DEBUG[7]: Mean prediction probability per class:")
            for c in sorted(set(y_true_cm)):
                print(f"    Class {c}: {probs_cm[:, c].mean().item():.6f}")

        if len(unique_preds) == 1:
            print(
                f"  ⚠️ WARNING: Model is predicting ONLY class {list(unique_preds)[0]}!"
            )

        # Save plots
        if VISUALIZATION_AVAILABLE:
            task_plot_dir = os.path.join(output_dir, "plots", f"task_{task_id}")
            os.makedirs(task_plot_dir, exist_ok=True)

            cm_path = os.path.join(task_plot_dir, "confusion_matrix.png")
            plot_confusion_matrix(
                y_true_cm,
                preds_cm,
                save_path=cm_path,
                title=f"Confusion Matrix (Task {task_id})",
            )

            metrics_path = os.path.join(task_plot_dir, "per_class_metrics.png")
            plot_per_class_metrics(y_true_cm, preds_cm, save_path=metrics_path)
            print(f"  📸 Saved plots to {task_plot_dir}")

    except Exception as e:
        print(f"  ⚠️ Visualization/Debug failed: {e}")


def _compute_forgetting(server, task_id, all_test_data, best_acc_per_task, trainer):
    """
    Compute per-task accuracy and average forgetting (AF).

    Returns:
        (current_task_accuracies, af) where af is the average forgetting measure.
    """
    print("  🔍 Computing Forgetting...")

    current_task_accuracies = {}
    for prev_tid, path in all_test_data.items():
        loaded_test = torch.load(path)
        server.test_data = loaded_test
        tm = server.evaluate_global(seen_classes_only=False)
        current_task_accuracies[prev_tid] = tm["accuracy"]
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

    # Feed AF back to trainer for μ reset mechanism (paper Eq. 8)
    if hasattr(trainer, "update_forgetting"):
        trainer.update_forgetting(current_task_accuracies)

    return current_task_accuracies, af


def _generate_fcil_report(all_history, config, output_dir):
    """Generate FCIL visualization report with heatmap and metrics."""
    if not FCIL_VISUALIZATION_AVAILABLE:
        return

    print("\n📊 Generating FCIL Visualization Report...")
    try:
        strategy_name = config.get("algorithm", "unknown")
        fcil_results = {strategy_name: {}}

        for entry in all_history["task_accuracies"]:
            tid = entry["task"]
            per_task_acc = entry.get("per_task_acc", {})

            acc_list = []
            for t in range(tid + 1):
                acc_list.append(per_task_acc.get(t, entry["accuracy"]))
            fcil_results[strategy_name][tid] = acc_list

        fcil_output_dir = os.path.join(output_dir, "fcil_plots")
        os.makedirs(fcil_output_dir, exist_ok=True)

        # 1. Task Accuracy Heatmap
        plot_task_accuracy_heatmap(
            fcil_results[strategy_name],
            strategy_name=strategy_name,
            save_path=os.path.join(fcil_output_dir, "task_accuracy_heatmap.png"),
        )

        # 2. Incremental Accuracy Curve
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
            "avg_forgetting": sum(forgetting) / len(forgetting) if forgetting else 0,
        }
        with open(os.path.join(fcil_output_dir, "fcil_metrics.json"), "w") as f:
            json.dump(fcil_metrics, f, indent=2)

        print(f"✅ FCIL report saved to: {fcil_output_dir}")

    except Exception as e:
        print(f"⚠️ FCIL visualization failed: {e}")
        import traceback

        traceback.print_exc()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def run_incremental_training(config: Dict[str, Any]):
    """
    Run the complete Federated Class Incremental Learning pipeline.

    This is the main entry point. The caller only needs to provide
    a CONFIG dict; all training logic is handled here.

    Args:
        config: Training configuration dict. Required keys:
            - "algorithm": Algorithm name (e.g., "cgofed", "nice", "der")
            - "data_dir": Path to data directory
            - "output_dir": Base output directory
            - "total_classes", "base_classes", "classes_per_task": Task structure
            - "rounds_per_task", "local_epochs", "learning_rate", "batch_size"
            - Algorithm-specific parameters (see CONFIG in training script)
    """
    mode = config.get("mode", "fed_il").lower()
    if mode == "il":
        from fed_learning.training.local_task_loop import run_local_incremental_training

        return run_local_incremental_training(config)
    elif mode == "decentralized":
        # Plexus decentralized FL (no central server)
        return run_decentralized_incremental_training(config)
    if mode != "fed_il":
        raise ValueError("Unsupported mode. Use 'fed_il', 'il', or 'decentralized'.")

    # 1. Setup
    set_seed(config.get("random_seed", 42))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{config['output_dir']}_{config['algorithm']}_{ts}"
    os.makedirs(output_dir, exist_ok=True)

    # Save Config
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print(f"🚀 FEDERATED CLASS INCREMENTAL LEARNING - {config['algorithm'].upper()}")
    print("=" * 80)

    cleanup_temp_folders()
    os.makedirs("./temp_test_data", exist_ok=True)

    # 2. Load Data
    data_loader = IncrementalDataLoader(data_dir=config["data_dir"])
    print(f"\n{data_loader}")

    # Update config with data-derived params
    config["input_shape"] = data_loader.input_shape
    config["num_classes"] = config["total_classes"]

    # 3. Get Strategy (Trainer & Aggregator)
    trainer, aggregator = get_strategy(**config)
    print(f"✓ Trainer: {trainer.__class__.__name__}")
    print(f"✓ Aggregator: {aggregator.__class__.__name__}")

    # 4. State Variables
    global_model = None
    all_history = {"task_accuracies": [], "task_forgetting": []}
    all_test_data = {}
    best_acc_per_task = {}
    persistent_clients: Dict[int, object] = {}

    # 5. Task Loop
    for task_id in range(data_loader.get_num_tasks()):
        print(
            f"\n{'=' * 80}\n📚 TASK {task_id}/{data_loader.get_num_tasks()}\n{'=' * 80}"
        )

        # 5a. Prepare Data
        new_classes = data_loader.get_task_classes(task_id)
        seen_classes = []
        for t in range(task_id + 1):
            seen_classes.extend(data_loader.get_task_classes(t))

        # Get client data
        client_data_map = {}
        for cid in data_loader.get_all_client_ids():
            X, y = data_loader.get_client_data(cid, task_id)
            if len(y) > 0:
                client_data_map[cid] = {"X_train": X, "y_train": y}

        print(f"  Clients with data: {len(client_data_map)}")
        if not client_data_map:
            print("  ⚠️ No data for this task, skipping.")
            continue

        # 5b. Prepare Test Data
        test_X, test_y = data_loader.get_test_data(task_id, cumulative=True)
        test_data = {"X_test": test_X, "y_test": test_y}
        test_data_path = os.path.join("./temp_test_data", f"test_task_{task_id}.pt")
        torch.save(test_data, test_data_path)
        all_test_data[task_id] = test_data_path

        # 5c. Manage Persistent Clients
        if task_id == 0:
            persistent_clients = {}

        participating_clients = []
        for cid, data in client_data_map.items():
            client = get_or_create_persistent_client(
                cid, data, config, persistent_clients
            )
            update_client_data(client, data, task_id, new_classes)
            participating_clients.append(client)

        # 5d. Prepare Server
        task_config = {
            **config,
            "num_classes": config["total_classes"],
            "num_rounds": config["rounds_per_task"],
        }

        # Dynamic param adjustment (e.g., LwF alpha decay)
        if "lwf" in config["algorithm"]:
            current_alpha = (
                config["lwf_alpha"]
                * (config.get("lwf_alpha_scale", 1.0) ** max(0, task_id - 1))
                if task_id > 0
                else config["lwf_alpha"]
            )
            task_config["lwf_alpha"] = current_alpha
            if hasattr(trainer, "lwf_alpha"):
                trainer.lwf_alpha = current_alpha
            print(f"   LwF Alpha: {current_alpha:.4f}")

        # NICE: pass is_last_task flag
        if config["algorithm"].lower() == "nice":
            is_last_task = task_id == data_loader.get_num_tasks() - 1
            task_config["is_last_task"] = is_last_task

        server = create_server(config, participating_clients, test_data, task_config)

        if global_model is not None:
            server.set_global_params(global_model)

        server.trainer = trainer
        server.aggregator = aggregator

        if hasattr(server, "set_task"):
            server.set_task(task_id, new_classes, seen_classes)
        if hasattr(trainer, "set_task"):
            trainer.set_task(task_id, new_classes)
        if hasattr(aggregator, "set_task"):
            aggregator.set_task(task_id)

        # 5e. Train (algorithm-specific dispatch)
        print(f"\n🎯 Training on {len(new_classes)} new classes...")
        algo = config["algorithm"].lower()

        if algo == "nice":
            _train_nice(server, participating_clients, config, data_loader, task_id)
        elif algo == "der":
            _train_der(server, participating_clients, config, trainer)
            # Weight Alignment after Stage 2
            if task_id > 0 and hasattr(server.global_model, "weight_align"):
                server.global_model.weight_align(len(new_classes))
        elif algo == "glfc":
            _train_glfc(server, participating_clients, config)
        elif algo == "refed":
            _train_refed(server, participating_clients, config, task_id)
        elif algo == "plexus":
            _train_plexus(server, participating_clients, config)
        else:
            train_federated_multigpu(server, task_config)

        # 5f. Post-Task Processing
        post_task_processing(
            trainer, server, client_data_map, config, participating_clients
        )

        # 5g. Update Global Model
        global_model = server.get_global_params()

        # 5h. Evaluate
        print(f"\n📊 Evaluation:")
        metrics = server.evaluate_global(
            compute_auc=(task_id == data_loader.num_tasks - 1)
        )
        print(
            f"  Accuracy: {metrics['accuracy'] * 100:.2f}% | "
            f"F1: {metrics['f1_macro'] * 100:.2f}%"
        )

        # 5i. Visualization & Debug
        _evaluate_and_visualize(server, task_id, output_dir, config)

        # 5j. Forgetting Calculation
        del test_data
        gc.collect()

        current_task_accuracies, af = _compute_forgetting(
            server, task_id, all_test_data, best_acc_per_task, trainer
        )

        # 5k. Save History
        all_history["task_accuracies"].append(
            {
                "task": task_id,
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "avg_forgetting": af,
                "per_task_acc": current_task_accuracies,
            }
        )

        # 5l. Checkpoint
        ckpt_path = os.path.join(output_dir, f"checkpoint_task_{task_id}.pt")
        torch.save(
            {
                "task_id": task_id,
                "model_state_dict": global_model,
                "config": config,
                "seen_classes": list(seen_classes),
            },
            ckpt_path,
        )
        print(f"💾 Checkpoint saved: {ckpt_path}")

        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()

    # 6. Final Summary
    print("\n" + "=" * 80)
    print("🏁 TRAINING COMPLETE")
    print("=" * 80)
    final = all_history["task_accuracies"][-1]
    print(f"Final Accuracy: {final['accuracy'] * 100:.2f}%")
    print(f"Final Forgetting: {final['avg_forgetting'] * 100:.2f}%")

    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(all_history, f, indent=2)

    # 7. FCIL Report
    _generate_fcil_report(all_history, config, output_dir)
