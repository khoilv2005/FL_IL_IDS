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
from typing import Dict, Any, List, Optional

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
from fed_learning.training.resume_state import (
    build_continuation_state,
    load_continuation_state,
    restore_aggregator_state,
    restore_client_state,
    restore_server_state,
    restore_trainer_state,
    save_continuation_state,
)
from fed_learning.plexus import run_plexus_training

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


def _refresh_server_clients(server, clients, config, test_data, task_config):
    """Refresh task participants while preserving server state when possible."""
    if hasattr(server, "test_data"):
        server.test_data = test_data
    if hasattr(server, "config"):
        server.config = task_config

    if hasattr(server, "update_clients"):
        server.update_clients(clients)
        return server

    if hasattr(server, "clients"):
        server.clients = clients
        print(
            "  Warning: server missing update_clients(); "
            "falling back to direct clients assignment."
        )
        return server

    print(
        "  Warning: server cannot refresh clients in-place; "
        "recreating server for this task."
    )
    return create_server(config, clients, test_data, task_config)


def _resolve_nice_schedule(config: Dict[str, Any]) -> tuple[int, int, int]:
    """
    Resolve NICE schedule.

    NICE in this repo is phase-based:

        max_phases x phase_epochs

    For tracking/debugging, we expose those phases explicitly in the task loop,
    so each phase gets its own metrics/checkpoint while the total local epoch
    budget per task remains unchanged.
    """
    max_phases = max(1, int(config.get("nice_max_phases", 5)))
    phase_epochs = max(1, int(config.get("nice_phase_epochs", 5)))
    effective_local_epochs = max_phases * phase_epochs
    return max_phases, phase_epochs, effective_local_epochs


# =============================================================================
# ALGORITHM-SPECIFIC TRAINING DISPATCHERS
# =============================================================================


def _train_nice(server, participating_clients, config, data_loader, task_id):
    """NICE: Phase-based training with single federated round."""
    max_phases, phase_epochs, effective_local_epochs = _resolve_nice_schedule(config)
    nice_rounds = max_phases
    is_last_task = task_id == data_loader.get_num_tasks() - 1

    print(
        f"\n  === NICE Training ({nice_rounds} phases) ==="
        f"\n  NICE local schedule: {max_phases} phases x {phase_epochs} epochs"
        f" = {effective_local_epochs} local epochs/client"
        f"{' [LAST EPISODE: tau=100%]' if is_last_task else ''}"
    )

    for r in range(nice_rounds):
        server.train_round(
            participating_clients=participating_clients,
            verbose=True,
            phase_offset=r,
            max_phases_override=1,
        )
        if (r + 1) % config.get("eval_every", 1) == 0:
            eval_metrics = server.evaluate_global()
            print(
                f"    Phase {r + 1}/{nice_rounds} -> "
                f"Acc: {eval_metrics['accuracy'] * 100:.2f}%"
            )


def _resolve_der_round_split(config: Dict[str, Any]) -> tuple[int, int]:
    """
    Resolve DER stage rounds from config.

    If stage-specific round counts are explicitly configured, keep them.
    Otherwise, treat rounds_per_task as the total DER budget and split it using
    the legacy 3:2 ratio between stage 1 and stage 2.
    """
    if "der_stage1_rounds" in config or "der_stage2_rounds" in config:
        stage1_rounds = int(config.get("der_stage1_rounds", config["rounds_per_task"]))
        stage2_rounds = int(config.get("der_stage2_rounds", 3))
        return max(1, stage1_rounds), max(1, stage2_rounds)

    total_rounds = max(1, int(config.get("rounds_per_task", 1)))
    if total_rounds == 1:
        return 1, 1

    stage1_rounds = max(1, round(total_rounds * 3 / 5))
    stage2_rounds = max(1, total_rounds - stage1_rounds)
    if stage1_rounds + stage2_rounds != total_rounds:
        stage2_rounds = max(1, total_rounds - stage1_rounds)
    return stage1_rounds, stage2_rounds


def _train_der(server, participating_clients, config, trainer):
    """DER: Two-stage federated training (representation + classifier)."""
    stage1_rounds, stage2_rounds = _resolve_der_round_split(config)

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


def _train_plexus_der(server, participating_clients, config, trainer):
    """PlexusDER: Decentralized DER with two-stage training."""
    stage1_rounds, stage2_rounds = _resolve_der_round_split(config)

    # Stage 1: Representation learning
    if hasattr(trainer, "set_stage"):
        trainer.set_stage(1)

    print(f"\n  === PlexusDER Stage 1: Representation Learning ({stage1_rounds} rounds) ===")
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

    if hasattr(server.global_model, "reset_classifier"):
        server.global_model.reset_classifier()
        print("  → Classifier H_t re-initialized (paper Section 3.2)")

    print(f"\n  === PlexusDER Stage 2: Classifier Learning ({stage2_rounds} rounds) ===")
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


def _train_plexus_nice(server, participating_clients, config, data_loader, task_id):
    """PlexusNICE: Decentralized NICE with phase-based training."""
    max_phases, phase_epochs, effective_local_epochs = _resolve_nice_schedule(config)
    nice_rounds = max_phases
    is_last_task = task_id == data_loader.get_num_tasks() - 1

    print(
        f"\n  === PlexusNICE Training ({nice_rounds} phases) ==="
        f"\n  PlexusNICE local schedule: {max_phases} phases x {phase_epochs} epochs"
        f" = {effective_local_epochs} local epochs/client"
        f"{' [LAST EPISODE: tau=100%]' if is_last_task else ''}"
    )

    for r in range(nice_rounds):
        server.train_round(
            participating_clients=participating_clients,
            is_last_task=is_last_task,
            verbose=True,
        )
        if (r + 1) % config.get("eval_every", 1) == 0:
            eval_metrics = server.evaluate_global()
            print(
                f"    Phase {r + 1}/{nice_rounds} -> "
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
        tm = server.evaluate_global(seen_classes_only=True)
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
        details = []
        for t in range(task_id):
            current = current_task_accuracies.get(t, 0.0)
            best = best_acc_per_task.get(t, 0.0)
            details.append(f"T{t}: current={current * 100:.2f}%, best={best * 100:.2f}%")
        if details:
            print("    Per-task old accuracy -> " + " | ".join(details))
    print(f"  Avg Forgetting: {af * 100:.2f}%")

    # Feed AF back to trainer for μ reset mechanism (paper Eq. 8)
    if hasattr(trainer, "update_forgetting"):
        trainer.update_forgetting(current_task_accuracies)

    return current_task_accuracies, af


def _write_training_history(output_dir: str, history: Dict[str, Any]):
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(history, f, indent=2, default=str)

    with open(os.path.join(output_dir, "round_metrics.json"), "w") as f:
        json.dump(history.get("round_metrics", []), f, indent=2, default=str)


def _write_phase_outputs(
    output_dir: str,
    history: Dict[str, Any],
    config: Dict[str, Any],
    completed_task_id: int,
):
    """
    Persist all aggregate outputs after each split-run phase/task.

    Split Kaggle runs may stop at task 2, 3, 4, or 5, so every completed task
    must leave a full metrics/report bundle, not only the final full run.
    """
    _write_training_history(output_dir, history)

    with open(os.path.join(output_dir, "task_metrics.json"), "w") as f:
        json.dump(history.get("task_accuracies", []), f, indent=2, default=str)

    phase_summary = {
        "algorithm": config.get("algorithm"),
        "mode": config.get("mode", "fed_il"),
        "task_start": int(config.get("task_start", 0)),
        "task_end": int(config.get("task_end", completed_task_id)),
        "completed_task": int(completed_task_id),
        "num_task_records": len(history.get("task_accuracies", [])),
        "num_round_records": len(history.get("round_metrics", [])),
    }
    with open(os.path.join(output_dir, "phase_summary.json"), "w") as f:
        json.dump(phase_summary, f, indent=2, default=str)

    if history.get("task_accuracies"):
        _generate_fcil_report(history, config, output_dir)


def _resolve_output_dir(config: Dict[str, Any], mode: str, algorithm: str) -> str:
    """Resolve output directory for fresh or resumed runs."""
    if config.get("resume_output_dir"):
        output_dir = config["resume_output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    resume_state_path = config.get("resume_state_path")
    if resume_state_path:
        output_dir = os.path.dirname(resume_state_path)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if mode == "il":
        output_dir = f"{config['output_dir']}_{algorithm}_{mode}_{ts}"
    elif mode == "decentralized":
        output_dir = f"{config['output_dir']}_plexus_{ts}"
    else:
        output_dir = f"{config['output_dir']}_{algorithm}_{ts}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _inject_persistent_algorithm_paths(
    config: Dict[str, Any], output_dir: str
) -> Dict[str, Any]:
    """
    Resolve algorithm-specific artifact directories under the run output.

    This is important on Kaggle because `/tmp` is session-scoped. For split-run
    continuation, artifacts such as CGoFed SVD bases and history snapshots must
    live alongside the run outputs unless the user explicitly overrides them.
    """
    resolved = dict(config)
    algorithm = resolved.get("algorithm", "").lower()

    if algorithm == "cgofed":
        cgofed_root = os.path.join(output_dir, "cgofed_artifacts")
        resolved.setdefault("cgofed_temp_dir", os.path.join(cgofed_root, "svd"))
        resolved.setdefault(
            "cgofed_history_dir", os.path.join(cgofed_root, "history")
        )
        os.makedirs(resolved["cgofed_temp_dir"], exist_ok=True)
        os.makedirs(resolved["cgofed_history_dir"], exist_ok=True)

    return resolved


def _resolve_task_bounds(
    config: Dict[str, Any], num_tasks: int, resume_state: Optional[Dict[str, Any]] = None
) -> tuple[int, int]:
    """Resolve start/end task bounds for a phase."""
    resume_from_task = 0
    if resume_state is not None:
        resume_from_task = int(resume_state["meta"].get("resume_from_task", 0))

    task_start = int(config.get("task_start", resume_from_task))
    task_end = int(config.get("task_end", num_tasks - 1))

    if task_start < 0 or task_start >= num_tasks:
        raise ValueError(f"task_start out of range: {task_start}")
    if task_end < task_start or task_end >= num_tasks:
        raise ValueError(f"task_end out of range: {task_end}")
    return task_start, task_end


def _rebuild_prior_test_snapshots(
    data_loader: IncrementalDataLoader, completed_task_ids: List[int]
) -> Dict[int, str]:
    """Rebuild cumulative test snapshots for previously completed tasks."""
    rebuilt = {}
    for prev_tid in completed_task_ids:
        test_X, test_y = data_loader.get_test_data(prev_tid, cumulative=True)
        test_data = {"X_test": test_X, "y_test": test_y}
        test_data_path = os.path.join("./temp_test_data", f"test_task_{prev_tid}.pt")
        torch.save(test_data, test_data_path)
        rebuilt[prev_tid] = test_data_path
    return rebuilt


def _save_fed_round_checkpoint(
    output_dir: str,
    task_id: int,
    round_id: int,
    global_params,
    config: Dict[str, Any],
    seen_classes: List[int],
    train_loss: float,
    round_time: float,
    metrics: Dict[str, Any],
    avg_forgetting: float,
):
    ckpt_path = os.path.join(output_dir, f"checkpoint_task_{task_id}_round_{round_id}.pt")
    torch.save(
        {
            "task_id": task_id,
            "round_id": round_id,
            "model_state_dict": global_params,
            "config": config,
            "seen_classes": list(seen_classes),
            "metrics": {
                "train_loss": train_loss,
                "round_time": round_time,
                **metrics,
                "avg_forgetting": avg_forgetting,
            },
        },
        ckpt_path,
    )
    print(f"  💾 Round checkpoint saved: {ckpt_path}")


def _save_fed_task_checkpoint(
    output_dir: str,
    task_id: int,
    final_round_id: int,
    global_params,
    config: Dict[str, Any],
    seen_classes: List[int],
    metrics: Dict[str, Any],
    avg_forgetting: float,
    task_accuracies: Optional[Dict[int, float]] = None,
):
    ckpt_path = os.path.join(output_dir, f"checkpoint_task_{task_id}.pt")
    torch.save(
        {
            "task_id": task_id,
            "final_round_id": final_round_id,
            "model_state_dict": global_params,
            "config": config,
            "seen_classes": list(seen_classes),
            "metrics": {**metrics, "avg_forgetting": avg_forgetting},
            "task_accuracies": task_accuracies or {},
        },
        ckpt_path,
    )
    print(f"💾 Checkpoint saved: {ckpt_path}")


def _record_fed_round(
    server,
    output_dir: str,
    history: Dict[str, Any],
    task_id: int,
    round_id: int,
    train_loss: float,
    round_time: float,
    all_test_data,
    best_acc_per_task,
    trainer,
    config: Dict[str, Any],
    seen_classes: List[int],
    is_last_task: bool,
    compute_forgetting: bool = True,
    evaluate: bool = True,
    save_checkpoint: bool = True,
):
    if evaluate:
        metrics = server.evaluate_global(compute_auc=False)
    else:
        metrics = {}

    if evaluate and compute_forgetting:
        current_task_accuracies, af = _compute_forgetting(
            server, task_id, all_test_data, best_acc_per_task, trainer
        )
    else:
        current_task_accuracies, af = {}, None

    round_record = {
        "task": task_id,
        "round": round_id,
        "train_loss": train_loss,
        "round_time": round_time,
        "test_loss": metrics.get("loss"),
        "accuracy": metrics.get("accuracy"),
        "precision_macro": metrics.get("precision_macro"),
        "recall_macro": metrics.get("recall_macro"),
        "f1_macro": metrics.get("f1_macro"),
        "f1_weighted": metrics.get("f1_weighted"),
        "avg_forgetting": af,
        "evaluated": evaluate,
    }
    history["round_metrics"].append(round_record)
    _write_training_history(output_dir, history)

    if evaluate:
        af_text = f"{af * 100:.2f}%" if af is not None else "N/A (final round only)"
        print(
            "    Metrics -> "
            f"train_loss={train_loss:.4f}, test_loss={metrics['loss']:.4f}, "
            f"accuracy={metrics['accuracy'] * 100:.2f}%, "
            f"f1={metrics['f1_macro'] * 100:.2f}%, "
            f"precision={metrics['precision_macro'] * 100:.2f}%, "
            f"recall={metrics['recall_macro'] * 100:.2f}%, "
            f"AF={af_text}"
        )
    else:
        print(
            "    Metrics skipped -> "
            f"train_loss={train_loss:.4f}, eval_every={config.get('eval_every', 1)}"
        )

    if save_checkpoint:
        _save_fed_round_checkpoint(
            output_dir,
            task_id,
            round_id,
            server.get_global_params(),
            config,
            seen_classes,
            train_loss,
            round_time,
            metrics,
            af,
        )
    return round_record


def _run_tracked_rounds(
    server,
    train_round_fn,
    total_rounds: int,
    task_id: int,
    output_dir: str,
    history: Dict[str, Any],
    all_test_data,
    best_acc_per_task,
    trainer,
    config: Dict[str, Any],
    seen_classes: List[int],
    is_last_task: bool,
    round_start: int = 0,
    round_total_last: Optional[int] = None,
    label: str = "",
):
    if round_total_last is None:
        round_total_last = round_start + total_rounds - 1

    eval_every = max(1, int(config.get("eval_every", 1)))
    checkpoint_every = config.get("round_checkpoint_every", 1)
    if checkpoint_every is not None:
        checkpoint_every = max(1, int(checkpoint_every))

    last_record = None
    for local_round in range(total_rounds):
        round_id = round_start + local_round
        round_suffix = f" {label}" if label else ""
        print(f"  🔁 ROUND {round_id}/{round_total_last}{round_suffix}")
        round_result = train_round_fn(local_round) or {}
        is_final_round = round_id == round_total_last
        evaluate_round = is_final_round or ((round_id + 1) % eval_every == 0)
        save_round_checkpoint = checkpoint_every is not None and (
            is_final_round or ((round_id + 1) % checkpoint_every == 0)
        )
        last_record = _record_fed_round(
            server,
            output_dir,
            history,
            task_id,
            round_id,
            float(round_result.get("train_loss", 0.0)),
            float(round_result.get("round_time", 0.0)),
            all_test_data,
            best_acc_per_task,
            trainer,
            config,
            seen_classes,
            is_last_task,
            compute_forgetting=is_final_round,
            evaluate=evaluate_round,
            save_checkpoint=save_round_checkpoint,
        )
    return last_record


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
            fcil_results[strategy_name][tid] = [entry["accuracy"]]

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


def _run_plexus_training(config: Dict[str, Any]) -> Dict:
    """
    Run pure Plexus decentralized training (Algorithm 1 & 2, no server, no incremental).

    This is the entry point when mode="decentralized".
    Plexus paper: Dhasade et al., EuroMLSys 2025

    Args:
        config: Training configuration dict. Required keys:
            - "data_dir": Path to data
            - "output_dir": Output directory
            - "total_classes": Total number of classes
            - "input_shape": Input shape for model
            - "num_rounds": Number of Plexus rounds
            - "plexus_sample_size": K (default 13)
            - "plexus_success_fraction": s_f (default 0.8)
            - "local_epochs", "learning_rate", "batch_size"

    Returns:
        Dict with task_accuracies (Plexus has 1 "task", no incremental)
    """
    # 1. Setup
    set_seed(config.get("random_seed", 42))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{config['output_dir']}_plexus_{ts}"
    os.makedirs(output_dir, exist_ok=True)

    # Save Config
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("🚀 PURE PLEXUS — Decentralized FL (No Server, No Incremental)")
    print("=" * 80)

    # 2. Load Data
    data_loader = IncrementalDataLoader(data_dir=config["data_dir"])
    print(f"\n{data_loader}")

    config["input_shape"] = data_loader.input_shape
    config["num_classes"] = config["total_classes"]

    # 3. Create model template
    from fed_learning.models.cnn_gru import CNN_GRU_Model
    model_template = CNN_GRU_Model(
        input_shape=config["input_shape"],
        num_classes=config["total_classes"],
    )

    # 4. Prepare node data (all clients, all data)
    node_data = {}
    for cid in data_loader.get_all_client_ids():
        X, y = data_loader.get_client_data(cid, task_id=0)  # Single task, no incremental
        if len(y) > 0:
            node_data[cid] = (X, y)

    print(f"  Nodes: {len(node_data)}")

    # 5. Prepare test data (all classes)
    test_X, test_y = data_loader.get_test_data(task_id=0, cumulative=True)
    test_data = {"X_test": test_X, "y_test": test_y}

    all_history = {
        "task_accuracies": [],
        "task_forgetting": [],
        "round_metrics": [],
        "plexus_history": {},
    }

    def _plexus_round_callback(round_id: int, global_params, round_metrics: Dict[str, Any]):
        round_record = {
            "task": 0,
            "round": round_id,
            "train_loss": round_metrics.get("train_loss", 0.0),
            "round_time": round_metrics.get("round_time", 0.0),
            "test_loss": round_metrics.get("loss"),
            "accuracy": round_metrics.get("accuracy"),
            "precision_macro": round_metrics.get("precision_macro"),
            "recall_macro": round_metrics.get("recall_macro"),
            "f1_macro": round_metrics.get("f1_macro"),
            "f1_weighted": round_metrics.get("f1_weighted"),
            "avg_forgetting": 0.0,
        }
        all_history["round_metrics"].append(round_record)
        _save_fed_round_checkpoint(
            output_dir,
            0,
            round_id,
            global_params,
            config,
            list(range(config["total_classes"])),
            float(round_metrics.get("train_loss", 0.0) or 0.0),
            float(round_metrics.get("round_time", 0.0) or 0.0),
            {
                "loss": round_metrics.get("loss", 0.0) or 0.0,
                "accuracy": round_metrics.get("accuracy", 0.0) or 0.0,
                "precision_macro": round_metrics.get("precision_macro", 0.0) or 0.0,
                "recall_macro": round_metrics.get("recall_macro", 0.0) or 0.0,
                "f1_macro": round_metrics.get("f1_macro", 0.0) or 0.0,
                "f1_weighted": round_metrics.get("f1_weighted", 0.0) or 0.0,
            },
            0.0,
        )
        _write_phase_outputs(output_dir, all_history, config, 0)

    # 6. Run Plexus
    result = run_plexus_training(
        node_ids=list(node_data.keys()),
        node_data=node_data,
        model_template=model_template,
        config=config,
        test_data=test_data,
        verbose=True,
        round_callback=_plexus_round_callback,
    )

    # 7. Format return (compatible with run_incremental_training format)
    history = result["history"]
    global_params = result["global_params"]

    # Compute metrics on test data
    model = CNN_GRU_Model(
        input_shape=config["input_shape"],
        num_classes=config["total_classes"],
    )
    model.load_state_dict({k: v.cpu() for k, v in global_params.items()})
    model.eval()

    X_test = test_data["X_test"]
    y_test = test_data["y_test"]

    all_preds = []
    all_targets = []
    with torch.no_grad():
        batch_size = config.get("batch_size", 32)
        for i in range(0, len(X_test), batch_size):
            batch_X = X_test[i:i+batch_size]
            batch_y = y_test[i:i+batch_size]
            output = model(batch_X)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    precision = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_targets, all_preds, average="macro", zero_division=0)

    print("\n" + "=" * 80)
    print("🏁 PURE PLEXUS COMPLETE")
    print(f"Final Accuracy: {acc*100:.2f}%")
    print(f"Final F1: {f1*100:.2f}%")
    print("=" * 80)

    final_metrics = {
        "loss": history["test_loss"][-1] if history.get("test_loss") else 0.0,
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "f1_weighted": history["test_f1_weighted"][-1] if history.get("test_f1_weighted") else None,
    }
    all_history["task_accuracies"] = [
        {
            "task": 0,
            "final_round": history["round"][-1] if history.get("round") else 0,
            "loss": final_metrics["loss"],
            "accuracy": acc,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
            "f1_weighted": final_metrics["f1_weighted"],
            "avg_forgetting": 0.0,
        }
    ]
    all_history["task_forgetting"] = [{"task": 0, "avg_forgetting": 0.0}]
    all_history["plexus_history"] = {k: v for k, v in history.items() if k != "sample"}

    _save_fed_task_checkpoint(
        output_dir,
        0,
        history["round"][-1] if history.get("round") else 0,
        global_params,
        config,
        list(range(config["total_classes"])),
        final_metrics,
        0.0,
        {0: acc},
    )
    _write_phase_outputs(output_dir, all_history, config, 0)

    return {
        "task_accuracies": all_history["task_accuracies"],
        "task_forgetting": all_history["task_forgetting"],
        "round_metrics": all_history["round_metrics"],
        "plexus_history": all_history["plexus_history"],
    }


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
        # Plexus decentralized IL: no server, class-incremental task loop.
        from fed_learning.training.decentralized_plexus_il import (
            run_decentralized_plexus_il,
        )

        return run_decentralized_plexus_il(config)
    if mode != "fed_il":
        raise ValueError("Unsupported mode. Use 'fed_il', 'il', or 'decentralized'.")

    # 1. Setup
    set_seed(config.get("random_seed", 42))

    resume_state = None
    if config.get("resume_state_path"):
        resume_state = load_continuation_state(config["resume_state_path"])
        resume_algo = resume_state["meta"].get("algorithm")
        if resume_algo != config["algorithm"]:
            raise ValueError(
                f"Resume state algorithm mismatch: {resume_algo} != {config['algorithm']}"
            )

    output_dir = _resolve_output_dir(config, mode, config["algorithm"])
    config = _inject_persistent_algorithm_paths(config, output_dir)

    config_name = "config_phase_resume.json" if resume_state else "config.json"
    with open(os.path.join(output_dir, config_name), "w") as f:
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
    global_neuron_ages = None  # NICE/PlexusNICE: preserve neuron ages across tasks
    all_history = {"task_accuracies": [], "task_forgetting": [], "round_metrics": []}
    all_test_data = {}
    best_acc_per_task = {}
    persistent_clients: Dict[int, object] = {}
    pending_client_states: Dict[int, Dict[str, Any]] = {}

    if resume_state is not None:
        global_model = resume_state.get("model_state_dict")
        global_neuron_ages = resume_state.get("global_neuron_ages")
        all_history = resume_state.get("all_history", all_history)
        best_acc_per_task = resume_state.get("best_acc_per_task", {})
        pending_client_states = {
            int(cid): state
            for cid, state in resume_state.get("persistent_clients_state", {}).items()
        }
        completed_task_ids = sorted(
            int(entry["task"]) for entry in all_history.get("task_accuracies", [])
        )
        all_test_data = _rebuild_prior_test_snapshots(data_loader, completed_task_ids)

    # 5. Task Loop
    # Create server ONCE and reuse for all tasks (fixes CGoFed aggregator state persistence)
    server = None

    num_tasks = data_loader.get_num_tasks()
    task_start, task_end = _resolve_task_bounds(config, num_tasks, resume_state)
    print(f"Task range: {task_start} -> {task_end}")

    for task_id in range(task_start, task_end + 1):
        print(
            f"\n{'=' * 80}\n📚 TASK {task_id}/{num_tasks - 1}\n{'=' * 80}"
        )

        # 5a. Prepare Data
        new_classes = data_loader.get_task_classes(task_id)
        seen_classes = []
        for t in range(task_id + 1):
            seen_classes.extend(data_loader.get_task_classes(t))

        # Get client data for current task
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
        # Bug 3 fix for dfca_il: use full client population for nested active scheduling.
        # The full population is needed so that task 0 (50%) + task 1 (60%) +
        # ... are truly nested supersets of each other.
        # All clients are included; those without current task data still participate
        # in graph communication and maintain their cluster banks.
        algo = config["algorithm"].lower()
        if algo == "dfca_il":
            all_client_ids = data_loader.get_all_client_ids()
            for cid in all_client_ids:
                if cid not in client_data_map:
                    client_data_map[cid] = {"X_train": torch.tensor([]), "y_train": torch.tensor([], dtype=torch.long)}
            participating_clients = []
            for cid in all_client_ids:
                data = client_data_map[cid]
                client = get_or_create_persistent_client(
                    cid, data, config, persistent_clients
                )
                if cid in pending_client_states:
                    restore_client_state(client, pending_client_states.pop(cid))
                update_client_data(client, data, task_id, new_classes)
                participating_clients.append(client)
        else:
            participating_clients = []
            for cid, data in client_data_map.items():
                client = get_or_create_persistent_client(
                    cid, data, config, persistent_clients
                )
                if cid in pending_client_states:
                    restore_client_state(client, pending_client_states.pop(cid))
                update_client_data(client, data, task_id, new_classes)
                participating_clients.append(client)

        # 5d. Prepare Server (create only for task 0, reuse for subsequent tasks)
        task_config = {
            **config,
            "num_classes": config["total_classes"],
            "num_rounds": config["rounds_per_task"],
            "num_tasks": num_tasks,  # For Plexus dynamic client scaling
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
            is_last_task = task_id == num_tasks - 1
            task_config["is_last_task"] = is_last_task

        # PlexusNICE: pass is_last_task flag
        if config["algorithm"].lower() == "plexus_nice":
            is_last_task = task_id == num_tasks - 1
            task_config["is_last_task"] = is_last_task

        # Create server only once, reuse for subsequent tasks
        if server is None:
            server = create_server(config, participating_clients, test_data, task_config)
            if resume_state is not None:
                restore_trainer_state(server.trainer, resume_state.get("trainer_state"))
                restore_aggregator_state(
                    server.aggregator, resume_state.get("aggregator_state")
                )
                restore_server_state(server, resume_state.get("server_state"))
        else:
            server = _refresh_server_clients(
                server,
                participating_clients,
                config,
                test_data,
                task_config,
            )

        if global_model is not None:
            server.set_global_params(global_model)
            # Restore neuron ages for NICE-based algorithms (unit_ranks NOT in state_dict)
            algo = config["algorithm"].lower()
            if global_neuron_ages is not None and algo in ("nice", "plexus_nice"):
                server.global_model.set_neuron_ages_state(global_neuron_ages)
                print(f"  Restored neuron ages from previous task")

        # Use server's trainer/aggregator (already configured with proper state like bandwidths)
        trainer = server.trainer
        aggregator = server.aggregator

        if hasattr(server, "set_task"):
            server.set_task(task_id, new_classes, seen_classes)
        if hasattr(trainer, "set_task"):
            trainer.set_task(task_id, new_classes)
        if hasattr(aggregator, "set_task"):
            aggregator.set_task(task_id)

        # 5e. Train (algorithm-specific dispatch)
        print(f"\n🎯 Training on {len(new_classes)} new classes...")
        algo = config["algorithm"].lower()
        is_last_task = task_id == num_tasks - 1
        last_round_record = None

        if algo == "nice":
            max_phases, phase_epochs, effective_local_epochs = _resolve_nice_schedule(
                config
            )
            nice_rounds = max_phases
            if is_last_task:
                print(
                    f"\n  === NICE Training ({nice_rounds} phases) ==="
                    f"\n  NICE local schedule: {max_phases} phases x {phase_epochs} epochs"
                    f" = {effective_local_epochs} local epochs/client"
                    "\n  [LAST EPISODE: tau=100%]"
                )
            else:
                print(
                    f"\n  === NICE Training ({nice_rounds} phases) ==="
                    f"\n  NICE local schedule: {max_phases} phases x {phase_epochs} epochs"
                    f" = {effective_local_epochs} local epochs/client"
                )
            last_round_record = _run_tracked_rounds(
                server,
                lambda _r: server.train_round(
                    participating_clients=participating_clients,
                    verbose=True,
                    phase_offset=_r,
                    max_phases_override=1,
                ),
                nice_rounds,
                task_id,
                output_dir,
                all_history,
                all_test_data,
                best_acc_per_task,
                trainer,
                config,
                seen_classes,
                is_last_task,
                label="[phase]",
            )
        elif algo == "der":
            stage1_rounds, stage2_rounds = _resolve_der_round_split(config)
            total_rounds = stage1_rounds + stage2_rounds

            if hasattr(trainer, "set_stage"):
                trainer.set_stage(1)
            print(f"\n  === DER Stage 1: Representation Learning ({stage1_rounds} rounds) ===")
            _run_tracked_rounds(
                server,
                lambda _r: server.train_round(
                    participating_clients=participating_clients,
                    stage=1,
                    verbose=True,
                ),
                stage1_rounds,
                task_id,
                output_dir,
                all_history,
                all_test_data,
                best_acc_per_task,
                trainer,
                config,
                seen_classes,
                is_last_task,
                round_start=0,
                round_total_last=total_rounds - 1,
                label="[stage=1]",
            )

            if hasattr(trainer, "set_stage"):
                trainer.set_stage(2)
            if hasattr(server.global_model, "reset_classifier"):
                server.global_model.reset_classifier()
                print("  → Classifier H_t re-initialized (paper Section 3.2)")

            print(f"\n  === DER Stage 2: Classifier Learning ({stage2_rounds} rounds) ===")
            last_round_record = _run_tracked_rounds(
                server,
                lambda _r: server.train_round(
                    participating_clients=participating_clients,
                    stage=2,
                    verbose=True,
                ),
                stage2_rounds,
                task_id,
                output_dir,
                all_history,
                all_test_data,
                best_acc_per_task,
                trainer,
                config,
                seen_classes,
                is_last_task,
                round_start=stage1_rounds,
                round_total_last=total_rounds - 1,
                label="[stage=2]",
            )
            # Weight Alignment after Stage 2
            if task_id > 0 and hasattr(server.global_model, "weight_align"):
                server.global_model.weight_align(len(new_classes))
        elif algo == "glfc":
            glfc_rounds = config.get("rounds_per_task", 5)
            print(f"\n  === GLFC Training ({glfc_rounds} rounds) ===")
            last_round_record = _run_tracked_rounds(
                server,
                lambda _r: server.train_round(
                    participating_clients=participating_clients,
                    verbose=True,
                ),
                glfc_rounds,
                task_id,
                output_dir,
                all_history,
                all_test_data,
                best_acc_per_task,
                trainer,
                config,
                seen_classes,
                is_last_task,
            )
            print("  Updating exemplar sets for all participants...")
            server.coordinate_exemplar_update(participating_clients, verbose=True)
        elif algo == "refed":
            refed_rounds = config.get("rounds_per_task", 5)
            if task_id > 0:
                print(f"\n  === Re-Fed: PIM Caching (Task {task_id}) ===")
            else:
                print("\n  === Re-Fed: Initial Caching (Task 0) ===")
            server.coordinate_pim_caching(participating_clients, verbose=True)
            print(f"\n  === Re-Fed Training ({refed_rounds} rounds) ===")
            last_round_record = _run_tracked_rounds(
                server,
                lambda _r: server.train_round(
                    participating_clients=participating_clients,
                    verbose=True,
                ),
                refed_rounds,
                task_id,
                output_dir,
                all_history,
                all_test_data,
                best_acc_per_task,
                trainer,
                config,
                seen_classes,
                is_last_task,
            )
        elif algo == "plexus":
            plexus_rounds = config.get("rounds_per_task", 5)
            
            # Set num_tasks for dynamic client scaling in PlexusServer
            if hasattr(server, '_num_tasks'):
                server._num_tasks = num_tasks
            
            print(f"\n  === Plexus Decentralized Training ({plexus_rounds} rounds) ===")
            if hasattr(server, 'scale_clients') and server.scale_clients:
                print(f"  📈 Dynamic scaling: {server.initial_client_ratio*100:.0f}% → {server.final_client_ratio*100:.0f}%")
            
            last_round_record = _run_tracked_rounds(
                server,
                lambda _r: server.train_round(
                    participating_clients=participating_clients,
                    task_id=task_id,
                    verbose=True,
                ),
                plexus_rounds,
                task_id,
                output_dir,
                all_history,
                all_test_data,
                best_acc_per_task,
                trainer,
                config,
                seen_classes,
                is_last_task,
            )
        elif algo == "plexus_der":
            stage1_rounds, stage2_rounds = _resolve_der_round_split(config)
            total_rounds = stage1_rounds + stage2_rounds

            # Set num_tasks for dynamic client scaling in PlexusServer
            if hasattr(server, '_num_tasks'):
                server._num_tasks = num_tasks

            if hasattr(trainer, "set_stage"):
                trainer.set_stage(1)
            print(f"\n  === PlexusDER Stage 1: Representation Learning ({stage1_rounds} rounds) ===")
            if hasattr(server, 'scale_clients') and server.scale_clients:
                print(f"  📈 Dynamic scaling: {server.initial_client_ratio*100:.0f}% → {server.final_client_ratio*100:.0f}%")
            _run_tracked_rounds(
                server,
                lambda _r: server.train_round(
                    participating_clients=participating_clients,
                    task_id=task_id,
                    stage=1,
                    verbose=True,
                ),
                stage1_rounds,
                task_id,
                output_dir,
                all_history,
                all_test_data,
                best_acc_per_task,
                trainer,
                config,
                seen_classes,
                is_last_task,
                round_start=0,
                round_total_last=total_rounds - 1,
                label="[stage=1]",
            )

            if hasattr(trainer, "set_stage"):
                trainer.set_stage(2)
            if hasattr(server.global_model, "reset_classifier"):
                server.global_model.reset_classifier()
                print("  → Classifier H_t re-initialized (paper Section 3.2)")

            print(f"\n  === PlexusDER Stage 2: Classifier Learning ({stage2_rounds} rounds) ===")
            last_round_record = _run_tracked_rounds(
                server,
                lambda _r: server.train_round(
                    participating_clients=participating_clients,
                    task_id=task_id,
                    stage=2,
                    verbose=True,
                ),
                stage2_rounds,
                task_id,
                output_dir,
                all_history,
                all_test_data,
                best_acc_per_task,
                trainer,
                config,
                seen_classes,
                is_last_task,
                round_start=stage1_rounds,
                round_total_last=total_rounds - 1,
                label="[stage=2]",
            )
            # Weight Alignment after Stage 2
            if task_id > 0 and hasattr(server.global_model, "weight_align"):
                server.global_model.weight_align(len(new_classes))
        elif algo == "plexus_nice":
            max_phases, phase_epochs, effective_local_epochs = _resolve_nice_schedule(
                config
            )
            nice_rounds = max_phases
            
            # Set num_tasks for dynamic client scaling in PlexusServer
            if hasattr(server, '_num_tasks'):
                server._num_tasks = num_tasks
            
            if is_last_task:
                print(
                    f"\n  === PlexusNICE Training ({nice_rounds} phases) ==="
                    f"\n  PlexusNICE local schedule: {max_phases} phases x {phase_epochs} epochs"
                    f" = {effective_local_epochs} local epochs/client"
                    "\n  [LAST EPISODE: tau=100%]"
                )
            else:
                print(
                    f"\n  === PlexusNICE Training ({nice_rounds} phases) ==="
                    f"\n  PlexusNICE local schedule: {max_phases} phases x {phase_epochs} epochs"
                    f" = {effective_local_epochs} local epochs/client"
                )
            if hasattr(server, 'scale_clients') and server.scale_clients:
                print(f"  📈 Dynamic scaling: {server.initial_client_ratio*100:.0f}% → {server.final_client_ratio*100:.0f}%")
            
            last_round_record = _run_tracked_rounds(
                server,
                lambda _r: server.train_round(
                    participating_clients=participating_clients,
                    task_id=task_id,
                    is_last_task=is_last_task,
                    verbose=True,
                ),
                nice_rounds,
                task_id,
                output_dir,
                all_history,
                all_test_data,
                best_acc_per_task,
                trainer,
                config,
                seen_classes,
                is_last_task,
                label="[phase]",
            )
        elif algo == "dfca_il":
            dfca_rounds = config.get("rounds_per_task", 5)
            print(f"\n  === DFCA-IL Training ({dfca_rounds} rounds) ===")
            print(f"  Active clients: {server.client_ratios[task_id]:.0%} of total")
            last_round_record = _run_tracked_rounds(
                server,
                lambda _r: server.train_round(
                    participating_clients=None,
                    task_id=task_id,
                    verbose=True,
                ),
                dfca_rounds,
                task_id,
                output_dir,
                all_history,
                all_test_data,
                best_acc_per_task,
                trainer,
                config,
                seen_classes,
                is_last_task,
            )
        else:
            total_rounds = config["rounds_per_task"]
            print(f"\n  === Standard Federated Training ({total_rounds} rounds) ===")
            last_round_record = _run_tracked_rounds(
                server,
                lambda _r: server.train_round(verbose=True),
                total_rounds,
                task_id,
                output_dir,
                all_history,
                all_test_data,
                best_acc_per_task,
                trainer,
                config,
                seen_classes,
                is_last_task,
            )

        # 5f. Post-Task Processing
        post_task_processing(
            trainer, server, client_data_map, config, participating_clients
        )

        # 5g. Update Global Model
        global_model = server.get_global_params()
        # Save neuron ages for NICE-based algorithms (unit_ranks NOT in state_dict)
        algo = config["algorithm"].lower()
        if algo in ("nice", "plexus_nice"):
            global_neuron_ages = server.global_model.get_neuron_ages_state()

        # 5h. Evaluate
        print(f"\n📊 Evaluation:")
        metrics = server.evaluate_global(compute_auc=False)
        print(
            "  ✅ Task summary -> "
            f"accuracy={metrics['accuracy'] * 100:.2f}%, "
            f"f1={metrics['f1_macro'] * 100:.2f}%, "
            f"precision={metrics['precision_macro'] * 100:.2f}%, "
            f"recall={metrics['recall_macro'] * 100:.2f}%"
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
                "final_round": last_round_record["round"] if last_round_record else 0,
                "loss": metrics["loss"],
                "accuracy": metrics["accuracy"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "f1_macro": metrics["f1_macro"],
                "f1_weighted": metrics.get("f1_weighted"),
                "avg_forgetting": af,
            }
        )
        all_history["task_forgetting"].append({"task": task_id, "avg_forgetting": af})

        # 5l. Checkpoint
        _save_fed_task_checkpoint(
            output_dir,
            task_id,
            last_round_record["round"] if last_round_record else 0,
            global_model,
            config,
            list(seen_classes),
            metrics,
            af,
            current_task_accuracies,
        )
        _write_phase_outputs(output_dir, all_history, config, task_id)

        if config.get("save_resume_after_task") == task_id:
            continuation_state = build_continuation_state(
                mode=mode,
                algorithm=config["algorithm"],
                task_id=task_id,
                config=config,
                output_dir=output_dir,
                model_state_dict=global_model,
                global_neuron_ages=global_neuron_ages,
                trainer=server.trainer,
                server=server,
                aggregator=server.aggregator,
                persistent_clients=persistent_clients,
                all_history=all_history,
                best_acc_per_task=best_acc_per_task,
                seen_classes=seen_classes,
            )
            continuation_path = save_continuation_state(
                output_dir, task_id, continuation_state
            )
            print(f"💾 Continuation state saved: {continuation_path}")

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

    _write_phase_outputs(
        output_dir,
        all_history,
        config,
        int(final.get("task", config.get("task_end", 0))),
    )
