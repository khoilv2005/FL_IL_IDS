"""Local incremental learning task loop (non-federated)."""

import gc
import json
import os
import time
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

from fed_learning.data.incremental_loader import IncrementalDataLoader
from fed_learning.factories.client_factory import create_client, update_client_data
from fed_learning.strategies.incremental import get_incremental_strategy
from fed_learning.strategies.incremental.nice import (
    increase_unit_ranks,
    update_freeze_masks,
)
from fed_learning.servers.nice_server import ContextDetector
from fed_learning.training.resume_state import (
    build_continuation_state,
    load_continuation_state,
    restore_client_state,
    restore_trainer_state,
    save_continuation_state,
)
from fed_learning.training.checkpoint_state import (
    CHECKPOINT_SCHEMA_VERSION,
    build_algorithm_state,
)
from fed_learning.training.nice_neuron_usage import append_nice_neuron_usage
from fed_learning.utils.cleanup import cleanup_temp_folders
from fed_learning.utils.seed import set_seed

try:
    import matplotlib.pyplot as plt

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


def _create_local_model(
    algorithm: str, config: Dict[str, Any], device: str
) -> nn.Module:
    algo = algorithm.lower()
    if algo == "der":
        from fed_learning.models.der_model import DERModel

        return DERModel(config["input_shape"], config["num_classes"]).to(device)
    if algo in ("rne", "rne_compress"):
        from fed_learning.models.rne_model import RNECompressModel, RNEModel

        model_cls = RNECompressModel if algo == "rne_compress" else RNEModel
        kwargs = {
            "recurrent_scale": config.get("rne_recurrent_scale", 1.0),
        }
        if algo == "rne_compress":
            kwargs["compressed_channels"] = config.get(
                "rne_compress_channels", (16, 32, 64, 25)
            )
        return model_cls(config["input_shape"], config["num_classes"], **kwargs).to(device)
    if algo == "nice":
        from fed_learning.models.nice_model import NICEModel

        return NICEModel(config["input_shape"], config["num_classes"]).to(device)

    if algo == "denice":
        from fed_learning.models.denice_model import DeNICEModel

        return DeNICEModel(config["input_shape"], config["num_classes"]).to(device)

    from fed_learning.models.cnn_gru import CNN_GRU_Model

    return CNN_GRU_Model(config["input_shape"], config["num_classes"]).to(device)


def _build_single_client_dataset(
    data_loader: IncrementalDataLoader, task_id: int
) -> Tuple[Dict[int, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    client_data_map: Dict[int, Dict[str, torch.Tensor]] = {}
    all_X: List[torch.Tensor] = []
    all_y: List[torch.Tensor] = []

    for cid in data_loader.get_all_client_ids():
        X, y = data_loader.get_client_data(cid, task_id)
        if len(y) > 0:
            client_data_map[cid] = {"X_train": X, "y_train": y}
            all_X.append(X)
            all_y.append(y)

    if not all_X:
        return client_data_map, {
            "X_train": torch.empty(0),
            "y_train": torch.empty(0, dtype=torch.long),
        }

    return client_data_map, {
        "X_train": torch.cat(all_X, dim=0),
        "y_train": torch.cat(all_y, dim=0),
    }


def _nice_seen_mask(model: nn.Module, seen_classes: List[int], device: str) -> torch.Tensor:
    num_classes = int(getattr(model, "num_classes", 0) or 0)
    if num_classes <= 0 and hasattr(model, "fc2"):
        num_classes = int(model.fc2.out_features)
    if num_classes <= 0:
        return torch.zeros(0, dtype=torch.bool, device=device)

    mask = torch.ones(num_classes, dtype=torch.bool, device=device)
    for cls_id in set(int(c) for c in seen_classes):
        if 0 <= cls_id < num_classes:
            mask[cls_id] = False
    return mask


def _allowed_nice_episode_classes(
    context_detector: ContextDetector,
    episode: int,
    seen_classes: List[int],
    num_classes: int,
) -> List[int]:
    seen_set = set(int(c) for c in seen_classes)
    allowed = [
        int(c)
        for c in context_detector.episode_classes.get(int(episode), [])
        if int(c) in seen_set and 0 <= int(c) < num_classes
    ]
    if allowed:
        return sorted(set(allowed))
    return sorted(c for c in seen_set if 0 <= c < num_classes)


def _apply_local_nice_context_mask(
    model: nn.Module,
    logits: torch.Tensor,
    X_batch: torch.Tensor,
    context_detector: ContextDetector,
    seen_classes: List[int],
    device: str,
    context_activations: Dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    masked = logits.clone()
    if not getattr(context_detector, "episode_classes", None):
        global_unseen = _nice_seen_mask(model, seen_classes, device)
        if len(global_unseen) == masked.shape[1]:
            masked[:, global_unseen] = float("-inf")
        return masked

    try:
        if context_activations is not None:
            binary_acts = context_detector.binarize_layer_activations(
                {
                    name: np.asarray(act.detach().cpu().tolist())
                    for name, act in context_activations.items()
                }
            )
        else:
            binary_acts = context_detector._binarize_per_sample(model, X_batch)
        pred_episodes = context_detector.predict_episodes_batch(binary_acts)
    except Exception:
        global_unseen = _nice_seen_mask(model, seen_classes, device)
        if len(global_unseen) == masked.shape[1]:
            masked[:, global_unseen] = float("-inf")
        return masked

    num_classes = masked.shape[1]
    for row_idx, episode in enumerate(pred_episodes):
        allowed = _allowed_nice_episode_classes(
            context_detector,
            int(episode),
            seen_classes,
            num_classes,
        )
        if allowed:
            masked[row_idx, allowed] = masked[row_idx, allowed] + 99999.0
    return masked


def _predict_labels(
    model: nn.Module,
    X_data: torch.Tensor,
    y_data: torch.Tensor,
    device: str,
    batch_size: int,
    context_detector: ContextDetector = None,
    seen_classes: List[int] = None,
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor | None]:
    if len(y_data) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), None

    model.eval()
    # DeNICE: context-routed adapter prediction (no probability aggregation).
    if (
        context_detector is not None
        and seen_classes is not None
        and hasattr(model, "adapter_registry")
    ):
        from fed_learning.training.denice_eval import denice_predict_batch

        denice_preds: List[int] = []
        denice_targets: List[int] = []
        with torch.no_grad():
            for i in range(0, len(y_data), batch_size):
                X_batch = X_data[i : i + batch_size].to(device)
                y_batch = y_data[i : i + batch_size]
                batch_preds = denice_predict_batch(
                    model, X_batch, context_detector, seen_classes, device
                )
                denice_preds.extend(batch_preds.detach().cpu().tolist())
                denice_targets.extend(y_batch.detach().cpu().tolist())
        return np.asarray(denice_targets), np.asarray(denice_preds), None

    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    prob_sum: torch.Tensor | None = None
    prob_count = 0

    with torch.no_grad():
        for i in range(0, len(y_data), batch_size):
            X_batch = X_data[i : i + batch_size].to(device)
            y_batch = y_data[i : i + batch_size].to(device)
            if (
                context_detector is not None
                and seen_classes is not None
                and hasattr(model, "get_output_and_context_activations")
            ):
                out, context_activations = model.get_output_and_context_activations(
                    X_batch
                )
            else:
                out = model(X_batch)
                context_activations = None
            if context_detector is not None and seen_classes is not None:
                out = _apply_local_nice_context_mask(
                    model,
                    out,
                    X_batch,
                    context_detector,
                    seen_classes,
                    device,
                    context_activations=context_activations,
                )
            probs = torch.softmax(out.detach().cpu(), dim=1)
            if prob_sum is None:
                prob_sum = probs.sum(dim=0)
            else:
                prob_sum += probs.sum(dim=0)
            prob_count += len(y_batch)
            preds.extend(out.argmax(dim=1).detach().cpu().tolist())
            targets.extend(y_batch.detach().cpu().tolist())
            del X_batch, y_batch, out, probs

    mean_probs = prob_sum / max(1, prob_count) if prob_sum is not None else None
    return np.asarray(targets), np.asarray(preds), mean_probs


def _evaluate_model(
    model: nn.Module,
    test_data: Dict[str, torch.Tensor],
    device: str,
    context_detector: ContextDetector = None,
    seen_classes: List[int] = None,
) -> Dict[str, float]:
    model.eval()
    # DeNICE: route evaluation through context-aware micro-adapters.
    if context_detector is not None and hasattr(model, "adapter_registry"):
        from fed_learning.training.denice_eval import evaluate_denice_model

        return evaluate_denice_model(
            model,
            test_data,
            device,
            context_detector,
            seen_classes or [],
        )

    criterion = nn.CrossEntropyLoss()
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]

    if len(y_test) == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
        }

    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    total_loss = 0.0

    with torch.no_grad():
        batch_size = 1024
        for i in range(0, len(y_test), batch_size):
            X_batch = X_test[i : i + batch_size].to(device)
            y_batch = y_test[i : i + batch_size].to(device)
            if (
                context_detector is not None
                and seen_classes is not None
                and hasattr(model, "get_output_and_context_activations")
            ):
                out, context_activations = model.get_output_and_context_activations(
                    X_batch
                )
            else:
                out = model(X_batch)
                context_activations = None
            if context_detector is not None and seen_classes is not None:
                loss_out = out.clone()
                global_unseen = _nice_seen_mask(model, seen_classes, device)
                if len(global_unseen) == loss_out.shape[1]:
                    loss_out[:, global_unseen] = float("-inf")
                pred_out = _apply_local_nice_context_mask(
                    model,
                    out,
                    X_batch,
                    context_detector,
                    seen_classes,
                    device,
                    context_activations=context_activations,
                )
            else:
                loss_out = out
                pred_out = out
            loss = criterion(loss_out, y_batch)
            total_loss += loss.item() * len(y_batch)
            preds.extend(pred_out.argmax(dim=1).detach().cpu().tolist())
            targets.extend(y_batch.detach().cpu().tolist())

    y_true = np.asarray(targets)
    y_pred = np.asarray(preds)
    zero_division: Any = 0
    return {
        "loss": total_loss / max(1, len(y_test)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=zero_division
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=zero_division
        ),
        "f1_macro": f1_score(
            y_true, y_pred, average="macro", zero_division=zero_division
        ),
    }


def _evaluate_and_visualize_local(
    model: nn.Module,
    test_data: Dict[str, torch.Tensor],
    device: str,
    task_id: int,
    output_dir: str,
    config: Dict[str, Any],
    context_detector: ContextDetector = None,
    seen_classes: List[int] = None,
):
    """Generate the same task-level diagnostic plots as federated IL."""
    print("\n  Visualization & Debugging:")
    if not VISUALIZATION_AVAILABLE:
        print("   Visualization unavailable; skipping plots")
        return

    try:
        X_test = test_data["X_test"]
        y_test = test_data["y_test"]
        if len(y_test) == 0:
            print("   Empty test set; skipping plots")
            return

        if len(y_test) > 10000:
            indices = torch.randperm(len(y_test))[:10000]
            X_test = X_test[indices]
            y_test = y_test[indices]

        eval_batch_size = int(config.get("eval_batch_size", config.get("batch_size", 1024)))
        eval_batch_size = max(1, eval_batch_size)
        y_true, y_pred, mean_probs = _predict_labels(
            model,
            X_test,
            y_test,
            device,
            eval_batch_size,
            context_detector=context_detector,
            seen_classes=seen_classes,
        )

        unique_preds = sorted(set(y_pred.tolist()))
        print(f"   Unique predicted classes: {unique_preds}")

        with torch.no_grad():
            if hasattr(model, "fc2"):
                fc2_weight = model.fc2.weight.detach().cpu()
            elif hasattr(model, "classifier"):
                fc2_weight = model.classifier.weight.detach().cpu()
            else:
                fc2_weight = None

            if fc2_weight is not None:
                print("  DEBUG[2]: Output layer weight norms per class:")
                for cls_id in sorted(set(y_true.tolist())):
                    if cls_id < fc2_weight.shape[0]:
                        print(f"    Class {cls_id}: {fc2_weight[cls_id].norm().item():.4f}")

            if mean_probs is not None:
                print("  DEBUG[7]: Mean prediction probability per class:")
                for cls_id in sorted(set(y_true.tolist())):
                    if cls_id < mean_probs.shape[0]:
                        print(f"    Class {cls_id}: {mean_probs[cls_id].item():.6f}")

        task_plot_dir = os.path.join(output_dir, "plots", f"task_{task_id}")
        os.makedirs(task_plot_dir, exist_ok=True)
        class_ids = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
        class_names = [str(cls_id) for cls_id in class_ids]

        cm_fig = plot_confusion_matrix(
            y_true,
            y_pred,
            class_names=class_names,
            save_path=os.path.join(task_plot_dir, "confusion_matrix.png"),
            title=f"Confusion Matrix (Task {task_id})",
        )
        metrics_fig = plot_per_class_metrics(
            y_true,
            y_pred,
            class_names=class_names,
            save_path=os.path.join(task_plot_dir, "per_class_metrics.png"),
        )
        plt.close(cm_fig)
        plt.close(metrics_fig)
        print(f"   Saved plots to {task_plot_dir}")
    except Exception as exc:
        print(f"   Visualization/Debug failed: {exc}")


def _compute_local_forgetting(
    model,
    device,
    data_loader,
    task_id,
    best_acc_per_task,
    trainer,
    context_detector: ContextDetector = None,
    seen_classes: List[int] = None,
):
    current_task_accuracies = {}
    for prev_tid in range(task_id + 1):
        X_prev, y_prev = data_loader.get_test_data(prev_tid, cumulative=False)
        metrics = _evaluate_model(
            model,
            {"X_test": X_prev, "y_test": y_prev},
            device,
            context_detector=context_detector,
            seen_classes=seen_classes,
        )
        current_task_accuracies[prev_tid] = metrics["accuracy"]
        best_acc_per_task[prev_tid] = max(
            best_acc_per_task.get(prev_tid, 0.0), metrics["accuracy"]
        )

    forgetting_values = [
        max(0.0, best_acc_per_task[t] - current_task_accuracies[t])
        for t in range(task_id)
    ]
    af = (
        float(sum(forgetting_values) / len(forgetting_values))
        if forgetting_values
        else 0.0
    )
    if hasattr(trainer, "update_forgetting"):
        trainer.update_forgetting(current_task_accuracies)
        if hasattr(trainer, "get_current_af"):
            af = trainer.get_current_af()
    return current_task_accuracies, af


def _write_local_history(output_dir: str, history: Dict[str, Any]):
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(history, f, indent=2, default=str)

    with open(os.path.join(output_dir, "round_metrics.json"), "w") as f:
        json.dump(history.get("round_metrics", []), f, indent=2, default=str)


def _generate_local_fcil_report(history: Dict[str, Any], config: Dict[str, Any], output_dir: str):
    if not FCIL_VISUALIZATION_AVAILABLE or not history.get("task_accuracies"):
        return

    try:
        strategy_name = config.get("algorithm", "unknown")
        fcil_results = {strategy_name: {}}
        for entry in history["task_accuracies"]:
            tid = int(entry["task"])
            fcil_results[strategy_name][tid] = [entry["accuracy"]]

        fcil_output_dir = os.path.join(output_dir, "fcil_plots")
        os.makedirs(fcil_output_dir, exist_ok=True)
        plot_task_accuracy_heatmap(
            fcil_results[strategy_name],
            strategy_name=strategy_name,
            save_path=os.path.join(fcil_output_dir, "task_accuracy_heatmap.png"),
        )
        plot_incremental_accuracy_curve(
            fcil_results,
            save_path=os.path.join(fcil_output_dir, "incremental_accuracy.png"),
            title=f"Incremental Learning - {strategy_name}",
        )

        aia = compute_average_incremental_accuracy(fcil_results[strategy_name])
        forgetting = compute_forgetting_measure(fcil_results[strategy_name])
        with open(os.path.join(fcil_output_dir, "fcil_metrics.json"), "w") as f:
            json.dump(
                {
                    "strategy": strategy_name,
                    "final_aia": aia[-1] if aia else 0,
                    "aia_per_task": aia,
                    "forgetting_per_task": forgetting,
                    "avg_forgetting": sum(forgetting) / len(forgetting) if forgetting else 0,
                },
                f,
                indent=2,
            )
    except Exception as exc:
        print(f"FCIL visualization failed: {exc}")


def _write_local_phase_outputs(
    output_dir: str,
    history: Dict[str, Any],
    config: Dict[str, Any],
    completed_task_id: int,
    generate_report: bool = True,
):
    _write_local_history(output_dir, history)

    with open(os.path.join(output_dir, "task_metrics.json"), "w") as f:
        json.dump(history.get("task_accuracies", []), f, indent=2, default=str)

    phase_summary = {
        "algorithm": config.get("algorithm"),
        "mode": config.get("mode", "il"),
        "task_start": int(config.get("task_start", 0)),
        "task_end": int(config.get("task_end", completed_task_id)),
        "completed_task": int(completed_task_id),
        "num_task_records": len(history.get("task_accuracies", [])),
        "num_round_records": len(history.get("round_metrics", [])),
    }
    with open(os.path.join(output_dir, "phase_summary.json"), "w") as f:
        json.dump(phase_summary, f, indent=2, default=str)

    if generate_report:
        _generate_local_fcil_report(history, config, output_dir)


def _get_seen_classes(data_loader: IncrementalDataLoader, task_id: int) -> List[int]:
    return sorted(
        {
            cls_id
            for prev_tid in range(task_id + 1)
            for cls_id in data_loader.get_task_classes(prev_tid)
        }
    )


def _save_local_round_checkpoint(
    output_dir: str,
    task_id: int,
    round_id: int,
    model: nn.Module,
    config: Dict[str, Any],
    seen_classes: List[int],
    train_loss: float,
    round_time: float,
    metrics: Dict[str, Any],
    avg_forgetting: float,
    algorithm_state: Optional[Dict[str, Any]] = None,
):
    ckpt_path = os.path.join(output_dir, f"checkpoint_task_{task_id}_round_{round_id}.pt")
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "mode": config.get("mode", "il"),
            "algorithm": config.get("algorithm"),
            "task_id": task_id,
            "round_id": round_id,
            "model_state_dict": OrderedDict(
                (k, v.detach().cpu().clone()) for k, v in model.state_dict().items()
            ),
            "config": config,
            "seen_classes": list(seen_classes),
            "algorithm_state": algorithm_state or {},
            "metrics": {
                "train_loss": train_loss,
                "round_time": round_time,
                **metrics,
                "avg_forgetting": avg_forgetting,
            },
        },
        ckpt_path,
    )
    print(f"  Round checkpoint saved: {ckpt_path}")


def _save_local_task_checkpoint(
    output_dir: str,
    task_id: int,
    final_round_id: int,
    model: nn.Module,
    config: Dict[str, Any],
    seen_classes: List[int],
    metrics: Dict[str, Any],
    avg_forgetting: float,
    current_task_accuracies: Optional[Dict[int, float]] = None,
    algorithm_state: Optional[Dict[str, Any]] = None,
):
    ckpt_path = os.path.join(output_dir, f"checkpoint_task_{task_id}.pt")
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "mode": config.get("mode", "il"),
            "algorithm": config.get("algorithm"),
            "task_id": task_id,
            "final_round_id": final_round_id,
            "model_state_dict": OrderedDict(
                (k, v.detach().cpu().clone()) for k, v in model.state_dict().items()
            ),
            "config": config,
            "seen_classes": list(seen_classes),
            "algorithm_state": algorithm_state or {},
            "metrics": {**metrics, "avg_forgetting": avg_forgetting},
            "task_accuracies": current_task_accuracies or {},
        },
        ckpt_path,
    )
    print(f"Checkpoint saved: {ckpt_path}")


def _record_local_round(
    model: nn.Module,
    device: str,
    data_loader: IncrementalDataLoader,
    output_dir: str,
    history: Dict[str, Any],
    task_id: int,
    round_id: int,
    train_loss: float,
    round_time: float,
    best_acc_per_task: Dict[int, float],
    trainer,
    config: Dict[str, Any],
    seen_classes: List[int],
    context_detector: ContextDetector = None,
    task_classes_history: Optional[Dict[int, List[int]]] = None,
    compute_forgetting: bool = True,
    evaluate: bool = True,
):
    if evaluate:
        test_X, test_y = data_loader.get_test_data(task_id, cumulative=True)
        metrics = _evaluate_model(
            model,
            {"X_test": test_X, "y_test": test_y},
            device,
            context_detector=context_detector,
            seen_classes=seen_classes,
        )
    else:
        metrics = {}
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
        "f1_weighted": None,
        "avg_forgetting": af,
    }
    history["round_metrics"].append(round_record)
    _write_local_history(output_dir, history)

    if evaluate:
        print(
            "    Metrics -> "
            f"train_loss={train_loss:.4f}, test_loss={metrics['loss']:.4f}, "
            f"accuracy={metrics['accuracy'] * 100:.2f}%, "
            f"f1={metrics['f1_macro'] * 100:.2f}%, "
            f"precision={metrics['precision_macro'] * 100:.2f}%, "
            f"recall={metrics['recall_macro'] * 100:.2f}%"
        )
    else:
        print(f"    Metrics skipped -> train_loss={train_loss:.4f} (post-task eval)")

    _save_local_round_checkpoint(
        output_dir,
        task_id,
        round_id,
        model,
        config,
        seen_classes,
        train_loss,
        round_time,
        metrics,
        af,
        build_algorithm_state(
            config.get("algorithm", ""),
            model=model,
            context_detector=context_detector,
            task_classes_history=task_classes_history,
            config=config,
        ),
    )
    return round_record


def _resolve_der_round_split(config: Dict[str, Any]) -> tuple[int, int]:
    """
    Resolve DER stage rounds from config.

    Rules:
    - If either der_stage1_rounds or der_stage2_rounds is explicitly provided,
      honor the explicit values exactly (with existing defaults for the missing side).
    - Otherwise, treat rounds_per_task as the TOTAL DER budget and split it using
      the historical 3:2 ratio between stage 1 and stage 2.
    """
    if "der_stage1_rounds" in config or "der_stage2_rounds" in config:
        stage1_rounds = max(
            1, int(config.get("der_stage1_rounds", config.get("rounds_per_task", 1)))
        )
        stage2_rounds = max(1, int(config.get("der_stage2_rounds", 3)))
        return stage1_rounds, stage2_rounds

    total_rounds = max(1, int(config.get("rounds_per_task", 1)))
    if total_rounds == 1:
        return 1, 1

    # Preserve the old default bias of 3 stage-1 rounds and 2 stage-2 rounds.
    stage1_rounds = max(1, round(total_rounds * 3 / 5))
    stage2_rounds = max(1, total_rounds - stage1_rounds)
    if stage1_rounds + stage2_rounds != total_rounds:
        stage2_rounds = max(1, total_rounds - stage1_rounds)
    return stage1_rounds, stage2_rounds


def _run_local_der(
    model,
    client,
    trainer,
    config,
    device,
    new_classes,
    round_callback=None,
):
    local_epochs = max(1, int(config.get("local_epochs", 1)))
    stage1_rounds, stage2_rounds = _resolve_der_round_split(config)
    final_round_id = stage1_rounds + stage2_rounds - 1
    round_records: List[Dict[str, float]] = []

    trainer.set_stage(1)
    client.setup_for_gpu(model, device)
    for stage_round in range(stage1_rounds):
        start_time = time.time()
        result = client.train(
            trainer=trainer,
            epochs=local_epochs,
            batch_size=config["batch_size"],
            lr=config["learning_rate"],
            global_params=None,
            stage=1,
            rne_feature_batch_size=config.get(
                "rne_feature_batch_size",
                min(config["batch_size"], 1024),
            ),
        )
        record = {
            "round": stage_round,
            "train_loss": float((result or {}).get("loss", 0.0)),
            "round_time": time.time() - start_time,
        }
        round_records.append(record)
        if round_callback is not None:
            round_callback(record, final_round_id)

    trainer.set_stage(2)
    if config.get("algorithm", "").lower() == "der" and hasattr(model, "reset_classifier"):
        model.reset_classifier()
    client.setup_for_gpu(model, device)
    for stage_round in range(stage2_rounds):
        round_id = stage1_rounds + stage_round
        start_time = time.time()
        result = client.train(
            trainer=trainer,
            epochs=local_epochs,
            batch_size=config["batch_size"],
            lr=config["learning_rate"],
            global_params=None,
            stage=2,
            rne_pseudo_per_class=config.get("rne_pseudo_per_class", 400),
            rne_pseudo_old_per_class=config.get(
                "rne_pseudo_old_per_class",
                config.get("rne_pseudo_per_class", 400),
            ),
            rne_pseudo_new_per_class=config.get("rne_pseudo_new_per_class", 100),
            rne_feature_batch_size=config.get(
                "rne_feature_batch_size",
                min(config["batch_size"], 1024),
            ),
        )
        record = {
            "round": round_id,
            "train_loss": float((result or {}).get("loss", 0.0)),
            "round_time": time.time() - start_time,
        }
        round_records.append(record)
        if round_callback is not None:
            round_callback(record, final_round_id)

    if (
        config.get("algorithm", "").lower() == "der"
        and getattr(model, "current_task", -1) > 0
        and hasattr(model, "weight_align")
    ):
        model.weight_align(len(new_classes))
    if hasattr(client, "update_exemplars"):
        client.update_exemplars(model)
    return round_records


def _run_local_nice(
    model,
    client,
    trainer,
    config,
    device,
    task_id,
    num_tasks,
    new_classes,
    context_detector: ContextDetector,
    round_callback=None,
):
    trainer.max_phases = max(1, int(config.get("nice_max_phases", 5)))
    trainer.phase_epochs = max(1, int(config.get("nice_phase_epochs", 5)))
    total_phase_rounds = trainer.max_phases

    for cls_id in new_classes:
        if cls_id < model.num_classes:
            model.unit_ranks["fc2"][cls_id] = 1
    context_detector.episode_classes[task_id] = list(new_classes)

    print(
        "  NICE local schedule: "
        f"{trainer.max_phases} phases x "
        f"{trainer.phase_epochs} epochs = "
        f"{total_phase_rounds * trainer.phase_epochs} total local epochs"
    )

    client.setup_for_gpu(model, device)
    round_records: List[Dict[str, float]] = []
    final_round_id = total_phase_rounds - 1
    for round_id in range(total_phase_rounds):
        print(f"    Round {round_id}/{total_phase_rounds - 1} [phase]")
        start_time = time.time()
        result = client.train(
            trainer=trainer,
            epochs=trainer.phase_epochs,
            batch_size=config["batch_size"],
            lr=config["learning_rate"],
            global_params=None,
            is_last_task=(task_id == num_tasks - 1),
            phase_offset=round_id,
            max_phases_override=1,
        )
        _update_local_nice_context_memory(
            context_detector,
            model,
            client.X_train,
            client.y_train,
            task_id,
            new_classes,
            device,
        )
        record = {
            "round": round_id,
            "train_loss": float((result or {}).get("loss", 0.0)),
            "round_time": time.time() - start_time,
        }
        round_records.append(record)
        if round_callback is not None:
            round_callback(record, final_round_id)

    increase_unit_ranks(model)
    update_freeze_masks(model)
    if hasattr(model, "freeze_bn_for_mature"):
        model.freeze_bn_for_mature()
    return round_records


def _update_local_nice_context_memory(
    context_detector: ContextDetector,
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    task_id: int,
    task_classes: List[int],
    device: str,
) -> None:
    """Update local NICE context memory from train samples, never test data."""
    if X_train is None or y_train is None or len(y_train) == 0:
        return

    context_detector.episode_classes[task_id] = list(task_classes)
    per_class = max(1, int(getattr(context_detector, "memo_per_class", 50)))
    chunks = []
    y_cpu = y_train.detach().cpu()

    for cls_id in task_classes:
        idx = torch.nonzero(y_cpu == int(cls_id), as_tuple=False).flatten()
        if len(idx) == 0:
            continue
        take = min(per_class, len(idx))
        selected = idx[torch.randperm(len(idx))[:take]]
        chunks.append(X_train[selected].detach().cpu())

    if not chunks:
        return

    sample_data = torch.cat(chunks, dim=0).to(device)
    context_detector.push_activations(
        model,
        sample_data,
        task_id,
        reference_data=sample_data,
    )
    context_detector.train_models(task_id)


def _sample_denice_reference(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    classes: List[int],
    per_class: int,
    device: str,
) -> torch.Tensor:
    """Sample min(per_class, available) examples per class for prototypes/novelty."""
    if X_train is None or y_train is None or len(y_train) == 0:
        return torch.empty(0)
    y_cpu = y_train.detach().cpu()
    chunks = []
    for cls_id in classes:
        idx = torch.nonzero(y_cpu == int(cls_id), as_tuple=False).flatten()
        if len(idx) == 0:
            continue
        take = min(per_class, len(idx))
        selected = idx[torch.randperm(len(idx))[:take]]
        chunks.append(X_train[selected].detach().cpu())
    if not chunks:
        return torch.empty(0)
    return torch.cat(chunks, dim=0).to(device)


def _run_local_denice(
    model,
    client,
    trainer,
    config,
    device,
    task_id,
    num_tasks,
    new_classes,
    context_detector: ContextDetector,
    novelty_estimator,
    prev_ages,
    round_callback=None,
):
    """DeNICE local training: novelty -> CANC -> adapters -> NICE phases.

    Follows plan section 5.2 (Client.prepare_task / compute_novelty / run_CANC /
    activate_neurons_or_adapters / local_train) and section 6 (task 0..5).
    Returns ``(round_records, canc_plan)``.
    """
    from fed_learning.strategies.incremental.denice_capacity import (
        CANCConfig,
        CapacityController,
        compute_capacity_state,
        compute_consumption,
    )
    from fed_learning.strategies.incremental.denice_recycling import (
        apply_graceful_recycling,
        revive_due_recycled_neurons,
    )

    trainer.max_phases = max(1, int(config.get("nice_max_phases", 5)))
    trainer.phase_epochs = max(1, int(config.get("nice_phase_epochs", 5)))
    total_phase_rounds = trainer.max_phases

    # prepare_task: mark new output neurons as learner, register episode classes.
    for cls_id in new_classes:
        if cls_id < model.num_classes:
            model.unit_ranks["fc2"][cls_id] = 1
    context_detector.episode_classes[task_id] = list(new_classes)

    # Reference subset for thresholds / novelty / prototype (train data only).
    per_class = max(1, int(getattr(context_detector, "memo_per_class", 50)))
    ref_data = _sample_denice_reference(
        client.X_train, client.y_train, list(new_classes), per_class, device
    )
    revived = revive_due_recycled_neurons(model, task_id, config)
    if revived:
        print(f"  DeNICE recycle: revived retired neurons {revived}")

    canc_config = getattr(trainer, "canc_config", None) or CANCConfig.from_dict(config)
    controller = CapacityController(canc_config)

    is_first_task = not novelty_estimator.has_history()

    # compute_novelty (plan section 4).
    novelty_info = {"novelty": 0.0}
    if ref_data.numel() > 0:
        model.eval()
        if novelty_estimator.thresholds is None:
            novelty_estimator.calibrate_thresholds(model, ref_data)
        if not is_first_task:
            novelty_info = novelty_estimator.compute_novelty(model, ref_data)
    novelty = float(novelty_info.get("novelty", 0.0))

    # run_CANC (plan section 2.2 / 3). Consumption compares the previous task's
    # start ages against this task's start ages (== previous task's end ages).
    start_ages = model.get_neuron_ages_state()
    capacity_state = compute_capacity_state(model)
    consumption = compute_consumption(prev_ages, start_ages)
    canc_plan = controller.plan_task(
        capacity_state, novelty, consumption, is_first_task=is_first_task
    )
    canc_plan["novelty"] = novelty
    canc_plan["start_ages"] = start_ages

    # activate_neurons_or_adapters (plan section 5.2).
    model.clear_active_adapters()
    for layer in canc_plan["adapters_to_add"]:
        key = model.add_adapter(task_id, layer, set_active=True)
        print(f"  DeNICE CANC: added adapter on '{layer}' (key={key})")

    recycle_summary = apply_graceful_recycling(
        model, ref_data, task_id, canc_plan, config
    )
    if recycle_summary.get("total_retired", 0):
        print(f"  DeNICE recycle: retired neurons {recycle_summary['retired']}")

    # HIGH_LAYER_ONLY: temporarily freeze low conv layers (rebuilt at end_task).
    if canc_plan["freeze_low_layers"]:
        for low in ("conv1", "conv2"):
            ranks = model.unit_ranks.get(low)
            if ranks is not None:
                model.freeze_masks[low] = np.ones(len(ranks), dtype=bool)
        print("  DeNICE CANC: freezing low layers conv1/conv2 (HIGH_LAYER_ONLY)")

    action_summary = {name: info["action"] for name, info in canc_plan["layers"].items()}
    print(
        f"  DeNICE: task {task_id} | novelty={novelty:.3f} | "
        f"actions={action_summary} | adapters={canc_plan['adapters_to_add']}"
    )

    print(
        "  DeNICE local schedule: "
        f"{trainer.max_phases} phases x {trainer.phase_epochs} epochs"
    )

    client.setup_for_gpu(model, device)
    round_records: List[Dict[str, float]] = []
    final_round_id = total_phase_rounds - 1
    for round_id in range(total_phase_rounds):
        print(f"    Round {round_id}/{total_phase_rounds - 1} [phase]")
        start_time = time.time()
        result = client.train(
            trainer=trainer,
            epochs=trainer.phase_epochs,
            batch_size=config["batch_size"],
            lr=config["learning_rate"],
            global_params=None,
            is_last_task=(task_id == num_tasks - 1),
            phase_offset=round_id,
            max_phases_override=1,
        )
        _update_local_nice_context_memory(
            context_detector,
            model,
            client.X_train,
            client.y_train,
            task_id,
            new_classes,
            device,
        )
        record = {
            "round": round_id,
            "train_loss": float((result or {}).get("loss", 0.0)),
            "round_time": time.time() - start_time,
        }
        round_records.append(record)
        if round_callback is not None:
            round_callback(record, final_round_id)

    # end_task: mature learner neurons, rebuild freeze masks, store prototype.
    increase_unit_ranks(model)
    update_freeze_masks(model)
    if hasattr(model, "freeze_bn_for_mature"):
        model.freeze_bn_for_mature()

    if ref_data.numel() > 0:
        proto = novelty_estimator.compute_prototype(model, ref_data)
        novelty_estimator.store_prototype(task_id, proto)

    model.clear_active_adapters()
    return round_records, canc_plan


def _run_local_generic(
    model, client, trainer, config, device, task_id, algorithm, round_callback=None
):
    local_epochs = max(1, int(config.get("local_epochs", 1)))
    num_rounds = max(1, int(config.get("rounds_per_task", 1)))
    client.setup_for_gpu(model, device)
    round_records: List[Dict[str, float]] = []

    print(f"  Local schedule: {num_rounds} rounds x {local_epochs} epochs")
    final_round_id = num_rounds - 1
    for round_id in range(num_rounds):
        print(f"    Round {round_id}/{num_rounds - 1}")
        start_time = time.time()
        result = client.train(
            trainer=trainer,
            epochs=local_epochs,
            batch_size=config["batch_size"],
            lr=config["learning_rate"],
            global_params=None,
        )
        record = {
            "round": round_id,
            "train_loss": float((result or {}).get("loss", 0.0)),
            "round_time": time.time() - start_time,
        }
        round_records.append(record)
        if round_callback is not None:
            round_callback(record, final_round_id)
    return round_records


def _post_task_local(algorithm, trainer, model, client, combined_data, config, device):
    algo = algorithm.lower()

    if algo == "ewc" and hasattr(trainer, "consolidate"):
        X, y = combined_data["X_train"], combined_data["y_train"]
        if len(y) > config.get("fisher_samples", 200):
            idx = torch.randperm(len(y))[: config.get("fisher_samples", 200)]
            X = X[idx]
            y = y[idx]
        loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
        trainer.consolidate(model, loader, device)
    elif algo == "lwf" and hasattr(trainer, "save_model_snapshot"):
        trainer.save_model_snapshot(model)


def _resolve_local_output_dir(config: Dict[str, Any], algorithm: str, mode: str) -> str:
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
    output_dir = f"{config['output_dir']}_{algorithm}_{mode}_{ts}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _resolve_local_task_bounds(
    config: Dict[str, Any],
    num_tasks: int,
    resume_state: Dict[str, Any] | None = None,
) -> Tuple[int, int]:
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


def run_local_incremental_training(config: Dict[str, Any]):
    """Run standalone incremental learning without federated aggregation."""
    set_seed(config.get("random_seed", 42))
    resume_state = None
    if config.get("resume_state_path"):
        resume_state = load_continuation_state(config["resume_state_path"])
        resume_algo = resume_state["meta"].get("algorithm")
        if resume_algo != config["algorithm"]:
            raise ValueError(
                f"Resume state algorithm mismatch: {resume_algo} != {config['algorithm']}"
            )

    output_dir = _resolve_local_output_dir(
        config, config["algorithm"], config.get("mode", "il")
    )

    config_name = "config_phase_resume.json" if resume_state else "config.json"
    with open(os.path.join(output_dir, config_name), "w") as f:
        json.dump(config, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print(f"🚀 LOCAL INCREMENTAL LEARNING - {config['algorithm'].upper()}")
    print("=" * 80)

    cleanup_temp_folders()
    data_loader = IncrementalDataLoader(data_dir=config["data_dir"])
    config["input_shape"] = data_loader.input_shape
    config["num_classes"] = config["total_classes"]

    trainer = get_incremental_strategy(
        config["algorithm"],
        **{k: v for k, v in config.items() if k != "algorithm"},
    )
    print(f"✓ Trainer: {trainer.__class__.__name__}")
    print("✓ Local IL algorithms supported: ewc, lwf, der, rne, rne_compress, nice, denice")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _create_local_model(config["algorithm"], config, device)
    all_history = {"task_accuracies": [], "task_forgetting": [], "round_metrics": []}
    best_acc_per_task: Dict[int, float] = {}
    der_task_classes_history: Dict[int, List[int]] = {}
    persistent_client = None
    pending_client_state = None
    local_model_state = None
    nice_previous_ages = None
    _algo_lower = config["algorithm"].lower()
    _is_nice_like = _algo_lower in ("nice", "denice")
    nice_context_detector = (
        ContextDetector(memo_per_class=int(config.get("nice_memo_per_class", config.get("memo_per_class", 50))))
        if _is_nice_like
        else None
    )
    # DeNICE: novelty estimator + per-layer consumption tracking for CANC.
    denice_novelty_estimator = None
    denice_prev_ages = None
    if _algo_lower == "denice":
        from fed_learning.strategies.incremental.denice_novelty import NoveltyEstimator

        denice_novelty_estimator = NoveltyEstimator(
            layer_weights=getattr(trainer, "novelty_layer_weights", None)
        )

    if resume_state is not None:
        all_history = resume_state.get("all_history", all_history)
        all_history.setdefault("round_metrics", [])
        best_acc_per_task = resume_state.get("best_acc_per_task", {})
        restore_trainer_state(trainer, resume_state.get("trainer_state"))
        pending_clients = resume_state.get("persistent_clients_state", {})
        pending_client_state = pending_clients.get(0) or pending_clients.get("0")

        completed_task_ids = sorted(
            int(entry["task"]) for entry in all_history.get("task_accuracies", [])
        )
        if config["algorithm"].lower() in ("der", "rne", "rne_compress"):
            for prev_tid in completed_task_ids:
                prev_classes = data_loader.get_task_classes(prev_tid)
                der_task_classes_history[prev_tid] = list(prev_classes)
                model.add_task(prev_classes, s_max=config.get("s_max", 15.0))

        saved_model_state = resume_state.get("model_state_dict", {})
        if saved_model_state:
            model.load_state_dict(
                OrderedDict((k, v.to(device)) for k, v in saved_model_state.items())
            )

        local_model_state = resume_state.get("global_neuron_ages")
        if local_model_state is not None and hasattr(model, "set_neuron_ages_state"):
            model.set_neuron_ages_state(local_model_state)
            nice_previous_ages = local_model_state
            if _is_nice_like:
                update_freeze_masks(model)
                if hasattr(model, "freeze_bn_for_mature"):
                    model.freeze_bn_for_mature()

    num_tasks = data_loader.get_num_tasks()
    task_start, task_end = _resolve_local_task_bounds(
        config, num_tasks, resume_state
    )
    print(f"Task range: {task_start} -> {task_end}")

    for task_id in range(task_start, task_end + 1):
        print(
            f"\n{'=' * 80}\nTASK {task_id}/{num_tasks - 1}\n{'=' * 80}"
        )
        new_classes = data_loader.get_task_classes(task_id)
        _, combined_data = _build_single_client_dataset(data_loader, task_id)
        if len(combined_data["y_train"]) == 0:
            print("  ⚠️ No data for this task, skipping.")
            continue

        if persistent_client is None:
            persistent_client = create_client(
                0,
                combined_data["X_train"],
                combined_data["y_train"],
                config,
            )
            if pending_client_state is not None:
                restore_client_state(persistent_client, pending_client_state)
                pending_client_state = None

        update_client_data(persistent_client, combined_data, task_id, new_classes)
        if hasattr(trainer, "set_task"):
            trainer.set_task(task_id, new_classes)

        if config["algorithm"].lower() in ("der", "rne", "rne_compress") and hasattr(model, "add_task"):
            der_task_classes_history[task_id] = list(new_classes)
            model.add_task(new_classes, s_max=config.get("s_max", 15.0))

        print(f"\n🎯 Local training on {len(new_classes)} new classes...")
        algo = config["algorithm"].lower()
        seen_classes = _get_seen_classes(data_loader, task_id)
        last_round_record = None
        denice_canc_plan = None

        def record_round_now(round_summary, final_round_id):
            nonlocal last_round_record
            round_id = int(round_summary["round"])
            eval_every = max(1, int(config.get("eval_every", 1)))
            is_final_round = round_id == int(final_round_id)
            evaluate_this_round = (
                ((round_id + 1) % eval_every == 0) and not is_final_round
            )
            last_round_record = _record_local_round(
                model,
                device,
                data_loader,
                output_dir,
                all_history,
                task_id,
                round_id,
                float(round_summary.get("train_loss", 0.0)),
                float(round_summary.get("round_time", 0.0)),
                best_acc_per_task,
                trainer,
                config,
                seen_classes,
                context_detector=nice_context_detector if algo in ("nice", "denice") else None,
                task_classes_history=(
                    der_task_classes_history if algo in ("der", "rne", "rne_compress") else None
                ),
                compute_forgetting=False,
                evaluate=evaluate_this_round,
            )

        if algo in ("der", "rne", "rne_compress"):
            round_records = _run_local_der(
                model,
                persistent_client,
                trainer,
                config,
                device,
                new_classes,
                round_callback=record_round_now,
            )
        elif algo == "nice":
            round_records = _run_local_nice(
                model,
                persistent_client,
                trainer,
                config,
                device,
                task_id,
                data_loader.get_num_tasks(),
                new_classes,
                nice_context_detector,
                round_callback=record_round_now,
            )
        elif algo == "denice":
            round_records, denice_canc_plan = _run_local_denice(
                model,
                persistent_client,
                trainer,
                config,
                device,
                task_id,
                data_loader.get_num_tasks(),
                new_classes,
                nice_context_detector,
                denice_novelty_estimator,
                denice_prev_ages,
                round_callback=record_round_now,
            )
        else:
            round_records = _run_local_generic(
                model,
                persistent_client,
                trainer,
                config,
                device,
                task_id,
                algo,
                round_callback=record_round_now,
            )

        _post_task_local(
            algo, trainer, model, persistent_client, combined_data, config, device
        )
        if algo in ("nice", "denice"):
            summary_path = append_nice_neuron_usage(
                output_dir,
                task_id,
                model,
                previous_state=nice_previous_ages,
            )
            print(f"  NICE neuron usage saved: {summary_path}")
            if algo == "denice":
                from fed_learning.training.denice_usage import append_adapter_usage

                adapter_path = append_adapter_usage(
                    output_dir,
                    task_id,
                    model,
                    canc_plan=denice_canc_plan,
                    novelty=(denice_canc_plan or {}).get("novelty"),
                )
                print(f"  DeNICE adapter usage saved: {adapter_path}")
                # Track this task's START ages so the next task can measure the
                # consumption that happened during this task.
                if denice_canc_plan is not None and denice_canc_plan.get("start_ages"):
                    denice_prev_ages = denice_canc_plan["start_ages"]
                elif hasattr(model, "get_neuron_ages_state"):
                    denice_prev_ages = model.get_neuron_ages_state()
            if hasattr(model, "get_neuron_ages_state"):
                nice_previous_ages = model.get_neuron_ages_state()

        test_X, test_y = data_loader.get_test_data(task_id, cumulative=True)
        metrics = _evaluate_model(
            model,
            {"X_test": test_X, "y_test": test_y},
            device,
            context_detector=nice_context_detector if algo in ("nice", "denice") else None,
            seen_classes=seen_classes,
        )
        print(
            "  Task summary -> "
            f"accuracy={metrics['accuracy'] * 100:.2f}%, "
            f"f1={metrics['f1_macro'] * 100:.2f}%, "
            f"precision={metrics['precision_macro'] * 100:.2f}%, "
            f"recall={metrics['recall_macro'] * 100:.2f}%"
        )
        is_final_global_task = task_id == num_tasks - 1
        if is_final_global_task:
            _evaluate_and_visualize_local(
                model,
                {"X_test": test_X, "y_test": test_y},
                device,
                task_id,
                output_dir,
                config,
                context_detector=nice_context_detector if algo in ("nice", "denice") else None,
                seen_classes=seen_classes,
            )
        else:
            print("  Visualization skipped (final task only)")

        current_task_accuracies, af = {}, None
        all_history["task_accuracies"].append(
            {
                "task": task_id,
                "final_round": last_round_record["round"] if last_round_record else 0,
                "loss": metrics["loss"],
                "accuracy": metrics["accuracy"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "f1_macro": metrics["f1_macro"],
                "f1_weighted": None,
            }
        )

        _save_local_task_checkpoint(
            output_dir,
            task_id,
            last_round_record["round"] if last_round_record else 0,
            model,
            config,
            seen_classes,
            metrics,
            af,
            current_task_accuracies,
            build_algorithm_state(
                config.get("algorithm", ""),
                model=model,
                context_detector=nice_context_detector if algo in ("nice", "denice") else None,
                task_classes_history=(
                    der_task_classes_history if algo in ("der", "rne", "rne_compress") else None
                ),
                config=config,
            ),
        )
        _write_local_phase_outputs(
            output_dir,
            all_history,
            config,
            task_id,
            generate_report=is_final_global_task,
        )

        if config.get("save_resume_after_task") == task_id:
            local_model_state = None
            if hasattr(model, "get_neuron_ages_state"):
                local_model_state = model.get_neuron_ages_state()

            continuation_state = build_continuation_state(
                mode=config.get("mode", "il"),
                algorithm=config["algorithm"],
                task_id=task_id,
                config=config,
                output_dir=output_dir,
                model_state_dict=OrderedDict(
                    (k, v.detach().cpu().clone()) for k, v in model.state_dict().items()
                ),
                global_neuron_ages=local_model_state,
                trainer=trainer,
                persistent_clients={0: persistent_client} if persistent_client else {},
                all_history=all_history,
                best_acc_per_task=best_acc_per_task,
                seen_classes=seen_classes,
            )
            continuation_path = save_continuation_state(
                output_dir, task_id, continuation_state
            )
            print(f"Continuation state saved: {continuation_path}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("🏁 LOCAL IL TRAINING COMPLETE")
    print("=" * 80)
    if all_history["task_accuracies"]:
        final = all_history["task_accuracies"][-1]
        print(f"Final Accuracy: {final['accuracy'] * 100:.2f}%")

    if all_history["task_accuracies"]:
        _write_local_phase_outputs(
            output_dir,
            all_history,
            config,
            int(all_history["task_accuracies"][-1].get("task", config.get("task_end", 0))),
            generate_report=int(
                all_history["task_accuracies"][-1].get("task", config.get("task_end", 0))
            )
            == num_tasks - 1,
        )
    else:
        _write_local_history(output_dir, all_history)

    return all_history
