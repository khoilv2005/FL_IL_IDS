"""Decentralized Plexus with class-incremental task scheduling."""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from fed_learning.data.incremental_loader import IncrementalDataLoader
from fed_learning.models.cnn_gru import CNN_GRU_Model
from fed_learning.plexus import run_plexus_training
from fed_learning.utils.seed import set_seed



def _eval_params(
    model_template: CNN_GRU_Model,
    params,
    test_data: Dict[str, torch.Tensor],
    batch_size: int,
    seen_classes: Optional[List[int]],
) -> Dict[str, float]:
    if params is None:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
        }

    model = model_template.__class__(model_template.input_shape, model_template.num_classes)
    model.load_state_dict({k: v.cpu() for k, v in params.items()})
    model.eval()

    X_test = test_data["X_test"]
    y_test = test_data["y_test"]
    seen_set = set(int(c) for c in seen_classes) if seen_classes else None
    if seen_set:
        mask = torch.tensor([int(y.item()) in seen_set for y in y_test])
        X_test = X_test[mask]
        y_test = y_test[mask]

    if len(y_test) == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
        }

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for i in range(0, len(y_test), batch_size):
            X_batch = X_test[i : i + batch_size]
            y_batch = y_test[i : i + batch_size]
            out = model(X_batch)
            if seen_set:
                unseen = torch.ones(out.shape[1], dtype=torch.bool)
                for cls_id in seen_set:
                    if 0 <= cls_id < out.shape[1]:
                        unseen[cls_id] = False
                out = out.clone()
                out[:, unseen] = float("-inf")
            loss = criterion(out, y_batch)
            total_loss += loss.item() * len(y_batch)
            all_preds.extend(out.argmax(dim=1).cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    return {
        "loss": total_loss / max(1, len(y_test)),
        "accuracy": accuracy_score(all_targets, all_preds),
        "precision_macro": precision_score(all_targets, all_preds, average="macro", zero_division=0),
        "recall_macro": recall_score(all_targets, all_preds, average="macro", zero_division=0),
        "f1_macro": f1_score(all_targets, all_preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(all_targets, all_preds, average="weighted", zero_division=0),
    }


def run_decentralized_plexus_il(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run no-server Plexus over incremental tasks with seen-class metrics."""
    from fed_learning.training.task_loop import (
        _evaluate_and_visualize,
        _resolve_output_dir,
        _save_fed_round_checkpoint,
        _save_fed_task_checkpoint,
        _write_phase_outputs,
        _write_training_history,
    )

    set_seed(config.get("random_seed", config.get("seed", 42)))

    output_dir = _resolve_output_dir(config, "decentralized", "plexus")

    print("\n" + "=" * 80)
    print("DECENTRALIZED PLEXUS-IL")
    print("=" * 80)

    data_loader = IncrementalDataLoader(data_dir=config["data_dir"])
    config["input_shape"] = data_loader.input_shape
    config["num_classes"] = config["total_classes"]
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    model_template = CNN_GRU_Model(config["input_shape"], config["total_classes"])
    num_tasks = data_loader.get_num_tasks()
    task_start = int(config.get("task_start", 0))
    task_end = int(config.get("task_end", num_tasks - 1))
    rounds_per_task = int(config.get("rounds_per_task", 5))
    batch_size = int(config.get("batch_size", 32))
    eval_every = max(1, int(config.get("eval_every", 1)))
    checkpoint_every = config.get("round_checkpoint_every", 1)
    if checkpoint_every is not None:
        checkpoint_every = max(1, int(checkpoint_every))

    history = {
        "task_accuracies": [],
        "task_forgetting": [],
        "round_metrics": [],
    }
    best_acc: Dict[int, float] = {}
    task_tests: Dict[int, Dict[str, torch.Tensor]] = {}
    global_params = None

    for task_id in range(task_start, task_end + 1):
        print(f"\n{'=' * 80}\nTASK {task_id}/{num_tasks - 1} - Decentralized Plexus-IL\n{'=' * 80}")

        seen_classes: List[int] = []
        for tid in range(task_id + 1):
            seen_classes.extend(data_loader.get_task_classes(tid))

        node_data = {}
        for cid in data_loader.get_all_client_ids():
            X, y = data_loader.get_client_data(cid, task_id)
            if len(y) > 0:
                node_data[cid] = (X, y)
        print(f"  Nodes with current task data: {len(node_data)}")
        if not node_data:
            continue

        test_X, test_y = data_loader.get_test_data(task_id, cumulative=True)
        task_tests[task_id] = {"X_test": test_X, "y_test": test_y}
        os.makedirs("./temp_test_data", exist_ok=True)
        torch.save(
            task_tests[task_id],
            os.path.join("./temp_test_data", f"decentralized_test_task_{task_id}.pt"),
        )

        task_config = {
            **config,
            "num_rounds": rounds_per_task,
            "num_classes": config["total_classes"],
        }

        def on_round(local_round: int, params, round_metrics: Dict[str, Any]) -> None:
            round_id = task_id * rounds_per_task + local_round
            is_final_round = local_round == rounds_per_task - 1
            evaluate = is_final_round or ((round_id + 1) % eval_every == 0)
            save_checkpoint = checkpoint_every is not None and (
                is_final_round or ((round_id + 1) % checkpoint_every == 0)
            )

            metrics = (
                _eval_params(model_template, params, task_tests[task_id], batch_size, seen_classes)
                if evaluate
                else {}
            )
            if evaluate and is_final_round:
                per_task_acc = {
                    tid: _eval_params(model_template, params, data, batch_size, seen_classes)["accuracy"]
                    for tid, data in task_tests.items()
                }
                for tid, acc in per_task_acc.items():
                    best_acc[tid] = max(best_acc.get(tid, 0.0), acc)
                old_tids = [tid for tid in per_task_acc if tid != task_id]
                af = (
                    sum(max(0.0, best_acc.get(tid, 0.0) - per_task_acc[tid]) for tid in old_tids)
                    / max(1, len(old_tids))
                    if old_tids
                    else 0.0
                )
            else:
                af = None

            history["round_metrics"].append(
                {
                    "task": task_id,
                    "round": round_id,
                    "train_loss": round_metrics.get("train_loss", 0.0),
                    "round_time": round_metrics.get("round_time", 0.0),
                    "test_loss": metrics.get("loss"),
                    "accuracy": metrics.get("accuracy"),
                    "precision_macro": metrics.get("precision_macro"),
                    "recall_macro": metrics.get("recall_macro"),
                    "f1_macro": metrics.get("f1_macro"),
                    "f1_weighted": metrics.get("f1_weighted"),
                    "avg_forgetting": af,
                    "evaluated": evaluate,
                }
            )
            _write_training_history(output_dir, history)
            if evaluate:
                af_text = f"{af * 100:.2f}%" if af is not None else "N/A (final round only)"
                print(
                    "    Metrics -> "
                    f"train_loss={float(round_metrics.get('train_loss', 0.0) or 0.0):.4f}, "
                    f"test_loss={metrics['loss']:.4f}, "
                    f"accuracy={metrics['accuracy'] * 100:.2f}%, "
                    f"f1={metrics['f1_macro'] * 100:.2f}%, "
                    f"precision={metrics['precision_macro'] * 100:.2f}%, "
                    f"recall={metrics['recall_macro'] * 100:.2f}%, "
                    f"AF={af_text}"
                )
            else:
                print(
                    "    Metrics skipped -> "
                    f"train_loss={float(round_metrics.get('train_loss', 0.0) or 0.0):.4f}, "
                    f"eval_every={config.get('eval_every', 1)}"
                )
            if save_checkpoint:
                _save_fed_round_checkpoint(
                    output_dir,
                    task_id,
                    round_id,
                    params,
                    config,
                    seen_classes,
                    float(round_metrics.get("train_loss", 0.0) or 0.0),
                    float(round_metrics.get("round_time", 0.0) or 0.0),
                    metrics,
                    af,
                )

        result = run_plexus_training(
            node_ids=list(node_data.keys()),
            node_data=node_data,
            model_template=model_template,
            config=task_config,
            test_data=None,
            initial_global_params=global_params,
            seen_classes=seen_classes,
            verbose=True,
            round_callback=on_round,
        )
        global_params = result["global_params"]

        final_metrics = _eval_params(
            model_template, global_params, task_tests[task_id], batch_size, seen_classes
        )
        per_task_acc = {
            tid: _eval_params(model_template, global_params, data, batch_size, seen_classes)["accuracy"]
            for tid, data in task_tests.items()
        }
        old_tids = [tid for tid in per_task_acc if tid != task_id]
        af = (
            sum(max(0.0, best_acc.get(tid, 0.0) - per_task_acc[tid]) for tid in old_tids)
            / max(1, len(old_tids))
            if old_tids
            else 0.0
        )
        history["task_accuracies"].append(
            {
                "task": task_id,
                "final_round": (task_id + 1) * rounds_per_task - 1,
                **final_metrics,
                "avg_forgetting": af,
            }
        )
        history["task_forgetting"].append({"task": task_id, "avg_forgetting": af})

        viz_model = CNN_GRU_Model(config["input_shape"], config["total_classes"])
        viz_model.load_state_dict({k: v.cpu() for k, v in global_params.items()})
        viz_wrapper = type("DecentralizedPlexusEvalWrapper", (), {})()
        viz_wrapper.global_model = viz_model
        viz_wrapper.test_data = task_tests[task_id]
        viz_wrapper.primary_device = "cpu"
        viz_wrapper.seen_classes = list(seen_classes)
        _evaluate_and_visualize(viz_wrapper, task_id, output_dir, config)

        final_round_id = (task_id + 1) * rounds_per_task - 1
        _save_fed_task_checkpoint(
            output_dir,
            task_id,
            final_round_id,
            global_params,
            config,
            seen_classes,
            final_metrics,
            af,
            per_task_acc,
        )
        _write_phase_outputs(output_dir, history, config, task_id)

    print("\n" + "=" * 80)
    print("DECENTRALIZED PLEXUS-IL COMPLETE")
    print("=" * 80)
    return history
