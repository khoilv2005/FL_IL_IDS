"""Local incremental learning task loop (non-federated)."""

import gc
import json
import os
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, List, Tuple

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
from fed_learning.utils.cleanup import cleanup_temp_folders
from fed_learning.utils.seed import set_seed


def _create_local_model(
    algorithm: str, config: Dict[str, Any], device: str
) -> nn.Module:
    algo = algorithm.lower()
    if algo == "der":
        from fed_learning.models.der_model import DERModel

        return DERModel(config["input_shape"], config["num_classes"]).to(device)
    if algo == "nice":
        from fed_learning.models.nice_model import NICEModel

        return NICEModel(config["input_shape"], config["num_classes"]).to(device)

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


def _evaluate_model(
    model: nn.Module, test_data: Dict[str, torch.Tensor], device: str
) -> Dict[str, float]:
    model.eval()
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
            out = model(X_batch)
            loss = criterion(out, y_batch)
            total_loss += loss.item() * len(y_batch)
            preds.append(out.argmax(dim=1).cpu().numpy())
            targets.append(y_batch.cpu().numpy())

    y_true = np.concatenate(targets)
    y_pred = np.concatenate(preds)
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


def _compute_local_forgetting(
    model, device, data_loader, task_id, best_acc_per_task, trainer
):
    current_task_accuracies = {}
    for prev_tid in range(task_id + 1):
        X_prev, y_prev = data_loader.get_test_data(prev_tid, cumulative=False)
        metrics = _evaluate_model(model, {"X_test": X_prev, "y_test": y_prev}, device)
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


def _run_local_der(model, client, trainer, config, device, new_classes):
    stage1_epochs = max(
        1,
        config.get("local_epochs", 1)
        * config.get("der_stage1_rounds", config.get("rounds_per_task", 1)),
    )
    stage2_epochs = max(
        1,
        config.get("local_epochs", 1) * config.get("der_stage2_rounds", 3),
    )

    trainer.set_stage(1)
    client.setup_for_gpu(model, device)
    client.train(
        trainer=trainer,
        epochs=stage1_epochs,
        batch_size=config["batch_size"],
        lr=config["learning_rate"],
        global_params=None,
        stage=1,
    )

    trainer.set_stage(2)
    if hasattr(model, "reset_classifier"):
        model.reset_classifier()
    client.setup_for_gpu(model, device)
    client.train(
        trainer=trainer,
        epochs=stage2_epochs,
        batch_size=config["batch_size"],
        lr=config["learning_rate"],
        global_params=None,
        stage=2,
    )

    if hasattr(client, "update_exemplars"):
        client.update_exemplars(model)
    if getattr(model, "current_task", -1) > 0 and hasattr(model, "weight_align"):
        model.weight_align(len(new_classes))


def _run_local_nice(
    model, client, trainer, config, device, task_id, num_tasks, new_classes
):
    trainer.max_phases = config.get(
        "rounds_per_task", getattr(trainer, "max_phases", 1)
    )
    trainer.phase_epochs = config.get(
        "local_epochs", getattr(trainer, "phase_epochs", 1)
    )

    for cls_id in new_classes:
        if cls_id < model.num_classes:
            model.unit_ranks["fc2"][cls_id] = 1

    print(
        f"  NICE local schedule: {trainer.max_phases} phases x "
        f"{trainer.phase_epochs} epochs"
    )

    client.setup_for_gpu(model, device)
    client.train(
        trainer=trainer,
        epochs=trainer.phase_epochs,
        batch_size=config["batch_size"],
        lr=config["learning_rate"],
        global_params=None,
        is_last_task=(task_id == num_tasks - 1),
    )

    increase_unit_ranks(model)
    update_freeze_masks(model)
    if hasattr(model, "freeze_bn_for_mature"):
        model.freeze_bn_for_mature()


def _run_local_generic(model, client, trainer, config, device, task_id, algorithm):
    effective_epochs = max(
        1, config.get("local_epochs", 1) * config.get("rounds_per_task", 1)
    )
    client.setup_for_gpu(model, device)

    client.train(
        trainer=trainer,
        epochs=effective_epochs,
        batch_size=config["batch_size"],
        lr=config["learning_rate"],
        global_params=None,
    )


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


def run_local_incremental_training(config: Dict[str, Any]):
    """Run standalone incremental learning without federated aggregation."""
    set_seed(config.get("random_seed", 42))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        f"{config['output_dir']}_{config['algorithm']}_{config.get('mode', 'il')}_{ts}"
    )
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/config.json", "w") as f:
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
    print("✓ Local IL algorithms supported: ewc, lwf, der, nice")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _create_local_model(config["algorithm"], config, device)
    all_history = {"task_accuracies": [], "task_forgetting": []}
    best_acc_per_task: Dict[int, float] = {}
    persistent_client = None

    for task_id in range(data_loader.get_num_tasks()):
        print(
            f"\n{'=' * 80}\n📚 TASK {task_id}/{data_loader.get_num_tasks()}\n{'=' * 80}"
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

        update_client_data(persistent_client, combined_data, task_id, new_classes)
        if hasattr(trainer, "set_task"):
            trainer.set_task(task_id, new_classes)

        if config["algorithm"].lower() == "der" and hasattr(model, "add_task"):
            model.add_task(new_classes, s_max=config.get("s_max", 15.0))

        print(f"\n🎯 Local training on {len(new_classes)} new classes...")
        algo = config["algorithm"].lower()
        if algo == "der":
            _run_local_der(
                model, persistent_client, trainer, config, device, new_classes
            )
        elif algo == "nice":
            _run_local_nice(
                model,
                persistent_client,
                trainer,
                config,
                device,
                task_id,
                data_loader.get_num_tasks(),
                new_classes,
            )
        else:
            _run_local_generic(
                model, persistent_client, trainer, config, device, task_id, algo
            )

        _post_task_local(
            algo, trainer, model, persistent_client, combined_data, config, device
        )

        test_X, test_y = data_loader.get_test_data(task_id, cumulative=True)
        metrics = _evaluate_model(model, {"X_test": test_X, "y_test": test_y}, device)
        print(
            f"  Accuracy: {metrics['accuracy'] * 100:.2f}% | F1: {metrics['f1_macro'] * 100:.2f}%"
        )

        current_task_accuracies, af = _compute_local_forgetting(
            model, device, data_loader, task_id, best_acc_per_task, trainer
        )
        all_history["task_accuracies"].append(
            {
                "task": task_id,
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "avg_forgetting": af,
                "per_task_acc": current_task_accuracies,
            }
        )

        ckpt_path = os.path.join(output_dir, f"checkpoint_task_{task_id}.pt")
        torch.save(
            {
                "task_id": task_id,
                "model_state_dict": OrderedDict(
                    (k, v.detach().cpu().clone()) for k, v in model.state_dict().items()
                ),
                "config": config,
                "seen_classes": sorted(
                    {
                        c
                        for t in range(task_id + 1)
                        for c in data_loader.get_task_classes(t)
                    }
                ),
            },
            ckpt_path,
        )
        print(f"💾 Checkpoint saved: {ckpt_path}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("🏁 LOCAL IL TRAINING COMPLETE")
    print("=" * 80)
    if all_history["task_accuracies"]:
        final = all_history["task_accuracies"][-1]
        print(f"Final Accuracy: {final['accuracy'] * 100:.2f}%")
        print(f"Final Forgetting: {final['avg_forgetting'] * 100:.2f}%")

    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(all_history, f, indent=2)

    return all_history
