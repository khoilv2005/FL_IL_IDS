"""
DFCA Evaluation - Cluster and ensemble evaluation without global model.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Any, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_ensemble_average(
    nodes: Dict[int, Any],
    representative_params: Dict[int, OrderedDict],
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    model_class,
    input_shape,
    num_classes: int,
    device: str = "cpu",
    batch_size: int = 1024,
) -> Dict[str, float]:
    """
    Evaluate using ensemble averaging across representative cluster models.

    Average softmax probabilities across all k representative cluster models,
    then argmax per sample. No cluster selection, no label leakage.

    This is the PRIMARY evaluation metric for pure DFCA.

    Args:
        nodes: Dict of DFCANode instances (for getting model class).
        representative_params: Dict[cluster_id -> OrderedDict of averaged params].
        X_test, y_test: Test tensors.
        model_class: Model class (e.g., CNN_GRU_Model).
        input_shape: Model input shape.
        num_classes: Number of classes.
        device: Device to evaluate on.
        batch_size: Batch size.

    Returns:
        Dict with loss, accuracy, precision_macro, recall_macro,
        precision_weighted, recall_weighted, f1_macro, f1_weighted.
    """
    eval_models: Dict[int, nn.Module] = {}

    for cid in range(len(representative_params)):
        if cid not in representative_params:
            continue
        params = representative_params[cid]
        if not params:
            continue

        model = model_class(input_shape, num_classes)
        model.to(device)
        model.load_state_dict({k: v.to(device) for k, v in params.items()})
        model.eval()
        eval_models[cid] = model

    if not eval_models:
        return {
            "loss": 0.0, "accuracy": 0.0,
            "precision_macro": 0.0, "recall_macro": 0.0,
            "precision_weighted": 0.0, "recall_weighted": 0.0,
            "f1_macro": 0.0, "f1_weighted": 0.0,
        }

    all_preds = []
    all_targets = []
    total_ce_loss = 0.0
    n_test = len(y_test)

    with torch.no_grad():
        for i in range(0, n_test, batch_size):
            X_batch = X_test[i : i + batch_size].to(device)
            y_batch = y_test[i : i + batch_size].to(device)

            all_probs: List[torch.Tensor] = []
            for cid, model in eval_models.items():
                out = model(X_batch)
                probs = torch.softmax(out, dim=1)
                all_probs.append(probs)

            stacked_probs = torch.stack(all_probs, dim=0)
            avg_probs = stacked_probs.mean(dim=0)

            preds = avg_probs.argmax(dim=1)

            ce_loss = -torch.gather(
                torch.log(avg_probs + 1e-8), dim=1, index=y_batch.unsqueeze(1)
            ).sum()
            total_ce_loss += ce_loss.item()

            all_preds.extend(preds.detach().cpu().tolist())
            all_targets.extend(y_batch.detach().cpu().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    zero_division = 0

    return {
        "loss": total_ce_loss / n_test,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=zero_division),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=zero_division),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=zero_division),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=zero_division),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=zero_division),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=zero_division),
    }


def evaluate_representative_clusters(
    representative_params: Dict[int, OrderedDict],
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    model_class,
    input_shape,
    num_classes: int,
    device: str = "cpu",
    batch_size: int = 1024,
) -> Dict[str, Any]:
    """
    Evaluate each representative cluster model individually.

    Useful for understanding which clusters have learned what.
    Returns per-cluster metrics and identifies the "best" cluster by loss.

    Args:
        representative_params: Dict[cluster_id -> OrderedDict].
        Others: same as evaluate_ensemble_average.

    Returns:
        Dict with per_cluster_metrics, best_cluster_by_loss, best_cluster_by_accuracy.
    """
    results = {}

    for cid in range(len(representative_params)):
        if cid not in representative_params:
            continue
        params = representative_params[cid]
        if not params:
            continue

        model = model_class(input_shape, num_classes)
        model.to(device)
        model.load_state_dict({k: v.to(device) for k, v in params.items()})
        model.eval()

        all_preds = []
        all_targets = []
        total_ce_loss = 0.0
        n_test = len(y_test)

        with torch.no_grad():
            for i in range(0, n_test, batch_size):
                X_batch = X_test[i : i + batch_size].to(device)
                y_batch = y_test[i : i + batch_size].to(device)

                out = model(X_batch)
                probs = torch.softmax(out, dim=1)

                ce_loss = -torch.gather(
                    torch.log(probs + 1e-8), dim=1, index=y_batch.unsqueeze(1)
                ).sum()
                total_ce_loss += ce_loss.item()

                preds = out.argmax(dim=1)
            all_preds.extend(preds.detach().cpu().tolist())
            all_targets.extend(y_batch.detach().cpu().tolist())

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        zero_division = 0

        results[cid] = {
            "loss": total_ce_loss / n_test,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=zero_division),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=zero_division),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=zero_division),
        }

    if not results:
        return {"per_cluster_metrics": {}, "best_cluster_by_loss": None, "best_cluster_by_accuracy": None}

    best_loss = min(results, key=lambda c: results[c]["loss"])
    best_acc = max(results, key=lambda c: results[c]["accuracy"])

    return {
        "per_cluster_metrics": results,
        "best_cluster_by_loss": best_loss,
        "best_cluster_by_accuracy": best_acc,
    }


def evaluate_oracle(
    representative_params: Dict[int, OrderedDict],
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    model_class,
    input_shape,
    num_classes: int,
    device: str = "cpu",
    batch_size: int = 1024,
) -> Dict[str, float]:
    """
    Oracle evaluation: pick the cluster with lowest loss on the test set.

    This is a DIAGNOSTIC metric only — it uses ground-truth labels to
    select the cluster, which is NOT valid for real deployment.

    Use evaluate_ensemble_average() as the primary metric.

    Args:
        Same as evaluate_representative_clusters.

    Returns:
        Dict with loss, accuracy (oracle — diagnostic only).
    """
    per_cluster = evaluate_representative_clusters(
        representative_params, X_test, y_test,
        model_class, input_shape, num_classes, device, batch_size
    )

    best_cid = per_cluster.get("best_cluster_by_loss")
    if best_cid is None:
        return {"oracle_accuracy": 0.0, "oracle_loss": 0.0}

    metrics = per_cluster["per_cluster_metrics"][best_cid]
    return {
        "oracle_loss": metrics["loss"],
        "oracle_accuracy": metrics["accuracy"],
    }
