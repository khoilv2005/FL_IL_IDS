"""Decentralized DeNICE-IL runner.

This wires Phase 3 of ``DeNICE_micro_adapter_implementation_plan.md`` into a
real no-server simulation:

- every client owns its DeNICE model and context detector;
- clients build NICE Context Capsules after local training;
- Dynamic-K context clustering forms collaboration groups;
- age-aware masked aggregation updates only compatible parameters;
- adapter parameters are averaged only when adapter ids/shapes match.

The runner is intentionally sequential for reproducibility and low complexity.
It is a protocol runner, not a central FedAvg server: the coordinator only
simulates communication rounds and writes artifacts.
"""

from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from fed_learning.data.incremental_loader import IncrementalDataLoader
from fed_learning.factories.client_factory import create_client, update_client_data
from fed_learning.models.denice_model import DeNICEModel
from fed_learning.servers.nice_server import (
    ContextDetector,
    build_pooled_context_detector,
)
from fed_learning.strategies.decentralized import (
    AggregationConfig,
    ClusteringConfig,
    SimilarityWeights,
    age_aware_aggregate,
    aggregate_adapters,
    aggregation_weights,
    build_context_capsule,
    collaboration_group,
    context_similarity,
    dynamic_ap_cluster,
    label_overlap,
    merge_neuron_ages,
)
from fed_learning.strategies.fed_incremental.nice import (
    increase_unit_ranks,
    update_freeze_masks,
)
from fed_learning.strategies.incremental import get_incremental_strategy
from fed_learning.strategies.incremental.denice_capacity import (
    CANCConfig,
    CapacityController,
    compute_capacity_state,
    compute_consumption,
)
from fed_learning.strategies.incremental.denice_novelty import NoveltyEstimator
from fed_learning.strategies.incremental.denice_recycling import (
    apply_graceful_recycling,
    revive_due_recycled_neurons,
)
from fed_learning.training.denice_eval import (
    evaluate_denice_ensemble,
    evaluate_denice_model,
)
from fed_learning.training.denice_delta_checkpoint import (
    cpu_client_model_states,
    save_delta_round_checkpoint,
    save_task_base_checkpoint,
    update_checkpoint_index,
)
from fed_learning.training.denice_usage import compute_adapter_usage
from fed_learning.training.checkpoint_state import (
    CHECKPOINT_SCHEMA_VERSION,
    restore_context_detector,
    snapshot_denice_state,
)
from fed_learning.training.local_task_loop import (
    _update_local_nice_context_memory,
)
from fed_learning.training.task_loop import _resolve_output_dir
from fed_learning.utils.seed import set_seed


def _state_dict(model: torch.nn.Module) -> "OrderedDict[str, torch.Tensor]":
    return OrderedDict((k, v.detach().clone()) for k, v in model.state_dict().items())

def _cpu_state_dict(model: torch.nn.Module) -> "OrderedDict[str, torch.Tensor]":
    return OrderedDict(
        (k, v.detach().cpu().clone()) for k, v in model.state_dict().items()
    )

def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, OrderedDict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return value

def _write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, default=str)

def _count_histogram(y: torch.Tensor) -> Dict[int, int]:
    if y is None or len(y) == 0:
        return {}
    labels, counts = torch.unique(y.detach().cpu(), return_counts=True)
    return {int(c): int(n.item()) for c, n in zip(labels, counts)}

def _param_max_abs_diff(
    left: "OrderedDict[str, torch.Tensor]",
    right: "OrderedDict[str, torch.Tensor]",
) -> float:
    max_diff = 0.0
    for key, value in left.items():
        other = right.get(key)
        if other is None or other.shape != value.shape:
            continue
        diff = (value.detach().cpu() - other.detach().cpu()).abs().max().item()
        max_diff = max(max_diff, float(diff))
    return max_diff

def _capacity_debug(model: DeNICEModel) -> Dict[str, Dict[str, float]]:
    return compute_capacity_state(model)


def _enforce_minimum_free_capacity(
    model: DeNICEModel,
    task_start_ages: Optional[Dict[str, np.ndarray]],
    minimum_free_ratio: float,
) -> Dict[str, int]:
    """Keep a deterministic reserve from neurons selected in the current task.

    NICE promotes all rank-1 units at task end.  In the decentralized setting
    that can leave no plastic capacity after the very first task, before CANC
    or adapters have a chance to react.  Only units that were free at the
    start of this task and remain learners are released back to free; mature
    knowledge from earlier tasks is never reopened.
    """
    ratio = float(minimum_free_ratio)
    if ratio <= 0.0 or not task_start_ages:
        return {}
    released: Dict[str, int] = {}
    for layer, start in task_start_ages.items():
        ranks = getattr(model, "unit_ranks", {}).get(layer)
        start = np.asarray(start)
        if ranks is None or np.asarray(ranks).shape != start.shape:
            continue
        required_free = int(np.ceil(ratio * len(ranks)))
        deficit = required_free - int((ranks == 0).sum())
        if deficit <= 0:
            continue
        eligible = np.flatnonzero((ranks == 1) & (start == 0))
        if not len(eligible):
            continue
        chosen = eligible[:deficit]
        ranks[chosen] = 0
        released[layer] = int(len(chosen))
    return released

def _round_float_stats(values: List[float]) -> Dict[str, Optional[float]]:
    clean = [float(v) for v in values if np.isfinite(float(v))]
    if not clean:
        return {"min": None, "mean": None, "max": None}
    arr = np.asarray(clean, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
    }


def _effective_cluster_assignment(
    cluster_result: Dict[str, Any],
    client_ids: List[int],
    previous_valid_cluster: Optional[Dict[str, Any]] = None,
    invalid_policy: str = "previous_valid_or_self_only",
) -> Tuple[np.ndarray, Optional[np.ndarray], str, Optional[str], Optional[Dict[str, Any]]]:
    """Resolve a safe collaboration assignment from a raw AP result.

    AP's ``valid`` flag is a control-flow signal, not merely diagnostic data.
    When a round is invalid, reusing labels is safe only when they were fitted
    for exactly the same ordered client set. Otherwise each client remains in
    a singleton group, avoiding invalid neighbor aggregation and age transfer.
    """
    raw_labels = np.asarray(cluster_result["labels"], dtype=np.int64)
    if raw_labels.shape != (len(client_ids),):
        raise ValueError("cluster labels must align with client_ids")

    if bool(cluster_result.get("valid", False)):
        next_state = {
            "client_ids": tuple(int(cid) for cid in client_ids),
            "labels": raw_labels.copy(),
        }
        return raw_labels, cluster_result.get("edges"), "raw_valid", None, next_state

    policy = str(invalid_policy or "previous_valid_or_self_only").lower()
    if policy not in {"previous_valid_or_self_only", "self_only"}:
        raise ValueError(f"Unsupported invalid cluster policy: {invalid_policy}")

    if policy == "previous_valid_or_self_only" and previous_valid_cluster is not None:
        previous_ids = tuple(int(cid) for cid in previous_valid_cluster.get("client_ids", ()))
        previous_labels = np.asarray(previous_valid_cluster.get("labels", []), dtype=np.int64)
        if previous_ids == tuple(int(cid) for cid in client_ids) and previous_labels.shape == raw_labels.shape:
            return (
                previous_labels.copy(),
                None,
                "previous_valid",
                "raw_cluster_invalid_reused_previous_valid_labels",
                previous_valid_cluster,
            )

    return (
        np.arange(len(client_ids), dtype=np.int64),
        None,
        "self_only",
        "raw_cluster_invalid_without_compatible_previous_valid_labels",
        previous_valid_cluster,
    )


def _bootstrap_source_client(
    models: Dict[int, DeNICEModel],
    context_detectors: Dict[int, ContextDetector],
    exclude_client: Optional[int] = None,
) -> Optional[int]:
    """Choose a deterministic current representative for a late joiner.

    More covered contexts are preferred; remaining ties prefer a model with
    more free capacity and finally the smallest id for reproducibility.
    """
    if not models:
        return None

    def score(cid: int) -> Tuple[int, int, float, int]:
        detector = context_detectors.get(int(cid))
        covered = {
            int(label)
            for labels in getattr(detector, "episode_classes", {}).values()
            for label in labels
        }
        capacity = _capacity_debug(models[cid])
        free = float(np.mean([v["rho0"] for v in capacity.values()])) if capacity else 0.0
        return (len(covered), len(getattr(detector, "episode_classes", {})), free, -int(cid))

    candidates = [int(cid) for cid in models if int(cid) != exclude_client]
    return max(candidates, key=score) if candidates else None


def _bootstrap_denice_model(
    source_model: DeNICEModel,
    config: Dict[str, Any],
    device: torch.device,
) -> DeNICEModel:
    """Clone a compatible DeNICE representative, including adapter structure."""
    model = _make_model(config, device)
    for meta in source_model.get_adapter_registry_state().values():
        model.add_adapter(
            int(meta["context_id"]),
            str(meta["layer_name"]),
            rank=int(meta["rank"]),
            set_active=False,
        )
    model.load_state_dict(_state_dict(source_model), strict=True)
    model.set_neuron_ages_state(source_model.get_neuron_ages_state())
    if hasattr(source_model, "get_recycling_state"):
        model.set_recycling_state(source_model.get_recycling_state())
    update_freeze_masks(model)
    if hasattr(model, "freeze_bn_for_mature"):
        model.freeze_bn_for_mature()
    model.clear_active_adapters()
    return model.to(device)


def _catch_up_rejoining_model(
    target_model: DeNICEModel,
    source_model: DeNICEModel,
    device: torch.device,
) -> float:
    """Synchronize only target-plastic parameters from a current representative."""
    for meta in source_model.get_adapter_registry_state().values():
        if not target_model.has_adapter(
            int(meta["context_id"]), str(meta["layer_name"]), int(meta["rank"])
        ):
            target_model.add_adapter(
                int(meta["context_id"]),
                str(meta["layer_name"]),
                rank=int(meta["rank"]),
                set_active=False,
            )
    before = _state_dict(target_model)
    source = _state_dict(source_model)
    updated = age_aware_aggregate(
        before,
        target_model.get_neuron_ages_state(),
        [_delta_to_target(before, source)],
        np.asarray([1.0]),
        AggregationConfig(eta=1.0, protect_mature=True),
    )
    target_model.load_state_dict(updated, strict=False)
    target_model.clear_active_adapters()
    target_model.to(device)
    return _param_max_abs_diff(before, updated)

def _select_eval_clients(
    client_ids: List[int],
    max_clients: Optional[int],
    context_detectors: Optional[Dict[int, ContextDetector]] = None,
    seen_classes: Optional[List[int]] = None,
    require_full_coverage: bool = True,
) -> List[int]:
    ordered = list(client_ids)
    if context_detectors is not None and seen_classes is not None:
        required = {int(c) for c in seen_classes}

        def coverage(cid: int) -> float:
            detector = context_detectors.get(int(cid))
            episode_classes = getattr(detector, "episode_classes", {}) if detector else {}
            known = {
                int(cls_id)
                for classes in episode_classes.values()
                for cls_id in classes
            }
            if not required:
                return 1.0
            return len(required & known) / max(1, len(required))

        full = [cid for cid in ordered if coverage(cid) >= 1.0]
        partial = sorted(
            [cid for cid in ordered if cid not in full],
            key=lambda cid: (-coverage(cid), int(cid)),
        )
        ordered = full if require_full_coverage and full else [*full, *partial]
    if max_clients is None or int(max_clients) <= 0:
        return ordered
    return ordered[: int(max_clients)]

def _limit_eval_samples(
    X: torch.Tensor,
    y: torch.Tensor,
    max_samples: Optional[int],
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    total = int(len(y))
    if max_samples is None or int(max_samples) <= 0 or total <= int(max_samples):
        return X, y, {"limited": False, "total": total, "used": total}
    used = int(max_samples)
    generator = torch.Generator().manual_seed(int(seed))
    idx = torch.randperm(total, generator=generator)[:used]
    return X[idx], y[idx], {"limited": True, "total": total, "used": used}

def _sample_reference_with_labels(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    classes: List[int],
    per_class: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a small labeled reference set for capsule prototypes."""
    if X_train is None or y_train is None or len(y_train) == 0:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    y_cpu = y_train.detach().cpu()
    x_chunks = []
    y_chunks = []
    for cls_id in classes:
        idx = torch.nonzero(y_cpu == int(cls_id), as_tuple=False).flatten()
        if len(idx) == 0:
            continue
        take = min(max(1, int(per_class)), len(idx))
        selected = idx[torch.randperm(len(idx))[:take]]
        x_chunks.append(X_train[selected].detach().cpu())
        y_chunks.append(y_train[selected].detach().cpu())
    if not x_chunks:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return torch.cat(x_chunks, dim=0).to(device), torch.cat(y_chunks, dim=0).to(device)


def _delta_to_target(
    target: "OrderedDict[str, torch.Tensor]",
    neighbor: "OrderedDict[str, torch.Tensor]",
) -> "OrderedDict[str, torch.Tensor]":
    delta = OrderedDict()
    for key, value in target.items():
        other = neighbor.get(key)
        if other is None or other.shape != value.shape:
            continue
        delta[key] = other.to(value.device) - value
    return delta


def _label_histogram(y: torch.Tensor) -> Dict[int, float]:
    if y is None or len(y) == 0:
        return {}
    labels, counts = torch.unique(y.detach().cpu(), return_counts=True)
    total = float(counts.sum().item())
    if total <= 0:
        return {}
    return {int(c): float(n.item() / total) for c, n in zip(labels, counts)}

def _cosine_distance_to_centroid(vectors: Dict[int, np.ndarray]) -> Dict[int, float]:
    """Cosine distance from each vector to the group centroid."""
    if not vectors:
        return {}
    width = max(int(np.asarray(v).size) for v in vectors.values())
    if width <= 0:
        return {int(k): 0.0 for k in vectors}
    rows = {}
    for cid, vec in vectors.items():
        arr = np.asarray(vec, dtype=np.float64).ravel()
        if arr.size < width:
            arr = np.pad(arr, (0, width - arr.size))
        rows[int(cid)] = arr
    centroid = np.mean(np.stack(list(rows.values()), axis=0), axis=0)
    centroid_norm = float(np.linalg.norm(centroid))
    distances: Dict[int, float] = {}
    for cid, arr in rows.items():
        denom = float(np.linalg.norm(arr)) * centroid_norm
        if denom <= 1e-12:
            distances[int(cid)] = 0.0
        else:
            distances[int(cid)] = float(1.0 - np.dot(arr, centroid) / denom)
    return distances


def _seen_classes(data_loader: IncrementalDataLoader, task_id: int) -> List[int]:
    seen: List[int] = []
    for tid in range(task_id + 1):
        seen.extend(int(c) for c in data_loader.get_task_classes(tid))
    return sorted(set(seen))

def _round_checkpoint_path(output_dir: str, task_id: int, round_id: int) -> str:
    return os.path.join(output_dir, f"checkpoint_task_{task_id}_round_{round_id}.pt")

def _base_checkpoint_path(output_dir: str, task_id: int) -> str:
    return os.path.join(output_dir, f"checkpoint_task_{task_id}_base.pt")

def _client_algorithm_states(
    client_ids: List[int],
    models: Dict[int, DeNICEModel],
    context_detectors: Dict[int, ContextDetector],
) -> Dict[int, Dict[str, Any]]:
    return {
        int(cid): snapshot_denice_state(models[cid], context_detectors.get(cid))
        for cid in client_ids
    }

def _adapter_states(model: DeNICEModel) -> Dict[str, "OrderedDict[str, torch.Tensor]"]:
    return {
        key: OrderedDict((k, v.detach().clone()) for k, v in adapter.state_dict().items())
        for key, adapter in getattr(model, "adapters", {}).items()
    }

def _inject_adapter_states(
    full_state: "OrderedDict[str, torch.Tensor]",
    adapter_states: Dict[str, "OrderedDict[str, torch.Tensor]"],
) -> None:
    for adapter_id, state in adapter_states.items():
        for name, value in state.items():
            full_state[f"adapters.{adapter_id}.{name}"] = value.detach().clone()

def _save_round_artifacts(
    *,
    output_dir: str,
    task_id: int,
    round_id: int,
    client_ids: List[int],
    models: Dict[int, DeNICEModel],
    context_detectors: Dict[int, ContextDetector],
    capsules: Dict[int, Any],
    cluster_summary: Dict[str, Any],
    adapter_usage: Dict[int, Dict[str, Any]],
    config: Dict[str, Any],
    seen_classes: List[int],
    round_record: Dict[str, Any],
    save_checkpoint: bool,
) -> None:
    _write_json(
        os.path.join(output_dir, f"context_capsule_task_{task_id}_round_{round_id}.json"),
        {"task": task_id, "round": round_id, "clients": capsules},
    )
    _write_json(
        os.path.join(output_dir, f"cluster_snapshot_task_{task_id}_round_{round_id}.json"),
        cluster_summary,
    )
    _write_json(
        os.path.join(output_dir, f"adapter_registry_task_{task_id}_round_{round_id}.json"),
        {"task": task_id, "round": round_id, "clients": adapter_usage},
    )

    if not save_checkpoint:
        return

    ckpt_path = _round_checkpoint_path(output_dir, task_id, round_id)
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "mode": "decentralized",
            "algorithm": "denice",
            "task_id": int(task_id),
            "round_id": int(round_id),
            "client_model_states": {
                int(cid): _cpu_state_dict(models[cid]) for cid in client_ids
            },
            "client_algorithm_states": _client_algorithm_states(
                client_ids, models, context_detectors
            ),
            "config": config,
            "seen_classes": list(seen_classes),
            "cluster": _json_safe(cluster_summary),
            "metrics": _json_safe(round_record),
        },
        ckpt_path,
    )
    print(f"   Round checkpoint saved: {ckpt_path}")

def _write_phase_outputs(
    output_dir: str,
    history: Dict[str, Any],
    cluster_history: List[Dict[str, Any]],
    adapter_history: List[Dict[str, Any]],
    debug_history: List[Dict[str, Any]],
    config: Dict[str, Any],
    completed_task_id: int,
) -> None:
    _write_json(os.path.join(output_dir, "training_history.json"), history)
    _write_json(os.path.join(output_dir, "results.json"), history)
    _write_json(os.path.join(output_dir, "round_metrics.json"), history.get("round_metrics", []))
    _write_json(os.path.join(output_dir, "task_metrics.json"), history.get("task_accuracies", []))
    _write_json(os.path.join(output_dir, "cluster_history.json"), cluster_history)
    _write_json(os.path.join(output_dir, "denice_debug_history.json"), debug_history)
    _write_json(os.path.join(output_dir, "denice_adapter_usage_summary.json"), {"clients": adapter_history})
    _write_json(os.path.join(output_dir, "all_task_metrics.json"), history.get("task_accuracies", []))
    _write_json(os.path.join(output_dir, "all_cluster_metrics.json"), cluster_history)
    _write_json(os.path.join(output_dir, "all_adapter_metrics.json"), adapter_history)
    _write_json(
        os.path.join(output_dir, "phase_summary.json"),
        {
            "algorithm": "denice",
            "mode": "decentralized",
            "task_start": int(config.get("task_start", 0)),
            "task_end": int(config.get("task_end", completed_task_id)),
            "completed_task": int(completed_task_id),
            "num_task_records": len(history.get("task_accuracies", [])),
            "num_round_records": len(history.get("round_metrics", [])),
            "num_cluster_records": len(cluster_history),
            "num_adapter_records": len(adapter_history),
            "num_debug_records": len(debug_history),
        },
    )
    _write_json(
        os.path.join(output_dir, "final_report.json"),
        {
            "algorithm": "denice",
            "mode": "decentralized",
            "completed_task": int(completed_task_id),
            "final_metrics": (
                history.get("task_accuracies", [{}])[-1]
                if history.get("task_accuracies")
                else {}
            ),
        },
    )


def _make_model(config: Dict[str, Any], device: torch.device) -> DeNICEModel:
    return DeNICEModel(config["input_shape"], config["num_classes"]).to(device)

def _compute_reference_ce_loss(
    model: DeNICEModel,
    ref_bank: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
    device: torch.device,
    batch_size: int = 8192,
) -> Optional[float]:
    """Local old-reference CE loss for CANC ``Delta L_val``.

    The reference bank stays inside each simulated client. Only the scalar delta
    is used by CANC; raw reference tensors are never placed in capsules.
    """
    if not ref_bank:
        return None
    was_training = model.training
    active_adapters = dict(getattr(model, "active_adapters", {}))
    total_loss = 0.0
    total_count = 0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    with torch.no_grad():
        for context_id, pair in sorted(ref_bank.items()):
            if not pair:
                continue
            X_ref, y_ref = pair
            if X_ref is None or y_ref is None or len(y_ref) == 0:
                continue
            if hasattr(model, "set_active_context"):
                model.set_active_context(int(context_id))
            X_ref = X_ref.to(device)
            y_ref = y_ref.to(device).long()
            for start in range(0, len(y_ref), max(1, int(batch_size))):
                xb = X_ref[start : start + batch_size]
                yb = y_ref[start : start + batch_size]
                logits = model(xb)
                total_loss += float(criterion(logits, yb).item())
                total_count += int(len(yb))
    model.active_adapters = active_adapters
    if was_training:
        model.train()
    if total_count <= 0:
        return None
    return float(total_loss / total_count)

def _compute_val_loss_delta(
    model: DeNICEModel,
    ref_bank: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
    baseline: Optional[float],
    device: torch.device,
    batch_size: int,
) -> float:
    current = _compute_reference_ce_loss(model, ref_bank, device, batch_size=batch_size)
    if current is None or baseline is None:
        return 0.0
    return float(max(0.0, current - float(baseline)))


def _prepare_client_task(
    *,
    cid: int,
    task_id: int,
    num_tasks: int,
    new_classes: List[int],
    model: DeNICEModel,
    client,
    trainer,
    config: Dict[str, Any],
    device: torch.device,
    context_detector: ContextDetector,
    novelty_estimator: NoveltyEstimator,
    prev_ages: Optional[Dict[str, np.ndarray]],
    old_ref_bank: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
    old_ref_loss_baseline: Optional[float] = None,
) -> Dict[str, Any]:
    """Run DeNICE prepare_task / novelty / CANC / adapter activation."""
    if hasattr(trainer, "set_task"):
        trainer.set_task(task_id, new_classes)

    for cls_id in new_classes:
        if 0 <= int(cls_id) < model.num_classes:
            model.unit_ranks["fc2"][int(cls_id)] = 1
    context_detector.episode_classes[task_id] = list(int(c) for c in new_classes)

    per_class = max(1, int(config.get("nice_memo_per_class", config.get("memo_per_class", 50))))
    ref_data, ref_labels = _sample_reference_with_labels(
        client.X_train, client.y_train, list(new_classes), per_class, device
    )
    revived = revive_due_recycled_neurons(model, task_id, config)

    is_first_task = not novelty_estimator.has_history()
    novelty = 0.0
    if ref_data.numel() > 0:
        model.eval()
        if novelty_estimator.thresholds is None:
            novelty_estimator.calibrate_thresholds(model, ref_data)
        if not is_first_task:
            novelty = float(novelty_estimator.compute_novelty(model, ref_data)["novelty"])

    start_ages = model.get_neuron_ages_state()
    capacity_state = compute_capacity_state(model)
    consumption = compute_consumption(prev_ages, start_ages)
    val_loss_delta = _compute_val_loss_delta(
        model,
        old_ref_bank,
        old_ref_loss_baseline,
        device,
        batch_size=int(config.get("denice_val_delta_batch_size", config.get("eval_batch_size", 8192))),
    )
    controller = CapacityController(CANCConfig.from_dict(config))
    plan = controller.plan_task(
        capacity_state,
        novelty,
        consumption,
        val_loss_delta=val_loss_delta,
        is_first_task=is_first_task,
    )
    plan["novelty"] = novelty
    plan["val_loss_delta"] = val_loss_delta
    plan["start_ages"] = start_ages
    plan["revived"] = revived

    model.clear_active_adapters()
    for layer in plan["adapters_to_add"]:
        model.add_adapter(task_id, layer, set_active=True)
    apply_graceful_recycling(model, ref_data, task_id, plan, config)

    if plan["freeze_low_layers"]:
        for low in ("conv1", "conv2"):
            ranks = model.unit_ranks.get(low)
            if ranks is not None:
                model.freeze_masks[low] = np.ones(len(ranks), dtype=bool)

    return {"plan": plan, "ref_data": ref_data, "ref_labels": ref_labels}


def _build_round_capsule(
    *,
    cid: int,
    task_id: int,
    round_id: int,
    model: DeNICEModel,
    client,
    context_detector: ContextDetector,
    ref_data: torch.Tensor,
    ref_labels: Optional[torch.Tensor],
    loss: float,
) -> Any:
    reliability = 1.0 / (1.0 + max(0.0, float(loss)))
    labels = sorted(set(int(c) for c in client.y_train.detach().cpu().tolist()))
    sample_count = int(len(client.y_train))
    thresholds = getattr(context_detector, "binarize_thresholds", None)
    device = next(model.parameters()).device
    capsule_data = ref_data if ref_data.numel() > 0 else client.X_train[: min(32, len(client.y_train))]
    capsule_data = capsule_data.to(device)
    if ref_labels is not None and ref_labels.numel() == len(capsule_data):
        capsule_labels = ref_labels.to(device)
    else:
        capsule_labels = client.y_train[: len(capsule_data)].to(device)
    return build_context_capsule(
        model,
        capsule_data,
        client_id=int(cid),
        task_id=int(task_id),
        round_id=int(round_id),
        label_histogram=_label_histogram(client.y_train),
        label_set=labels,
        sample_count=sample_count,
        reliability=reliability,
        thresholds=thresholds,
        context_detector=context_detector,
        labels=capsule_labels,
    )


def _aggregate_round(
    *,
    client_ids: List[int],
    models: Dict[int, DeNICEModel],
    capsules: Dict[int, Any],
    config: Dict[str, Any],
    device: torch.device,
    previous_valid_cluster: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Cluster capsules and apply age-aware decentralized aggregation."""
    ordered_caps = [capsules[cid] for cid in client_ids]
    # Context graph E_ij = 1 iff clients are context-compatible (proposal section 6).
    # delta<=0 lets clustering derive an adaptive threshold and mutual top-k graph.
    delta_sim = float(config.get("denice_cluster_delta_sim", 0.0))
    cluster_config = ClusteringConfig(
        delta_sim=delta_sim,
        theta_s=float(config.get("denice_cluster_theta_s", 0.5)),
        edge_top_k=int(config.get("denice_cluster_edge_top_k", 20)),
        edge_quantile=float(config.get("denice_cluster_edge_quantile", 0.40)),
        min_signal_std=float(config.get("denice_cluster_min_signal_std", 0.02)),
    )
    sim_weights = SimilarityWeights()
    cluster_result = dynamic_ap_cluster(
        ordered_caps, config=cluster_config, weights=sim_weights
    )
    raw_labels = np.asarray(cluster_result["labels"], dtype=np.int64)
    labels, context_edges, cluster_policy, fallback_reason, next_valid_cluster = (
        _effective_cluster_assignment(
            cluster_result,
            client_ids,
            previous_valid_cluster,
            invalid_policy=str(
                config.get(
                    "denice_cluster_invalid_policy", "previous_valid_or_self_only"
                )
            ),
        )
    )
    # When enabled, restrict each collaboration group to context neighbors
    # (s_ij > delta), i.e. G_i = {j | C[j]=C[i] AND s_ij > delta} (Đề xuất §6).
    use_context_edges = bool(config.get("denice_collab_use_context_edges", True))
    require_label_overlap = bool(config.get("denice_require_label_overlap", True))
    centroid_gate_threshold = float(config.get("denice_centroid_gate_threshold", 0.75))
    agg_config = AggregationConfig(
        eta=float(config.get("denice_aggregation_eta", 1.0)),
        protect_mature=bool(config.get("denice_protect_mature", True)),
        method=str(config.get("denice_aggregation_method", "weighted_mean")),
        trim_ratio=float(config.get("denice_aggregation_trim_ratio", 0.1)),
    )

    old_states = {cid: _state_dict(models[cid]) for cid in client_ids}
    old_ages = {cid: models[cid].get_neuron_ages_state() for cid in client_ids}
    old_adapter_states = {cid: _adapter_states(models[cid]) for cid in client_ids}
    new_states: Dict[int, OrderedDict] = {}
    new_ages: Dict[int, Dict[str, np.ndarray]] = {}
    groups: Dict[int, List[int]] = {}
    alpha_debug: Dict[int, Dict[str, Any]] = {}
    age_peer_promotions: Dict[int, Dict[str, int]] = {}
    group_sizes: List[int] = []
    alpha_values: List[float] = []

    for idx, cid in enumerate(client_ids):
        neighbors = None
        if (
            use_context_edges
            and context_edges is not None
            and getattr(context_edges, "shape", (0,))[0] == len(client_ids)
        ):
            neighbors = [
                j for j in range(len(client_ids)) if int(context_edges[idx, j]) == 1
            ]
        group_indices = collaboration_group(idx, labels, neighbors=neighbors)
        group_ids = [client_ids[g] for g in group_indices]
        if require_label_overlap:
            group_ids = [
                gid
                for gid in group_ids
                if gid == cid
                or label_overlap(capsules[cid].label_set, capsules[gid].label_set) > 0.0
            ]
            if cid not in group_ids:
                group_ids.append(cid)
            group_ids = sorted(set(int(gid) for gid in group_ids))
        centroid_distances = _cosine_distance_to_centroid(
            {gid: capsules[gid].proto_vector() for gid in group_ids}
        )
        if centroid_gate_threshold > 0 and len(group_ids) > 2:
            group_ids = [
                gid
                for gid in group_ids
                if gid == cid
                or centroid_distances.get(int(gid), 0.0) <= centroid_gate_threshold
            ]
            if cid not in group_ids:
                group_ids.append(cid)
            group_ids = sorted(set(int(gid) for gid in group_ids))
        groups[int(cid)] = [int(x) for x in group_ids]
        group_sizes.append(len(group_ids))

        sims = []
        counts = []
        rels = []
        for gid in group_ids:
            if gid == cid:
                sims.append(1.0)
            else:
                sims.append(max(0.0, context_similarity(capsules[cid], capsules[gid], sim_weights)))
            counts.append(float(capsules[gid].sample_count))
            rels.append(float(capsules[gid].reliability))
        self_index = group_ids.index(cid)
        alphas = aggregation_weights(
            sims,
            counts,
            rels,
            self_index=self_index,
            count_transform=str(config.get("denice_aggregation_count_transform", "log")),
            self_floor=float(config.get("denice_aggregation_self_floor", 0.25)),
        )
        alpha_values.extend(float(a) for a in alphas)
        alpha_debug[int(cid)] = {
            "group_ids": [int(x) for x in group_ids],
            "similarities": [float(x) for x in sims],
            "sample_counts": [float(x) for x in counts],
            "reliabilities": [float(x) for x in rels],
            "alphas": [float(x) for x in alphas],
            "centroid_distances": {
                int(gid): float(centroid_distances.get(int(gid), 0.0))
                for gid in group_ids
            },
            "centroid_gate_threshold": float(centroid_gate_threshold),
        }

        target_state = old_states[cid]
        deltas = [_delta_to_target(target_state, old_states[gid]) for gid in group_ids]
        new_states[cid] = age_aware_aggregate(
            target_state,
            old_ages[cid],
            deltas,
            alphas,
            agg_config,
        )
        neighbor_adapter_states = []
        neighbor_adapter_weights = []
        for pos, gid in enumerate(group_ids):
            if gid == cid:
                continue
            neighbor_adapter_states.append(old_adapter_states[gid])
            neighbor_adapter_weights.append(float(alphas[pos]))
        merged_adapters = aggregate_adapters(
            old_adapter_states[cid],
            neighbor_adapter_states,
            neighbor_adapter_weights,
            target_weight=float(alphas[self_index]),
        )
        age_merge_policy = str(config.get("denice_age_merge_policy", "consensus"))
        age_consensus_threshold = float(
            config.get("denice_age_merge_consensus_threshold", 0.5)
        )
        _inject_adapter_states(new_states[cid], merged_adapters)
        new_ages[cid] = merge_neuron_ages(
            old_ages[cid],
            [old_ages[gid] for gid in group_ids if gid != cid],
            neighbor_weights=neighbor_adapter_weights,
            policy=age_merge_policy,
            consensus_threshold=age_consensus_threshold,
        )
        age_peer_promotions[int(cid)] = {
            layer: int(
                ((np.asarray(new_ages[cid][layer]) >= 2)
                 & (np.asarray(old_ages[cid][layer]) < 2)).sum()
            )
            for layer in old_ages[cid]
            if layer in new_ages[cid]
            and np.asarray(new_ages[cid][layer]).shape == np.asarray(old_ages[cid][layer]).shape
        }

    for cid in client_ids:
        models[cid].load_state_dict(new_states[cid], strict=False)
        models[cid].set_neuron_ages_state(new_ages[cid])
        update_freeze_masks(models[cid])
        if hasattr(models[cid], "freeze_bn_for_mature"):
            models[cid].freeze_bn_for_mature()
        models[cid].to(device)

    capacity_after_aggregation = {
        int(cid): _capacity_debug(models[cid]) for cid in client_ids
    }
    capacity_guardrails: Dict[int, Dict[str, Any]] = {}
    for cid, state in capacity_after_aggregation.items():
        free_by_layer = {layer: float(info.get("rho0", 0.0)) for layer, info in state.items()}
        min_free = min(free_by_layer.values()) if free_by_layer else 1.0
        level = (
            "critical" if min_free < 0.01 else
            "severe" if min_free < 0.05 else
            "warning" if min_free < 0.10 else
            None
        )
        if level is not None:
            capacity_guardrails[int(cid)] = {
                "level": level,
                "min_rho0": float(min_free),
                "layers": [layer for layer, rho0 in free_by_layer.items() if rho0 < 0.10],
            }

    label_map = {int(cid): int(labels[i]) for i, cid in enumerate(client_ids)}
    cluster_sizes = {
        int(label): int((labels == label).sum())
        for label in sorted(set(int(x) for x in labels.tolist()))
    }
    sim_matrix = np.asarray(cluster_result.get("similarity", np.zeros((0, 0))))
    finite_sim = sim_matrix[(sim_matrix > -1e8) & np.isfinite(sim_matrix)]
    return {
        "K_t": int(len(set(int(x) for x in labels.tolist()))),
        "raw_K_t": int(cluster_result["K_t"]),
        "effective_K_t": int(len(set(int(x) for x in labels.tolist()))),
        "silhouette": (
            None
            if not np.isfinite(cluster_result["silhouette"])
            else float(cluster_result["silhouette"])
        ),
        "valid": bool(cluster_result["valid"]),
        "raw_valid": bool(cluster_result["valid"]),
        "effective_policy": cluster_policy,
        "fallback_reason": fallback_reason,
        "labels": label_map,
        "groups": groups,
        "cluster_sizes": cluster_sizes,
        "similarity_stats": _round_float_stats(finite_sim.tolist()),
        "group_size_stats": _round_float_stats([float(x) for x in group_sizes]),
        "alpha_stats": _round_float_stats(alpha_values),
        "alpha_debug": alpha_debug,
        "age_peer_promotions": age_peer_promotions,
        "capacity_after_aggregation": capacity_after_aggregation,
        "capacity_guardrails": capacity_guardrails,
        "next_valid_cluster": next_valid_cluster,
    }


def _build_shared_context_detector(
    context_detectors: Dict[int, ContextDetector],
    memo_per_class: int,
    max_per_episode: Optional[int],
    seed: int,
    client_ids: Optional[List[int]] = None,
    router_mode: str = "chained",
    require_compatible_calibration: bool = True,
) -> Optional[ContextDetector]:
    """Pool selected clients' per-episode sketches into a context bank.

    ``client_ids=None`` keeps the old global behavior. Passing a collaboration
    group implements the proposal's neighbor/cluster capsule sharing: a client
    routes with context sketches from its decentralized group, not the whole
    system.
    """
    selected = (
        list(context_detectors.values())
        if client_ids is None
        else [context_detectors[cid] for cid in client_ids if cid in context_detectors]
    )
    detectors = [
        det
        for det in selected
        if getattr(det, "activation_memory", None)
    ]
    if not detectors:
        return None
    shared = build_pooled_context_detector(
        detectors,
        memo_per_class=memo_per_class,
        max_per_episode=max_per_episode,
        seed=seed,
        router_mode=router_mode,
        require_compatible_calibration=require_compatible_calibration,
    )
    if shared is None or not shared.activation_memory:
        return None
    return shared


def _evaluate_clients(
    *,
    client_ids: List[int],
    models: Dict[int, DeNICEModel],
    context_detectors: Dict[int, ContextDetector],
    test_data: Dict[str, torch.Tensor],
    seen_classes: List[int],
    batch_size: int,
    device: torch.device,
    label: str = "eval",
    progress_every_clients: int = 1,
    progress_every_batches: int = 0,
    use_shared_context: bool = True,
    shared_context_scope: str = "cluster",
    context_groups: Optional[Dict[int, List[int]]] = None,
    shared_context_max_per_episode: Optional[int] = None,
    shared_context_memo_per_class: int = 50,
    shared_context_seed: int = 0,
    router_mode: str = "chained",
    require_compatible_calibration: bool = True,
    route_mode: str = "hard",
    route_topk: int = 1,
    report_nomask: bool = True,
    report_representative_ensemble: bool = True,
) -> Dict[str, Any]:
    metrics = []
    per_client = {}
    total_samples = int(len(test_data.get("y_test", [])))
    eval_start = time.time()

    # Context routing bank. The proposal uses capsule exchange inside
    # decentralized groups, so the default is cluster/neighbor pooling. Global
    # pooling is kept only for ablation/debug because it is slower and less
    # decentralized.
    shared_detector: Optional[ContextDetector] = None
    detector_cache: Dict[Tuple[int, ...], Optional[ContextDetector]] = {}
    shared_context_scope = str(shared_context_scope or "cluster").lower()
    if use_shared_context and shared_context_scope == "global":
        shared_detector = _build_shared_context_detector(
            context_detectors,
            memo_per_class=shared_context_memo_per_class,
            max_per_episode=shared_context_max_per_episode,
            seed=shared_context_seed,
            router_mode=router_mode,
            require_compatible_calibration=require_compatible_calibration,
        )
    if not use_shared_context or shared_context_scope == "local":
        routing_mode = "per-client"
    elif shared_context_scope == "global" and shared_detector is not None:
        routing_mode = "global"
    elif shared_context_scope == "cluster":
        routing_mode = "cluster"
    else:
        routing_mode = "per-client"

    print(
        f"  DeNICE eval start [{label}]: clients={len(client_ids)}, "
        f"test_samples={total_samples}, batch_size={batch_size}, "
        f"routing={routing_mode}, "
        f"episodes={sorted(shared_detector.episode_classes) if shared_detector else 'group/local'}, "
        f"seen_classes={list(seen_classes)}",
        flush=True,
    )
    for pos, cid in enumerate(client_ids, start=1):
        client_start = time.time()
        detector = context_detectors[cid]
        if routing_mode == "global" and shared_detector is not None:
            detector = shared_detector
        elif routing_mode == "cluster":
            group_ids = (context_groups or {}).get(int(cid), [int(cid)])
            if int(cid) not in group_ids:
                group_ids = [int(cid), *group_ids]
            group_key = tuple(sorted(set(int(x) for x in group_ids)))
            if group_key not in detector_cache:
                detector_cache[group_key] = _build_shared_context_detector(
                    context_detectors,
                    memo_per_class=shared_context_memo_per_class,
                    max_per_episode=shared_context_max_per_episode,
                    seed=shared_context_seed + int(cid),
                    client_ids=list(group_key),
                    router_mode=router_mode,
                    require_compatible_calibration=require_compatible_calibration,
                )
            detector = detector_cache[group_key] or context_detectors[cid]
        client_metrics = evaluate_denice_model(
            models[cid],
            test_data,
            device=str(device),
            context_detector=detector,
            seen_classes=seen_classes,
            batch_size=batch_size,
            progress_label=f"eval cid={cid}",
            progress_every_batches=progress_every_batches,
            route_mode=route_mode,
            route_topk=route_topk,
            include_route_diagnostics=True,
        )
        if report_nomask and str(route_mode).lower() != "nomask":
            nomask_metrics = evaluate_denice_model(
                models[cid],
                test_data,
                device=str(device),
                context_detector=detector,
                seen_classes=seen_classes,
                batch_size=batch_size,
                route_mode="nomask",
                route_topk=route_topk,
            )
            for key in ("loss", "accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"):
                client_metrics[f"nomask_{key}"] = float(nomask_metrics[key])
        metrics.append(client_metrics)
        per_client[int(cid)] = client_metrics
        client_elapsed = time.time() - client_start
        should_print = (
            pos == 1
            or pos == len(client_ids)
            or (progress_every_clients > 0 and pos % progress_every_clients == 0)
        )
        if should_print:
            print(
                f"    Eval client {pos}/{len(client_ids)} cid={cid} done: "
                f"acc={client_metrics['accuracy'] * 100:.2f}%, "
                f"f1={client_metrics['f1_macro'] * 100:.2f}%, "
                f"route_acc={client_metrics.get('route_accuracy', 0.0) * 100:.2f}%, "
                f"time={client_elapsed:.1f}s, total_elapsed={time.time() - eval_start:.1f}s",
                flush=True,
            )
    if not metrics:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
            "route_accuracy": 0.0,
            "route_coverage": 0.0,
            "routing_mode": routing_mode,
            "per_client": {},
            "per_client_accuracy_stats": {"min": None, "mean": None, "max": None},
        }
    numeric_keys = [
        key for key, value in metrics[0].items()
        if isinstance(value, (int, float, np.floating, np.integer))
    ]
    averaged = {key: float(np.mean([m[key] for m in metrics])) for key in numeric_keys}
    route_confusion: Dict[str, Dict[str, int]] = {}
    for metric in metrics:
        for true_ep, row in metric.get("route_confusion", {}).items():
            total_row = route_confusion.setdefault(str(true_ep), {})
            for pred_ep, count in row.items():
                total_row[str(pred_ep)] = int(total_row.get(str(pred_ep), 0) + int(count))
    averaged["per_client"] = per_client
    averaged["routing_mode"] = routing_mode
    averaged["route_mode"] = str(route_mode)
    averaged["route_topk"] = int(route_topk)
    averaged["route_confusion"] = route_confusion
    averaged["per_client_accuracy_stats"] = _round_float_stats(
        [float(m.get("accuracy", 0.0)) for m in metrics]
    )
    averaged["per_client_route_accuracy_stats"] = _round_float_stats(
        [float(m.get("route_accuracy", 0.0)) for m in metrics]
    )
    if report_representative_ensemble:
        representative_ids: List[int] = []
        seen_groups = set()
        for cid in client_ids:
            group = tuple(
                sorted(set(int(x) for x in (context_groups or {}).get(int(cid), [int(cid)])))
            )
            if group in seen_groups:
                continue
            seen_groups.add(group)
            representative_ids.append(min(group))
        representative_ids = [
            cid for cid in representative_ids
            if cid in models and cid in context_detectors
        ]
        if representative_ids:
            averaged["representative_ensemble"] = evaluate_denice_ensemble(
                [(models[cid], context_detectors[cid]) for cid in representative_ids],
                test_data,
                device=str(device),
                seen_classes=seen_classes,
                batch_size=batch_size,
                route_mode=route_mode,
                route_topk=route_topk,
            )
    print(
        f"  DeNICE eval done [{label}]: "
        f"accuracy={averaged['accuracy'] * 100:.2f}%, "
        f"f1={averaged['f1_macro'] * 100:.2f}%, "
        f"route_acc={averaged.get('route_accuracy', 0.0) * 100:.2f}% "
        f"(routing={routing_mode}), "
        f"elapsed={time.time() - eval_start:.1f}s",
        flush=True,
    )
    return averaged


def run_decentralized_denice_il(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run DeNICE with decentralized capsule clustering and age-aware aggregation."""
    set_seed(config.get("random_seed", config.get("seed", 42)))
    output_dir = _resolve_output_dir(config, "decentralized", "denice")
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("DECENTRALIZED DeNICE-IL")
    print("=" * 80)

    data_loader = IncrementalDataLoader(data_dir=config["data_dir"])
    config["input_shape"] = data_loader.input_shape
    config["num_classes"] = config["total_classes"]
    _write_json(os.path.join(output_dir, "config.json"), config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = get_incremental_strategy(
        "denice",
        **{k: v for k, v in config.items() if k != "algorithm"},
    )
    rounds_per_task = max(1, int(config.get("rounds_per_task", 5)))
    eval_every = max(1, int(config.get("eval_every", rounds_per_task)))
    checkpoint_every = config.get("round_checkpoint_every", 1)
    if checkpoint_every is not None:
        checkpoint_every = max(1, int(checkpoint_every))
    save_round_artifacts = bool(config.get("denice_save_round_artifacts", False))
    checkpoint_format = str(config.get("denice_checkpoint_format", "full")).lower()
    if checkpoint_format not in {"full", "delta"}:
        raise ValueError("denice_checkpoint_format must be 'full' or 'delta'.")
    batch_size = int(config.get("batch_size", 128))
    eval_batch_size = int(config.get("eval_batch_size", 8192))
    post_task_eval = bool(config.get("denice_post_task_eval", True))
    denice_debug = bool(config.get("denice_debug", False))
    eval_progress_every_clients = max(
        1, int(config.get("denice_eval_progress_every_clients", 1))
    )
    eval_progress_every_batches = max(
        0, int(config.get("denice_eval_progress_every_batches", 0))
    )
    eval_max_clients_raw = config.get("denice_eval_max_clients")
    denice_eval_max_clients = (
        None if eval_max_clients_raw is None else int(eval_max_clients_raw)
    )
    eval_max_samples_raw = config.get("denice_eval_max_samples")
    denice_eval_max_samples = (
        None if eval_max_samples_raw is None else int(eval_max_samples_raw)
    )
    max_clients = config.get("denice_max_clients")
    max_clients = None if max_clients is None else int(max_clients)
    max_train_samples_raw = config.get("denice_max_train_samples_per_client")
    denice_max_train_samples_per_client = (
        None if max_train_samples_raw is None else max(1, int(max_train_samples_raw))
    )
    use_shared_context = bool(config.get("denice_shared_context_eval", True))
    shared_context_scope = str(config.get("denice_shared_context_scope", "cluster")).lower()
    denice_router_mode = str(config.get("denice_router_mode", "chained")).lower()
    shared_context_require_compatible_calibration = bool(
        config.get("denice_shared_context_require_compatible_calibration", True)
    )
    denice_eval_route_mode = str(config.get("denice_eval_route_mode", "hard")).lower()
    denice_eval_route_topk = max(1, int(config.get("denice_eval_route_topk", 1)))
    denice_eval_report_nomask = bool(config.get("denice_eval_report_nomask", True))
    denice_eval_representative_ensemble = bool(
        config.get("denice_eval_representative_ensemble", True)
    )
    shared_ctx_cap_raw = config.get("denice_shared_context_max_per_episode")
    shared_context_max_per_episode = (
        None if shared_ctx_cap_raw is None else int(shared_ctx_cap_raw)
    )
    shared_context_memo_per_class = int(
        config.get("nice_memo_per_class", config.get("memo_per_class", 50))
    )
    initial_template = _make_model(config, device)
    initial_model_state = _cpu_state_dict(initial_template)
    del initial_template

    models: Dict[int, DeNICEModel] = {}
    clients: Dict[int, Any] = {}
    context_detectors: Dict[int, ContextDetector] = {}
    novelty_estimators: Dict[int, NoveltyEstimator] = {}
    prev_ages: Dict[int, Optional[Dict[str, np.ndarray]]] = {}
    ref_data: Dict[int, torch.Tensor] = {}
    ref_labels: Dict[int, torch.Tensor] = {}
    old_ref_banks: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = {}
    old_ref_loss_baselines: Dict[int, Optional[float]] = {}
    canc_plans: Dict[int, Dict[str, Any]] = {}
    last_active_task: Dict[int, int] = {}

    history = {"task_accuracies": [], "task_forgetting": [], "round_metrics": []}
    cluster_history: List[Dict[str, Any]] = []
    adapter_history: List[Dict[str, Any]] = []
    debug_history: List[Dict[str, Any]] = []

    num_tasks = data_loader.get_num_tasks()
    task_start = int(config.get("task_start", 0))
    task_end = int(config.get("task_end", num_tasks - 1))

    for task_id in range(task_start, task_end + 1):
        print(f"\n{'=' * 80}\nTASK {task_id}/{num_tasks - 1} - Decentralized DeNICE-IL\n{'=' * 80}")
        new_classes = data_loader.get_task_classes(task_id)
        active_ids = []
        new_model_init_diffs: List[float] = []
        bootstrap_events: List[Dict[str, Any]] = []
        task_debug = {
            "type": "task_start",
            "task": int(task_id),
            "new_classes": [int(c) for c in new_classes],
            "client_data": {},
        }
        for cid in data_loader.get_all_client_ids():
            X, y = data_loader.get_client_data(cid, task_id)
            if (
                denice_max_train_samples_per_client is not None
                and len(y) > denice_max_train_samples_per_client
            ):
                generator = torch.Generator().manual_seed(
                    int(config.get("random_seed", config.get("seed", 42)))
                    + 10_000 * int(task_id)
                    + int(cid)
                )
                selected = torch.randperm(len(y), generator=generator)[:denice_max_train_samples_per_client]
                X, y = X[selected], y[selected]
            if len(y) > 0:
                # Apply the smoke/debug client cap before constructing models
                # and client state; the previous post-loop slice allocated
                # unused models for every active client.
                if max_clients is not None and len(active_ids) >= max_clients:
                    continue
                active_ids.append(int(cid))
                task_debug["client_data"][int(cid)] = {
                    "num_samples": int(len(y)),
                    "class_hist": _count_histogram(y),
                    "labels": sorted(int(c) for c in set(y.detach().cpu().tolist())),
                }
                data = {"X_train": X, "y_train": y}
                if cid not in clients:
                    clients[cid] = create_client(cid, X, y, {**config, "algorithm": "denice"})
                else:
                    update_client_data(clients[cid], data, task_id, new_classes)
                if cid not in models:
                    source_id = (
                        _bootstrap_source_client(models, context_detectors)
                        if task_id > 0
                        else None
                    )
                    if source_id is None:
                        models[cid] = _make_model(config, device)
                        models[cid].load_state_dict(
                            {k: v.to(device) for k, v in initial_model_state.items()},
                            strict=False,
                        )
                        context_detectors[cid] = ContextDetector(
                            memo_per_class=int(config.get("nice_memo_per_class", config.get("memo_per_class", 50))),
                            router_mode=denice_router_mode,
                        )
                        bootstrap_events.append(
                            {
                                "client_id": int(cid),
                                "bootstrap_policy": "initial_template",
                                "bootstrap_source": None,
                                "param_distance_to_initial": 0.0,
                            }
                        )
                    else:
                        models[cid] = _bootstrap_denice_model(
                            models[source_id], config, device
                        )
                        source_detector_state = snapshot_denice_state(
                            models[source_id], context_detectors[source_id]
                        ).get("context_detector")
                        context_detectors[cid] = ContextDetector(
                            memo_per_class=int(config.get("nice_memo_per_class", config.get("memo_per_class", 50))),
                            router_mode=denice_router_mode,
                        )
                        # The new model is an exact clone at this point, so the
                        # source detector's sketches/calibration are compatible.
                        restore_context_detector(
                            context_detectors[cid], source_detector_state
                        )
                        bootstrap_events.append(
                            {
                                "client_id": int(cid),
                                "bootstrap_policy": "representative_clone",
                                "bootstrap_source": int(source_id),
                                "param_distance_to_source": _param_max_abs_diff(
                                    _cpu_state_dict(models[cid]),
                                    _cpu_state_dict(models[source_id]),
                                ),
                                "param_distance_to_initial": _param_max_abs_diff(
                                    _cpu_state_dict(models[cid]), initial_model_state
                                ),
                            }
                        )
                    new_model_init_diffs.append(
                        _param_max_abs_diff(_cpu_state_dict(models[cid]), initial_model_state)
                    )
                    novelty_estimators[cid] = NoveltyEstimator(
                        layer_weights=getattr(trainer, "novelty_layer_weights", None)
                    )
                    prev_ages[cid] = None
                    old_ref_banks[cid] = {}
                    old_ref_loss_baselines[cid] = None
                elif last_active_task.get(int(cid), task_id - 1) < task_id - 1:
                    source_id = _bootstrap_source_client(
                        models, context_detectors, exclude_client=int(cid)
                    )
                    if source_id is not None:
                        updated_distance = _catch_up_rejoining_model(
                            models[cid], models[source_id], device
                        )
                        bootstrap_events.append(
                            {
                                "client_id": int(cid),
                                "bootstrap_policy": "rejoining_plastic_catch_up",
                                "bootstrap_source": int(source_id),
                                "missed_tasks": int(task_id - last_active_task[int(cid)] - 1),
                                "catch_up_param_distance": float(updated_distance),
                            }
                        )
                last_active_task[int(cid)] = int(task_id)

        active_ids = sorted(active_ids)
        print(f"  Active DeNICE clients: {len(active_ids)}")
        if not active_ids:
            continue
        task_debug["new_model_count"] = len(new_model_init_diffs)
        # ``False`` when no model was created in this task; reporting ``True``
        # for an empty list previously looked like a late client had silently
        # been initialized from the random template.
        task_debug["new_model_shared_init"] = bool(
            new_model_init_diffs and max(new_model_init_diffs) <= 1e-12
        )
        task_debug["new_model_max_init_param_diff"] = (
            max(new_model_init_diffs) if new_model_init_diffs else None
        )
        task_debug["active_client_count"] = len(active_ids)
        task_debug["total_samples"] = int(
            sum(v["num_samples"] for v in task_debug["client_data"].values())
        )
        task_debug["bootstrap_events"] = bootstrap_events
        debug_history.append(task_debug)
        if denice_debug:
            print(
                "  DeNICE debug: "
                f"new_model_shared_init={task_debug['new_model_shared_init']}, "
                f"new_model_max_init_param_diff={task_debug['new_model_max_init_param_diff']}, "
                f"samples={task_debug['total_samples']}"
            )

        for cid in active_ids:
            old_ref_banks.setdefault(cid, {})
            old_ref_loss_baselines.setdefault(cid, None)
            prep = _prepare_client_task(
                cid=cid,
                task_id=task_id,
                num_tasks=num_tasks,
                new_classes=new_classes,
                model=models[cid],
                client=clients[cid],
                trainer=trainer,
                config=config,
                device=device,
                context_detector=context_detectors[cid],
                novelty_estimator=novelty_estimators[cid],
                prev_ages=prev_ages.get(cid),
                old_ref_bank=old_ref_banks.get(cid),
                old_ref_loss_baseline=old_ref_loss_baselines.get(cid),
            )
            canc_plans[cid] = prep["plan"]
            ref_data[cid] = prep["ref_data"]
            ref_labels[cid] = prep.get("ref_labels", torch.empty(0, dtype=torch.long))
        if denice_debug:
            debug_history.append(
                {
                    "type": "canc_plan",
                    "task": int(task_id),
                    "clients": {
                        int(cid): {
                            "novelty": float(canc_plans[cid].get("novelty", 0.0)),
                            "actions": {
                                name: info.get("action")
                                for name, info in canc_plans[cid].get("layers", {}).items()
                            },
                            "capacity": {
                                name: {
                                    "rho0": float(info.get("rho0", 0.0)),
                                    "rhom": float(info.get("rhom", 0.0)),
                                    "retired": float(info.get("retired", 0.0)),
                                    "u": float(info.get("u", 0.0)),
                                    "kappa": float(info.get("kappa", 0.0)),
                                    "val_loss_delta": float(info.get("val_loss_delta", 0.0)),
                                }
                                for name, info in canc_plans[cid].get("layers", {}).items()
                            },
                            "adapters_to_add": list(canc_plans[cid].get("adapters_to_add", [])),
                            "val_loss_delta": float(canc_plans[cid].get("val_loss_delta", 0.0)),
                            "recycling": canc_plans[cid].get("recycling", {}),
                        }
                        for cid in active_ids
                    },
                }
            )

        delta_base_path: Optional[str] = None
        previous_round_checkpoint_path: Optional[str] = None
        previous_model_states: Dict[int, OrderedDict[str, torch.Tensor]] = {}
        if checkpoint_every is not None and checkpoint_format == "delta":
            delta_base_path = _base_checkpoint_path(output_dir, task_id)
            save_task_base_checkpoint(
                delta_base_path,
                task_id=task_id,
                client_ids=active_ids,
                models=models,
                client_algorithm_states=_client_algorithm_states(
                    active_ids, models, context_detectors
                ),
                config=config,
                seen_classes=_seen_classes(data_loader, task_id),
            )
            update_checkpoint_index(
                output_dir,
                {
                    "kind": "base",
                    "task_id": int(task_id),
                    "round_id": -1,
                    "path": os.path.basename(delta_base_path),
                    "checkpoint_type": "denice_delta_base",
                },
            )
            previous_model_states = cpu_client_model_states(active_ids, models)
            print(f"   Task base checkpoint saved: {delta_base_path}")

        previous_valid_cluster: Optional[Dict[str, Any]] = None
        for round_id in range(rounds_per_task):
            print(f"  Round {round_id}/{rounds_per_task - 1}")
            start = time.time()
            losses: Dict[int, float] = {}
            client_round_debug: Dict[int, Dict[str, Any]] = {}
            train_time_total = 0.0
            context_time_total = 0.0
            for cid in active_ids:
                model = models[cid]
                client = clients[cid]
                client.setup_for_gpu(model, str(device))
                client_train_start = time.time()
                result = client.train(
                    trainer=trainer,
                    epochs=max(1, int(config.get("nice_phase_epochs", 1))),
                    batch_size=batch_size,
                    lr=float(config.get("learning_rate", 0.001)),
                    global_params=None,
                    is_last_task=(task_id == num_tasks - 1),
                    phase_offset=round_id,
                    max_phases_override=1,
                )
                client_train_time = time.time() - client_train_start
                train_time_total += client_train_time
                losses[cid] = float((result or {}).get("loss", 0.0))
                # Keep the reserve during the task as well as at task end;
                # otherwise a single local NICE phase can transiently consume
                # all plastic units before CANC/aggregation can react.
                reserve_after_train = _enforce_minimum_free_capacity(
                    model,
                    canc_plans.get(cid, {}).get("start_ages", prev_ages.get(cid)),
                    float(config.get("denice_min_free_capacity_ratio", 0.10)),
                )
                update_freeze_masks(model)
                if hasattr(model, "freeze_bn_for_mature"):
                    model.freeze_bn_for_mature()
                context_start = time.time()
                _update_local_nice_context_memory(
                    context_detectors[cid],
                    model,
                    client.X_train,
                    client.y_train,
                    task_id,
                    new_classes,
                    str(device),
                )
                context_time = time.time() - context_start
                context_time_total += context_time
                client_round_debug[int(cid)] = {
                    "loss": losses[cid],
                    "train_time": client_train_time,
                    "context_update_time": context_time,
                    "num_samples": int(len(client.y_train)),
                    "class_hist": _count_histogram(client.y_train),
                    "adapter_usage": compute_adapter_usage(model),
                    "capacity_after_train": _capacity_debug(model),
                    "capacity_reserve_released": reserve_after_train,
                }

            capsule_start = time.time()
            capsules = {
                cid: _build_round_capsule(
                    cid=cid,
                    task_id=task_id,
                    round_id=round_id,
                    model=models[cid],
                    client=clients[cid],
                    context_detector=context_detectors[cid],
                    ref_data=ref_data[cid],
                    ref_labels=ref_labels.get(cid),
                    loss=losses[cid],
                )
                for cid in active_ids
            }
            capsule_time = time.time() - capsule_start
            aggregation_start = time.time()
            cluster_summary = _aggregate_round(
                client_ids=active_ids,
                models=models,
                capsules=capsules,
                config=config,
                device=device,
                previous_valid_cluster=previous_valid_cluster,
            )
            previous_valid_cluster = cluster_summary.pop("next_valid_cluster", None)
            aggregation_time = time.time() - aggregation_start
            router_refresh: Dict[int, Dict[str, int]] = {}
            if bool(config.get("denice_refresh_router_memory_after_aggregation", True)):
                for cid in active_ids:
                    router_refresh[int(cid)] = context_detectors[cid].refresh_activation_memory(
                        models[cid]
                    )
            if cluster_summary.get("capacity_guardrails"):
                print(
                    "    DeNICE capacity guardrail: "
                    f"{cluster_summary['capacity_guardrails']}",
                    flush=True,
                )
            cluster_summary.update({"task": task_id, "round": round_id})
            cluster_summary["router_memory_refresh"] = router_refresh
            cluster_history.append(cluster_summary)

            round_record = {
                "task": task_id,
                "round": round_id,
                "train_loss": float(np.mean(list(losses.values()))),
                "round_time": time.time() - start,
                "train_time": train_time_total,
                "context_update_time": context_time_total,
                "capsule_time": capsule_time,
                "aggregation_time": aggregation_time,
                "checkpoint_time": None,
                "test_loss": None,
                "accuracy": None,
                "precision_macro": None,
                "recall_macro": None,
                "f1_macro": None,
                "f1_weighted": None,
                "avg_forgetting": None,
                "evaluated": False,
                "num_clients": len(active_ids),
                "K_t": cluster_summary["K_t"],
                "silhouette": cluster_summary["silhouette"],
            }
            history["round_metrics"].append(round_record)

            if (round_id + 1) % eval_every == 0 and round_id != rounds_per_task - 1:
                test_X, test_y = data_loader.get_test_data(task_id, cumulative=True)
                test_X, test_y, sample_info = _limit_eval_samples(
                    test_X,
                    test_y,
                    denice_eval_max_samples,
                    seed=int(config.get("random_seed", config.get("seed", 42))) + task_id,
                )
                seen_classes_eval = _seen_classes(data_loader, task_id)
                eval_ids = _select_eval_clients(
                    active_ids,
                    denice_eval_max_clients,
                    context_detectors=context_detectors,
                    seen_classes=seen_classes_eval,
                    require_full_coverage=bool(
                        config.get("denice_eval_require_full_coverage", True)
                    ),
                )
                print(
                    f"  DeNICE eval workload [{task_id}:{round_id}]: "
                    f"clients={len(eval_ids)}/{len(active_ids)}, "
                    f"samples={sample_info['used']}/{sample_info['total']}, "
                    f"sample_limited={sample_info['limited']}",
                    flush=True,
                )
                metrics = _evaluate_clients(
                    client_ids=eval_ids,
                    models=models,
                    context_detectors=context_detectors,
                    test_data={"X_test": test_X, "y_test": test_y},
                    seen_classes=seen_classes_eval,
                    batch_size=eval_batch_size,
                    device=device,
                    label=f"task={task_id},round={round_id}",
                    progress_every_clients=eval_progress_every_clients,
                    progress_every_batches=eval_progress_every_batches,
                    use_shared_context=use_shared_context,
                    shared_context_scope=shared_context_scope,
                    context_groups=cluster_summary.get("groups", {}),
                    shared_context_max_per_episode=shared_context_max_per_episode,
                    shared_context_memo_per_class=shared_context_memo_per_class,
                    shared_context_seed=int(config.get("random_seed", config.get("seed", 42))),
                    router_mode=denice_router_mode,
                    require_compatible_calibration=shared_context_require_compatible_calibration,
                    route_mode=denice_eval_route_mode,
                    route_topk=denice_eval_route_topk,
                    report_nomask=denice_eval_report_nomask,
                    report_representative_ensemble=denice_eval_representative_ensemble,
                )
                metrics["eval_client_count"] = len(eval_ids)
                metrics["eval_total_client_count"] = len(active_ids)
                metrics["eval_sample_count"] = int(len(test_y))
                metrics["eval_total_sample_count"] = int(sample_info["total"])
                metrics["eval_sample_limited"] = bool(sample_info["limited"])
                round_record.update(
                    {
                        "test_loss": metrics.get("loss"),
                        "accuracy": metrics.get("accuracy"),
                        "precision_macro": metrics.get("precision_macro"),
                        "recall_macro": metrics.get("recall_macro"),
                        "f1_macro": metrics.get("f1_macro"),
                        "f1_weighted": metrics.get("f1_weighted"),
                        "evaluated": True,
                    }
                )
                print(
                    f"    eval accuracy={metrics['accuracy'] * 100:.2f}% "
                    f"f1={metrics['f1_macro'] * 100:.2f}% K={cluster_summary['K_t']}"
                )
            else:
                print(
                    f"    Metrics skipped -> train_loss={round_record['train_loss']:.4f}, "
                    f"eval_every={eval_every}"
                )

            save_round_checkpoint = checkpoint_every is not None and (
                round_id == rounds_per_task - 1 or ((round_id + 1) % checkpoint_every == 0)
            )
            checkpoint_start = time.time()
            if save_round_artifacts or (
                save_round_checkpoint and checkpoint_format == "full"
            ):
                _save_round_artifacts(
                    output_dir=output_dir,
                    task_id=task_id,
                    round_id=round_id,
                    client_ids=active_ids,
                    models=models,
                    context_detectors=context_detectors,
                    capsules=capsules,
                    cluster_summary=cluster_summary,
                    adapter_usage={
                        int(cid): compute_adapter_usage(models[cid]) for cid in active_ids
                    },
                    config=config,
                    seen_classes=_seen_classes(data_loader, task_id),
                    round_record=round_record,
                    save_checkpoint=save_round_checkpoint and checkpoint_format == "full",
                )
            if save_round_checkpoint and checkpoint_format == "delta":
                if delta_base_path is None:
                    raise RuntimeError("Delta checkpoint base was not initialized.")
                round_ckpt_path = _round_checkpoint_path(output_dir, task_id, round_id)
                save_delta_round_checkpoint(
                    round_ckpt_path,
                    task_id=task_id,
                    round_id=round_id,
                    base_path=delta_base_path,
                    previous_round_path=previous_round_checkpoint_path,
                    client_ids=active_ids,
                    models=models,
                    previous_model_states=previous_model_states,
                    client_algorithm_states=_client_algorithm_states(
                        active_ids, models, context_detectors
                    ),
                    config=config,
                    seen_classes=_seen_classes(data_loader, task_id),
                    cluster=_json_safe(cluster_summary),
                    metrics=_json_safe(round_record),
                )
                update_checkpoint_index(
                    output_dir,
                    {
                        "kind": "round",
                        "task_id": int(task_id),
                        "round_id": int(round_id),
                        "path": os.path.basename(round_ckpt_path),
                        "base": os.path.basename(delta_base_path),
                        "previous": (
                            os.path.basename(previous_round_checkpoint_path)
                            if previous_round_checkpoint_path
                            else None
                        ),
                        "checkpoint_type": "denice_delta_round",
                    },
                )
                previous_model_states = cpu_client_model_states(active_ids, models)
                previous_round_checkpoint_path = round_ckpt_path
                print(f"   Delta round checkpoint saved: {round_ckpt_path}")
            round_record["checkpoint_time"] = time.time() - checkpoint_start
            round_record["round_time"] = time.time() - start
            debug_round = {
                "type": "round",
                "task": int(task_id),
                "round": int(round_id),
                "timing": {
                    "round_time": round_record["round_time"],
                    "train_time": train_time_total,
                    "context_update_time": context_time_total,
                    "capsule_time": capsule_time,
                    "aggregation_time": aggregation_time,
                    "checkpoint_time": round_record["checkpoint_time"],
                },
                "loss_stats": _round_float_stats(list(losses.values())),
                "clients": client_round_debug if denice_debug else {},
                "cluster": cluster_summary,
            }
            debug_history.append(debug_round)
            if denice_debug:
                _write_json(
                    os.path.join(output_dir, "denice_debug_history.json"),
                    debug_history,
                )
                print(
                    "    DeNICE debug: "
                    f"train={train_time_total:.1f}s, ctx={context_time_total:.1f}s, "
                    f"capsule={capsule_time:.1f}s, agg={aggregation_time:.1f}s, "
                    f"ckpt={round_record['checkpoint_time']:.1f}s, "
                    f"K={cluster_summary['K_t']}, clusters={cluster_summary.get('cluster_sizes')}"
                )

        capacity_reserve_released: Dict[int, Dict[str, int]] = {}
        minimum_free_capacity_ratio = float(
            config.get("denice_min_free_capacity_ratio", 0.10)
        )
        for cid in active_ids:
            model = models[cid]
            capacity_reserve_released[int(cid)] = _enforce_minimum_free_capacity(
                model,
                canc_plans.get(cid, {}).get("start_ages", prev_ages.get(cid)),
                minimum_free_capacity_ratio,
            )
            increase_unit_ranks(model)
            update_freeze_masks(model)
            if hasattr(model, "freeze_bn_for_mature"):
                model.freeze_bn_for_mature()
            if ref_data[cid].numel() > 0:
                proto = novelty_estimators[cid].compute_prototype(model, ref_data[cid])
                novelty_estimators[cid].store_prototype(task_id, proto)
                old_ref_banks.setdefault(cid, {})[int(task_id)] = (
                    ref_data[cid].detach().cpu(),
                    ref_labels[cid].detach().cpu().long(),
                )
                old_ref_loss_baselines[cid] = _compute_reference_ce_loss(
                    model,
                    old_ref_banks.get(cid),
                    device,
                    batch_size=int(
                        config.get("denice_val_delta_batch_size", config.get("eval_batch_size", 8192))
                    ),
                )
            prev_ages[cid] = canc_plans[cid].get("start_ages", model.get_neuron_ages_state())
            adapter_history.append(
                {
                    "task": task_id,
                    "client_id": int(cid),
                    "canc": canc_plans[cid],
                    **compute_adapter_usage(model),
                }
            )
            model.clear_active_adapters()

        final_round_id = rounds_per_task - 1
        seen_classes_eval = _seen_classes(data_loader, task_id)
        final_train_loss = None
        if history.get("round_metrics"):
            final_train_loss = history["round_metrics"][-1].get("train_loss")

        if post_task_eval:
            test_X, test_y = data_loader.get_test_data(task_id, cumulative=True)
            test_X, test_y, sample_info = _limit_eval_samples(
                test_X,
                test_y,
                denice_eval_max_samples,
                seed=int(config.get("random_seed", config.get("seed", 42))) + task_id,
            )
            eval_ids = _select_eval_clients(
                active_ids,
                denice_eval_max_clients,
                context_detectors=context_detectors,
                seen_classes=seen_classes_eval,
                require_full_coverage=bool(config.get("denice_eval_require_full_coverage", True)),
            )
            print(
                f"  Starting post-task DeNICE eval: task={task_id}, "
                f"clients={len(eval_ids)}/{len(active_ids)}, "
                f"test_samples={len(test_y)}/{sample_info['total']}, "
                f"sample_limited={sample_info['limited']}, batch_size={eval_batch_size}",
                flush=True,
            )
            metrics = _evaluate_clients(
                client_ids=eval_ids,
                models=models,
                context_detectors=context_detectors,
                test_data={"X_test": test_X, "y_test": test_y},
                seen_classes=seen_classes_eval,
                batch_size=eval_batch_size,
                device=device,
                label=f"task={task_id},post_task",
                progress_every_clients=eval_progress_every_clients,
                progress_every_batches=eval_progress_every_batches,
                use_shared_context=use_shared_context,
                shared_context_scope=shared_context_scope,
                context_groups=cluster_summary.get("groups", {}),
                shared_context_max_per_episode=shared_context_max_per_episode,
                shared_context_memo_per_class=shared_context_memo_per_class,
                shared_context_seed=int(config.get("random_seed", config.get("seed", 42))),
                router_mode=denice_router_mode,
                require_compatible_calibration=shared_context_require_compatible_calibration,
                route_mode=denice_eval_route_mode,
                route_topk=denice_eval_route_topk,
                report_nomask=denice_eval_report_nomask,
                report_representative_ensemble=denice_eval_representative_ensemble,
            )
            metrics["eval_client_count"] = len(eval_ids)
            metrics["eval_total_client_count"] = len(active_ids)
            metrics["eval_sample_count"] = int(len(test_y))
            metrics["eval_total_sample_count"] = int(sample_info["total"])
            metrics["eval_sample_limited"] = bool(sample_info["limited"])
            metrics["eval_skipped"] = False
        else:
            print(
                "  Post-task DeNICE eval skipped -> denice_post_task_eval=False",
                flush=True,
            )
            metrics = {
                "task": int(task_id),
                "final_round": int(final_round_id),
                "loss": final_train_loss,
                "train_loss": final_train_loss,
                "test_loss": None,
                "accuracy": None,
                "precision_macro": None,
                "recall_macro": None,
                "f1_macro": None,
                "f1_weighted": None,
                "route_accuracy": None,
                "route_coverage": None,
                "routing_mode": "skipped",
                "eval_skipped": True,
                "eval_reason": "denice_post_task_eval=False",
                "per_client": {},
                "eval_client_count": 0,
                "eval_total_client_count": len(active_ids),
                "eval_sample_count": 0,
                "eval_total_sample_count": 0,
                "eval_sample_limited": False,
            }
        debug_history.append(
            {
                "type": "task_summary",
                "task": int(task_id),
                "metrics": metrics,
                "adapter_usage": {
                    int(cid): compute_adapter_usage(models[cid]) for cid in active_ids
                },
                "capacity_end": {
                    int(cid): _capacity_debug(models[cid]) for cid in active_ids
                },
                "capacity_reserve_released": capacity_reserve_released,
            }
        )
        history["task_accuracies"].append(
            {"task": task_id, "final_round": final_round_id, **metrics, "avg_forgetting": None}
        )
        history["task_forgetting"].append({"task": task_id, "avg_forgetting": None})
        if metrics.get("eval_skipped"):
            print(
                "  Task summary -> eval skipped, "
                f"train_loss={final_train_loss if final_train_loss is not None else 'n/a'}"
            )
        else:
            print(
                "  Task summary -> "
                f"accuracy={metrics['accuracy'] * 100:.2f}%, "
                f"f1={metrics['f1_macro'] * 100:.2f}%, "
                f"route_acc={metrics.get('route_accuracy', 0.0) * 100:.2f}% "
                f"(routing={metrics.get('routing_mode', 'per-client')})"
            )

        if bool(config.get("save_resume_after_task", True)):
            torch.save(
                {
                    "task": task_id,
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "mode": "decentralized",
                    "algorithm": "denice",
                    "final_round_id": final_round_id,
                    "client_model_states": {
                        int(cid): _cpu_state_dict(models[cid])
                        for cid in active_ids
                    },
                    "client_algorithm_states": _client_algorithm_states(
                        active_ids, models, context_detectors
                    ),
                    "client_neuron_ages": {
                        int(cid): models[cid].get_neuron_ages_state()
                        for cid in active_ids
                    },
                    "config": config,
                    "seen_classes": _seen_classes(data_loader, task_id),
                    "metrics": metrics,
                },
                os.path.join(output_dir, f"checkpoint_task_{task_id}.pt"),
            )
        _write_phase_outputs(
            output_dir, history, cluster_history, adapter_history, debug_history, config, task_id
        )

    return {
        "output_dir": output_dir,
        "history": history,
        "cluster_history": cluster_history,
        "adapter_history": adapter_history,
    }
