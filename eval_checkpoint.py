"""Evaluate a saved training checkpoint.

Examples:
    python eval_checkpoint.py --checkpoint results/checkpoint_task_5.pt
    python eval_checkpoint.py --checkpoint results/checkpoint_task_2_round_19.pt --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from fed_learning.data.incremental_loader import IncrementalDataLoader
from fed_learning.models.cnn_gru import CNN_GRU_Model
from fed_learning.models.der_model import DERModel
from fed_learning.models.denice_model import DeNICEModel
from fed_learning.models.nice_model import NICEModel
from fed_learning.servers.nice_server import ContextDetector
from fed_learning.strategies.incremental.nice import update_freeze_masks
from fed_learning.training.checkpoint_state import restore_context_detector
from fed_learning.training.denice_delta_checkpoint import load_denice_checkpoint
from fed_learning.training.denice_eval import (
    _denice_routed_logits_with_episodes,
    _label_to_episode_map,
    evaluate_denice_ensemble,
    evaluate_denice_model,
)
from fed_learning.training.der_worker import _reconstruct_model_structure
from fed_learning.training.local_task_loop import _evaluate_model


def _load_checkpoint(path: str) -> Dict[str, Any]:
    checkpoint = load_denice_checkpoint(path)
    # Task-end resume checkpoints predate the delta schema and store ``task``
    # rather than ``task_id``. Normalize at the boundary so every evaluator
    # mode can consume both formats.
    if "task_id" not in checkpoint and "task" in checkpoint:
        checkpoint = dict(checkpoint)
        checkpoint["task_id"] = int(checkpoint["task"])
    return checkpoint

def _dict_get_int(mapping: Dict[Any, Any], key: int, default=None):
    if key in mapping:
        return mapping[key]
    text_key = str(key)
    if text_key in mapping:
        return mapping[text_key]
    return default


def _safe_entropy(probabilities: np.ndarray) -> float:
    if probabilities.size == 0:
        return 0.0
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)
    return float(np.mean(-(clipped * np.log(clipped)).sum(axis=1)))


def _router_memory_arrays(detector_state: Dict[str, Any]) -> Dict[int, np.ndarray]:
    arrays: Dict[int, np.ndarray] = {}
    for episode, value in (detector_state.get("activation_memory") or {}).items():
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim == 2 and len(arr):
            arrays[int(episode)] = arr
    return arrays


def _evaluate_router_predictions(
    detector: ContextDetector,
    memory: Dict[int, np.ndarray],
) -> Dict[str, Any]:
    if not memory:
        return {
            "sample_count": 0,
            "accuracy": None,
            "balanced_accuracy": None,
            "per_episode_recall": {},
            "predicted_episode_distribution": {},
            "mean_confidence": None,
            "mean_entropy": None,
        }
    X = np.concatenate([memory[episode] for episode in sorted(memory)], axis=0)
    y = np.concatenate(
        [np.full(len(memory[episode]), episode, dtype=np.int64) for episode in sorted(memory)]
    )
    predictions, probabilities = detector.predict_episodes_with_scores(X)
    predictions = np.asarray(predictions, dtype=np.int64)
    per_episode_recall = {
        str(episode): float((predictions[y == episode] == episode).mean())
        for episode in sorted(memory)
    }
    unique, counts = np.unique(predictions, return_counts=True)
    return {
        "sample_count": int(len(y)),
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "per_episode_recall": per_episode_recall,
        "predicted_episode_distribution": {
            str(int(episode)): int(count) for episode, count in zip(unique, counts)
        },
        "mean_confidence": float(np.max(probabilities, axis=1).mean()),
        "mean_entropy": _safe_entropy(probabilities),
    }


def _split_router_memory(
    memory: Dict[int, np.ndarray], *, seed: int, holdout_fraction: float = 0.2
) -> tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    train: Dict[int, np.ndarray] = {}
    holdout: Dict[int, np.ndarray] = {}
    generator = np.random.default_rng(int(seed))
    for episode, values in memory.items():
        if len(values) < 2:
            continue
        order = generator.permutation(len(values))
        holdout_count = max(1, int(round(len(values) * holdout_fraction)))
        holdout_count = min(holdout_count, len(values) - 1)
        holdout[int(episode)] = values[order[:holdout_count]]
        train[int(episode)] = values[order[holdout_count:]]
    return train, holdout


def audit_denice_router_states(
    checkpoint: Dict[str, Any], *, seed: int = 42, holdout_fraction: float = 0.2
) -> Dict[str, Any]:
    """Audit saved local router state without reconstructing the model delta chain.

    ``persisted_memory`` checks the actual serialized router against its stored
    sketches.  ``refit_holdout`` retrains a router on a deterministic 80/20
    per-episode split, testing whether the stored binary feature space can
    separate tasks at all.  It intentionally does not claim anything about
    current-model feature drift; that is a later model-dependent diagnostic.
    """
    algorithm_states = checkpoint.get("client_algorithm_states", {})
    config = checkpoint.get("config", {})
    records: list[Dict[str, Any]] = []
    coverage_histogram: Dict[str, int] = {}
    for client_id_raw, state in algorithm_states.items():
        client_id = int(client_id_raw)
        denice_state = (state or {}).get("denice", state or {})
        detector_state = denice_state.get("context_detector") or {}
        memory = _router_memory_arrays(detector_state)
        episode_classes = {
            int(ep): [int(cls) for cls in classes]
            for ep, classes in (detector_state.get("episode_classes") or {}).items()
        }
        memory_episodes = sorted(memory)
        coverage_key = ",".join(map(str, memory_episodes)) or "none"
        coverage_histogram[coverage_key] = coverage_histogram.get(coverage_key, 0) + 1
        detector = ContextDetector(
            memo_per_class=int(detector_state.get("memo_per_class", config.get("memo_per_class", 50))),
            router_mode=str(detector_state.get("router_mode", config.get("denice_router_mode", "chained"))),
        )
        restore_context_detector(detector, detector_state)
        persisted = _evaluate_router_predictions(detector, memory)
        train_memory, holdout_memory = _split_router_memory(
            memory, seed=int(seed) + client_id, holdout_fraction=holdout_fraction
        )
        refit = None
        if len(train_memory) >= 2 and len(holdout_memory) >= 2:
            refit_detector = ContextDetector(
                memo_per_class=detector.memo_per_class,
                router_mode=detector.router_mode,
                calibration_provenance=detector.calibration_provenance,
            )
            refit_detector.activation_memory = {ep: values.copy() for ep, values in train_memory.items()}
            refit_detector.context_masks = {
                int(ep): np.asarray(
                    _dict_get_int(detector_state.get("context_masks") or {}, int(ep), []),
                    dtype=bool,
                ).copy()
                for ep in train_memory
            }
            refit_detector.episode_classes = {
                ep: list(episode_classes.get(ep, [])) for ep in train_memory
            }
            refit_detector.binarize_thresholds = detector.binarize_thresholds
            refit_detector.train_models(max(train_memory))
            refit = _evaluate_router_predictions(refit_detector, holdout_memory)
        records.append(
            {
                "client_id": client_id,
                "router_mode": detector.router_mode,
                "episode_classes": sorted(episode_classes),
                "memory_episodes": memory_episodes,
                "router_classes": [int(ep) for ep in getattr(detector.multiclass_router, "classes_", [])]
                if detector.multiclass_router is not None else [],
                "calibration_signature": detector.calibration_signature(),
                "memory_sample_count": int(sum(len(values) for values in memory.values())),
                "persisted_memory": persisted,
                "refit_holdout": refit,
            }
        )
    eligible = [record for record in records if record["refit_holdout"] is not None]
    return {
        "audit_protocol": "saved_activation_memory_in_sample_and_refit_holdout",
        "seed": int(seed),
        "holdout_fraction": float(holdout_fraction),
        "client_count": len(records),
        "eligible_holdout_client_count": len(eligible),
        "memory_episode_coverage_histogram": coverage_histogram,
        "mean_persisted_balanced_accuracy": float(np.mean([
            record["persisted_memory"]["balanced_accuracy"]
            for record in records
            if record["persisted_memory"]["balanced_accuracy"] is not None
        ])) if records else None,
        "mean_refit_holdout_balanced_accuracy": float(np.mean([
            record["refit_holdout"]["balanced_accuracy"] for record in eligible
        ])) if eligible else None,
        "worst_refit_holdout_clients": sorted(
            eligible,
            key=lambda record: record["refit_holdout"]["balanced_accuracy"],
        )[:10],
        "clients": records,
    }


def _balanced_episode_indices(
    test_y: torch.Tensor,
    task_classes: Dict[int, list[int]],
    *,
    samples_per_episode: int,
    seed: int,
) -> tuple[torch.Tensor, np.ndarray]:
    """Choose a deterministic, class-imbalanced-safe router audit subset."""
    class_to_episode = {
        int(class_id): int(episode)
        for episode, classes in task_classes.items()
        for class_id in classes
    }
    max_class = max(class_to_episode, default=-1)
    lookup = torch.full((max_class + 1,), -1, dtype=torch.long)
    for class_id, episode in class_to_episode.items():
        lookup[class_id] = int(episode)
    labels = test_y.detach().cpu().long()
    target_episodes = torch.full_like(labels, -1)
    valid = (labels >= 0) & (labels <= max_class)
    target_episodes[valid] = lookup[labels[valid]]
    generator = torch.Generator().manual_seed(int(seed))
    selected: list[torch.Tensor] = []
    expected: list[np.ndarray] = []
    for episode in sorted(task_classes):
        candidates = torch.nonzero(target_episodes == int(episode), as_tuple=False).flatten()
        if not len(candidates):
            continue
        count = min(int(samples_per_episode), len(candidates))
        candidates = candidates[torch.randperm(len(candidates), generator=generator)[:count]]
        selected.append(candidates)
        expected.append(np.full(count, int(episode), dtype=np.int64))
    if not selected:
        return torch.empty(0, dtype=torch.long), np.empty(0, dtype=np.int64)
    return torch.cat(selected), np.concatenate(expected)


def _binary_current_context_features(
    model: DeNICEModel, detector: ContextDetector, X: torch.Tensor
) -> np.ndarray:
    activations = model.get_context_activations_per_sample(X)
    layer_acts = {
        name: np.asarray(value.detach().cpu().tolist(), dtype=np.float32)
        for name, value in activations.items()
    }
    return detector.binarize_layer_activations(layer_acts)


@torch.no_grad()
def audit_denice_router_current_features(
    checkpoint: Dict[str, Any],
    test_X: torch.Tensor,
    test_y: torch.Tensor,
    task_classes: Dict[int, list[int]],
    *,
    device: str,
    seed: int = 42,
    max_clients: int = 10,
    samples_per_episode: int = 256,
) -> Dict[str, Any]:
    """Measure router accuracy on features emitted by the *final* saved model.

    Router memory for an old task was captured when the old model state was
    current.  This audit therefore compares the stored binary prototype against
    a final-model prototype on the same task's held-out samples.  It only uses
    full-coverage clients by default, separating feature drift from clients
    that never learned an episode.
    """
    client_ids = [
        int(cid) for cid in checkpoint.get("client_ids", checkpoint.get("client_model_states", {}))
    ]
    # ``task_classes`` contains the whole dataset schedule. A checkpoint from
    # task t must be audited only on task episodes that actually occur in its
    # cumulative test labels, never future episodes that it could not learn.
    class_to_episode = {
        int(class_id): int(episode)
        for episode, classes in task_classes.items()
        for class_id in classes
    }
    required_episodes = {
        int(class_to_episode[int(label)])
        for label in test_y.detach().cpu().tolist()
        if int(label) in class_to_episode
    }
    coverage = {
        client_id: _client_router_episode_coverage(checkpoint, client_id)
        for client_id in client_ids
    }
    full_coverage = [
        client_id for client_id in client_ids
        if required_episodes.issubset(set(coverage[client_id]["supported_episodes"]))
    ]
    selected_clients = sorted(full_coverage)[:max(1, int(max_clients))]
    indices, true_episodes = _balanced_episode_indices(
        test_y,
        task_classes,
        samples_per_episode=samples_per_episode,
        seed=seed,
    )
    if not len(indices):
        raise ValueError("Current-feature router audit found no task-mapped test samples")
    required_episodes = set(int(episode) for episode in true_episodes)
    X_selected = test_X[indices]
    records: list[Dict[str, Any]] = []
    for client_id in selected_clients:
        model, detector = _make_denice_client_model(checkpoint, client_id, device)
        features = _binary_current_context_features(model, detector, X_selected.to(device))
        predictions, probabilities = detector.predict_episodes_with_scores(features)
        predictions = np.asarray(predictions, dtype=np.int64)
        current_metrics = {
            "sample_count": int(len(true_episodes)),
            "accuracy": float(accuracy_score(true_episodes, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(true_episodes, predictions)),
            "per_episode_recall": {
                str(episode): float((predictions[true_episodes == episode] == episode).mean())
                for episode in sorted(set(int(ep) for ep in true_episodes))
            },
            "predicted_episode_distribution": {
                str(int(ep)): int(count)
                for ep, count in zip(*np.unique(predictions, return_counts=True))
            },
            "mean_confidence": float(np.max(probabilities, axis=1).mean()),
            "mean_entropy": _safe_entropy(probabilities),
        }
        denice_state = _dict_get_int(checkpoint.get("client_algorithm_states", {}), client_id, {}) or {}
        denice_state = denice_state.get("denice", denice_state)
        old_memory = _router_memory_arrays((denice_state.get("context_detector") or {}))
        drift = {}
        for episode in sorted(required_episodes.intersection(old_memory)):
            current_rows = features[true_episodes == episode]
            old_rows = old_memory[episode]
            if not len(current_rows) or not len(old_rows):
                continue
            current_prototype = current_rows.mean(axis=0)
            old_prototype = old_rows.mean(axis=0)
            denominator = float(np.linalg.norm(current_prototype) * np.linalg.norm(old_prototype))
            drift[str(episode)] = {
                "prototype_mean_abs_delta": float(np.abs(current_prototype - old_prototype).mean()),
                "prototype_cosine_similarity": (
                    float(np.dot(current_prototype, old_prototype) / denominator)
                    if denominator > 1e-12 else None
                ),
            }
        records.append({
            "client_id": client_id,
            "coverage": coverage[client_id],
            "current_feature_router": current_metrics,
            "prototype_drift_by_episode": drift,
        })
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return {
        "audit_protocol": "final_model_feature_router_audit_on_balanced_global_test_subset",
        "seed": int(seed),
        "samples_per_episode": int(samples_per_episode),
        "total_selected_sample_count": int(len(indices)),
        "required_episodes": sorted(required_episodes),
        "full_coverage_client_count": len(full_coverage),
        "audited_client_count": len(records),
        "mean_current_feature_balanced_accuracy": float(np.mean([
            record["current_feature_router"]["balanced_accuracy"] for record in records
        ])) if records else None,
        "worst_current_feature_clients": sorted(
            records,
            key=lambda record: record["current_feature_router"]["balanced_accuracy"],
        )[:10],
        "clients": records,
    }

def _make_denice_client_model(
    ckpt: Dict[str, Any],
    client_id: int,
    device: str,
    router_mode: str | None = None,
):
    config = dict(ckpt["config"])
    input_shape = config["input_shape"]
    num_classes = config.get("num_classes", config.get("total_classes"))
    model = DeNICEModel(input_shape, num_classes)

    algorithm_states = ckpt.get("client_algorithm_states", {})
    client_alg = _dict_get_int(algorithm_states, int(client_id), {}) or {}
    denice_state = client_alg.get("denice", client_alg)

    adapter_registry = denice_state.get("adapter_registry", {}) or {}
    for meta in adapter_registry.values():
        layer_name = meta.get("layer_name")
        context_id = meta.get("context_id")
        rank = meta.get("rank")
        if layer_name is not None and context_id is not None:
            model.add_adapter(int(context_id), str(layer_name), rank=rank, set_active=False)

    neuron_ages = denice_state.get("neuron_ages")
    if neuron_ages:
        model.set_neuron_ages_state(neuron_ages)
        update_freeze_masks(model)
    freeze_masks = denice_state.get("freeze_masks")
    if freeze_masks:
        model.freeze_masks = freeze_masks
    recycling = denice_state.get("recycling_registry")
    if recycling and hasattr(model, "set_recycling_state"):
        model.set_recycling_state(recycling)

    state_dict = OrderedDict(
        _dict_get_int(ckpt["client_model_states"], int(client_id), {}) or {}
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"warning: client {client_id} missing keys: {len(missing)}")
    if unexpected:
        print(f"warning: client {client_id} unexpected keys: {len(unexpected)}")

    context_detector = ContextDetector(
        memo_per_class=int(config.get("memo_per_class", 50)),
        router_mode=str(router_mode or config.get("denice_router_mode", "chained")),
    )
    restore_context_detector(context_detector, denice_state.get("context_detector"))
    if router_mode is not None and context_detector.router_mode != str(router_mode).lower():
        context_detector.router_mode = str(router_mode).lower()
        if context_detector.activation_memory:
            context_detector.train_models(max(context_detector.activation_memory))

    model.to(device)
    model.eval()
    return model, context_detector


def _make_model(ckpt: Dict[str, Any], device: str):
    config = dict(ckpt["config"])
    algorithm = str(ckpt.get("algorithm") or config.get("algorithm", "")).lower()
    input_shape = config["input_shape"]
    num_classes = config.get("num_classes", config.get("total_classes"))
    state_dict = OrderedDict(ckpt["model_state_dict"])
    algorithm_state = ckpt.get("algorithm_state", {})

    context_detector = None

    if algorithm == "nice":
        model = NICEModel(input_shape, num_classes)
        nice_state = algorithm_state.get("nice", {})
        neuron_ages = nice_state.get("neuron_ages")
        if neuron_ages:
            model.set_neuron_ages_state(neuron_ages)
            update_freeze_masks(model)
        freeze_masks = nice_state.get("freeze_masks")
        if freeze_masks:
            model.freeze_masks = freeze_masks
        context_detector = ContextDetector(
            memo_per_class=int(config.get("memo_per_class", 50))
        )
        restore_context_detector(context_detector, nice_state.get("context_detector"))

    elif algorithm == "der":
        model = DERModel(input_shape, num_classes)
        der_state = algorithm_state.get("der", {})
        task_classes_history = der_state.get("task_classes_history", {})
        recon_config = {
            **config,
            "task_classes_history": task_classes_history,
            "s_max": der_state.get("s_max", config.get("s_max", 15.0)),
        }
        _reconstruct_model_structure(model, state_dict, recon_config)

    else:
        model = CNN_GRU_Model(input_shape, num_classes)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, context_detector


def _client_router_episode_coverage(
    ckpt: Dict[str, Any], client_id: int
) -> Dict[str, Any]:
    """Return only episodes that a saved local router can legitimately score.

    A local detector is *not* assumed to cover every globally seen task.  The
    coverage is the intersection of declared episode classes and stored router
    memory; for the saved multiclass router it must also be a fitted class.
    """
    algorithm_states = ckpt.get("client_algorithm_states", {})
    client_alg = _dict_get_int(algorithm_states, int(client_id), {}) or {}
    denice_state = client_alg.get("denice", client_alg)
    detector = denice_state.get("context_detector") or {}
    episode_classes = {
        int(ep): [int(cls) for cls in classes]
        for ep, classes in (detector.get("episode_classes") or {}).items()
    }
    memory_episodes = {int(ep) for ep in (detector.get("activation_memory") or {})}
    router = detector.get("multiclass_router")
    router_classes = (
        {int(ep) for ep in getattr(router, "classes_", [])}
        if router is not None
        else set()
    )
    supported = set(episode_classes).intersection(memory_episodes)
    if router is not None:
        supported.intersection_update(router_classes)
    adapter_contexts = sorted(
        {
            int(meta.get("context_id"))
            for meta in (denice_state.get("adapter_registry") or {}).values()
            if meta.get("context_id") is not None
        }
    )
    return {
        "supported_episodes": sorted(supported),
        "episode_classes": episode_classes,
        "memory_episodes": sorted(memory_episodes),
        "router_classes": sorted(router_classes),
        "adapter_contexts": adapter_contexts,
    }


def _build_coverage_aware_partitions(
    test_y: torch.Tensor,
    client_ids: list[int],
    task_classes: Dict[int, list[int]],
    coverage_by_client: Dict[int, Dict[str, Any]],
    *,
    seed: int,
) -> tuple[Dict[int, torch.Tensor], Dict[str, Any]]:
    """Assign every supportable test sample to one client with true-episode coverage.

    This intentionally uses the dataset task map only to construct a valid
    evaluation assignment.  The evaluator still gives the selected client no
    oracle route at inference time; it measures that client's predicted router.
    """
    class_to_episode = {
        int(class_id): int(episode)
        for episode, classes in task_classes.items()
        for class_id in classes
    }
    target_episodes = torch.as_tensor(
        [class_to_episode.get(int(label), -1) for label in test_y.tolist()], dtype=torch.long
    )
    generator = torch.Generator().manual_seed(int(seed))
    assigned: Dict[int, list[torch.Tensor]] = {int(cid): [] for cid in client_ids}
    unsupported_by_episode: Dict[str, int] = {}
    eligible_by_episode: Dict[str, int] = {}

    for episode in sorted(set(int(ep) for ep in target_episodes.tolist())):
        episode_indices = torch.nonzero(target_episodes == episode, as_tuple=False).flatten()
        if episode < 0:
            unsupported_by_episode["unknown"] = int(len(episode_indices))
            continue
        eligible = [
            int(cid)
            for cid in client_ids
            if episode in set(coverage_by_client[int(cid)]["supported_episodes"])
        ]
        eligible_by_episode[str(episode)] = len(eligible)
        if not eligible:
            unsupported_by_episode[str(episode)] = int(len(episode_indices))
            continue
        shuffled = episode_indices[torch.randperm(len(episode_indices), generator=generator)]
        for offset, sample_index in enumerate(shuffled):
            assigned[eligible[offset % len(eligible)]].append(sample_index.reshape(1))

    partitions = {
        cid: torch.cat(indices) if indices else torch.empty(0, dtype=torch.long)
        for cid, indices in assigned.items()
    }
    assigned_count = sum(int(indices.numel()) for indices in partitions.values())
    return partitions, {
        "partition_protocol": "coverage_aware_disjoint_global_test_per_client",
        "assignment_seed": int(seed),
        "class_to_episode": {str(key): int(value) for key, value in class_to_episode.items()},
        "eligible_client_count_by_episode": eligible_by_episode,
        "unsupported_sample_count_by_episode": unsupported_by_episode,
        "unsupported_sample_count": int(sum(unsupported_by_episode.values())),
        "assigned_sample_count": int(assigned_count),
        "coverage_by_client": {
            str(cid): coverage_by_client[int(cid)] for cid in client_ids
        },
    }


@torch.no_grad()
def _evaluate_denice_partitioned_clients(
    ckpt: Dict[str, Any],
    client_ids: list[int],
    test_X: torch.Tensor,
    test_y: torch.Tensor,
    *,
    device: str,
    router_mode: str | None,
    route_mode: str,
    route_topk: int,
    eval_seed: int,
    task_id: int,
    partitions: Dict[int, torch.Tensor] | None = None,
    protocol_debug: Dict[str, Any] | None = None,
    inference_policy: str | None = None,
    class_to_episode: Dict[int, int] | None = None,
) -> Dict[str, Any]:
    """Evaluate one disjoint, reproducible global-test partition per client.

    This is a personalized/distributed test protocol, not a global ensemble:
    every test sample is routed through exactly one saved client model.  Metrics
    are calculated after concatenating predictions from all partitions, so the
    reported F1 remains a proper dataset-level metric rather than an average of
    per-client F1 values.
    """
    if not client_ids:
        raise ValueError("partitioned_local evaluation needs at least one client")

    config = ckpt["config"]
    seed = int(eval_seed) + int(task_id)
    generator = torch.Generator().manual_seed(seed)
    shuffled = torch.randperm(len(test_y), generator=generator)
    if partitions is None:
        partition_pairs = zip(client_ids, torch.tensor_split(shuffled, len(client_ids)))
        protocol_name = "random_disjoint_global_test_per_client"
    else:
        partition_pairs = ((cid, partitions.get(int(cid), torch.empty(0, dtype=torch.long))) for cid in client_ids)
        protocol_name = str((protocol_debug or {}).get("partition_protocol", "coverage_aware"))
    criterion = nn.CrossEntropyLoss()
    all_predictions: list[int] = []
    all_targets: list[int] = []
    partition_sizes: Dict[str, int] = {}
    partition_records: list[Dict[str, Any]] = []
    route_confusion: Dict[str, Dict[str, int]] = {}
    total_loss = 0.0
    route_correct = 0
    route_total = 0
    evaluated_client_count = 0
    batch_size = int(config.get("eval_batch_size", 8192))
    seen_classes = ckpt.get("seen_classes")
    mask_violation_count = 0
    routing_diagnostics: Dict[str, int] = {
        "adapter_active_sample_count": 0,
        "missing_adapter_sample_count": 0,
        "hard_mask_sample_count": 0,
        "topk_mask_sample_count": 0,
        "adaptive_hard_sample_count": 0,
        "adaptive_topk_sample_count": 0,
        "adaptive_nomask_sample_count": 0,
    }

    for client_id, indices in partition_pairs:
        partition_sizes[str(client_id)] = int(indices.numel())
        if indices.numel() == 0:
            continue
        model, context_detector = _make_denice_client_model(
            ckpt, client_id, device, router_mode=router_mode
        )
        evaluated_client_count += 1
        X_partition = test_X[indices]
        y_partition = test_y[indices]
        label2episode = _label_to_episode_map(context_detector)
        partition_predictions: list[int] = []
        partition_targets: list[int] = []
        partition_loss = 0.0
        partition_route_correct = 0
        partition_route_total = 0
        for start in range(0, len(y_partition), max(1, batch_size)):
            X_batch = X_partition[start : start + batch_size].to(device)
            y_batch = y_partition[start : start + batch_size].to(device)
            oracle_episodes = None
            if inference_policy in {"oracle_adapter_nomask", "oracle_hard"}:
                if not class_to_episode:
                    raise ValueError("oracle evaluation needs a class-to-episode map")
                oracle_episodes = np.asarray(
                    [class_to_episode.get(int(label), -1) for label in y_batch.cpu().tolist()],
                    dtype=np.int64,
                )
                if np.any(oracle_episodes < 0):
                    raise ValueError("oracle evaluation encountered an unmapped test class")
                if inference_policy == "oracle_hard":
                    for label, episode in zip(y_batch.cpu().tolist(), oracle_episodes):
                        allowed = context_detector.episode_classes.get(int(episode), [])
                        if int(label) not in {int(cls) for cls in allowed}:
                            mask_violation_count += 1
            logits, episodes = _denice_routed_logits_with_episodes(
                model,
                X_batch,
                context_detector,
                seen_classes,
                device,
                route_mode,
                route_topk,
                inference_policy=inference_policy,
                oracle_episodes=oracle_episodes,
                routing_diagnostics=routing_diagnostics,
            )
            batch_loss = criterion(logits, y_batch).item() * len(y_batch)
            total_loss += batch_loss
            partition_loss += batch_loss
            batch_predictions = logits.argmax(dim=1).detach().cpu().tolist()
            batch_targets = y_batch.detach().cpu().tolist()
            all_predictions.extend(batch_predictions)
            all_targets.extend(batch_targets)
            partition_predictions.extend(batch_predictions)
            partition_targets.extend(batch_targets)
            if episodes is not None and label2episode:
                true_episodes = np.asarray(
                    [label2episode.get(int(label), -1) for label in y_batch.cpu().numpy()],
                    dtype=np.int64,
                )
                known = true_episodes >= 0
                known_count = int(known.sum())
                correct_count = int((episodes[known] == true_episodes[known]).sum())
                route_total += known_count
                route_correct += correct_count
                partition_route_total += known_count
                partition_route_correct += correct_count
                for true_episode, predicted_episode in zip(
                    true_episodes[known], episodes[known]
                ):
                    true_key = str(int(true_episode))
                    predicted_key = str(int(predicted_episode))
                    row = route_confusion.setdefault(true_key, {})
                    row[predicted_key] = int(row.get(predicted_key, 0) + 1)
        partition_targets_np = np.asarray(partition_targets)
        partition_predictions_np = np.asarray(partition_predictions)
        partition_records.append(
            {
                "client_id": int(client_id),
                "sample_count": int(len(partition_targets_np)),
                "unique_target_class_count": int(len(np.unique(partition_targets_np))),
                "loss": partition_loss / max(1, len(partition_targets_np)),
                "accuracy": accuracy_score(partition_targets_np, partition_predictions_np),
                "route_accuracy": (
                    partition_route_correct / partition_route_total
                    if partition_route_total
                    else None
                ),
                "route_coverage": partition_route_total / max(1, len(partition_targets_np)),
            }
        )
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    y_true = np.asarray(all_targets)
    y_pred = np.asarray(all_predictions)
    per_class = {}
    for class_id in sorted(int(c) for c in np.unique(y_true)):
        class_mask = y_true == class_id
        per_class[str(class_id)] = {
            "support": int(class_mask.sum()),
            "accuracy": float((y_pred[class_mask] == y_true[class_mask]).mean()),
        }
    compact_debug = {
        "per_class": per_class,
        "route_confusion": route_confusion,
        "worst_partitions_by_accuracy": sorted(
            partition_records, key=lambda row: (row["accuracy"], -row["sample_count"])
        )[:10],
        "worst_partitions_by_loss": sorted(
            partition_records, key=lambda row: (-row["loss"], -row["sample_count"])
        )[:10],
        "worst_partitions_by_route_accuracy": sorted(
            (row for row in partition_records if row["route_accuracy"] is not None),
            key=lambda row: (row["route_accuracy"], -row["sample_count"]),
        )[:10],
    }
    return {
        "loss": total_loss / max(1, len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "route_accuracy": route_correct / route_total if route_total else 0.0,
        "route_coverage": route_total / len(y_true) if len(y_true) else 0.0,
        "partition_protocol": protocol_name,
        "partition_seed": seed,
        "partition_count": len(client_ids),
        "evaluated_client_count": evaluated_client_count,
        "partition_sizes": partition_sizes,
        "debug": compact_debug,
        "protocol_debug": protocol_debug or {},
        "inference_policy": inference_policy or "routed_default",
        "oracle_mask_violation_count": int(mask_violation_count),
        "routing_diagnostics": routing_diagnostics,
    }


def evaluate_checkpoint(
    checkpoint_path: str,
    device: str | None = None,
    data_dir: str | None = None,
    route_mode: str = "hard",
    route_topk: int = 1,
    router_mode: str | None = None,
    evaluation_mode: str = "local",
    max_samples: int | None = None,
    eval_seed: int = 42,
    inference_policy: str | None = None,
) -> Dict[str, Any]:
    ckpt = _load_checkpoint(checkpoint_path)
    config = dict(ckpt["config"])
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = data_dir or config["data_dir"]
    config["data_dir"] = data_dir
    ckpt["config"] = config
    task_id = int(ckpt["task_id"])
    data_loader = IncrementalDataLoader(data_dir=data_dir)
    test_X, test_y = data_loader.get_test_data(task_id, cumulative=True)
    total_test_samples = int(len(test_y))
    if max_samples is not None and 0 < int(max_samples) < len(test_y):
        generator = torch.Generator().manual_seed(int(eval_seed) + int(task_id))
        indices = torch.randperm(len(test_y), generator=generator)[: int(max_samples)]
        test_X, test_y = test_X[indices], test_y[indices]

    if ckpt.get("algorithm") == "denice" and "client_model_states" in ckpt:
        client_ids = [
            int(cid) for cid in ckpt.get("client_ids", ckpt["client_model_states"].keys())
        ]
        requested_evaluation_mode = str(evaluation_mode or "local").lower()
        evaluation_mode = requested_evaluation_mode
        if requested_evaluation_mode == "representative_global":
            evaluation_mode = "representative"
        if evaluation_mode not in {
            "local", "ensemble", "representative", "partitioned_local", "coverage_aware_local"
        }:
            raise ValueError(
                "evaluation_mode must be local, ensemble, representative, partitioned_local, "
                "representative_global, or coverage_aware_local"
            )
        if evaluation_mode in {"partitioned_local", "coverage_aware_local"}:
            partitions = None
            protocol_debug = None
            if evaluation_mode == "coverage_aware_local":
                coverage_by_client = {
                    int(cid): _client_router_episode_coverage(ckpt, int(cid))
                    for cid in client_ids
                }
                partitions, protocol_debug = _build_coverage_aware_partitions(
                    test_y,
                    client_ids,
                    data_loader.task_classes,
                    coverage_by_client,
                    seed=int(eval_seed) + int(task_id),
                )
            metrics = _evaluate_denice_partitioned_clients(
                ckpt,
                client_ids,
                test_X,
                test_y,
                device=device,
                router_mode=router_mode,
                route_mode=route_mode,
                route_topk=route_topk,
                eval_seed=eval_seed,
                task_id=task_id,
                partitions=partitions,
                protocol_debug=protocol_debug,
                inference_policy=inference_policy,
                class_to_episode={
                    int(class_id): int(episode)
                    for episode, classes in data_loader.task_classes.items()
                    for class_id in classes
                },
            )
            return {
                "checkpoint": str(checkpoint_path),
                "checkpoint_type": ckpt.get("checkpoint_type"),
                "task_id": task_id,
                "round_id": ckpt.get("round_id", ckpt.get("final_round_id")),
                "algorithm": "denice",
                "evaluation_mode": requested_evaluation_mode,
                "client_ids": client_ids,
                "metrics": metrics,
                "route_mode": route_mode,
                "inference_policy": inference_policy or "routed_default",
                "route_topk": int(route_topk),
                "router_mode": router_mode or config.get("denice_router_mode", "chained"),
                "checkpoint_sha256": hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
                "config_sha256": hashlib.sha256(
                    json.dumps(config, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest(),
                "eval_sample_count": int(len(test_y)),
                "eval_total_sample_count": total_test_samples,
                "eval_seed": int(eval_seed),
            }
        if evaluation_mode in {"ensemble", "representative"}:
            pairs = []
            selected_ids = client_ids
            representative_coverage_debug: Dict[str, Any] = {}
            if evaluation_mode == "representative":
                class_to_episode = {
                    int(class_id): int(episode)
                    for episode, classes in data_loader.task_classes.items()
                    for class_id in classes
                }
                required_episodes = {
                    int(class_to_episode[int(label)])
                    for label in test_y.detach().cpu().tolist()
                    if int(label) in class_to_episode
                }
                coverage_by_client = {
                    int(cid): _client_router_episode_coverage(ckpt, int(cid))
                    for cid in client_ids
                }
                eligible_ids = [
                    int(cid)
                    for cid in client_ids
                    if required_episodes.issubset(
                        set(coverage_by_client[int(cid)]["supported_episodes"])
                    )
                ]
                if not eligible_ids:
                    raise ValueError(
                        "representative_global has no client with full router episode coverage"
                    )
                groups = (ckpt.get("cluster") or {}).get("groups", {})
                seen_groups = set()
                selected_ids = []
                for cid in eligible_ids:
                    group = tuple(sorted(set(int(x) for x in _dict_get_int(groups, int(cid), [cid]))))
                    if group in seen_groups:
                        continue
                    seen_groups.add(group)
                    eligible_group = sorted(int(x) for x in group if int(x) in eligible_ids)
                    selected_ids.append(eligible_group[0])
                representative_coverage_debug = {
                    "required_episodes": sorted(required_episodes),
                    "eligible_client_count": len(eligible_ids),
                    "eligible_client_ids": eligible_ids,
                    "coverage_by_representative": {
                        str(cid): coverage_by_client[int(cid)] for cid in selected_ids
                    },
                }
            for cid in selected_ids:
                pairs.append(_make_denice_client_model(ckpt, cid, device, router_mode=router_mode))
            metrics = evaluate_denice_ensemble(
                pairs,
                {"X_test": test_X, "y_test": test_y},
                device=device,
                seen_classes=ckpt.get("seen_classes"),
                batch_size=int(config.get("eval_batch_size", 8192)),
                route_mode=route_mode,
                route_topk=route_topk,
                inference_policy=inference_policy,
                class_to_episode={
                    int(class_id): int(episode)
                    for episode, classes in data_loader.task_classes.items()
                    for class_id in classes
                },
            )
            return {
                "checkpoint": str(checkpoint_path),
                "checkpoint_type": ckpt.get("checkpoint_type"),
                "task_id": task_id,
                "round_id": ckpt.get("round_id", ckpt.get("final_round_id")),
                "algorithm": "denice",
                "evaluation_mode": requested_evaluation_mode,
                "representative_client_ids": selected_ids,
                "representative_coverage_debug": representative_coverage_debug,
                "metrics": metrics,
                "route_mode": route_mode,
                "inference_policy": inference_policy or "routed_default",
                "route_topk": int(route_topk),
                "router_mode": router_mode or config.get("denice_router_mode", "chained"),
                "checkpoint_sha256": hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
                "config_sha256": hashlib.sha256(
                    json.dumps(config, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest(),
                "eval_sample_count": int(len(test_y)),
                "eval_total_sample_count": total_test_samples,
                "eval_seed": int(eval_seed),
            }

        per_client = []
        for cid in client_ids:
            model, context_detector = _make_denice_client_model(
                ckpt, cid, device, router_mode=router_mode
            )
            metrics = evaluate_denice_model(
                model,
                {"X_test": test_X, "y_test": test_y},
                device,
                context_detector=context_detector,
                seen_classes=ckpt.get("seen_classes"),
                batch_size=int(config.get("eval_batch_size", 8192)),
                route_mode=route_mode,
                route_topk=route_topk,
            )
            per_client.append({"client_id": cid, **metrics})
            del model
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()

        metric_keys = [
            "loss",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "f1_weighted",
        ]
        mean_metrics = {
            key: sum(float(row[key]) for row in per_client) / max(1, len(per_client))
            for key in metric_keys
        }
        return {
            "checkpoint": str(checkpoint_path),
            "checkpoint_type": ckpt.get("checkpoint_type"),
            "task_id": task_id,
            "round_id": ckpt.get("round_id", ckpt.get("final_round_id")),
            "algorithm": ckpt.get("algorithm", config.get("algorithm")),
            "mode": ckpt.get("mode", config.get("mode")),
            "eval_client_count": len(per_client),
            "metrics": mean_metrics,
            "per_client_metrics": per_client,
            "route_mode": route_mode,
            "route_topk": int(route_topk),
            "router_mode": router_mode or config.get("denice_router_mode", "chained"),
            "checkpoint_sha256": hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
            "config_sha256": hashlib.sha256(
                json.dumps(config, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest(),
            "eval_sample_count": int(len(test_y)),
            "eval_total_sample_count": total_test_samples,
            "eval_seed": int(eval_seed),
        }

    model, context_detector = _make_model(ckpt, device)
    metrics = _evaluate_model(
        model,
        {"X_test": test_X, "y_test": test_y},
        device,
        context_detector=context_detector,
        seen_classes=ckpt.get("seen_classes"),
    )
    return {
        "checkpoint": str(checkpoint_path),
        "task_id": task_id,
        "round_id": ckpt.get("round_id", ckpt.get("final_round_id")),
        "algorithm": ckpt.get("algorithm", config.get("algorithm")),
        "mode": ckpt.get("mode", config.get("mode")),
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--route-mode", default="hard", choices=["hard", "topk", "nomask", "adaptive"]
    )
    parser.add_argument("--route-topk", type=int, default=1)
    parser.add_argument("--router-mode", default=None, choices=["chained", "multiclass"])
    parser.add_argument(
        "--evaluation-mode",
        default="local",
        choices=[
            "local", "ensemble", "representative", "representative_global", "partitioned_local", "coverage_aware_local"
        ],
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument(
        "--inference-policy",
        default=None,
        choices=[
            "pred_hard",
            "pred_adapter_nomask",
            "oracle_adapter_nomask",
            "oracle_hard",
            "backbone_nomask",
        ],
        help="Component-isolation policy; requires partitioned or coverage-aware local evaluation.",
    )
    parser.add_argument(
        "--router-audit",
        action="store_true",
        help="Audit serialized DeNICE router memory and exit; does not evaluate test data.",
    )
    parser.add_argument(
        "--router-current-feature-audit",
        action="store_true",
        help="Audit router accuracy on final-model features using a balanced test subset.",
    )
    parser.add_argument("--router-audit-max-clients", type=int, default=10)
    parser.add_argument("--router-audit-samples-per-episode", type=int, default=256)
    parser.add_argument(
        "--route-modes",
        default=None,
        help="Comma-separated ablation modes; emits one result per mode on identical checkpoint/data.",
    )
    args = parser.parse_args()

    if args.router_audit:
        audit = audit_denice_router_states(_load_checkpoint(args.checkpoint), seed=args.eval_seed)
        print(json.dumps(audit, indent=2))
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        return

    if args.router_current_feature_audit:
        checkpoint = _load_checkpoint(args.checkpoint)
        config = dict(checkpoint["config"])
        actual_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        data_loader = IncrementalDataLoader(args.data_dir or config["data_dir"])
        test_X, test_y = data_loader.get_test_data(int(checkpoint["task_id"]), cumulative=True)
        audit = audit_denice_router_current_features(
            checkpoint,
            test_X,
            test_y,
            data_loader.task_classes,
            device=actual_device,
            seed=args.eval_seed,
            max_clients=args.router_audit_max_clients,
            samples_per_episode=args.router_audit_samples_per_episode,
        )
        print(json.dumps(audit, indent=2))
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        return

    modes = [args.route_mode]
    if args.route_modes:
        modes = [m.strip() for m in args.route_modes.split(",") if m.strip()]
    if any(m not in {"hard", "topk", "nomask", "adaptive"} for m in modes):
        parser.error("--route-modes accepts only hard,topk,nomask,adaptive")
    results = {
        mode: evaluate_checkpoint(
            args.checkpoint,
            device=args.device,
            data_dir=args.data_dir,
            route_mode=mode,
            route_topk=args.route_topk,
            router_mode=args.router_mode,
            evaluation_mode=args.evaluation_mode,
            max_samples=args.max_samples,
            eval_seed=args.eval_seed,
            inference_policy=args.inference_policy,
        )
        for mode in modes
    }
    result = results[modes[0]] if len(modes) == 1 else {"ablations": results}
    print(json.dumps(result, indent=2))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
