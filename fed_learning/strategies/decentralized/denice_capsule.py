"""
NICE Context Capsule for DeNICE (plan section 2.3, Đề xuất section 5).

Each client builds a capsule and sends it to neighbors (no raw data)::

    Psi_i^t = {
        activation_prototypes,      # P_i^t : per-layer binary activation prototype
        class_activation_prototypes,# P_i,c^t : per-class binary activation prototype
        age_mask,                   # M_i^t : selected/age mask per layer
        neuron_importance,          # A_i^t : activation-based importance per layer
        reliability,                # R_i^t : local validation reliability
        context_detector_summary,   # Q_i^t : episode -> classes summary
        capacity_histogram,         # H_i^t : young/learner/mature ratio per layer
        label_histogram,            # Y_i^t : class distribution
        sample_count,               # n_i^t
        architecture_version,
        adapter_registry            # adapter metadata for adapter aggregation
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Dict, List, Optional

import numpy as np
import torch

CAPSULE_LAYERS: List[str] = ["conv1", "conv2", "conv3", "gru"]


@dataclass
class ContextCapsule:
    """Serializable NICE Context Capsule (no raw data)."""

    client_id: int
    task_id: int
    round_id: int
    activation_prototypes: Dict[str, np.ndarray]
    age_mask: Dict[str, np.ndarray]
    neuron_importance: Dict[str, np.ndarray]
    capacity_histogram: Dict[str, Dict[str, float]]
    label_histogram: Dict[int, float]
    label_set: List[int]
    sample_count: int
    reliability: float
    context_detector_summary: Dict[str, Any]
    architecture_version: int = 1
    adapter_registry: Dict[str, Dict] = field(default_factory=dict)
    update_summary: Optional[np.ndarray] = None
    class_activation_prototypes: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)

    def proto_vector(self) -> np.ndarray:
        """Flatten class-balanced per-layer prototypes into one vector.

        The proposal defines ``P_i,c^t`` per class. For a fixed-width clustering
        feature we average those class prototypes per layer. Older checkpoints
        without class prototypes fall back to the task/client mean prototype.
        """
        if self.class_activation_prototypes:
            parts = []
            for name in CAPSULE_LAYERS:
                class_parts = [
                    np.asarray(layer_proto[name]).ravel()
                    for _, layer_proto in sorted(self.class_activation_prototypes.items())
                    if name in layer_proto
                ]
                if class_parts:
                    parts.append(np.stack(class_parts, axis=0).mean(axis=0))
            return np.concatenate(parts) if parts else np.zeros(0)
        parts = [
            np.asarray(self.activation_prototypes[name]).ravel()
            for name in CAPSULE_LAYERS
            if name in self.activation_prototypes
        ]
        return np.concatenate(parts) if parts else np.zeros(0)

    def age_mask_vector(self) -> np.ndarray:
        parts = [
            np.asarray(self.age_mask[name]).ravel().astype(np.float32)
            for name in CAPSULE_LAYERS
            if name in self.age_mask
        ]
        return np.concatenate(parts) if parts else np.zeros(0)

    def importance_vector(self) -> np.ndarray:
        parts = [
            np.asarray(self.neuron_importance[name]).ravel()
            for name in CAPSULE_LAYERS
            if name in self.neuron_importance
        ]
        return np.concatenate(parts) if parts else np.zeros(0)

    def capacity_vector(self) -> np.ndarray:
        vals = []
        for name in CAPSULE_LAYERS:
            hist = self.capacity_histogram.get(name)
            if hist is None:
                continue
            vals.extend([hist.get("young", 0.0), hist.get("learner", 0.0), hist.get("mature", 0.0)])
        return np.asarray(vals, dtype=np.float32)


def _binary_prototype(
    acts: Dict[str, torch.Tensor], thresholds: Optional[Dict[str, float]]
) -> Dict[str, np.ndarray]:
    """Mean binary activation per layer (P_i^t)."""
    proto: Dict[str, np.ndarray] = {}
    for name in CAPSULE_LAYERS:
        act = acts[name].detach().cpu().numpy()
        if thresholds is not None and name in thresholds:
            binary = (act > thresholds[name]).astype(np.float32)
        else:
            binary = (act > 0).astype(np.float32)
        proto[name] = binary.mean(axis=0)
    return proto

def _class_binary_prototypes(
    acts: Dict[str, torch.Tensor],
    labels: Optional[torch.Tensor],
    thresholds: Optional[Dict[str, float]],
) -> Dict[int, Dict[str, np.ndarray]]:
    """Per-class mean binary activation prototype ``P_i,c^t``."""
    if labels is None or len(labels) == 0:
        return {}
    first_layer = next(iter(acts.values()), None)
    if first_layer is None or len(labels) != len(first_layer):
        return {}
    labels_np = labels.detach().cpu().numpy().astype(int)
    binary_by_layer: Dict[str, np.ndarray] = {}
    for name in CAPSULE_LAYERS:
        act = acts[name].detach().cpu().numpy()
        if thresholds is not None and name in thresholds:
            binary_by_layer[name] = (act > thresholds[name]).astype(np.float32)
        else:
            binary_by_layer[name] = (act > 0).astype(np.float32)

    class_proto: Dict[int, Dict[str, np.ndarray]] = {}
    for cls_id in sorted(set(int(x) for x in labels_np.tolist())):
        mask = labels_np == int(cls_id)
        if not mask.any():
            continue
        class_proto[int(cls_id)] = {
            name: binary_by_layer[name][mask].mean(axis=0)
            for name in CAPSULE_LAYERS
        }
    return class_proto


def _importance(acts: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    """Activation-based neuron importance per layer (A_i^t)."""
    return {
        name: acts[name].detach().cpu().numpy().mean(axis=0)
        for name in CAPSULE_LAYERS
    }


def _capacity_histogram(model) -> Dict[str, Dict[str, float]]:
    hist: Dict[str, Dict[str, float]] = {}
    for name, ranks in getattr(model, "unit_ranks", {}).items():
        ranks = np.asarray(ranks)
        total = max(1, int(ranks.size))
        hist[name] = {
            "young": float((ranks == 0).sum()) / total,
            "learner": float((ranks == 1).sum()) / total,
            "mature": float((ranks >= 2).sum()) / total,
        }
    return hist


def _age_mask(model) -> Dict[str, np.ndarray]:
    """Selected/used-neuron mask (age >= 1) per capsule layer (M_i^t)."""
    return {
        name: (np.asarray(model.unit_ranks[name]) >= 1).astype(np.float32)
        for name in CAPSULE_LAYERS
        if name in getattr(model, "unit_ranks", {})
    }

def _context_detector_summary(context_detector: Any) -> Dict[str, Any]:
    """Compact ``Q_i^t`` summary for routing/debug without raw samples."""
    if context_detector is None:
        return {}
    episode_classes = {
        int(ep): [int(c) for c in cls]
        for ep, cls in getattr(context_detector, "episode_classes", {}).items()
    }
    activation_memory = getattr(context_detector, "activation_memory", {}) or {}
    memory_counts = {
        int(ep): int(len(mat))
        for ep, mat in activation_memory.items()
    }
    feature_dims = {
        int(ep): int(np.asarray(mat).shape[1]) if np.asarray(mat).ndim == 2 else 0
        for ep, mat in activation_memory.items()
    }
    activation_stats = {}
    for ep, mat in activation_memory.items():
        arr = np.asarray(mat)
        if arr.ndim != 2 or arr.size == 0:
            activation_stats[int(ep)] = {
                "mean_density": 0.0,
                "active_feature_ratio": 0.0,
            }
            continue
        activation_stats[int(ep)] = {
            "mean_density": float(arr.mean()),
            "active_feature_ratio": float((arr.mean(axis=0) > 0).mean()),
        }

    def _array_hash(value: Any) -> str:
        arr = np.asarray(value, dtype=np.float32)
        if arr.size == 0:
            return ""
        return hashlib.sha256(arr.tobytes()).hexdigest()[:16]

    learner_summary = []
    for idx, clf in enumerate(getattr(context_detector, "context_learners", []) or []):
        if clf is None:
            learner_summary.append({"episode": int(idx), "trained": False})
            continue
        coef = getattr(clf, "coef_", None)
        intercept = getattr(clf, "intercept_", None)
        coef_arr = np.asarray(coef, dtype=np.float32) if coef is not None else np.asarray([])
        intercept_arr = (
            np.asarray(intercept, dtype=np.float32) if intercept is not None else np.asarray([])
        )
        learner_summary.append(
            {
                "episode": int(idx),
                "trained": True,
                "coef_shape": list(coef_arr.shape),
                "coef_l2": float(np.linalg.norm(coef_arr)) if coef_arr.size else 0.0,
                "coef_mean": float(coef_arr.mean()) if coef_arr.size else 0.0,
                "coef_std": float(coef_arr.std()) if coef_arr.size else 0.0,
                "coef_hash": _array_hash(coef_arr),
                "intercept_l2": float(np.linalg.norm(intercept_arr)) if intercept_arr.size else 0.0,
                "intercept_mean": float(intercept_arr.mean()) if intercept_arr.size else 0.0,
                "intercept_hash": _array_hash(intercept_arr),
            }
        )

    thresholds = getattr(context_detector, "binarize_thresholds", None) or {}

    return {
        "episode_classes": episode_classes,
        "memory_counts": memory_counts,
        "feature_dims": feature_dims,
        "activation_stats": activation_stats,
        "threshold_layers": sorted(str(k) for k in thresholds.keys()),
        "threshold_values": {str(k): float(v) for k, v in thresholds.items()},
        "num_learners": int(len(getattr(context_detector, "context_learners", []) or [])),
        "learners": learner_summary,
    }


def build_context_capsule(
    model,
    data: torch.Tensor,
    *,
    client_id: int,
    task_id: int,
    round_id: int,
    label_histogram: Dict[int, float],
    label_set: List[int],
    sample_count: int,
    reliability: float,
    thresholds: Optional[Dict[str, float]] = None,
    context_detector: Any = None,
    update_summary: Optional[np.ndarray] = None,
    labels: Optional[torch.Tensor] = None,
) -> ContextCapsule:
    """Build a :class:`ContextCapsule` from a trained DeNICE/NICE model.

    ``data`` is a *train* subset used only to read activations; raw data is never
    placed in the capsule.
    """
    model.eval()
    acts = model.get_context_activations_per_sample(data)
    proto = _binary_prototype(acts, thresholds)
    class_proto = _class_binary_prototypes(acts, labels, thresholds)
    importance = _importance(acts)
    age_mask = _age_mask(model)
    capacity = _capacity_histogram(model)

    detector_summary = _context_detector_summary(context_detector)

    adapter_registry: Dict[str, Dict] = {}
    if hasattr(model, "get_adapter_registry_state"):
        adapter_registry = model.get_adapter_registry_state()

    return ContextCapsule(
        client_id=int(client_id),
        task_id=int(task_id),
        round_id=int(round_id),
        activation_prototypes=proto,
        age_mask=age_mask,
        neuron_importance=importance,
        capacity_histogram=capacity,
        label_histogram={int(k): float(v) for k, v in label_histogram.items()},
        label_set=[int(c) for c in label_set],
        sample_count=int(sample_count),
        reliability=float(reliability),
        context_detector_summary=detector_summary,
        architecture_version=int(getattr(model, "architecture_version", 1)),
        adapter_registry=adapter_registry,
        update_summary=update_summary,
        class_activation_prototypes=class_proto,
    )
