"""
NICE Context Capsule for DeNICE (plan section 2.3, Đề xuất section 5).

Each client builds a capsule and sends it to neighbors (no raw data)::

    Psi_i^t = {
        activation_prototypes,      # P_i^t : per-layer binary activation prototype
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
    context_detector_summary: Dict[int, List[int]]
    architecture_version: int = 1
    adapter_registry: Dict[str, Dict] = field(default_factory=dict)
    update_summary: Optional[np.ndarray] = None

    def proto_vector(self) -> np.ndarray:
        """Flatten per-layer prototypes into a single vector (for f_i features)."""
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
    model, data: torch.Tensor, thresholds: Optional[Dict[str, float]]
) -> Dict[str, np.ndarray]:
    """Mean binary activation per layer (P_i^t)."""
    acts = model.get_context_activations_per_sample(data)
    proto: Dict[str, np.ndarray] = {}
    for name in CAPSULE_LAYERS:
        act = acts[name].detach().cpu().numpy()
        if thresholds is not None and name in thresholds:
            binary = (act > thresholds[name]).astype(np.float32)
        else:
            binary = (act > 0).astype(np.float32)
        proto[name] = binary.mean(axis=0)
    return proto


def _importance(model, data: torch.Tensor) -> Dict[str, np.ndarray]:
    """Activation-based neuron importance per layer (A_i^t)."""
    acts = model.get_context_activations_per_sample(data)
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
) -> ContextCapsule:
    """Build a :class:`ContextCapsule` from a trained DeNICE/NICE model.

    ``data`` is a *train* subset used only to read activations; raw data is never
    placed in the capsule.
    """
    model.eval()
    proto = _binary_prototype(model, data, thresholds)
    importance = _importance(model, data)
    age_mask = _age_mask(model)
    capacity = _capacity_histogram(model)

    detector_summary: Dict[int, List[int]] = {}
    if context_detector is not None:
        detector_summary = {
            int(ep): list(cls)
            for ep, cls in getattr(context_detector, "episode_classes", {}).items()
        }

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
    )
