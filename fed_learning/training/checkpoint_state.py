"""Checkpoint state helpers for standalone evaluation."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Optional

import numpy as np
import torch


CHECKPOINT_SCHEMA_VERSION = 2


def _clone_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, OrderedDict):
        return OrderedDict((k, _clone_value(v)) for k, v in value.items())
    if isinstance(value, dict):
        return {k: _clone_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(v) for v in value)
    if isinstance(value, set):
        return {_clone_value(v) for v in value}
    return value


def snapshot_context_detector(context_detector: Any) -> Optional[Dict[str, Any]]:
    if context_detector is None:
        return None
    return {
        "memo_per_class": getattr(context_detector, "memo_per_class", 50),
        "router_reference_per_class": getattr(
            context_detector,
            "router_reference_per_class",
            getattr(context_detector, "memo_per_class", 50),
        ),
        "router_mode": getattr(context_detector, "router_mode", "chained"),
        "calibration_provenance": getattr(
            context_detector, "calibration_provenance", None
        ),
        "activation_memory": _clone_value(
            getattr(context_detector, "activation_memory", {})
        ),
        "reference_input_memory": _clone_value(
            getattr(context_detector, "reference_input_memory", {})
        ),
        "context_masks": _clone_value(getattr(context_detector, "context_masks", {})),
        "binarize_thresholds": _clone_value(
            getattr(context_detector, "binarize_thresholds", None)
        ),
        "context_learners": list(getattr(context_detector, "context_learners", [])),
        "multiclass_router": getattr(context_detector, "multiclass_router", None),
        "multiclass_episodes": _clone_value(
            getattr(context_detector, "multiclass_episodes", [])
        ),
        "episode_classes": _clone_value(
            getattr(context_detector, "episode_classes", {})
        ),
        "router_state_fresh": bool(
            getattr(context_detector, "router_state_fresh", False)
        ),
        "router_last_refresh_task": getattr(
            context_detector, "router_last_refresh_task", None
        ),
        "router_last_refresh_round": getattr(
            context_detector, "router_last_refresh_round", None
        ),
        "router_stale_reason": getattr(
            context_detector, "router_stale_reason", None
        ),
    }


def restore_context_detector(context_detector: Any, state: Optional[Dict[str, Any]]) -> None:
    if context_detector is None or not state:
        return
    context_detector.memo_per_class = int(state.get("memo_per_class", 50))
    context_detector.router_reference_per_class = int(
        state.get("router_reference_per_class", context_detector.memo_per_class)
    )
    context_detector.router_mode = str(state.get("router_mode", "chained")).lower()
    context_detector.calibration_provenance = state.get("calibration_provenance")
    context_detector.activation_memory = _clone_value(
        state.get("activation_memory", {})
    )
    context_detector.reference_input_memory = _clone_value(
        state.get("reference_input_memory", {})
    )
    context_detector.context_masks = _clone_value(state.get("context_masks", {}))
    context_detector.binarize_thresholds = _clone_value(
        state.get("binarize_thresholds")
    )
    context_detector.context_learners = list(state.get("context_learners", []))
    context_detector.multiclass_router = state.get("multiclass_router")
    context_detector.multiclass_episodes = [
        int(ep) for ep in state.get("multiclass_episodes", [])
    ]
    context_detector.episode_classes = _clone_value(state.get("episode_classes", {}))
    context_detector.router_state_fresh = bool(state.get("router_state_fresh", False))
    context_detector.router_last_refresh_task = state.get("router_last_refresh_task")
    context_detector.router_last_refresh_round = state.get("router_last_refresh_round")
    context_detector.router_stale_reason = state.get(
        "router_stale_reason",
        None if context_detector.router_state_fresh else "checkpoint_missing_freshness_metadata",
    )


def snapshot_nice_state(model: Any, context_detector: Any = None) -> Dict[str, Any]:
    state: Dict[str, Any] = {}
    if model is not None and hasattr(model, "get_neuron_ages_state"):
        state["neuron_ages"] = _clone_value(model.get_neuron_ages_state())
    if model is not None and hasattr(model, "freeze_masks"):
        state["freeze_masks"] = _clone_value(getattr(model, "freeze_masks", {}))
    state["context_detector"] = snapshot_context_detector(context_detector)
    return state


def snapshot_denice_state(model: Any, context_detector: Any = None) -> Dict[str, Any]:
    """Snapshot NICE state plus the DeNICE adapter registry metadata."""
    state = snapshot_nice_state(model, context_detector)
    if model is not None and hasattr(model, "get_adapter_registry_state"):
        state["adapter_registry"] = _clone_value(model.get_adapter_registry_state())
        state["architecture_version"] = int(getattr(model, "architecture_version", 1))
    if model is not None and hasattr(model, "get_recycling_state"):
        state["recycling_registry"] = _clone_value(model.get_recycling_state())
    return state


def restore_denice_state(
    model: Any, context_detector: Any, state: Optional[Dict[str, Any]]
) -> None:
    """Restore non-``state_dict`` DeNICE state after adapter reconstruction.

    Callers must create adapters from ``adapter_registry`` before loading model
    tensor weights.  NICE ranks/masks are Python/NumPy state rather than model
    buffers, so restoring a tensor checkpoint alone is insufficient.
    """
    if model is None or not state:
        return
    adapter_registry = state.get("adapter_registry") or {}
    for meta in adapter_registry.values():
        if not hasattr(model, "add_adapter"):
            break
        layer_name = meta.get("layer_name")
        context_id = meta.get("context_id")
        if layer_name is not None and context_id is not None:
            model.add_adapter(
                int(context_id), str(layer_name), rank=meta.get("rank"), set_active=False
            )
    neuron_ages = state.get("neuron_ages")
    if neuron_ages and hasattr(model, "set_neuron_ages_state"):
        model.set_neuron_ages_state(_clone_value(neuron_ages))
    if "freeze_masks" in state and hasattr(model, "freeze_masks"):
        model.freeze_masks = _clone_value(state.get("freeze_masks") or {})
    recycling = state.get("recycling_registry")
    if recycling and hasattr(model, "set_recycling_state"):
        model.set_recycling_state(_clone_value(recycling))
    restore_context_detector(context_detector, state.get("context_detector"))


def snapshot_der_state(
    model: Any,
    task_classes_history: Optional[Dict[int, list]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    current_task = getattr(model, "current_task", -1)
    if current_task is None:
        current_task = -1
    return {
        "task_classes_history": _clone_value(task_classes_history or {}),
        "num_extractors": int(getattr(model, "num_extractors", 0) or 0),
        "current_task": int(current_task),
        "s_max": float(getattr(model, "_s_max", (config or {}).get("s_max", 15.0))),
    }


def build_algorithm_state(
    algorithm: str,
    *,
    model: Any = None,
    server: Any = None,
    context_detector: Any = None,
    task_classes_history: Optional[Dict[int, list]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    algo = (algorithm or "").lower()
    source_model = model if model is not None else getattr(server, "global_model", None)
    if context_detector is None and server is not None:
        context_detector = getattr(server, "context_detector", None)

    state: Dict[str, Any] = {}
    if algo == "nice":
        state["nice"] = snapshot_nice_state(source_model, context_detector)
    elif algo == "denice":
        state["denice"] = snapshot_denice_state(source_model, context_detector)
    elif algo in ("der", "rne", "rne_compress"):
        if task_classes_history is None and server is not None:
            task_classes_history = getattr(server, "_task_classes_history", {})
        state[algo] = snapshot_der_state(
            source_model,
            task_classes_history=task_classes_history,
            config=config,
        )
    return state
