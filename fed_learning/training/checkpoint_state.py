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
        "activation_memory": _clone_value(
            getattr(context_detector, "activation_memory", {})
        ),
        "context_masks": _clone_value(getattr(context_detector, "context_masks", {})),
        "binarize_thresholds": _clone_value(
            getattr(context_detector, "binarize_thresholds", None)
        ),
        "context_learners": list(getattr(context_detector, "context_learners", [])),
        "episode_classes": _clone_value(
            getattr(context_detector, "episode_classes", {})
        ),
    }


def restore_context_detector(context_detector: Any, state: Optional[Dict[str, Any]]) -> None:
    if context_detector is None or not state:
        return
    context_detector.memo_per_class = int(state.get("memo_per_class", 50))
    context_detector.activation_memory = _clone_value(
        state.get("activation_memory", {})
    )
    context_detector.context_masks = _clone_value(state.get("context_masks", {}))
    context_detector.binarize_thresholds = _clone_value(
        state.get("binarize_thresholds")
    )
    context_detector.context_learners = list(state.get("context_learners", []))
    context_detector.episode_classes = _clone_value(state.get("episode_classes", {}))


def snapshot_nice_state(model: Any, context_detector: Any = None) -> Dict[str, Any]:
    state: Dict[str, Any] = {}
    if model is not None and hasattr(model, "get_neuron_ages_state"):
        state["neuron_ages"] = _clone_value(model.get_neuron_ages_state())
    if model is not None and hasattr(model, "freeze_masks"):
        state["freeze_masks"] = _clone_value(getattr(model, "freeze_masks", {}))
    state["context_detector"] = snapshot_context_detector(context_detector)
    return state


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
    elif algo in ("der", "rne"):
        if task_classes_history is None and server is not None:
            task_classes_history = getattr(server, "_task_classes_history", {})
        state[algo] = snapshot_der_state(
            source_model,
            task_classes_history=task_classes_history,
            config=config,
        )
    return state
