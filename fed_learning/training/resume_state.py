"""Helpers for saving/loading continuation state across task phases."""

from __future__ import annotations

import os
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn


_OBJECT_MARKER = "__resume_object__"

_CLIENT_EXCLUDED_ATTRS = {
    "X_train",
    "y_train",
    "num_samples",
    "model",
    "device",
    "use_amp",
    "old_model",
    "_original_X",
    "_original_y",
}

_TRAINER_EXCLUDED_ATTRS = {
    "_cached_old_model",
    "_cached_device",
}

_SERVER_EXCLUDED_ATTRS = {
    "clients",
    "global_model",
    "test_data",
    "trainer",
    "aggregator",
    "config",
    "history",
    "primary_device",
    "num_gpus",
    "use_cpu",
}

_AGGREGATOR_EXCLUDED_ATTRS = set()


def _clone_resume_value(value: Any) -> Any:
    """Convert a value into a CPU-safe serializable structure."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, OrderedDict):
        return OrderedDict((k, _clone_resume_value(v)) for k, v in value.items())
    if isinstance(value, dict):
        return {k: _clone_resume_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_resume_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_resume_value(v) for v in value)
    if isinstance(value, set):
        return {_clone_resume_value(v) for v in value}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, nn.Module):
        return OrderedDict(
            (k, v.detach().cpu().clone()) for k, v in value.state_dict().items()
        )
    if hasattr(value, "__dict__"):
        attrs = {
            name: _clone_resume_value(attr_value)
            for name, attr_value in value.__dict__.items()
            if not callable(attr_value) and not name.startswith("__")
        }
        return {
            _OBJECT_MARKER: value.__class__.__name__,
            "attrs": attrs,
        }
    return value


def _restore_resume_value(existing: Any, saved: Any) -> Any:
    """Restore a previously serialized value."""
    if isinstance(saved, dict) and saved.get(_OBJECT_MARKER):
        if existing is not None and hasattr(existing, "__dict__"):
            _restore_object_state(existing, {"attrs": saved["attrs"]})
            return existing
        return {
            name: _restore_resume_value(None, value)
            for name, value in saved["attrs"].items()
        }
    if isinstance(saved, OrderedDict):
        return OrderedDict((k, _restore_resume_value(None, v)) for k, v in saved.items())
    if isinstance(saved, dict):
        return {k: _restore_resume_value(None, v) for k, v in saved.items()}
    if isinstance(saved, list):
        return [_restore_resume_value(None, v) for v in saved]
    if isinstance(saved, tuple):
        return tuple(_restore_resume_value(None, v) for v in saved)
    if isinstance(saved, set):
        return {_restore_resume_value(None, v) for v in saved}
    if isinstance(saved, torch.Tensor):
        return saved.detach().cpu().clone()
    if isinstance(saved, np.ndarray):
        return saved.copy()
    return saved


def _snapshot_object_state(obj: Any, excluded_attrs: Iterable[str]) -> Dict[str, Any]:
    """Snapshot object attrs unless the object exposes a custom resume hook."""
    if obj is None:
        return {"class_name": None, "attrs": {}}

    if hasattr(obj, "get_resume_state"):
        state = obj.get_resume_state()
        attrs = _clone_resume_value(state)
    else:
        attrs = {
            name: _clone_resume_value(value)
            for name, value in obj.__dict__.items()
            if name not in set(excluded_attrs)
            and not callable(value)
            and not name.startswith("__")
        }

    return {
        "class_name": obj.__class__.__name__,
        "attrs": attrs,
    }


def _restore_object_state(obj: Any, state: Optional[Dict[str, Any]]) -> None:
    """Restore attrs into an already-created object."""
    if obj is None or not state:
        return

    attrs = state.get("attrs", {})
    if hasattr(obj, "load_resume_state"):
        obj.load_resume_state(_restore_resume_value(None, attrs))
        return

    restored_attrs = _restore_resume_value(None, attrs)
    for name, value in restored_attrs.items():
        current_value = getattr(obj, name, None)
        if (
            isinstance(attrs.get(name), dict)
            and attrs[name].get(_OBJECT_MARKER)
            and current_value is not None
            and hasattr(current_value, "__dict__")
        ):
            _restore_object_state(current_value, {"attrs": attrs[name]["attrs"]})
            continue
        setattr(obj, name, value)


def snapshot_client_state(client: Any) -> Dict[str, Any]:
    return _snapshot_object_state(client, _CLIENT_EXCLUDED_ATTRS)


def snapshot_trainer_state(trainer: Any) -> Dict[str, Any]:
    return _snapshot_object_state(trainer, _TRAINER_EXCLUDED_ATTRS)


def snapshot_server_state(server: Any) -> Dict[str, Any]:
    return _snapshot_object_state(server, _SERVER_EXCLUDED_ATTRS)


def snapshot_aggregator_state(aggregator: Any) -> Dict[str, Any]:
    return _snapshot_object_state(aggregator, _AGGREGATOR_EXCLUDED_ATTRS)


def restore_client_state(client: Any, state: Optional[Dict[str, Any]]) -> None:
    _restore_object_state(client, state)


def restore_trainer_state(trainer: Any, state: Optional[Dict[str, Any]]) -> None:
    _restore_object_state(trainer, state)


def restore_server_state(server: Any, state: Optional[Dict[str, Any]]) -> None:
    _restore_object_state(server, state)


def restore_aggregator_state(aggregator: Any, state: Optional[Dict[str, Any]]) -> None:
    _restore_object_state(aggregator, state)


def build_continuation_state(
    *,
    mode: str,
    algorithm: str,
    task_id: int,
    config: Dict[str, Any],
    output_dir: str,
    model_state_dict,
    global_neuron_ages: Optional[Dict[str, Any]] = None,
    trainer: Any = None,
    server: Any = None,
    aggregator: Any = None,
    persistent_clients: Optional[Dict[int, Any]] = None,
    all_history: Optional[Dict[str, Any]] = None,
    best_acc_per_task: Optional[Dict[int, float]] = None,
    seen_classes: Optional[Iterable[int]] = None,
) -> Dict[str, Any]:
    """Create a continuation-state payload for split training."""
    return {
        "meta": {
            "saved_at": datetime.now().isoformat(),
            "mode": mode,
            "algorithm": algorithm,
            "completed_task": task_id,
            "resume_from_task": task_id + 1,
            "output_dir": output_dir,
        },
        "config": _clone_resume_value(config),
        "model_state_dict": _clone_resume_value(model_state_dict),
        "global_neuron_ages": _clone_resume_value(global_neuron_ages),
        "all_history": _clone_resume_value(all_history or {}),
        "best_acc_per_task": _clone_resume_value(best_acc_per_task or {}),
        "seen_classes": list(seen_classes or []),
        "trainer_state": snapshot_trainer_state(trainer) if trainer is not None else None,
        "server_state": snapshot_server_state(server) if server is not None else None,
        "aggregator_state": (
            snapshot_aggregator_state(aggregator) if aggregator is not None else None
        ),
        "persistent_clients_state": {
            int(cid): snapshot_client_state(client)
            for cid, client in (persistent_clients or {}).items()
        },
    }


def save_continuation_state(
    output_dir: str, task_id: int, state: Dict[str, Any]
) -> str:
    """Persist continuation state to disk."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"continuation_state_task_{task_id}.pt")
    torch.save(state, path)
    return path


def load_continuation_state(path: str) -> Dict[str, Any]:
    """Load a continuation state file."""
    return torch.load(path, map_location="cpu")
