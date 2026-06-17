"""Compact DeNICE round checkpoints with reconstructable fp16 deltas."""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch
import numpy as np

from fed_learning.training.checkpoint_state import CHECKPOINT_SCHEMA_VERSION


FLOAT_STORAGE_DTYPE = torch.float16


def _compact_tensor(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach().cpu().clone()
    if torch.is_floating_point(value):
        return value.to(FLOAT_STORAGE_DTYPE)
    return value


def compact_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> "OrderedDict[str, torch.Tensor]":
    return OrderedDict((k, _compact_tensor(v)) for k, v in state_dict.items())


def compact_client_model_states(
    client_ids: Iterable[int],
    models: Mapping[int, torch.nn.Module],
) -> Dict[int, "OrderedDict[str, torch.Tensor]"]:
    return {
        int(cid): compact_state_dict(models[int(cid)].state_dict())
        for cid in client_ids
    }


def _compact_metadata(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        arr = value.copy()
        if np.issubdtype(arr.dtype, np.floating) and arr.size > 0:
            arr_min = float(np.nanmin(arr))
            arr_max = float(np.nanmax(arr))
            if arr_min >= 0.0 and arr_max <= 1.0:
                return arr.astype(np.uint8)
            return arr.astype(np.float32)
        return arr
    if isinstance(value, torch.Tensor):
        return _compact_tensor(value)
    if isinstance(value, dict):
        return {k: _compact_metadata(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_compact_metadata(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_compact_metadata(v) for v in value)
    return value


def compact_algorithm_states(
    states: Mapping[int, Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    return {int(cid): _compact_metadata(state) for cid, state in states.items()}


def _tensor_delta(
    current: torch.Tensor,
    previous: Optional[torch.Tensor],
) -> torch.Tensor:
    cur = current.detach().cpu()
    if previous is None or previous.shape != cur.shape or previous.dtype != cur.dtype:
        return _compact_tensor(cur)
    prev = previous.detach().cpu()
    if torch.is_floating_point(cur):
        return (cur - prev).to(FLOAT_STORAGE_DTYPE)
    return cur.clone()


def build_client_model_deltas(
    client_ids: Iterable[int],
    models: Mapping[int, torch.nn.Module],
    previous_states: Mapping[int, Mapping[str, torch.Tensor]],
) -> Dict[int, "OrderedDict[str, torch.Tensor]"]:
    deltas: Dict[int, "OrderedDict[str, torch.Tensor]"] = {}
    for cid_raw in client_ids:
        cid = int(cid_raw)
        prev_state = previous_states.get(cid, {})
        delta_state = OrderedDict()
        for key, value in models[cid].state_dict().items():
            delta_state[key] = _tensor_delta(value, prev_state.get(key))
        deltas[cid] = delta_state
    return deltas


def cpu_client_model_states(
    client_ids: Iterable[int],
    models: Mapping[int, torch.nn.Module],
) -> Dict[int, "OrderedDict[str, torch.Tensor]"]:
    return {
        int(cid): OrderedDict(
            (k, v.detach().cpu().clone()) for k, v in models[int(cid)].state_dict().items()
        )
        for cid in client_ids
    }


def apply_client_model_deltas(
    base_states: Mapping[int, Mapping[str, torch.Tensor]],
    delta_states: Mapping[int, Mapping[str, torch.Tensor]],
) -> Dict[int, "OrderedDict[str, torch.Tensor]"]:
    reconstructed: Dict[int, "OrderedDict[str, torch.Tensor]"] = {
        int(cid): OrderedDict((k, v.detach().cpu().clone()) for k, v in state.items())
        for cid, state in base_states.items()
    }
    for cid_raw, delta_state in delta_states.items():
        cid = int(cid_raw)
        current = reconstructed.setdefault(cid, OrderedDict())
        for key, delta in delta_state.items():
            delta_cpu = delta.detach().cpu()
            previous = current.get(key)
            if (
                previous is not None
                and torch.is_floating_point(previous)
                and torch.is_floating_point(delta_cpu)
                and previous.shape == delta_cpu.shape
            ):
                current[key] = (previous.float() + delta_cpu.float()).to(previous.dtype)
            else:
                current[key] = delta_cpu.clone()
    return reconstructed


def save_task_base_checkpoint(
    path: str,
    *,
    task_id: int,
    client_ids: List[int],
    models: Mapping[int, torch.nn.Module],
    client_algorithm_states: Mapping[int, Dict[str, Any]],
    config: Dict[str, Any],
    seen_classes: List[int],
) -> None:
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_type": "denice_delta_base",
            "storage_dtype": "fp16",
            "mode": "decentralized",
            "algorithm": "denice",
            "task_id": int(task_id),
            "round_id": -1,
            "client_ids": [int(cid) for cid in client_ids],
            "client_model_states": compact_client_model_states(client_ids, models),
            "client_algorithm_states": compact_algorithm_states(client_algorithm_states),
            "config": config,
            "seen_classes": list(seen_classes),
        },
        path,
    )


def save_delta_round_checkpoint(
    path: str,
    *,
    task_id: int,
    round_id: int,
    base_path: str,
    previous_round_path: Optional[str],
    client_ids: List[int],
    models: Mapping[int, torch.nn.Module],
    previous_model_states: Mapping[int, Mapping[str, torch.Tensor]],
    client_algorithm_states: Mapping[int, Dict[str, Any]],
    config: Dict[str, Any],
    seen_classes: List[int],
    cluster: Dict[str, Any],
    metrics: Dict[str, Any],
) -> None:
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_type": "denice_delta_round",
            "storage_dtype": "fp16_delta",
            "mode": "decentralized",
            "algorithm": "denice",
            "task_id": int(task_id),
            "round_id": int(round_id),
            "base_path": os.path.basename(base_path),
            "previous_round_path": (
                os.path.basename(previous_round_path) if previous_round_path else None
            ),
            "client_ids": [int(cid) for cid in client_ids],
            "client_model_deltas": build_client_model_deltas(
                client_ids, models, previous_model_states
            ),
            "client_algorithm_states": compact_algorithm_states(client_algorithm_states),
            "config": config,
            "seen_classes": list(seen_classes),
            "cluster": cluster,
            "metrics": metrics,
        },
        path,
    )


def load_denice_checkpoint(path: str) -> Dict[str, Any]:
    """Load a full or delta DeNICE checkpoint and return full client states."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("checkpoint_type") != "denice_delta_round":
        return ckpt

    root = os.path.dirname(path)
    base_path = os.path.join(root, ckpt["base_path"])
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    states = base["client_model_states"]

    chain: List[str] = []
    cursor = path
    while True:
        current = torch.load(cursor, map_location="cpu", weights_only=False)
        chain.append(cursor)
        prev = current.get("previous_round_path")
        if not prev:
            break
        cursor = os.path.join(root, prev)
    for delta_path in reversed(chain):
        delta = torch.load(delta_path, map_location="cpu", weights_only=False)
        states = apply_client_model_deltas(states, delta["client_model_deltas"])

    full = dict(ckpt)
    full.pop("client_model_deltas", None)
    full["checkpoint_type"] = "denice_reconstructed_round"
    full["client_model_states"] = states
    return full


def update_checkpoint_index(output_dir: str, record: Dict[str, Any]) -> None:
    path = os.path.join(output_dir, "checkpoint_index.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            index = json.load(f)
    else:
        index = {"schema_version": CHECKPOINT_SCHEMA_VERSION, "checkpoints": []}
    checkpoints = index.setdefault("checkpoints", [])
    key = (record.get("task_id"), record.get("round_id"), record.get("kind"))
    checkpoints[:] = [
        item
        for item in checkpoints
        if (item.get("task_id"), item.get("round_id"), item.get("kind")) != key
    ]
    checkpoints.append(record)
    checkpoints.sort(
        key=lambda item: (
            int(item.get("task_id", -1)),
            int(item.get("round_id", -1)),
            str(item.get("kind", "")),
        )
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, default=str)
