"""NICE neuron usage summary writer."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import numpy as np


SUMMARY_FILENAME = "nice_neuron_usage_summary.json"


def _layer_usage(name: str, ranks: Any, previous_ranks: Optional[Any] = None) -> Dict[str, int]:
    arr = np.asarray(ranks, dtype=np.int32)
    prev = None if previous_ranks is None else np.asarray(previous_ranks, dtype=np.int32)

    total = int(arr.size)
    young = int((arr == 0).sum())
    learner = int((arr == 1).sum())
    mature = int((arr >= 2).sum())
    used = learner + mature
    free = young
    new_used = used
    if prev is not None and prev.shape == arr.shape:
        new_used = int(((prev == 0) & (arr > 0)).sum())

    return {
        "layer": name,
        "total": total,
        "used": used,
        "free": free,
        "new_used_this_task": new_used,
        "young": young,
        "learner": learner,
        "mature": mature,
    }


def compute_nice_neuron_usage(model: Any, previous_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compute cumulative NICE neuron usage from model.unit_ranks."""
    unit_ranks = getattr(model, "unit_ranks", None)
    if not unit_ranks:
        return {}

    layer_order = list(getattr(model, "LAYER_NAMES", unit_ranks.keys()))
    layers = []
    totals = {
        "total": 0,
        "used": 0,
        "free": 0,
        "new_used_this_task": 0,
        "young": 0,
        "learner": 0,
        "mature": 0,
    }

    for name in layer_order:
        if name not in unit_ranks:
            continue
        prev = previous_state.get(name) if previous_state else None
        stats = _layer_usage(name, unit_ranks[name], prev)
        layers.append(stats)
        for key in totals:
            totals[key] += int(stats[key])

    return {"totals": totals, "layers": layers}


def append_nice_neuron_usage(
    output_dir: str,
    task_id: int,
    model: Any,
    previous_state: Optional[Dict[str, Any]] = None,
) -> str:
    """Append per-task NICE neuron usage to one JSON summary file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, SUMMARY_FILENAME)

    usage = compute_nice_neuron_usage(model, previous_state)
    if not usage:
        return path

    record = {
        "task": int(task_id),
        **usage,
    }

    payload: Dict[str, Any]
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = {
            "schema": {
                "used": "neurons with age > 0 (learner + mature)",
                "free": "young/surplus neurons with age == 0",
                "new_used_this_task": "neurons that changed from age 0 to age > 0 since previous task",
            },
            "tasks": [],
        }

    tasks = [entry for entry in payload.get("tasks", []) if int(entry.get("task", -1)) != int(task_id)]
    tasks.append(record)
    tasks.sort(key=lambda entry: int(entry["task"]))
    payload["tasks"] = tasks

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return path
