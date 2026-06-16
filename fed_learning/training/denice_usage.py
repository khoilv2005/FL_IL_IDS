"""DeNICE adapter usage summary writer (plan section 11)."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

ADAPTER_SUMMARY_FILENAME = "denice_adapter_usage_summary.json"


def compute_adapter_usage(model: Any) -> Dict[str, Any]:
    """Summarize the adapter registry of a :class:`DeNICEModel`."""
    registry = getattr(model, "adapter_registry", None)
    if not registry:
        return {
            "num_adapters": 0,
            "total_adapter_params": 0,
            "per_layer": {},
            "adapters": [],
        }

    per_layer: Dict[str, Dict[str, int]] = defaultdict(lambda: {"count": 0, "params": 0})
    adapters: List[Dict[str, Any]] = []
    total_params = 0

    for key, meta in registry.items():
        layer = meta["layer_name"]
        per_layer[layer]["count"] += 1
        per_layer[layer]["params"] += int(meta["param_count"])
        total_params += int(meta["param_count"])
        adapters.append(
            {
                "key": key,
                "context_id": int(meta["context_id"]),
                "layer": layer,
                "rank": int(meta["rank"]),
                "dim": int(meta["dim"]),
                "architecture_version": int(meta["architecture_version"]),
                "param_count": int(meta["param_count"]),
            }
        )

    return {
        "num_adapters": len(registry),
        "total_adapter_params": int(total_params),
        "per_layer": {k: dict(v) for k, v in per_layer.items()},
        "adapters": adapters,
    }


def append_adapter_usage(
    output_dir: str,
    task_id: int,
    model: Any,
    canc_plan: Optional[Dict[str, Any]] = None,
    novelty: Optional[float] = None,
) -> str:
    """Append per-task adapter usage + CANC decisions to one JSON summary."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, ADAPTER_SUMMARY_FILENAME)

    usage = compute_adapter_usage(model)
    record: Dict[str, Any] = {"task": int(task_id), **usage}
    if novelty is not None:
        record["novelty"] = float(novelty)
    if canc_plan is not None:
        record["canc"] = {
            "freeze_low_layers": bool(canc_plan.get("freeze_low_layers", False)),
            "adapters_to_add": list(canc_plan.get("adapters_to_add", [])),
            "layers": {
                name: {
                    "action": info.get("action"),
                    "kappa": round(float(info.get("kappa", 0.0)), 4),
                    "rho0": round(float(info.get("rho0", 0.0)), 4),
                    "rhom": round(float(info.get("rhom", 0.0)), 4),
                    "u": round(float(info.get("u", 0.0)), 4),
                }
                for name, info in canc_plan.get("layers", {}).items()
            },
        }

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = {
            "schema": {
                "num_adapters": "total micro-adapters instantiated so far",
                "total_adapter_params": "sum of U/V parameters across adapters",
                "canc": "Capacity-Aware Neurogenesis Controller decisions for the task",
            },
            "tasks": [],
        }

    tasks = [
        entry for entry in payload.get("tasks", [])
        if int(entry.get("task", -1)) != int(task_id)
    ]
    tasks.append(record)
    tasks.sort(key=lambda entry: int(entry["task"]))
    payload["tasks"] = tasks

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return path
