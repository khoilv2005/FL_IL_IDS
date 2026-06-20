"""Graceful recycling for DeNICE Phase 4.

Recycling is intentionally conservative:

1. Only layers selected by CANC are touched.
2. Only mature neurons (age >= 2) may be retired.
3. Retired neurons move to age=-1 first, are masked out, and cannot be selected
   by NICE training.
4. A later task revives them as young (age=0) after a grace period.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from .denice_capacity import CANCConfig

def _activation_scores(model: Any, data: torch.Tensor) -> Dict[str, np.ndarray]:
    if data is None or data.numel() == 0:
        return {}
    model.eval()
    acts = model.get_activations(data)
    return {
        name: np.asarray(t.detach().cpu().tolist(), dtype=np.float64)
        for name, t in acts.items()
    }

def _choose_low_importance_mature(
    ranks: np.ndarray,
    scores: np.ndarray,
    ratio: float,
    min_count: int,
    max_count: int,
    usage_recent_threshold: float,
) -> List[int]:
    mature = np.where(np.asarray(ranks) >= 2)[0]
    if mature.size == 0:
        return []
    if scores is not None and scores.shape[0] == ranks.shape[0]:
        score_max = float(np.max(scores)) if scores.size else 0.0
        if score_max > 0.0:
            usage_recent = scores / score_max
            mature = mature[usage_recent[mature] <= float(usage_recent_threshold)]
        else:
            usage_recent = np.zeros_like(scores, dtype=np.float64)
        if mature.size == 0:
            return []
    count = max(int(min_count), int(np.ceil(mature.size * float(ratio))))
    count = min(count, int(max_count), int(mature.size))
    if count <= 0:
        return []
    if scores is None or scores.shape[0] != ranks.shape[0]:
        order = mature[np.argsort(ranks[mature])]
    else:
        order = mature[np.argsort(scores[mature])]
    return [int(idx) for idx in order[:count]]

def _old_metric_check_passed(canc_plan: Dict[str, Any], config: CANCConfig) -> bool:
    """Rule-5 safety: do not recycle if old-task quality check is missing or bad."""
    if not bool(config.recycle_require_old_check):
        return True
    if "old_metric_delta" in canc_plan:
        return float(canc_plan.get("old_metric_delta", 0.0)) <= float(
            config.recycle_max_old_metric_drop
        )
    if "old_validation_drop" in canc_plan:
        return float(canc_plan.get("old_validation_drop", 0.0)) <= float(
            config.recycle_max_old_metric_drop
        )
    old_check = canc_plan.get("old_metric_check")
    if isinstance(old_check, dict):
        if "passed" in old_check:
            return bool(old_check.get("passed"))
        if "delta" in old_check:
            return float(old_check.get("delta", 0.0)) <= float(
                config.recycle_max_old_metric_drop
            )
    return False

def revive_due_recycled_neurons(model: Any, task_id: int, config: Dict[str, Any]) -> Dict[str, List[int]]:
    """Revive retired units that have passed the configured grace period."""
    if not hasattr(model, "revive_retired_neurons"):
        return {}
    c = CANCConfig.from_dict(config)
    return model.revive_retired_neurons(task_id, c.recycle_grace_tasks)

def apply_graceful_recycling(
    model: Any,
    ref_data: torch.Tensor,
    task_id: int,
    canc_plan: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Retire low-importance mature neurons selected by CANC.

    Returns a JSON-safe summary and injects it into ``canc_plan["recycling"]``.
    """
    recycle_layers = list(canc_plan.get("recycle_layers", []))
    if not recycle_layers or not hasattr(model, "retire_neurons"):
        summary = {"enabled": False, "retired": {}, "total_retired": 0}
        canc_plan["recycling"] = summary
        return summary

    c = CANCConfig.from_dict(config)
    if not _old_metric_check_passed(canc_plan, c):
        summary = {
            "enabled": True,
            "recycle_layers": recycle_layers,
            "retired": {},
            "total_retired": 0,
            "blocked": True,
            "reason": "old_metric_check_failed_or_missing",
            "max_old_metric_drop": float(c.recycle_max_old_metric_drop),
        }
        canc_plan["recycling"] = summary
        return summary

    scores_by_layer = _activation_scores(model, ref_data)
    retired: Dict[str, List[int]] = {}
    usage_recent: Dict[str, Dict[int, float]] = {}

    for layer in recycle_layers:
        ranks = getattr(model, "unit_ranks", {}).get(layer)
        if ranks is None:
            continue
        ranks = np.asarray(ranks)
        scores = scores_by_layer.get(layer)
        chosen = _choose_low_importance_mature(
            ranks,
            scores,
            c.recycle_ratio,
            c.recycle_min,
            c.recycle_max_per_layer,
            c.recycle_usage_recent_threshold,
        )
        if scores is not None and scores.shape[0] == ranks.shape[0] and scores.size:
            denom = float(np.max(scores)) or 1.0
            usage_recent[layer] = {int(idx): float(scores[int(idx)] / denom) for idx in chosen}
        actual = model.retire_neurons(layer, chosen, task_id)
        if actual:
            retired[layer] = actual

    summary = {
        "enabled": True,
        "recycle_layers": recycle_layers,
        "retired": retired,
        "total_retired": int(sum(len(v) for v in retired.values())),
        "grace_tasks": int(c.recycle_grace_tasks),
        "usage_recent_threshold": float(c.recycle_usage_recent_threshold),
        "usage_recent": usage_recent,
        "old_metric_check": "passed",
        "note": "retired neurons use age=-1 and are revived as age=0 after grace_tasks",
    }
    canc_plan["recycling"] = summary
    return summary
