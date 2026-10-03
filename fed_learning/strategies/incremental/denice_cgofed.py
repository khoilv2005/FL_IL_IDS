"""Client-local CGoFed subspace protection for DeNICE classifier updates.

The module deliberately stores activation bases per client. It does not share
representations between independently trained DeNICE models.
"""

from __future__ import annotations

import math
import hashlib
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


CGOFED_STATE_SCHEMA_VERSION = 1


def cgofed_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the DeNICE+CGoFed head-protection controls."""
    mode = str(config.get("denice_cgofed_optimizer", "adam_delta")).lower()
    if mode not in {"adam_delta", "sgd_gradient"}:
        raise ValueError("denice_cgofed_optimizer must be adam_delta or sgd_gradient")
    mu = float(config.get("denice_cgofed_mu", 0.5))
    decay = float(config.get("denice_cgofed_decay", 0.8))
    energy = float(config.get("denice_cgofed_energy", 0.95))
    beta = float(config.get("denice_cgofed_beta", 1.0))
    samples = config.get("denice_cgofed_max_samples", 512)
    capture_batch_size = config.get("denice_cgofed_capture_batch_size", 512)
    max_rank = config.get("denice_cgofed_max_rank")
    if not math.isfinite(mu) or not 0.0 <= mu <= 1.0:
        raise ValueError("denice_cgofed_mu must be finite and in [0, 1]")
    if not math.isfinite(decay) or not 0.0 <= decay <= 1.0:
        raise ValueError("denice_cgofed_decay must be finite and in [0, 1]")
    if not math.isfinite(energy) or not 0.0 < energy <= 1.0:
        raise ValueError("denice_cgofed_energy must be finite and in (0, 1]")
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError("denice_cgofed_beta must be finite and non-negative")
    if isinstance(samples, bool) or int(samples) != samples or int(samples) < 1:
        raise ValueError("denice_cgofed_max_samples must be a positive integer")
    if (
        isinstance(capture_batch_size, bool)
        or int(capture_batch_size) != capture_batch_size
        or int(capture_batch_size) < 1
    ):
        raise ValueError("denice_cgofed_capture_batch_size must be a positive integer")
    if max_rank is not None and (
        isinstance(max_rank, bool) or int(max_rank) != max_rank or int(max_rank) < 1
    ):
        raise ValueError("denice_cgofed_max_rank must be a positive integer or null")
    return {
        "optimizer": mode,
        "mu": mu,
        "decay": decay,
        "energy": energy,
        "beta": beta,
        "max_samples": int(samples),
        "capture_batch_size": int(capture_batch_size),
        "max_rank": None if max_rank is None else int(max_rank),
        "peer_projection": bool(config.get("denice_cgofed_peer_projection", True)),
    }


def empty_projection_state() -> Dict[str, Any]:
    return {
        "schema_version": CGOFED_STATE_SCHEMA_VERSION,
        "completed_local_tasks": 0,
        "tasks": [],
    }


def ensure_projection_state(model) -> Dict[str, Any]:
    state = getattr(model, "cgofed_projection_state", None)
    if not isinstance(state, dict) or int(state.get("schema_version", -1)) != CGOFED_STATE_SCHEMA_VERSION:
        state = empty_projection_state()
        model.cgofed_projection_state = state
    state.setdefault("completed_local_tasks", 0)
    state.setdefault("tasks", [])
    return state


def scheduled_mu(state: Dict[str, Any], controls: Dict[str, Any]) -> float:
    completed = max(0, int(state.get("completed_local_tasks", 0)))
    return float(controls["mu"]) * float(controls["decay"]) ** max(0, completed - 1)


@torch.no_grad()
def collect_task_features(
    model,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    max_samples: int,
    batch_size: int,
    seed: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Collect balanced, train-only inputs to the actual masked ``fc2`` path."""
    n = int(labels.numel())
    if n == 0:
        return torch.empty((0, int(model.fc2.in_features))), {"sample_count": 0}
    labels_cpu = labels.detach().cpu().long().reshape(-1)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    classes = torch.unique(labels_cpu, sorted=True)
    selected = []
    base_quota, extra = divmod(int(max_samples), max(1, len(classes)))
    for class_index, cls in enumerate(classes):
        indices = torch.nonzero(labels_cpu == cls, as_tuple=False).flatten()
        order = torch.randperm(len(indices), generator=generator)
        quota = base_quota + int(class_index < extra)
        if quota:
            selected.append(indices[order[:quota]])
    indices = torch.cat(selected) if selected else torch.empty(0, dtype=torch.long)

    module_modes = {module: bool(module.training) for module in model.modules()}
    active_adapters = dict(getattr(model, "active_adapters", {}))
    model.eval()
    features = []
    device = next(model.parameters()).device
    try:
        for start in range(0, len(indices), max(1, int(batch_size))):
            batch_idx = indices[start : start + max(1, int(batch_size))]
            x = inputs[batch_idx].to(device, non_blocking=True)
            # DeNICEModel.penultimate_features executes the same masked fc1 and
            # active adapter path as forward(), immediately before fc2 dropout.
            z = model.penultimate_features(x)
            features.append(z.detach().float().cpu())
    finally:
        model.active_adapters = {
            layer: adapter_key
            for layer, adapter_key in active_adapters.items()
            if adapter_key in getattr(model, "adapters", {})
        }
        # Restore each module's mode independently. DeNICE may deliberately keep
        # mature BatchNorm modules in eval mode while the rest of the model trains.
        for module, training in module_modes.items():
            module.training = training
    values = torch.cat(features, dim=0) if features else torch.empty((0, model.fc2.in_features))
    if values.numel() and not torch.isfinite(values).all():
        raise FloatingPointError("Non-finite DeNICE fc2 activations; CGoFed basis was not updated")
    audit = {
        "sample_count": int(values.shape[0]),
        "class_count": int(len(classes)),
        "classes": [int(v) for v in classes.tolist()],
        "feature_dim": int(values.shape[1]),
        "adapter_context": {str(k): str(v) for k, v in active_adapters.items()},
        "topology_signature": hashlib.sha256(
            model.weight_masks["fc2"].detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest(),
    }
    return values, audit


@torch.no_grad()
def append_task_basis(
    state: Dict[str, Any],
    features: torch.Tensor,
    *,
    task_id: int,
    energy_threshold: float,
    beta: float,
    max_rank: Optional[int],
    class_support=None,
    adapter_context=None,
    topology_signature: Optional[str] = None,
) -> Dict[str, Any]:
    """Build and append an FP32 activation basis using a small feature Gram matrix."""
    x = features.detach().to(device="cpu", dtype=torch.float64)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        entry = {
            "task_id": int(task_id),
            "layers": {},
            "sample_count": int(x.shape[0]) if x.ndim == 2 else 0,
            "class_support": [int(v) for v in (class_support or [])],
            "adapter_context": dict(adapter_context or {}),
            "topology_signature": topology_signature,
        }
        state["tasks"].append(entry)
        state["completed_local_tasks"] = int(state.get("completed_local_tasks", 0)) + 1
        return {"rank": 0, "energy_retained": 0.0, "sample_count": entry["sample_count"]}
    if not torch.isfinite(x).all():
        raise FloatingPointError("Non-finite CGoFed activation samples")
    gram = x.T @ x
    if not torch.isfinite(gram).all():
        raise FloatingPointError("Non-finite CGoFed activation Gram matrix")
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0.0)
    eigenvectors = eigenvectors[:, order]
    positive = eigenvalues > max(1e-12, float(eigenvalues.max().item()) * 1e-10)
    eigenvalues = eigenvalues[positive]
    eigenvectors = eigenvectors[:, positive]
    if eigenvalues.numel() == 0:
        keep = 0
        retained = 0.0
    else:
        cumulative = eigenvalues.cumsum(0) / eigenvalues.sum().clamp_min(1e-30)
        keep = int(torch.searchsorted(cumulative, torch.tensor(float(energy_threshold), dtype=cumulative.dtype)).item()) + 1
        if max_rank is not None:
            keep = min(keep, int(max_rank))
        keep = max(1, min(keep, int(eigenvalues.numel())))
        retained = float(eigenvalues[:keep].sum().item() / eigenvalues.sum().item())
    basis = eigenvectors[:, :keep].to(torch.float32).contiguous()
    importance = torch.sigmoid(float(beta) * eigenvalues[:keep].sqrt()).to(torch.float32)
    layer_state = {
        "basis": basis,
        "importance": importance,
        "input_dimension": int(x.shape[1]),
        "rank": int(keep),
        "sample_count": int(x.shape[0]),
        "energy_retained": retained,
    }
    state["tasks"].append(
        {
            "task_id": int(task_id),
            "layers": {"fc2": layer_state},
            "sample_count": int(x.shape[0]),
            "class_support": [int(v) for v in (class_support or [])],
            "adapter_context": dict(adapter_context or {}),
            "topology_signature": topology_signature,
        }
    )
    state["completed_local_tasks"] = int(state.get("completed_local_tasks", 0)) + 1
    return {
        "rank": int(keep),
        "energy_retained": retained,
        "sample_count": int(x.shape[0]),
        "feature_dim": int(x.shape[1]),
    }


@torch.no_grad()
def project_fc2_delta(
    delta: torch.Tensor,
    *,
    weight_mask: torch.Tensor,
    unit_ranks,
    state: Dict[str, Any],
    mu: float,
    projector_cache: Optional[Dict[Any, Any]] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Project mature output-row updates into each row's allowed input support."""
    result = delta.detach().clone()
    mask = weight_mask.to(device=delta.device, dtype=torch.bool)
    ranks = np.asarray(unit_ranks, dtype=np.int64).reshape(-1)
    tasks = state.get("tasks", [])
    if float(mu) <= 0.0:
        return torch.where(mask, result, torch.zeros_like(result)), {
            "projected_rows": 0,
            "projection_ratio": 0.0,
            "mu": float(mu),
        }
    if not tasks:
        ranks = np.asarray(unit_ranks, dtype=np.int64).reshape(-1)
        result = torch.where(mask, result, torch.zeros_like(result))
        mature = torch.as_tensor(ranks >= 2, device=delta.device)
        result[mature] = 0.0
        return result, {
            "projected_rows": 0,
            "projection_ratio": 0.0,
            "mu": float(mu),
            "reason": "no_historical_basis_mature_rows_frozen",
        }
    if not torch.isfinite(delta).all():
        return torch.zeros_like(result), {
            "projected_rows": 0,
            "projection_ratio": 0.0,
            "mu": float(mu),
            "reason": "non_finite_update_zeroed",
        }
    input_dim = int(delta.shape[1])
    task_signature = tuple(int(task.get("task_id", -1)) for task in tasks)
    bases = []
    for task in tasks:
        layer = (task.get("layers") or {}).get("fc2")
        if not layer:
            continue
        basis = layer.get("basis")
        importance = layer.get("importance")
        if not torch.is_tensor(basis) or basis.ndim != 2 or basis.shape[0] != input_dim:
            continue
        if not torch.is_tensor(importance) or importance.numel() != basis.shape[1]:
            continue
        bases.append((basis.to(delta.device, torch.float64), importance.to(delta.device, torch.float64)))
    if not bases:
        ranks = np.asarray(unit_ranks, dtype=np.int64).reshape(-1)
        result = torch.where(mask, result, torch.zeros_like(result))
        mature = torch.as_tensor(ranks >= 2, device=delta.device)
        result[mature] = 0.0
        return result, {
            "projected_rows": 0,
            "projection_ratio": 0.0,
            "mu": float(mu),
            "reason": "no_compatible_basis_mature_rows_frozen",
        }

    original_norm_sq = torch.zeros((), device=delta.device, dtype=torch.float64)
    removed_norm_sq = torch.zeros_like(original_norm_sq)
    projected_rows = 0
    support_masks = mask.detach().cpu().numpy()
    per_call_device_cache = {}
    for row, rank in enumerate(ranks):
        if int(rank) < 2:  # Keep the current learner row plastic.
            result[row] = torch.where(mask[row], delta[row], torch.zeros_like(delta[row]))
            continue
        allowed_indices = np.flatnonzero(support_masks[row])
        if allowed_indices.size == 0:
            result[row].zero_()
            continue
        allowed = torch.as_tensor(allowed_indices, device=delta.device, dtype=torch.long)
        d = delta[row, allowed].to(torch.float64)
        cache_key = (str(delta.device), input_dim, task_signature, tuple(allowed_indices.tolist()))
        cached = per_call_device_cache.get(cache_key)
        if cached is None:
            cached = projector_cache.get(cache_key) if projector_cache is not None else None
        if cached is None:
            gram = torch.zeros((len(allowed), len(allowed)), device=delta.device, dtype=torch.float64)
            for basis, importance in bases:
                weighted = basis[allowed] * importance.unsqueeze(0)
                gram.add_(weighted @ weighted.T)
            gram = 0.5 * (gram + gram.T)
            vals, vecs = torch.linalg.eigh(gram)
            vals = vals.clamp_min(0.0)
            max_value = vals.max() if vals.numel() else vals.new_zeros(())
            if float(max_value.item()) <= 1e-12:
                result[row].zero_()
                continue
            significant = vals > max_value * 1e-8
            q = vecs[:, significant].to(torch.float32).contiguous()
            weights = (vals[significant] / max_value).to(torch.float32).contiguous()
            cached = (q.detach().cpu(), weights.detach().cpu())
            if projector_cache is not None:
                if len(projector_cache) >= 128:
                    projector_cache.clear()
                projector_cache[cache_key] = cached
        q, weights = cached
        if q.device != delta.device:
            q = q.to(device=delta.device)
            weights = weights.to(device=delta.device)
        q = q.to(dtype=torch.float64)
        weights = weights.to(dtype=torch.float64)
        per_call_device_cache[cache_key] = (q, weights)
        # Low-rank multiplication avoids materializing a dense d-by-d matrix
        # on every client training step.
        projected = d - float(mu) * ((d @ q * weights) @ q.T)
        result[row].zero_()
        result[row, allowed] = projected.to(result.dtype)
        removed = d - projected
        original_norm_sq += d.square().sum()
        removed_norm_sq += removed.square().sum()
        projected_rows += 1
    ratio = float((removed_norm_sq.sqrt() / original_norm_sq.sqrt().clamp_min(1e-12)).item())
    return result, {
        "projected_rows": int(projected_rows),
        "projection_ratio": ratio,
        "mu": float(mu),
        "basis_tasks": int(len(bases)),
    }


@torch.no_grad()
def apply_local_fc2_delta(model, before: torch.Tensor, controls: Dict[str, Any]) -> Dict[str, Any]:
    """Commit Adam's proposed fc2 delta after applying the receiver-local bank."""
    state = ensure_projection_state(model)
    cache = getattr(model, "_cgofed_projector_cache", None)
    if cache is None:
        cache = {}
        model._cgofed_projector_cache = cache
    parameter = model.fc2.weight
    mask = model.weight_masks["fc2"].to(device=parameter.device, dtype=torch.bool)
    ranks = model.unit_ranks["fc2"]
    candidate = parameter.detach().clone()
    delta = candidate - before.to(candidate.device)
    if not torch.isfinite(candidate).all() or not torch.isfinite(delta).all():
        parameter.copy_(before.to(parameter.device))
        raise FloatingPointError("Non-finite Adam fc2 update; restored pre-step CGoFed weights")
    projected, audit = project_fc2_delta(
        delta,
        weight_mask=mask,
        unit_ranks=ranks,
        state=state,
        mu=scheduled_mu(state, controls),
        projector_cache=cache,
    )
    # Unallocated rows stay fixed; allowed current/young rows retain their
    # ordinary Adam update, while mature rows receive the projected update.
    active_rows = torch.as_tensor(ranks > 0, device=parameter.device)
    committed = before.to(parameter.device) + projected
    parameter.copy_(torch.where(active_rows[:, None], committed, before.to(parameter.device)))
    audit.update({"mature_rows": int((torch.as_tensor(ranks) >= 2).sum()), "optimizer": "adam_delta"})
    return audit


def project_local_gradient(model, controls: Dict[str, Any]) -> Dict[str, Any]:
    """Reference SGD path: project masked mature-row gradients before the step."""
    grad = model.fc2.weight.grad
    if grad is None:
        return {"projected_rows": 0, "projection_ratio": 0.0}
    state = ensure_projection_state(model)
    cache = getattr(model, "_cgofed_projector_cache", None)
    if cache is None:
        cache = {}
        model._cgofed_projector_cache = cache
    projected, audit = project_fc2_delta(
        grad,
        weight_mask=model.weight_masks["fc2"],
        unit_ranks=model.unit_ranks["fc2"],
        state=state,
        mu=scheduled_mu(state, controls),
        projector_cache=cache,
    )
    grad.copy_(projected)
    audit["optimizer"] = "sgd_gradient"
    return audit


def project_peer_fc2_correction(model, correction: torch.Tensor, controls: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Filter the post-local peer correction using this receiver's basis."""
    state = ensure_projection_state(model)
    cache = getattr(model, "_cgofed_projector_cache", None)
    if cache is None:
        cache = {}
        model._cgofed_projector_cache = cache
    return project_fc2_delta(
        correction,
        weight_mask=model.weight_masks["fc2"],
        unit_ranks=model.unit_ranks["fc2"],
        state=state,
        mu=scheduled_mu(state, controls),
        projector_cache=cache,
    )
