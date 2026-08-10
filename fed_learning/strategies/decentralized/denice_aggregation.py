"""
Age-aware decentralized aggregation for DeNICE (plan section 2.5, Alg. 2 phase 6).

Neighbor weight (plan)::

    alpha_ij = s_ij * n_j * R_j / sum_k (s_ik * n_k * R_k)

Update (masked by NICE-compatible mask)::

    theta_i <- theta_i + eta * sum_j alpha_ij * (M_ij o Delta theta_j)

The aggregation mask ``M_ij`` only enables a parameter when it is plastic on the
receiver side (receiver neuron not mature), so a neighbor update can never
overwrite the receiver's mature (frozen) knowledge. Adapters are aggregated
separately and only when ``adapter_id`` / shape / architecture match.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch


@dataclass
class AggregationConfig:
    eta: float = 1.0           # aggregation step size (eta_agg)
    epsilon: float = 1e-8
    protect_mature: bool = True
    # Robust in-cluster aggregation (Đề xuất section 7 / plan section 2.5).
    #   "weighted_mean"     -> sum_j alpha_ij * Delta_j  (default, faithful alpha)
    #   "coordinate_median" -> coordinate-wise median of neighbor deltas
    #   "trimmed_mean"      -> coordinate-wise trimmed mean of neighbor deltas
    method: str = "weighted_mean"
    trim_ratio: float = 0.1    # fraction trimmed each side for trimmed_mean


# Layer name -> mapping helpers (mirror NICEModel.reset_frozen_gradients).
_BN_LAYER_MAP = {"bn1": "conv1", "bn2": "conv2", "bn3": "conv3"}


def aggregation_weights(
    similarities: List[float],
    counts: List[float],
    reliabilities: List[float],
    self_index: Optional[int] = None,
    count_transform: str = "log",
    self_floor: float = 0.0,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Normalized alpha_ij over a collaboration group (plan section 2.5).

    If the total weight collapses, the receiver keeps its own update
    (``alpha_ii = 1``) following Algorithm 2 phase 6.
    """
    s = np.maximum(np.asarray(similarities, dtype=np.float64), 0.0)
    n = np.maximum(np.asarray(counts, dtype=np.float64), 0.0)
    if count_transform == "log":
        n = np.log1p(n)
    elif count_transform == "sqrt":
        n = np.sqrt(n)
    r = np.maximum(np.asarray(reliabilities, dtype=np.float64), 0.0)
    raw = s * n * r
    total = float(raw.sum())
    if total < epsilon:
        alpha = np.zeros_like(raw)
        if self_index is not None and 0 <= self_index < len(alpha):
            alpha[self_index] = 1.0
        return alpha
    alpha = raw / total
    if (
        self_floor > 0.0
        and self_index is not None
        and 0 <= self_index < len(alpha)
        and len(alpha) > 1
        and alpha[self_index] < self_floor
    ):
        floor = min(float(self_floor), 1.0)
        other_total = float(alpha.sum() - alpha[self_index])
        if other_total > epsilon:
            scale = (1.0 - floor) / other_total
            alpha = alpha * scale
            alpha[self_index] = floor
        else:
            alpha = np.zeros_like(alpha)
            alpha[self_index] = 1.0
    return alpha


def _mature_mask_for_param(
    param_name: str,
    param_shape: torch.Size,
    ages: Dict[str, np.ndarray],
    gru_hidden: int = 100,
) -> Optional[np.ndarray]:
    """Return a boolean mask (True = mature, must NOT be updated) for a param.

    Returns ``None`` when the parameter has no age-tracked output dimension
    (then it is treated as fully plastic).
    """
    layer = param_name.split(".")[0]

    if layer in _BN_LAYER_MAP:
        layer = _BN_LAYER_MAP[layer]
    if layer not in ages:
        return None

    ranks = np.asarray(ages[layer])
    mature = ranks >= 2

    if layer == "gru":
        # GRU gate params have first dim 3*hidden.
        if len(param_shape) >= 1 and param_shape[0] == 3 * gru_hidden:
            return np.tile(mature, 3)
        if len(param_shape) >= 1 and param_shape[0] == len(mature):
            return mature
        return None

    if len(param_shape) >= 1 and param_shape[0] == len(mature):
        return mature
    return None


def build_compatible_mask(
    target_params: "OrderedDict[str, torch.Tensor]",
    target_ages: Dict[str, np.ndarray],
    gru_hidden: int = 100,
) -> Dict[str, torch.Tensor]:
    """Per-parameter keep-mask (1 = may receive update, 0 = protected mature)."""
    masks: Dict[str, torch.Tensor] = {}
    for name, param in target_params.items():
        mature = _mature_mask_for_param(name, param.shape, target_ages, gru_hidden)
        if mature is None:
            masks[name] = torch.ones_like(param)
            continue
        keep = torch.ones_like(param)
        mature_idx = torch.as_tensor(mature.tolist(), dtype=torch.bool)
        # Zero the rows (dim 0) belonging to mature receiver neurons.
        keep[mature_idx] = 0.0
        masks[name] = keep
    return masks


def _robust_combine(
    stacked: torch.Tensor, method: str, trim_ratio: float
) -> torch.Tensor:
    """Combine neighbor deltas along dim 0 with a robust estimator.

    ``stacked`` has shape ``[k, *param_shape]``. Returns a tensor of
    ``param_shape``. Used to replace the weighted average for Byzantine
    robustness (Đề xuất section 7).
    """
    k = stacked.shape[0]
    if k == 0:
        return torch.zeros(stacked.shape[1:], device=stacked.device)
    if k == 1:
        return stacked[0]

    if method == "coordinate_median":
        return torch.median(stacked, dim=0).values

    if method == "trimmed_mean":
        trim = int(np.floor(float(trim_ratio) * k))
        if 2 * trim >= k:
            return stacked.mean(dim=0)
        ordered, _ = torch.sort(stacked, dim=0)
        kept = ordered[trim : k - trim]
        return kept.mean(dim=0)

    # Unknown method -> plain mean (safe fallback).
    return stacked.mean(dim=0)


def age_aware_aggregate(
    target_params: "OrderedDict[str, torch.Tensor]",
    target_ages: Dict[str, np.ndarray],
    neighbor_deltas: List["OrderedDict[str, torch.Tensor]"],
    alphas: np.ndarray,
    config: Optional[AggregationConfig] = None,
    gru_hidden: int = 100,
) -> "OrderedDict[str, torch.Tensor]":
    """Apply masked neighbor deltas to the receiver params.

    ``neighbor_deltas[j]`` is ``theta_j_after - reference`` for neighbor j.

    With ``config.method == "weighted_mean"`` (default) the update is the
    faithful ``sum_j alpha_ij * Delta_j``. The robust variants
    (``coordinate_median`` / ``trimmed_mean``) replace the weighted average by a
    coordinate-wise robust estimator over the neighbor deltas (Đề xuất §7).
    """
    config = config or AggregationConfig()
    masks = (
        build_compatible_mask(target_params, target_ages, gru_hidden)
        if config.protect_mature
        else {name: torch.ones_like(p) for name, p in target_params.items()}
    )
    robust = config.method in ("coordinate_median", "trimmed_mean")

    new_params = OrderedDict()
    for name, param in target_params.items():
        if robust:
            contribs = [
                delta[name].to(param.device)
                for delta in neighbor_deltas
                if name in delta and delta[name].shape == param.shape
            ]
            if contribs:
                stacked = torch.stack(contribs, dim=0)
                agg = _robust_combine(stacked, config.method, config.trim_ratio)
            else:
                agg = torch.zeros_like(param)
        else:
            agg = torch.zeros_like(param)
            for j, delta in enumerate(neighbor_deltas):
                if name not in delta:
                    continue
                d = delta[name]
                if d.shape != param.shape:
                    continue
                agg = agg + float(alphas[j]) * d.to(param.device)
        mask = masks.get(name, torch.ones_like(param)).to(param.device)
        new_params[name] = param + config.eta * (mask * agg)
    return new_params


def merge_neuron_ages(
    target_ages: Dict[str, np.ndarray],
    neighbor_ages: List[Dict[str, np.ndarray]],
    neighbor_weights: Optional[List[float]] = None,
    policy: str = "consensus",
    consensus_threshold: float = 0.5,
) -> Dict[str, np.ndarray]:
    """Merge peer maturity without turning disjoint selections into a union.

    ``policy="consensus"`` (the default) only promotes a receiver neuron to
    mature when peers that mark that same neuron mature contribute at least
    ``consensus_threshold`` total neighbor weight.  Existing receiver maturity
    is never reduced.  This prevents the old element-wise max rule from
    exhausting every client's capacity after repeated decentralized rounds.

    ``policy="max"`` is retained strictly for backwards-compatible ablations.
    It must not be used by the DeNICE runner default.
    """
    merged = {k: np.asarray(v).copy() for k, v in target_ages.items()}
    if not neighbor_ages:
        return merged

    mode = str(policy or "consensus").lower()
    if mode not in {"consensus", "max", "none"}:
        raise ValueError(f"Unsupported neuron-age merge policy: {policy}")
    if mode == "none":
        return merged

    if neighbor_weights is None:
        weights = np.full(len(neighbor_ages), 1.0 / len(neighbor_ages), dtype=np.float64)
    else:
        weights = np.asarray(neighbor_weights, dtype=np.float64)
        if weights.shape != (len(neighbor_ages),):
            raise ValueError("neighbor_weights must align with neighbor_ages")
        weights = np.maximum(weights, 0.0)
        if float(weights.sum()) <= 0.0:
            return merged

    for layer, target in merged.items():
        compatible = []
        compatible_weights = []
        for ages, weight in zip(neighbor_ages, weights):
            arr = np.asarray(ages.get(layer, []))
            if arr.shape != target.shape:
                continue
            compatible.append(arr)
            compatible_weights.append(float(weight))
        if not compatible:
            continue

        stack = np.stack(compatible, axis=0)
        if mode == "max":
            merged[layer] = np.maximum(target, stack.max(axis=0))
            continue

        layer_weights = np.asarray(compatible_weights, dtype=np.float64)
        if float(layer_weights.sum()) <= 0.0:
            continue
        mature_vote = ((stack >= 2) * layer_weights.reshape((-1,) + (1,) * target.ndim)).sum(axis=0)
        promote = (target < 2) & (mature_vote >= float(consensus_threshold))
        peer_max = stack.max(axis=0)
        merged[layer] = np.where(promote, peer_max, target)
    return merged


def aggregate_adapters(
    target_adapter_states: Dict[str, "OrderedDict[str, torch.Tensor]"],
    neighbor_adapter_states: List[Dict[str, "OrderedDict[str, torch.Tensor]"]],
    neighbor_weights: List[float],
    target_weight: float = 1.0,
) -> Dict[str, "OrderedDict[str, torch.Tensor]"]:
    """FedAvg adapters, matched strictly by adapter key (plan section 2.5).

    A neighbor only contributes to an adapter it actually owns (same key ->
    same context_id / layer / rank / architecture_version). Clients without the
    adapter are skipped for that adapter's average.
    """
    merged: Dict[str, OrderedDict] = {}
    target_weight = max(0.0, float(target_weight))
    for key, target_state in target_adapter_states.items():
        contributors = [(target_state, target_weight)]
        for nb_states, w in zip(neighbor_adapter_states, neighbor_weights):
            nb = nb_states.get(key)
            if nb is None:
                continue
            # Shape compatibility check.
            if any(
                p not in nb or nb[p].shape != target_state[p].shape
                for p in target_state
            ):
                continue
            contributors.append((nb, float(w)))

        total_w = sum(w for _, w in contributors)
        if total_w <= 0:
            merged[key] = target_state
            continue

        avg = OrderedDict()
        for p_name, p_val in target_state.items():
            acc = torch.zeros_like(p_val)
            for state, w in contributors:
                acc = acc + w * state[p_name].to(p_val.device)
            avg[p_name] = acc / total_w
        merged[key] = avg
    return merged
