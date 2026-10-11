"""Canonical BN from current-BASE moments on sequential post-aggregate states."""
from dataclasses import dataclass

import numpy as np
import torch

from .contracts import array, state_hash
from .state import restore, snapshot


def model_hash(state):
    return state_hash(state.model_state)


def like(values, reference):
    if isinstance(reference, torch.Tensor):
        return torch.as_tensor(values, dtype=reference.dtype).clone()
    return np.asarray(values, dtype=reference.dtype).copy()


def bn_layers(state):
    names = {k.removesuffix('.running_mean') for k in state.bn_state if k.endswith('.running_mean')}
    # Native CNN forward order, rather than order of upload/leader choice.
    if names - {'bn1', 'bn2', 'bn3'}:
        raise ValueError('Unsupported BN topology; explicit forward-order contract required')
    return tuple(name for name in ('bn1', 'bn2', 'bn3') if name in names)


def protected_bn_channels(state, layer):
    width = len(array(state.model_state[layer + '.running_mean']))
    conv = {'bn1': 'conv1', 'bn2': 'conv2', 'bn3': 'conv3'}[layer]
    ranks = state.algorithm_state['unit_ranks'].get(conv)
    if ranks is None or array(ranks).shape != (width,):
        raise ValueError('BN requires matching canonical convolution ranks')
    protected = array(ranks) >= 2
    for suffix in ('running_mean', 'running_var', 'weight', 'bias'):
        name = layer + '.' + suffix
        if name in state.routing_dependencies:
            protected[:] = True
        if name in state.protected_parameters:
            protected |= array(state.protected_parameters[name]).astype(bool)
    return protected


@dataclass(frozen=True)
class BNMoment:
    client_id: int
    task: int
    round: int
    layer: str
    model_hash: str
    plan_hash: str
    sample_rows: int
    counts: np.ndarray
    sums: np.ndarray
    squared_sums: np.ndarray
    data_task: int
    role: str = 'base'


def recalibrate_bn(state, plan, provider):
    """provider(stage_clone, layer) returns moments, never raw inputs.

    Each later layer binds the hash AFTER preceding BN recalibration. Local
    running buffers are not inputs here. Population mean and unbiased variance
    overwrite mutable channels; canonical counters/protected channels persist.
    """
    candidate = restore(snapshot(state))
    receipt = []
    for layer in bn_layers(candidate):
        protected = protected_bn_channels(candidate, layer)
        mutable = ~protected
        if not mutable.any():
            receipt.append(dict(layer=layer, policy='reference_protected', contributors=()))
            continue
        if provider is None:
            raise ValueError(f'Current-BASE post-aggregation moments required: {layer}')
        expected = model_hash(candidate)
        packets = tuple(provider(restore(snapshot(candidate)), layer))
        ids = [p.client_id for p in packets]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate BN contributor')
        counts = np.zeros(len(protected), dtype=np.float64)
        sums, squared = counts.copy(), counts.copy()
        for packet in sorted(packets, key=lambda p: p.client_id):
            if (packet.client_id not in plan.base_counts or packet.task != plan.reference.task
                or packet.round != plan.target_round or packet.data_task != packet.task or packet.role != 'base'
                or packet.layer != layer or packet.model_hash != expected or packet.plan_hash != plan.hash):
                raise ValueError('Stale/unauthorized BN moment')
            if type(packet.sample_rows) is not int or not 0 < packet.sample_rows <= sum(plan.base_counts[packet.client_id].values()):
                raise ValueError('Invalid BASE sample authority for BN')
            n, total, sq = map(np.asarray, (packet.counts, packet.sums, packet.squared_sums))
            if any(v.shape != counts.shape or not np.isfinite(v).all() for v in (n, total, sq)):
                raise ValueError('Invalid BN moment vectors')
            if n.dtype.kind not in 'iu' or np.any(n < packet.sample_rows) or np.any(sq < 0):
                raise ValueError('Invalid BN observation counts/squared sums')
            # Moments must describe a realizable distribution, allowing roundoff.
            residual = sq.astype(np.float64) - total.astype(np.float64) ** 2 / n
            if np.any(residual < -1e-10 * np.maximum(1, sq)):
                raise ValueError('Inconsistent BN moments')
            counts += n
            sums += total
            squared += sq
        if np.any(counts[mutable] < 2):
            raise ValueError('Insufficient current BN evidence; transaction rejected')
        mean = sums[mutable] / counts[mutable]
        var = np.maximum(0, squared[mutable] - sums[mutable] ** 2 / counts[mutable]) / (counts[mutable] - 1)
        for suffix, values in [('running_mean', mean), ('running_var', var)]:
            name = layer + '.' + suffix
            result = array(candidate.model_state[name]).copy()
            result[mutable] = values
            candidate.model_state[name] = like(result, candidate.model_state[name])
            candidate.bn_state[name] = like(result, candidate.bn_state[name])
        candidate.validate()
        receipt.append(dict(layer=layer, policy='current_base_global_moments', contributors=tuple(sorted(ids)),
            mutable_channels=int(mutable.sum()), protected_channels=int(protected.sum()),
            input_model_hash=expected, output_model_hash=model_hash(candidate), counts=counts.tolist()))
    return candidate, receipt
