"""P1 pure common-reference aggregation. No native runner/router fitting."""
from dataclasses import asdict, dataclass
import re

import numpy as np

from .contracts import array, binary, decode, encode, state_hash, routing_parameter_mask
from .knowledge import ClassKnowledgeLedger, LocalLearningEvent
from .normalization import like, protected_bn_channels, recalibrate_bn
from .state import broadcast, restore, snapshot


def reference_hash(state):
    """Bind canonical training state; coordinator identity never affects weights."""
    payload = snapshot(state)
    for key in ('coordinator_id', 'ownership_metadata', 'parent_lineage', 'training_rng_state'):
        payload.pop(key)
    return state_hash(payload)


def topology_hash(state):
    return state_hash(dict(versions=state.versions, model_spec=state.model_spec,
        algorithm_state=state.algorithm_state, protected=state.protected_parameters,
        router=state.router_state, dependencies=state.routing_dependencies, routing_hash=state.routing_space_hash))


def parameter_names(state):
    return tuple(sorted(k for k in state.model_state if k.endswith(('.weight', '.bias'))
        or re.fullmatch(r'gru\.(weight|bias)_(ih|hh)_l\d+', k)))


def permissions(state):
    """Canonical learner-only masks. Unknown parameter families fail closed."""
    result = {}
    alg = state.algorithm_state
    for name in parameter_names(state):
        value = state.model_state[name]
        layer = name.split('.')[0]
        allowed = np.zeros(value.shape, dtype=bool)
        if layer in alg['unit_ranks']:
            ranks = array(alg['unit_ranks'][layer])
            rows = (ranks == 1) & ~array(alg['freeze_masks'].get(layer, np.zeros(len(ranks), bool))).astype(bool)
            if layer == 'gru':
                rows &= array(alg['weight_masks'][layer]).astype(bool) & array(alg['bias_masks'][layer]).astype(bool)
                rows = np.tile(rows, 3)
                allowed = np.broadcast_to(rows.reshape((-1,) + (1,) * (value.ndim - 1)), value.shape).copy()
                connection_masks = alg.get('gru_connection_masks', {})
                short_name = name.removeprefix('gru.')
                if name in connection_masks and short_name in connection_masks:
                    raise ValueError('Ambiguous GRU connection mask key')
                mask = connection_masks.get(name, connection_masks.get(short_name))
                if mask is not None:
                    if array(mask).shape != value.shape or not binary(mask):
                        raise ValueError('Invalid canonical GRU connection mask')
                    allowed &= array(mask).astype(bool)
            else:
                mask = alg['weight_masks' if name.endswith('.weight') else 'bias_masks'][layer]
                allowed = array(mask).astype(bool) & rows.reshape((-1,) + (1,) * (value.ndim - 1))
        elif layer in ('bn1', 'bn2', 'bn3'):
            allowed = ~protected_bn_channels(state, layer)
        if name in state.protected_parameters:
            allowed &= ~array(state.protected_parameters[name]).astype(bool)
        if name in state.routing_dependencies:
            allowed &= ~routing_parameter_mask(state,name)
        result[name] = allowed
    return result


@dataclass
class RoundPlan:
    reference: object
    current_classes: tuple
    base_counts: dict
    base_role_hash: str
    target_round: int
    reference_hash: str
    topology_hash: str
    parameter_permissions: dict
    hash: str

    def validate(self):
        self.reference.validate()
        if self.target_round != self.reference.round + 1:
            raise ValueError('Invalid target round')
        if self.reference_hash != reference_hash(self.reference) or self.topology_hash != topology_hash(self.reference):
            raise ValueError('Round reference mutated')
        if state_hash(self.parameter_permissions) != state_hash(permissions(self.reference)):
            raise ValueError('Canonical permissions mutated')
        if self.hash != state_hash(self.binding()):
            raise ValueError('Round authority mutated')

    def binding(self):
        return dict(reference_hash=self.reference_hash, topology_hash=self.topology_hash,
            current_classes=self.current_classes, base_counts=self.base_counts,
            base_role_hash=self.base_role_hash, target_round=self.target_round,
            parameter_permissions=self.parameter_permissions)


def prepare_round(reference, *, current_classes, base_counts, base_role_hash):
    canonical = restore(snapshot(reference))
    if any(array(canonical.model_state[k]).dtype.kind != 'f' for k in parameter_names(canonical)):
        raise ValueError('Trainable parameter tensors must be floating point')
    current_classes = tuple(sorted(current_classes))
    if not current_classes or len(set(current_classes)) != len(current_classes) or not set(current_classes) <= set(canonical.versions['class_ids']):
        raise ValueError('Invalid current task class authority')
    if not base_role_hash or set(base_counts) != set(canonical.members):
        raise ValueError('Current BASE authority must cover exactly the cluster members')
    counts = decode(encode(base_counts))
    if any(type(c) is not int or c not in current_classes or type(n) is not int or n <= 0
           for row in counts.values() for c, n in row.items()):
        raise ValueError('Only positive current-task BASE counts are allowed')
    plan = RoundPlan(canonical, current_classes, counts, base_role_hash, canonical.round + 1,
        reference_hash(canonical), topology_hash(canonical), permissions(canonical), '')
    plan.hash = state_hash(plan.binding())
    plan.validate()
    return plan


@dataclass
class WorkerSnapshot:
    client_id: int
    plan_hash: str
    initial_reference_hash: str
    initial_topology_hash: str
    state: object


def start_worker(plan, client_id):
    plan.validate()
    if client_id not in plan.reference.members:
        raise ValueError('Worker outside cluster')
    state = restore(snapshot(plan.reference))
    return WorkerSnapshot(client_id, plan.hash, reference_hash(state), topology_hash(state), state)


@dataclass
class LocalContribution:
    event: LocalLearningEvent
    plan_hash: str
    initial_reference_hash: str
    initial_topology_hash: str
    deltas: dict
    masks: dict


def _validate_event(plan, event):
    if not isinstance(event, LocalLearningEvent) or event.client_id not in plan.base_counts:
        raise ValueError('Invalid contributor event/member')
    if event.task != plan.reference.task or event.round != plan.target_round or not event.completed:
        raise ValueError('Incomplete/stale contributor event')
    # Validate the existing local receipt contract without changing the live ledger.
    # Task IDs need not start at zero in a round. Reuse ledger with global static
    # class mapping, and check current membership explicitly before recording.
    probe = ClassKnowledgeLedger({0: tuple(plan.reference.versions['class_ids'])})
    if any(type(v) is not int for v in (event.task, event.round, event.data_task)):
        raise ValueError('Invalid training receipt clock')
    rewritten = LocalLearningEvent(**{**asdict(event), 'task': 0, 'data_task': 0})
    if event.role != 'base' or event.data_task != event.task:
        raise ValueError('Historical/non-BASE contribution is forbidden')
    probe.record(rewritten, at_task=0, at_round=event.round)
    trained = dict(event.successful_class_counts)
    if any(c not in plan.current_classes or c not in plan.base_counts[event.client_id] for c in trained):
        raise ValueError('Training class has no authorized current BASE support')
    return trained


def finish_worker(plan, worker, event, *, contribution_masks=None):
    plan.validate()
    trained = _validate_event(plan, event)
    if (event.client_id != worker.client_id or worker.plan_hash != plan.hash
        or worker.initial_reference_hash != plan.reference_hash or worker.initial_topology_hash != plan.topology_hash
        or topology_hash(worker.state) != plan.topology_hash or worker.state.cluster_uid != plan.reference.cluster_uid
        or worker.state.task != plan.reference.task or worker.state.round != plan.reference.round
        or worker.state.model_spec != plan.reference.model_spec or set(worker.state.model_state) != set(plan.reference.model_state)):
        raise ValueError('Worker did not retain canonical reference/topology')
    names = parameter_names(plan.reference)
    if contribution_masks is not None and set(contribution_masks) != set(names):
        raise ValueError('Contribution mask schema mismatch')
    deltas, masks = {}, {}
    for name, old in plan.reference.model_state.items():
        local = worker.state.model_state[name]
        if (tuple(local.shape), str(local.dtype)) != plan.reference.model_spec[name] or not np.isfinite(array(local)).all():
            raise ValueError('Local tensor shape/dtype/finite mismatch')
        before, after = array(old), array(local)
        allowed = plan.parameter_permissions.get(name, np.zeros(old.shape, bool))
        if name in plan.reference.bn_state and not name.endswith('num_batches_tracked'):
            allowed = ~protected_bn_channels(plan.reference, name.split('.')[0])
        if name.endswith('num_batches_tracked'):
            # Local counters are discarded along with local running buffers.
            continue
        if state_hash(before[~allowed]) != state_hash(after[~allowed]):
            raise ValueError(f'Protected/unauthorized local mutation: {name}')
        if name not in names:
            continue  # Mutable local BN buffers are never averaged.
        delta = after.astype(np.float64) - before.astype(np.float64)
        if not event.successful_steps and np.any(delta):
            raise ValueError('Weights changed without a successful optimizer step')
        mask = allowed.copy()
        if contribution_masks is not None:
            requested = array(contribution_masks[name])
            if requested.shape != old.shape or not binary(requested) or np.any(requested.astype(bool) & ~allowed):
                raise ValueError('Contribution expands canonical permissions')
            mask &= requested.astype(bool)
        if name.startswith('fc2.'):
            mask &= np.isin(np.arange(len(before)), tuple(trained)).reshape((-1,) + (1,) * (old.ndim - 1))
        if not event.successful_steps:
            mask[:] = False
        masks[name], deltas[name] = mask, np.where(mask, delta, 0)
    return LocalContribution(event, plan.hash, worker.initial_reference_hash, worker.initial_topology_hash, deltas, masks)


def _validate_contribution(plan, packet):
    trained = _validate_event(plan, packet.event)
    if (packet.plan_hash != plan.hash or packet.initial_reference_hash != plan.reference_hash
        or packet.initial_topology_hash != plan.topology_hash):
        raise ValueError('Stale/mismatched common-reference packet')
    names = set(parameter_names(plan.reference))
    if set(packet.deltas) != names or set(packet.masks) != names:
        raise ValueError('Delta/mask parameter schema mismatch')
    for name in names:
        delta, mask = array(packet.deltas[name]), array(packet.masks[name])
        if delta.shape != plan.reference.model_state[name].shape or mask.shape != delta.shape or not np.isfinite(delta).all() or not binary(mask):
            raise ValueError('Invalid delta/mask tensor')
        allowed = plan.parameter_permissions[name].copy()
        if name.startswith('fc2.'):
            allowed &= np.isin(np.arange(len(delta)), tuple(trained)).reshape((-1,) + (1,) * (delta.ndim - 1))
        if not packet.event.successful_steps:
            allowed[:] = False
        if np.any(mask.astype(bool) & ~allowed) or np.any(delta[~mask.astype(bool)] != 0):
            raise ValueError('Packet violates contributor permissions')
    return trained


@dataclass
class AggregationResult:
    state: object
    receipt: dict

    def broadcast(self):
        return broadcast(self.state)


def aggregate_round(plan, contributions, *, eta=1.0, bn_provider=None, acceptance=None, groups=None):
    """Build a shadow candidate; reference never mutates, including on failure."""
    plan.validate()
    if not np.isfinite(eta) or not 0 <= eta <= 1:
        raise ValueError('Aggregation eta must be in [0,1]')
    packets = tuple(sorted(contributions, key=lambda p: p.event.client_id))
    if len({p.event.client_id for p in packets}) != len(packets):
        raise ValueError('Duplicate client contribution')
    trained = {p.event.client_id: _validate_contribution(plan, p) for p in packets}
    if groups is not None:
        flat = [cid for group in groups for cid in group]
        if len(flat) != len(set(flat)) or set(flat) != set(trained) or any(not g for g in groups):
            raise ValueError('Hierarchy must partition every contributor exactly once')
    packet_groups = ([packets] if groups is None else
        [tuple(p for p in packets if p.event.client_id in group) for group in groups])
    candidate = restore(snapshot(plan.reference))
    coverage = {}
    for name in parameter_names(candidate):
        old = candidate.model_state[name]
        numerator, denominator = np.zeros(old.shape, np.float64), np.zeros(old.shape, np.float64)
        for group in packet_groups:
            # Only these sufficient statistics cross cluster boundaries. Never
            # average group means equally: clusters can have unequal support.
            group_sum, group_weight = np.zeros_like(numerator), np.zeros_like(denominator)
            for packet in group:
                counts = plan.base_counts[packet.event.client_id]
                actual_classes = trained[packet.event.client_id]
                if name.startswith('fc2.'):
                    weight = np.array([counts.get(c, 0) if c in actual_classes else 0 for c in range(len(old))], np.float64)
                    weight = weight.reshape((-1,) + (1,) * (old.ndim - 1))
                else:
                    weight = float(sum(counts[c] for c in actual_classes))
                weighted_mask = weight * array(packet.masks[name]).astype(bool)
                group_sum += weighted_mask * array(packet.deltas[name])
                group_weight += weighted_mask
            numerator += group_sum
            denominator += group_weight
        touched = denominator > 0
        result = array(old).copy()
        if eta:
            update = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=touched)
            result[touched] = result[touched] + eta * update[touched]
        candidate.model_state[name] = like(result, old)
        coverage[name] = dict(contributed_coordinates=int(touched.sum()), total_coordinates=int(touched.size))
    # All metadata/masks/ranks/router/BN remain canonical; no OR or per-worker aging.
    candidate.validate()
    candidate, bn_receipt = recalibrate_bn(candidate, plan, bn_provider)
    candidate.round = plan.target_round
    candidate.validate()
    if acceptance is not None:
        witness = restore(snapshot(candidate))
        passed, reason = acceptance(witness)
        if not passed:
            raise ValueError(f'AGGREGATION_ROLLBACK: {reason}')
        if witness.fingerprint() != candidate.fingerprint():
            raise ValueError('Acceptance callback mutated candidate')
    receipt = dict(policy='class_sample_count_masked_delta_v1', reference_hash=plan.reference_hash,
        plan_hash=plan.hash, task=candidate.task, round=candidate.round, eta=float(eta),
        contributors=tuple(p.event.client_id for p in packets), leader_weight_bonus=False,
        fc2_contributors={c: tuple(p.event.client_id for p in packets if c in trained[p.event.client_id])
                          for c in plan.current_classes}, parameter_coverage=coverage, bn=bn_receipt,
        result_model_hash=state_hash(candidate.model_state), result_state_hash=candidate.fingerprint())
    if groups is not None:
        receipt.update(policy='hierarchical_class_sample_count_masked_delta_v1',
            groups=tuple(tuple(g) for g in groups), shared_payload='weighted_delta_sums_and_coordinate_weights',
            raw_training_rows_shared=False)
    return AggregationResult(candidate, receipt)
