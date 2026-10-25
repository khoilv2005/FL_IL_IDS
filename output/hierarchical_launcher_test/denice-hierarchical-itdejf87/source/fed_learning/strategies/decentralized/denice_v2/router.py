"""P2 canonical binary-cosine router proposals, with a capacity commit gate.

The P0/P1 dependency contract freezes complete named tensors. Its conservative
native closure is explicit: selected conv features include upstream Conv/BN;
any selected GRU feature includes both recurrent layers. No historical raw bank.
"""
from dataclasses import dataclass
import json

import numpy as np
import torch

from fed_learning.servers.nice_server import ContextDetector
from .aggregation import _validate_contribution, parameter_names, permissions, topology_hash
from .contracts import ClusterState, array, decode, encode, state_hash, routing_parameter_mask
from .native import load_native, _validate_base
from .normalization import model_hash
from .state import restore, snapshot


LAYERS = ('conv1', 'conv2', 'conv3', 'gru')
CAPACITY_POLICY = 'whole_dependency_max_backbone_lock_0.90_v1'
MAX_BACKBONE_LOCK = .90  # Engineering gate locked before the real-data probe.


def allocated_feature_mask(state, feature_policy='allocated_native_v1'):
    """Baseline mask: allocated native context units, fixed on first memory."""
    if feature_policy not in ('allocated_native_v1', 'conv1_only_v1','cnn_coordinate_router_v1'):
        raise ValueError('Unknown routing feature policy')
    if feature_policy=='cnn_coordinate_router_v1':
        target=state.algorithm_state['coordinate_router_layer']
        return np.concatenate([array(state.algorithm_state['unit_ranks'][k])>0 if k==target
            else np.zeros_like(array(state.algorithm_state['unit_ranks'][k]),dtype=bool) for k in LAYERS])
    return np.concatenate([array(state.algorithm_state['unit_ranks'][k]) > 0
        if feature_policy == 'allocated_native_v1' or k == 'conv1'
        else np.zeros_like(array(state.algorithm_state['unit_ranks'][k]), dtype=bool) for k in LAYERS])


def dependency_closure(state, feature_mask):
    """Whole-tensor closure, not a claim of minimal coordinate-level closure."""
    state.validate()
    if state.algorithm_state.get('routing_contract_version')=='cnn_coordinate_router_v1':
        from .coordinate_router import cnn_closure
        closure=cnn_closure(state,feature_mask)
        return tuple(sorted(closure['parameters'])),closure['selected']
    if tuple(state.router_state['context_layers']) != LAYERS or state.versions['architecture'] != 'denice_legacy_plain_v1':
        raise ValueError('Unsupported routing architecture/layers')
    mask = np.asarray(feature_mask)
    width = sum(len(state.algorithm_state['unit_ranks'][k]) for k in LAYERS)
    if mask.shape != (width,) or not np.isin(mask, [0, 1]).all() or not mask.any():
        raise ValueError('Empty/invalid routing feature mask')
    names, offset = set(), 0
    selected = {}
    for layer in LAYERS:
        size = len(state.algorithm_state['unit_ranks'][layer])
        selected[layer] = mask[offset:offset+size].astype(bool)
        offset += size
    deepest = max((i for i, layer in enumerate(LAYERS[:3]) if selected[layer].any()), default=-1)
    for i in range(deepest + 1):
        names.update(k for k in state.model_state if k.startswith((f'conv{i+1}.', f'bn{i+1}.')))
    if selected['gru'].any():
        # Dense recurrence mixes hidden units and layer1 depends on layer0.
        # Even sparse recurrence is conservatively frozen in this contract.
        names.update(k for k in state.model_state if k.startswith('gru.'))
    return tuple(sorted(names)), {k: int(v.sum()) for k, v in selected.items()}


def capacity_receipt(state, dependencies):
    names = parameter_names(state)
    backbone = [k for k in names if k.startswith(('conv', 'bn', 'gru.'))]
    total = sum(state.model_state[k].numel() for k in names)
    backbone_total = sum(state.model_state[k].numel() for k in backbone)
    dependencies = set(dependencies) | set(state.routing_dependencies)
    protected = {}
    for k in names:
        mask = np.asarray(array(state.protected_parameters.get(k, torch.zeros_like(state.model_state[k], dtype=torch.bool))), bool).copy()
        layer = k.split('.')[0]
        if layer in state.algorithm_state['freeze_masks']:
            frozen = np.asarray(state.algorithm_state['freeze_masks'][layer], dtype=bool)
            if layer == 'gru':
                frozen = np.tile(frozen, 3)
            mask |= np.broadcast_to(frozen.reshape((-1,) + (1,) * (mask.ndim - 1)), mask.shape)
        if k in dependencies:
            mask |= routing_parameter_mask(state,k) if k in state.routing_dependencies else np.ones(mask.shape,bool)
        if k.startswith('bn'):
            from .normalization import protected_bn_channels
            mask |= protected_bn_channels(state, k.split('.')[0])
        protected[k] = mask
    locked = sum(int(protected[k].sum()) for k in backbone)
    # Compare canonical permissions before and after the proposed closure;
    # rank0 reserve is not mislabeled as currently trainable.
    allowed=permissions(state)
    before = sum(int(m.sum()) for m in allowed.values())
    remaining={k:m & ~(routing_parameter_mask(state,k) if k in state.routing_dependencies else np.ones(m.shape,bool))
        if k in dependencies else m for k,m in allowed.items()}
    after = sum(int(m.sum()) for m in remaining.values())
    ratio = locked / backbone_total
    passed = ratio < MAX_BACKBONE_LOCK and after > 0
    return dict(policy=CAPACITY_POLICY, max_backbone_locked_fraction=MAX_BACKBONE_LOCK,
        passed=passed, decision='GO' if passed else 'NO_GO',
        reason=None if passed else 'BACKBONE_CLOSURE_TOO_LARGE' if ratio >= MAX_BACKBONE_LOCK else 'NO_TRAINABLE_PARAMETERS',
        total_parameters=total, backbone_parameters=backbone_total, locked_backbone_parameters=locked,
        locked_backbone_fraction=ratio, remaining_backbone_parameters=backbone_total-locked,
        currently_trainable_before=before, currently_trainable_after=after,
        currently_trainable_after_fraction=after / total,
        per_layer={layer: dict(total=sum(state.model_state[k].numel() for k in names if k.split('.')[0] == layer),
            closure_locked=sum(int((routing_parameter_mask(state,k) if k in state.routing_dependencies else
                np.ones(state.model_state[k].shape,bool)).sum()) for k in names if k.split('.')[0] == layer and k in dependencies),
            protected=sum(int(protected[k].sum()) for k in names if k.split('.')[0] == layer),
            currently_trainable_after=sum(int(m.sum()) for k, m in remaining.items() if k.split('.')[0] == layer))
            for layer in ('conv1', 'conv2', 'conv3', 'gru', 'bn1', 'bn2', 'bn3', 'fc1', 'fc2')},
        closure_granularity=('CNN_output_row_coordinates' if state.algorithm_state.get('routing_contract_version')=='cnn_coordinate_router_v1'
                            else 'complete_named_tensors'), classifier_learning_may_remain=True)


def probe_router_capacity(state, feature_policy='allocated_native_v1'):
    _check_feature_policy(state, feature_policy)
    mask = (np.asarray(state.router_state['stable_feature_mask'], bool) if state.router_state['activation_memory']
            else allocated_feature_mask(state, feature_policy))
    if feature_policy=='cnn_coordinate_router_v1':
        from .coordinate_router import protect_coordinate_closure
        proposed,closure=protect_coordinate_closure(state,mask)
        capacity=capacity_receipt(proposed,())
        capacity['currently_trainable_before']=sum(int(m.sum()) for m in permissions(state).values())
        return dict(selected_features=closure['selected'],selected_total=int(mask.sum()),
            dependencies=proposed.routing_dependencies,capacity=capacity,
            upstream_channels={k:int(v.sum()) for k,v in closure['channels'].items()})
    deps, selected = dependency_closure(state, mask)
    deps = tuple(sorted(set(deps) | set(state.routing_dependencies)))
    return dict(selected_features=selected, selected_total=int(mask.sum()), dependencies=deps,
                capacity=capacity_receipt(state, deps))


def _check_feature_policy(state, policy):
    if policy=='cnn_coordinate_router_v1' and state.algorithm_state.get('routing_contract_version')!='cnn_coordinate_router_v1':
        raise ValueError('Coordinate routing requires a fresh versioned state')
    allocated_feature_mask(state, policy)  # Validate even for historical state.
    if state.router_state['activation_memory']:
        existing = state.algorithm_state.get('routing_contract_version', 'whole_native_closure_v1')
        expected = policy if policy in ('conv1_only_v1','cnn_coordinate_router_v1') else 'whole_native_closure_v1'
        if existing != expected:
            raise ValueError('Cannot migrate a committed routing policy implicitly')
        if policy == 'conv1_only_v1' and np.asarray(state.router_state['stable_feature_mask'])[64:].any():
            raise ValueError('Conv1-only contract contains deep features')


@dataclass(frozen=True)
class RoutingMoment:
    client_id: int
    task: int
    round: int
    plan_hash: str
    model_hash: str
    feature_mask_hash: str
    role_hash: str
    sample_rows: int
    moments: dict  # layer: (count, sum, squared_sum), no raw inputs.


@dataclass(frozen=True)
class SketchPacket:
    client_id: int
    task: int
    round: int
    plan_hash: str
    model_hash: str
    routing_hash: str
    role_hash: str
    class_counts: dict
    bits: np.ndarray


def _acts(model, x, batch_size, device):
    chunks = {k: [] for k in LAYERS}
    for batch in x.split(batch_size):
        values = model.get_context_activations_per_sample(batch.to(device))
        for k in LAYERS:
            chunks[k].append(values[k].detach().cpu().numpy())
    return {k: np.concatenate(v) for k, v in chunks.items()}


def collect_routing_moments(state, plan, base, mask, *, batch_size=16, device='cpu'):
    _validate_base(plan, base)
    model = load_native(state, device).eval()
    moments = {k: (0, 0., 0.) for k in LAYERS}
    for batch in base.x.split(batch_size):
        values = _acts(model, batch, batch_size, device)
        offset = 0
        for k in LAYERS:
            n = values[k].shape[1]
            selected = np.asarray(mask[offset:offset+n], bool)
            z = values[k][:, selected].astype(np.float64)
            count, total, squared = moments[k]
            moments[k] = (count+int(z.size), total+float(z.sum()), squared+float((z*z).sum()))
            offset += n
    return RoutingMoment(base.client_id, state.task, state.round, plan.hash, model_hash(state),
                         state_hash(mask), plan.base_role_hash, len(base.y), moments)


def aggregate_thresholds(packets, state, plan, mask):
    ids = [p.client_id for p in packets]
    if tuple(sorted(ids)) != tuple(sorted(cid for cid, counts in plan.base_counts.items() if sum(counts.values()))):
        raise ValueError('Missing/duplicate threshold contributor')
    threshold = {}
    offset = 0
    for k in LAYERS:
        width = len(state.algorithm_state['unit_ranks'][k])
        selected_count = int(np.asarray(mask[offset:offset+width], bool).sum())
        offset += width
        count, total, squared = 0, 0., 0.
        for p in sorted(packets, key=lambda p: p.client_id):
            if (p.task != state.task or p.round != state.round or p.plan_hash != plan.hash
                or p.model_hash != model_hash(state) or p.feature_mask_hash != state_hash(mask)
                or p.role_hash != plan.base_role_hash or set(p.moments) != set(LAYERS)
                or p.sample_rows != sum(plan.base_counts[p.client_id].values())):
                raise ValueError('Stale/noncurrent routing moments')
            n, s, sq = p.moments[k]
            if (type(n) is not int or n != p.sample_rows*selected_count or not np.isfinite([s, sq]).all()
                or sq < 0 or (n == 0 and (s != 0 or sq != 0))
                or (n > 0 and sq - s*s/n < -1e-8*max(1, sq))):
                raise ValueError('Invalid routing moments')
            count += n
            total += s
            squared += sq
        mean = total/count if count else 0.
        var = max(0., squared-total*total/count)/(count-1) if count > 1 else 0.
        threshold[k] = float(mean + np.sqrt(var))
    return threshold


def collect_sketch(state, plan, base, trained_classes, *, memo_per_class=8, batch_size=16, device='cpu'):
    _validate_base(plan, base)
    selected = torch.cat([torch.where(base.y == c)[0][:memo_per_class] for c in sorted(trained_classes)])
    if not len(selected):
        raise ValueError('No actual trained-class BASE sketches')
    model = load_native(state, device).eval()
    values = _acts(model, base.x[selected], batch_size, device)
    thresholds = state.router_state['binarize_thresholds']
    bits = np.concatenate([(values[k] > thresholds[k]).astype(np.uint8) for k in LAYERS], axis=1)
    bits[:, ~np.asarray(state.router_state['stable_feature_mask'], bool)] = 0
    labels, counts = torch.unique(base.y[selected], return_counts=True)
    return SketchPacket(base.client_id, state.task, state.round, plan.hash, model_hash(state),
        state.routing_space_hash, plan.base_role_hash, {int(c): int(n) for c, n in zip(labels, counts)}, bits)


def merge_sketches(state, plan, packets, actual_classes, memo_per_class, *, engineering_baseline=False):
    ids = [p.client_id for p in packets]
    expected = {cid for cid, labels in actual_classes.items() if labels}
    if len(set(ids)) != len(ids) or set(ids) != expected:
        raise ValueError('Missing/duplicate sketch contributor')
    width = len(state.router_state['stable_feature_mask'])
    for p in packets:
        if (p.task != state.task or p.round != state.round or p.plan_hash != plan.hash or p.model_hash != model_hash(state)
            or p.routing_hash != state.routing_space_hash or p.role_hash != plan.base_role_hash
            or set(p.class_counts) != set(actual_classes[p.client_id])
            or any(type(n) is not int or n != min(memo_per_class, plan.base_counts[p.client_id][c]) for c, n in p.class_counts.items())
            or p.bits.shape != (sum(p.class_counts.values()), width) or p.bits.dtype != np.uint8
            or not np.isin(p.bits, [0, 1]).all() or np.any(p.bits[:, ~np.asarray(state.router_state['stable_feature_mask'], bool)])):
            raise ValueError('Incompatible/stale/unauthorized sketch packet')
    if not packets:
        raise ValueError('Empty current router bank')
    bits = np.concatenate([p.bits for p in sorted(packets, key=lambda p: p.client_id)])
    if not bits.any() and not engineering_baseline:
        raise ValueError('DEGENERATE_BINARY_MEMORY')
    labels = tuple(sorted({c for p in packets for c in p.class_counts}))
    return bits, labels


@dataclass
class RouterProposal:
    state: object
    receipt: dict
    binding: str

    def commit(self, *, engineering_baseline=False):
        if self.binding != state_hash(dict(state=self.state.fingerprint(), receipt=self.receipt)):
            raise ValueError('Router proposal mutated')
        if not self.receipt['capacity']['passed'] and not engineering_baseline:
            raise ValueError('P2_NO_GO: ' + self.receipt['capacity']['reason'])
        return restore(snapshot(self.state))


def stage_cluster_router(state, plan, contributions, bases, *, memo_per_class=8, batch_size=16, device='cpu',
                         feature_policy='allocated_native_v1', engineering_baseline=False):
    """Build a shadow only. NO-GO proposals cannot commit; no history replay."""
    state.validate()
    plan.validate()
    if (state.task != plan.reference.task or state.round != plan.target_round or state.members != plan.reference.members
        or state.cluster_uid != plan.reference.cluster_uid or topology_hash(state) != plan.topology_hash
        or type(memo_per_class) is not int or memo_per_class < 1 or type(batch_size) is not int or batch_size < 1):
        raise ValueError('Wrong router round/configuration')
    if state_hash({cid: b.fingerprint() for cid, b in bases.items()}) != plan.base_role_hash:
        raise ValueError('BASE content/authority differs from trained round')
    actual = {}
    for packet in contributions:
        if packet.event.client_id in actual:
            raise ValueError('Duplicate trained contributor')
        labels = _validate_contribution(plan, packet)
        actual[packet.event.client_id] = tuple(labels)
    if set(actual) != set(state.members):
        raise ValueError('Missing training receipts')
    historical = state.router_state['activation_memory']
    if historical and state.task < max(historical):
        raise ValueError('Future router bank at current task')
    probe = probe_router_capacity(state, feature_policy)
    payload = snapshot(state)
    if not historical:
        if feature_policy=='cnn_coordinate_router_v1':
            from .coordinate_router import protect_coordinate_closure
            staging,_=protect_coordinate_closure(state,allocated_feature_mask(state,feature_policy))
            payload=snapshot(staging)
        else:
            payload['router_state']['stable_feature_mask'] = allocated_feature_mask(state, feature_policy)
            payload['routing_dependencies'] = tuple(sorted(set(state.routing_dependencies) | set(probe['dependencies'])))
            for name in payload['routing_dependencies']:
                payload['protected_parameters'][name] = torch.ones_like(state.model_state[name], dtype=torch.bool)
            payload['versions']['feature'] = ('p2_conv1_only_binary_v1' if feature_policy == 'conv1_only_v1'
                                               else 'p2_whole_dependency_binary_v1')
            payload['algorithm_state']['routing_contract_version'] = ('conv1_only_v1' if feature_policy == 'conv1_only_v1'
                                                                      else 'whole_native_closure_v1')
    else:
        if not set(probe['dependencies']) <= set(state.routing_dependencies):
            raise ValueError('Historical dependency closure incomplete')
    staging = ClusterState.create(**{k: v for k, v in payload.items() if k not in ('model_spec', 'routing_space_hash')})
    first = not historical
    moment_packets = []
    # Full-state download accounting for ephemeral statistic/sketch workers.
    import json
    def state_bytes(s):
        return len(json.dumps(encode(snapshot(s)),sort_keys=True,separators=(',',':')).encode())
    threshold_download_bytes = state_bytes(staging) * sum(cid != state.coordinator_id and bool(len(base.y)) for cid,base in bases.items()) if first else 0
    if first:
        mask = staging.router_state['stable_feature_mask']
        moment_packets = [collect_routing_moments(staging, plan, base, mask, batch_size=batch_size, device=device)
                          for cid, base in sorted(bases.items()) if len(base.y)]
        payload = snapshot(staging)
        payload['router_state']['binarize_thresholds'] = aggregate_thresholds(moment_packets, staging, plan, mask)
        staging = ClusterState.create(**{k: v for k, v in payload.items() if k not in ('model_spec', 'routing_space_hash')})
    sketch_download_bytes = state_bytes(staging) * sum(cid != state.coordinator_id and bool(labels) for cid,labels in actual.items())
    sketches = [collect_sketch(staging, plan, bases[cid], labels, memo_per_class=memo_per_class,
                              batch_size=batch_size, device=device) for cid, labels in sorted(actual.items()) if labels]
    bits, labels = merge_sketches(staging, plan, sketches, actual, memo_per_class,
        engineering_baseline=engineering_baseline)
    payload = snapshot(staging)
    payload['router_state']['activation_memory'][state.task] = bits
    payload['router_state']['episode_classes'][state.task] = tuple(sorted(set(labels) |
        set(payload['router_state']['episode_classes'].get(state.task, ()))))
    if feature_policy=='cnn_coordinate_router_v1':
        payload['algorithm_state']['learned_classes_by_task'][state.task]=payload['router_state']['episode_classes'][state.task]
        payload['algorithm_state']['seen_classes']=tuple(sorted({c for cs in payload['router_state']['episode_classes'].values() for c in cs}))
    candidate = ClusterState.create(**{k: v for k, v in payload.items() if k not in ('model_spec', 'routing_space_hash')})
    receipt = dict(task=state.task, round=state.round, reference_state_hash=state.fingerprint(),
        reference_download_bytes=threshold_download_bytes+sketch_download_bytes,
        result_state_hash=candidate.fingerprint(), feature_policy=feature_policy,
        capacity=probe['capacity'], selected_features=probe['selected_features'],
        threshold_policy='first_current_BASE_global_mean_plus_unbiased_std', threshold_calibrated=first,
        thresholds=candidate.router_state['binarize_thresholds'], dependency_hash=candidate.routing_space_hash,
        memory_rows=len(bits), trained_classes=labels, sketch_contributors=tuple(p.client_id for p in sketches),
        degenerate_binary_memory=not bool(bits.any()), engineering_baseline=engineering_baseline,
        historical_memory_unchanged=all(state_hash(bank) == state_hash(candidate.router_state['activation_memory'][t])
                                        for t, bank in historical.items() if t < state.task),
        historical_raw_read=False, raw_reference_bank=False, model_weights_changed=False,
        wire_format='typed_json_utf8', wire_scope='moment_and_sketch_uploads_only_excludes_broadcast_and_native_traffic',
        threshold_wire_bytes=sum(len(json.dumps(encode(p.__dict__), sort_keys=True, separators=(',', ':')).encode()) for p in moment_packets),
        threshold_wire_bytes_by_client={p.client_id:len(json.dumps(encode(p.__dict__),sort_keys=True,separators=(',',':')).encode()) for p in moment_packets},
        sketch_wire_bytes_by_client={p.client_id:len(json.dumps(encode(p.__dict__),sort_keys=True,separators=(',',':')).encode()) for p in sketches},
        sketch_wire_bytes=sum(len(json.dumps(encode(p.__dict__), sort_keys=True, separators=(',', ':')).encode()) for p in sketches))
    return RouterProposal(candidate, receipt, state_hash(dict(state=candidate.fingerprint(), receipt=receipt)))


class ClusterRouter:
    """One native predictor: x -> context bits -> task -> registered classes."""
    def __init__(self, state, device='cpu'):
        self.state = restore(snapshot(state))
        router = self.state.router_state
        if self.state.algorithm_state.get('routing_contract_version') == 'conv1_only_v1':
            _check_feature_policy(self.state, 'conv1_only_v1')
        if self.state.algorithm_state.get('routing_contract_version')=='cnn_coordinate_router_v1':
            _check_feature_policy(self.state,'cnn_coordinate_router_v1')
        if not router['activation_memory'] or router['binarize_thresholds'] is None:
            raise ValueError('Router is not built')
        deps, _ = dependency_closure(self.state, router['stable_feature_mask'])
        if not set(deps) <= set(self.state.routing_dependencies):
            raise ValueError('Incomplete routing dependency closure')
        self.model = load_native(self.state, device).eval()
        self.device = device
        self.detector = ContextDetector(router_mode='binary_cosine')
        self.detector.retain_reference_inputs = False
        for key in ('activation_memory', 'episode_classes', 'binarize_thresholds', 'stable_feature_mask'):
            setattr(self.detector, key, decode(encode(router[key])))

    def predict(self, x):
        """No y or task identifier is accepted at inference."""
        self.state.validate()
        with torch.no_grad():
            logits, acts = self.model.get_output_and_context_activations(x.to(self.device))
        values = {k: v.detach().cpu().numpy() for k, v in acts.items()}
        bits = self.detector.binarize_layer_activations(values)
        tasks, scores = self.detector.predict_episodes_with_scores(bits)
        predictions = np.empty(len(x), np.int64)
        raw = logits.detach().cpu().numpy()
        for task in sorted(set(tasks.tolist())):
            labels = self.state.router_state['episode_classes'][task]
            if not labels:
                raise ValueError('Chosen episode has no registered classes')
            rows = np.flatnonzero(tasks == task)
            predictions[rows] = np.asarray(labels)[raw[np.ix_(rows, labels)].argmax(axis=1)]
        return predictions, tasks, scores, bits
