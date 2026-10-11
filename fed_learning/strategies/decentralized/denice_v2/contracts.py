"""Lossless cluster state contract. P0 validates declared routing dependencies;
P2 must establish that this dependency closure is functionally complete.
"""
import base64
from dataclasses import asdict, dataclass, field
import hashlib
import json
import re

import numpy as np
import torch


def encode(value):
    """Typed JSON codec, including integer keys and exact tensor bytes."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return ['tensor', str(tensor.dtype).removeprefix('torch.'), list(tensor.shape), base64.b64encode(raw).decode()]
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError('Object arrays are not checkpoint state')
        array = np.ascontiguousarray(value)
        return ['array', value.dtype.str, list(value.shape), base64.b64encode(array.tobytes()).decode()]
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, dict):
        pairs = [[encode(k), encode(v)] for k, v in value.items()]
        pairs.sort(key=lambda pair: json.dumps(pair[0], separators=(',', ':')))
        return ['dict', pairs]
    if isinstance(value, (tuple, list)):
        return ['tuple' if isinstance(value, tuple) else 'list', [encode(v) for v in value]]
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError('Nonfinite scalar metadata')
        return ['scalar', value]
    raise TypeError(f'Unsupported state value {type(value).__name__}')


def decode(value):
    kind = value[0]
    if kind == 'scalar':
        return value[1]
    if kind == 'dict':
        return {decode(k): decode(v) for k, v in value[1]}
    if kind in {'tuple', 'list'}:
        result = [decode(v) for v in value[1]]
        return tuple(result) if kind == 'tuple' else result
    raw = base64.b64decode(value[3], validate=True)
    if kind == 'array':
        dtype = np.dtype(value[1])
        if dtype.hasobject:
            raise ValueError('Object dtype forbidden')
        return np.frombuffer(raw, dtype=dtype).copy().reshape(value[2])
    if kind == 'tensor':
        dtype = getattr(torch, value[1], None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError('Unsupported tensor dtype')
        if not raw:
            if np.prod(value[2], dtype=np.int64) != 0:
                raise ValueError('Missing tensor bytes')
            return torch.empty(value[2], dtype=dtype)
        return torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy()).view(dtype).reshape(value[2])
    raise ValueError('Unknown checkpoint value kind')


def state_hash(value):
    return hashlib.sha256(json.dumps(encode(value), separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().numpy() if value.dtype == torch.bfloat16 else value.detach().cpu().numpy()
    return np.asarray(value)


def binary(value):
    a = array(value)
    return bool(np.isfinite(a).all() and np.isin(a, [0, 1]).all())


@dataclass
class ClusterState:
    cluster_uid: str
    task: int
    round: int
    members: tuple[int, ...]
    coordinator_id: int
    versions: dict
    model_state: dict
    model_spec: dict
    algorithm_state: dict
    router_state: dict
    bn_state: dict
    protected_parameters: dict
    routing_dependencies: tuple[str, ...]
    routing_space_hash: str
    parent_lineage: tuple[str, ...] = ()
    ownership_metadata: dict = field(default_factory=dict)
    training_rng_state: dict = field(default_factory=dict)
    schema_version: int = 1

    def payload(self):
        return asdict(self)

    def fingerprint(self):
        self.validate()
        return state_hash(self.payload())

    def routing_fingerprint(self):
        router = self.router_state
        return state_hash(dict(dependencies={k: self.model_state[k] for k in self.routing_dependencies},
            context_layers=tuple(router['context_layers']),
            weight_masks={k: self.algorithm_state['weight_masks'][k] for k in router['context_layers']},
            bias_masks={k: self.algorithm_state['bias_masks'][k] for k in router['context_layers']},
            thresholds=router['binarize_thresholds'], feature_mask=router['stable_feature_mask'],
            feature_version=self.versions['feature'], architecture=self.versions['architecture']))

    def validate(self):
        if self.schema_version != 1 or not isinstance(self.cluster_uid, str) or not self.cluster_uid or self.task < 0 or self.round < -1:
            raise ValueError('Invalid state schema/identity/time')
        if any(type(n) is not int for n in (self.task, self.round, self.coordinator_id, *self.members)):
            raise ValueError('State clocks and client IDs must be integers')
        if not self.members or tuple(sorted(set(self.members))) != self.members or any(c < 0 for c in self.members):
            raise ValueError('Invalid cluster members')
        if self.coordinator_id not in self.members:
            raise ValueError('Coordinator must be a member')
        if len(set(self.parent_lineage)) != len(self.parent_lineage) or any(not isinstance(p, str) or not p for p in self.parent_lineage):
            raise ValueError('Invalid lineage')
        if any(not self.versions.get(k) for k in ['architecture', 'feature', 'class_mapping']):
            raise ValueError('Missing version binding')
        mapping = self.versions.get('class_ids')
        if mapping != tuple(range(len(mapping or ()))) or not mapping:
            raise ValueError('Invalid class mapping')
        if not self.model_state or set(self.model_state) != set(self.model_spec):
            raise ValueError('Model state does not match declared model schema')
        for name, value in self.model_state.items():
            if not isinstance(value, (np.ndarray, torch.Tensor)) or not np.isfinite(array(value)).all():
                raise ValueError('Model tensors must be finite')
            if (tuple(value.shape), str(value.dtype)) != self.model_spec[name]:
                raise ValueError(f'Model shape/dtype mismatch: {name}')
        alg = self.algorithm_state
        for key in ['unit_ranks', 'weight_masks', 'bias_masks', 'freeze_masks']:
            if key not in alg:
                raise ValueError(f'Missing algorithm state: {key}')
        ranks = alg['unit_ranks']
        if set(ranks) != set(alg['weight_masks']) or set(ranks) != set(alg['bias_masks']):
            raise ValueError('Rank/mask layer sets differ')
        if not set(alg['freeze_masks']) <= set(ranks):
            raise ValueError('Unknown freeze-mask layer')
        for layer, values in ranks.items():
            age = array(values)
            if age.ndim != 1 or age.dtype.kind not in 'iu' or np.any(age < 0):
                raise ValueError('Neuron ranks must be nonnegative integer vectors')
            wm, bm = alg['weight_masks'][layer], alg['bias_masks'][layer]
            shape = (len(age),) if layer == 'gru' else tuple(self.model_state[layer + '.weight'].shape)
            if tuple(wm.shape) != shape or tuple(bm.shape) != (len(age),) or not binary(wm) or not binary(bm):
                raise ValueError(f'Invalid masks: {layer}')
            if layer != 'gru' and shape[0] != len(age):
                raise ValueError('Ranks do not match model output width')
            if layer == 'gru':
                params = [v for k, v in self.model_state.items() if re.fullmatch(r'gru\.(weight|bias)_(ih|hh)_l\d+', k)]
                if not params or any(v.shape[0] != 3 * len(age) for v in params):
                    raise ValueError('GRU ranks do not match gate dimensions')
            frozen = array(alg['freeze_masks'].get(layer, np.zeros(len(age), dtype=bool)))
            if frozen.shape != age.shape or not binary(frozen) or np.any((age >= 2) & ~frozen.astype(bool)):
                raise ValueError('Mature neurons must be frozen')
        if len(ranks.get('fc2', ())) != len(mapping):
            raise ValueError('Classifier/class mapping mismatch')
        expected_bn = {k for k in self.model_state if k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
        if set(self.bn_state) != expected_bn or any(state_hash(v) != state_hash(self.model_state[k]) for k, v in self.bn_state.items()):
            raise ValueError('BN buffers missing or inconsistent with model')
        for name, mask in self.protected_parameters.items():
            if name not in self.model_state or tuple(mask.shape) != tuple(self.model_state[name].shape) or not binary(mask):
                raise ValueError('Invalid protected-parameter mask')
        for name, value in self.model_state.items():
            layer = name.split('.')[0]
            if layer in ranks and (name.endswith(('weight', 'bias')) or re.fullmatch(r'gru\.(weight|bias)_(ih|hh)_l\d+', name)):
                mature = array(ranks[layer]) >= 2
                if layer == 'gru':
                    mature = np.tile(mature, 3)
                if mature.any():
                    mask = self.protected_parameters.get(name)
                    if mask is None or not np.all(array(mask)[mature] == 1):
                        raise ValueError('Missing mature-parameter protection')
        router = self.router_state
        allowed_router_fields = {'router_mode', 'context_layers', 'stable_feature_mask',
            'binarize_thresholds', 'activation_memory', 'episode_classes'}
        if set(router) != allowed_router_fields:
            raise ValueError('Unknown/missing router fields; raw reference data is not allowed')
        if router.get('router_mode') != 'binary_cosine':
            raise ValueError('P0 contract currently supports binary_cosine only')
        layers = tuple(router['context_layers'])
        if not layers or len(set(layers)) != len(layers) or not set(layers) <= set(ranks):
            raise ValueError('Invalid routing layers')
        width = sum(len(ranks[k]) for k in layers)
        mask = router['stable_feature_mask']
        if array(mask).shape != (width,) or not binary(mask):
            raise ValueError('Invalid stable feature mask')
        thresholds = router['binarize_thresholds']
        if thresholds is not None and (set(thresholds) != set(layers) or not all(np.isfinite(v) for v in thresholds.values())):
            raise ValueError('Invalid router thresholds')
        memory, classes = router['activation_memory'], router['episode_classes']
        if memory and thresholds is None:
            raise ValueError('Memory requires calibrated shared thresholds')
        for task, labels in classes.items():
            if not isinstance(task, int) or not 0 <= task <= self.task or len(set(labels)) != len(labels) or any(c not in mapping for c in labels):
                raise ValueError('Future/invalid episode class registration')
        for task, values in memory.items():
            if task not in classes or array(values).ndim != 2 or array(values).shape[1] != width or not binary(values):
                raise ValueError('Invalid activation memory')
            if np.any(array(values)[:, ~array(mask).astype(bool)]):
                raise ValueError('Memory uses excluded routing features')
        if not self.routing_dependencies or len(set(self.routing_dependencies)) != len(self.routing_dependencies) or not set(self.routing_dependencies) <= set(self.model_state):
            raise ValueError('Missing/invalid routing dependencies')
        if self.routing_space_hash != self.routing_fingerprint():
            raise ValueError('Routing dependency/threshold/mask version mismatch')
        encode(self.payload())  # Reject unsupported/opaque objects; no raw CAL fields in router schema.

    @classmethod
    def create(cls, **kwargs):
        model = kwargs['model_state']
        kwargs['model_spec'] = {k: (tuple(v.shape), str(v.dtype)) for k, v in model.items()}
        kwargs['routing_space_hash'] = ''
        state = cls(**kwargs)
        state.routing_space_hash = state.routing_fingerprint()
        state.validate()
        return cls(**decode(encode(state.payload())))  # Own every buffer; no alias to caller.
