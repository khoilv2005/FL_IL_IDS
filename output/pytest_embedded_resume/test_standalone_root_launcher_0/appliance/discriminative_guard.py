"""Experimental 16D linear patch guard fitted from class sufficient statistics.

BASE moments are training support, never CAL certification. Historical raw
examples and individual sketches are not retained. No production installer.
"""
import copy
import gzip
import hashlib
import json
import numpy as np
from .config import Rejected
from .current_base_data import CurrentBaseData
from .portable_route import SharedSketch
from .role_sketch_moments import _entry, _checked_entry
from .state import digest

VERSION = 'appliance_discriminative_sketch_ridge_v1'
RIDGE = .001  # Fixed before development evaluation; no parameter search.


class CurrentBaseSketchMoments:
    def __init__(self, view, sketch):
        if (not isinstance(view, CurrentBaseData) or not isinstance(sketch, SharedSketch)
                or sketch.dimension != 16 or tuple(view.store['input_shape']) != sketch.input_shape
                or view.store['metadata_sha256'] != sketch.preprocessing_sha256):
            raise Rejected('SCOPED_BASE_SKETCH_REQUIRED')
        self.owner, self.sketch = view.client_id, sketch
        self.role_sha = view.store['role_manifest_sha256']
        self.entries, self.events = {}, []

    def observe_current(self, view):
        if (not isinstance(view, CurrentBaseData) or view.client_id != self.owner
                or view.store['role_manifest_sha256'] != self.role_sha
                or view.store['metadata_sha256'] != self.sketch.preprocessing_sha256
                or tuple(view.store['input_shape']) != self.sketch.input_shape
                or (self.events and view.task != self.events[-1]['task'] + 1)):
            raise Rejected('BASE_MOMENT_SCOPE_CHANGED')
        pool = view.current_pool(self.owner, 'base', view.store['task_classes'][str(view.task)])
        z, valid = self.sketch.features(pool['X'])
        entries = copy.deepcopy(self.entries)
        for c in np.unique(pool['y']):
            if str(int(c)) in entries:
                raise Rejected('BASE_MOMENT_CLASS_REPLAY')
            ix = pool['y'] == c
            entries[str(int(c))] = dict(task=view.task, moments=_entry(z[ix], valid[ix]))
        event = dict(owner=self.owner, task=view.task, role='current owned BASE; not CAL',
                     role_manifest_sha256=self.role_sha, partition_sha256=pool['partition_sha256'],
                     rows_sha256=hashlib.sha256(pool['rows'].astype('<i8').tobytes()).hexdigest(),
                     class_counts={str(int(c)): int((pool['y'] == c).sum()) for c in np.unique(pool['y'])})
        event['event_digest'] = digest(event)
        self.entries, self.events = entries, self.events + [event]

    def packet(self, classes, receiver, source_version):
        selected = sorted(set(map(int, classes)))
        if not selected or any(str(c) not in self.entries for c in selected):
            raise Rejected('MISSING_BASE_NEGATIVE_MOMENTS')
        body = dict(version=VERSION, owner=self.owner, receiver=receiver,
                    source_version=source_version, sketch=self.sketch.manifest(), role_sha=self.role_sha,
                    entries={str(c): self.entries[str(c)] for c in selected}, events=self.events,
                    retained_examples=0, retained_individual_features=0, substitutes_CAL=False)
        body['state_digest'] = digest(body)
        return gzip.compress(json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode(), mtime=0)


def read_negative_packet(packet, owner, receiver, authorized, source_version, sketch, role_sha, task):
    if owner not in authorized or owner == receiver:
        raise Rejected('UNAUTHORIZED_NEGATIVE_SOURCE')
    try:
        body = json.loads(gzip.decompress(packet))
    except (ValueError, OSError) as exc:
        raise Rejected('INVALID_NEGATIVE_PACKET') from exc
    required = {'version', 'owner', 'receiver', 'source_version', 'sketch', 'role_sha', 'entries',
                'events', 'retained_examples', 'retained_individual_features', 'substitutes_CAL', 'state_digest'}
    if (set(body) != required or body['version'] != VERSION or body['owner'] != owner
            or body['receiver'] != receiver or body['source_version'] != source_version
            or body['sketch'] != sketch.manifest() or body['role_sha'] != role_sha
            or body['retained_examples'] != 0 or body['retained_individual_features'] != 0
            or body['substitutes_CAL'] is not False
            or body['state_digest'] != digest({k: v for k, v in body.items() if k != 'state_digest'})):
        raise Rejected('NEGATIVE_PACKET_BINDING_CHANGED')
    declared = {}
    for i, e in enumerate(body['events']):
        if (set(e) != {'owner','task','role','role_manifest_sha256','partition_sha256',
                       'rows_sha256','class_counts','event_digest'}
                or e['owner'] != owner or e['task'] != i or i > task
                or e['role'] != 'current owned BASE; not CAL' or e['role_manifest_sha256'] != role_sha
                or e['event_digest'] != digest({k: v for k, v in e.items() if k != 'event_digest'})):
            raise Rejected('NEGATIVE_EVENT_CHANGED')
        for c, n in e['class_counts'].items():
            if c in declared or type(n) is not int or n <= 0:
                raise Rejected('NEGATIVE_COUNTS_CHANGED')
            declared[c] = (i, n)
    if not body['entries']:
        raise Rejected('EMPTY_NEGATIVE_MOMENTS')
    for c, e in body['entries'].items():
        if str(int(c)) != c or not 0 <= int(c) < 34 or set(e) != {'task', 'moments'}:
            raise Rejected('NEGATIVE_CLASS_CHANGED')
        n, usable, _, _ = _checked_entry(e['moments'], sketch.dimension)
        if declared.get(c) != (e['task'], n) or usable < 1:
            raise Rejected('NEGATIVE_MOMENT_PROVENANCE_CHANGED')
    return body


def fit_linear_guard(positive_features, positive_valid, negative_packets, binding):
    """Balanced ridge least squares, exact from n/mean/scatter, not logistic LR.

    Positive side mass=1/2; negative side mass=1/2, equal across classes.
    Within a negative class, sources are weighted by usable BASE row count.
    No synthetic samples, Gaussian draws, raw negative replay, or CAL labels.
    """
    pos = _entry(positive_features, positive_valid)
    _, n, pm, ps = _checked_entry(pos, 16)
    if n < 32 or not negative_packets:
        raise Rejected('INSUFFICIENT_DISCRIMINATIVE_FIT')
    groups = {}
    for p in negative_packets:
        if (p.get('version') != VERSION or p.get('role_sha') != binding['role_sha']
                or p.get('sketch') != binding['sketch'] or p.get('receiver') != binding['receiver']
                or p.get('owner') not in binding['authorized_sources']
                or p.get('state_digest') != digest({k:v for k,v in p.items() if k!='state_digest'})
                or p.get('substitutes_CAL') is not False):
            raise Rejected('NEGATIVE_FIT_AUTHORITY_CHANGED')
        for c, e in p['entries'].items():
            if int(c) == binding['class_id']:
                raise Rejected('POSITIVE_CLASS_IN_NEGATIVES')
            _, count, mean, scatter = _checked_entry(e['moments'], 16)
            groups.setdefault(c, []).append((count, mean, scatter))
    components = [(.5, pm, ps/n, 1.)]
    for values in groups.values():
        total = sum(v[0] for v in values)
        components += [(.5/len(groups)*count/total, mean, scatter/count, -1.)
                       for count, mean, scatter in values]
    mean = sum(mass * mu for mass, mu, _, _ in components)
    second = sum(mass * (cov + np.outer(mu, mu)) for mass, mu, cov, _ in components)
    scale = np.sqrt(np.maximum(np.diag(second) - mean**2, 1e-12))
    gram, rhs = np.zeros((17, 17)), np.zeros(17)
    for mass, mu, cov, label in components:
        u = (mu-mean)/scale
        block = np.zeros((17, 17)); block[:16, :16] = cov / np.outer(scale, scale) + np.outer(u, u)
        block[:16, 16] = block[16, :16] = u; block[16, 16] = 1
        gram += mass*block; rhs += mass*label*np.r_[u, 1.]
    penalty = np.diag(np.r_[np.full(16, RIDGE), 0.])
    coef = np.linalg.solve(gram+penalty, rhs)
    weight = (coef[:16]/scale).astype(np.float32)
    bias = np.float32(coef[16] - mean @ (coef[:16]/scale))
    body = dict(version=VERSION, method='balanced standardized ridge least squares from sufficient statistics',
                ridge=RIDGE, weight=weight.tolist(), bias=float(bias), binding=copy.deepcopy(binding),
                positive_valid_rows=n, negative_classes=sorted(map(int, groups)),
                negative_packet_digests=[p['state_digest'] for p in negative_packets],
                positive_moments_digest=digest(pos), individual_negative_features_retained=0,
                inference_requires_labels=False, score='tanh(linear), ranking score; not calibrated probability')
    body['state_digest'] = digest(body)
    return LinearGuard(body)


class LinearGuard:
    def __init__(self, state):
        body = copy.deepcopy(state)
        if (body.get('version') != VERSION or body.get('ridge') != RIDGE
                or body.get('state_digest') != digest({k:v for k,v in body.items() if k!='state_digest'})
                or np.asarray(body.get('weight')).shape != (16,)
                or not np.isfinite(body['weight']).all() or not np.isfinite(body['bias'])
                or body.get('inference_requires_labels') is not False):
            raise Rejected('DISCRIMINATIVE_GUARD_STATE_CHANGED')
        self.state = body

    def score(self, x, sketch, binding):
        if binding != self.state['binding'] or sketch.manifest() != binding['sketch']:
            raise Rejected('DISCRIMINATIVE_GUARD_BINDING_CHANGED')
        z, valid = sketch.features(x)
        raw = z @ np.asarray(self.state['weight'], np.float32) + np.float32(self.state['bias'])
        return np.tanh(raw.astype(np.float64)), valid
