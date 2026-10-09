"""Fixed-sketch class moments for finite recorded-role activation bounds.

These summaries do not certify unseen-population FAR or authorize installation.
Cantelli is applied to the uniform finite set of recorded valid sketch rows.
A FP32 dot-product allowance makes the threshold conservative. No Gaussian
density assumption and no historical samples are needed for a later query.
"""
import copy
import hashlib
import math
import numpy as np
from .config import Rejected
from .current_calibration_data import CurrentCalibrationData
from .imported_route import ROUTE_RULES, stratified_roles
from .portable_route import SharedSketch
from .state import digest

VERSION = 'appliance_current_role_sketch_full_moments_v1'
ROLES = ('fit', 'selection', 'holdout')
FEATURE_BOUND = 1.0001


def _entry(features, valid):
    z = np.asarray(features, np.float32)
    valid = np.asarray(valid, bool)
    if z.ndim != 2 or valid.shape != (len(z),) or not np.isfinite(z).all():
        raise Rejected('INVALID_SKETCH_MOMENT_INPUT')
    values = z[valid].astype(np.float64)
    if values.size and (np.abs(values).max() > FEATURE_BOUND or
                       np.linalg.norm(values, axis=1).max() > FEATURE_BOUND):
        raise Rejected('NONUNIT_SKETCH_MOMENT_INPUT')
    d = z.shape[1]
    mean = values.mean(axis=0) if len(values) else np.zeros(d)
    centered = values - mean
    scatter = centered.T @ centered
    return dict(rows=len(z), valid_rows=len(values), mean=mean.tolist(),
                scatter=scatter.tolist())


def _checked_entry(entry, d):
    if set(entry) != {'rows', 'valid_rows', 'mean', 'scatter'}:
        raise Rejected('SKETCH_MOMENT_SCHEMA_CHANGED')
    n, valid = entry['rows'], entry['valid_rows']
    mean, scatter = np.asarray(entry['mean'], np.float64), np.asarray(entry['scatter'], np.float64)
    if (type(n) is not int or type(valid) is not int or not 0 <= valid <= n or n < 1 or
            mean.shape != (d,) or scatter.shape != (d, d) or
            not np.isfinite(mean).all() or not np.isfinite(scatter).all() or
            np.linalg.norm(mean) > FEATURE_BOUND or
            not np.allclose(scatter, scatter.T, rtol=0, atol=1e-10 * max(1, valid)) or
            np.linalg.eigvalsh((scatter + scatter.T) / 2).min() < -1e-10 * max(1, valid) or
            np.trace(scatter) > valid * FEATURE_BOUND**2 + 1e-10 * max(1, valid)):
        raise Rejected('INVALID_SKETCH_MOMENTS')
    if not valid and (np.any(mean) or np.any(scatter)):
        raise Rejected('INVALID_EMPTY_SKETCH_MOMENTS')
    return n, valid, mean, scatter


def finite_activation_bound(entry, prototype, tau):
    """Bound count(score_FP32 > tau) in the summarized finite role only.

    Mathematical argument: FP32 activation implies exact real dot > tau-e.
    For a uniform recorded row and a>mean, Markov applied to
    (s-mean + variance/(a-mean))**2 gives Cantelli
    P(s>a) <= variance/(variance+(a-mean)**2).
    Invalid sketch rows never activate. This is not a sampling-confidence bound.
    """
    w = np.asarray(prototype)
    if (w.dtype != np.float32 or w.ndim != 1 or not np.isfinite(w).all() or
            not np.isclose(np.linalg.norm(w.astype(np.float64)), 1, rtol=1e-5, atol=1e-6) or
            isinstance(tau, bool) or not np.isfinite(tau) or not -1 <= tau <= 1):
        raise Rejected('INVALID_FINITE_ACTIVATION_QUERY')
    n, count, mean, scatter = _checked_entry(entry, len(w))
    wf = w.astype(np.float64)
    # Covers separate multiply/add FP32, not just fused dot. No fast-math/FP16
    # authorization is inferred from this diagnostic numeric contract.
    eps = np.finfo(np.float32).eps
    operations = 2 * len(w) + 2
    gamma = operations * eps / (1 - operations * eps)
    dot_error = gamma * FEATURE_BOUND * float(np.abs(wf).sum())
    numeric_padding = 1e-10 * (1 + count)
    projected_mean = float(mean @ wf)
    variance = max(0., float(wf @ scatter @ wf) / max(count, 1)) + numeric_padding
    lowered = float(tau) - dot_error - numeric_padding
    if tau == 1 or not count:
        fraction, upper = 0., 0
    elif lowered <= projected_mean:
        fraction, upper = 1., count
    else:
        delta = lowered - projected_mean
        fraction = min(1., variance / (variance + delta * delta) + numeric_padding)
        # An integer count is <= floor of a conservative real upper bound.
        # nextafter protects rounding at an integer endpoint.
        upper = min(count, math.floor(np.nextafter(count * fraction, np.inf)))
    return dict(total_rows=n, valid_rows=count, activation_count_upper=upper,
                activation_fraction_upper=upper/n,
                projected_mean=projected_mean, projected_variance_upper=variance,
                threshold=tau, effective_threshold_lower=lowered,
                fp32_dot_error_allowance=dot_error,
                method='finite recorded-row Cantelli; padded real bound then integer floor',
                scope='finite recorded calibration role only; not unseen population',
                full_guard_conjunction_also_bounded=True,
                unseen_population_far_certified=False, main_install_authorized=False)


class CurrentRoleSketchMoments:
    """Owner-local current CAL roles summarized once per task; no raw retention."""
    def __init__(self, client_id, sketch, role_sha256):
        if (type(client_id) is not int or client_id < 0 or
                not isinstance(sketch, SharedSketch) or
                type(role_sha256) is not str or len(role_sha256) != 64):
            raise Rejected('INVALID_ROLE_SKETCH_OWNER')
        self.client_id = client_id
        self.sketch = sketch
        self.role_sha256 = role_sha256
        self.entries, self.provenance = {}, []

    def observe_current(self, scoped):
        if (not isinstance(scoped, CurrentCalibrationData) or
                scoped.client_id != self.client_id or
                scoped.store['metadata_sha256'] != self.sketch.preprocessing_sha256 or
                scoped.store['role_manifest_sha256'] != self.role_sha256):
            raise Rejected('ROLE_SKETCH_SCOPE_CHANGED')
        if self.provenance and scoped.task <= self.provenance[-1]['task']:
            raise Rejected('ROLE_SKETCH_PAST_OR_SEALED_TASK')
        classes = scoped.store['task_classes'][str(scoped.task)]
        if any(str(c) in self.entries for c in classes):
            raise Rejected('ROLE_SKETCH_CLASS_REPLAY')
        pool = scoped.current_pool(self.client_id, 'calibration', classes)
        split = stratified_roles(pool, ROUTE_RULES['seed'] + self.client_id)
        pending = copy.deepcopy(self.entries)
        event = dict(owner=self.client_id, task=scoped.task,
                     partition_sha256=pool['partition_sha256'],
                     role_manifest_sha256=self.role_sha256,
                     sketch_sha256=digest(self.sketch.manifest()), roles={})
        for role in ROLES:
            ix = split[role]
            z, valid = self.sketch.features(pool['X'][ix])
            event['roles'][role] = dict(
                row_ids_sha256=hashlib.sha256(np.asarray(pool['rows'][ix], dtype='<i8').tobytes()).hexdigest(),
                class_counts={str(c): int((pool['y'][ix] == c).sum()) for c in classes})
            for c in classes:
                selected = pool['y'][ix] == c
                if selected.any():
                    pending.setdefault(str(c), dict(task=scoped.task, roles={}))['roles'][role] = _entry(z[selected], valid[selected])
        event['event_id'] = digest(event)
        self.entries, self.provenance = pending, self.provenance + [event]
        # Store only task/class moments and provenance hashes; pool dies here.
        return copy.deepcopy(event)

    def state(self):
        body = dict(version=VERSION, client_id=self.client_id,
                    sketch=self.sketch.manifest(), role_manifest_sha256=self.role_sha256,
                    entries=copy.deepcopy(self.entries), provenance=copy.deepcopy(self.provenance),
                    retained_raw_examples=0, retained_per_sample_features=0,
                    old_class_safety_certified=False, main_install_authorized=False)
        return dict(body, state_digest=digest(body))

    @classmethod
    def restore(cls, state):
        fields = {'version','client_id','sketch','role_manifest_sha256','entries','provenance',
                  'retained_raw_examples','retained_per_sample_features','old_class_safety_certified',
                  'main_install_authorized','state_digest'}
        if set(state) != fields:
            raise Rejected('ROLE_SKETCH_STATE_SCHEMA_CHANGED')
        body = {k:copy.deepcopy(v) for k,v in state.items() if k != 'state_digest'}
        if (body['version'] != VERSION or state['state_digest'] != digest(body) or
                body['retained_raw_examples'] != 0 or body['retained_per_sample_features'] != 0 or
                body['old_class_safety_certified'] is not False or body['main_install_authorized'] is not False):
            raise Rejected('ROLE_SKETCH_STATE_CHANGED')
        sig = body['sketch']
        obj = cls(body['client_id'], SharedSketch(tuple(sig['input_shape']),sig['dimension'],
                  sig['preprocessing_sha256'],sig['seed']),body['role_manifest_sha256'])
        if obj.sketch.manifest() != sig:
            raise Rejected('ROLE_SKETCH_FUNCTION_CHANGED')
        obj.entries, obj.provenance = body['entries'], body['provenance']
        tasks, declared = [], set()
        for event in obj.provenance:
            if set(event) != {'owner','task','partition_sha256','role_manifest_sha256','sketch_sha256','roles','event_id'}:
                raise Rejected('ROLE_SKETCH_EVENT_SCHEMA_CHANGED')
            expected = digest({k:v for k,v in event.items() if k != 'event_id'})
            t = event['task']
            if (type(t) is not int or not 0 <= t <= 5 or event['owner'] != obj.client_id or
                    event['role_manifest_sha256'] != obj.role_sha256 or
                    event['sketch_sha256'] != digest(sig) or expected != event['event_id'] or
                    set(event['roles']) != set(ROLES)):
                raise Rejected('ROLE_SKETCH_EVENT_CHANGED')
            tasks.append(t)
            for role, evidence in event['roles'].items():
                if set(evidence) != {'row_ids_sha256','class_counts'} or len(evidence['row_ids_sha256']) != 64:
                    raise Rejected('ROLE_SKETCH_ROLE_PROVENANCE_CHANGED')
                for c, n in evidence['class_counts'].items():
                    if (str(int(c)) != c or not 0 <= int(c) < 34 or type(n) is not int or n < 0):
                        raise Rejected('ROLE_SKETCH_CLASS_PROVENANCE_CHANGED')
                    record = obj.entries.get(c)
                    if n:
                        if record is None or record['task'] != t or role not in record['roles']:
                            raise Rejected('ROLE_SKETCH_COUNT_PROVENANCE_CHANGED')
                        rows, _, _, _ = _checked_entry(record['roles'][role], sig['dimension'])
                        if rows != n:
                            raise Rejected('ROLE_SKETCH_COUNT_PROVENANCE_CHANGED')
                        declared.add((c, role))
        if tasks != sorted(set(tasks)):
            raise Rejected('ROLE_SKETCH_TASK_ORDER_CHANGED')
        for c, entry in obj.entries.items():
            if set(entry) != {'task','roles'} or not entry['roles'] or not set(entry['roles']).issubset(ROLES):
                raise Rejected('ROLE_SKETCH_ENTRY_SCHEMA_CHANGED')
            for role, value in entry['roles'].items():
                _checked_entry(value, sig['dimension'])
                if (c,role) not in declared:
                    raise Rejected('ROLE_SKETCH_UNDECLARED_MOMENTS')
        return obj

    def bounds(self, packet, role, required_classes):
        from .portable_route import ProtectedRoute
        if role not in ROLES:
            raise Rejected('UNKNOWN_ROLE_SKETCH_QUERY')
        route = ProtectedRoute.from_packet(packet)
        if route.metadata['signature'] != self.sketch.manifest():
            raise Rejected('ROLE_SKETCH_QUERY_FUNCTION_CHANGED')
        required = sorted(set(required_classes))
        if (any(type(c) is not int or not 0 <= c < 34 for c in required) or
                route.metadata['class_id'] in required):
            raise Rejected('INVALID_OLD_CLASS_QUERY')
        found = {str(c):finite_activation_bound(self.entries[str(c)]['roles'][role],
                    route.prototype,route.metadata['tau'])
                 for c in required if str(c) in self.entries and role in self.entries[str(c)]['roles']}
        missing = [c for c in required if str(c) not in found]
        return dict(packet_sha256=hashlib.sha256(packet).hexdigest(), role=role,
                    required_classes=required, class_bounds=found, missing_classes=missing,
                    summary_state_digest=self.state()['state_digest'],
                    historical_raw_reads_required=False,
                    unseen_population_far_certified=False, main_install_authorized=False)
