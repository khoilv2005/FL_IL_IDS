"""Versioned, scope-bound patch lifecycle; no calibration data access.

An old certificate is evidence in its original domain only. A task number,
router prediction, BASE veto, or a small new CAL pool never proves domain
equivalence. Runtime scope is declared by the application protocol, not y_true.
"""
import copy
import hashlib
import json
from .config import Rejected

VERSION = 'appliance_scope_lifecycle_v1'
ACTIVE_STATES = ('COMMITTED', 'CARRY_FORWARD')


def sealed_sha256(certificate):
    return hashlib.sha256(json.dumps(certificate,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def checked_scope(scope):
    if (not isinstance(scope, dict) or set(scope) != {'domain_id', 'classes'} or
            not isinstance(scope['domain_id'], str) or not scope['domain_id'] or
            not isinstance(scope['classes'], (list, tuple)) or not scope['classes'] or
            any(type(c) is not int or not 0 <= c < 34 for c in scope['classes']) or
            len(set(scope['classes'])) != len(scope['classes'])):
        raise Rejected('INVALID_CERTIFICATE_DOMAIN_SCOPE')
    return dict(domain_id=scope['domain_id'], classes=sorted(scope['classes']))


def bind_initial_certificate(entry, scope):
    """Seal existing initial acceptance. No larger class/domain grant is inferred."""
    if 'lifecycle_certificate' in entry:
        raise Rejected('LIFECYCLE_ALREADY_BOUND')
    scope = checked_scope(scope)
    a = entry['current_acceptance']
    supported = {entry['class_id']} | set(map(int, a.get('receiver_far_by_class', {})))
    if (not a.get('passed_current_scope') or a.get('positive_rows', 0) < 32 or
            a.get('receiver_rows', 0) < 32 or a.get('recall', 0) < .95 or
            a.get('receiver_far', 1) > .001 or a.get('break_count', 1) != 0 or
            a.get('rescue', 0) < 1 or
            any(v > .001 for v in a.get('receiver_far_by_class', {}).values()) or
            not set(scope['classes']).issubset(supported) or entry['class_id'] not in scope['classes']):
        raise Rejected('LIFECYCLE_INITIAL_ACCEPTANCE_OR_SCOPE_INVALID')
    sealed = dict(version=VERSION, scope=scope, class_id=entry['class_id'],
        receiver=entry['receiver'], task=entry['task'], patch_id=entry['patch_id'],
        packet_sha256=hashlib.sha256(entry['packet']).hexdigest(),
        patch_version=entry['patch_version'], route_version=entry['route_version'],
        activation_policy=entry['activation_policy'], guard_declaration=copy.deepcopy(entry['guard_declaration']),
        guard_function_fingerprint=entry['guard_function_fingerprint'],
        initial_acceptance=copy.deepcopy(a), dependency_drift_budget=0.,
        threshold_changes_allowed=False, scope_expansion_allowed=False,
        raw_calibration_retained=False, population_FAR_claim=False)
    entry.update(lifecycle_certificate=sealed, lifecycle_certificate_sha256=sealed_sha256(sealed),
        lifecycle_state='COMMITTED', lifecycle_reason=None, lifecycle_history=[])
    return copy.deepcopy(sealed)


def certificate_binding_current(entry):
    c = entry.get('lifecycle_certificate')
    try:
        if not isinstance(c, dict) or c.get('version') != VERSION or sealed_sha256(c) != entry.get('lifecycle_certificate_sha256'):
            return False
        checked_scope(c['scope'])
        if c['dependency_drift_budget'] != 0. or c['scope_expansion_allowed'] or c['threshold_changes_allowed']:
            return False
        return all(entry.get(k) == c[k] for k in (
            'class_id', 'receiver', 'task', 'patch_id', 'patch_version', 'route_version',
            'activation_policy', 'guard_function_fingerprint', 'guard_declaration'
        )) and (
            entry['current_acceptance'] == c['initial_acceptance'] and
            hashlib.sha256(entry['packet']).hexdigest() == c['packet_sha256'])
    except (KeyError, TypeError, ValueError, Rejected):
        return False


def reconcile(entry, function_current, scope, task, conflict=False, new_cal_rows=None):
    """Insufficient CAL is not certificate corruption. Scope expansion suspends."""
    if type(task) is not int or task < entry['task'] or task < entry.get('lifecycle_last_task', entry['task']):
        raise Rejected('NONCHRONOLOGICAL_LIFECYCLE_TASK')
    bound = certificate_binding_current(entry)
    try:
        requested = checked_scope(scope)
    except Rejected:
        requested = None
    old = entry.get('lifecycle_certificate', {}).get('scope')
    contained = bool(requested and old and requested['domain_id'] == old['domain_id'] and
                     set(requested['classes']).issubset(old['classes']))
    reason = ('certificate_binding_changed' if not bound else
              'dependency_head_or_guard_drift' if not function_current else
              'new_protection_conflict' if conflict else
              'runtime_domain_not_proven_inside_certificate_scope' if not contained else None)
    state = 'SUSPENDED' if reason else ('COMMITTED' if task == entry['task'] else 'CARRY_FORWARD')
    report = dict(version=VERSION, task=task, state=state, reason=reason,
        certificate_binding_current=bound, function_current=bool(function_current),
        runtime_scope=requested, inside_original_scope=contained, new_conflict=bool(conflict),
        new_cal_rows=new_cal_rows, new_CAL_certificate_issued=False,
        insufficient_CAL_corrupts_old_certificate=False, historical_raw_reads=0,
        patch_retained=True, protection_retained=True)
    entry.update(lifecycle_state=state, lifecycle_reason=reason, lifecycle_last_task=task,
                 lifecycle_runtime_scope=requested)
    # Keep bounded metadata; neither raw data nor an ever-growing per-round log.
    history = entry.setdefault('lifecycle_history', [])
    if not history or history[-1] != report:
        history.append(report)
        del history[:-32]
    return report


def route_authorized(entry, scope):
    if entry.get('lifecycle_state') not in ACTIVE_STATES or not certificate_binding_current(entry):
        return False
    try:
        requested = checked_scope(scope)
    except Rejected:
        return False
    old = entry['lifecycle_certificate']['scope']
    return requested['domain_id'] == old['domain_id'] and set(requested['classes']).issubset(old['classes'])
