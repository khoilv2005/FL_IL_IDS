"""Experimental FIT-only compatibility probes before guard acceptance.

Donor head metadata permits maturity rejection BEFORE a receiver capsule is
sent. Receiver-space scoring still needs the receiver function at the donor
endpoint; this setup cost must be counted, never described as a KiB transfer.
These probes rank donors. They do not relax the separate 95% CAL HOLDOUT gate.
"""
import copy
import numpy as np
import torch
from .closure import effective_linear
from .config import Rejected
from .current_calibration_data import CurrentCalibrationData
from .dependency_boundary import head_dependency_boundary
from .imported_route import ROUTE_RULES, stratified_roles
from .ledger import wilson_lower
from .portable_route import SharedSketch, prototype_summary, fit_support_cosine_floor
from .stable_head import references
from .state import digest

VERSION = 'appliance_receiver_aware_current_FIT_probe_v1'


def head_offer(model, class_id, model_version):
    if int(model.unit_ranks['fc2'][class_id]) < 2:
        raise Rejected('PRECHECK_DONOR_HEAD_NOT_MATURE')
    weight, bias = effective_linear(model, 'fc2')
    body = dict(version=VERSION, class_id=int(class_id), model_version=model_version,
                weight=weight[class_id].numpy().tolist(), bias=float(bias[class_id]))
    return dict(body, digest=digest(body))


def maturity_precheck(receiver, router, offer, task, seen):
    if (set(offer) != {'version', 'class_id', 'model_version', 'weight', 'bias', 'digest'} or
            offer['version'] != VERSION or offer['digest'] != digest({k: v for k, v in offer.items() if k != 'digest'})):
        raise Rejected('PRECHECK_HEAD_OFFER_CHANGED')
    if type(offer['model_version']) is not str or not offer['model_version'] or not np.isfinite(offer['bias']):
        raise Rejected('PRECHECK_HEAD_VERSION_OR_BIAS_CHANGED')
    c = offer['class_id']
    if type(c) is not int or c not in seen or int(receiver.unit_ranks['fc2'][c]) != 0:
        raise Rejected('PRECHECK_RECEIVER_OUTPUT_NOT_FREE')
    refs = references(router, task, c, seen)
    if any(int(receiver.unit_ranks['fc2'][k]) < 2 for k in refs):
        return dict(eligible=False, reason='NONMATURE_STABLE_MARGIN_REFERENCE', reference_classes=refs,
                    before_capsule=True, authorizes_installation=False)
    w, b = effective_linear(receiver, 'fc2')
    patch = torch.as_tensor(offer['weight'], dtype=w.dtype).reshape(1, -1)
    boundary = head_dependency_boundary(receiver, torch.cat((patch, w[refs]), 0))
    return dict(eligible=boundary['all_dependencies_mature'],
        reason=None if boundary['all_dependencies_mature'] else 'NONMATURE_STABLE_GUARD_DEPENDENCY',
        reference_classes=refs, dependency_version=boundary['canonical_value_function_sha'],
        head_offer_digest=offer['digest'], receiver_reference_version=digest(dict(weight=w[refs], bias=b[refs])),
        maturity_by_layer=boundary['mature_dependency_scope'], before_capsule=True,
        authorizes_installation=False)


def current_fit(view):
    if not isinstance(view, CurrentCalibrationData):
        raise Rejected('PROBE_CURRENT_CAL_AUTHORITY_REQUIRED')
    classes = view.store['task_classes'][str(view.task)]
    pool = view.current_pool(view.client_id, 'calibration', classes)
    fit = stratified_roles(pool, ROUTE_RULES['seed'] + view.client_id)['fit']
    return {k: pool[k][fit] for k in ('X', 'y', 'rows')}


@torch.no_grad()
def fit_probe(receiver, offer, precheck, receiver_view, donor_view, protection, required, batch_size=512):
    """Trusted simulator executes each view at its owner; returns counts only.

    Never use SELECTION or HOLDOUT predictions here. Donor FIT positives passed
    through a receiver replica assess portability, not receiver-owned recall.
    Fixed route cosine floor and zero margin are ranking diagnostics, not the
    final installed guard. The subsequent installer must still select/freeze
    thresholds on SELECTION and pass unchanged CAL HOLDOUT acceptance.
    """
    if not precheck['eligible']:
        raise Rejected('PROBE_REQUIRES_MATURE_DEPENDENCIES')
    if (not isinstance(receiver_view, CurrentCalibrationData) or not isinstance(donor_view, CurrentCalibrationData)):
        raise Rejected('PROBE_CURRENT_CAL_AUTHORITY_REQUIRED')
    if (protection.owner != receiver_view.client_id or protection.own.role_sha != receiver_view.store['role_manifest_sha256'] or
            protection.own.pp_sha != receiver_view.store['metadata_sha256']):
        raise Rejected('PROBE_PROTECTION_OWNER_OR_ROLE_CHANGED')
    if (precheck.get('head_offer_digest') != offer['digest'] or
            offer['digest'] != digest({k: v for k, v in offer.items() if k != 'digest'})):
        raise Rejected('PROBE_HEAD_CHANGED_AFTER_PRECHECK')
    if (receiver_view.task != donor_view.task or receiver_view.client_id == donor_view.client_id or
            receiver_view.store['metadata_sha256'] != donor_view.store['metadata_sha256'] or
            receiver_view.store['role_manifest_sha256'] != donor_view.store['role_manifest_sha256']):
        raise Rejected('PROBE_OWNER_OR_DATA_PROTOCOL_CHANGED')
    c = offer['class_id']
    own = receiver_view.manifest['clients'][str(receiver_view.client_id)]['role_class_counts']
    donor = donor_view.manifest['clients'][str(donor_view.client_id)]['role_class_counts']
    if int(own['base'].get(str(c), 0)) or int(own['calibration'].get(str(c), 0)) or int(donor['base'].get(str(c), 0)) <= 0:
        raise Rejected('PROBE_MISSING_CLASS_OR_DONOR_OWNERSHIP_CHANGED')
    head = np.asarray(offer['weight'], np.float32); bias = offer['bias']
    local_w, local_b = effective_linear(receiver, 'fc2')
    refs = precheck['reference_classes']
    boundary = head_dependency_boundary(receiver, torch.cat((torch.as_tensor(head)[None, :], local_w[refs]), 0))
    if (not boundary['all_dependencies_mature'] or boundary['canonical_value_function_sha'] != precheck['dependency_version'] or
            digest(dict(weight=local_w[refs], bias=local_b[refs])) != precheck['receiver_reference_version']):
        raise Rejected('PROBE_RECEIVER_CHANGED_AFTER_PRECHECK')
    rp, dp = current_fit(receiver_view), current_fit(donor_view)
    dx = dp['X'][dp['y'] == c]
    if len(dx) < 32 or len(rp['y']) < 32:
        raise Rejected('PROBE_FIT_SUPPORT_TOO_SMALL')
    sketch = SharedSketch(tuple(receiver_view.store['input_shape']), 16, receiver_view.store['metadata_sha256'])
    z, valid = sketch.features(dx)
    proto, _, support = prototype_summary(z, valid)
    floor = fit_support_cosine_floor(support)
    original_modes = [(m, m.training) for m in receiver.modules()]
    active = copy.deepcopy(receiver.active_adapters)
    def hit(x):
        outputs = []
        for start in range(0, len(x), batch_size):
            part = x[start:start+batch_size]
            value = torch.as_tensor(part, dtype=torch.float32, device=next(receiver.parameters()).device)
            with torch.autocast(device_type=value.device.type, enabled=False):
                h = receiver.penultimate_features(value).cpu()
            margin = h.numpy() @ head + bias - torch.nn.functional.linear(h, local_w[refs], local_b[refs]).numpy().max(1)
            z, valid = sketch.features(part)
            outputs.append(valid & (z @ proto > floor) & (margin > 0) & ~protection.veto(part, required))
        return np.concatenate(outputs)
    try:
        receiver.eval(); receiver.clear_active_adapters()
        positive, negative = hit(dx), hit(rp['X'])
    finally:
        receiver.active_adapters = active
        for module, mode in original_modes:
            module.training = mode
    n, h, fp = len(positive), int(positive.sum()), int(negative.sum())
    return dict(version=VERSION, receiver=receiver_view.client_id, donor=donor_view.client_id,
        class_id=c, task=receiver_view.task, head_offer_digest=offer['digest'],
        receiver_model_dependency_version=precheck['dependency_version'],
        protection_snapshot_digest=digest(protection.state()),
        donor_FIT_positive_rows=n, compatible_positive_activations=h,
        compatible_positive_lcb=wilson_lower(h, n), receiver_FIT_negative_rows=len(negative),
        receiver_FIT_false_activation=fp, receiver_FIT_far=fp/len(negative),
        no_receiver_owned_positive_evidence=True, selection_predictions_opened=False,
        holdout_predictions_opened=False, historical_raw_CAL_opened=False,
        acceptance_recall_gate=.95, authorizes_installation=False)


def rank_fit_probes(probes, donor_qualities):
    """Deterministic rank; frozen before any SELECTION/HOLDOUT acceptance."""
    if any(p['authorizes_installation'] or p['holdout_predictions_opened'] or p['selection_predictions_opened'] for p in probes):
        raise Rejected('PROBE_ACCEPTANCE_OR_HOLDOUT_LEAKAGE')
    return sorted(probes, key=lambda p: (
        -(p['compatible_positive_lcb'] - p['receiver_FIT_far']),
        -donor_qualities[p['donor']], p['donor']))
