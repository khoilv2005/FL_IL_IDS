"""Native availability probes before train-time transfer; no live installation.

Metadata creates requests, never proves functional absence. Endpoint statistics
classify an observed gap separately from risk and installation authorization.
Raw CAL remains at each current owner. Summary counts are not old CAL evidence.
"""
import copy
import math

import numpy as np

from .config import Rejected
from .current_calibration_data import CurrentCalibrationData
from .evaluate import Predictor
from .imported_route import ROUTE_RULES, stratified_roles
from .state import complete_hash, digest


VERSION = 'appliance_functional_gap_native_v1'
RULES = dict(min_positive=32, min_negative=32, recall=.95, FAR=.001,
             negative_break=0, net_rescue_min=1)


def availability_shadow(model, router, target, task):
    """Clone and expose an original row; do not change masks/ranks/weights."""
    if type(target) is not int or not 0 <= target < model.fc2.out_features:
        raise Rejected('FUNCTIONAL_GAP_TARGET_INVALID')
    if type(task) is not int or task < 0:
        raise Rejected('FUNCTIONAL_GAP_TASK_INVALID')
    candidate, detector = copy.deepcopy(model), copy.deepcopy(router)
    detector.episode_classes[task] = sorted(set(detector.episode_classes.get(task, [])) | {target})
    return candidate, detector


def metadata_candidates(receiver, model, router, models, counts, groups, alphas,
                        task, current_classes):
    """Enumerate every positive-alpha neighbor owning a mature current class.

    Donor native router accuracy is not an eligibility filter. BASE ownership
    and ranks only authorize a probe. The caller binds metadata to its role lock.
    """
    if receiver not in groups or receiver not in alphas or receiver not in counts:
        raise Rejected('FUNCTIONAL_GAP_GRAPH_RECEIVER_MISSING')
    peers = groups[receiver]
    positive = {int(p) for p, a in alphas[receiver].items() if p != receiver and a > 0}
    if set(peers) != positive or len(peers) != len(set(peers)):
        raise Rejected('FUNCTIONAL_GAP_GRAPH_CHANGED')
    if any(not math.isfinite(a) or a < 0 for a in alphas[receiver].values()):
        raise Rejected('FUNCTIONAL_GAP_ALPHA_INVALID')
    missing = sorted(set(current_classes) - set(router.episode_classes.get(task, [])))
    requests, candidates = [], []
    for c in missing:
        # Existing local BASE support suggests a local registration problem,
        # not permission to ask a peer to overwrite knowledge already owned.
        local = int(counts[receiver].get(str(c), 0))
        requests.append(dict(receiver=receiver, class_id=c, task=task,
            locally_owned_BASE=local, receiver_rank=int(model.unit_ranks['fc2'][c]),
            functional_gap_established=False))
        for donor in sorted(peers):
            if donor not in models or donor not in counts:
                raise Rejected('FUNCTIONAL_GAP_NEIGHBOR_AUTHORITY_MISSING')
            n = int(counts[donor].get(str(c), 0))
            rank = int(models[donor].unit_ranks['fc2'][c])
            if n > 0 and rank >= 2:
                candidates.append(dict(receiver=receiver, donor=donor, class_id=c,
                    task=task, alpha=float(alphas[receiver][donor]),
                    donor_current_owned_BASE=n, donor_rank=rank,
                    receiver_current_owned_BASE=local))
    return requests, candidates


def current_split(view, split):
    if not isinstance(view, CurrentCalibrationData) or split not in ('fit', 'holdout'):
        raise Rejected('FUNCTIONAL_GAP_CURRENT_CAL_REQUIRED')
    classes = view.store['task_classes'][str(view.task)]
    pool = view.current_pool(view.client_id, 'calibration', classes)
    partitions = stratified_roles(pool, ROUTE_RULES['seed'] + view.client_id)
    indices = partitions[split]
    # Deduplicate inside this endpoint split. Original role materialization
    # already groups repeated content within the same role; no iid claim.
    selected, seen = [], {}
    excluded = {digest(np.ascontiguousarray(pool['X'][ix]))
        for stage in ('fit', 'selection') for ix in partitions[stage]} if split == 'holdout' else set()
    overlap_excluded = 0
    for ix in indices:
        key = digest(np.ascontiguousarray(pool['X'][ix]))
        label = int(pool['y'][ix])
        if key in excluded:
            overlap_excluded += 1
            continue
        if key in seen:
            if seen[key] != label:
                raise Rejected('FUNCTIONAL_GAP_CONTENT_LABEL_CONFLICT')
            continue
        seen[key] = label
        selected.append(int(ix))
    selected = np.asarray(selected, np.int64)
    return dict(X=pool['X'][selected], y=pool['y'][selected], rows=pool['rows'][selected],
        binding=dict(owner=view.client_id, task=view.task, split=split,
            role='current CAL', role_sha256=view.store['role_manifest_sha256'],
            partition_sha256=pool['partition_sha256'],
            rows_digest=digest(pool['rows'][selected]), unique_rows=len(selected),
            earlier_current_split_content_excluded=overlap_excluded,
            cross_owner_content_independence_established=False))


def comparison(reference, candidate, labels, target):
    reference, candidate, labels = [np.asarray(v, np.int64) for v in (reference, candidate, labels)]
    if any(v.ndim != 1 for v in (reference, candidate, labels)) or not (len(reference) == len(candidate) == len(labels)):
        raise Rejected('FUNCTIONAL_GAP_PREDICTION_ALIGNMENT')
    pos, neg = labels == target, labels != target
    correct0, correct1 = reference == labels, candidate == labels
    rescue = int((~correct0 & correct1).sum())
    broken = int((correct0 & ~correct1).sum())
    delta = int(correct1.sum()) - int(correct0.sum())
    if delta != rescue - broken:
        raise Rejected('FUNCTIONAL_GAP_ACCOUNTING')
    return dict(rows=len(labels), positive=int(pos.sum()), target_hits=int((candidate[pos] == target).sum()),
        reference_target_hits=int((reference[pos] == target).sum()), negative=int(neg.sum()),
        false_positive=int((candidate[neg] == target).sum()), rescue=rescue, break_count=broken,
        net_rescue=delta, negative_break=int((correct0 & ~correct1 & neg).sum()),
        recall=float((candidate[pos] == target).mean()) if pos.any() else None,
        reference_recall=float((reference[pos] == target).mean()) if pos.any() else None,
        FAR=float((candidate[neg] == target).mean()) if neg.any() else None,
        by_negative_class={str(int(c)):dict(rows=int((labels == c).sum()),
            false_positive=int((candidate[labels == c] == target).sum()),
            FAR=float((candidate[labels == c] == target).mean())) for c in np.unique(labels[neg])})


def endpoint_comparison(reference, candidate, pool, target, seen, device, batch_size=256):
    """Run at an owner endpoint with x only; return aggregate evidence counts."""
    predictor = Predictor(seen, device, batch_size)
    before_hashes = [complete_hash(*v) for v in (reference, candidate)]
    p0 = predictor(*reference, pool['X'])
    p1 = predictor(*candidate, pool['X'])
    if before_hashes != [complete_hash(*v) for v in (reference, candidate)]:
        raise Rejected('FUNCTIONAL_GAP_PROBE_MUTATED_MODEL')
    return dict(binding=pool['binding'], class_id=target, reference_function=before_hashes[0],
        candidate_function=before_hashes[1], stats=comparison(p0, p1, pool['y'], target),
        raw_examples_transmitted=0, labels_used_by_predictor=False)


def classify(donor_evidence, receiver_evidence, target):
    """CAL-FIT decides observed gap only, never authorizes registration."""
    db, rb = (v['binding'] for v in (donor_evidence, receiver_evidence))
    if (db['split'] != 'fit' or rb['split'] != 'fit' or db['owner'] == rb['owner'] or
            db['task'] != rb['task'] or db['role_sha256'] != rb['role_sha256'] or
            donor_evidence['class_id'] != target or receiver_evidence['class_id'] != target or
            donor_evidence['reference_function'] != receiver_evidence['reference_function'] or
            donor_evidence['candidate_function'] != receiver_evidence['candidate_function']):
        raise Rejected('FUNCTIONAL_GAP_FIT_BINDING')
    ds, rs = donor_evidence['stats'], receiver_evidence['stats']
    sufficient = ds['positive'] >= RULES['min_positive']
    gap = ('registration_gap' if ds['recall'] >= RULES['recall'] else 'functional_gap') if sufficient else 'unknown_gap'
    risk_sufficient = rs['negative'] >= RULES['min_negative']
    risk_pass = bool(risk_sufficient and rs['FAR'] <= RULES['FAR'] and
        all(v['FAR'] <= RULES['FAR'] for e in (ds, rs) for v in e['by_negative_class'].values()) and
        ds['negative_break'] + rs['negative_break'] <= RULES['negative_break'])
    return dict(version=VERSION, class_id=target, gap=gap,
        scope='observed donor current CAL-FIT only; not all-owner competence',
        observed_receiver_current_risk_pass=risk_pass,
        trial_transfer_requested=bool(gap == 'functional_gap' and risk_pass),
        donor_training_required=bool(gap == 'functional_gap' and risk_pass),
        registration_requires_acceptance=True, historical_protection_status='unknown',
        installation_authorized=False, native_smoke_authorized=False)


def select_probes(probes):
    """Registration cannot hide an observed risk by choosing a benign donor.

    Every peer evaluates the same Availability-only function for a class. Its
    negative evidence therefore remains relevant across donor choices. Transfer
    candidates still require their own sufficient positive/current risk FIT;
    their *new* function needs fresh risk evaluation, not recycled old scores.
    """
    selected, protection = [], {}
    for c in sorted({p['class_id'] for p in probes}):
        options = [p for p in probes if p['class_id'] == c]
        functions = {p['donor_FIT']['candidate_function'] for p in options}
        if len(functions) != 1:
            raise Rejected('FUNCTIONAL_GAP_PROBE_FUNCTION_VERSIONS_DIFFER')
        violations = []
        for p in options:
            for name in ('donor_FIT', 'receiver_FIT'):
                evidence = p[name]
                for label, value in evidence['stats']['by_negative_class'].items():
                    if value['FAR'] > RULES['FAR']:
                        violation = dict(owner=evidence['binding']['owner'], class_id=int(label),
                            rows=value['rows'], false_positive=value['false_positive'], FAR=value['FAR'])
                        if violation not in violations:
                            violations.append(violation)
                if evidence['stats']['negative_break']:
                    violations.append(dict(owner=evidence['binding']['owner'],
                        negative_break=evidence['stats']['negative_break']))
        protection[c] = dict(availability_function=next(iter(functions)),
            current_negative_veto=bool(violations), violations=violations,
            no_veto_is_not_broad_risk_authorization=True, old_risk='unknown')
        def rank(p):
            safe = p['observed_receiver_current_risk_pass']
            priority = (0 if safe and p['gap'] == 'registration_gap' and not violations else
                        1 if safe and p['gap'] == 'functional_gap' else 2)
            return priority, -p['donor_FIT']['stats']['positive'], p['pair']['donor']
        selected.append(min(options, key=rank))
    return selected, protection


def assess_holdout(donor, receiver, action, independent_risk=None, protected_receipts=()):
    """Compare learned update to Availability-only, not to a closed mask.

    Independent old-risk evidence is a separate requirement. This development
    API never commits state; a production installer would need verified receipt
    authority and provenance, not a caller-supplied independence boolean.
    """
    if action not in ('registration', 'transfer'):
        raise Rejected('FUNCTIONAL_GAP_ACTION_INVALID')
    db, rb = (v['binding'] for v in (donor, receiver))
    if (db['split'] != 'holdout' or rb['split'] != 'holdout' or db['owner'] == rb['owner'] or
            db['task'] != rb['task'] or db['role_sha256'] != rb['role_sha256'] or
            donor['candidate_function'] != receiver['candidate_function'] or
            donor['reference_function'] != receiver['reference_function'] or
            donor['class_id'] != receiver['class_id']):
        raise Rejected('FUNCTIONAL_GAP_HOLDOUT_BINDING')
    ds, rs = donor['stats'], receiver['stats']
    for witness in protected_receipts:
        wb = witness['binding']
        if (wb['split'] != 'holdout' or wb['task'] != db['task'] or
                wb['role_sha256'] != db['role_sha256'] or witness['class_id'] != donor['class_id'] or
                witness['reference_function'] != donor['reference_function'] or
                witness['candidate_function'] != donor['candidate_function']):
            raise Rejected('FUNCTIONAL_GAP_PROTECTED_HOLDOUT_BINDING')
    negative_evidence = [ds, rs] + [e['stats'] for e in protected_receipts]
    useful = ds['net_rescue'] + rs['net_rescue'] >= RULES['net_rescue_min']
    observed_pass = bool(ds['positive'] >= RULES['min_positive'] and rs['negative'] >= RULES['min_negative'] and
        ds['recall'] >= RULES['recall'] and rs['FAR'] <= RULES['FAR'] and
        all(v['FAR'] <= RULES['FAR'] for e in negative_evidence for v in e['by_negative_class'].values()) and
        sum(e['negative_break'] for e in negative_evidence) <= RULES['negative_break'] and useful and
        (action != 'transfer' or ds['target_hits'] > ds['reference_target_hits']))
    return dict(action=action, reference='availability_only' if action == 'transfer' else 'closed_mask',
        observed_current_CAL_holdout_pass=observed_pass, net_rescue=ds['net_rescue'] + rs['net_rescue'],
        additional_protected_HOLDOUT_receipts=len(protected_receipts),
        independent_risk=independent_risk or dict(status='unknown', reason='no independently qualified old-risk receipt'),
        evidence_authorization_verified=False, installation_authorized=False,
        native_smoke_authorized=False, decision='qualification_only_not_authorized')
