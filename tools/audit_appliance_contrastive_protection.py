"""Helpers for a frozen negative-head policy, not production installation.

All previously FIT-selected task requests are tested; no test-error class filter.
Negative source choice is fixed before CAL: patch donor first, then largest own
BASE support count, then sender ID. Geometry/head receipts are simulated from
the actual terminal positive graph. Historical raw CAL is never reopened.
"""
import copy
import hashlib
import numpy as np
from appliance.config import Rejected
from appliance.contrastive_protection import ContrastiveProtection
from appliance.receiver_aware_discovery import head_offer
from appliance.provenance_protection import frozen_veto
from appliance.state import complete_hash, digest


def build_counter(ckpt, ledger, required, c, preferred):
    from tools.audit_appliance_provenance_acceptance import original_model
    offers = {}
    models = {}
    for k in sorted(set(required)):
        if not any(k in r['receipt']['class_rows'] for r in ledger.receipts.values()):
            continue
        choices = []
        for r in ledger.receipts.values():
            sender = r['receipt']['sender']
            state = ckpt['client_algorithm_states'][sender]
            state = state.get('denice', state)
            if k not in r['receipt']['class_rows'] or int(state['neuron_ages']['fc2'][k]) < 2:
                continue
            source = ledger.sources[r['source_id']]
            count = source['memory']['entries'][str(k)]['count']
            choices.append((sender != preferred, -count, sender, r['source_id'], r))
        if not choices:
            raise Rejected('CONTRASTIVE_NO_MATURE_OWNED_NEGATIVE_HEAD', str(k))
        _, _, sender, sid, r = min(choices, key=lambda v: v[:4])
        if sender not in models:
            models[sender], _ = original_model(ckpt, sender)
        offer = head_offer(models[sender], k, digest(ckpt['client_model_states'][sender]))
        if offer['model_version'] != r['receipt']['sender_model_version']:
            raise Rejected('CONTRASTIVE_SOURCE_MODEL_CHANGED')
        offers[k] = dict(sender=sender, class_id=k, offer=offer, source_id=sid, source_head_mature=True)
    return ContrastiveProtection(ledger, required, c, offers)


def controls(counter, model, router, route, refs, x, seen, batch_size):
    from appliance.provenance_protection import sealed
    from appliance.portable_route import receiver_signals
    from appliance.closure import effective_linear
    import torch
    checks = {}
    before = complete_hash(model, router)
    function = counter.function(model, route, refs)['fingerprint']
    sig = counter.signals(model, router, x, seen, route, refs, batch_size, 'cpu')
    restored = type(counter).restore(counter.state())
    rs = restored.signals(model, router, x, seen, route, refs, batch_size, 'cpu')
    checks['save_restore_exact'] = all(np.array_equal(sig[k], rs[k]) for k in sig)
    checks['label_blind_prediction'] = 'y' not in sig and 'true_class' not in sig
    checks['no_receiver_mutation'] = before == complete_hash(model, router)
    checks['all_extra_dependencies_mature'] = counter.function(model, route, refs)['all_dependencies_mature']
    checks['no_receiver_ownership_or_FAR_claim'] = (counter.state()['installs_classifier_knowledge'] is False
                                                  and counter.state()['population_FAR_certified'] is False)
    # Negative counter-evidence is mandatory: deleting it must not drop a veto silently.
    if counter.heads:
        state = copy.deepcopy(counter.state()); state['negative_heads'].pop(next(iter(state['negative_heads'])))
        state = sealed({k: v for k, v in state.items() if k != 'state_digest'})
        try:
            type(counter).restore(state)
            checks['missing_counter_head_rejected'] = False
        except Rejected:
            checks['missing_counter_head_rejected'] = True
        state = copy.deepcopy(counter.state()); p = next(iter(state['negative_heads'].values()))
        p['offer']['model_version'] = 'wrong-source'
        p['offer']['digest'] = digest({k: v for k, v in p['offer'].items() if k != 'digest'})
        state = sealed({k: v for k, v in state.items() if k != 'state_digest'})
        try:
            type(counter).restore(state)
            checks['head_source_version_rejected'] = False
        except Rejected:
            checks['head_source_version_rejected'] = True
        changed = type(counter).restore(counter.state())
        next(iter(changed.heads.values()))['offer']['bias'] += .25
        try:
            checks['counter_head_changes_guard_fingerprint'] = changed.function(model, route, refs)['fingerprint'] != function
        except Rejected:
            checks['counter_head_changes_guard_fingerprint'] = True
    if hasattr(counter, 'offsets') and counter.offsets:
        changed = type(counter).restore(counter.state())
        changed.offsets[next(iter(changed.offsets))] += .25
        checks['FIT_offset_changes_guard_fingerprint'] = changed.function(model, route, refs)['fingerprint'] != function
        state = copy.deepcopy(counter.state()); state['fit_evidence']['quantile'] = .95
        state = sealed({k: v for k, v in state.items() if k != 'state_digest'})
        try:
            type(counter).restore(state)
            checks['changed_FIT_quantile_rejected'] = False
        except Rejected:
            checks['changed_FIT_quantile_rejected'] = True
    # Own geometry remains a hard veto irrespective of peer head scores.
    base = receiver_signals(model, router, x, seen, route.metadata['task'], route.metadata['class_id'], batch_size, 'cpu')
    p = route.signals(base, x)['patch_logit']
    checks['owned_support_never_weakened'] = np.all(~counter.ledger.own.veto(x, counter.local) |
                                                  counter.veto(x, base['imported_features'], p))
    # Pure label-blind inputs are shape-checked; no malformed/nonfinite evidence is accepted.
    try:
        counter.veto(x, base['imported_features'], np.r_[p, 0])
        checks['misaligned_signal_rejected'] = False
    except Rejected:
        checks['misaligned_signal_rejected'] = True
    bad = base['imported_features'].copy(); bad[0, 0] = np.nan
    try:
        counter.veto(x, bad, p)
        checks['nonfinite_signal_rejected'] = False
    except Rejected:
        checks['nonfinite_signal_rejected'] = True
    if not all(checks.values()):
        raise AssertionError(checks)
    return checks


def development(ckpt, graph, model, router, route, refs, ledger, counter, a):
    from fed_learning.data.denice_clean_roles import CleanRoleData
    from appliance.stable_head import stable_signals
    from appliance.portable_route import transitions
    roles = CleanRoleData(a.roles, source_data_dir=a.data)
    cid, c, donor = route.metadata['receiver'], route.metadata['class_id'], route.metadata['donor']
    frozen = ledger.freeze(counter.required, c)
    pools = [('receiver_validation', cid, set(ckpt['seen_classes'])),
             ('donor_validation_positive', donor, {c})]
    from appliance.selector import lookup
    edge = lookup(graph['alpha_debug'], cid)
    for peer, weight in zip(edge['group_ids'], edge['alphas']):
        if peer == cid or weight <= 0:
            continue
        raw = lookup(ckpt['client_algorithm_states'], peer); ps = raw.get('denice', raw)
        supported = {int(k) for k, e in ps['appliance_base_sketch_shield_state']['memory']['entries'].items()
                     if e['task'] < ckpt['task']}
        if supported:
            pools.append((f'peer_{peer}_validation_negative', int(peer), supported))
    rows = []
    before = complete_hash(model, router)
    for name, owner, classes in pools:
        x, y, ids = roles.client_role(owner, 'validation')
        ix = np.concatenate([np.flatnonzero(y == k)[:a.per_class] for k in sorted(classes)])
        x, y, ids = x[ix], y[ix], ids[ix]
        if not len(x):
            continue
        old = stable_signals(model, router, x, ckpt['seen_classes'], route, refs, a.batch_size, 'cpu')
        new = counter.signals(model, router, x, ckpt['seen_classes'], route, refs, a.batch_size, 'cpu')
        hit = lambda s: s['signature_valid'] & (s['signature_score'] > route.metadata['tau']) & (s['margin'] > route.metadata['gamma'])
        previous = hit(old) & ~ledger.own.veto(x, counter.local)
        union = hit(old) & ~frozen_veto(frozen, x)
        contrastive = hit(new)
        def metric(h):
            pred = np.where(h, c, old['local_pred'])
            return dict(activations=int(h.sum()), false_activations=int((h & (y != c)).sum()),
                        positive_activations=int((h & (y == c)).sum()),
                        transitions=transitions(y, old['local_pred'], pred))
        rows.append(dict(pool=name, owner=owner, rows=len(y),
            target_rows=int((y == c).sum()), class_counts={str(k): int((y == k).sum()) for k in sorted(set(y))},
            row_ids_sha256=hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest(),
            own_only=metric(previous), union_peer_veto=metric(union), contrastive=metric(contrastive),
            identical_final_tau_gamma=True, final_test_opened=False))
        print(f'Counter development receiver={cid} class={c} {name}: own={previous.sum()} union={union.sum()} counter={contrastive.sum()}', flush=True)
    if complete_hash(model, router) != before:
        raise AssertionError('Development mutated receiver')
    return rows
