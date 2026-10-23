"""Retrospective current-CAL acceptance of locked receiver-aware candidates.

This creates NEW experimental guard declarations, not replacements for existing
CUDA certificates. No installs/backbone updates/test or historical raw CAL.
Simulation cannot prove the old run transmitted row-level provenance receipts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from appliance.base_sketch_shield import CurrentBaseSketchShield
from appliance.closure import effective_linear
from appliance.config import Rejected
from appliance.current_calibration_data import CurrentCalibrationData
from appliance.distributed_calibration import (CalibrationSession, GuardGrid, LocalCalibrationEndpoint,
                                              select_distributed_guard, current_acceptance)
from appliance.guarded_head import put_head
from appliance.imported_route import ROUTE_RULES, stratified_roles
from appliance.portable_route import SharedSketch, ProtectedRoute, prototype_summary, fit_support_cosine_floor, PORTABLE_RULES
from appliance.provenance_protection import ProvenanceProtection, export_support, frozen_veto
from appliance.receiver_aware_discovery import head_offer, maturity_precheck
from appliance.selector import lookup
from appliance.stable_head import guard_function, stable_signals
from appliance.state import complete_hash, digest, boundary_hash, write_json
from eval_checkpoint import _make_denice_client_model


def protection(ckpt, graph, cid, required, protection_only=False):
    states = {int(i): s.get('denice', s) for i, s in ckpt['client_algorithm_states'].items()}
    ledger = ProvenanceProtection(CurrentBaseSketchShield.restore(states[cid]['appliance_base_sketch_shield_state']))
    alpha = lookup(graph['alpha_debug'], cid)
    peers = {int(j): float(w) for j, w in zip(alpha['group_ids'], alpha['alphas']) if int(j) != cid and w > 0}
    for j, weight in peers.items():
        own = CurrentBaseSketchShield.restore(states[j]['appliance_base_sketch_shield_state'])
        classes = (sorted(int(c) for c, e in own.memory.entries.items() if e['task'] < ckpt['task'])
                   if protection_only else
                   [c for c in required if str(c) in own.memory.entries and int(states[j]['neuron_ages']['fc2'][c]) >= 2])
        if not classes:
            continue
        packet = export_support(own, digest(ckpt['client_model_states'][j]), digest(states[j]['connection_masks']))
        receipt = dict(sender=j, receiver=cid, task=int(ckpt['task']), round=int(ckpt['final_round_id']),
            kind='protection_only' if protection_only else 'aggregation', alpha=weight, class_rows=classes, sender_model_version=packet['sender_model_version'],
            receiver_model_version=digest(ckpt['client_model_states'][cid]), dependency_version=packet['dependency_version'])
        receipt['receipt_id'] = digest(receipt)
        ledger.receive(packet, receipt, current_task=ckpt['task'], authorized_senders=peers)
    return ledger


def original_model(ckpt, cid):
    model, router = _make_denice_client_model(ckpt, cid, 'cpu')
    state = lookup(ckpt['client_algorithm_states'], cid)
    state = state.get('denice', state)
    for c, entry in state.get('appliance_guarded_head_entries', {}).items():
        put_head(model, int(c), entry['backup'])
    return model, router


def candidate(ckpt, graph, decision, a):
    cid, donor, c = decision['receiver'], decision['selected_FIT_donor'], decision['class_id']
    model, router = original_model(ckpt, cid); dm, _ = original_model(ckpt, donor)
    original = complete_hash(model, router)
    views = {i: CurrentCalibrationData(a.calibration_store, i, int(ckpt['task']),
                                     ckpt['config']['denice_data_roles_sha256']) for i in (cid, donor)}
    task = int(ckpt['task']); classes = views[cid].store['task_classes'][str(task)]
    head = head_offer(dm, c, digest(dm.state_dict()))
    pre = maturity_precheck(model, router, head, task, ckpt['seen_classes'])
    if not pre['eligible']:
        raise Rejected(pre['reason'])
    required = sorted(k for k in ckpt['seen_classes'] if k not in classes and int(model.unit_ranks['fc2'][k]) >= 2)
    ledger = protection(ckpt, graph, cid, required, a.protection_only or a.contrastive)
    if a.protection_only or a.contrastive:
        required = sorted(set(required) | {k for r in ledger.receipts.values() for k in r['receipt']['class_rows']})
    snapshot = ledger.freeze(required, c)
    pools = {i: view.current_pool(i, 'calibration', classes) for i, view in views.items()}
    split = {i: stratified_roles(p, ROUTE_RULES['seed'] + i) for i, p in pools.items()}
    fit = split[donor]['fit']; pos = fit[pools[donor]['y'][fit] == c]
    sketch = SharedSketch(tuple(ckpt['config']['input_shape']), 16, views[cid].store['metadata_sha256'])
    z, valid = sketch.features(pools[donor]['X'][pos]); proto, var, support = prototype_summary(z, valid)
    floor = fit_support_cosine_floor(support)
    w, b = effective_linear(dm, 'fc2')
    route = ProtectedRoute(dict(kind='protected_imported_route', version=PORTABLE_RULES['version'],
        receiver=cid, donor=donor, class_id=c, task=task, tau=1., gamma=0., beta=1.,
        receiver_feature_hash=boundary_hash(model, True), signature=sketch.manifest(), support=support,
        role_manifest_sha256=views[cid].store['role_manifest_sha256'], margin_reference_scope='import_context',
        guard_calibration_version='experimental-current-peer-protection-v1', fit_support_min_cosine=floor,
        fit_support_policy='donor-FIT p95 radius; fixed before SELECTION', self_confidence=PORTABLE_RULES['self_confidence']),
        proto, var, w[c].numpy(), float(b[c]))
    refs = pre['reference_classes']
    counter = None
    if a.contrastive:
        from tools.audit_appliance_contrastive_protection import build_counter
        counter = build_counter(ckpt, ledger, required, c, donor)
        counter.function(model, route, refs)  # Precheck all extra head dependencies before opening SELECTION.
        if a.fit_counter:
            from appliance.fit_contrastive_protection import FitContrastiveProtection
            counter = FitContrastiveProtection.fit(counter, model, router, route, refs,
                pools[donor]['X'][pos], pools[donor]['rows'][pos], views[donor], ckpt['seen_classes'], a.batch_size)
        write_json(a.out/f'counter_receiver_{cid}_class_{c}.json', counter.state())
    def endpoints(session, holdout):
        result = {}
        for i, pool in pools.items():
            private = {}
            for role in ('fit', 'selection', 'holdout'):
                indices = split[i][role] if role != 'holdout' or holdout else np.zeros(0, np.int64)
                x = pool['X'][indices]
                if len(x):
                    if counter is not None:
                        sig = counter.signals(model, router, x, ckpt['seen_classes'], route, refs, a.batch_size, 'cpu')
                    else:
                        sig = stable_signals(model, router, x, ckpt['seen_classes'], route, refs, a.batch_size, 'cpu')
                        sig['signature_valid'] &= ~frozen_veto(snapshot, x)
                    sig['local_confidence'] = np.zeros(len(x), np.float64)
                else:
                    sig = dict(signature_score=np.zeros(0), signature_valid=np.zeros(0, bool),
                        margin=np.zeros(0), local_confidence=np.zeros(0), local_pred=np.zeros(0, np.int64))
                private[role] = dict(y=pool['y'][indices], row_id=pool['rows'][indices], signals=sig)
            result[i] = LocalCalibrationEndpoint(i, session, private)
        return result
    def session(stage):
        declaration = dict(policy='experimental_peer_negative_head_v1' if counter is not None else 'experimental_peer_BASE_veto_v1',
            protection_only_support=a.protection_only,
            guard_function=guard_function(model, route, refs)['fingerprint'],
            packet_sha256=hashlib.sha256(route.packet()).hexdigest(),
            protection_snapshot_digest=snapshot['state_digest'], source_receipts_simulated=True)
        if counter is not None:
            declaration.update(counter_state_digest=counter.state()['state_digest'],
                               counter_function=counter.function(model, route, refs)['fingerprint'])
        return CalibrationSession(cid, donor, c, task, ckpt['final_round_id'], tuple(classes), original,
            digest(declaration), ckpt['config']['denice_data_roles_sha256'], f'locked-FIT-{cid}-{c}-{stage}'), declaration
    s, decl = session('selection'); eps = endpoints(s, False)
    proposed = GuardGrid.from_quantiles(s, [eps[cid].quantiles(), eps[donor].quantiles()], floor)
    grid = GuardGrid(s, proposed.taus, proposed.gammas, [1.])
    selected = select_distributed_guard(s, grid, eps[cid].count_packet(grid), eps[donor].count_packet(grid))
    route.metadata.update(tau=selected['tau'], gamma=selected['gamma'])
    s, decl = session('holdout'); final = GuardGrid(s, [selected['tau']], [selected['gamma']], [1.])
    out = a.out/f'receiver_{cid}_class_{c}'; out.mkdir()
    write_json(out/'guard_lock_before_HOLDOUT.json', dict(pair=decision, selected=selected, declaration=decl,
        session=s.manifest(), protection_snapshot_digest=snapshot['state_digest'],
        holdout_predictions_opened=False, acceptance_recall_gate=.95, no_HOLDOUT_fallback=True))
    eps = endpoints(s, True)
    for ep in eps.values():
        ep.lock_guard(final)
    acceptance = current_acceptance(s, final, eps[cid].count_packet(final, 'holdout'), eps[donor].count_packet(final, 'holdout'))
    if complete_hash(model, router) != original:
        raise AssertionError('Acceptance changed receiver state')
    result = dict(receiver=cid, donor=donor, class_id=c, selection=selected, acceptance=acceptance,
        thresholds_locked_before_HOLDOUT=True, installation_authorized=False, installed=False,
        source_row_receipts_simulated=True, certificate_revalidated_from_old_run=False)
    if counter is not None:
        from tools.audit_appliance_contrastive_protection import controls, development
        result['counter_controls'] = controls(counter, model, router, route, refs, pools[donor]['X'][pos][:32],
                                              ckpt['seen_classes'], a.batch_size)
        state = lookup(ckpt['client_algorithm_states'], cid)
        entry = state.get('denice', state).get('appliance_guarded_head_entries', {}).get(c)
        if entry is None:
            entry = state.get('denice', state).get('appliance_guarded_head_entries', {}).get(str(c))
        original_task_install = entry is not None and entry['task'] == task
        result['original_task_install'] = original_task_install
        if a.development and (original_task_install or acceptance['passed_current_scope']):
            result['development'] = development(ckpt, graph, model, router, route, refs, ledger, counter, a)
        result['negative_head_count'] = len(counter.heads)
        result['negative_head_FP32_tensor_bytes'] = sum((len(p['offer']['weight']) + 1) * 4 for p in counter.heads.values())
        result['fixed_FIT_p99_offsets'] = a.fit_counter
    write_json(out/'result.json', result)
    return result


def run(a):
    a.out.mkdir(parents=True, exist_ok=False)
    decisions = json.loads(a.decisions.read_text())
    # File-level lock precedes every new CAL SELECTION/HOLDOUT prediction.
    write_json(a.out/'FIT_candidates_lock.json', dict(decisions=decisions,
        decisions_sha256=hashlib.file_digest(a.decisions.open('rb'), 'sha256').hexdigest(),
        no_HOLDOUT_fallback=True, test_opened=False,
        contrastive_policy=a.contrastive,
        FIT_p99_offsets=a.fit_counter,
        negative_source_choice='patch donor first, then largest owned BASE count, then sender ID',
        negative_guard='own hard veto; every peer-supported old class uses box AND negative head >= imported head',
        negative_margin_fixed=0, validation_scope='CAL-pass candidates plus all original task installs',
        acceptance_recall_gate=.95, final_test_opened=False, production_enabled=False))
    write_json(a.out/'completion.json', dict(completed=False))
    ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    graph = next(g for g in json.loads(a.graphs.read_text()) if g['task'] == ckpt['task'] and g['round'] == ckpt['final_round_id'])
    results = []
    for d in decisions:
        if d['selected_FIT_donor'] is None:
            results.append(dict(receiver=d['receiver'], class_id=d['class_id'], skipped='no mature FIT candidate'))
            continue
        try:
            r = candidate(ckpt, graph, d, a)
        except Rejected as e:
            r = dict(receiver=d['receiver'], class_id=d['class_id'], rejected=e.reason, detail=e.detail)
        results.append(r)
        write_json(a.out/'results.json', results)
        print(f'Peer-protection CAL receiver={d["receiver"]}, class={d["class_id"]}: {r.get("acceptance", r.get("rejected"))}', flush=True)
    report = dict(completed=True, request_count=len(decisions),
        protection_only_support=a.protection_only,
        contrastive_peer_support=a.contrastive,
        fixed_FIT_p99_offsets=a.fit_counter,
        acceptance_evaluated=sum('acceptance' in r for r in results),
        current_CAL_pass=sum(r.get('acceptance', {}).get('passed_current_scope', False) for r in results),
        source_receipts_simulated=True, installed=0, production_enabled=False,
        test_opened=False, raw_historical_CAL_opened=False, backbone_training=False,
        inference='new CPU FP32 guard; old CUDA certificates were not modified or revalidated',
        communication_not_measured=True, limitation='same retrospective current CAL fixture; not independent final evidence')
    write_json(a.out/'completion.json', report);print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('checkpoint', 'graphs', 'decisions', 'calibration-store', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--protection-only', action='store_true')
    p.add_argument('--contrastive', action='store_true', help='Experimental peer box AND negative-head counter-evidence')
    p.add_argument('--fit-counter', action='store_true', help='Fix per-negative-head offsets from donor current FIT p99')
    p.add_argument('--development', action='store_true')
    p.add_argument('--roles', type=Path)
    p.add_argument('--data', type=Path)
    p.add_argument('--per-class', type=int, default=256)
    a = p.parse_args();torch.set_num_threads(4)
    if a.fit_counter and not a.contrastive:
        p.error('--fit-counter requires --contrastive')
    if a.development and (not a.contrastive or not a.roles or not a.data):
        p.error('--development requires --contrastive, --roles and --data')
    with threadpool_limits(limits=1):
        run(a)
