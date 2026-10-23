"""Development-only P1/P2 probes on the saved Task3 request panel.

No installs, no threshold selection, no CAL SELECTION/HOLDOUT predictions and
no test. Request/offer metadata were locked by the original training service.
P0 receipts are reconstructed conservatively as a simulation, not provenance
that the original run actually transmitted.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from appliance.base_sketch_shield import CurrentBaseSketchShield
from appliance.config import Rejected
from appliance.current_calibration_data import CurrentCalibrationData
from appliance.guarded_head import put_head
from appliance.provenance_protection import ProvenanceProtection, export_support
from appliance.receiver_aware_discovery import head_offer, maturity_precheck, fit_probe, rank_fit_probes
from appliance.selector import lookup
from appliance.state import complete_hash, digest, write_json
from eval_checkpoint import _make_denice_client_model


def run(a):
    a.out.mkdir(parents=True, exist_ok=False)
    write_json(a.out/'completion.json', dict(completed=False))
    ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    graphs = json.loads(a.graphs.read_text(encoding='utf-8'))
    history = json.loads(a.history.read_text(encoding='utf-8'))
    task = int(ckpt['task'])
    graph = next(r for r in graphs if r['task'] == task and r['round'] == ckpt['final_round_id'])
    d = next(r for r in history if r['task'] == task and r['phase'] == 'task_finalized')['discovery']
    states = {int(i): s.get('denice', s) for i, s in ckpt['client_algorithm_states'].items()}
    role_sha = ckpt['config']['denice_data_roles_sha256']
    # All original selected requests, not a test-derived class/pair shortlist.
    write_json(a.out/'request_panel_lock.json', dict(pairs=d['pairs'], task=task,
        rule='original selected request panel; all positive-edge donors offering same class',
        donor_quality_source='original current CAL FIT offers', final_test_opened=False,
        CAL_holdout_for_selection=False, no_threshold_tuning=True))
    models, views = {}, {}
    def model(i):
        if i not in models:
            m, r = _make_denice_client_model(ckpt, i, 'cpu')
            # Counterfactual pre-install head view, not a rewrite of a certificate.
            for c, e in states[i].get('appliance_guarded_head_entries', {}).items():
                put_head(m, int(c), e['backup'])
            models[i] = (m, r)
        return models[i]
    def view(i):
        if i not in views:
            views[i] = CurrentCalibrationData(a.calibration_store, i, task, role_sha)
        return views[i]
    probes, decisions = [], []
    for pair in d['pairs']:
        cid, c = pair['receiver'], pair['class_id']
        m, router = model(cid)
        shield = CurrentBaseSketchShield.restore(states[cid]['appliance_base_sketch_shield_state'])
        ledger = ProvenanceProtection(shield)
        alpha = lookup(graph['alpha_debug'], cid)
        weights = {int(j): float(w) for j, w in zip(alpha['group_ids'], alpha['alphas']) if int(j) != cid and w > 0}
        current = view(cid).store['task_classes'][str(task)]
        required = sorted(c for c in ckpt['seen_classes'] if c not in current and int(m.unit_ranks['fc2'][c]) >= 2)
        for j, weight in weights.items():
            source = CurrentBaseSketchShield.restore(states[j]['appliance_base_sketch_shield_state'])
            rows = [k for k in required if int(states[j]['neuron_ages']['fc2'][k]) >= 2 and str(k) in source.memory.entries]
            if not rows:
                continue
            packet = export_support(source, digest(ckpt['client_model_states'][j]), digest(states[j]['connection_masks']))
            receipt = dict(sender=j, receiver=cid, task=task, round=ckpt['final_round_id'], kind='aggregation',
                alpha=weight, class_rows=rows, sender_model_version=packet['sender_model_version'],
                receiver_model_version=digest(ckpt['client_model_states'][cid]), dependency_version=packet['dependency_version'])
            receipt['receipt_id'] = digest(receipt)
            ledger.receive(packet, receipt, current_task=task, authorized_senders=weights)
        missing = ledger.coverage(required)['missing_protection_classes']
        candidates = [o for o in d['offers'] if o['class_id'] == c and o['donor'] in weights]
        result = []
        for offer in candidates:
            j = offer['donor']; dm, _ = model(j)
            record = dict(receiver=cid, donor=j, class_id=c, donor_quality_lcb=offer['quality_lcb'])
            head = head_offer(dm, c, digest(dm.state_dict()))
            try:
                pre = maturity_precheck(m, router, head, task, ckpt['seen_classes'])
                record['precheck'] = pre
                if not pre['eligible']:
                    record['reason'] = pre['reason']
                elif missing:
                    record['reason'] = 'missing_potential_peer_support'
                else:
                    before = complete_hash(m, router)
                    scores = fit_probe(m, head, pre, view(cid), view(j), ledger, required, a.batch_size)
                    record['FIT_probe'] = scores
                    if complete_hash(m, router) != before:
                        raise AssertionError('FIT probe mutated receiver')
                    result.append(scores)
            except Rejected as e:
                record['reason'] = e.reason
            probes.append(record)
        ranked = rank_fit_probes(result, {o['donor']: o['quality_lcb'] for o in candidates})
        decisions.append(dict(receiver=cid, class_id=c, original_donor=pair['donor'],
            candidates=len(candidates), maturity_eligible=sum(r['precheck']['eligible'] for r in probes
                if r['receiver'] == cid and r['class_id'] == c and 'precheck' in r),
            FIT_ranked_donors=[r['donor'] for r in ranked], selected_FIT_donor=ranked[0]['donor'] if ranked else None,
            future_CAL_acceptance_unchanged=True, installed=False))
        write_json(a.out/'probes.json', probes);write_json(a.out/'decisions.json', decisions)
        print(f'Receiver-aware FIT receiver={cid}, class={c}, donors={len(candidates)}, probed={len(ranked)}', flush=True)
    summary = dict(completed=True, original_requests=len(d['pairs']), donor_candidates=len(probes),
        maturity_rejected_before_capsule=sum(r.get('precheck', {}).get('eligible') is False for r in probes),
        FIT_probed=sum('FIT_probe' in r for r in probes),
        donor_selection_changed=sum(r['selected_FIT_donor'] is not None and r['selected_FIT_donor'] != r['original_donor'] for r in decisions),
        no_eligible_donor=sum(r['selected_FIT_donor'] is None for r in decisions),
        new_guard_CAL_pass_count=None, installation_count=0, production_enabled=False,
        retained_acceptance_recall_gate=.95, raw_historical_CAL_opened=False,
        CAL_selection_holdout_predictions_opened=False, final_test_opened=False,
        calibration_access=[r for v in views.values() for r in v.access_log],
        receiver_capsule_and_setup_wire_not_measured=True,
        limitation='CPU retrospective compatibility scores; not native installs or acceptance evidence')
    write_json(a.out/'completion.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'calibration_access'}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('checkpoint', 'graphs', 'history', 'calibration-store', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=512)
    a = p.parse_args()
    torch.set_num_threads(4)
    with threadpool_limits(limits=1):
        run(a)
