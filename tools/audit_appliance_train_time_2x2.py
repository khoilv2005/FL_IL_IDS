"""Frozen class-6 mechanism ablation; native routed predictions, no refitting.

VALIDATION is role-disjoint from the sealed BASE/CAL/FIT experiment. Earlier
CME/audits used this role, so global untouched status is deliberately UNKNOWN.
This offline qualification cannot authorize a production install/native smoke.
"""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from appliance.evaluate import Predictor
from appliance.runner import load_input
from appliance.selector import recorded_graph
from appliance.state import complete_hash, write_json
from appliance.train_time_transfer import RULES, put_readout
from eval_checkpoint import _make_denice_client_model
from fed_learning.data.denice_clean_roles import CleanRoleData, file_sha256
from tools.run_appliance_train_time_transfer import counters


VARIANTS = ('baseline', 'availability_only', 'update_only', 'full_v2')
CAP = 1024


def content_hash(x):
    return hashlib.sha256(np.ascontiguousarray(x, dtype='<f4').tobytes()).hexdigest()


def select_witnesses(roles, groups):
    """Choose graph witnesses from counts, never prediction quality/labels."""
    totals = {c: 0 for c in range(6)}
    peers = sorted(set(groups[2]) - {2, 62})
    owners = []
    counts = roles.manifest['clients']
    while peers and any(n < 32 for n in totals.values()):
        def gain(owner):
            v = counts[str(owner)]['role_class_counts']['validation']
            return sum(min(max(0, 32-totals[c]), int(v.get(str(c), 0))) for c in totals)
        owner = min(peers, key=lambda p: (-gain(p), p))
        if gain(owner) == 0:
            break
        owners.append(owner)
        for c in totals:
            totals[c] += min(CAP, int(counts[str(owner)]['role_class_counts']['validation'].get(str(c), 0)))
        peers.remove(owner)
    return owners, totals


def qualify(records, predictions):
    y = np.array([v['class_id'] for v in records], dtype=np.int64)
    metrics = {name: counters(predictions['baseline'], predictions[name], y, 6) for name in VARIANTS}
    per_class = {}
    for c in range(12):
        ix = y == c
        n = int(ix.sum())
        per_class[str(c)] = dict(rows=n, variants={name: dict(
            correct=int((pred[ix] == c).sum()),
            accuracy=float((pred[ix] == c).mean()) if n else None,
            target_false_positives=int((pred[ix] == 6).sum()) if c != 6 else None)
            for name, pred in predictions.items()})
    transitions = {}
    for before, after in (('baseline', 'availability_only'), ('baseline', 'update_only'),
                          ('availability_only', 'full_v2'), ('update_only', 'full_v2')):
        transitions[f'{before}_to_{after}'] = counters(predictions[before], predictions[after], y, 6)
    return dict(metrics=metrics, per_class=per_class, transitions=transitions)


def run(a):
    a.out.mkdir(parents=True, exist_ok=False)
    write_json(a.out/'completion.json', dict(completed=False))
    prior = json.loads((a.prepared/'completion.json').read_text())
    if not prior['completed']:
        raise ValueError('Frozen transfer incomplete')
    ckpt, hashes = load_input(a.checkpoint, 1, 19)
    if any(hashes[k] != prior['protocol'][k] for k in hashes):
        raise ValueError('Task1 checkpoint/graph binding changed')
    roles = CleanRoleData(a.roles, source_data_dir=a.data)
    if file_sha256(a.roles/'role_manifest.json') != prior['protocol']['role_manifest_sha256']:
        raise ValueError('Role authority changed')
    if ckpt['config'].get('denice_cl_method') != 'legacy' or sorted(ckpt['seen_classes']) != list(range(12)):
        raise ValueError('Legacy Task1 required')
    groups, alphas = recorded_graph(ckpt, 1, 19)
    if alphas[2].get(62, 0) <= 0:
        raise ValueError('Donor not a positive-alpha Task1 neighbor')
    witnesses, expected = select_witnesses(roles, groups)
    update_dir = a.prepared/'receiver_2_donor_62_class_6'
    row = torch.load(update_dir/'trained_row.pt', map_location='cpu', weights_only=False)
    lock = json.loads((update_dir/'update_lock_before_CAL_development.json').read_text())
    model, router = _make_denice_client_model(ckpt, 2, 'cpu')
    baseline_hash = complete_hash(model, router)
    if row['metadata']['receiver_sha256'] != baseline_hash:
        raise ValueError('Update receiver function binding changed')
    candidates = {}
    for name in VARIANTS:
        m, r = copy.deepcopy(model), copy.deepcopy(router)
        if name in ('update_only', 'full_v2'):
            put_readout(m, 6, row['weight'], row['bias'])
        if name in ('availability_only', 'full_v2'):
            r.episode_classes[1] = sorted(set(r.episode_classes.get(1, [])) | {6})
        candidates[name] = (m, r)
    frozen_hashes = {name: complete_hash(m, r) for name, (m, r) in candidates.items()}
    if frozen_hashes['full_v2'] != lock['candidate_sha256']:
        raise ValueError('Frozen Full V2 function changed')
    # Availability-only must leave all model parameters/masks/ranks intact.
    if complete_hash(candidates['availability_only'][0], router) != baseline_hash:
        raise ValueError('Availability-only changed model state')
    for name, (m, r) in candidates.items():
        for key, value in model.state_dict().items():
            other = m.state_dict()[key]
            if key in ('fc2.weight', 'fc2.bias'):
                keep = [c for c in range(model.fc2.out_features) if c != 6]
                if not torch.equal(value[keep], other[keep]):
                    raise ValueError('Non-target head row changed')
            elif not torch.equal(value, other):
                raise ValueError('Receiver prefix/buffers changed')
        expected_router = copy.deepcopy(router)
        expected_router.episode_classes = copy.deepcopy(r.episode_classes)
        if complete_hash(model, expected_router) != complete_hash(model, r):
            raise ValueError('Router state changed beyond availability')
    protocol = dict(kind='native-routed class6 frozen 2x2 role-disjoint qualification',
        **hashes, role_manifest_sha256=prior['protocol']['role_manifest_sha256'],
        prepared_completion_sha256=file_sha256(a.prepared/'completion.json'),
        trained_row_file_sha256=file_sha256(update_dir/'trained_row.pt'),
        candidate_hashes=frozen_hashes, pair=dict(receiver=2, donor=62, class_id=6),
        primary_inference='pred_hard, existing binary_cosine router, seen classes 0..11',
        earlier_all_seen_primary_unchanged=True, variants=list(VARIANTS),
        update_intervention='frozen trained fc2 row/bias plus original sealed mask=1 and maturity=2 registration',
        availability_intervention='add class6 to existing Task1 episode_classes only; no refit',
        update_frozen=True, thresholds_tuned=False, refitting=False,
        witness_selection='graph positive-alpha peers, greedy metadata coverage up to 32/class, ID tie',
        witness_owners=witnesses, expected_old_counts_before_content_dedup=expected,
        witness_alphas={str(o): alphas[2][o] for o in witnesses},
        evaluation_role='original validation; receiver current classes, donor target, graph witnesses old classes',
        cap_per_owner_class=CAP, sampling='ascending original row IDs; global feature-content dedup',
        role_disjoint_from_sealed_BASE_CAL_FIT=True,
        globally_untouched_evidence_established=False,
        independence_reason='prior CME and APPLIANCE audits accessed validation; complete prior row/content exposure ledger unavailable',
        evaluation_is_development_not_final_confirmation=True,
        new_native_empirical_gates=dict(target_min_rows=32, recall=.95, negative_min_rows=32,
            pooled_and_each_observed_class_FAR=.001, negative_break=0,
            old_accuracy_drop=RULES['max_development_old_accuracy_drop'], rescue_positive=True),
        broad_evidence_coverage_check='>=32 unique feature contents for each seen negative class; qualification only, not a changed production CAL gate',
        population_FAR_claim=False, CAL_acceptance_authorized=False,
        production_install_authorized=False, native_smoke_authorized=False,
        historical_raw_CAL_reads=0, BASE_reads=0, FIT_reads=0, final_test_opened=False,
        retrospective_pair_selection=True, raw_features_transmitted=0,
        source_sha256=file_sha256(Path(__file__)), training_rules_unchanged=RULES)
    # Written before loading any VALIDATION example or computing outcomes.
    write_json(a.out/'protocol_before_validation.json', protocol)
    seen_content = {}
    records = []
    predictions = {name: [] for name in VARIANTS}
    predictor = Predictor(list(range(12)), 'cpu', RULES['batch_size'])
    pools = [(62, [6], 'donor_positive'), (2, list(range(6, 12)), 'receiver_current_negative')]
    pools += [(owner, list(range(6)), 'graph_old_witness') for owner in witnesses]
    pool_reports = []
    for owner, classes, scope in pools:
        x, y, rows = roles.client_role(owner, 'validation')
        selected = []
        duplicate = conflicts = 0
        for c in classes:
            indices = np.flatnonzero(y == c)
            indices = indices[np.argsort(rows[indices], kind='stable')]
            count = 0
            for ix in indices:
                h = content_hash(x[ix])
                if h in seen_content:
                    if seen_content[h] != int(c):
                        conflicts += 1
                        raise ValueError('Identical input has conflicting labels')
                    duplicate += 1
                    continue
                seen_content[h] = int(c)
                selected.append((int(ix), h))
                count += 1
                if count == CAP:
                    break
        ix = np.array([v[0] for v in selected], dtype=np.int64)
        local_records = [dict(owner=owner, row_id=int(rows[i]), class_id=int(y[i]),
            scope=scope, content_sha256=h) for i, h in selected]
        write_json(a.out/f'panel_{owner}_{scope}_before_prediction.json', dict(
            role='validation', rows=local_records, duplicates_excluded=duplicate,
            label_conflicts=conflicts, available_counts={str(c): int((y == c).sum()) for c in classes}))
        result = {name: predictor.records(m, r, x[ix]) for name, (m, r) in candidates.items()}
        for name in VARIANTS:
            # Availability must not change the router's sample task decision.
            if not np.array_equal(result[name]['task'], result['baseline']['task']):
                raise ValueError('Native sample route changed in frozen ablation')
            predictions[name].extend(result[name]['pred'].tolist())
        local_pred = {name: result[name]['pred'] for name in VARIANTS}
        for k, record in enumerate(local_records):
            record['pred'] = {name: int(local_pred[name][k]) for name in VARIANTS}
            record['predicted_task'] = int(result['baseline']['task'][k])
        records.extend(local_records)
        pool_reports.append(dict(owner=owner, scope=scope, duplicates_excluded=duplicate,
            **qualify(local_records, local_pred)))
        print(f'Native 2x2: owner={owner}, scope={scope}, unique rows={len(ix)}, Full recall={pool_reports[-1]["metrics"]["full_v2"]["target_recall"]}', flush=True)
        del x, y, rows, result
        gc.collect()
    predictions = {name: np.asarray(values, dtype=np.int64) for name, values in predictions.items()}
    pooled = qualify(records, predictions)
    full = pooled['metrics']['full_v2']
    counts = {int(c): v['rows'] for c, v in pooled['per_class'].items()}
    missing = [c for c in range(12) if c != 6 and counts[c] < 32]
    far_pass = all(v <= .001 for v in full['negative_FAR_by_class'].values())
    old_drop = (full['old_accuracy_before'] - full['old_accuracy_after']) if full['old_rows'] else None
    empirical_pass = bool(full['positive_rows'] >= 32 and full['negative_rows'] >= 32 and
        full['target_recall'] >= .95 and full['negative_FAR'] <= .001 and far_pass and
        full['negative_break'] == 0 and old_drop is not None and old_drop <= .01 and full['rescue'] > 0)
    if complete_hash(model, router) != baseline_hash:
        raise ValueError('Live baseline mutated')
    if any(complete_hash(m, r) != frozen_hashes[name] for name, (m, r) in candidates.items()):
        raise ValueError('Ablation function mutated during evaluation')
    n = full['negative_rows']
    report = dict(completed=True, protocol=protocol, pooled=pooled, pools=pool_reports,
        mechanism=dict(update_gain_with_registration=pooled['transitions']['availability_only_to_full_v2'],
            availability_gain_without_update=pooled['transitions']['baseline_to_availability_only']),
        empirical_observed_gate_pass=empirical_pass,
        broad_evidence_coverage_pass=not missing, classes_below32_or_absent=missing,
        population_FAR_certified=False,
        zero_FP_iid_95_percent_upper_bound_if_applicable=(1-.05**(1/n)) if n and full['negative_FAR'] == 0 else None,
        bound_assumptions_not_established='iid population sampling not established; content dedup alone does not establish independence',
        globally_untouched_evidence_established=False, production_install_authorized=False,
        native_smoke_authorized=False, decision='qualification_only_no_independent_acceptance',
        communication=next(r['communication'] for r in prior['reports'] if r['pair']['class_id'] == 6),
        qualification_wire_cost='offline local emulator, not implemented/measured as decentralized wire cost',
        historical_raw_CAL_reads=0, refitting=False, final_test_opened=False,
        full_training_started=False, integrity=dict(full_frozen_function_hash_match=True,
            native_task_decisions_equal=True, source_receiver_unchanged=True,
            all_four_functions_unchanged=True, global_content_dedup=True,
            rescue_break_accounting_exact=True, label_blind_predictor_inputs=True))
    write_json(a.out/'prediction_records.json', records)
    write_json(a.out/'completion.json', report)
    write_json(a.publish, report)
    print(json.dumps(dict(completed=True, rows=len(records), empirical_pass=empirical_pass,
        broad_coverage=not missing, native_smoke_authorized=False), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'prepared', 'roles', 'data', 'out', 'publish'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):
        run(args)
