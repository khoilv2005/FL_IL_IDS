"""Audit saved discovery/CAL aggregates and predictions without fitting models.

Reads metadata and the existing full-test prediction CSV only. No raw BASE/CAL,
checkpoint unpickling, threshold tuning, installation or new inference.
"""
import argparse
import gzip
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def member(z, suffix):
    matches = [n for n in z.namelist() if n.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f'Expected one member ending in {suffix}: {matches}')
    return json.loads(z.read(matches[0]))


def run(a):
    a.out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.results) as z:
        history = member(z, '/appliance/automatic_history.json')
        clusters = member(z, '/cluster_history.json')
        contract = member(z, '/appliance/service_contract.json')
        communication = member(z, '/appliance/communication.json')
        roles = member(z, 'denice_clean_roles/role_manifest.json')
        debug = member(z, '/denice_debug_history.json')
        guard_locks = {n.split('/appliance/', 1)[1]: json.loads(z.read(n))
                       for n in z.namelist() if '/appliance/' in n and
                       n.endswith('/guard_lock_before_CURRENT_HOLDOUT.json')}
        wire_records = {n.split('/appliance/', 1)[1]: [json.loads(line) for line in z.read(n).decode().splitlines()]
                        for n in z.namelist() if '/appliance/' in n and n.endswith('/application_wire.jsonl')}
    graphs = {(r['task'], r['round']): r for r in clusters}
    starts = {r['task']: r for r in debug if r.get('type') == 'task_start'}
    missing_provenance, selection_rows = [], []
    wire_by_outcome = Counter()
    transaction_rows, request_rows, task_rows, lifecycle_rows = [], [], [], []
    protocol_flags = dict(historical_raw_CAL_reads=0, discovery_test_opened=False,
                          discovery_holdout_used=False)
    for row in history:
        d = row['discovery']
        protocol_flags['historical_raw_CAL_reads'] += row['historical_raw_CAL_reads']
        protocol_flags['discovery_test_opened'] |= d['final_test_opened']
        protocol_flags['discovery_holdout_used'] |= d['holdout_predictions_used']
        for t in row['transactions']:
            p, tx, acc = t['pair'], t['transaction'], t.get('acceptance', {})
            transaction_rows.append(dict(task=row['task'], round=row['round'], phase=row['phase'],
                receiver=p['receiver'], donor=p['donor'], class_id=p['class_id'],
                applied=tx['applied'], reason=tx.get('reason', 'committed'),
                rollback_verified=tx.get('rollback_verified'), recall=acc.get('recall'),
                CAL_negative_rows=acc.get('receiver_rows'), CAL_positive_rows=acc.get('positive_rows'),
                CAL_far=acc.get('receiver_far'), CAL_break=acc.get('break_count'),
                setup_bytes=t.get('setup_capsule_bytes'), capability_bytes=t.get('capability_bytes')))
            prefix = (f'task_{row["task"]}_round_{row["round"]}_{row["phase"]}/'
                      f'receiver_{p["receiver"]}_class_{p["class_id"]}/')
            guard = guard_locks.get(prefix + 'guard_lock_before_CURRENT_HOLDOUT.json')
            if guard:
                s = guard['selected']
                selection_rows.append(dict(task=row['task'], receiver=p['receiver'], donor=p['donor'],
                    class_id=p['class_id'], applied=tx['applied'],
                    selection_recall=s['selection_target_activations'] / s['selection_positive_rows'],
                    holdout_recall=acc.get('recall'), tau=s['tau'], gamma=s['gamma']))
            wire_by_outcome[tx.get('reason', 'committed')] += sum(
                r['bytes'] for r in wire_records.get(prefix + 'application_wire.jsonl', []))
        for r in row['lifecycle']:
            m = r.get('monitor') or {}
            l = r['lifecycle']
            lifecycle_rows.append(dict(task=row['task'], round=row['round'], phase=row['phase'],
                receiver=r['receiver'], class_id=r['class_id'], state=l['state'], reason=l['reason'],
                head_exact=r['head_exact_after'], certificate_current=r['certificate_current'],
                monitor_rows=m.get('rows'), monitor_far=m.get('far'), monitor_break=m.get('break_count'),
                monitor_evaluated=m.get('monitor_evaluated')))
        if row['phase'] != 'task_finalized':
            continue
        for cid, gate in d['receiver_preflight'].items():
            if gate['reason'] != 'missing_owned_old_BASE_support':
                continue
            for c in gate['missing_owned_classes']:
                prior = [task for task, start in starts.items() if task < row['task'] and
                         int(start['client_data'].get(str(cid), {}).get('class_hist', {}).get(str(c), 0)) > 0]
                missing_provenance.append(dict(task=row['task'], receiver=int(cid), class_id=c,
                    prior_active_tasks=prior, status='previous_active_BASE_missing' if prior else
                    'not_previously_observed_current_BASE'))
        g = graphs[(row['task'], row['round'])]
        selected = {(p['receiver'], p['class_id']) for p in d['pairs']}
        selected_receivers = {p['receiver'] for p in d['pairs']}
        funnel = Counter()
        missing = Counter()
        for req in d['requests']:
            cid, c = req['receiver'], req['class_id']
            gate = d['receiver_preflight'][str(cid)]
            alpha = g['alpha_debug'][str(cid)]
            peers = {int(i) for i, weight in zip(alpha['group_ids'], alpha['alphas'])
                     if int(i) != cid and weight > 0}
            global_offers = [o for o in d['offers'] if o['class_id'] == c]
            donors = [o['donor'] for o in global_offers if o['donor'] in peers]
            if not gate['eligible']:
                stage = gate['reason']
            elif not global_offers:
                stage = 'no_offer_for_requested_class'
            elif not donors:
                stage = 'no_positive_neighbor_offer'
            elif (cid, c) in selected:
                stage = 'selected'
            elif cid in selected_receivers:
                stage = 'receiver_selected_for_other_class'
            elif d['coverage'][str(c)]['selected_receivers'] >= contract['max_receivers_per_class']:
                stage = 'class_budget'
            elif len(selected) >= contract['max_transactions']:
                stage = 'transaction_budget'
            else:
                stage = 'scheduler_not_selected'
            funnel[stage] += 1
            missing.update(gate['missing_owned_classes'])
            request_rows.append(dict(task=row['task'], receiver=cid, class_id=c, stage=stage,
                global_offers=len(global_offers), positive_neighbor_offers=len(donors),
                eligible_donors=donors, missing_owned_classes=gate['missing_owned_classes']))
        applied = [t for t in row['transactions'] if t['transaction']['applied']]
        task_rows.append(dict(task=row['task'], active_receivers=len(d['receiver_preflight']),
            requests=len(d['requests']), offers=len(d['offers']),
            donor_offer_classes=sorted({o['class_id'] for o in d['offers']}),
            request_classes=sorted({r['class_id'] for r in d['requests']}),
            receiver_preflight=dict(Counter(p['reason'] or 'eligible' for p in d['receiver_preflight'].values())),
            request_funnel=dict(funnel), missing_owned_classes_frequency=dict(missing),
            donor_reject_reasons=dict(Counter(r['reason'] for r in d['rejected_offers'])),
            attempted=len(row['transactions']), committed=len(applied),
            rejected=len(row['transactions']) - len(applied),
            eligible_request_class_pairs=sum(v['eligible_current_receivers'] for v in d['coverage'].values())))
    rejects = [r for r in transaction_rows if not r['applied']]
    acceptance_fails = [r for r in rejects if r['reason'] == 'STABLE_CURRENT_AGGREGATE_ACCEPTANCE_REQUIRED']
    acceptance_reasons = Counter()
    for r in acceptance_fails:
        if r['recall'] is None:
            acceptance_reasons['no_acceptance_report'] += 1
        elif r['recall'] < .95:
            acceptance_reasons['target_recall_below_95pct'] += 1
        elif r['CAL_far'] > 0 or r['CAL_break'] > 0:
            acceptance_reasons['negative_damage'] += 1
        else:
            acceptance_reasons['other_gate'] += 1
    with zipfile.ZipFile(a.evaluation) as z:
        evaluation = member(z, 'completion.json')
        evaluation_lock = member(z, 'pipeline_lock.json')
        if (not evaluation['completed'] or contract['role_sha'] != evaluation_lock['role_manifest_sha256']
                or contract['scope_mode'] != evaluation_lock['scope_mode']):
            raise ValueError('Training/evaluation role and deployment locks must match')
        clients = {r['receiver']: r for r in member(z, 'client_metrics.json')}
        # Diagnostic analysis of every active capability, never a fitted selector.
        targets = {}
        for cid, client in clients.items():
            active = [int(c) for c, route in client['appliance_routes'].items() if route['authorized']]
            if len(active) > 1:
                raise ValueError('This V1 audit expects one installed active capability per receiver')
            if active:
                targets[cid] = active[0]
        negatives = {cid: Counter() for cid in targets}
        all_rows = {cid: Counter() for cid in targets}
        break_rows = {cid: Counter() for cid in targets}
        with z.open('test_predictions.csv.gz') as packed, gzip.GzipFile(fileobj=packed) as stream:
            for df in pd.read_csv(stream, chunksize=250000,
                                  usecols=['client_id', 'y_true', 'MulticlassSelf', 'APPLIANCE']):
                for cid, sub in df[df.client_id.isin(targets)].groupby('client_id', sort=False):
                    c = targets[cid]
                    all_rows[cid].update(sub.y_true.tolist())
                    fp = sub[(sub.APPLIANCE == c) & (sub.y_true != c)]
                    negatives[cid].update(fp.y_true.tolist())
                    broken = sub[(sub.MulticlassSelf == sub.y_true) & (sub.APPLIANCE != sub.y_true)]
                    break_rows[cid].update(broken.y_true.tolist())
    commits = {(r['receiver'], r['class_id']): r for r in transaction_rows if r['applied']}
    install_records = {(t['pair']['receiver'], t['pair']['class_id']): t
                       for row in history for t in row['transactions'] if t['transaction']['applied']}
    false_activation = []
    for cid, c in targets.items():
        install = install_records[(cid, c)]
        acc = install['acceptance']
        scope = set(map(int, acc['receiver_far_by_class'])) | {c}
        prior = set(install['required_old_classes'])
        task = install['pair']['task']
        task_classes = set(starts[task]['new_classes'])
        rows_by_class = [dict(class_id=int(label), false_activations=int(n),
                             test_class_rows=int(all_rows[cid][label]),
                             false_activation_rate=n / all_rows[cid][label],
                             inside_initial_CAL_scope=label in scope,
                             required_owned_BASE_protection=label in prior,
                             inside_install_task=label in task_classes)
                         for label, n in negatives[cid].most_common()]
        false_activation.append(dict(receiver=cid, class_id=c, donor=install['pair']['donor'],
            initial_CAL_scope=sorted(scope), initial_CAL_negative_rows=acc['receiver_rows'],
            initial_CAL_far=acc['receiver_far'], initial_CAL_break=acc['break_count'],
            required_old_classes=sorted(prior), false_activations=sum(negatives[cid].values()),
            outside_initial_CAL_scope=sum(n for label, n in negatives[cid].items() if label not in scope),
            breaks_by_true_class=dict(break_rows[cid]), by_true_class=rows_by_class))
    class28_offers = [o for row in history if row['phase'] == 'task_finalized'
                     for o in row['discovery']['offers'] if o['class_id'] == 28]
    summary = dict(results=str(a.results), evaluation=str(a.evaluation), callbacks=len(history),
        transactions=len(transaction_rows), committed=sum(r['applied'] for r in transaction_rows),
        rejected=len(rejects), rejection_reasons=dict(Counter(r['reason'] for r in rejects)),
        acceptance_failure_reasons=dict(acceptance_reasons), task_rows=task_rows,
        round_callbacks_with_offers=sum(bool(r['discovery']['offers']) for r in history if r['phase'] == 'round'),
        round_callbacks_with_transactions=sum(bool(r['transactions']) for r in history if r['phase'] == 'round'),
        all_rollbacks_verified=all(r['rollback_verified'] for r in rejects),
        all_observed_heads_exact=all(r['head_exact'] for r in lifecycle_rows),
        all_observed_certificates_current=all(r['certificate_current'] for r in lifecycle_rows),
        missing_owned_provenance=dict(Counter(r['status'] for r in missing_provenance)),
        wire_by_transaction_outcome=dict(wire_by_outcome),
        selection_vs_holdout=dict(committed_with_selection_below_95=sum(r['applied'] and r['selection_recall'] < .95 for r in selection_rows),
            rejected_with_selection_below_95=sum(not r['applied'] and r['selection_recall'] < .95 for r in selection_rows)),
        protocol_flags=protocol_flags, communication=communication,
        false_activation=false_activation, class28=dict(role_coverage=roles['global_coverage']['28'],
            terminal_offers=len(class28_offers)),
        evaluation_activity=evaluation['appliance_activity'],
        limitation='Saved counts lack per-sample route/margin/sketch values; no attribution to a specific gate or new threshold can be validated from these artifacts')
    write_json(a.out / 'summary.json', summary)
    write_json(a.out / 'task_funnel.json', task_rows)
    write_json(a.out / 'false_activation_by_true_class.json', false_activation)
    write_json(a.out / 'missing_owned_provenance.json', missing_provenance)
    write_json(a.out / 'selection_vs_holdout.json', selection_rows)
    pd.DataFrame(transaction_rows).to_csv(a.out / 'transactions.csv', index=False)
    pd.DataFrame(request_rows).to_csv(a.out / 'request_funnel.csv', index=False)
    pd.DataFrame(lifecycle_rows).to_csv(a.out / 'lifecycle.csv', index=False)
    display = {k: summary[k] for k in ('transactions', 'committed', 'rejected',
        'rejection_reasons', 'acceptance_failure_reasons', 'missing_owned_provenance', 'class28')}
    display['tasks'] = [{k: r[k] for k in ('task', 'requests', 'offers', 'attempted', 'committed', 'request_funnel')}
                        for r in task_rows]
    print(json.dumps(display, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
