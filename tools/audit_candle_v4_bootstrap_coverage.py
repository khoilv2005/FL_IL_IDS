"""Read-only: compare inherited native router support with V4 continuous memory.

No raw data, model forward, classifier fit, training, or prediction changes.
Uses saved terminal confusion matrices only to quantify the support mismatch.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'candle_v4_runtime'))
    with warnings.catch_warnings(record=True) as caught:
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    report = json.loads(Path(args.report).read_text(encoding='utf-8'))
    assert state['meta']['completed_task'] == 5 and report['completed']
    continuous = state['candle_v4_shadow_state']['continuous_routers']
    events = {}
    local_tasks = {}
    for record in state['debug_history']:
        if 'bootstrap_events' in record:
            for event in record['bootstrap_events']:
                events.setdefault(int(event['client_id']), []).append(dict(task=record['task'], **event))
        if 'client_data' in record:
            for cid, data in record['client_data'].items():
                if data.get('num_samples', 0):
                    local_tasks.setdefault(int(cid), []).append(int(record['task']))
    rows = {}
    for cid, cs in sorted(continuous.items()):
        alg = state['client_algorithm_states'][cid]
        native = alg.get('denice', alg)['context_detector']
        native_tasks = sorted(int(t) for t, v in native['activation_memory'].items() if len(v))
        continuous_tasks = sorted({int(c)//6 for c in cs['memory']})
        missing = sorted(set(native_tasks)-set(continuous_tasks))
        pc = report['per_client'][str(cid)]
        n = pc['V1']['samples']
        counts = np.asarray(pc['V1']['per_class_counts'])
        diag_native = np.diag(pc['V1']['confusion_matrix'])
        missing_classes = np.asarray([c//6 in missing for c in range(34)])
        outside_support = np.asarray([c//6 not in continuous_tasks for c in range(34)])
        n_missing = int(counts[missing_classes].sum())
        correct_in_missing = {arm: int(np.diag(m['confusion_matrix'])[missing_classes].sum())
                              for arm, m in pc.items()}
        # Native labels must remain reachable only through predicted tasks.
        assert correct_in_missing['continuous_native'] == correct_in_missing['coverage'] == 0
        predicted_tasks = {}
        for arm in ('continuous_native', 'coverage'):
            predictions = np.asarray(pc[arm]['confusion_matrix']).sum(0)
            predicted_tasks[arm] = [t for t in range(6) if predictions[t*6:min(34,t*6+6)].sum()]
            assert set(predicted_tasks[arm]).issubset(continuous_tasks)
        rows[str(cid)] = dict(native_tasks=native_tasks, continuous_tasks=continuous_tasks,
            continuous_classes=sorted(map(int, cs['memory'])), missing_inherited_tasks=missing,
            local_training_tasks=sorted(set(local_tasks.get(cid, []))), bootstrap_events=events.get(cid, []),
            test_rows=n, missing_inherited_test_rows=n_missing,
            missing_inherited_native_correct=int(diag_native[missing_classes].sum()),
            correct_in_missing=correct_in_missing,
            continuous_support_accuracy_ceiling=float(1-counts[outside_support].sum()/n),
            predicted_task_support=predicted_tasks,
            accuracy={arm:m['accuracy'] for arm,m in pc.items()})
    affected = [cid for cid,row in rows.items() if row['missing_inherited_tasks']]
    clone_events = [dict(client_id=int(cid), **{k:v for k,v in e.items() if k!='client_id'})
                    for cid,row in rows.items() for e in row['bootstrap_events']
                    if e['bootstrap_policy']=='representative_clone']
    assert set(map(int, affected)) == {e['client_id'] for e in clone_events}
    groups = {}
    for name, ids in [('missing_inherited_memory', set(map(int, affected))),
                      ('no_inherited_memory_gap', set(map(int, rows))-set(map(int, affected)))]:
        n = sum(report['per_client'][str(c)]['V1']['samples'] for c in ids)
        groups[name] = dict(clients=len(ids), rows=n, accuracy={arm:
            sum(sum(report['per_client'][str(c)][arm]['confusion_matrix'][k][k]
                    for k in range(34)) for c in ids)/n
            for arm in ('V1','continuous_native','coverage')})
    summary = dict(completed=True, source_checkpoint=str(Path(args.checkpoint).resolve()),
        protocol=state['candle_v4_shadow_state']['spec'].get('version'),
        task=5, no_training=True, no_new_inference=True, no_raw_data_access=True,
        no_runtime_changes=True, no_TEST_fitting=True,
        estimator_version_warnings=[str(w.message) for w in caught][:1],
        warning_scope='No estimator prediction/fit used; analysis reads saved memory support and confusion counts',
        clients=len(rows), affected_count=len(affected), affected_ids=list(map(int, affected)),
        representative_clone_count=len(clone_events),
        affected_all_had_representative_clone=True,
        group_metrics=groups,
        missing_inherited_test_rows=sum(rows[c]['missing_inherited_test_rows'] for c in affected),
        native_correct_unreachable_in_continuous=sum(rows[c]['missing_inherited_native_correct'] for c in affected),
        clients_detail=rows)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('clients_detail','estimator_version_warnings')},indent=2))
    for cid in (35,43,59,5):
        r=rows[str(cid)]
        print(cid,json.dumps({k:r[k] for k in ('native_tasks','continuous_tasks','bootstrap_events',
             'missing_inherited_test_rows','missing_inherited_native_correct','continuous_support_accuracy_ceiling','accuracy')}))


if __name__ == '__main__':
    main()
