"""Evaluate completed cluster-server training, one predictor per active server on FULL TEST."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

from fed_learning.strategies.decentralized.denice_v2.archive import TaskArchive
from fed_learning.strategies.decentralized.denice_v2.data import TaskData
from fed_learning.strategies.decentralized.denice_v2.evaluation import atomic_json, evaluate_cluster, aggregate_evaluations
from fed_learning.strategies.decentralized.denice_v2.state import restore


def completed_payload(checkpoint_root):
    root = Path(checkpoint_root)
    if not (root/'completion.json').is_file():
        raise ValueError('Finish TRAIN Task 0-5: completion.json is missing')
    receipt = json.loads((root/'completion.json').read_text(encoding='utf-8'))
    if not receipt.get('completed') or receipt.get('tasks_completed') != 6:
        raise ValueError('Finish TRAIN Task 0-5 before running eval_selective')
    if receipt.get('checkpoint_format') != 'denice_v2_cluster_archive_v1':
        raise ValueError('Not a cluster-server checkpoint; use the legacy notebook branch')
    payload = TaskArchive(root, 5).load()
    if (payload['lock'] != receipt['protocol_lock'] or payload['next_task'] != 6
            or payload['task_end_pending'] or not payload['active']):
        raise ValueError('Final checkpoint is incomplete or differs from completion.json')
    return payload


def evaluate_servers(checkpoint_root, data_dir, output_dir, *, batch_size=8192, device='cpu'):
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    payload = completed_payload(checkpoint_root)  # Must pass before reading TEST.
    output = Path(output_dir)
    states = {uid: restore(payload['states'][uid]) for uid in payload['active']}
    if any(s.task != 5 for s in states.values()):
        raise ValueError('Expected final Task 5 server models')
    shapes = {tuple(s.algorithm_state['native_input_shape']) for s in states.values()}
    if len(shapes) != 1:
        raise ValueError('Server input shape mismatch')
    # Eval needs only global_test_data.npz, never scans or loads client TRAIN.
    source = SimpleNamespace(dataset=Path(data_dir), cache=(output/'test_cache').resolve(),
                             input_shape=next(iter(shapes)))
    x, y, binding = TaskData.test(source)
    if len(y) == 0:
        raise ValueError('Empty full test dataset')
    results, rows = [], []
    for uid, state in sorted(states.items()):
        result = evaluate_cluster(state, x, y, binding, output/f'evaluation_{uid}.json',
                                  batch_size=batch_size, device=device)
        if result['metrics']['rows'] != len(y):
            raise ValueError('A server did not evaluate every TEST row')
        results.append(result)
        rows.append(dict(cluster_uid=uid, temporary_server=state.coordinator_id,
            member_count=len(state.members), members=list(state.members),
            test_samples=len(y), accuracy=result['metrics']['accuracy'],
            macro_f1=result['metrics']['macro_f1']))
    summary = aggregate_evaluations(results)
    summary.update(task=5, checkpoint_round=payload['config']['rounds_per_task'],
        model_stage='task_boundary_consolidated', test_sha256=binding,
        protocol_lock=payload['lock'], per_temporary_server=rows, full_test=True)
    atomic_json(output/'temporary_server_metrics.json', summary)
    with (output/'temporary_server_metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return summary
