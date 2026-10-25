"""Read-only paired routing/oracle diagnostic; never an official benchmark."""
import json
from pathlib import Path

import numpy as np
import torch

from tools.eval_cluster_servers import completed_payload
from fed_learning.strategies.decentralized.denice_v2.state import restore
from fed_learning.strategies.decentralized.denice_v2.router import ClusterRouter
from fed_learning.strategies.decentralized.denice_v2.evaluation import metrics, atomic_json
from fed_learning.strategies.decentralized.denice_v2.contracts import state_hash


def masked_predictions(logits, tasks, registered):
    """Missing task/classes count as failure, never silently drop those rows."""
    result = np.full(len(tasks), -1, dtype=np.int64)
    for task in np.unique(tasks):
        labels = registered.get(int(task), ())
        rows = np.flatnonzero(tasks == task)
        if len(labels):
            labels = np.asarray(labels, dtype=np.int64)
            result[rows] = labels[logits[np.ix_(rows, labels)].argmax(axis=1)]
    return result


def diagnose(checkpoint_root, x, y, output_dir, *, batch_size=1024, device='cpu',
             per_class=256, seed=42, server_ids=None):
    if batch_size < 1 or per_class < 1:
        raise ValueError('Positive batch size and per-class sample count required')
    payload = completed_payload(checkpoint_root)
    rng = np.random.default_rng(seed)
    indices = np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c),
        min(per_class, int(np.sum(y == c))), replace=False) for c in range(34)]))
    if not len(indices):
        raise ValueError('Empty diagnostic sample')
    truth = np.asarray(y[indices], np.int64)
    inputs = np.asarray(x[indices], np.float32)
    if not np.isfinite(inputs).all():
        raise ValueError('Nonfinite diagnostic inputs')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    binding = state_hash(dict(indices=indices, x=inputs, y=truth))
    rows = []
    for uid in sorted(payload['active']):
        state = restore(payload['states'][uid])
        if state.router_state['router_mode'] == 'routerless_v1':
            raise ValueError('This checkpoint predicts without a task router. Set DIAGNOSTIC=False for full evaluation.')
        if server_ids is not None and state.coordinator_id not in server_ids:
            continue
        predictor = ClusterRouter(state, device=device)
        registered = state.router_state['episode_classes']
        cms = {k: np.zeros((34, 35), np.int64) for k in ('routed', 'oracle')}
        route = np.zeros((6, 7), np.int64)
        zero_bits = ties = 0
        for start in range(0, len(truth), batch_size):
            target = truth[start:start+batch_size]
            true_tasks = np.minimum(target // 6, 5)
            with torch.inference_mode():
                logits, acts = predictor.model.get_output_and_context_activations(
                    torch.from_numpy(inputs[start:start+batch_size]).to(device))
            raw = logits.cpu().numpy()
            bits = predictor.detector.binarize_layer_activations(
                {k: v.cpu().numpy() for k, v in acts.items()})
            tasks, scores = predictor.detector.predict_episodes_with_scores(bits)
            if not np.isfinite(raw).all() or not np.isfinite(scores).all():
                raise ValueError('Nonfinite predictions')
            zero_bits += int(np.sum(~np.any(bits, axis=1)))
            ties += int(np.sum(np.sum(scores == scores.max(axis=1, keepdims=True), axis=1) > 1))
            predictions = dict(routed=masked_predictions(raw, tasks, registered),
                               oracle=masked_predictions(raw, true_tasks, registered))
            for key, pred in predictions.items():
                np.add.at(cms[key], (target, np.where(pred < 0, 34, pred)), 1)
            np.add.at(route, (true_tasks, np.where(tasks < 0, 6, tasks)), 1)
        memory = {}
        for task, bank in state.router_state['activation_memory'].items():
            bank = np.asarray(bank)
            memory[int(task)] = dict(rows=len(bank), zero_rows=int(np.sum(~np.any(bank, axis=1))),
                unique_rows=int(len(np.unique(bank, axis=0))), classes=list(registered.get(task, ())))
        result = dict(cluster_uid=uid, temporary_server=state.coordinator_id, members=list(state.members),
            sample_binding=binding, checkpoint_state=state.fingerprint(), diagnostic_only=True,
            oracle_uses_true_task=True, samples=len(truth), memory=memory,
            zero_query_fraction=zero_bits/len(truth), score_tie_fraction=ties/len(truth),
            router_accuracy=float(np.trace(route[:, :6])/len(truth)), router_confusion=route.tolist(),
            routed=metrics(cms['routed'], range(34)), oracle=metrics(cms['oracle'], range(34)))
        result['accuracy_gain'] = result['oracle']['accuracy'] - result['routed']['accuracy']
        atomic_json(output/f'diagnostic_{uid}.json', result)
        rows.append(result)
        print(f"Diagnostic server {state.coordinator_id}: routed={result['routed']['accuracy']:.4f}, "
              f"oracle={result['oracle']['accuracy']:.4f}, router={result['router_accuracy']:.4f}", flush=True)
        del predictor
    if not rows:
        raise ValueError('No selected servers found')
    summary = dict(diagnostic_only=True, sampling='fixed stratified TEST subset; not full-test benchmark',
        runtime=dict(torch=torch.__version__, device=device),
        seed=seed, per_class=per_class, samples=len(truth), sample_binding=binding,
        servers=len(rows), protocol_lock=payload['lock'],
        mean_routed_accuracy=float(np.mean([r['routed']['accuracy'] for r in rows])),
        mean_oracle_accuracy=float(np.mean([r['oracle']['accuracy'] for r in rows])),
        mean_router_accuracy=float(np.mean([r['router_accuracy'] for r in rows])), results=rows)
    atomic_json(output/'diagnostic_summary.json', summary)
    return summary
