"""Each predictor sees the identical full cumulative set; labels only for metrics."""
import json
import os
from pathlib import Path

import numpy as np
import torch

from .router import ClusterRouter
from .contracts import state_hash


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def metrics(confusion, classes):
    cm = np.asarray(confusion, np.int64)
    support = cm.sum(1)
    hits = np.diag(cm[:, :34])
    predicted = cm[:, :34].sum(0)
    f1 = np.divide(2*hits, support+predicted, out=np.zeros(34), where=(support+predicted)>0)
    total = int(support.sum())
    return dict(rows=total, correct=int(hits.sum()), accuracy=float(hits.sum()/total) if total else None,
        macro_f1=float(f1[list(classes)].mean()),
        per_class={int(c): dict(support=int(support[c]), recall=float(hits[c]/support[c]) if support[c] else 0.,
            precision=float(hits[c]/predicted[c]) if predicted[c] else 0., f1=float(f1[c])) for c in classes},
        unregistered_prediction_rows=int(cm[:, 34].sum()))


def evaluate_cluster(state, x, y, test_binding, output, *, batch_size=8192, device='cpu',evaluation_role='test'):
    scope = tuple(range(min(6*(state.task+1), 34)))
    binding = state_hash(dict(state=state.fingerprint(), test=test_binding, scope=scope, total_rows=len(y),role=evaluation_role))
    output = Path(output)
    cm, route = np.zeros((34, 35), np.int64), np.zeros((6, 7), np.int64)
    cursor = 0
    if output.exists():
        saved = json.loads(output.read_text(encoding='utf-8'))
        if saved['binding'] != binding:
            raise ValueError('Evaluation resume state/test/scope mismatch')
        if saved.get('receipt_sha256')!=state_hash({k:v for k,v in saved.items() if k!='receipt_sha256'}):
            raise ValueError('Evaluation resume receipt checksum mismatch')
        cm, route, cursor = np.asarray(saved['confusion']), np.asarray(saved['router_confusion']), saved['cursor']
        if (not 0<=cursor<=len(y) or cm.shape!=(34,35) or route.shape!=(6,7) or
            np.any(cm<0) or np.any(route<0) or cm.sum()!=route.sum() or
            cm.sum()!=np.isin(y[:cursor],scope).sum()):
            raise ValueError('Invalid evaluation resume accounting')
        if saved.get('completed'):
            return saved
    predictor = ClusterRouter(state, device=device) if state.router_state['activation_memory'] else None
    task_cm = {t: np.zeros((34,35), np.int64) for t in range(state.task+1)}
    if cursor:
        task_cm = {int(t): np.asarray(v, np.int64) for t,v in saved['task_confusions'].items()}
    for start in range(cursor, len(y), batch_size):
        end = min(start+batch_size, len(y))
        labels = np.asarray(y[start:end], np.int64)
        keep = np.isin(labels, scope)
        truth = labels[keep]
        # Ground truth controls cumulative benchmark membership only. It is
        # never passed to predictor, routing, availability or model selection.
        if len(truth):
            inputs = torch.from_numpy(np.asarray(x[start:end][keep], np.float32).copy())
            if not torch.isfinite(inputs).all():raise ValueError('Nonfinite TEST input')
            if predictor:
                pred, tasks, scores, _ = predictor.predict(inputs)
                if not np.isfinite(scores).all():
                    raise ValueError('Nonfinite router output')
            else:
                pred, tasks = np.full(len(truth), -1), np.full(len(truth), -1)
            if not np.isin(pred, [-1, *scope]).all():
                raise ValueError('Prediction outside cumulative class scope')
            columns, task_columns = np.where(pred<0,34,pred), np.where(tasks<0,6,tasks)
            true_tasks = np.minimum(truth//6, 5)
            np.add.at(cm, (truth, columns), 1)
            np.add.at(route, (true_tasks, task_columns), 1)
            for t in range(state.task+1):
                mask = true_tasks == t
                np.add.at(task_cm[t], (truth[mask], columns[mask]), 1)
        saved = dict(binding=binding, task=state.task, cluster_uid=state.cluster_uid, members=state.members,
            parents=state.parent_lineage, cursor=end, source_rows=len(y), completed=end==len(y),
            confusion=cm.tolist(), router_confusion=route.tolist(),
            task_confusions={t:v.tolist() for t,v in task_cm.items()},
            metrics=metrics(cm,scope), task_metrics={t:metrics(v,range(t*6,min(t*6+6,34))) for t,v in task_cm.items()},
            inference_uses_true_task=False,evaluation_role=evaluation_role,
            evaluated_full_cumulative_test=evaluation_role=='test')
        saved = json.loads(json.dumps(saved))
        saved['receipt_sha256']=state_hash(saved)
        atomic_json(output, saved)
        if start==cursor or end==len(y) or (start//batch_size)%50==0:
            print(f'Full cumulative {evaluation_role.upper()} T{state.task} cluster={state.cluster_uid}: {end:,}/{len(y):,}; evaluated={cm.sum():,}',flush=True)
    expected = int(np.isin(y, scope).sum())
    if int(cm.sum()) != expected or sum(v.sum() for v in task_cm.values()) != expected:
        raise ValueError('Full TEST row accounting mismatch')
    return saved


def aggregate_evaluations(results):
    if not results or not all(r['completed'] for r in results):
        raise ValueError('Incomplete cluster evaluation')
    rows = {r['metrics']['rows'] for r in results}
    if len(rows)!=1 or not next(iter(rows)):
        raise ValueError('Clusters must see the same nonempty full cumulative test')
    return dict(K=len(results), unique_test_rows=next(iter(rows)), prediction_evaluations=sum(r['metrics']['rows'] for r in results),
        mean_cluster_accuracy=float(np.mean([r['metrics']['accuracy'] for r in results])),
        mean_cluster_macro_f1=float(np.mean([r['metrics']['macro_f1'] for r in results])),
        accuracy_std=float(np.std([r['metrics']['accuracy'] for r in results])),
        macro_f1_std=float(np.std([r['metrics']['macro_f1'] for r in results])),
        cluster_metrics={r['cluster_uid']:r['metrics'] for r in results}, primary_weighting='uniform_clusters')


def forgetting(results, history):
    """Best past task accuracy minus current; split inherits, ambiguous merge N/A."""
    details = []
    for row in results:
        for task in range(row['task']):
            candidates = [h for h in history if h['task']<row['task'] and
                (h['cluster_uid']==row['cluster_uid'] or h['cluster_uid'] in row['parents']) and
                str(task) in h['task_metrics'] and h['task_metrics'][str(task)]['accuracy'] is not None]
            by_time = {}
            for h in candidates:
                by_time.setdefault(h['task'], []).append(h)
            if not candidates or row['task_metrics'][str(task)]['accuracy'] is None or any(len(v)>1 for v in by_time.values()):
                details.append(dict(cluster=row['cluster_uid'],task=task,forgetting=None,reason='NO_UNAMBIGUOUS_LINEAGE'))
                continue
            best = max(h['task_metrics'][str(task)]['accuracy'] for h in candidates)
            current = row['task_metrics'][str(task)]['accuracy']
            details.append(dict(cluster=row['cluster_uid'],task=task,forgetting=best-current))
    known = [d['forgetting'] for d in details if d['forgetting'] is not None]
    return dict(mean_forgetting=float(np.mean(known)) if known else None, lineage_task_pairs=details,
        undefined_pairs=sum(d['forgetting'] is None for d in details), policy='best_past_accuracy_single_lineage')
