"""Replay frozen predictions and enumerate only each router's allowed actions.

Ground truth is used by explicitly labelled diagnostics, never to fit a router.
Requires the previous diagnostic ZIP to lock sample IDs, routes and profiles.
"""
from pathlib import Path
import json
import zipfile
import numpy as np
import pandas as pd
import torch

from eval_checkpoint import _make_denice_client_model
from tools.eval_denice_tip_router import classify, write_json
from fed_learning.strategies.incremental.denice_tip_router import encoder_fingerprint
from fed_learning.training.denice_eval import _allowed_classes_for_episode


def run_matched_oracle(ckpt, shards, classes, ids, out, device, diagnostic_zip, batch_size=512):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    source = Path(diagnostic_zip)
    if source.is_dir():
        previous = pd.read_csv(source/'predictions.csv')
        protocol = json.loads((source/'protocol.json').read_text())
        manifest = json.loads((source/'profile_manifest.json').read_text())
        diagnostics = json.loads((source/'router_diagnostics.json').read_text())
    else:
        with zipfile.ZipFile(source) as archive:
            previous = pd.read_csv(archive.open('predictions.csv'))
            protocol = json.loads(archive.read('protocol.json'))
            manifest = json.loads(archive.read('profile_manifest.json'))
            diagnostics = json.loads(archive.read('router_diagnostics.json'))
    if protocol['client_ids'] != ids or not protocol['training_commit'].startswith('03b9b53'):
        raise ValueError('Prior diagnostic panel/checkpoint differs')
    if previous.global_test_row.duplicated().any():
        raise ValueError('Duplicate sample identities in the prior panel')
    seen = sorted({c for values in classes.values() for c in values})
    mapping = {c:t for t,values in classes.items() for c in values}
    methods = ('Router', 'Multiclass', 'Mahalanobis', 'TIP')
    rows, frames, lifecycle = [], [], []
    for position, cid in enumerate(ids, 1):
        model, detector = _make_denice_client_model(ckpt, cid, device)
        signature = encoder_fingerprint(model)
        if signature != manifest[str(cid)]['encoder_hash']:
            raise ValueError(f'Client {cid}: checkpoint fingerprint differs from prior run')
        x, y = shards[cid]['X_test'], shards[cid]['y_test'].numpy()
        prior = previous[previous.client_id == cid].sort_values('sample_in_shard').reset_index(drop=True)
        if (len(prior) != len(y) or not np.array_equal(prior.y_true, y)
                or not np.array_equal(prior.global_test_row, shards[cid]['sample_ids'])):
            raise ValueError(f'Client {cid}: sample identity/order mismatch')
        truth_task = np.asarray([mapping[int(label)] for label in y])
        if not np.array_equal(prior.true_task, truth_task):
            raise ValueError('Ground-truth task map changed')
        baseline, baseline_routes = classify(model,detector,x,seen,device,batch_size)
        if not (np.array_equal(baseline,prior.Router) and np.array_equal(baseline_routes,prior.Router_task)):
            raise ValueError(f'Client {cid}: legacy replay differs from saved predictions')
        memory_tasks = sorted(int(t) for t, memory in detector.activation_memory.items() if len(memory))
        if not memory_tasks:
            raise ValueError(f'Client {cid}: empty binary task bank')
        available_sets = dict(Router=memory_tasks, Multiclass=memory_tasks,
                              Mahalanobis=diagnostics[str(cid)]['fitted_tasks'],
                              TIP=diagnostics[str(cid)]['fitted_tasks'])
        all_routes = sorted({t for tasks in available_sets.values() for t in tasks})
        cache = {}
        # Prediction for every legal action; labels are not used in these forwards.
        for task in all_routes:
            cache[task], _ = classify(model,detector,x,seen,device,batch_size,
                                     np.full(len(y),task,dtype=np.int64),'oracle_hard')
        for task in sorted(classes):
            local = detector.episode_classes.get(task, [])
            evidence = next((entry['evidence'] for entry in manifest[str(cid)]['tasks'] if entry['task']==task),None)
            lifecycle.append(dict(client_id=cid,task=task,local_class_count=len(local),
                                  binary_memory_present=task in memory_tasks,
                                  continuous_profile_present=task in available_sets['TIP'],
                                  profile_evidence=evidence,
                                  status='profile_present' if evidence else 'needs_participation_provenance_audit'))
        for method in methods:
            tasks = sorted(available_sets[method])
            route = prior[method+'_task'].to_numpy(dtype=np.int64)
            if not np.isin(route,tasks).all():
                raise ValueError(f'Client {cid}/{method}: saved route outside allowed bank')
            predictions = np.stack([cache[t] for t in tasks],axis=1)
            columns = {t:j for j,t in enumerate(tasks)}
            selected = np.asarray([columns[t] for t in route])
            normal = predictions[np.arange(len(y)),selected]
            if not np.array_equal(normal,prior[method]):
                raise ValueError(f'Client {cid}/{method}: forced-route replay differs')
            available = np.isin(truth_task,tasks)
            # Same fallback as the original policy when true task is unavailable:
            # retain that policy's original selected route; do not invent a route.
            matched_routes = np.where(available,truth_task,route)
            matched = predictions[np.arange(len(y)),[columns[t] for t in matched_routes]]
            solvable = (predictions == y[:,None]).any(axis=1)
            correct = normal == y
            if np.any(correct & ~solvable) or np.any((matched==y)&~solvable):
                raise RuntimeError('Allowed-route upper bound violated')
            class_supported = np.asarray([int(label) in detector.episode_classes.get(int(task),[])
                                          for label,task in zip(y,truth_task)])
            effective_supported = np.asarray([int(label) in _allowed_classes_for_episode(
                detector,int(task),seen,model.num_classes) for label,task in zip(y,truth_task)])
            # Partition errors only. Correct predictions take precedence even if
            # fallback/another route made an unsupported-task sample correct.
            bucket = np.full(len(y),'correct',dtype=object)
            wrong = ~correct
            bucket[wrong & ~available] = 'task_profile_unavailable'
            bucket[wrong & available & ~class_supported] = 'class_unavailable'
            bucket[wrong & available & class_supported & (route!=truth_task)] = 'wrong_task'
            bucket[wrong & available & class_supported & (route==truth_task)] = 'classifier_wrong'
            shared = available & class_supported
            frame = pd.DataFrame(dict(client_id=cid,policy=method,global_test_row=prior.global_test_row,
                y_true=y,true_task=truth_task,predicted_task=route,prediction=normal,
                task_available=available,class_supported=class_supported,
                effective_class_supported=effective_supported,route_correct=route==truth_task,
                matched_task=matched_routes,matched_prediction=matched,
                best_allowed_correct=solvable,bucket=bucket))
            frames.append(frame)
            row = dict(client_id=cid,policy=method,samples=len(y),correct=int(correct.sum()),
                matched_correct=int((matched==y).sum()),best_allowed_correct=int(solvable.sum()),
                shared_supported_samples=int(shared.sum()),
                shared_normal_correct=int((correct&shared).sum()),
                shared_matched_correct=int(((matched==y)&shared).sum()))
            for name in ('correct','task_profile_unavailable','class_unavailable','wrong_task','classifier_wrong'):
                row['bucket_'+name] = int((bucket==name).sum())
            rows.append(row)
        if encoder_fingerprint(model) != signature:
            raise RuntimeError('Diagnostic changed model state')
        pd.DataFrame(rows).to_csv(out/'matched_per_client.csv',index=False)
        print(f'Matched Oracle {position}/{len(ids)} client={cid}',flush=True)
        del model,detector
        if device=='cuda': torch.cuda.empty_cache()
    all_predictions = pd.concat(frames,ignore_index=True)
    all_predictions.to_csv(out/'matched_predictions.csv',index=False)
    pd.DataFrame(lifecycle).to_csv(out/'profile_lifecycle_audit.csv',index=False)
    table = pd.DataFrame(rows)
    results = []
    for method,group in table.groupby('policy',sort=False):
        n = int(group.samples.sum()); shared = int(group.shared_supported_samples.sum())
        item = dict(policy=method,samples=n)
        for key in ('correct','matched_correct','best_allowed_correct'):
            item[key+'_pooled_accuracy'] = float(group[key].sum()/n)
            item[key+'_client_mean_accuracy'] = float((group[key]/group.samples).mean())
        item['shared_supported_samples'] = shared
        for key in ('shared_normal_correct','shared_matched_correct'):
            item[key+'_accuracy'] = float(group[key].sum()/shared) if shared else None
        for column in group.columns:
            if column.startswith('bucket_'): item[column] = int(group[column].sum())
        item['router_only_50_excluded_on_panel_pooled'] = bool(item['best_allowed_correct_pooled_accuracy']<0.5)
        item['router_only_50_excluded_on_panel_client_mean'] = bool(item['best_allowed_correct_client_mean_accuracy']<0.5)
        results.append(item)
    summary = pd.DataFrame(results)
    summary.to_csv(out/'matched_summary.csv',index=False)
    write_json(out/'matched_definition.json',dict(
        matched='Use true task only if in this router bank; otherwise retain original route',
        mask='Unchanged local class policy, including original all-seen fallback for empty masks',
        best_allowed='Any correct classification among all permitted task routes; uses test label, diagnostic only',
        scope='Frozen checkpoint, fixed allowed task actions, hard masks/adapters and same sampled panel only',
        buckets='Errors partitioned by unavailable task, unavailable class, wrong task, classifier; successes separate',
        profile_missing='Does not establish a software bug without participation provenance'))
    return summary
