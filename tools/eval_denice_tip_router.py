"""P0/P1/P2 retrospective router evaluation; never starts training.

Reference implementation reviewed: seohyeon-cha/FedProTIP@54193fa2,
client.compute_references and server._eval_cnn. This evaluator uses sample-level
norms, local frozen encoders and no server voting; RMS references are an ablation.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from threadpoolctl import threadpool_limits

from eval_checkpoint import _make_denice_client_model, _load_checkpoint
from fed_learning.data.incremental_loader import IncrementalDataLoader
from fed_learning.training.decentralized_denice_il import _limit_eval_samples, _partition_test_data_by_client
from fed_learning.training.denice_eval import _denice_routed_logits_with_episodes
from fed_learning.strategies.incremental.denice_router_baselines import PrototypeRouter
from fed_learning.strategies.incremental.denice_tip_router import (
    SubspaceRouter, continuous_features, encoder_fingerprint)


def write_json(path, value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    Path(path).write_text(json.dumps(value, indent=2, default=convert, allow_nan=False), encoding='utf-8')


@torch.no_grad()
def classify(model, detector, inputs, seen, device, batch_size, routes=None, policy='pred_hard'):
    predictions, episodes = [], []
    for start in range(0, len(inputs), batch_size):
        # oracle_hard is also the existing forced-route adapter/mask executor.
        # For learned routers, routes contains predicted IDs, never test labels.
        logits, ep = _denice_routed_logits_with_episodes(
            model, inputs[start:start+batch_size].to(device), detector, seen, device,
            inference_policy=policy,
            oracle_episodes=None if routes is None else routes[start:start+batch_size])
        predictions.append(logits.argmax(1).cpu().numpy())
        if ep is not None:
            episodes.append(np.asarray(ep))
    return np.concatenate(predictions), np.concatenate(episodes) if episodes else None


def metric_row(frame, prediction, route, available):
    y, truth_task = frame.y_true.to_numpy(), frame.true_task.to_numpy()
    covered = frame.covered.to_numpy(dtype=bool)
    correct = prediction == y
    result = dict(samples=len(y), correct=int(correct.sum()), accuracy=float(correct.mean()),
                  f1_macro=float(f1_score(y, prediction, average='macro', zero_division=0)),
                  covered_count=int(covered.sum()), coverage=float(covered.mean()))
    if route is not None:
        routed = route == truth_task
        task_available = np.isin(truth_task, available)
        result.update(route_correct=int(routed.sum()), route_total=len(y),
                      route_accuracy_global=float(routed.mean()),
                      route_covered_correct=int((routed & covered).sum()),
                      route_covered_total=int(covered.sum()),
                      route_available_correct=int((routed & task_available).sum()),
                      route_available_total=int(task_available.sum()),
                      correct_task_missing_class=int((routed & ~covered).sum()),
                      correct_task_class_present_wrong_class=int((routed & covered & ~correct).sum()))
    return result


def balanced_cap(indices, labels, limit, rng):
    pools = [list(rng.permutation(indices[labels[indices] == label]))
             for label in np.unique(labels[indices])]
    chosen = []
    while pools and len(chosen) < limit:
        for pool in pools:
            if pool and len(chosen) < limit:
                chosen.append(pool.pop())
        pools = [pool for pool in pools if pool]
    return np.asarray(chosen, dtype=np.int64)


def local_profiles(model, detector, loader, cid, final_task, seed, max_samples, batch_size):
    """Only recorded completed tasks, plus the latest locally refreshed task.

    The latter is needed for final-round checkpoints saved before task-end basis
    capture. Local class support further restricts historical data access.
    """
    records = getattr(model, 'cgofed_projection_state', {}) or {}
    evidence = {int(entry['task_id']): 'cgofed_completed_task'
                for entry in records.get('tasks', []) if 'task_id' in entry}
    refreshed = getattr(detector, 'router_last_refresh_task', None)
    if refreshed is not None:
        evidence.setdefault(int(refreshed), 'checkpoint_local_router_refresh')
    evidence = {t: source for t, source in evidence.items()
                if t <= final_task and detector.episode_classes.get(t)}
    if not evidence:
        raise ValueError(f'Client {cid}: no recorded local participation; refusing to infer all-task access')
    inputs, labels_tensor = loader.get_client_full_data(cid)
    labels = labels_tensor.numpy()
    if not len(labels):
        raise ValueError(f'Client {cid}: original local train data missing')
    train, validation, manifests = {}, {}, []
    for task, source in sorted(evidence.items()):
        allowed = sorted(set(loader.get_task_classes(task)) & set(detector.episode_classes[task]))
        candidates = np.flatnonzero(np.isin(labels, allowed))
        rng = np.random.default_rng(seed + cid * 1009 + task * 9176)
        fitting, held = [], []
        for label in np.unique(labels[candidates]):
            indices = rng.permutation(candidates[labels[candidates] == label])
            nval = min(len(indices)-1, max(1, int(len(indices)*0.2))) if len(indices)>1 else 0
            held.extend(indices[:nval]); fitting.extend(indices[nval:])
        fit = balanced_cap(np.asarray(fitting, dtype=np.int64), labels, max_samples, rng)
        val = balanced_cap(np.asarray(held, dtype=np.int64), labels, max(32, max_samples//4), rng)
        if not len(fit):
            continue
        train[task] = continuous_features(model, inputs[fit], batch_size)
        if len(val):
            validation[task] = continuous_features(model, inputs[val], batch_size)
        manifests.append(dict(task=task, evidence=source, classes=allowed,
                              fit_row_ids=fit, validation_row_ids=val,
                              available_rows=len(candidates)))
    if not train:
        raise ValueError(f'Client {cid}: no usable profiles')
    return train, validation, manifests


def validation_score(router, validation):
    # Task-balanced selection prevents the largest task dominating selection.
    return float(np.mean([np.mean(router.predict(z) == task)
                          for task, z in validation.items()])) if validation else None


def fit_candidates(train, validation):
    models = {'Centroid': PrototypeRouter().fit(train)}
    audit = []
    for family, candidates in (
        ('Mahalanobis', [PrototypeRouter('mahalanobis', x) for x in (0.1, 0.01, 0.5)]),
        ('TIP', [SubspaceRouter(max_rank=rank, basis_mode=mode, reference_mode=ref)
                 for mode in ('independent', 'residual')
                 for ref in ('rms', 'mean') for rank in (32, 16, 64)]),
    ):
        best, best_score = None, -float('inf')
        for candidate in candidates:
            candidate.fit(train)
            score = validation_score(candidate, validation)
            description = {key: getattr(candidate, key) for key in
                           ('method', 'shrinkage', 'max_rank', 'basis_mode', 'reference_mode')
                           if hasattr(candidate, key)}
            audit.append(dict(family=family, config=description, validation_task_macro_accuracy=score))
            effective = score if score is not None else 0.0
            if best is None or effective > best_score:
                best, best_score = candidate, effective
        models[family] = best
    return models, audit


@torch.no_grad()
def binary_multiclass_predictions(model, detector, inputs, device, batch_size):
    memory = detector.activation_memory
    matrices, targets = [], []
    for task, values in sorted(memory.items()):
        z = np.asarray(values)
        if z.ndim == 2 and len(z):
            matrices.append(z); targets.extend([int(task)] * len(z))
    if not targets:
        raise ValueError('Missing binary memory for multiclass baseline')
    if len(set(targets)) == 1:
        return np.full(len(inputs), targets[0]), dict(single_task=targets[0])
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=0)
    clf.fit(np.concatenate(matrices), targets)
    result = []
    for start in range(0, len(inputs), batch_size):
        acts = model.get_context_activations_per_sample(inputs[start:start+batch_size].to(device))
        binary = detector.binarize_layer_activations({k: v.cpu().numpy() for k,v in acts.items()})
        result.append(clf.predict(binary))
    return np.concatenate(result), dict(classes=clf.classes_, coef=clf.coef_, intercept=clf.intercept_)


def summarize(frames, rows, out, seen, seed):
    predictions = pd.concat(frames, ignore_index=True)
    clients = pd.DataFrame(rows)
    predictions.to_csv(out/'predictions.csv', index=False)
    clients.to_csv(out/'per_client_metrics.csv', index=False)
    policies = list(clients.policy.unique())
    summaries, task_rows, confusions, class_confusions = [], [], [], []
    rng = np.random.default_rng(seed)
    baseline = clients[clients.policy == 'Router'].set_index('client_id')
    for policy in policies:
        pred = predictions[policy].to_numpy()
        selected = clients[clients.policy == policy].set_index('client_id').loc[baseline.index]
        differences = (selected.accuracy - baseline.accuracy).to_numpy()
        draws = rng.integers(0, len(differences), (2000, len(differences)))
        interval = np.quantile(differences[draws].mean(1), [0.025, 0.975]) * 100
        item = dict(policy=policy, accuracy=float(accuracy_score(predictions.y_true, pred)),
                              f1_macro=float(f1_score(predictions.y_true, pred, labels=seen, average='macro', zero_division=0)),
                              client_mean_accuracy=float(selected.accuracy.mean()),
                              client_mean_f1_macro=float(selected.f1_macro.mean()),
                              delta_client_mean_pp=float(differences.mean()*100),
                              paired_client_ci_low_pp=float(interval[0]), paired_client_ci_high_pp=float(interval[1]))
        if selected.route_total.notna().any():
            for scope, numerator, denominator in (
                ('global', 'route_correct', 'route_total'),
                ('covered', 'route_covered_correct', 'route_covered_total'),
                ('available', 'route_available_correct', 'route_available_total'),
            ):
                total = int(selected[denominator].sum())
                correct = int(selected[numerator].sum())
                item['route_'+scope+'_correct'] = correct
                item['route_'+scope+'_total'] = total
                item['route_accuracy_'+scope] = correct/total if total else None
        summaries.append(item)
        class_counts = predictions.groupby(['y_true', policy]).size()
        class_confusions.extend(dict(policy=policy, true_class=int(a), predicted_class=int(b), count=int(n))
                                for (a,b), n in class_counts.items())
        for task, group in predictions.groupby('true_task'):
            task_rows.append(dict(policy=policy, task=int(task), samples=len(group),
                                  accuracy=float(np.mean(group[policy] == group.y_true)),
                                  f1_macro=float(f1_score(group.y_true,group[policy],average='macro',zero_division=0))))
        route_column = policy + '_task'
        if route_column in predictions:
            counts = predictions.groupby(['true_task', route_column]).size()
            confusions.extend(dict(policy=policy, true_task=int(a), predicted_task=int(b), count=int(n))
                              for (a,b), n in counts.items())
    summary = pd.DataFrame(summaries).set_index('policy')
    summary.to_csv(out/'summary.csv')
    pd.DataFrame(task_rows).to_csv(out/'per_task_metrics.csv', index=False)
    pd.DataFrame(confusions).to_csv(out/'route_confusion.csv', index=False)
    pd.DataFrame(class_confusions).to_csv(out/'class_confusion.csv', index=False)
    # Primary gate uses mean-client accuracy with the matching paired client CI.
    tip = summary.loc['TIP']
    gate = dict(accuracy_gain_pp=float(tip.delta_client_mean_pp),
                macro_f1_change_pp=float(100*(tip.f1_macro-summary.loc['Router','f1_macro'])),
                paired_ci_low_pp=float(tip.paired_client_ci_low_pp),
                passed=bool(tip.delta_client_mean_pp >= 2
                            and tip.f1_macro >= summary.loc['Router','f1_macro']-0.005
                            and tip.paired_client_ci_low_pp > 0))
    # Additional paired comparisons separate continuous features from TIP's rule.
    comparisons = []
    tip_clients = clients[clients.policy == 'TIP'].set_index('client_id').loc[baseline.index]
    for comparator in ('Centroid', 'Mahalanobis'):
        other = clients[clients.policy == comparator].set_index('client_id').loc[baseline.index]
        diff = (tip_clients.accuracy-other.accuracy).to_numpy()
        ci = np.quantile(diff[draws].mean(1), [0.025, 0.975])*100
        comparisons.append(dict(comparator=comparator, delta_pp=float(diff.mean()*100), ci_pp=ci))
    write_json(out/'gate.json', dict(primary=gate, continuous_comparisons=comparisons,
                                    ci_unit='client', bootstrap_draws=2000))
    return summary


def run_suite(ckpt, config, report, loader, shards, classes, ids, out, device,
              batch_size=512, max_profile_samples=512, seed=42):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if batch_size < 1 or max_profile_samples < 1:
        raise ValueError('Batch and profile sample limits must be positive')
    if not str(config.get('git_commit', '')).startswith('03b9b53'):
        raise ValueError('Frozen baseline protocol requires checkpoint 03b9b53')
    if [int(cid) for cid in report['final_metrics']['per_client']] != list(ids):
        raise ValueError('Client order differs from the original evaluation panel')
    seen = sorted({c for values in classes.values() for c in values})
    label_task = {c:t for t, values in classes.items() for c in values}
    if len(seen) != sum(map(len, classes.values())):
        raise ValueError('Ambiguous dataset class-to-task map')
    final_task = max(classes)
    if int(report['completed_task']) != final_task:
        raise ValueError('Dataset task span differs from the final report')
    frames, rows = [], []
    print('P0: restoring legacy baseline on the complete original panel', flush=True)
    for cid in ids:
        model, detector = _make_denice_client_model(ckpt, cid, device)
        stored = ckpt['client_model_states'].get(cid, ckpt['client_model_states'].get(str(cid)))
        if stored is None or set(stored) != set(model.state_dict()):
            raise ValueError(f'Client {cid}: restored model/checkpoint keys differ')
        x, y = shards[cid]['X_test'], shards[cid]['y_test'].numpy()
        if not len(y):
            raise ValueError(f'Empty evaluation shard: {cid}')
        known = {int(c) for values in detector.episode_classes.values() for c in values}
        frame = pd.DataFrame(dict(client_id=cid, sample_in_shard=np.arange(len(y)), y_true=y,
                                  true_task=[label_task[int(v)] for v in y], covered=np.isin(y, list(known))))
        if 'sample_ids' in shards[cid]:
            frame['global_test_row'] = shards[cid]['sample_ids']
        pred, route = classify(model, detector, x, seen, device, batch_size)
        if route is None:
            raise ValueError(f'Client {cid}: baseline detector produced no task IDs')
        frame['Router'], frame['Router_task'] = pred, route
        rows.append(dict(client_id=cid, policy='Router', **metric_row(frame, pred, route, list(detector.episode_classes))))
        frames.append(frame)
        print(f'P0 client {cid}: accuracy={np.mean(pred==y):.4f}', flush=True)
        del model, detector
    baseline = float(np.mean([r['accuracy'] for r in rows]))
    write_json(out/'baseline_guard.json', dict(observed=baseline, expected=0.2279, tolerance=0.001,
                                              passed=abs(baseline-0.2279)<=0.001))
    pd.DataFrame(rows).to_csv(out/'per_client_metrics.csv', index=False)
    if abs(baseline - 0.2279) > 0.001:
        raise RuntimeError(f'P0 baseline guard failed: {baseline:.6f}, expected 0.2279 +/- 0.001')
    manifests, diagnostics = {}, {}
    profiles = out/'profiles'; profiles.mkdir(exist_ok=True)
    print('P0 passed. P1/P2: train-only profile fitting and frozen evaluation', flush=True)
    for index, cid in enumerate(ids):
        start_time = time.perf_counter()
        model, detector = _make_denice_client_model(ckpt, cid, device)
        fingerprint = encoder_fingerprint(model)
        train, validation, manifest = local_profiles(
            model, detector, loader, cid, final_task, seed, max_profile_samples, batch_size)
        routers, selection = fit_candidates(train, validation)
        fit_seconds = time.perf_counter()-start_time
        # Persist selection before reading this client's test features.
        manifests[str(cid)] = dict(encoder_hash=fingerprint, tasks=manifest,
                                  validation_kind='router holdout; backbone may have seen these rows',
                                  selection=selection)
        write_json(out/'profile_manifest.json', manifests)
        frame = frames[index]
        x = shards[cid]['X_test']
        z = continuous_features(model, x, batch_size)
        # Runtime protocol invariant, checked on a bounded probe (not model selection).
        nprobe = min(8, len(x))
        z_single = continuous_features(model, x[:nprobe], 1)
        policy_routes = {}
        score_data = {}
        router_timing = {}
        for policy, router in routers.items():
            route_start = time.perf_counter()
            route = router.predict(z)
            router_timing[policy] = (time.perf_counter()-route_start)/max(len(z),1)
            probe_route = router.predict(z_single)
            if not np.array_equal(route[:nprobe], probe_route):
                raise RuntimeError(f'{cid}/{policy}: feature routing depends on batch composition')
            if not np.array_equal(router.predict(z[::-1])[::-1], route):
                raise RuntimeError(f'{cid}/{policy}: routing depends on sample order')
            policy_routes[policy] = route
            score_data[policy] = router.scores(z)
            score_data[policy+'_task_columns'] = router.tasks
        multiclass, binary_state = binary_multiclass_predictions(model, detector, x, device, batch_size)
        policy_routes['Multiclass'] = multiclass
        np.savez_compressed(profiles/f'client_{cid}_scores.npz', **score_data)
        for policy, route in policy_routes.items():
            pred, _ = classify(model, detector, x, seen, device, batch_size, route, 'oracle_hard')
            frame[policy], frame[policy+'_task'] = pred, route
            available = routers[policy].tasks if policy in routers else list(detector.activation_memory)
            rows.append(dict(client_id=cid, policy=policy, **metric_row(frame, pred, route, available)))
        true_tasks = frame.true_task.to_numpy()
        global_detector = copy.copy(detector)
        global_detector.episode_classes = classes
        for policy, oracle_detector, inference in (
            ('OracleLocal', detector, 'oracle_hard'),
            ('OracleGlobal', global_detector, 'oracle_hard'),
            ('AllClasses', detector, 'backbone_nomask'),
        ):
            pred, _ = classify(model, oracle_detector, x, seen, device, batch_size,
                               true_tasks if inference == 'oracle_hard' else None, inference)
            frame[policy] = pred
            rows.append(dict(client_id=cid, policy=policy, **metric_row(frame, pred, None, [])))
        if encoder_fingerprint(model) != fingerprint:
            raise RuntimeError(f'Client {cid}: evaluation mutated model state')
        bank = dict(schema_version=1, encoder_hash=fingerprint,
                    routers={key:value.state_dict() for key,value in routers.items()}, binary_multiclass=binary_state)
        torch.save(bank, profiles/f'client_{cid}_router_profiles.pt')
        diagnostics[str(cid)] = dict(seconds=time.perf_counter()-start_time,
                                    profile_fit_seconds=fit_seconds,
                                    router_seconds_per_sample=router_timing,
                                    feature_dim=int(z.shape[1]), feature_zero_rows=int((np.linalg.norm(z,axis=1)==0).sum()),
                                    tip_zero_relevance_rows=int((np.linalg.norm(routers['TIP'].relevance(z),axis=1)==0).sum()),
                                    fitted_tasks=sorted(train), batch_probe_samples=nprobe,
                                    tip=routers['TIP'].diagnostics,
                                    selected_tip={k:getattr(routers['TIP'], k) for k in ('max_rank','basis_mode','reference_mode')},
                                    mahalanobis_shrinkage=routers['Mahalanobis'].shrinkage,
                                    mahalanobis_jitter=routers['Mahalanobis'].jitter,
                                    profile_file_bytes=(profiles/f'client_{cid}_router_profiles.pt').stat().st_size)
        write_json(out/'router_diagnostics.json', diagnostics)
        frame.to_csv(profiles/f'client_{cid}_predictions.csv', index=False)
        pd.DataFrame(rows).to_csv(out/'per_client_metrics.csv', index=False)
        print(f'Client {index+1}/{len(ids)} {cid}: '+str({p:round(float(np.mean(frame[p]==frame.y_true))*100,2)
                                                       for p in policy_routes}), flush=True)
        del model, detector, train, validation, routers, z
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    return summarize(frames, rows, out, seen, seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--batch-size', type=int, default=512)
    args = parser.parse_args()
    checkpoint = _load_checkpoint(args.checkpoint)
    config = checkpoint['config']
    if not str(config.get('git_commit', '')).startswith('03b9b53'):
        raise ValueError('This protocol requires the baseline 03b9b53 checkpoint')
    report = json.loads(Path(args.report).read_text(encoding='utf-8'))
    ids = [int(cid) for cid in report['final_metrics']['per_client']]
    loader = IncrementalDataLoader(args.data_dir)
    task = int(checkpoint['task_id'])
    classes = {t:list(map(int,loader.get_task_classes(t))) for t in range(task+1)}
    seed = int(config.get('random_seed', config.get('seed',42)))
    x, y = loader.get_test_data(task,cumulative=True)
    indices, selected_y, sampling = _limit_eval_samples(torch.arange(len(y))[:,None],y,50000,seed+task)
    index_shards, partition = _partition_test_data_by_client(indices,selected_y,ids,seed+104729*task)
    shards = {}
    for cid, shard in index_shards.items():
        selected = shard['X_test'].reshape(-1).long()
        shards[cid] = dict(X_test=x[selected],y_test=shard['y_test'],sample_ids=selected.numpy())
    out = Path(args.out); out.mkdir(parents=True,exist_ok=True)
    import subprocess
    source_commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    digest = hashlib.sha256()
    with Path(args.checkpoint).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024), b''):
            digest.update(block)
    write_json(out/'protocol.json',dict(kind='retrospective historical-local-train refit',
                                        sampling=sampling, partition=partition, seed=seed,
                                        evaluation_commit=source_commit, checkpoint_file_sha256=digest.hexdigest(),
                                        task_classes=classes, client_ids=ids,
                                        feature='adapter-free fc1, identity transform',
                                        training_commit=config['git_commit'], data_dir=args.data_dir))
    with threadpool_limits(limits=1):
        result = run_suite(checkpoint,config,report,loader,shards,classes,ids,out,
                           'cuda' if torch.cuda.is_available() else 'cpu',args.batch_size,seed=seed)
    print(result.to_string())


if __name__ == '__main__':
    main()
