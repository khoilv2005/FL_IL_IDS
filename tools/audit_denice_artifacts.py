"""Read-only source artifact analysis; writes compact derived audit evidence.

Uses the historical repo JSON/CSV and optional Downloads/42.zip ... 46.zip.
Never extracts or changes source archives, and never loads pickle checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from zipfile import ZipFile


def stats(values):
    values = [float(v) for v in values if v is not None]
    return dict(n=len(values), mean=mean(values), min=min(values), max=max(values),
                std=stdev(values) if len(values) > 1 else 0) if values else {}


def cluster_summary(rows):
    result = {}
    for task in sorted({r['task'] for r in rows}):
        group = [r for r in rows if r['task'] == task]
        result[task] = {
            'rounds': len(group), 'raw_valid': sum(bool(r.get('raw_valid', r.get('valid'))) for r in group),
            'policies': dict(Counter(str(r.get('effective_policy', 'unrecorded')) for r in group)),
            'K': stats(r.get('K_t') for r in group),
            'active_clients': stats(len(r.get('labels', {})) for r in group),
            'peer_fraction': stats(r['peer_aggregated_client_count']/len(r['labels'])
                                   for r in group if r.get('labels') and 'peer_aggregated_client_count' in r),
            'peer_alpha': stats(r.get('peer_alpha_sum_stats', {}).get('mean') for r in group),
            'group_size': stats(r.get('group_size_stats', {}).get('mean') for r in group),
            'singleton_clusters': stats(sum(v == 1 for v in r.get('cluster_sizes', {}).values()) for r in group),
            'silhouette': stats(r.get('silhouette') for r in group),
            'last_capacity': {},
        }
        last = group[-1]
        capacities = list(last.get('capacity_after_aggregation', {}).values())
        for layer in ('conv1', 'conv2', 'conv3', 'gru', 'fc1', 'fc2'):
            result[task]['last_capacity'][layer] = {
                key: stats(c.get(layer, {}).get(key) for c in capacities)
                for key in ('rho0', 'rhom', 'learner', 'mature', 'free', 'retired', 'total')
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--downloads', type=Path, default=Path(r'C:\Users\khoak\Downloads'))
    parser.add_argument('--output', type=Path, default=Path('audit_denice/artifact_evidence.json'))
    args = parser.parse_args()
    evidence = {'historical': {}, 'full_d2': {}}
    with Path('denice_eval_5seed_summary.csv').open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    evidence['historical']['zero_training_std_rows'] = sum(float(r['train_loss_std']) == 0 for r in rows)
    evidence['historical']['total_rows'] = len(rows)
    evidence['historical']['final_metrics'] = [r for r in rows if r['round_id'] == '19']
    log = json.loads(Path('denice_log.json').read_text(encoding='utf-8'))
    evidence['historical']['log_entry_types'] = dict(Counter(r.get('type') for r in log))
    evidence['historical']['data_by_task'] = {}
    for row in log:
        if row.get('type') != 'task_start':
            continue
        clients = row['client_data']
        hist = Counter()
        for c in clients.values():
            hist.update({int(k): v for k, v in c['class_hist'].items()})
        evidence['historical']['data_by_task'][row['task']] = {
            'logged_active': len(clients), 'samples': stats(c['num_samples'] for c in clients.values()),
            'class_hist': dict(sorted(hist.items())),
            'missing_class_fraction': stats(1-len(c['labels'])/len(row['new_classes']) for c in clients.values()),
        }
    cluster_file = Path('.tmp_denice2_json/cluster_history.json')
    if cluster_file.exists():
        evidence['historical']['clusters'] = cluster_summary(json.loads(cluster_file.read_text(encoding='utf-8')))
    for seed in range(42, 47):
        archive = args.downloads / f'{seed}.zip'
        if not archive.exists():
            continue
        with ZipFile(archive) as z:
            prefix = f'denice_full_d2_seed_{seed}/'
            terminal = prefix + 'terminal_task_5/'
            def read(name):
                return json.loads(z.read(name))
            cfg = read(prefix+'base_tasks_0_to_4/config.json')
            summary = read(terminal+'p6_evaluation/p6_evaluation_summary.json')
            hist = read(terminal+'training_history.json')
            policies = summary['summary']['coverage_aware_local']
            record = {
                'archive': str(archive), 'training_seed_in_config': cfg['random_seed'],
                'summary_seed': summary['training_seed'], 'config': cfg,
                'policies': policies, 'clusters': cluster_summary(read(terminal+'cluster_history.json')),
                'first_round_loss': hist['round_metrics'][0]['train_loss'],
                'base_validation': read(prefix+'base_tasks_0_to_4/audit_validation.json'),
                'terminal_validation': read(terminal+'audit_validation.json'),
                'per_task_final_accuracy': {}, 'route_confusions': {}, 'trace_audits': {},
            }
            # Stream the actual checkpoint bytes to corroborate recorded hashes.
            digest = hashlib.sha256()
            with z.open(terminal+'checkpoint_task_5_round_19.pt') as f:
                for chunk in iter(lambda: f.read(8*1024*1024), b''):
                    digest.update(chunk)
            record['actual_checkpoint_sha256'] = digest.hexdigest()
            record['hash_matches_all_policies'] = all(p['checkpoint_sha256'] == digest.hexdigest() for p in policies.values())
            for policy in policies:
                result = read(terminal+f'p6_evaluation/coverage_aware_local_{policy}.json')
                debug = result['metrics'].get('debug', {})
                per_class = debug.get('per_class', {})
                record['per_task_final_accuracy'][policy] = {
                    task: mean(float(per_class[str(c)]['accuracy']) for c in range(task*6, min(task*6+6,34)))
                    for task in range(6)
                } if len(per_class) == 34 else {}
                record['route_confusions'][policy] = debug.get('route_confusion', {})
                trace = debug.get('prediction_trace', {})
                targets, predictions = trace.get('targets', []), trace.get('predictions', [])
                if targets and len(targets) == len(predictions):
                    per_class_metrics = {}
                    for cls in range(34):
                        tp = sum(y == cls and p == cls for y,p in zip(targets,predictions))
                        support = targets.count(cls)
                        predicted = predictions.count(cls)
                        per_class_metrics[cls] = {
                            'support': support, 'recall': tp/support if support else 0,
                            'f1': 2*tp/(support+predicted) if support+predicted else 0,
                        }
                    record['trace_audits'][policy] = {
                        'sample_count': len(targets),
                        'accuracy': mean(y == p for y,p in zip(targets,predictions)),
                        'macro_f1': mean(c['f1'] for c in per_class_metrics.values()),
                        'per_class': per_class_metrics,
                        'per_task_global_class_macro_f1': {
                            task: mean(per_class_metrics[c]['f1'] for c in range(task*6,min(task*6+6,34)))
                            for task in range(6)
                        },
                        'routing_diagnostics': result['metrics'].get('routing_diagnostics', {}),
                        'per_episode_router_recall': result['metrics'].get('per_episode_router_recall', {}),
                        'first_errors': [
                            {'sample_subset_index': idx, 'client_id': cid, 'target': y, 'prediction': p}
                            for idx,cid,y,p in zip(trace.get('source_test_indices', []),
                                                  trace.get('client_ids', []), targets, predictions) if y != p
                        ][:5],
                    }
            evidence['full_d2'][seed] = record
            print(f'seed {seed}: hash_match={record["hash_matches_all_policies"]}, first_loss={record["first_round_loss"]}', flush=True)
    if evidence['full_d2']:
        seeds = list(evidence['full_d2'].values())
        evidence['full_d2_aggregate'] = {
            policy: {metric: stats(r['policies'][policy].get(metric) for r in seeds)
                     for metric in ('accuracy','f1_macro','f1_weighted','route_accuracy')}
            for policy in seeds[0]['policies']
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
