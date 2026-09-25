"""Paired DENICE readout experiment; synthetic by default, real JSON optional.

This measures hard routing vs the incremental readout on identical models and
test sets. It is NOT a replay-vs-no-replay end-to-end benchmark.
"""
import argparse
import contextlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from fed_learning.training.decentralized_denice_il import run_decentralized_denice_il


def synthetic_data(path, seed):
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(6,16,1)).astype('float32')
    for cid in range(2):
        labels = np.repeat(np.arange(6), 40)
        inputs = centers[labels] + .2*rng.normal(size=(len(labels),16,1)) + cid*.05
        np.savez(path/f'client_{cid}_train.npz', X_train=inputs.astype('float32'), y_train=labels)
    labels = np.repeat(np.arange(6), 80)
    test = centers[labels] + .2*rng.normal(size=(len(labels),16,1))
    np.savez(path/'global_test_data.npz', X_test=test.astype('float32'), y_test=labels)
    (path/'metadata.json').write_text(json.dumps({
        'task_structure': {'total_classes':6, 'task_classes':{'0':[0,1],'1':[2,3],'2':[4,5]}}}))
    return dict(algorithm='denice', mode='decentralized', data_dir=str(path), total_classes=6,
                num_clients=2, denice_max_clients=2, task_end=2, rounds_per_task=2,
                batch_size=32, nice_phase_epochs=1, learning_rate=.001, eval_every=9999,
                denice_structural_protection=True, denice_fixed_task_allocation=True,
                denice_allocation_policy='fixed_per_class', denice_memory_policy='local_replay',
                denice_router_mode='binary_cosine', denice_router_update_schedule='every_round',
                denice_refresh_router_memory_after_aggregation=False,
                denice_shared_context_eval=False, denice_capsule_mode='paper',
                denice_clustering_mode='paper', denice_validation_fraction=.2,
                denice_aggregation_update_mode='local_delta', denice_aggregation_rho='reserve',
                denice_fisher_samples=2, denice_canc_schedule='task_end', denice_canc_mode='paper',
                denice_adapter_mode='linear_input', denice_collaboration_guard_mode='off',
                denice_checkpoint_format='full', save_continuation_every_task=True,
                denice_replay_capacity=48, denice_replay_batch_size=12,
                denice_replay_candidate_limit=48, denice_classifier_per_class=24)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='Real experiment JSON; omit for synthetic data.')
    parser.add_argument('--output-dir', type=Path, default=Path('output/denice_incremental_probe'))
    parser.add_argument('--seeds', nargs='+', type=int, default=[23,42,73])
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for seed in args.seeds:
        config = (json.loads(args.config.read_text(encoding='utf-8')) if args.config
                  else synthetic_data(args.output_dir/f'data_{seed}', seed))
        config.update(seed=seed, random_seed=seed, task_start=0, resume_state_path=None,
                      output_dir=str(args.output_dir/f'seed_{seed}'),
                      denice_classifier_enabled=True, denice_classifier_validation_select=True,
                      denice_replay_selection='herding', denice_eval_route_mode='local_lda',
                      denice_post_task_eval=True, denice_eval_report_nomask=False,
                      denice_eval_representative_ensemble=False)
        with (args.output_dir/f'seed_{seed}.log').open('w', encoding='utf-8') as log:
            with contextlib.redirect_stdout(log):
                result = run_decentralized_denice_il(config)
        rows = []
        for metric in result['history']['task_accuracies']:
            rows.append({k: metric.get(k) for k in (
                'task','accuracy','f1_macro','hard_accuracy','hard_f1_macro',
                'gain_vs_hard_accuracy','route_accuracy')})
        summaries.append({'seed':seed, 'output_dir':result['output_dir'], 'tasks':rows})
        print(json.dumps(summaries[-1]), flush=True)
    summary = {'data_kind':'real' if args.config else 'synthetic_non_IDS',
               'comparison':'paired_readout_same_backbone_not_end_to_end_baseline', 'runs':summaries}
    (args.output_dir/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
