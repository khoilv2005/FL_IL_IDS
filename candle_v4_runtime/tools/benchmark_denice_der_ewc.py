"""Paired DENICE DER/EWC experiments; reports results, never chooses using test data."""
import argparse
import contextlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from tools.benchmark_denice_incremental import synthetic_data
from fed_learning.strategies.incremental.denice_variants import variant_preset
from fed_learning.training.decentralized_denice_il import run_decentralized_denice_il


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--overrides', type=Path, help='JSON applied after each variant preset')
    parser.add_argument('--methods', nargs='+', choices=['der', 'derpp', 'ewc', 'ewc_zero'], default=['derpp', 'ewc'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[23, 42, 73])
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overrides = json.loads(args.overrides.read_text(encoding='utf-8')) if args.overrides else {}
    report = dict(data_kind='real_IDS_config' if args.config else 'synthetic_non_IDS', runs=[])
    for seed in args.seeds:
        base = json.loads(args.config.read_text(encoding='utf-8')) if args.config else synthetic_data(args.output_dir / f'data_{seed}', seed)
        for method in args.methods:
            config = {**base, **variant_preset('ewc' if method == 'ewc_zero' else method)}
            if not args.config:
                config.update(denice_ewc_fisher_samples=8, denice_replay_capacity=48 if method in ('der','derpp') else 0)
            config.update(overrides)
            if method == 'ewc_zero':
                config['denice_ewc_lambda'] = 0.
            config.setdefault('denice_post_task_eval', True)
            config.update(seed=seed, random_seed=seed, resume_state_path=None, task_start=0,
                          output_dir=str(args.output_dir / f'{method}_{seed}'))
            with (args.output_dir / f'{method}_{seed}.log').open('w', encoding='utf-8') as log:
                with contextlib.redirect_stdout(log):
                    result = run_decentralized_denice_il(config)
            row = dict(method=method, seed=seed, output_dir=result['output_dir'],
                       validation_tasks=[{k: value.get(k) for k in ('task','accuracy','f1_macro','avg_forgetting')}
                                         for value in result['history'].get('validation_task_accuracies', [])],
                       tasks=[{k: value.get(k) for k in ('task','accuracy','f1_macro','route_accuracy','avg_forgetting')}
                              for value in result['history']['task_accuracies']],
                       final_rounds=[{k: r.get(k) for k in ('task','round','accuracy','f1_macro')}
                                     for r in result['history']['round_metrics'] if r['round'] == config['rounds_per_task'] - 1])
            report['runs'].append(row)
            (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
