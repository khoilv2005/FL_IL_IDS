"""End-to-end paired DENICE transfer ablation. Test results never select weights."""
import argparse
import contextlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from tools.benchmark_denice_incremental import synthetic_data
from fed_learning.training.decentralized_denice_il import run_decentralized_denice_il


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='JSON training config for the real IDS dataset')
    parser.add_argument('--output-dir', type=Path, default=Path('output/denice_transfer_probe'))
    parser.add_argument('--seeds', nargs='+', type=int, default=[23, 42, 73])
    parser.add_argument('--method', choices=['transfer', 'router_replay', 'continual', 'continual_bn', 'normalization', 'plasticity', 'capacity'], default='transfer')
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {'data_kind': 'real' if args.config else 'synthetic_non_IDS', 'method': args.method, 'runs': []}
    for seed in args.seeds:
        base = (json.loads(args.config.read_text(encoding='utf-8')) if args.config else
                synthetic_data(args.output_dir / f'data_{seed}', seed))
        base.update(seed=seed, random_seed=seed, task_start=0, resume_state_path=None,
                    denice_continual_width=0,
                    denice_plasticity_enabled=False,
                    denice_calibrate_plastic_bn=False,
                    denice_transfer_enabled=False, denice_router_replay_enabled=False,
                    denice_classifier_enabled=False, denice_classifier_validation_select=False,
                    denice_replay_selection='priority', denice_eval_route_mode='hard',
                    denice_post_task_eval=True, denice_eval_final_round=True)
        result_rows = {'seed': seed}
        for enabled in (False, True):
            name = args.method if enabled else 'baseline'
            config = {**base, 'denice_' + args.method + '_enabled': enabled,
                      'output_dir': str(args.output_dir / f'{name}_{seed}')}
            if args.method in ('continual', 'continual_bn'):
                config.update(denice_continual_width=128 if enabled else 0,
                              denice_router_replay_enabled=True,
                              denice_calibrate_plastic_bn=enabled and args.method == 'continual_bn',
                              denice_eval_route_mode='nomask' if enabled else 'hard')
            if args.method == 'normalization':
                config.update(denice_router_replay_enabled=True, denice_calibrate_plastic_bn=enabled)
            if args.method in ('plasticity', 'capacity'):
                config.update(denice_router_replay_enabled=True, denice_calibrate_plastic_bn=True,
                              denice_plasticity_enabled=enabled)
                if args.method == 'capacity':
                    config['denice_mature_fraction'] = 1.0
            with (args.output_dir / f'{name}_{seed}.log').open('w', encoding='utf-8') as log:
                with contextlib.redirect_stdout(log):
                    result = run_decentralized_denice_il(config)
            result_rows[name] = [{k: row.get(k) for k in ('task', 'accuracy', 'f1_macro', 'route_accuracy')}
                                   for row in result['history']['task_accuracies']]
            for row in result_rows[name]:
                rounds = [r for r in result['history']['round_metrics'] if r['task'] == row['task']]
                row['final_round_accuracy'] = rounds[-1].get('accuracy') if rounds else None
        summary['runs'].append(result_rows)
        (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(result_rows), flush=True)


if __name__ == '__main__':
    main()
