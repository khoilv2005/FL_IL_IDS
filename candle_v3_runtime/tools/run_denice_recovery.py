"""Explicit Kaggle ablations; run from a complete source checkout."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def build_config(arm, data_dir, output_dir, seed=42):
    if arm not in ('amp_control', 'fp32', 'balanced_router', 'cluster_control', 'row_overlay'):
        raise ValueError(f'Unknown arm: {arm}')
    config = {
        'data_dir': str(data_dir), 'output_dir': str(output_dir),
        'resume_output_dir': None, 'resume_state_path': None,
        'task_start': 0, 'task_end': 5, 'seed': seed, 'random_seed': seed,
        'denice_appliance_enabled': False, 'appliance_enabled': False,
        'denice_expected_method': 'denice_base',
        'denice_amp_enabled': arm == 'amp_control',
        'denice_fail_on_zero_update': arm != 'amp_control',
        'denice_router_mode': 'binary_cosine' if arm in ('amp_control', 'fp32') else 'multiclass_balanced',
        # Include every active client; do not switch silently from a coverage-
        # selected subpopulation to all clients when no complete client exists.
        'denice_eval_require_full_coverage': False,
        'denice_eval_max_clients': None,
        'denice_eval_max_samples': 50000,
        'denice_eval_final_task_only': True,
        'denice_eval_final_round': False,
        'denice_eval_local_validation': True,
        'eval_every': 999999,
    }
    if arm in ('cluster_control', 'row_overlay'):
        # Matched pair: same clustering, allocation, split and router. Only
        # row transfer differs. These arms are NOT a single-factor comparison
        # against balanced_router.
        config.update(json.loads((ROOT / 'configs/denice_appliance.json').read_text()))
        config.update(denice_appliance_enabled=arm == 'row_overlay',
                      denice_expected_method='row_overlay' if arm == 'row_overlay' else 'denice_base',
                      denice_eval_max_samples=50000,
                      denice_eval_require_full_coverage=False)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['amp_control', 'fp32', 'balanced_router', 'cluster_control', 'row_overlay'], required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--full-test', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = build_config(args.arm, args.data_dir, args.output_dir, args.seed)
    if args.full_test:
        config['denice_eval_max_samples'] = None
    print(json.dumps(config, indent=2), flush=True)
    if args.dry_run:
        return
    if not Path(args.data_dir).is_dir():
        parser.error('Dataset directory does not exist')
    if Path(args.output_dir).exists() and any(Path(args.output_dir).iterdir()):
        parser.error('Use a new empty output directory for each experiment')
    env = dict(os.environ)
    env.pop('DENICE_CONFIG_FILE', None)
    env.update(DENICE_TRAIN_PHASE='5', DENICE_VARIANT='legacy',
               DENICE_SEED=str(args.seed), DENICE_CODE_DIR=str(ROOT),
               DENICE_OUTPUT_DIR=str(args.output_dir),
               DENICE_CONFIG_OVERRIDES=json.dumps(config))
    subprocess.run([sys.executable, str(ROOT / 'train_incremental_kaggle.py')],
                   cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
