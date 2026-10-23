"""Three-task, two-client synthetic integration test; no IDS claims."""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import os
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from tools.benchmark_denice_incremental import synthetic_data
from fed_learning.training.decentralized_denice_il import run_decentralized_denice_il


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('output/appliance_smoke'))
    parser.add_argument('--entrypoint', action='store_true', help='Exercise the Kaggle entry point in a subprocess')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    base = synthetic_data(args.output / 'data', 42)
    preset = json.loads((Path(__file__).resolve().parents[1] / 'configs/denice_appliance.json').read_text())
    base.update(preset)
    base.update(seed=42, random_seed=42, output_dir=str(args.output / 'run'),
                denice_post_task_eval=True, denice_eval_report_nomask=False,
                denice_eval_representative_ensemble=False,
                denice_router_replay_enabled=True, denice_router_replay_per_class=8)
    with (args.output / 'training.log').open('w', encoding='utf-8') as log:
        if args.entrypoint:
            root = Path(__file__).resolve().parents[1]
            env = os.environ.copy()
            env.update(DENICE_TRAIN_PHASE='5', DENICE_CODE_DIR=str(root),
                       DENICE_CONFIG_FILE=str(root / 'configs/denice_appliance.json'),
                       DENICE_CONFIG_OVERRIDES=json.dumps(base), PYTHONUTF8='1',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            subprocess.run([sys.executable, '-u', str(root / 'train_incremental_kaggle.py')],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            folders = list(args.output.glob('run*'))
            assert len(folders) == 1, 'Use a fresh output directory'
            result = {'output_dir': str(folders[0])}
        else:
            with contextlib.redirect_stdout(log):
                result = run_decentralized_denice_il(base)
    records = [json.loads(line) for line in (Path(result['output_dir']) / 'appliance_rounds.jsonl').read_text().splitlines()]
    assert len(records) == 6, 'Expected three tasks times two rounds; use a fresh output directory'
    assert {(r['task'], r['round']) for r in records} == {(t, r) for t in range(3) for r in range(2)}
    assert all('leaders' in r and 'decisions' in r for r in records)
    summary = dict(data_kind='synthetic_non_IDS', integration_rounds=len(records),
                   packets=sum(r['packets'] for r in records), accepted=sum(r['accepted'] for r in records),
                   status='passed', output_dir=result['output_dir'])
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
