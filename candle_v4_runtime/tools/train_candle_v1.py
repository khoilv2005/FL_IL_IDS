"""Kaggle CANDLE V1 baseline: full current-task TRAIN, no validation or experts."""
import argparse
import gc
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys


def resolve_data(root):
    root = Path(root).resolve()
    candidates = [root] if (root / 'metadata.json').is_file() else [
        path.parent for path in root.rglob('metadata.json')]
    candidates = [path for path in candidates
                  if (path / 'global_test_data.npz').is_file()
                  and next(path.glob('client_*_train.npz'), None) is not None]
    if len(candidates) != 1:
        raise ValueError(f'Set --data-dir to one original dataset; found {candidates}')
    return candidates[0]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def main():
    # Shared native path; an empty expert-arm set creates no expert or shadow models.
    if '--variant' in sys.argv:
        raise ValueError('CANDLE V1 always uses native-only')
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from tools.train_candle_v4 import main as train_full
    sys.argv.extend(['--variant','native-only'])
    train_full()


if __name__ == '__main__':
    main()
