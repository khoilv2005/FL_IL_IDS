"""Paste into a separate Kaggle notebook after Task 0-5 completes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

CHECKPOINT_ROOT = None  # Set to FINAL output, not the incomplete V59 output.
SOURCE_ZIP = None
DATA_DIR = '/kaggle/input/datasets/khoilv2005/100-clients/100-clients'
OUTPUT_DIR = '/kaggle/working/eval_cluster_servers'

def exactly_one(paths, label):
    paths = list(paths)
    if len(paths) != 1:
        raise ValueError(f'Set {label} explicitly. Candidates: {paths}')
    return paths[0]

root = Path(CHECKPOINT_ROOT) if CHECKPOINT_ROOT else exactly_one(
    (p.parent for p in Path('/kaggle/input').rglob('completion.json')
     if (p.parent/'checkpoint_task_5_all_rounds.zip').is_file()), 'CHECKPOINT_ROOT')
receipt = json.loads((root/'completion.json').read_text())
assert receipt.get('completed') and receipt.get('tasks_completed') == 6, 'Finish Task 0-5 first'
bundle = Path(SOURCE_ZIP) if SOURCE_ZIP else exactly_one(
    Path('/kaggle/input').rglob('denice_v59_exact_source.zip'), 'SOURCE_ZIP')
source = Path(tempfile.mkdtemp(prefix='denice-v59-eval-'))
with zipfile.ZipFile(bundle) as archive:
    if any(not (source/n).resolve().is_relative_to(source.resolve()) for n in archive.namelist()):
        raise ValueError('Unsafe archive path')
    archive.extractall(source)
env = dict(os.environ, DENICE_CODE_DIR=str(source))
subprocess.run([sys.executable, '-u', str(source/'eval_selective.py'),
    '--checkpoint-root', str(root), '--data-dir', DATA_DIR,
    '--output', OUTPUT_DIR, '--batch-size', '8192'],
    env=env, cwd=source, check=True)
