"""Paste into a fresh Kaggle notebook cell after adding V59 outputs and source ZIP."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

# Set explicit paths if several matching inputs are mounted.
CHECKPOINT_ROOT = None
SOURCE_ZIP = None
DATA_DIR = '/kaggle/input/datasets/khoilv2005/100-clients/100-clients'
OUTPUT_DIR = '/kaggle/working/results_denice_cluster_server_seed_42_resume'

def exactly_one(paths, label):
    paths = list(paths)
    if len(paths) != 1:
        raise ValueError(f'Set {label} explicitly. Candidates: {paths}')
    return paths[0]

root = Path(CHECKPOINT_ROOT) if CHECKPOINT_ROOT else exactly_one(
    (p.parent for p in Path('/kaggle/input').rglob('execution_provenance.json')
     if (p.parent/'checkpoint_task_2_round_000.zip').is_file()), 'CHECKPOINT_ROOT')
bundle = Path(SOURCE_ZIP) if SOURCE_ZIP else exactly_one(
    Path('/kaggle/input').rglob('denice_v59_exact_source.zip'), 'SOURCE_ZIP')
provenance = json.loads((root/'execution_provenance.json').read_text())
import torch
if torch.__version__ != provenance['torch']:
    raise RuntimeError(f"Required PyTorch {provenance['torch']}; found {torch.__version__}")
if torch.cuda.device_count() != 2:
    raise RuntimeError('Select GPU T4 x2, matching the original CUDA RNG topology.')

source = Path(tempfile.mkdtemp(prefix='denice-v59-'))
with zipfile.ZipFile(bundle) as archive:
    if any(not (source/n).resolve().is_relative_to(source.resolve()) for n in archive.namelist()):
        raise ValueError('Unsafe archive path')
    archive.extractall(source)
for name, expected in provenance['source_sha256'].items():
    if hashlib.sha256((source/name).read_bytes()).hexdigest() != expected:
        raise ValueError(f'Original source mismatch: {name}')

env = dict(os.environ)
for key in ('DENICE_CONFIG_FILE', 'DENICE_CONFIG_OVERRIDES', 'DENICE_RESUME_STATE_PATH',
            'DENICE_VARIANT', 'DENICE_SOURCE_ZIP'):
    env.pop(key, None)
env.update(DENICE_CODE_DIR=str(source), DENICE_RUN_PROFILE='cluster_server',
    DENICE_TRAIN_PHASE='5', DENICE_TASK_END='5', DENICE_SEED='42',
    DENICE_CLUSTER_RESUME_ROOT=str(root), DENICE_CLUSTER_RESUME_TASK='2',
    DENICE_CLUSTER_RESUME_ROUND='0', DENICE_DATA_DIR=DATA_DIR,
    DENICE_OUTPUT_DIR=OUTPUT_DIR, DENICE_BATCH_SIZE='2048',
    DENICE_DEVICE='cuda:0', DENICE_WORKER_DEVICES='cuda:0,cuda:1',
    DENICE_OUTPUT_BUDGET_GIB='18')
print('Verified original source. Resume T2 R0 -> next T2 R1.', flush=True)
subprocess.run([sys.executable, '-u', str(source/'train_incremental_kaggle.py')],
               env=env, cwd=source, check=True)
