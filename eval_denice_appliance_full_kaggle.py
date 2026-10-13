"""Evaluate completed DeNICE + APPLIANCE Task5; never restart training."""
import json
import os
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError

# Set either the mounted output path or its Google Drive share link.
RESULTS_DRIVE_URL = os.environ.get('APPLIANCE_RESULTS_DRIVE_URL', '')
RESULTS_PATH = os.environ.get('APPLIANCE_RESULTS_PATH',
    '' if RESULTS_DRIVE_URL else '/kaggle/input/datasets/luuquanghuy636/appliance-v2')
DATA_DIR = os.environ.get('DENICE_DATA_DIR', '/kaggle/input/datasets/khoilv2005/100-clients/100-clients')
ROLE_DIR = os.environ.get('DENICE_CLEAN_ROLES_DIR')
BATCH_SIZE = 512
STAMP = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
OUT = Path('/kaggle/working') / f'denice_appliance_full_eval_task_5_{STAMP}'
WORK = Path('/kaggle/working') / f'appliance_eval_inputs_{STAMP}'

code = os.environ.get('DENICE_CODE_DIR')
if not code:
    code = tempfile.mkdtemp(prefix='denice-appliance-eval-source-')
    subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/khoilv2005/FL_IL_IDS.git', code],
                   env={**os.environ, 'GIT_LFS_SKIP_SMUDGE':'1'}, check=True)
sys.path.insert(0, code)
revision = subprocess.check_output(['git', '-C', code, 'rev-parse', 'HEAD'], text=True).strip()
print('Evaluation source:', revision, flush=True)
try: sklearn_version = version('scikit-learn')
except PackageNotFoundError: sklearn_version = None
if sklearn_version != '1.6.1':
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', 'scikit-learn==1.6.1'], check=True)
if RESULTS_DRIVE_URL:
    if RESULTS_PATH: raise ValueError('Set RESULTS_PATH or RESULTS_DRIVE_URL, not both')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', 'gdown'], check=True)
    import gdown
    RESULTS_PATH = f'/kaggle/working/appliance_training_output_{STAMP}.zip'
    if not gdown.download(url=RESULTS_DRIVE_URL, output=RESULTS_PATH, fuzzy=True):
        raise RuntimeError('Google Drive download failed; mount the training output as a Kaggle Dataset')
if not RESULTS_PATH:
    raise ValueError('Set RESULTS_PATH to this completed training output, or RESULTS_DRIVE_URL to its ZIP. '
                     'This notebook does not use an older checkpoint by default.')
if not Path(DATA_DIR, 'metadata.json').is_file():
    candidates = [p.parent for p in Path('/kaggle/input').rglob('metadata.json')
                  if next(p.parent.glob('client_*_train.npz'), None) is not None]
    if len(candidates) != 1: raise FileNotFoundError('Set DENICE_DATA_DIR to the original 100-client dataset')
    DATA_DIR = str(candidates[0])
from tools.run_appliance_full_evaluation import run
try:
    summary = run(RESULTS_PATH, DATA_DIR, OUT, WORK, role_dir=ROLE_DIR, batch_size=BATCH_SIZE)
    print('Full test metrics:', json.dumps(summary['metrics'], indent=2), flush=True)
    print('Actual APPLIANCE activity:', json.dumps(summary['appliance_activity'], indent=2), flush=True)
except Exception as exc:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'evaluation_error.json').write_text(json.dumps(dict(completed=False, error=str(exc),
        error_type=type(exc).__name__, evaluation_commit=revision), indent=2), encoding='utf-8')
    raise
finally:
    if OUT.exists():
        print('Evaluation output ZIP:', shutil.make_archive(str(OUT), 'zip', root_dir=OUT), flush=True)
