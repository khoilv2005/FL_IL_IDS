"""Download results (8), audit class28 without backbone retraining or final test."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from importlib.metadata import version,PackageNotFoundError

RESULTS_DRIVE_URL=os.environ.get('DENICE_RESULTS8_DRIVE_URL','PASTE_RESULTS8_GOOGLE_DRIVE_URL')
DATA_DIR=os.environ.get('DENICE_DATA_DIR','/kaggle/input/datasets/khoilv2005/100-clients/100-clients')
RESULTS_ZIP=Path(os.environ.get('DENICE_RESULTS8_ZIP','/kaggle/working/results8_class28_source.zip'))
EXTRACT_DIR=Path('/kaggle/working/results8_class28_selected')
OUTPUT_DIR=Path('/kaggle/working/denice_class28_diagnostics')
BATCH_SIZE=512

code=os.environ.get('DENICE_CODE_DIR')
if not code:
    local=Path(globals().get('__file__','')).resolve().parent
    if (local/'fed_learning').is_dir():code=str(local)
if not code:
    code=tempfile.mkdtemp(prefix='denice-class28-source-')
    subprocess.run(['git','clone','--depth','1','https://github.com/khoilv2005/FL_IL_IDS.git',code],
                   env={**os.environ,'GIT_LFS_SKIP_SMUDGE':'1'},check=True)
sys.path.insert(0,code)
try:installed=version('scikit-learn')
except PackageNotFoundError:installed=None
if installed!='1.6.1':subprocess.run([sys.executable,'-m','pip','install','--quiet','scikit-learn==1.6.1'],check=True)
if not RESULTS_ZIP.is_file():
    if 'PASTE_' in RESULTS_DRIVE_URL:raise ValueError('Set RESULTS_DRIVE_URL to the results (8).zip Google Drive share link')
    subprocess.run([sys.executable,'-m','pip','install','--quiet','gdown'],check=True)
    import gdown
    RESULTS_ZIP.parent.mkdir(parents=True,exist_ok=True)
    downloaded=gdown.download(url=RESULTS_DRIVE_URL,output=str(RESULTS_ZIP),fuzzy=True,quiet=False)
    if not downloaded:raise RuntimeError('Google Drive download failed; check that the link is shared')
if not Path(DATA_DIR,'metadata.json').is_file():
    candidates=[p.parent for p in Path('/kaggle/input').rglob('metadata.json')
                if next(p.parent.glob('client_*_train.npz'),None) is not None]
    if len(candidates)!=1:raise FileNotFoundError('Set DATA_DIR to the original 100-client dataset mount')
    DATA_DIR=str(candidates[0])
if EXTRACT_DIR.exists() or OUTPUT_DIR.exists():raise ValueError('Use a fresh session or new extract/output paths; do not overwrite an audit')
from tools.debug_denice_class28 import unpack_results,run_debug
roles,run=unpack_results(RESULTS_ZIP,EXTRACT_DIR)
import torch
device='cuda' if torch.cuda.is_available() else 'cpu'
print('Class28 debug: no backbone training, no final-test read. Dataset:',DATA_DIR,flush=True)
print('Checkpoints: task4/task5 exact FP32 terminals; device:',device,flush=True)
print('Source config:',json.loads((run/'config.json').read_text()).get('denice_similarity_threshold'),flush=True)
try:
    run_debug(run,roles,DATA_DIR,OUTPUT_DIR,device=device,batch_size=BATCH_SIZE)
finally:
    if OUTPUT_DIR.exists():
        packed=shutil.make_archive(str(OUTPUT_DIR),'zip',root_dir=OUTPUT_DIR)
        print('Output ZIP (check completion.json):',packed,flush=True)
