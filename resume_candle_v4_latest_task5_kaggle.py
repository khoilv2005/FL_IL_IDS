# CANDLE V4: explicit repaired Task4 import, train Task5, V4-only final TEST.
import os
from pathlib import Path
import subprocess
import sys

CODE_REF = "2dbb0c0ead5b5f865d07a089f6de8357087e631a"
DATA_DIR = "/kaggle/input/datasets/khoilv2005/100-clients"
OUTPUT_DIR = "/kaggle/working/results_candle_v4_latest_resume_task5_seed42"

if not Path('/kaggle/working').is_dir():raise RuntimeError('Run on Kaggle with GPU + Internet')
env={**os.environ,'GIT_LFS_SKIP_SMUDGE':'1','PYTHONUNBUFFERED':'1'}
for name in ('DENICE_CONFIG_FILE','DENICE_CONFIG_OVERRIDES','DENICE_RESUME_STATE_PATH','DENICE_CONFIG_ONLY','DENICE_TASK_END'):
    env.pop(name,None)
code=Path('/kaggle/temp/candle_v4_latest_resume_source');code.parent.mkdir(parents=True,exist_ok=True)
if not code.exists():
    subprocess.check_call(['git','clone','--filter=blob:none','--no-checkout','https://github.com/khoilv2005/FL_IL_IDS.git',str(code)],env=env)
    subprocess.check_call(['git','sparse-checkout','set','--no-cone','/candle_v4_runtime/','/tools/migrate_candle_v4_task4.py','/tools/candle_v4_historical_repair.py'],cwd=code,env=env)
    subprocess.check_call(['git','checkout','--detach',CODE_REF],cwd=code,env=env)
actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()
if actual!=CODE_REF:raise ValueError('Source differs from the locked version; use a fresh Kaggle session')
runtime=code/'candle_v4_runtime'
subprocess.check_call([sys.executable,'-m','pip','install','-q','-r',str(runtime/'requirements-candle-v4-kaggle.txt')],env=env)

import zipfile
RESULTS16_ZIP = ""  # Optional mounted /kaggle/input/.../results (16).zip
RESULTS16_DRIVE_URL = "https://drive.google.com/file/d/12m7ZyoowmKNhKPiSQQ0FH2mzhAweD7tw/view?usp=drive_link"
cache = Path('/kaggle/temp/candle_v4_task4_input'); cache.mkdir(parents=True, exist_ok=True)
if RESULTS16_ZIP:
    archive = Path(RESULTS16_ZIP)
else:
    archive = cache/'results16.zip'
    if not archive.exists():
        subprocess.check_call([sys.executable,'-m','pip','install','-q','gdown==5.2.0'],env=env)
        import gdown
        part = archive.with_suffix('.zip.download')
        print('Downloading results16 (~4.70 GB); only T0-T4 boundary states will be extracted.',flush=True)
        done = gdown.download(url=RESULTS16_DRIVE_URL,output=str(part),fuzzy=True,quiet=False,resume=True,use_cookies=False)
        if not done or not zipfile.is_zipfile(part):raise RuntimeError('Drive quota/sharing error; mount results16 ZIP instead')
        part.replace(archive)
if not zipfile.is_zipfile(archive):raise ValueError('results16 is not a ZIP')

print('Repair saved router memory; train T5/R0..19; final V4-only TEST.',flush=True)
subprocess.check_call([sys.executable,'-u',str(runtime/'tools/train_candle_v4.py'),
    '--data-dir',DATA_DIR,'--output',OUTPUT_DIR,'--variant','coverage',
    '--router-mode','continuous_class_v1','--continuous-bootstrap','--v4-only','--import-task4-archive',str(archive)],cwd=runtime,env=env)
import json
receipt=json.loads((Path(OUTPUT_DIR)/'candle_v4_completion.json').read_text())
print('Completed:',receipt['completed'],'V4 final metrics:',receipt['final_test']['V4'])
print('Use Kaggle Save Version to keep checkpoints and metrics.')
