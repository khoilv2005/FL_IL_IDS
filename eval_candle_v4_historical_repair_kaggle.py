# EVALUATION ONLY. No backbone or expert training.
import os
from pathlib import Path
import subprocess
import sys
import zipfile

CODE_REF = "842a59ad4e2d334f32d0150490f69322b1ca9c01"
DATA_DIR = "/kaggle/input/datasets/khoilv2005/100-clients"
RESULTS16_DRIVE_URL = "https://drive.google.com/file/d/12m7ZyoowmKNhKPiSQQ0FH2mzhAweD7tw/view?usp=drive_link"
RESULTS17_DRIVE_URL = ""  # Paste the results (17).zip Drive sharing link here.
RESULTS16_ZIP = ""  # Optional mounted /kaggle/input/.../results (16).zip.
RESULTS17_ZIP = ""  # Optional mounted /kaggle/input/.../results (17).zip.
OUTPUT_DIR = "/kaggle/working/results_candle_v4_historical_repair_seed42"

if not Path('/kaggle/working').is_dir():raise RuntimeError('Run this notebook on Kaggle')
env={**os.environ,'GIT_LFS_SKIP_SMUDGE':'1','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
for name in ('DENICE_CONFIG_FILE','DENICE_CONFIG_OVERRIDES','DENICE_RESUME_STATE_PATH','DENICE_CONFIG_ONLY','DENICE_TASK_END'):
    env.pop(name,None)
cache=Path('/kaggle/temp/candle_v4_historical_repair');cache.mkdir(parents=True,exist_ok=True)

def input_zip(path,url,label):
    if path:
        target=Path(path)
        if not zipfile.is_zipfile(target):raise ValueError('Invalid mounted ZIP: '+str(target))
        return target
    if not url:
        candidates=[]
        for p in Path('/kaggle/input').rglob('*.zip'):
            try:
                with zipfile.ZipFile(p) as z:
                    if any(n.endswith('/checkpoint_task_5_all_rounds.zip') or n=='checkpoint_task_5_all_rounds.zip' for n in z.namelist()):candidates.append(p)
            except zipfile.BadZipFile:continue
        if label=='results17' and len(candidates)==1:return candidates[0]
        raise ValueError('Set RESULTS17_DRIVE_URL after upload, or mount results17 ZIP and set RESULTS17_ZIP.')
    target=cache/(label+'.zip')
    if not target.exists():
        subprocess.check_call([sys.executable,'-m','pip','install','-q','gdown==5.2.0'],env=env)
        import gdown
        part=target.with_suffix('.zip.download')
        print('Downloading '+label+' from Google Drive...',flush=True)
        done=gdown.download(url=url,output=str(part),fuzzy=True,quiet=False,resume=True,use_cookies=False)
        if not done or not zipfile.is_zipfile(part):raise RuntimeError('Drive download is not a ZIP; check sharing/quota or mount the ZIP')
        part.replace(target)
    if not zipfile.is_zipfile(target):raise ValueError('Invalid cached results ZIP')
    return target

# Resolve results17 first: do not download4.7GB results16 before its input is ready.
r17=input_zip(RESULTS17_ZIP,RESULTS17_DRIVE_URL,'results17')
r16=input_zip(RESULTS16_ZIP,RESULTS16_DRIVE_URL,'results16')
code=cache/'source'
if not code.exists():
    subprocess.check_call(['git','clone','--filter=blob:none','--no-checkout','https://github.com/khoilv2005/FL_IL_IDS.git',str(code)],env=env)
    subprocess.check_call(['git','sparse-checkout','set','--no-cone','/candle_v4_runtime/',
        '/tools/candle_v4_historical_repair.py','/tools/eval_candle_v4_historical_repair.py'],cwd=code,env=env)
    subprocess.check_call(['git','checkout','--detach',CODE_REF],cwd=code,env=env)
actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()
if actual!=CODE_REF:raise ValueError('Cached source differs; use a fresh Kaggle session')
subprocess.check_call([sys.executable,'-m','pip','install','-q','-r',str(code/'candle_v4_runtime/requirements-candle-v1-kaggle.txt')],env=env)
print('Repair saved TRAIN feature memory; keep frozen V1/backbone/experts. No retraining tasks.',flush=True)
subprocess.check_call([sys.executable,'-u',str(code/'tools/eval_candle_v4_historical_repair.py'),
    '--results16',str(r16),'--results17',str(r17),'--data-dir',DATA_DIR,
    '--output',OUTPUT_DIR,'--cache',str(cache/'continuations')],cwd=code,env=env)
import json
receipt=json.loads((Path(OUTPUT_DIR)/'repair_completion.json').read_text())
print('Completed:',receipt['completed'],'repaired clients:',receipt['repaired_clients'])
print('Old V4:',receipt['old_v4_metrics'],'Readout:',receipt['metrics'])
print('Native control exact matrix match:',receipt['native_control_exact_matrix_match'])
print('Use Kaggle Save Version to retain output. This is a repaired-checkpoint development readout, not fresh training.')
