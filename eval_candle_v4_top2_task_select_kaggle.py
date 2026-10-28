# Frozen CANDLE V4 evaluation only. No training, no oracle task labels.
import os
from pathlib import Path
import subprocess
import sys
import zipfile
import json

CODE_REF = "13e78ce5a974aa238210d76c8e86f1f9e0e013d8"
DATA_DIR = "/kaggle/input/datasets/khoilv2005/100-clients"
RESULTS19_ZIP = ""  # Optional mounted ZIP containing T0-T4 archives.
RESULTS19_DRIVE_URL = "https://drive.google.com/file/d/1Qmnb2pHEq7JL55oOTrNm5e963XW09tYp/view?usp=drive_link"
RESULTS21_ZIP = ""  # Optional mounted results (21).zip instead of Drive.
RESULTS21_DRIVE_URL = "https://drive.google.com/file/d/1o5SJXPtDYfga2TLzQfBi5KfEttGT5PbD/view?usp=drive_link"
OUTPUT_DIR = "/kaggle/working/eval_candle_v4_round19_top2_task_select"

if not Path('/kaggle/working').is_dir():raise RuntimeError('Run on Kaggle GPU + Internet')
env={**os.environ,'GIT_LFS_SKIP_SMUDGE':'1','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
cache=Path('/kaggle/temp/candle_v4_six_task_select');cache.mkdir(parents=True,exist_ok=True)

def input_zip(path,url,label):
    if path:
        target=Path(path)
    elif url:
        target=cache/(label+'.zip')
        if not target.exists():
            subprocess.check_call([sys.executable,'-m','pip','install','-q','gdown==5.2.0'],env=env)
            import gdown
            part=target.with_suffix('.zip.download')
            print('Downloading '+label+'...',flush=True)
            done=gdown.download(url=url,output=str(part),fuzzy=True,quiet=False,resume=True,use_cookies=False)
            if not done or not zipfile.is_zipfile(part):raise RuntimeError('Drive download failed; mount ZIP instead')
            part.replace(target)
    else:
        matches=[]
        for p in Path('/kaggle/input').rglob('*.zip'):
            try:
                with zipfile.ZipFile(p) as z:
                    names=z.namelist()
                    if (any(n.endswith('/checkpoint_task_5_all_rounds.zip') for n in names)
                            and any(n.endswith('/candle_v4_final_test.json') for n in names)):matches.append(p)
            except zipfile.BadZipFile:continue
        if len(matches)!=1:raise ValueError('Mount results21 ZIP and set RESULTS21_ZIP; or set RESULTS21_DRIVE_URL. Found '+str(matches))
        target=matches[0]
    if not zipfile.is_zipfile(target):raise ValueError('Invalid ZIP: '+str(target))
    return target

# Resolve required results21 before downloading 4.54GB results19.
r21=input_zip(RESULTS21_ZIP,RESULTS21_DRIVE_URL,'results21')
r19=input_zip(RESULTS19_ZIP,RESULTS19_DRIVE_URL,'results19')
code=cache/'source'
if not code.exists():
    subprocess.check_call(['git','clone','--filter=blob:none','--no-checkout','https://github.com/khoilv2005/FL_IL_IDS.git',str(code)],env=env)
    subprocess.check_call(['git','sparse-checkout','set','--no-cone','/candle_v4_runtime/',
        '/tools/eval_candle_v4_task_round19_topk.py','/tools/candle_v4_historical_repair.py','/tools/candle_v4_top2_task_select.py'],cwd=code,env=env)
    subprocess.check_call(['git','checkout','--detach',CODE_REF],cwd=code,env=env)
if subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()!=CODE_REF:
    raise ValueError('Cached source differs; use a fresh session')
subprocess.check_call([sys.executable,'-m','pip','install','-q','-r',str(code/'candle_v4_runtime/requirements-candle-v4-kaggle.txt')],env=env)
subprocess.check_call([sys.executable,'-u',str(code/'tools/eval_candle_v4_task_round19_topk.py'),
    '--results19',str(r19),'--results21',str(r21),'--data-dir',DATA_DIR,
    '--cache',str(cache/'continuations'),'--output',OUTPUT_DIR,'--top2-policy','mass'],cwd=code,env=env)
receipt=json.loads((Path(OUTPUT_DIR)/'completion.json').read_text())
print('Complete:',receipt['completed'])
for row in receipt['summary']:
    print('Task',row['task'],row['mode'],'Acc',round(100*row['accuracy'],4),'Macro-F1',round(100*row['macro_f1'],4))
print('Save Version to retain summary.csv and six per-task reports. No training checkpoints changed.')
