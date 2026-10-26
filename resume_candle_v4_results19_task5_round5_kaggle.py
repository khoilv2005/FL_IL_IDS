# CANDLE V4: exact results19 continuation, Task5 Round5..19 + final V4-only TEST.
import os
from pathlib import Path
import subprocess
import sys

CODE_REF = "3731fe1cee41c693fbfb7327b7fc3b2e86a0f3ab"
DATA_DIR = "/kaggle/input/datasets/khoilv2005/100-clients"
OUTPUT_DIR = "/kaggle/working/results_candle_v4_only_seed42"

if not Path('/kaggle/working').is_dir():raise RuntimeError('Run on Kaggle with GPU + Internet')
env={**os.environ,'GIT_LFS_SKIP_SMUDGE':'1','PYTHONUNBUFFERED':'1'}
for name in ('DENICE_CONFIG_FILE','DENICE_CONFIG_OVERRIDES','DENICE_RESUME_STATE_PATH','DENICE_CONFIG_ONLY','DENICE_TASK_END'):
    env.pop(name,None)
code=Path('/kaggle/temp/candle_v4_results19_resume_source');code.parent.mkdir(parents=True,exist_ok=True)
if not code.exists():
    subprocess.check_call(['git','clone','--filter=blob:none','--no-checkout','https://github.com/khoilv2005/FL_IL_IDS.git',str(code)],env=env)
    subprocess.check_call(['git','sparse-checkout','set','--no-cone','/candle_v4_runtime/'],cwd=code,env=env)
    subprocess.check_call(['git','checkout','--detach',CODE_REF],cwd=code,env=env)
actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()
if actual!=CODE_REF:raise ValueError('Source differs from the locked version; use a fresh Kaggle session')
runtime=code/'candle_v4_runtime'
subprocess.check_call([sys.executable,'-m','pip','install','-q','-r',str(runtime/'requirements-candle-v4-kaggle.txt')],env=env)

import hashlib
import json
import shutil
import zipfile

RESULTS19_ZIP = ""  # Optional mounted /kaggle/input/.../results (19).zip
RESULTS19_DRIVE_URL = "https://drive.google.com/file/d/1Qmnb2pHEq7JL55oOTrNm5e963XW09tYp/view?usp=drive_link"
EXPECTED_STATE_SHA = "bcf6cbc092a0e45bc0dc8d5689c5439ea87e9fa712ce4f628ab4bfdefb8dc3b8"
EXPECTED_LOCK_SHA = "f5c22edadc48584b4589b38dee1093b46d00ce1c4740c03af97526299a83adff"

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def restore_results19(archive,output):
    out=Path(output)
    if out.exists() and any(out.iterdir()):raise ValueError('Output is not empty; use a fresh Kaggle session')
    prefix='results_candle_v4_only_seed42/'
    with zipfile.ZipFile(archive) as z:
        lock_bytes=z.read(prefix+'candle_v4_run_lock.json')
        if hashlib.sha256(lock_bytes).hexdigest()!=EXPECTED_LOCK_SHA:raise ValueError('Different results19 run lock')
        lock=json.loads(lock_bytes)
        if lock['source_commit']!=CODE_REF or lock['config']['output_dir']!=str(out):raise ValueError('Locked source/output mismatch')
        for name in ['continuation_state_latest.pt','checkpoint_index.json','checkpoint_task_5_base.zip',
                     *[f'checkpoint_task_5_round_{r}.zip' for r in range(5)]]:
            z.getinfo(prefix+name)
        if prefix+'candle_v4_final_test.json' in z.namelist():raise ValueError('Input already contains final TEST; expected partial results19')
        out.mkdir(parents=True,exist_ok=True)
        for item in z.infolist():
            if item.is_dir() or not item.filename.startswith(prefix):continue
            name=item.filename[len(prefix):]
            if '/' in name or '\\' in name:raise ValueError('Unexpected nested output member')
            if name.startswith('checkpoint_task_') and not name.startswith('checkpoint_task_5_'):continue
            target=out/name
            with z.open(item) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst,8*1024*1024)
    state=out/'continuation_state_latest.pt'
    if sha(state)!=EXPECTED_STATE_SHA:raise ValueError('Continuation checksum failed')
    (out/'results19_resume_receipt.json').write_text(json.dumps(dict(source_commit=CODE_REF,
        state_sha256=EXPECTED_STATE_SHA,lock_sha256=EXPECTED_LOCK_SHA,resume_task=5,resume_round=5,
        policy='Exact round continuation; no memory migration or T0-T4 retraining',
        historical_task_archives='T0-T4 archives remain in original results19 ZIP'),indent=2))
    return state

cache=Path('/kaggle/temp/candle_v4_results19');cache.mkdir(parents=True,exist_ok=True)
if RESULTS19_ZIP:
    archive=Path(RESULTS19_ZIP)
else:
    archive=cache/'results19.zip'
    if not archive.exists():
        subprocess.check_call([sys.executable,'-m','pip','install','-q','gdown==5.2.0'],env=env)
        import gdown
        part=archive.with_suffix('.zip.download')
        print('Downloading results19 (~4.54 GB)...',flush=True)
        done=gdown.download(url=RESULTS19_DRIVE_URL,output=str(part),fuzzy=True,quiet=False,resume=True,use_cookies=False)
        if not done or not zipfile.is_zipfile(part):raise RuntimeError('Drive quota/sharing error; mount results19 ZIP instead')
        part.replace(archive)
if not zipfile.is_zipfile(archive):raise ValueError('results19 input is not a ZIP')
RESUME_STATE=restore_results19(archive,OUTPUT_DIR)
probe=r"""
import sys,torch,sklearn
if sklearn.__version__!='1.6.1':raise RuntimeError('Pinned sklearn1.6.1 required')
s=torch.load(sys.argv[1],map_location='cpu',weights_only=False,mmap=True)
m=s['meta'];r=s.get('round_state') or {};v=s['candle_v4_shadow_state']
if (m.get('boundary')!='round' or m.get('completed_task')!=4 or m.get('resume_from_task')!=5
        or r.get('next_round')!=5 or list(v['boundary'])!=[5,4]
        or v['version']!='candle_v4_continuous_bootstrap_v8'
        or v['spec'].get('evaluation_mode')!='v4_only' or len(s['client_model_states'])!=100):
    raise ValueError('Expected verified results19 T5/R4 boundary')
print('Checkpoint PASS: 100 clients; resume T5/R5 through R19, then V4-only Full TEST.',flush=True)
"""
subprocess.check_call([sys.executable,'-c',probe,str(RESUME_STATE)],cwd=runtime,env=env)

subprocess.check_call([sys.executable,'-u',str(runtime/'tools/train_candle_v4.py'),
    '--data-dir',DATA_DIR,'--output',OUTPUT_DIR,'--variant','coverage',
    '--router-mode','continuous_class_v1','--continuous-bootstrap','--v4-only','--resume',str(RESUME_STATE)],cwd=runtime,env=env)
import json
receipt=json.loads((Path(OUTPUT_DIR)/'candle_v4_completion.json').read_text())
print('Completed:',receipt['completed'],'V4 final metrics:',receipt['final_test']['V4'])
print('Use Kaggle Save Version to keep checkpoints and metrics.')
