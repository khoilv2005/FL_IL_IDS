# CANDLE V4 coverage-aware full development. Edit paths only.
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

CODE_REF = "37a277c8f6ae85e74c9553d4d30e5e27f052ca93"
DATA_DIR = "/kaggle/input/datasets/khoilv2005/100-clients"
OUTPUT_DIR = "/kaggle/working/results_candle_v4_continuous_class_seed42"
VARIANT = "coverage"  # V4 only. Keep the separate V1 notebook unchanged.
ROUTER_MODE = "continuous_class_v1"  # Locked in config; binary_cosine remains an explicit variant.
# Optional mounted round continuation .pt or task archive .zip; empty=fresh T0.
# Resume needs the same continuous-v7 run lock and router configuration.
RESUME_STATE = ""

if not Path('/kaggle/working').is_dir():
    raise RuntimeError('Run this notebook on Kaggle, not locally')
env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
for name in ('DENICE_CONFIG_FILE','DENICE_CONFIG_OVERRIDES','DENICE_RESUME_STATE_PATH',
             'DENICE_CONFIG_ONLY','DENICE_TASK_END'):
    env.pop(name,None)
code = Path('/kaggle/working/candle_v4_continuous_source')
if not code.exists():
    subprocess.check_call(['git','clone','--filter=blob:none','--no-checkout',
        'https://github.com/khoilv2005/FL_IL_IDS.git',str(code)],env=env)
    subprocess.check_call(['git','sparse-checkout','set','--no-cone','/candle_v4_runtime/'],cwd=code,env=env)
    subprocess.check_call(['git','checkout','--detach',CODE_REF],cwd=code,env=env)
actual = subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()
if actual != CODE_REF:raise ValueError('Source pin mismatch; use a fresh source folder')
runtime = code/'candle_v4_runtime'
subprocess.check_call([sys.executable,'-m','pip','install','-q','-r',
                       str(runtime/'requirements-candle-v1-kaggle.txt')],env=env)
args = [sys.executable,'-u',str(runtime/'tools/train_candle_v4.py'),
        '--data-dir',DATA_DIR,'--output',OUTPUT_DIR,'--variant',VARIANT,'--router-mode',ROUTER_MODE]
if RESUME_STATE:
    source = Path(RESUME_STATE);out=Path(OUTPUT_DIR);out.mkdir(parents=True,exist_ok=True)
    if source.suffix=='.zip':
        with zipfile.ZipFile(source) as archive:
            for name in archive.namelist():
                relative=Path(name)
                if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe resume archive')
                if name=='candle_v4_run_lock.json' or name.startswith(('v4_threshold_locks/','v4_gate_journal/')):
                    target=out/relative;target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(archive.read(name))
    else:
        if source.parent.resolve()!=out.resolve():
            shutil.copy2(source.parent/'candle_v4_run_lock.json',out/'candle_v4_run_lock.json')
        if source.parent.resolve()!=out.resolve() and (source.parent/'v4_threshold_locks').is_dir():
            shutil.copytree(source.parent/'v4_threshold_locks',out/'v4_threshold_locks',dirs_exist_ok=True)
        if source.parent.resolve()!=out.resolve() and (source.parent/'v4_gate_journal').is_dir():
            shutil.copytree(source.parent/'v4_gate_journal',out/'v4_gate_journal',dirs_exist_ok=True)
    args += ['--resume',RESUME_STATE]
print('Pinned source:',actual,flush=True)
print('Full TEST runs only after Task5/Round19 consolidation; retain output with Kaggle Save Version.',flush=True)
subprocess.check_call(args,cwd=runtime,env=env)
import json
completion=json.loads(Path(OUTPUT_DIR,'candle_v1_completion.json' if VARIANT=='native-only' else 'candle_v4_completion.json').read_text())
print('Completed:',completion['completed'],'metrics:',
      {arm:{k:m[k] for k in ('accuracy','macro_f1')} for arm,m in completion['final_test'].items()})
