"""Generate a pinned GitHub-cloning Kaggle full-development notebook."""
import argparse
import json
from pathlib import Path

SOURCE='''# CANDLE V4 coverage-aware full development. Edit paths only.
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

CODE_REF = "__REF__"
DATA_DIR = "/kaggle/input/datasets/khoilv2005/100-clients"
ROLES_DIR = "/kaggle/working/candle_v4_roles"
OUTPUT_DIR = "/kaggle/working/results_candle_v4_seed42"
# Optional mounted round continuation .pt or task archive .zip; empty=fresh T0.
# Resume also needs the original run lock/threshold locks in that folder/archive.
RESUME_STATE = ""

if not Path('/kaggle/working').is_dir():
    raise RuntimeError('Run this notebook on Kaggle, not locally')
env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
for name in ('DENICE_CONFIG_FILE','DENICE_CONFIG_OVERRIDES','DENICE_RESUME_STATE_PATH',
             'DENICE_CONFIG_ONLY','DENICE_TASK_END'):
    env.pop(name,None)
code = Path('/kaggle/working/candle_v4_source')
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
        '--data-dir',DATA_DIR,'--roles',ROLES_DIR,'--output',OUTPUT_DIR]
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
print(Path(OUTPUT_DIR,'candle_v4_completion.json').read_text())
'''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--ref',required=True);args=parser.parse_args()
    if len(args.ref)!=40 or any(c not in '0123456789abcdef' for c in args.ref):raise ValueError('Full public SHA required')
    root=Path(__file__).resolve().parents[1];source=SOURCE.replace('__REF__',args.ref)
    markdown='''# CANDLE V4 — native coverage-aware, development seed 42

100 benchmark clients, six tasks ×20 rounds; native V1 batch2048/AMP, one GPU.
Experts use donor positives, receiver/donor negatives and current eligible
auxiliary peers. Includes matched no-auxiliary and donor-only controls plus V1.
ACTIVE only after unchanged risk gates; rejection does not prevent completing training.

Uses the locked V1 BASE/FIT/CAL/validation split. Cumulative VALIDATION after each
task; **full 34-class TEST runs once after Task5/Round19 and consolidation**.
V1 and three V4 arms use identical disjoint receiver shards covering the entire TEST.
This remains a development run, not independent confirmation. TEST never tunes gates.
No historical raw-data replay. Round continuation and task ZIPs contain expert state.
Enable **GPU and Internet**. Source is cloned at a fixed Git SHA; no embedded runtime.
Keep the output with Kaggle Save Version. A stopped interactive session is not an
automatic saved notebook output. Set RESUME_STATE to resume a retained continuation.
'''
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python')),
        cells=[dict(cell_type='markdown',metadata={},source=markdown.splitlines(True)),
               dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=source.splitlines(True))])
    (root/'train_candle_v4_kaggle.py').write_text(source,encoding='utf-8')
    (root/'train_candle_v4_kaggle.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print('Created train_candle_v4_kaggle.ipynb pinned to',args.ref)


if __name__=='__main__':main()
