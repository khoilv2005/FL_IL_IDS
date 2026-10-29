"""Build one V4-only full train + terminal eval Kaggle notebook."""
import argparse
import json
from pathlib import Path

SOURCE=r'''# CANDLE V4: fresh full training + V4-only final evaluation.
import os
from pathlib import Path
import subprocess
import sys

CODE_REF = "__REF__"
DATA_DIR = "/kaggle/input/datasets/khoilv2005/100-clients"
OUTPUT_DIR = "/kaggle/working/results_candle_v4_only_seed42"

if not Path('/kaggle/working').is_dir():raise RuntimeError('Run on Kaggle with GPU + Internet')
env={**os.environ,'GIT_LFS_SKIP_SMUDGE':'1','PYTHONUNBUFFERED':'1'}
for name in ('DENICE_CONFIG_FILE','DENICE_CONFIG_OVERRIDES','DENICE_RESUME_STATE_PATH','DENICE_CONFIG_ONLY','DENICE_TASK_END'):
    env.pop(name,None)
code=Path('/kaggle/temp/candle_v4_only_source');code.parent.mkdir(parents=True,exist_ok=True)
if not code.exists():
    subprocess.check_call(['git','clone','--filter=blob:none','--no-checkout','https://github.com/khoilv2005/FL_IL_IDS.git',str(code)],env=env)
    subprocess.check_call(['git','sparse-checkout','set','--no-cone','/candle_v4_runtime/'],cwd=code,env=env)
    subprocess.check_call(['git','checkout','--detach',CODE_REF],cwd=code,env=env)
actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()
if actual!=CODE_REF:raise ValueError('Source differs from the locked version; use a fresh Kaggle session')
runtime=code/'candle_v4_runtime'
subprocess.check_call([sys.executable,'-m','pip','install','-q','-r',str(runtime/'requirements-candle-v4-kaggle.txt')],env=env)
print('CANDLE V4 only: seed42, T0..T5 x20 rounds, final TEST after T5/R19 consolidation.',flush=True)
subprocess.check_call([sys.executable,'-u',str(runtime/'tools/train_candle_v4.py'),
    '--data-dir',DATA_DIR,'--output',OUTPUT_DIR,'--variant','coverage',
    '--router-mode','continuous_class_v1','--continuous-bootstrap','--v4-only'],cwd=runtime,env=env)
import json
receipt=json.loads((Path(OUTPUT_DIR)/'candle_v4_completion.json').read_text())
print('Completed:',receipt['completed'],'V4 final metrics:',receipt['final_test']['V4'])
print('Use Kaggle Save Version to keep checkpoints and metrics.')
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--ref',required=True);a=p.parse_args()
    if len(a.ref)!=40 or any(c not in '0123456789abcdef' for c in a.ref):raise ValueError('Full public SHA required')
    text='''# CANDLE V4 — TRAIN + EVAL duy nhất

Bật **GPU + Internet**, gắn dataset **100-clients**, rồi **Run All**.
Đây là **fresh training từ Task0**, không dùng results16/17 hoặc Google Drive.

- 100 clients, seed42, 6 tasks ×20 rounds, batch2048, AMP.
- 100% current-task TRAIN, validation0%; không BASE/FIT/CAL, không raw replay.
- Continuous router + coverage-aware experts + exact-bootstrap memory fix.
- Expert learning/threshold0,5/scoped fusion giữ nguyên; không quality gate.
- Chỉ đánh giá predictor **V4** sau T5/R19 và consolidation.
- Không chạy baseline hoặc ablation readout song song. Native backbone/router
  là thành phần nội bộ của V4, không phải một phương pháp được train/eval riêng.
- Giữ cùng TEST disjoint shards seed523687 và pooled Accuracy/Macro-F1.
- Checkpoint mỗi round, task archive, output `results_candle_v4_only_seed42`.

Save Version để giữ output. Cấu hình và requirements riêng của V4 được khóa
theo commit; cấu hình/budget training không đổi so với bản bootstrap-fixed.
'''
    source=SOURCE.replace('__REF__',a.ref)
    n=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python')),
        cells=[dict(cell_type='markdown',metadata={},source=text.splitlines(True)),dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=source.splitlines(True))])
    root=Path(__file__).resolve().parents[1];stem='train_eval_candle_v4_only_kaggle'
    (root/(stem+'.py')).write_text(source,encoding='utf-8')
    (root/(stem+'.ipynb')).write_text(json.dumps(n,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print('Created',stem+'.ipynb')


if __name__=='__main__':main()
