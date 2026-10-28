"""Kaggle evaluation only: six frozen task checkpoints, actual Top1/Top2."""
import argparse
import json
from pathlib import Path

SOURCE=r'''# Frozen CANDLE V4 evaluation only. No training, no oracle task labels.
import os
from pathlib import Path
import subprocess
import sys
import zipfile
import json

CODE_REF = "__REF__"
DATA_DIR = "/kaggle/input/datasets/khoilv2005/100-clients"
RESULTS19_ZIP = ""  # Optional mounted ZIP containing T0-T4 archives.
RESULTS19_DRIVE_URL = "https://drive.google.com/file/d/1Qmnb2pHEq7JL55oOTrNm5e963XW09tYp/view?usp=drive_link"
RESULTS21_ZIP = ""  # Mount results (21).zip; auto-detected if only one matching ZIP.
RESULTS21_DRIVE_URL = ""  # Optional Drive URL instead of mounted results21 ZIP.
OUTPUT_DIR = "/kaggle/working/eval_candle_v4_round19_top1_top2"

if not Path('/kaggle/working').is_dir():raise RuntimeError('Run on Kaggle GPU + Internet')
env={**os.environ,'GIT_LFS_SKIP_SMUDGE':'1','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
cache=Path('/kaggle/temp/candle_v4_six_task_topk');cache.mkdir(parents=True,exist_ok=True)

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
        '/tools/eval_candle_v4_task_round19_topk.py','/tools/candle_v4_historical_repair.py'],cwd=code,env=env)
    subprocess.check_call(['git','checkout','--detach',CODE_REF],cwd=code,env=env)
if subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()!=CODE_REF:
    raise ValueError('Cached source differs; use a fresh session')
subprocess.check_call([sys.executable,'-m','pip','install','-q','-r',str(code/'candle_v4_runtime/requirements-candle-v4-kaggle.txt')],env=env)
subprocess.check_call([sys.executable,'-u',str(code/'tools/eval_candle_v4_task_round19_topk.py'),
    '--results19',str(r19),'--results21',str(r21),'--data-dir',DATA_DIR,
    '--cache',str(cache/'continuations'),'--output',OUTPUT_DIR],cwd=code,env=env)
receipt=json.loads((Path(OUTPUT_DIR)/'completion.json').read_text())
print('Complete:',receipt['completed'])
for row in receipt['summary']:
    print('Task',row['task'],row['mode'],'Acc',round(100*row['accuracy'],4),'Macro-F1',round(100*row['macro_f1'],4))
print('Save Version to retain summary.csv and six per-task reports. No training checkpoints changed.')
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--ref',required=True);a=p.parse_args()
    if len(a.ref)!=40 or any(c not in '0123456789abcdef' for c in a.ref):raise ValueError('Full public SHA required')
    source=SOURCE.replace('__REF__',a.ref);compile(source,'notebook','exec')
    text='''# CANDLE V4 — eval Round19 của Task0–5: Top-1 và Top-2 thực tế

**Chỉ eval, không train lại, không sửa V1.** Bật GPU + Internet và gắn dataset100-clients.
Gắn thêm **results (21).zip** vào Kaggle Input. Nếu không tự tìm được ZIP, điền `RESULTS21_ZIP`.
Cũng có thể điền `RESULTS21_DRIVE_URL`. results19 đã có link Drive; notebook tự tải.

- T0–T4 lấy từ results19, T5 lấy từ results21; chỉ trích continuation cuối task.
- Mỗi task dùng checkpoint Round19 **sau consolidation** và cumulative TEST T0 đến task đó.
- Mỗi TEST row được giao một lần qua các client shards; seed42+104729×task.
- **Top-1:** giữ nguyên đường dự đoán V4 hiện tại.
- **Top-2:** mở hợp native classes và experts của hai task router chọn từ input;
  chỉ chọn task thực sự có memory. Giữ adapter theo Top-1, threshold expert0,5.
- Không dùng task/class label thật để chọn task hoặc dự đoán. Đây không phải oracle61,30%.
- Báo Accuracy, Macro-F1 fixed34 classes như native, thêm Macro-F1 trên seen classes,
  Top-1 route accuracy, Top-2 task recall, rescue/break và chi tiết từng client/class.
- T5 Top-1 bắt buộc khớp confusion matrix results21 trên từng client; lệch thì dừng.
- Top-1 route accuracy và Top-2 task recall là hai đại lượng khác nhau.
- Không tune threshold, không refit router, không dùng raw TRAIN lịch sử.

Output: `/kaggle/working/eval_candle_v4_round19_top1_top2` gồm summary.csv,
6 JSON task reports và completion.json. Save Version để giữ kết quả.
Đây là readout development trên các checkpoint hiện có; chưa chứng nhận thuật toán mới.
Chuẩn bị có thể cần khoảng18GiB cache; full cumulative TEST qua6 checkpoints sẽ tốn thời gian.
'''
    n=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python')),
        cells=[dict(cell_type='markdown',metadata={},source=text.splitlines(True)),dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=source.splitlines(True))])
    root=Path(__file__).resolve().parents[1];stem='eval_candle_v4_tasks0_5_round19_top1_top2_kaggle'
    (root/(stem+'.py')).write_text(source,encoding='utf-8')
    (root/(stem+'.ipynb')).write_text(json.dumps(n,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print('Created',stem+'.ipynb')


if __name__=='__main__':main()
