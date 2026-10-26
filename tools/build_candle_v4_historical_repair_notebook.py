"""Separate Kaggle notebook: repair saved routing memory, evaluate all clients."""
import argparse
import json
from pathlib import Path


SOURCE=r'''# EVALUATION ONLY. No backbone or expert training.
import os
from pathlib import Path
import subprocess
import sys
import zipfile

CODE_REF = "__REF__"
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
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--ref',required=True);a=p.parse_args()
    if len(a.ref)!=40 or any(c not in '0123456789abcdef' for c in a.ref):raise ValueError('Full public commit SHA required')
    source=SOURCE.replace('__REF__',a.ref)
    markdown='''# CANDLE V4 — sửa memory checkpoint cũ và đánh giá toàn bộ

**Không train lại Task0–5, backbone hoặc expert. Không sửa V1.**

1. Kaggle: bật **GPU + Internet**, gắn dataset **100-clients**.
2. Điền **RESULTS17_DRIVE_URL** khi upload results17 xong (hoặc gắn ZIP results17 vào Input).
3. Link results16 đã điền sẵn. Run All, cuối cùng Save Version để giữ output.

Notebook lấy historical continuationsT0–T4 từ results16 và finalT5 từ results17.
Phục dựng inherited memory theo exact bootstrap lineage; kiểm tra sketches,
thresholds, feature indices và task profiles. Giữ nguyên các router không thiếu
memory; chỉ fit lại LR theo công thức cũ trên TRAIN features đã lưu cho các bank
được phục dựng. Không mở raw TRAIN và không dùng TEST chọn ngưỡng/fit.

Sau đó chạy **100 frozen client predictors** trên cùng original disjoint TEST
shards, seed523687, đủ34 classes; mỗi TEST row được tính một lần. Accuracy và
Macro-F1 tính pooled từ confusion matrices, đúng evaluator cũ. Threshold0,5 giữ
nguyên. Báo cả native control, repaired-native, repaired-V4 và so kết quả V4 cũ.

Tất cả50 client thiếu memory được xử lý hoặc báo lỗi structural, không chọn
những client có kết quả đẹp. Đây là development counterfactual sau audit TEST,
không phải fresh V8 training hoặc independent benchmark. sklearn giữ1.6.1.

Output riêng: `results_candle_v4_historical_repair_seed42`, gồm final TEST JSON,
repair_completion.json, provenance và router repair bundle. Bundle **không phải
training continuation**. Results16/17 và notebook V1/V4 cũ giữ nguyên.

Download/cache dùng `/kaggle/temp`, chỉ trích task continuations; không giải nén
hàng trăm full round models. Cần khoảng14GiB temporary disk cho hai ZIP và cache.
'''
    n=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python')),
        cells=[dict(cell_type='markdown',metadata={},source=markdown.splitlines(True)),
               dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=source.splitlines(True))])
    root=Path(__file__).resolve().parents[1];stem='eval_candle_v4_historical_repair_kaggle'
    (root/(stem+'.py')).write_text(source,encoding='utf-8')
    (root/(stem+'.ipynb')).write_text(json.dumps(n,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print('Created',stem+'.ipynb')


if __name__=='__main__':main()
