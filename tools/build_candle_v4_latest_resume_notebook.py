"""Build explicit repaired Task4 -> latest V4-only Task5 continuation."""
import argparse
import json
from pathlib import Path
from build_candle_v4_only_notebook import SOURCE

PREP = r'''
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
'''


def build(ref):
    source = SOURCE.replace('__REF__',ref)
    source = source.replace('# CANDLE V4: fresh full training + V4-only final evaluation.',
                            '# CANDLE V4: explicit repaired Task4 import, train Task5, V4-only final TEST.')
    source = source.replace('results_candle_v4_only_seed42','results_candle_v4_latest_resume_task5_seed42')
    source = source.replace('candle_v4_only_source','candle_v4_latest_resume_source')
    source = source.replace("'/candle_v4_runtime/'", "'/candle_v4_runtime/','/tools/migrate_candle_v4_task4.py','/tools/candle_v4_historical_repair.py'")
    marker = "print('CANDLE V4 only: seed42, T0..T5 x20 rounds, final TEST after T5/R19 consolidation.',flush=True)"
    assert marker in source
    source = source.replace(marker,PREP+"\nprint('Repair saved router memory; train T5/R0..19; final V4-only TEST.',flush=True)")
    source = source.replace("'--continuous-bootstrap','--v4-only']", "'--continuous-bootstrap','--v4-only','--import-task4-archive',str(archive)]")
    compile(source,'notebook','exec')
    return source


def main():
    p=argparse.ArgumentParser();p.add_argument('--ref',required=True);args=p.parse_args()
    if len(args.ref)!=40 or any(c not in '0123456789abcdef' for c in args.ref):raise ValueError('Full published SHA required')
    source=build(args.ref)
    text='''# CANDLE V4 mới nhất — resume Task5 từ Task4

Bật GPU + Internet, gắn dataset **100-clients**, rồi Run All.

- Tự tải **results (16)** từ Drive; không cần results17/18.
- Chỉ lấy checkpoint biên T0–T4, kiểm tra checksum Task4.
- Khôi phục continuous router memory bị thiếu bằng saved TRAIN feature memory.
  Không đọc lại raw TRAIN task cũ; không train lại T0–T4.
- Giữ backbone, expert weights, Adam, masks, native router và RNG.
- Chuyển rõ ràng từ protocol cũ sang bootstrap-fixed V4; xuất migration receipt.
  Đây là continuation có sửa memory, không phải exact resume của code cũ hoặc fresh full run.
- Train **Task5, Round0–19**, consolidation rồi **Full TEST 34 classes, V4-only**.
- Giữ seed42, batch2048, AMP, 100% current-task TRAIN; validation0, threshold0,5.
- Không sửa V1, không chạy baseline V1, không acceptance quality gate.
- TEST giữ disjoint client shards và pooled metrics như run trước.

Nguồn, cache và download nằm ở `/kaggle/temp`; cần khoảng 14 GiB trống cho khâu chuẩn bị.
Output: `/kaggle/working/results_candle_v4_latest_resume_task5_seed42`.
Dùng session mới. Save Version để giữ checkpoint và kết quả.
Nếu Drive hết quota, điền đường dẫn ZIP đã gắn vào `RESULTS16_ZIP`.
'''
    n=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python')),
           cells=[dict(cell_type='markdown',metadata={},source=text.splitlines(True)),dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=source.splitlines(True))])
    root=Path(__file__).resolve().parents[1];stem='resume_candle_v4_latest_task5_kaggle'
    (root/(stem+'.py')).write_text(source,encoding='utf-8')
    (root/(stem+'.ipynb')).write_text(json.dumps(n,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print('Created',stem+'.ipynb')


if __name__=='__main__':main()
