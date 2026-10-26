"""Build exact V4-only round continuation from results19; no runtime changes."""
import json
from pathlib import Path

PREP=r'''
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
'''


def build(root):
    source=(root/'train_eval_candle_v4_only_kaggle.py').read_text(encoding='utf-8')
    assert '3731fe1cee41c693fbfb7327b7fc3b2e86a0f3ab' in source
    source=source.replace('# CANDLE V4: fresh full training + V4-only final evaluation.',
                          '# CANDLE V4: exact results19 continuation, Task5 Round5..19 + final V4-only TEST.')
    source=source.replace('candle_v4_only_source','candle_v4_results19_resume_source')
    marker="print('CANDLE V4 only: seed42, T0..T5 x20 rounds, final TEST after T5/R19 consolidation.',flush=True)"
    assert marker in source
    source=source.replace(marker,PREP)
    source=source.replace("'--continuous-bootstrap','--v4-only']", "'--continuous-bootstrap','--v4-only','--resume',str(RESUME_STATE)]")
    compile(source,'notebook','exec')
    return source


def main():
    root=Path(__file__).resolve().parents[1];source=build(root)
    markdown='''# CANDLE V4 — resume results19 tại Task5 / Round5

Bật **GPU + Internet**, gắn dataset **100-clients**, rồi **Run All**.

- Tự tải results19 từ Drive (~4,54 GB); có thể điền `RESULTS19_ZIP` nếu đã gắn ZIP vào Input.
- Khóa đúng source của phiên results19: `3731fe1cee41c693fbfb7327b7fc3b2e86a0f3ab`.
- Kiểm tra SHA256 continuation/run lock và metadata Task5/R4 đã hoàn tất.
- Restore backbone, experts/Adam, masks, router, RNG và quota/communication.
- Chạy tiếp **Task5 Round5–19**, consolidation rồi **Full TEST chỉ V4**.
- Không migration, không train lại T0–T4, không sửa V1 hoặc đổi training protocol.
- Giữ batch2048, seed42, 100% current-task TRAIN, validation0, threshold0,5.
- Khôi phục các ZIP base/round0–4 để tiếp tục chuỗi delta checkpoint Task5.
  Archive T0–T4 giữ trong ZIP results19 gốc, không sao chép sang output mới.

Output giữ đúng `/kaggle/working/results_candle_v4_only_seed42` để khớp run lock.
Dùng session mới. **Save Version** để giữ checkpoint và kết quả cuối.
Không dùng notebook resume từ results16 cho phiên này.
'''
    n=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python')),
           cells=[dict(cell_type='markdown',metadata={},source=markdown.splitlines(True)),dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=source.splitlines(True))])
    stem='resume_candle_v4_results19_task5_round5_kaggle'
    (root/(stem+'.py')).write_text(source,encoding='utf-8')
    (root/(stem+'.ipynb')).write_text(json.dumps(n,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print('Created',stem+'.ipynb')


if __name__=='__main__':main()
