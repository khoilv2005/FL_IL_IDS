# CANDLE V4: resume Task5 from results (16), completed Task4.
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
code = Path('/kaggle/temp/candle_v4_continuous_resume_source')
code.parent.mkdir(parents=True, exist_ok=True)
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

import hashlib
import json

EXPECTED_CONTINUATION_SHA256 = "1336e1d91f4301bac73b78005c9ecd195bbd3ec36b9d4832e7b3b63967bed075"
EXPECTED_LOCK_SHA256 = "35419e763dcfb410025ccf978716dc39657723f4ba7a4fafe26cd1d632ff99c7"

def digest_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def prepare_task4(results_zip, cache_dir, output_dir,
                  expected_sha=EXPECTED_CONTINUATION_SHA256,
                  expected_lock_sha=EXPECTED_LOCK_SHA256):
    """Extract only the explicit task-boundary continuation, never latest/round state."""
    cache, out = Path(cache_dir), Path(output_dir)
    cache.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'continuation_state_latest.pt').exists() or any(out.glob('checkpoint_task_5*')) or (out/'candle_v4_completion.json').exists():
        raise ValueError('Task5 output already exists. Use a fresh Kaggle session; do not overwrite it.')
    with zipfile.ZipFile(results_zip) as outer:
        if 'checkpoint_archive_manifest.json' in outer.namelist():
            task_zip = Path(results_zip)
        else:
            matches = [i for i in outer.infolist()
                       if i.filename.endswith('/checkpoint_task_4_all_rounds.zip') or i.filename == 'checkpoint_task_4_all_rounds.zip']
            if len(matches) != 1:
                raise ValueError('Expected exactly one Task4 all-rounds archive')
            item = matches[0]
            task_zip = cache/'checkpoint_task_4_all_rounds.zip'
            if not task_zip.exists() or task_zip.stat().st_size != item.file_size:
                temporary = task_zip.with_suffix('.zip.partial')
                with outer.open(item) as src, temporary.open('wb') as dst:
                    shutil.copyfileobj(src, dst, 8*1024*1024)
                temporary.replace(task_zip)
    with zipfile.ZipFile(task_zip) as archive:
        manifest = json.loads(archive.read('checkpoint_archive_manifest.json'))
        member = manifest.get('continuation_checkpoint')
        if manifest.get('task_id') != 4 or manifest.get('completed_rounds') != list(range(20)) or member != 'continuation_state_task_4.pt':
            raise ValueError('Archive is not a complete Task4 boundary, rounds0..19')
        if manifest['checksums'].get(member) != expected_sha:
            raise ValueError('This is a different continuation than results (16)')
        lock_bytes = archive.read('candle_v4_run_lock.json')
        if hashlib.sha256(lock_bytes).hexdigest() != expected_lock_sha or manifest['checksums'].get('candle_v4_run_lock.json') != expected_lock_sha:
            raise ValueError('Original run lock checksum mismatch')
        lock = json.loads(lock_bytes)
        if lock['source_commit'] != CODE_REF or lock['version'] != 'candle_v4_continuous_class_router_v7':
            raise ValueError('Source/protocol differs from the original run')
        if lock['config']['output_dir'] != str(out) or lock['config']['candle_v4_full_service']['router_mode'] != ROUTER_MODE:
            raise ValueError('Keep the original output path and router mode for exact resume')
        target = out/member
        if not target.exists() or digest_file(target) != expected_sha:
            temporary = target.with_suffix('.pt.partial')
            with archive.open(member) as src, temporary.open('wb') as dst:
                shutil.copyfileobj(src, dst, 8*1024*1024)
            if digest_file(temporary) != expected_sha:
                raise ValueError('Continuation checksum failed; training will not start')
            temporary.replace(target)
        lock_path = out/'candle_v4_run_lock.json'
        if lock_path.exists() and lock_path.read_bytes() != lock_bytes:
            raise ValueError('Output contains another run lock')
        lock_path.write_bytes(lock_bytes)
    receipt = dict(source_commit=CODE_REF, resume_from_task=5, completed_task=4,
                   selected_member=member, sha256=expected_sha,
                   policy='Task4 task-boundary only; ignore outer latest and any partial Task5')
    (out/'resume_task5_preflight.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print('Verified Task4 continuation; resume Task5 round0. Original config retained.', flush=True)
    return str(target)


# Source/cache/download stay outside /kaggle/working output exports.
RESULTS_DRIVE_URL = "https://drive.google.com/file/d/12m7ZyoowmKNhKPiSQQ0FH2mzhAweD7tw/view?usp=drive_link"
MOUNTED_RESULTS_ZIP = ""  # Optional /kaggle/input/.../results (16).zip instead of Drive.
cache = Path('/kaggle/temp/candle_v4_resume_task5_results16')
cache.mkdir(parents=True, exist_ok=True)
if MOUNTED_RESULTS_ZIP:
    results_zip = Path(MOUNTED_RESULTS_ZIP)
else:
    results_zip = cache/'results_16.zip'
    if not results_zip.exists():
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'gdown==5.2.0'], env=env)
        import gdown
        partial = results_zip.with_suffix('.zip.download')
        print('Downloading results (16) from Google Drive (~4.70 GB)...', flush=True)
        result = gdown.download(url=RESULTS_DRIVE_URL, output=str(partial), fuzzy=True,
                                quiet=False, resume=True, use_cookies=False)
        if not result or not zipfile.is_zipfile(partial):
            raise RuntimeError('Drive did not return a ZIP. Check sharing/download quota, or mount the ZIP.')
        partial.replace(results_zip)
if not zipfile.is_zipfile(results_zip):
    raise ValueError('Results input is not a ZIP')
RESUME_STATE = prepare_task4(results_zip, cache, OUTPUT_DIR)
# Inspect in a short-lived CPU subprocess: no model training and no retained RAM.
probe = r"""
import json, sys, torch, sklearn
if sklearn.__version__ != '1.6.1': raise RuntimeError('Resume requires the original sklearn1.6.1')
s = torch.load(sys.argv[1], map_location='cpu', weights_only=False, mmap=True)
m = s['meta']
if m.get('completed_task') != 4 or m.get('resume_from_task') != 5 or m.get('boundary') != 'task' or s.get('round_state') is not None:
    raise ValueError('Not a Task4 boundary; refusing earlier tasks or partial Task5')
v = s['candle_v4_shadow_state']
if v['version'] != 'candle_v4_continuous_class_router_v7' or len(v.get('continuous_routers', {})) != len(s['client_model_states']):
    raise ValueError('Missing continuous router state')
print('Resume metadata PASS:', m, 'stored clients:', len(s['client_model_states']), flush=True)
"""
subprocess.check_call([sys.executable, '-c', probe, RESUME_STATE], cwd=runtime, env=env)

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
