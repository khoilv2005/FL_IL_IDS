"""Build a self-contained one-cell launcher; saved runs carry their exact source."""
import base64
import hashlib
import re
from pathlib import Path
import zipfile

root=Path(__file__).resolve().parents[1]
out=root/'output/kaggle_hierarchical'
out.mkdir(parents=True,exist_ok=True)
archive=out/'denice_cluster_server_source.zip'
files=[*sorted((root/'fed_learning').rglob('*.py')),
    *[root/p for p in ('tools/train_denice_v2.py','tools/train_cluster_server.py',
        'tools/eval_cluster_servers.py','train_incremental_kaggle.py','eval_selective.py')]]
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in files:
        info=zipfile.ZipInfo(p.relative_to(root).as_posix(),date_time=(2026,1,1,0,0,0))
        info.compress_type=zipfile.ZIP_DEFLATED
        z.writestr(info,p.read_bytes())
payload=base64.b64encode(archive.read_bytes()).decode()
sha=hashlib.sha256(archive.read_bytes()).hexdigest()
header='''"""Paste this ENTIRE file into one Kaggle cell: fresh hierarchical TRAIN, no TEST."""
# Fresh run: leave empty. Resume: link ZIP output from THIS hierarchical version only.
RESUME_DRIVE_URL = ""
DATA_DIR = '/kaggle/input/datasets/khoilv2005/100-clients/100-clients'
OUTPUT_DIR = '/kaggle/working/results_denice_hierarchical_seed_42'
BATCH_SIZE = 2048

import base64, hashlib, os, subprocess, sys, tempfile, zipfile
from pathlib import Path
'''
footer='''
raw=base64.b64decode(SOURCE_ZIP_BASE64)
if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA256:
    raise ValueError('Embedded source corrupted: copy the entire launcher again')
directory=Path(tempfile.mkdtemp(prefix='denice-hierarchical-'))
archive=directory/'source.zip'
archive.write_bytes(raw)
source=directory/'source'
with zipfile.ZipFile(archive) as z:
    z.extractall(source)
env=dict(os.environ, DENICE_CODE_DIR=str(source), DENICE_SOURCE_ZIP=str(archive),
    DENICE_RUN_PROFILE='cluster_server', DENICE_PREDICTION_MODE='routerless_v1',
    DENICE_SERVER_EXCHANGE='1', DENICE_RESUME_DRIVE_URL=RESUME_DRIVE_URL,
    DENICE_DATA_DIR=DATA_DIR, DENICE_OUTPUT_DIR=OUTPUT_DIR, DENICE_BATCH_SIZE=str(BATCH_SIZE))
for name in ('DENICE_CLUSTER_RESUME_ROOT','DENICE_CLUSTER_RESUME_TASK','DENICE_CLUSTER_RESUME_ROUND'):
    env.pop(name,None)
# Fresh interpreter: no notebook argparse flags or cached imported modules.
subprocess.run([sys.executable,'-u',str(source/'train_incremental_kaggle.py')],cwd=source,env=env,check=True)
'''
launcher=out/'train_incremental_kaggle.py'
launcher.write_text(header+'\nSOURCE_SHA256 = '+repr(sha)+'\nSOURCE_ZIP_BASE64 = '+repr(payload)+'\n'+footer,encoding='utf-8')
eval_source=(root/'eval_selective.py').read_text(encoding='utf-8')
eval_source=re.sub(r'^EVAL_DRIVE_URL = .*$', 'EVAL_DRIVE_URL = ""  # Link ZIP output hierarchical da train du Task 0-5.',
    eval_source,flags=re.MULTILINE)
eval_source=re.sub(r'^DIAGNOSTIC = .*$', 'DIAGNOSTIC = False  # Full TEST for the routerless hierarchical run.',
    eval_source,flags=re.MULTILINE)
(out/'eval_selective.py').write_text(eval_source,encoding='utf-8')
print(f'Built {launcher} ({launcher.stat().st_size:,} bytes); source SHA256={sha}')
