"""Generate the frozen remaining-test panel notebook, with no fit calls."""
import json
from pathlib import Path
from build_denice_tip_notebook import cell
ROOT=Path(__file__).resolve().parents[1]


def main():
    cells=[cell('markdown','''# DeNICE frozen remaining-test panel

Attach the original **100-clients** dataset, enable **Internet** and a GPU.
Use a fresh kernel/session. This is inference on new inputs; it queries
**self + 16 frozen experts** per sample. It does not train or select any policy.

Fixed: checkpoint 03b9b53, candidate seed 42, Gate V2 MLP_top1,
StandardScaler + ClassLR_C0.1, original feature schema and action/tie policy.
All 98 fitted donor routers are shipped in the repo; no router refitting occurs.

Evaluate 50k unique remaining-test inputs with panel seed 20261006, excluding
all old panel row IDs/content and all V2 gate role content hashes. Some rare
classes were exhausted by the old panel. Export the precise missing-class scope;
this is **not a new full-34-class result directly comparable to 50.732%**.
The protocol is locked before reading dataset labels. Stratification uses labels
as declared; final prediction functions do not receive them.

Output: `denice_frozen_panel_diagnostics.zip`. No fitting or test-based selection.
'''),cell('code','''import os, sys, subprocess, importlib.metadata, json, zipfile, shutil
from pathlib import Path
WORK = Path('/kaggle/working')
SOURCE = WORK / 'denice_frozen_panel_source'
OUT = WORK / 'denice_frozen_remaining_panel_03b9b53'
DATA_DIR = ''  # Original dataset auto-discovery below.
BATCH_SIZE = 512
RESULTS_URL = 'https://drive.google.com/file/d/1BEjP4iGJbPT0uFcHZ7WXX4oM_1vyvx0M/view?usp=sharing'
GATE_URL = 'https://drive.google.com/file/d/1gweD4iZ4_NYmEsQyGDOlInTITwcxHrnP/view?usp=sharing'
META_URL = 'https://drive.google.com/file/d/1YA8ixsHaf0_oHFMvE0R3VNIBE6CzPZdM/view?usp=drive_link'
if OUT.exists() and any(OUT.iterdir()):
    raise RuntimeError('Use a fresh Kaggle session/output directory to avoid mixing runs.')
if 'sklearn' in sys.modules and sys.modules['sklearn'].__version__ != '1.6.1':
    raise RuntimeError('Restart kernel and Run All before importing sklearn; frozen estimators require 1.6.1.')
try:
    version = importlib.metadata.version('scikit-learn')
except importlib.metadata.PackageNotFoundError:
    version = None
if version != '1.6.1':
    subprocess.run([sys.executable,'-m','pip','install','--quiet','scikit-learn==1.6.1'],check=True)
subprocess.run([sys.executable,'-m','pip','install','--quiet','gdown','threadpoolctl'],check=True)
clone_env = dict(os.environ,GIT_LFS_SKIP_SMUDGE='1')
if not SOURCE.exists():
    subprocess.run(['git','clone','--depth','1','https://github.com/khoilv2005/FL_IL_IDS.git',str(SOURCE)],check=True,env=clone_env)
else:
    subprocess.run(['git','-C',str(SOURCE),'pull','--ff-only'],check=True,env=clone_env)
sys.path.insert(0,str(SOURCE))
evaluation_commit = subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
print('Evaluation source:',evaluation_commit)
'''),cell('code','''import gdown
def download_archive(url, name, environment_key):
    explicit = os.environ.get(environment_key)
    target = Path(explicit) if explicit else WORK / name
    if not target.exists():
        if explicit:
            raise FileNotFoundError(target)
        partial = target.with_suffix('.download')
        result = gdown.download(url=url,output=str(partial),fuzzy=True,quiet=False)
        if not result or not zipfile.is_zipfile(partial):
            raise RuntimeError(f'Drive download did not produce a ZIP: {name}')
        partial.replace(target)
    if not zipfile.is_zipfile(target):
        raise ValueError(f'Invalid ZIP: {target}')
    return target
GATE_ZIP = download_archive(GATE_URL,'denice_competence_gate_v2_diagnostics.zip','DENICE_GATE_V2_ZIP')
META_ZIP = download_archive(META_URL,'denice_class_meta_diagnostics.zip','DENICE_CLASS_META_ZIP')
RESULTS_ZIP = download_archive(RESULTS_URL,'results_03b9b53_full.zip','DENICE_RESULTS4_ZIP')
ROUTERS = SOURCE / 'artifacts' / 'denice_multiclass_routers_03b9b53.zip'
if not ROUTERS.exists():
    raise FileNotFoundError('Frozen donor routers missing from cloned source.')
if not DATA_DIR or not Path(DATA_DIR,'global_test_data.npz').exists():
    candidates = sorted({p.parent for p in Path('/kaggle/input').rglob('metadata.json')
                         if (p.parent / 'global_test_data.npz').exists()})
    historical = Path('/kaggle/input/datasets/khoilv2005/100-clients/100-clients')
    if historical in candidates:
        DATA_DIR = str(historical)
    elif len(candidates) == 1:
        DATA_DIR = str(candidates[0])
    else:
        raise FileNotFoundError(f'Attach original 100-clients dataset or set DATA_DIR. Candidates: {candidates}')
print('Dataset:',DATA_DIR)

# Keep checkpoint files outside OUT so result ZIP contains metrics/features only.
CHECKPOINT_ROOT = WORK / 'denice_frozen_checkpoint_03b9b53'
CHECKPOINT_ROOT.mkdir(exist_ok=True)
checkpoint_name = 'checkpoint_task_5_round_19.pt'
with zipfile.ZipFile(RESULTS_ZIP) as archive:
    matches = [n for n in archive.namelist() if Path(n).name == checkpoint_name]
    if len(matches) != 1:
        raise ValueError(f'Expected one original checkpoint; found {len(matches)}')
    prefix = matches[0][:-len(checkpoint_name)]
    for name in archive.namelist():
        if not name.startswith(prefix) or name.endswith('/'):
            continue
        base = Path(name).name
        if base == 'config.json' or base == 'checkpoint_task_5_base.pt' or base.startswith('checkpoint_task_5_round_'):
            with archive.open(name) as src, (CHECKPOINT_ROOT/base).open('wb') as dst:
                shutil.copyfileobj(src,dst)
from fed_learning.training.denice_delta_checkpoint import load_denice_checkpoint
from tools.eval_denice_frozen_panel import file_hash
checkpoint_path = CHECKPOINT_ROOT / checkpoint_name
checkpoint_sha256 = file_hash(checkpoint_path)
ckpt = load_denice_checkpoint(str(checkpoint_path))
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Frozen checkpoint:',checkpoint_sha256,'device:',device)
'''),cell('code','''from threadpoolctl import threadpool_limits
from tools.eval_denice_frozen_panel import run_frozen_panel
try:
    with threadpool_limits(limits=1):
        summary = run_frozen_panel(ckpt,DATA_DIR,GATE_ZIP,META_ZIP,ROUTERS,OUT,
            checkpoint_sha256,evaluation_commit=evaluation_commit,device=device,batch_size=BATCH_SIZE)
finally:
    if OUT.exists():
        path = shutil.make_archive(str(WORK/'denice_frozen_panel_diagnostics'),'zip',root_dir=OUT)
        print('Output ZIP (check frozen_panel_completion.json):',path)
display(summary)
print('Panel scope:',json.loads((OUT/'panel_scope.json').read_text()))
''')]
    for index,c in enumerate(cells):c['id']=f'denice-frozen-panel-{index}'
    notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python',version='3.11')),nbformat=4,nbformat_minor=5)
    path=ROOT/'eval_denice_frozen_panel_kaggle.ipynb'
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8');print(path)


if __name__=='__main__':main()
