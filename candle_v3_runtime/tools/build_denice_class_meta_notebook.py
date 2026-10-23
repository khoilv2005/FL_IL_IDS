"""Build a CPU-only cached Gate V2 class-meta experiment notebook."""
import json
from pathlib import Path
from build_denice_tip_notebook import cell

ROOT=Path(__file__).resolve().parents[1]


def main():
    cells=[cell('markdown','''# Frozen DeNICE class-level meta-ensemble

Run on Kaggle with **Internet enabled**. CPU is sufficient. No dataset attachment,
checkpoint download, backbone training or new expert inference is required.
The configured Google Drive ZIP is the completed Gate V2 diagnostic archive.

The original checkpoint remains **03b9b53**, candidate seed 42 and budgets 4/8/16.
Fit class-level LR/MLP on cached fitting rows; select a policy on all validation
rows; save and restore the frozen artifact before accessing test caches.
The final class must have been predicted by an actually queried expert.
Actual-routed Oracle is therefore the same upper bound as in V2.

This is a shared retrospective diagnostic on the existing development panel,
not an untouched final test or a streaming/private deployment implementation.
Primary V2 accuracy remains 46.388% until this experiment has finished.
'''),cell('code','''import os, sys, subprocess, importlib.metadata
from pathlib import Path

DRIVE_URL = 'https://drive.google.com/file/d/1gweD4iZ4_NYmEsQyGDOlInTITwcxHrnP/view?usp=sharing'
WORK = Path('/kaggle/working')
SOURCE = WORK / 'denice_class_meta_source'
ARCHIVE = Path(os.environ.get('DENICE_GATE_V2_ZIP', str(WORK / 'denice_competence_gate_v2_diagnostics.zip')))
OUT = WORK / 'denice_class_meta_03b9b53'

# Run this cell in a fresh kernel before any sklearn import.
if 'sklearn' in sys.modules and sys.modules['sklearn'].__version__ != '1.6.1':
    raise RuntimeError('Restart the kernel, then Run All: archived V2 estimators require sklearn 1.6.1.')
try:
    current_version = importlib.metadata.version('scikit-learn')
except importlib.metadata.PackageNotFoundError:
    current_version = None
if current_version != '1.6.1':
    subprocess.run([sys.executable,'-m','pip','install','--quiet','scikit-learn==1.6.1'],check=True)
subprocess.run([sys.executable,'-m','pip','install','--quiet','gdown','pandas','matplotlib','threadpoolctl'],check=True)

clone_env = dict(os.environ, GIT_LFS_SKIP_SMUDGE='1')
if not SOURCE.exists():
    subprocess.run(['git','clone','--depth','1','https://github.com/khoilv2005/FL_IL_IDS.git',str(SOURCE)],env=clone_env,check=True)
else:
    subprocess.run(['git','-C',str(SOURCE),'pull','--ff-only'],env=clone_env,check=True)
sys.path.insert(0,str(SOURCE))
print('Evaluation source:',subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip())
'''),cell('code','''import zipfile, gdown
if not ARCHIVE.exists():
    if 'DENICE_GATE_V2_ZIP' in os.environ:
        raise FileNotFoundError(f'Configured archive missing: {ARCHIVE}')
    partial = ARCHIVE.with_suffix('.download')
    downloaded = gdown.download(url=DRIVE_URL, output=str(partial), fuzzy=True, quiet=False)
    if not downloaded or not zipfile.is_zipfile(partial):
        raise RuntimeError('Drive download did not produce a ZIP. Check sharing permissions and download quota.')
    partial.replace(ARCHIVE)
if not zipfile.is_zipfile(ARCHIVE):
    raise ValueError(f'Invalid Gate V2 ZIP: {ARCHIVE}')
print('Cached Gate V2 archive:', ARCHIVE)
'''),cell('code','''import shutil
from threadpoolctl import threadpool_limits
from tools.eval_denice_class_meta import run_class_meta

if OUT.exists() and any(OUT.iterdir()):
    raise RuntimeError('Output directory contains an earlier run. Choose a new OUT or restart a fresh Kaggle session.')
try:
    with threadpool_limits(limits=1):
        summary = run_class_meta(ARCHIVE, OUT, budgets=(4,8,16))
finally:
    if OUT.exists():
        output_zip = shutil.make_archive(str(WORK / 'denice_class_meta_diagnostics'),'zip',root_dir=OUT)
        print('Output archive (check class_meta_completion.json):', output_zip)

display(summary[summary.policy.isin(['majority','GateV2','GateAllVote','ValidationSelected'])])
from IPython.display import Image, display
display(Image(filename=str(OUT / 'class_meta_vs_budget.png')))
''')]
    for index,item in enumerate(cells):item['id']=f'denice-class-meta-{index}'
    notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python',version='3.11')),nbformat=4,nbformat_minor=5)
    path=ROOT/'eval_denice_class_meta_kaggle.ipynb'
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(path)


if __name__=='__main__':main()
