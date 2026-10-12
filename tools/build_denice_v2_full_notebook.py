"""Small Kaggle launcher pinned to a committed clean repository revision."""
import argparse
import json
from pathlib import Path
import subprocess


SOURCE = '''# DeNICE V2 engineering main: edit paths/resume below; do not change the locked method.
import os
from pathlib import Path
import subprocess
import sys

CODE_REF = "__CODE_REF__"
REPOSITORY = "https://github.com/khoilv2005/FL_IL_IDS.git"
DATA_DIR = "/kaggle/input/datasets/khoilv2005/100-clients"  # Mount root; nested dataset directory is detected below
ROLES_DIR = ""  # Existing denice_clean_roles with role_lock.json and indices; empty -> prepare once
OUTPUT_DIR = "/kaggle/working/results_denice_v2_conv2_xi_0p8_seed_42"
RESUME_ROOT = ""  # Mounted previous output directory containing execution_provenance.json and ZIPs
RESUME_TASK = None
RESUME_ROUND = None  # None restores latest completed round/task-boundary continuation
OUTPUT_BUDGET_GIB = 18.0

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
code = Path("/kaggle/working/denice_v2_source")
if not code.exists():
    subprocess.check_call(["git", "clone", "--no-checkout", REPOSITORY, str(code)])
    subprocess.check_call(["git", "checkout", "--detach", CODE_REF], cwd=code)
actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code, text=True).strip()
if actual != CODE_REF:
    raise ValueError("Existing source checkout differs from pinned revision; use a fresh Kaggle session")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(code/"requirements-denice-v2-kaggle.txt")])
sys.path.insert(0, str(code))
import json
import torch

data_root=Path(DATA_DIR) if DATA_DIR else Path("/kaggle/input")
if (data_root/"metadata.json").is_file() and (data_root/"global_test_data.npz").is_file() and list(data_root.glob("client_*_train.npz")):
    candidates=[data_root]
else:
    candidates=[]
    for metadata in data_root.rglob("metadata.json"):
        parent=metadata.parent
        if (parent/"global_test_data.npz").is_file() and list(parent.glob("client_*_train.npz")):
            candidates.append(parent)
if len(candidates)!=1:
    raise ValueError("Set DATA_DIR explicitly. Found original dataset candidates under " + str(data_root) + ": " + str(candidates))
DATA_DIR=str(candidates[0])
data=Path(DATA_DIR)
if not (data/"global_test_data.npz").is_file():
    raise FileNotFoundError("DATA_DIR must include the entire global_test_data.npz")
if not ROLES_DIR:
    from fed_learning.data.denice_clean_roles import file_sha256
    expected=file_sha256(data/"metadata.json")
    candidates=[]
    for manifest in Path("/kaggle/input").rglob("role_manifest.json"):
        if manifest.parent.name!="denice_clean_roles":
            continue
        value=json.loads(manifest.read_text(encoding="utf-8"))
        if value.get("completed") and value.get("content_disjoint") and value.get("metadata_sha256")==expected:
            candidates.append(manifest.parent)
    if len(candidates)>1:
        raise ValueError("Set ROLES_DIR explicitly; multiple clean role locks found: " + str(candidates))
    if candidates:
        ROLES_DIR=str(candidates[0])
    elif RESUME_TASK is not None:
        raise ValueError("Resume needs the original locked denice_clean_roles; set ROLES_DIR")
    else:
        ROLES_DIR="/kaggle/working/denice_v2_locked_roles/denice_clean_roles"
        from tools.prepare_denice_clean_roles import prepare
        if not (Path(ROLES_DIR)/"role_lock.json").exists():
            print("Preparing content-disjoint roles before fresh training; TEST not used.", flush=True)
            prepare(DATA_DIR,ROLES_DIR,20261006)
        print("Preserve this role directory with the output for any future resume:",ROLES_DIR,flush=True)
if not torch.cuda.is_available():
    raise RuntimeError("Enable a Kaggle GPU before full training")
print("Pinned source:", actual, flush=True)
print("Data:", DATA_DIR, "Roles:", ROLES_DIR, flush=True)
print("DeNICE V2: auto K / xi=.8 / Conv2 coordinate / binary_cosine / seed42 / 6x20.", flush=True)
print("All BASE rows; batch2048; 1 epoch/round; FP32; cuda:0. FULL TEST only after Task 5 / Round 19.", flush=True)
command=[sys.executable,"-u",str(code/"tools/train_denice_v2.py"),"--data-dir",DATA_DIR,
    "--roles-dir",ROLES_DIR,"--output",OUTPUT_DIR,"--device","cuda:0",
    "--output-budget-gib",str(OUTPUT_BUDGET_GIB)]
if RESUME_TASK is not None:
    command += ["--resume-task",str(RESUME_TASK)]
    if RESUME_ROOT:command += ["--resume-root",RESUME_ROOT]
    if RESUME_ROUND is not None:command += ["--resume-round",str(RESUME_ROUND)]
subprocess.check_call(command,cwd=code)
print(Path(OUTPUT_DIR,"completion.json").read_text(encoding="utf-8"),flush=True)
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ref')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    ref=args.ref or subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    if len(ref)!=40 or any(c not in '0123456789abcdef' for c in ref):raise ValueError('Use a full commit SHA')
    source=SOURCE.replace('__CODE_REF__',ref)
    (root/'train_denice_v2_full_kaggle.py').write_text(source,encoding='utf-8')
    markdown='''# DeNICE V2 — Full engineering run

Shared cluster model + binary-cosine router; Conv2 coordinate protection.
Auto K, xi=0.8, fresh seed 42, Task 0–5, 20 rounds/task, uncapped BASE.
Only after Task 5 / Round 19: each cluster runs the SAME full 34-class TEST.
Headline: Mean Cluster Accuracy / Mean Cluster Macro-F1.

Enable GPU + Internet. Set DATA_DIR if more than one original dataset is mounted.
Uses a pinned GitHub checkout; no embedded sources or artifacts.
Round checkpoints are content-addressed; each task seals one verified ZIP.
Low quality does not stop training. Correctness errors still stop.
Preserve locked roles + execution_provenance.json + ZIPs for resume.
Large auto-K means many full-test evaluations; continuation across Kaggle sessions may be necessary.
'''
    notebook=dict(cells=[dict(cell_type='markdown',metadata={},source=markdown.splitlines(True)),
        dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=source.splitlines(True))],
        metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
            language_info=dict(name='python',version='3.12')),nbformat=4,nbformat_minor=5)
    target=root/'train_denice_v2_full_kaggle.ipynb'
    target.write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
    if target.stat().st_size>=1_000_000:raise ValueError('Notebook exceeds Kaggle source limit')
    print(f'{target.name}: {target.stat().st_size:,} bytes; source ref={ref}')


if __name__=='__main__':main()
