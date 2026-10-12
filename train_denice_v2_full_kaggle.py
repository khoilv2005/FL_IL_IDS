# DeNICE V2 engineering main: edit paths/resume below; do not change the locked method.
import os
from pathlib import Path
import subprocess
import sys

CODE_REF = "3c02c38a6529c28103213b9caac5903c5d808466"
REPOSITORY = "https://github.com/khoilv2005/FL_IL_IDS.git"
DATA_DIR = ""   # Directory with metadata.json, client_*_train.npz, global_test_data.npz
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

if not DATA_DIR:
    candidates=[]
    for metadata in Path("/kaggle/input").rglob("metadata.json"):
        parent=metadata.parent
        if (parent/"global_test_data.npz").is_file() and list(parent.glob("client_*_train.npz")):
            candidates.append(parent)
    if len(candidates)!=1:
        raise ValueError("Set DATA_DIR explicitly. Found original dataset candidates: " + str(candidates))
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
print("All BASE rows; batch2048; 1 epoch/round; FP32; cuda:0. Evaluation at each task end.", flush=True)
command=[sys.executable,"-u",str(code/"tools/train_denice_v2.py"),"--data-dir",DATA_DIR,
    "--roles-dir",ROLES_DIR,"--output",OUTPUT_DIR,"--device","cuda:0",
    "--output-budget-gib",str(OUTPUT_BUDGET_GIB)]
if RESUME_TASK is not None:
    command += ["--resume-task",str(RESUME_TASK)]
    if RESUME_ROOT:command += ["--resume-root",RESUME_ROOT]
    if RESUME_ROUND is not None:command += ["--resume-round",str(RESUME_ROUND)]
subprocess.check_call(command,cwd=code)
print(Path(OUTPUT_DIR,"completion.json").read_text(encoding="utf-8"),flush=True)
