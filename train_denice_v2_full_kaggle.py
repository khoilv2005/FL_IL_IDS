# DeNICE V2 engineering main: edit paths/resume below; do not change the locked method.
import os
from pathlib import Path
import subprocess
import sys

CODE_REF = "3e79c746fd9e648ae0ba1ad5a3d580ace166b52a"
REPOSITORY = "https://github.com/khoilv2005/FL_IL_IDS.git"
DATA_DIR = "/kaggle/input/datasets/khoilv2005/100-clients"  # Mount root; nested dataset directory is detected below
OUTPUT_DIR = "/kaggle/working/results_denice_v2_conv2_xi_0p8_seed_42_2gpu_bs4086"
BATCH_SIZE = 4086  # Per-client mini-batch, not a global DDP batch
WORKER_DEVICES = "cuda:0,cuda:1"
RESUME_ROOT = ""  # Mounted previous output directory containing execution_provenance.json and ZIPs
RESUME_TASK = None
RESUME_ROUND = None  # None restores latest completed round/task-boundary continuation
OUTPUT_BUDGET_GIB = 18.0

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("GIT_LFS_SKIP_SMUDGE", "1")
code = Path("/kaggle/working/denice_v2_source")
if not code.exists():
    subprocess.check_call(["git","clone","--filter=blob:none","--no-checkout",REPOSITORY,str(code)])
    subprocess.check_call(["git","sparse-checkout","set","--no-cone",
        "/fed_learning/","/tools/","/requirements-denice-v2-kaggle.txt"],cwd=code)
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
if not torch.cuda.is_available() or torch.cuda.device_count()<2:
    raise RuntimeError("Enable Kaggle T4 x2: this notebook requires two visible GPUs")
print("Pinned source:", actual, flush=True)
print("Data:",DATA_DIR,"; using 100% original client TRAIN, current task only.",flush=True)
print("DeNICE V2: auto K / xi=.8 / Conv2 coordinate / binary_cosine / seed42 / 6x20.", flush=True)
print(f"All TRAIN rows; batch/client={BATCH_SIZE}; 1 epoch/round; FP32; parallel workers cuda:0 + cuda:1.",flush=True)
print("FULL TEST only after Task 5 / Round 19. Output:",OUTPUT_DIR,flush=True)
command=[sys.executable,"-u",str(code/"tools/train_denice_v2.py"),"--data-dir",DATA_DIR,
    "--output",OUTPUT_DIR,"--device","cuda:0",
    "--batch-size",str(BATCH_SIZE),"--worker-devices",WORKER_DEVICES,
    "--output-budget-gib",str(OUTPUT_BUDGET_GIB)]
if RESUME_TASK is not None:
    command += ["--resume-task",str(RESUME_TASK)]
    if RESUME_ROOT:command += ["--resume-root",RESUME_ROOT]
    if RESUME_ROUND is not None:command += ["--resume-round",str(RESUME_ROUND)]
subprocess.check_call(command,cwd=code)
print(Path(OUTPUT_DIR,"completion.json").read_text(encoding="utf-8"),flush=True)
