# CANDLE V2: evaluate the saved-router repair, without training.
import os
from pathlib import Path
import subprocess
import sys

CODE_REF = "97e922ccab67f433ba07793cf0f2e23738bab144"
# Add the original results (14) dataset and the original 100-clients dataset.
# Supports either extracted results or an outer results ZIP.
RESULTS_ROOT = "/kaggle/input"
DATA_DIR = "/kaggle/input/datasets/khoilv2005/100-clients"
OUTPUT_DIR = "/kaggle/working/candle_v2_routed_repair_task5"
BATCH_SIZE = 4096
# Only needed if more than one CANDLE V2 run is mounted.
CHECKPOINT = ""
# Empty => use the original candle_v2_roles.zip already in results (14).
ROLES_DIR = ""

env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "PYTHONUNBUFFERED": "1"}
code = Path("/kaggle/temp/candle_v2_repair_source")
code.parent.mkdir(parents=True, exist_ok=True)
if not code.exists():
    subprocess.check_call(["git", "clone", "--filter=blob:none", "--no-checkout",
        "https://github.com/khoilv2005/FL_IL_IDS.git", str(code)], env=env)
    subprocess.check_call(["git", "sparse-checkout", "set", "--no-cone",
        "/fed_learning/", "/appliance/", "/tools/", "/configs/",
        "/eval_checkpoint.py", "/requirements-candle-v1-kaggle.txt"], cwd=code, env=env)
    subprocess.check_call(["git", "checkout", "--detach", CODE_REF], cwd=code, env=env)
actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code, text=True).strip()
if actual != CODE_REF:
    raise ValueError("Use a fresh session: cached source differs from pinned revision")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r",
    str(code / "requirements-candle-v1-kaggle.txt")])
command = [sys.executable, "-u", str(code / "tools/eval_candle_v2_repair.py"),
    "--results-root", RESULTS_ROOT, "--data-dir", DATA_DIR,
    "--output", OUTPUT_DIR, "--batch-size", str(BATCH_SIZE)]
if CHECKPOINT:
    command += ["--checkpoint", CHECKPOINT]
if ROLES_DIR:
    command += ["--roles", ROLES_DIR]
subprocess.check_call(command, cwd=code, env=env)
print(Path(OUTPUT_DIR, "repair_comparison.json").read_text(encoding="utf-8"))
