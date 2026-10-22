# CANDLE V2: training-time prototype alignment and injection.
import os
from pathlib import Path
import subprocess
import sys

CODE_REF = "97e922ccab67f433ba07793cf0f2e23738bab144"
DATA_DIR = "/kaggle/input/datasets/khoilv2005/100-clients"
OUTPUT_DIR = "/kaggle/working/results_candle_v2_xi_0p8_seed_42"
# Reuse a mounted V1 roles directory if available; otherwise prepare the same split.
ROLES_DIR = "/kaggle/temp/candle_v2_roles"
BATCH_SIZE = 2048
INFERENCE_POLICY = "binary_cosine_registered_v1"
# Empty => fresh Task0..5. For resume, mount a CANDLE V2 continuation/task ZIP.
RESUME_STATE = ""

env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "PYTHONUNBUFFERED": "1"}
code = Path("/kaggle/temp/candle_v2_source")
code.parent.mkdir(parents=True, exist_ok=True)
if not code.exists():
    subprocess.check_call(["git", "clone", "--filter=blob:none", "--no-checkout",
        "https://github.com/khoilv2005/FL_IL_IDS.git", str(code)], env=env)
    subprocess.check_call(["git", "sparse-checkout", "set", "--no-cone",
        "/fed_learning/", "/appliance/", "/tools/", "/configs/",
        "/train_incremental_kaggle.py", "/eval_checkpoint.py",
        "/requirements-candle-v1-kaggle.txt"], cwd=code, env=env)
    subprocess.check_call(["git", "checkout", "--detach", CODE_REF], cwd=code, env=env)
actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code, text=True).strip()
if actual != CODE_REF:
    raise ValueError("CANDLE V2 source differs from pinned revision; use a fresh session")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r",
    str(code / "requirements-candle-v1-kaggle.txt")])
print("CANDLE V2 source:", actual, flush=True)
command = [sys.executable, "-u", str(code / "tools/train_candle_v2.py"),
    "--data-dir", DATA_DIR, "--output", OUTPUT_DIR, "--roles", ROLES_DIR,
    "--batch-size", str(BATCH_SIZE), "--inference-policy", INFERENCE_POLICY]
if RESUME_STATE:
    command += ["--resume", RESUME_STATE]
subprocess.check_call(command, cwd=code, env=env)
print(Path(OUTPUT_DIR, "candle_v2_completion.json").read_text(encoding="utf-8"))
