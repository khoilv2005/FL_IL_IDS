# CANDLE V1: historical DeNICE V1 configuration; edit paths only.
import os
from pathlib import Path
import subprocess
import sys

CODE_REF = "a9714ac96b3ecec3910985d6fda8224b1584b570"
DATA_DIR = "/kaggle/input/datasets/khoilv2005/100-clients"
OUTPUT_DIR = "/kaggle/working/results_candle_v1_xi_0p8_seed_42"
# Point to mounted historical role_lock.json's DIRECTORY to reuse the exact split.
# An empty directory here recreates roles from original TRAIN using split seed20261006.
ROLES_DIR = "/kaggle/working/candle_v1_roles"

env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
code = Path("/kaggle/working/candle_v1_source")
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
    raise ValueError("Source differs from pinned CANDLE V1 revision; use a fresh session")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r",
    str(code / "requirements-candle-v1-kaggle.txt")])
print("CANDLE V1 supporting source:", actual, flush=True)
subprocess.check_call([sys.executable, "-u", str(code / "tools/train_candle_v1.py"),
    "--data-dir", DATA_DIR, "--output", OUTPUT_DIR, "--roles", ROLES_DIR], cwd=code, env=env)
print(Path(OUTPUT_DIR, "candle_v1_completion.json").read_text(encoding="utf-8"))
