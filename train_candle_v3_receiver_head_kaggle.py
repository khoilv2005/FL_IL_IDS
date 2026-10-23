# CANDLE V3 receiver-encoder head learning; full engineering baseline.
import os
from pathlib import Path
import subprocess
import sys

CODE_REF = "e0565dd4c1a18c11d84007d0b60cb2bc9110dcee"
DATA_DIR = "/kaggle/input/datasets/khoilv2005/100-clients"
OUTPUT_DIR = "/kaggle/working/results_candle_v3_receiver_head_xi_0p8_seed_42"
# Point to mounted historical role_lock.json's DIRECTORY to reuse the exact split.
# An empty directory here recreates roles from original TRAIN using split seed20261006.
ROLES_DIR = "/kaggle/working/candle_v3_roles"

env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
code = Path("/kaggle/temp/candle_v3_receiver_head_source")
code.parent.mkdir(parents=True, exist_ok=True)
if not code.exists():
    subprocess.check_call(["git", "clone", "--filter=blob:none", "--no-checkout",
        "https://github.com/khoilv2005/FL_IL_IDS.git", str(code)], env=env)
    subprocess.check_call(["git", "sparse-checkout", "set", "--no-cone",
        "/candle_v3_runtime/", "/tools/", "/configs/",
        "/requirements-candle-v3-kaggle.txt"], cwd=code, env=env)
    subprocess.check_call(["git", "checkout", "--detach", CODE_REF], cwd=code, env=env)
actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code, text=True).strip()
if actual != CODE_REF:
    raise ValueError("Source differs from pinned CANDLE V3 revision; use a fresh session")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r",
    str(code / "requirements-candle-v3-kaggle.txt")])
print("CANDLE V3 supporting source:", actual, flush=True)
subprocess.check_call([sys.executable, "-u", str(code / "tools/train_candle_v3.py"),
    "--peer-teaching", "--data-dir", DATA_DIR, "--output", OUTPUT_DIR, "--roles", ROLES_DIR], cwd=code, env=env)
print(Path(OUTPUT_DIR, "candle_v3_completion.json").read_text(encoding="utf-8"))
