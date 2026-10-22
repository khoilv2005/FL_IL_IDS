"""Build a small, pinned Kaggle notebook for the CANDLE V2 training variant."""
import argparse
import json
from pathlib import Path
import subprocess

SOURCE = '''# CANDLE V2: training-time prototype alignment and injection.
import os
from pathlib import Path
import subprocess
import sys

CODE_REF = "__CODE_REF__"
DATA_DIR = "/kaggle/input/datasets/khoilv2005/100-clients"
OUTPUT_DIR = "/kaggle/working/results_candle_v2_xi_0p8_seed_42"
# Reuse a mounted V1 roles directory if available; otherwise prepare the same split.
ROLES_DIR = "/kaggle/temp/candle_v2_roles"
BATCH_SIZE = 2048
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
    "--batch-size", str(BATCH_SIZE)]
if RESUME_STATE:
    command += ["--resume", RESUME_STATE]
subprocess.check_call(command, cwd=code, env=env)
print(Path(OUTPUT_DIR, "candle_v2_completion.json").read_text(encoding="utf-8"))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ref')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    ref = args.ref or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if len(ref) != 40 or any(c not in '0123456789abcdef' for c in ref):
        raise ValueError('Use a full source commit SHA')
    source = SOURCE.replace('__CODE_REF__', ref)
    markdown = '''# CANDLE V2 — fresh training with peer prototypes

Enable **Internet** and a **GPU** in Kaggle. One GPU is used by this runner.
Fresh seed42, Tasks0–5, 20 rounds/task, xi=0.8, batch2048 by default.
One personalized model/client; decentralized neighborhood aggregation.

Changes from V1: peer-prototype representation alignment, virtual missing-class
CE, current-input teacher preservation, and task-free cosine FC2 inference.
Context-specific adapters are disabled; no oracle/task router at primary inference.
No raw historical replay, CGoFed, CME, APPLIANCE, or donor models at inference.

Uses the same content-disjoint V1 BASE training split and internal local holdout;
**not 100% original TRAIN**. No CAL/FIT/TEST drives prototype fitting.
Round r post-aggregation prototypes train round r+1; old-task banks are discarded.

Checkpoint each round, ZIP/task, 12GiB output guard. Source/roles live in
`/kaggle/temp`; only results are exported from `/kaggle/working`.
Only terminal Task5/Round19 is evaluated, on the full34-class TEST partitioned
across receiver models exactly once, matching V1 sample allocation.
Read `candle_v2_completion.json` for Accuracy and Macro-F1.

This is a new implementation/configuration awaiting training and smoke results.
Accuracy gains and historical safety have not been established.
The notebook clones pinned GitHub source; no embedded source/runtime ZIP.
'''
    notebook = dict(nbformat=4, nbformat_minor=5, metadata=dict(
        kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        language_info=dict(name='python', version='3.12')),
        cells=[dict(cell_type='markdown', metadata={}, source=markdown.splitlines(True)),
               dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=source.splitlines(True))])
    (root/'train_candle_v2_kaggle.py').write_text(source, encoding='utf-8')
    target = root/'train_candle_v2_kaggle.ipynb'
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
    if target.stat().st_size >= 1_000_000:
        raise ValueError('Kaggle source exceeds1MB')
    print(f'{target.name}: {target.stat().st_size} bytes; source={ref}')


if __name__ == '__main__':
    main()
