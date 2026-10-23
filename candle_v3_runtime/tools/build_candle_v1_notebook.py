"""Generate a small Kaggle notebook that clones the frozen CANDLE V1 launcher."""
import argparse
import json
from pathlib import Path
import subprocess


SOURCE = '''# CANDLE V1: historical DeNICE V1 configuration; edit paths only.
import os
from pathlib import Path
import subprocess
import sys

CODE_REF = "__CODE_REF__"
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
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ref')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    ref = args.ref or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if len(ref) != 40 or any(c not in '0123456789abcdef' for c in ref):
        raise ValueError('Use a full Git commit SHA')
    source = SOURCE.replace('__CODE_REF__', ref)
    markdown = '''# CANDLE V1 — audited DeNICE V1 baseline configuration

Uses the existing DeNICE legacy implementation, with an independent CANDLE V1 run name.
Fresh seed42; 6 tasks x20 rounds; batch2048; Adam lr.001; paper neighborhoods xi=.8.
One local model/router per client; binary_cosine during training; AMP enabled as in V1.
Historical content-disjoint BASE split, not 100% original TRAIN. No replay/APPLIANCE/CME.
Every round is checkpointed/compressed; each task seals one ZIP; 12 GiB output guard.
After final Task5/Round19 and terminal consolidation: full 34-class TEST on disjoint receiver shards.
Report BinarySelf as primary and frozen MulticlassSelf as secondary, without retraining the backbone.
Enable Internet and GPU. Only one GPU is used to preserve the V1 execution path.
Source is cloned from a pinned Git commit; no embedded source bundle/checkpoints/data.
Reuse mounted historical roles by setting ROLES_DIR; otherwise role preparation may take time.
Same configuration does not promise identical numerical results or an untouched confirmation test.
'''
    notebook = dict(nbformat=4, nbformat_minor=5, metadata=dict(
        kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        language_info=dict(name='python', version='3.12')),
        cells=[dict(cell_type='markdown', metadata={}, source=markdown.splitlines(True)),
               dict(cell_type='code', metadata={}, execution_count=None, outputs=[],
                    source=source.splitlines(True))])
    (root / 'train_candle_v1_kaggle.py').write_text(source, encoding='utf-8')
    destination = root / 'train_candle_v1_kaggle.ipynb'
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    if destination.stat().st_size >= 1_000_000:
        raise ValueError('Kaggle notebook exceeds 1 MB source limit')
    print(f'{destination.name}: {destination.stat().st_size:,} bytes; source={ref}')


if __name__ == '__main__':
    main()
