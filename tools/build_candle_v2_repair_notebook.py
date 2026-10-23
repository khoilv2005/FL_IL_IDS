"""Publish a small pinned Kaggle notebook to re-evaluate existing CANDLE V2 weights."""
import argparse
import json
from pathlib import Path
import subprocess

SOURCE='''# CANDLE V2: evaluate the saved-router repair, without training.
import os
from pathlib import Path
import subprocess
import sys
import zipfile

CODE_REF = "__CODE_REF__"
# Results download runs inside Kaggle. Only mount the original 100-clients dataset.
RESULTS_DRIVE_URL = "https://drive.google.com/file/d/1HTPtASQr4Nh-lwy-RwJCk9iogWBKtpIQ/view?usp=sharing"
# To use mounted results instead, clear RESULTS_DRIVE_URL and set RESULTS_ROOT.
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
if RESULTS_DRIVE_URL and not CHECKPOINT:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown==5.2.0"])
    import gdown
    archive = Path("/kaggle/temp/candle_v2_results14.zip")
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        partial = archive.with_suffix(".download")
        print("Downloading results (14) from Google Drive...", flush=True)
        downloaded = gdown.download(url=RESULTS_DRIVE_URL, output=str(partial),
            fuzzy=True, quiet=False, resume=True, use_cookies=False)
        if not downloaded or not zipfile.is_zipfile(partial):
            raise RuntimeError("Drive download did not produce a ZIP; check file sharing or download quota")
        partial.replace(archive)
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"Invalid cached results ZIP: {archive}")
    RESULTS_ROOT = str(archive)
    print("Results ZIP ready:", archive, flush=True)
command = [sys.executable, "-u", str(code / "tools/eval_candle_v2_repair.py"),
    "--results-root", RESULTS_ROOT, "--data-dir", DATA_DIR,
    "--output", OUTPUT_DIR, "--batch-size", str(BATCH_SIZE)]
if CHECKPOINT:
    command += ["--checkpoint", CHECKPOINT]
if ROLES_DIR:
    command += ["--roles", ROLES_DIR]
subprocess.check_call(command, cwd=code, env=env)
print(Path(OUTPUT_DIR, "repair_comparison.json").read_text(encoding="utf-8"))
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ref')
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    ref=a.ref or subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    if len(ref)!=40 or any(c not in '0123456789abcdef' for c in ref):raise ValueError('Full source commit SHA required')
    source=SOURCE.replace('__CODE_REF__',ref)
    markdown='''# CANDLE V2 — saved-router inference repair (evaluation only)

Enable **GPU** and **Internet**. Mount the original **100-clients** dataset.

The notebook downloads **results (14)** directly from the configured Google Drive
link inside Kaggle, then prepares task5 checkpoint and the original roles.
Download/cache/source stay under `/kaggle/temp`, outside the exported output.
To use mounted results, clear `RESULTS_DRIVE_URL` and set `RESULTS_ROOT`.

**No training. No extra runtime ZIP.** The notebook clones pinned GitHub code.
Only task5 checkpoint and original roles are prepared under `/kaggle/temp`.

Policy: saved `binary_cosine` router predicts task from input, then permits only
classes in the saved CANDLE `available + class_tasks` registry for that task.
One local model / one backbone forward per sample; no oracle or donor inference.
No changes to weights, masks, thresholds, or saved activation memory.

Evaluate the full34-class TEST with the original disjoint receiver allocation:
each global TEST row is evaluated once. The same forward also computes the old
task-free baseline, paired rescue/break, Accuracy, pooled Macro-F1 and route accuracy.
The original full TEST confusion is checked when present in mounted results.

Read `repair_comparison.json` and `completion.json`. Results are also exported as
`candle_v2_routed_repair_task5.zip`, without checkpoint duplicates.

Development showed improvement but also breaks and lower new-class recall.
This notebook measures full TEST performance after a development-selected repair;
it is not untouched scientific confirmation or a claim that CANDLE V2 beats V1.
'''
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(
        kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3.12')),
        cells=[dict(cell_type='markdown',metadata={},source=markdown.splitlines(True)),
               dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=source.splitlines(True))])
    (root/'eval_candle_v2_repair_kaggle.py').write_text(source,encoding='utf-8')
    path=root/'eval_candle_v2_repair_kaggle.ipynb'
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    if path.stat().st_size>=1_000_000:raise ValueError('Kaggle source exceeds 1MB')
    print(f'{path.name}: {path.stat().st_size} bytes; source={ref}')


if __name__=='__main__':main()
