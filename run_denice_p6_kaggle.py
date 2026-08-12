"""Kaggle P6 launcher: full 6-task DeNICE training plus E0--E6 evaluation.

Set ``DENICE_SEED`` to 42, 43, or 44. The job writes all artifacts under
``/kaggle/working/results_denice_seed_<seed>`` and runs the P6 evaluation
matrix only after a successful full task-5/round-19 checkpoint exists.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


SEED = int(os.environ.get("DENICE_SEED", "42"))
DATA_DIR = os.environ.get(
    "DENICE_DATA_DIR", "/kaggle/input/datasets/khoilv2005/100-clients/100-clients"
)
OUTPUT_DIR = Path(os.environ.get("DENICE_OUTPUT_DIR", f"/kaggle/working/results_denice_seed_{SEED}"))
REPO_DIR = Path(os.environ.get("DENICE_REPO_DIR", "/kaggle/working/FL_IL_IDS"))


def main() -> None:
    if not Path(DATA_DIR).is_dir():
        raise FileNotFoundError(f"DENICE_DATA_DIR does not exist: {DATA_DIR}")
    if not REPO_DIR.is_dir():
        raise FileNotFoundError(f"DENICE_REPO_DIR does not exist: {REPO_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "DENICE_SEED": str(SEED),
        "DENICE_TRAIN_PHASE": "5",
        "DENICE_OUTPUT_DIR": str(OUTPUT_DIR),
    }
    subprocess.run([sys.executable, str(REPO_DIR / "train_incremental_kaggle.py")], check=True, env=env)
    checkpoint = OUTPUT_DIR / "checkpoint_task_5_round_19.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Full training did not create final checkpoint: {checkpoint}")
    subprocess.run(
        [
            sys.executable,
            str(REPO_DIR / "run_denice_p6_eval.py"),
            "--checkpoint", str(checkpoint),
            "--data-dir", DATA_DIR,
            "--output-dir", str(OUTPUT_DIR / "p6_evaluation"),
            "--device", "cuda",
            "--seed", str(SEED),
        ],
        check=True,
        env=env,
    )


if __name__ == "__main__":
    main()
