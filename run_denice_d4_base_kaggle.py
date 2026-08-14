"""Build the single audited task-4 continuation required by D4 branches."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


SEED = int(os.environ.get("DENICE_SEED", "42"))
DATA_DIR = os.environ.get("DENICE_DATA_DIR", "/kaggle/input/datasets/khoilv2005/100-clients/100-clients")
REPO_DIR = Path(os.environ.get("DENICE_REPO_DIR", "/kaggle/working/FL_IL_IDS"))
OUTPUT_ROOT = Path(os.environ.get("D4_BASE_OUTPUT_ROOT", f"/kaggle/working/denice_d4_base_seed_{SEED}"))
MAX_CLIENTS = int(os.environ.get("D4_MAX_CLIENTS", "20"))
ROUNDS = int(os.environ.get("D4_ROUNDS_PER_TASK", "5"))


def main() -> None:
    if not REPO_DIR.is_dir() or not Path(DATA_DIR).is_dir():
        raise FileNotFoundError("D4_BASE needs existing DENICE_REPO_DIR and DENICE_DATA_DIR.")
    continuation = OUTPUT_ROOT / "continuation_state_task_4.pt"
    if continuation.is_file():
        print(f"D4 base already exists: {continuation}", flush=True)
    else:
        overrides = {
            "data_dir": DATA_DIR, "output_dir": str(OUTPUT_ROOT), "resume_output_dir": str(OUTPUT_ROOT),
            "resume_state_path": None, "task_start": 0, "task_end": 4,
            "save_resume_after_task": True, "random_seed": SEED, "seed": SEED,
            "rounds_per_task": ROUNDS, "eval_every": 9999, "round_checkpoint_every": ROUNDS,
            "denice_checkpoint_format": "full", "denice_post_task_eval": False,
            "denice_max_clients": MAX_CLIENTS,
        }
        env = {**os.environ, "DENICE_SEED": str(SEED), "DENICE_TRAIN_PHASE": "5",
               "DENICE_OUTPUT_DIR": str(OUTPUT_ROOT),
               "DENICE_CONFIG_OVERRIDES": json.dumps(overrides, sort_keys=True)}
        print("D4 base: train baseline tasks 0--4 and save a full continuation state.", flush=True)
        subprocess.run([sys.executable, str(REPO_DIR / "train_incremental_kaggle.py")], check=True, env=env)
    if not continuation.is_file():
        raise FileNotFoundError(f"D4 base did not create {continuation}")
    subprocess.run([sys.executable, str(REPO_DIR / "tools" / "validate_denice_run.py"),
                    "--run-dir", str(OUTPUT_ROOT), "--expected-task-end", "4",
                    "--expected-rounds-per-task", str(ROUNDS),
                    "--output", str(OUTPUT_ROOT / "audit_validation.json")], check=True)
    (OUTPUT_ROOT / "d4_base_manifest.json").write_text(json.dumps({
        "purpose": "shared D4 baseline continuation through task 4", "seed": SEED,
        "continuation": str(continuation), "validation": str(OUTPUT_ROOT / "audit_validation.json"),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
