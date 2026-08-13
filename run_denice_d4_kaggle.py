"""Compare terminal task-5 capacity reserve from one identical task-4 state."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


SEED = int(os.environ.get("DENICE_SEED", "42"))
DATA_DIR = os.environ.get("DENICE_DATA_DIR", "/kaggle/input/datasets/khoilv2005/100-clients/100-clients")
REPO_DIR = Path(os.environ.get("DENICE_REPO_DIR", "/kaggle/working/FL_IL_IDS"))
BASE_CONTINUATION = Path(os.environ.get("D4_BASE_CONTINUATION", ""))
OUTPUT_ROOT = Path(os.environ.get("D4_OUTPUT_ROOT", f"/kaggle/working/denice_d4_seed_{SEED}"))
MAX_CLIENTS = int(os.environ.get("D4_MAX_CLIENTS", "20"))
ROUNDS = int(os.environ.get("D4_ROUNDS_PER_TASK", "5"))
SAMPLES_PER_CLASS = int(os.environ.get("D4_SAMPLES_PER_CLASS", "100"))
EVAL_DEVICE = os.environ.get("DENICE_EVAL_DEVICE", "cuda")


def main() -> None:
    if not REPO_DIR.is_dir() or not Path(DATA_DIR).is_dir() or not BASE_CONTINUATION.is_file():
        raise FileNotFoundError("D4 needs existing DENICE_REPO_DIR, DENICE_DATA_DIR, and D4_BASE_CONTINUATION.")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {"purpose": "D4 task-5 capacity-reserve checkpoint branch", "seed": SEED,
                "data_dir": DATA_DIR, "base_continuation": str(BASE_CONTINUATION), "variants": {}}
    for name, reserve in (("reserve_010", 0.10), ("reserve_000", 0.00)):
        run_dir = OUTPUT_ROOT / name
        overrides = {"data_dir": DATA_DIR, "output_dir": str(run_dir), "resume_output_dir": str(run_dir),
                     "resume_state_path": str(BASE_CONTINUATION), "task_end": 5, "random_seed": SEED,
                     "seed": SEED, "rounds_per_task": ROUNDS, "eval_every": 9999,
                     "round_checkpoint_every": ROUNDS, "denice_checkpoint_format": "full",
                     "denice_post_task_eval": False, "denice_max_clients": MAX_CLIENTS,
                     "denice_min_free_capacity_ratio": reserve}
        env = {**os.environ, "DENICE_SEED": str(SEED), "DENICE_TRAIN_PHASE": "5",
               "DENICE_OUTPUT_DIR": str(run_dir), "DENICE_CONFIG_OVERRIDES": json.dumps(overrides, sort_keys=True)}
        print(f"D4 {name}: resuming exactly {BASE_CONTINUATION.name}; only reserve={reserve} differs.", flush=True)
        subprocess.run([sys.executable, str(REPO_DIR / "train_incremental_kaggle.py")], check=True, env=env)
        checkpoint = run_dir / f"checkpoint_task_5_round_{ROUNDS - 1}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"D4 branch did not create {checkpoint}")
        evaluation = run_dir / "d4_evaluation"
        subprocess.run([sys.executable, str(REPO_DIR / "run_denice_p6_eval.py"), "--checkpoint", str(checkpoint),
                        "--data-dir", DATA_DIR, "--output-dir", str(evaluation), "--device", EVAL_DEVICE,
                        "--seed", str(SEED), "--protocols", "coverage_aware_local", "--samples-per-class",
                        str(SAMPLES_PER_CLASS), "--class-balanced-with-replacement"], check=True, env=env)
        manifest["variants"][name] = {"reserve": reserve, "checkpoint": str(checkpoint),
                                      "evaluation_dir": str(evaluation)}
        (OUTPUT_ROOT / "d4_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
