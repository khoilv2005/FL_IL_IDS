"""Run the preregistered D3 one-factor imbalance matrix on a fixed protocol.

Runs baseline, D3-A (class-balanced local batches), and D3-B (clipped,
smoothed effective-number CE).  Each variant uses the same seed, split,
training budget, and class-balanced strict P6 support.  Prediction traces are
saved solely to enable a paired-bootstrap decision after the runs finish.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


SEED = int(os.environ.get("DENICE_SEED", "42"))
DATA_DIR = os.environ.get(
    "DENICE_DATA_DIR", "/kaggle/input/datasets/khoilv2005/100-clients/100-clients"
)
REPO_DIR = Path(os.environ.get("DENICE_REPO_DIR", "/kaggle/working/FL_IL_IDS"))
OUTPUT_ROOT = Path(os.environ.get("D3_OUTPUT_ROOT", f"/kaggle/working/denice_d3_seed_{SEED}"))
TASK_END = int(os.environ.get("D3_TASK_END", "2"))
MAX_CLIENTS = int(os.environ.get("D3_MAX_CLIENTS", "20"))
ROUNDS_PER_TASK = int(os.environ.get("D3_ROUNDS_PER_TASK", "5"))
MAX_TRAIN_SAMPLES = int(os.environ.get("D3_MAX_TRAIN_SAMPLES", "300"))
SAMPLES_PER_CLASS = int(os.environ.get("D3_SAMPLES_PER_CLASS", "100"))
EVAL_DEVICE = os.environ.get("DENICE_EVAL_DEVICE", "cuda")


VARIANTS: Dict[str, Dict[str, Any]] = {
    "baseline": {},
    "class_balanced_batches": {"denice_batch_sampling": "class_balanced"},
    "effective_number_ce": {
        "denice_class_weight_mode": "effective_number",
        "denice_class_weight_smoothing": 1.0,
        "denice_class_weight_effective_beta": 0.999,
        "denice_class_weight_min": 0.25,
        "denice_class_weight_max": 4.0,
    },
}


def _shared_overrides(output_dir: Path) -> Dict[str, Any]:
    return {
        "data_dir": DATA_DIR,
        "output_dir": str(output_dir),
        "random_seed": SEED,
        "seed": SEED,
        "task_start": 0,
        "task_end": TASK_END,
        "rounds_per_task": ROUNDS_PER_TASK,
        "eval_device": EVAL_DEVICE,
        "resume_output_dir": str(output_dir),
        "eval_every": 9999,
        "round_checkpoint_every": ROUNDS_PER_TASK,
        "denice_checkpoint_format": "full",
        "denice_post_task_eval": False,
        "denice_max_clients": MAX_CLIENTS,
        "denice_max_train_samples_per_client": MAX_TRAIN_SAMPLES,
        "denice_d1_row_drift_audit": True,
    }


def main() -> None:
    if not REPO_DIR.is_dir() or not Path(DATA_DIR).is_dir():
        raise FileNotFoundError("Set DENICE_REPO_DIR and DENICE_DATA_DIR to existing directories.")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "purpose": "D3 one-factor local class-imbalance ablation",
        "seed": SEED,
        "data_dir": DATA_DIR,
        "task_end": TASK_END,
        "variants": {},
    }
    for name, variant in VARIANTS.items():
        run_dir = OUTPUT_ROOT / name
        overrides = {**_shared_overrides(run_dir), **variant}
        env = {**os.environ, "DENICE_SEED": str(SEED), "DENICE_TRAIN_PHASE": "5",
               "DENICE_OUTPUT_DIR": str(run_dir),
               "DENICE_CONFIG_OVERRIDES": json.dumps(overrides, sort_keys=True)}
        print(f"D3 {name}: only imbalance factor differs from baseline.", flush=True)
        subprocess.run([sys.executable, str(REPO_DIR / "train_incremental_kaggle.py")], check=True, env=env)
        checkpoint = run_dir / f"checkpoint_task_{TASK_END}_round_{ROUNDS_PER_TASK - 1}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"D3 training did not create {checkpoint}")
        evaluation = run_dir / "d3_evaluation"
        subprocess.run([
            sys.executable, str(REPO_DIR / "run_denice_p6_eval.py"),
            "--checkpoint", str(checkpoint), "--data-dir", DATA_DIR,
            "--output-dir", str(evaluation), "--device", EVAL_DEVICE,
            "--seed", str(SEED), "--protocols", "coverage_aware_local",
            "--samples-per-class", str(SAMPLES_PER_CLASS),
            "--class-balanced-with-replacement", "--save-prediction-trace",
        ], check=True, env=env)
        subprocess.run([
            sys.executable, str(REPO_DIR / "tools" / "validate_denice_run.py"),
            "--run-dir", str(run_dir), "--expected-task-end", str(TASK_END),
            "--expected-rounds-per-task", str(ROUNDS_PER_TASK), "--require-evaluation",
            "--output", str(run_dir / "audit_validation.json"),
        ], check=True, env=env)
        manifest["variants"][name] = {"overrides": variant, "checkpoint": str(checkpoint),
                                      "evaluation_dir": str(evaluation),
                                      "validation": str(run_dir / "audit_validation.json")}
        (OUTPUT_ROOT / "d3_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    subprocess.run([
        sys.executable, str(REPO_DIR / "tools" / "analyze_denice_d3.py"),
        "--manifest", str(OUTPUT_ROOT / "d3_manifest.json"),
        "--output", str(OUTPUT_ROOT / "d3_decision_report.json"),
    ], check=True, env=os.environ)


if __name__ == "__main__":
    main()
