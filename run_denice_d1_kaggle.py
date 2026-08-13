"""Run the controlled D1 peer-aggregation ablation on Kaggle.

The three variants share a seed, data split, task range, client cap, local
training budget, and class-balanced E0--E6 evaluation protocol.  Only the
peer-aggregation control changes:

* ``peer_default``: recovered DeNICE aggregation;
* ``self_only``: no peer parameters, adapters, or neuron ages are merged;
* ``peer_self_floor_050``: peer aggregation with at least 50% local weight.

Example::

    python run_denice_d1_kaggle.py

The default is the larger decision run from the consolidated audit (20
clients, tasks 0--2, five rounds/task).  Use environment variables to adjust
the budget without modifying this file: ``D1_TASK_END``, ``D1_MAX_CLIENTS``,
``D1_ROUNDS_PER_TASK``, ``D1_MAX_TRAIN_SAMPLES``, and
``D1_SAMPLES_PER_CLASS``.
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
OUTPUT_ROOT = Path(os.environ.get("D1_OUTPUT_ROOT", f"/kaggle/working/denice_d1_seed_{SEED}"))
TASK_END = int(os.environ.get("D1_TASK_END", "2"))
MAX_CLIENTS = int(os.environ.get("D1_MAX_CLIENTS", "20"))
ROUNDS_PER_TASK = int(os.environ.get("D1_ROUNDS_PER_TASK", "5"))
MAX_TRAIN_SAMPLES = int(os.environ.get("D1_MAX_TRAIN_SAMPLES", "300"))
SAMPLES_PER_CLASS = int(os.environ.get("D1_SAMPLES_PER_CLASS", "100"))
EVAL_DEVICE = os.environ.get("DENICE_EVAL_DEVICE", "cuda")


VARIANTS: Dict[str, Dict[str, Any]] = {
    "peer_default": {
        "denice_aggregation_mode": "peer",
        "denice_collaboration_guard_mode": "error",
    },
    "self_only": {
        "denice_aggregation_mode": "self_only",
        # A collaboration guard would correctly reject this intentional control.
        "denice_collaboration_guard_mode": "off",
    },
    "peer_self_floor_050": {
        "denice_aggregation_mode": "peer",
        "denice_aggregation_self_floor": 0.50,
        "denice_collaboration_guard_mode": "error",
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
        "max_clients": MAX_CLIENTS,
        "rounds_per_task": ROUNDS_PER_TASK,
        "eval_device": EVAL_DEVICE,
        "save_resume_after_task": None,
        "resume_state_path": None,
        # The DeNICE runner otherwise appends an algorithm/timestamp suffix,
        # which would make the deterministic checkpoint path below ambiguous.
        "resume_output_dir": str(output_dir),
        "rounds_per_task": ROUNDS_PER_TASK,
        "eval_every": 9999,
        "round_checkpoint_every": ROUNDS_PER_TASK,
        "denice_checkpoint_format": "full",
        "denice_post_task_eval": False,
        "denice_max_clients": MAX_CLIENTS,
        "denice_max_train_samples_per_client": MAX_TRAIN_SAMPLES,
        "denice_d1_row_drift_audit": True,
    }


def main() -> None:
    if not REPO_DIR.is_dir():
        raise FileNotFoundError(f"DENICE_REPO_DIR does not exist: {REPO_DIR}")
    if not Path(DATA_DIR).is_dir():
        raise FileNotFoundError(f"DENICE_DATA_DIR does not exist: {DATA_DIR}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "purpose": "D1 controlled peer-aggregation ablation",
        "seed": SEED,
        "task_end": TASK_END,
        "variants": {},
    }

    for name, variant in VARIANTS.items():
        run_dir = OUTPUT_ROOT / name
        overrides = {**_shared_overrides(run_dir), **variant}
        env = {
            **os.environ,
            "DENICE_SEED": str(SEED),
            "DENICE_TRAIN_PHASE": "5",
            "DENICE_OUTPUT_DIR": str(run_dir),
            "DENICE_CONFIG_OVERRIDES": json.dumps(overrides, sort_keys=True),
        }
        subprocess.run(
            [sys.executable, str(REPO_DIR / "train_incremental_kaggle.py")],
            check=True,
            env=env,
        )
        checkpoint = run_dir / f"checkpoint_task_{TASK_END}_round_{ROUNDS_PER_TASK - 1}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"D1 training did not create {checkpoint}")
        eval_dir = run_dir / "d1_evaluation"
        subprocess.run(
            [
                sys.executable,
                str(REPO_DIR / "run_denice_p6_eval.py"),
                "--checkpoint", str(checkpoint),
                "--data-dir", DATA_DIR,
                "--output-dir", str(eval_dir),
                "--device", EVAL_DEVICE,
                "--seed", str(SEED),
                "--protocols", "coverage_aware_local",
                "--samples-per-class", str(SAMPLES_PER_CLASS),
                "--class-balanced-with-replacement",
            ],
            check=True,
            env=env,
        )
        subprocess.run(
            [
                sys.executable,
                str(REPO_DIR / "tools" / "validate_denice_run.py"),
                "--run-dir", str(run_dir),
                "--expected-task-end", str(TASK_END),
                "--expected-rounds-per-task", str(ROUNDS_PER_TASK),
                "--require-evaluation",
                "--output", str(run_dir / "audit_validation.json"),
            ],
            check=True,
            env=env,
        )
        manifest["variants"][name] = {
            "overrides": overrides,
            "checkpoint": str(checkpoint),
            "evaluation_summary": str(eval_dir / "p6_evaluation_summary.json"),
            "validation": str(run_dir / "audit_validation.json"),
        }
        (OUTPUT_ROOT / "d1_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    # This is a decision report, not a success/failure check: it exits zero
    # when D2 correctly remains closed for insufficient evidence.
    subprocess.run(
        [
            sys.executable,
            str(REPO_DIR / "tools" / "analyze_denice_d1.py"),
            "--manifest", str(OUTPUT_ROOT / "d1_manifest.json"),
            "--output", str(OUTPUT_ROOT / "d1_decision_report.json"),
        ],
        check=True,
        env=os.environ,
    )


if __name__ == "__main__":
    main()
