"""Frozen full DeNICE run after the two-seed D2 confirmation.

The selected method keeps peer aggregation for shared parameters, but accepts a
peer contribution to a plastic classifier row only when that peer trained on
the row's class.  The reserve schedule matches the D2 causal runs: 10% through
task 4, then no reserve restriction for the terminal task.
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
OUTPUT_ROOT = Path(
    os.environ.get("FULL_D2_OUTPUT_ROOT", f"/kaggle/working/denice_full_d2_seed_{SEED}")
)
MAX_CLIENTS = int(os.environ.get("FULL_D2_MAX_CLIENTS", "100"))
ROUNDS = int(os.environ.get("FULL_D2_ROUNDS_PER_TASK", "20"))
SAMPLES_PER_CLASS = int(os.environ.get("FULL_D2_SAMPLES_PER_CLASS", "100"))
EVAL_DEVICE = os.environ.get("DENICE_EVAL_DEVICE", "cuda")


def _train(overrides: Dict[str, Any], output_dir: Path) -> None:
    env = {
        **os.environ,
        "DENICE_SEED": str(SEED),
        "DENICE_TRAIN_PHASE": "5",
        "DENICE_OUTPUT_DIR": str(output_dir),
        "DENICE_CONFIG_OVERRIDES": json.dumps(overrides, sort_keys=True),
    }
    subprocess.run(
        [sys.executable, str(REPO_DIR / "train_incremental_kaggle.py")],
        check=True,
        env=env,
    )


def _validate(run_dir: Path, task_end: int, require_evaluation: bool) -> None:
    command = [
        sys.executable,
        str(REPO_DIR / "tools" / "validate_denice_run.py"),
        "--run-dir",
        str(run_dir),
        "--expected-task-end",
        str(task_end),
        "--expected-rounds-per-task",
        str(ROUNDS),
        "--output",
        str(run_dir / "audit_validation.json"),
    ]
    if require_evaluation:
        command.append("--require-evaluation")
    subprocess.run(command, check=True, env=os.environ)


def _common(output_dir: Path) -> Dict[str, Any]:
    return {
        "data_dir": DATA_DIR,
        "output_dir": str(output_dir),
        "resume_output_dir": str(output_dir),
        "random_seed": SEED,
        "seed": SEED,
        "task_start": 0,
        "task_end": 5,
        "rounds_per_task": ROUNDS,
        "eval_every": 9999,
        "round_checkpoint_every": ROUNDS,
        "denice_checkpoint_format": "full",
        "denice_post_task_eval": False,
        "denice_max_clients": MAX_CLIENTS,
        "denice_aggregation_mode": "peer",
        "denice_collaboration_guard_mode": "error",
        "denice_selective_fc2_peer_rows": True,
    }


def main() -> None:
    if not REPO_DIR.is_dir() or not Path(DATA_DIR).is_dir():
        raise FileNotFoundError("FULL_D2 needs existing DENICE_REPO_DIR and DENICE_DATA_DIR.")
    if MAX_CLIENTS <= 0 or ROUNDS <= 0 or SAMPLES_PER_CLASS <= 0:
        raise ValueError("FULL_D2_MAX_CLIENTS, FULL_D2_ROUNDS_PER_TASK, and samples per class must be positive.")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    base_dir = OUTPUT_ROOT / "base_tasks_0_to_4"
    terminal_dir = OUTPUT_ROOT / "terminal_task_5"
    continuation = base_dir / "continuation_state_task_4.pt"
    checkpoint = terminal_dir / f"checkpoint_task_5_round_{ROUNDS - 1}.pt"
    evaluation = terminal_dir / "p6_evaluation"

    if not continuation.is_file():
        base = _common(base_dir)
        base.update(
            {
                "task_start": 0,
                "task_end": 4,
                "save_resume_after_task": True,
                "resume_state_path": None,
                "denice_min_free_capacity_ratio": 0.10,
            }
        )
        print(
            "FULL_D2: tasks 0--4 with 10% reserve and confirmed selective-fc2 peer protection.",
            flush=True,
        )
        _train(base, base_dir)
    if not continuation.is_file():
        raise FileNotFoundError(f"FULL_D2 missing continuation {continuation}")
    _validate(base_dir, 4, False)

    if not checkpoint.is_file() or not (evaluation / "p6_evaluation_summary.json").is_file():
        terminal = _common(terminal_dir)
        terminal.update(
            {
                "task_start": 5,
                "task_end": 5,
                "save_resume_after_task": None,
                "resume_state_path": str(continuation),
                "denice_min_free_capacity_ratio": 0.0,
            }
        )
        print("FULL_D2: terminal task 5 with zero reserve, then P6 strict evaluation.", flush=True)
        _train(terminal, terminal_dir)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"FULL_D2 missing terminal checkpoint {checkpoint}")
        subprocess.run(
            [
                sys.executable,
                str(REPO_DIR / "run_denice_p6_eval.py"),
                "--checkpoint",
                str(checkpoint),
                "--data-dir",
                DATA_DIR,
                "--output-dir",
                str(evaluation),
                "--device",
                EVAL_DEVICE,
                "--seed",
                str(SEED),
                "--protocols",
                "coverage_aware_local",
                "--samples-per-class",
                str(SAMPLES_PER_CLASS),
                "--class-balanced-with-replacement",
                "--save-prediction-trace",
            ],
            check=True,
            env=os.environ,
        )
    _validate(terminal_dir, 5, True)

    manifest = {
        "purpose": "Frozen full DeNICE run after two-seed D2 confirmation",
        "seed": SEED,
        "schedule": {
            "tasks_0_to_4_min_free_capacity_ratio": 0.10,
            "task_5_min_free_capacity_ratio": 0.0,
        },
        "selected_method": {
            "denice_aggregation_mode": "peer",
            "denice_selective_fc2_peer_rows": True,
            "scope": "plastic fc2 rows only; shared layers and adapters retain peer aggregation",
        },
        "budget": {"max_clients": MAX_CLIENTS, "rounds_per_task": ROUNDS},
        "fixed_evaluation": {
            "protocol": "coverage_aware_local",
            "samples_per_class": SAMPLES_PER_CLASS,
            "class_balanced_with_replacement": True,
        },
        "base_dir": str(base_dir),
        "continuation": str(continuation),
        "checkpoint": str(checkpoint),
        "evaluation_summary": str(evaluation / "p6_evaluation_summary.json"),
        "validation": str(terminal_dir / "audit_validation.json"),
    }
    (OUTPUT_ROOT / "full_d2_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
