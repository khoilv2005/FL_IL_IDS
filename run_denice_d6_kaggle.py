"""D6: isolate the effect of peer aggregation on the D4 training budget.

Each branch follows the exact D4 reserve schedule: tasks 0--4 retain the
10-percent capacity reserve, then the terminal task runs with reserve zero.
The peer and self-only branches differ only in ``denice_aggregation_mode``.
Both branches are trained from scratch; disabling peers after a completed
peer-aggregated checkpoint would not be a causal ablation.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


SEED = int(os.environ.get("DENICE_SEED", "42"))
DATA_DIR = os.environ.get("DENICE_DATA_DIR", "/kaggle/input/datasets/khoilv2005/100-clients/100-clients")
REPO_DIR = Path(os.environ.get("DENICE_REPO_DIR", "/kaggle/working/FL_IL_IDS"))
OUTPUT_ROOT = Path(os.environ.get("D6_OUTPUT_ROOT", f"/kaggle/working/denice_d6_seed_{SEED}"))
MAX_CLIENTS = int(os.environ.get("D6_MAX_CLIENTS", "20"))
ROUNDS = int(os.environ.get("D6_ROUNDS_PER_TASK", "5"))
SAMPLES_PER_CLASS = int(os.environ.get("D6_SAMPLES_PER_CLASS", "100"))
EVAL_DEVICE = os.environ.get("DENICE_EVAL_DEVICE", "cuda")


VARIANTS: Dict[str, Dict[str, Any]] = {
    "peer": {"denice_aggregation_mode": "peer", "denice_collaboration_guard_mode": "error"},
    "self_only": {"denice_aggregation_mode": "self_only", "denice_collaboration_guard_mode": "off"},
}


def _run_training(overrides: Dict[str, Any], output_dir: Path) -> None:
    env = {
        **os.environ,
        "DENICE_SEED": str(SEED),
        "DENICE_TRAIN_PHASE": "5",
        "DENICE_OUTPUT_DIR": str(output_dir),
        "DENICE_CONFIG_OVERRIDES": json.dumps(overrides, sort_keys=True),
    }
    subprocess.run([sys.executable, str(REPO_DIR / "train_incremental_kaggle.py")], check=True, env=env)


def _validate(run_dir: Path, *, expected_task_end: int, require_evaluation: bool) -> None:
    command = [
        sys.executable, str(REPO_DIR / "tools" / "validate_denice_run.py"), "--run-dir", str(run_dir),
        "--expected-task-end", str(expected_task_end), "--expected-rounds-per-task", str(ROUNDS),
        "--output", str(run_dir / "audit_validation.json"),
    ]
    if require_evaluation:
        command.append("--require-evaluation")
    subprocess.run(command, check=True, env=os.environ)


def _common(output_dir: Path, variant: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "data_dir": DATA_DIR, "output_dir": str(output_dir), "resume_output_dir": str(output_dir),
        "random_seed": SEED, "seed": SEED, "rounds_per_task": ROUNDS, "eval_every": 9999,
        "round_checkpoint_every": ROUNDS, "denice_checkpoint_format": "full",
        "denice_post_task_eval": False, "denice_max_clients": MAX_CLIENTS,
        **variant,
    }


def main() -> None:
    if not REPO_DIR.is_dir() or not Path(DATA_DIR).is_dir():
        raise FileNotFoundError("D6 needs existing DENICE_REPO_DIR and DENICE_DATA_DIR.")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "purpose": "D6 D4-budget peer-aggregation causal ablation", "seed": SEED,
        "d4_schedule": {"tasks_0_to_4_min_free_capacity_ratio": 0.10, "task_5_min_free_capacity_ratio": 0.0},
        "fixed_evaluation": {"protocol": "coverage_aware_local", "samples_per_class": SAMPLES_PER_CLASS},
        "variants": {},
    }
    for name, variant in VARIANTS.items():
        variant_root = OUTPUT_ROOT / name
        base_dir = variant_root / "base_tasks_0_to_4"
        terminal_dir = variant_root / "terminal_task_5"
        continuation = base_dir / "continuation_state_task_4.pt"
        checkpoint = terminal_dir / f"checkpoint_task_5_round_{ROUNDS - 1}.pt"
        evaluation = terminal_dir / "d6_evaluation"
        if not continuation.is_file():
            base = _common(base_dir, variant)
            base.update({"task_start": 0, "task_end": 4, "save_resume_after_task": True,
                         "resume_state_path": None, "denice_min_free_capacity_ratio": 0.10})
            print(f"D6 {name}: tasks 0--4 with D4's 10% reserve.", flush=True)
            _run_training(base, base_dir)
        if not continuation.is_file():
            raise FileNotFoundError(f"D6 {name} failed to create {continuation}")
        _validate(base_dir, expected_task_end=4, require_evaluation=False)
        if not checkpoint.is_file() or not (evaluation / "p6_evaluation_summary.json").is_file():
            terminal = _common(terminal_dir, variant)
            terminal.update({"task_start": 5, "task_end": 5, "save_resume_after_task": None,
                             "resume_state_path": str(continuation), "denice_min_free_capacity_ratio": 0.0})
            print(f"D6 {name}: terminal task 5 with reserve 0; aggregation mode={variant['denice_aggregation_mode']}.", flush=True)
            _run_training(terminal, terminal_dir)
            if not checkpoint.is_file():
                raise FileNotFoundError(f"D6 {name} failed to create {checkpoint}")
            subprocess.run([
                sys.executable, str(REPO_DIR / "run_denice_p6_eval.py"), "--checkpoint", str(checkpoint),
                "--data-dir", DATA_DIR, "--output-dir", str(evaluation), "--device", EVAL_DEVICE,
                "--seed", str(SEED), "--protocols", "coverage_aware_local", "--samples-per-class",
                str(SAMPLES_PER_CLASS), "--class-balanced-with-replacement", "--save-prediction-trace",
            ], check=True, env=os.environ)
        _validate(terminal_dir, expected_task_end=5, require_evaluation=True)
        manifest["variants"][name] = {
            "aggregation_mode": variant["denice_aggregation_mode"], "base_dir": str(base_dir),
            "continuation": str(continuation), "checkpoint": str(checkpoint),
            "evaluation_summary": str(evaluation / "p6_evaluation_summary.json"),
            "validation": str(terminal_dir / "audit_validation.json"),
        }
        (OUTPUT_ROOT / "d6_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    subprocess.run([
        sys.executable, str(REPO_DIR / "tools" / "analyze_denice_d6.py"), "--manifest",
        str(OUTPUT_ROOT / "d6_manifest.json"), "--output", str(OUTPUT_ROOT / "d6_decision_report.json"),
    ], check=True, env=os.environ)


if __name__ == "__main__":
    main()
