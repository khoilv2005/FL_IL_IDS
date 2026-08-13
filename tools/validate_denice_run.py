"""Validate DeNICE run artifacts before metrics are used for a decision.

The validator intentionally checks protocol evidence, not model quality.  A
run with a high score but a partial coverage denominator, missing task history,
or mismatched fixed evaluation support is invalid for the audit plan.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _find_task_ids(rows: Iterable[Dict[str, Any]]) -> List[int]:
    return sorted({int(row["task"]) for row in rows if "task" in row})


def _coverage_errors(summary: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    protocols = summary.get("summary") or {}
    for protocol, policies in protocols.items():
        if not isinstance(policies, dict):
            continue
        for policy_name, record in policies.items():
            sampling = record.get("evaluation_sampling") or {}
            if not sampling:
                errors.append(f"{protocol}/{policy_name}: missing evaluation sampling provenance")
            for metric in ("accuracy", "f1_macro", "loss"):
                if record.get(metric) is not None and not _finite(record[metric]):
                    errors.append(f"{protocol}/{policy_name}: non-finite {metric}")
            coverage = record.get("coverage_protocol") or {}
            if protocol == "coverage_aware_local" and not coverage:
                errors.append(f"{protocol}/{policy_name}: missing coverage denominator evidence")
            if coverage:
                if bool(coverage.get("partial_coverage", False)):
                    errors.append(f"{protocol}/{policy_name}: partial coverage metric")
                if int(coverage.get("unsupported_sample_count", 0) or 0) != 0:
                    errors.append(f"{protocol}/{policy_name}: unsupported samples in strict metric")
                if coverage.get("assigned_sample_count") != coverage.get("requested_sample_count"):
                    errors.append(f"{protocol}/{policy_name}: assignment denominator mismatch")
    return errors


def validate_denice_run(
    run_dir: str | Path,
    *,
    expected_task_end: int | None = None,
    expected_rounds_per_task: int | None = None,
    require_evaluation: bool = False,
) -> Dict[str, Any]:
    """Return a machine-readable validity report for one finished run."""
    root = Path(run_dir)
    errors: List[str] = []
    warnings: List[str] = []
    evidence: Dict[str, Any] = {"run_dir": str(root)}
    if not root.is_dir():
        return {"valid": False, "errors": [f"missing run directory: {root}"], "warnings": [], "evidence": evidence}

    config_path = root / "config.json"
    if not config_path.exists():
        config_path = root / "config_phase_resume.json"
    config = _read_json(config_path) if config_path.exists() else None
    if config is None:
        errors.append("missing resolved config.json/config_phase_resume.json")
    else:
        evidence["config"] = {
            key: config.get(key)
            for key in ("algorithm", "mode", "random_seed", "task_start", "task_end", "rounds_per_task")
        }
        if str(config.get("algorithm", "")).lower() != "denice":
            errors.append("config algorithm is not denice")
        if str(config.get("mode", "")).lower() != "decentralized":
            errors.append("config mode is not decentralized")

    history_path = root / "training_history.json"
    if not history_path.exists():
        errors.append("missing training_history.json")
        history = {}
    else:
        history = _read_json(history_path)
    task_rows = list(history.get("task_accuracies") or [])
    round_rows = list(history.get("round_metrics") or [])
    task_ids = _find_task_ids(task_rows)
    evidence["task_ids"] = task_ids
    evidence["round_record_count"] = len(round_rows)
    end = expected_task_end
    if end is None and config is not None:
        end = config.get("task_end")
    if end is not None:
        expected_tasks = list(range(int((config or {}).get("task_start", 0)), int(end) + 1))
        if task_ids != expected_tasks:
            errors.append(f"task history mismatch: got {task_ids}, expected {expected_tasks}")
        final_ckpt = root / f"checkpoint_task_{int(end)}.pt"
        if not final_ckpt.exists():
            errors.append(f"missing final task checkpoint: {final_ckpt.name}")

    rounds = expected_rounds_per_task
    if rounds is None and config is not None:
        rounds = config.get("rounds_per_task")
    if rounds is not None and task_ids:
        for task_id in task_ids:
            records = [row for row in round_rows if int(row.get("task", -1)) == task_id]
            if len(records) != int(rounds):
                errors.append(
                    f"task {task_id}: {len(records)} round records, expected {int(rounds)}"
                )

    for row in round_rows:
        for metric in ("train_loss", "loss", "accuracy", "f1_macro"):
            value = row.get(metric)
            if value is not None and not _finite(value):
                errors.append(f"task {row.get('task')} round {row.get('round')}: non-finite {metric}")

    evaluation_path = next(
        (
            root / directory / "p6_evaluation_summary.json"
            for directory in ("d1_evaluation", "d3_evaluation", "p6_evaluation")
            if (root / directory / "p6_evaluation_summary.json").exists()
        ),
        root / "p6_evaluation" / "p6_evaluation_summary.json",
    )
    if evaluation_path.exists():
        summary = _read_json(evaluation_path)
        evidence["evaluation_summary"] = str(evaluation_path)
        errors.extend(_coverage_errors(summary))
        policies = set(summary.get("policies") or [])
        required = {
            "e0_backbone_nomask", "e1_pred_adapter_nomask",
            "e2_oracle_adapter_nomask", "e3_oracle_routed_system_ceiling",
            "e3b_oracle_hard_no_adapter", "e4_pred_hard",
        }
        missing = sorted(required - policies)
        if missing:
            errors.append(f"evaluation missing required policy semantics: {missing}")
    elif require_evaluation:
        errors.append("missing P6 evaluation summary")
    else:
        warnings.append("no P6 evaluation summary found")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "evidence": evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-task-end", type=int, default=None)
    parser.add_argument("--expected-rounds-per-task", type=int, default=None)
    parser.add_argument("--require-evaluation", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = validate_denice_run(
        args.run_dir,
        expected_task_end=args.expected_task_end,
        expected_rounds_per_task=args.expected_rounds_per_task,
        require_evaluation=args.require_evaluation,
    )
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
