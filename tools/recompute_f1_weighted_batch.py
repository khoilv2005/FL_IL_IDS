"""
Batch recompute f1_weighted for saved round checkpoints without retraining.

This is a convenience wrapper around recompute_f1_weighted_from_checkpoint.py.
It writes one CSV/JSON per algorithm plus a combined CSV/JSON.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fed_learning.data.incremental_loader import IncrementalDataLoader
from tools.recompute_f1_weighted_from_checkpoint import (
    _create_model,
    _evaluate,
    _infer_input_shape,
    _infer_num_classes,
    _load_state,
    _resolve_device,
)


CHECKPOINT_RE = re.compile(r"checkpoint_task_(\d+)_round_(\d+)\.pt$")


DEFAULT_ALGORITHM_DIRS = {
    "der": [
        "results_der_phase1",
        "results_der_phase2",
        "results_der_phase3",
        "results_der_phase4",
        "results_der_phase5",
    ],
    "ewc": ["results_ewc"],
    "lwf": ["results_lwf"],
}


def _read_existing_rows(path: Path) -> Dict[Tuple[str, int, int], Dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    existing = {}
    for row in rows:
        try:
            key = (row["algorithm"], int(row["task"]), int(row["round"]))
            existing[key] = row
        except (KeyError, ValueError):
            continue
    return existing


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_json(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def _checkpoint_sort_key(path: Path) -> Tuple[int, int, str]:
    parsed = _parse_checkpoint_name(path)
    if parsed is None:
        return (10**9, 10**9, str(path))
    task_id, round_id = parsed
    return (task_id, round_id, str(path))


def _parse_checkpoint_name(path: Path) -> Optional[Tuple[int, int]]:
    match = CHECKPOINT_RE.match(path.name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _find_checkpoints(
    results_root: Path,
    algorithm: str,
    task_start: int,
    task_end: int,
    rounds_per_task: int,
    round_ids: Optional[set[int]],
) -> List[Path]:
    checkpoints = []
    for dir_name in DEFAULT_ALGORITHM_DIRS[algorithm]:
        root = results_root / dir_name
        if not root.exists():
            print(f"[WARN] Missing directory for {algorithm}: {root}")
            continue
        for path in root.rglob("checkpoint_task_*_round_*.pt"):
            parsed = _parse_checkpoint_name(path)
            if parsed is None:
                continue
            task_id, round_id = parsed
            if task_start <= task_id <= task_end and 0 <= round_id < rounds_per_task:
                if round_ids is not None and round_id not in round_ids:
                    continue
                checkpoints.append(path)
    checkpoints.sort(key=_checkpoint_sort_key)
    return checkpoints


def _metric(metrics: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metrics:
            return metrics[key]
    return None


def _evaluate_checkpoint(
    checkpoint_path: Path,
    data_loader: IncrementalDataLoader,
    device: torch.device,
    batch_size: int,
    strict: bool,
    apply_seen_mask: bool,
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"{checkpoint_path} is missing model_state_dict")

    config = checkpoint.get("config", {})
    metrics_old = checkpoint.get("metrics", {}) or {}
    task_id = int(checkpoint.get("task_id", _parse_checkpoint_name(checkpoint_path)[0]))
    round_id = int(checkpoint.get("round_id", _parse_checkpoint_name(checkpoint_path)[1]))
    algorithm = str(config.get("algorithm") or checkpoint.get("algorithm") or "unknown").lower()
    seen_classes = [int(c) for c in checkpoint.get("seen_classes", [])]
    input_shape = _infer_input_shape(config, data_loader)
    num_classes = _infer_num_classes(config, seen_classes or range(data_loader.get_total_classes()))
    state_dict = checkpoint["model_state_dict"]

    model = _create_model(algorithm, input_shape, num_classes, state_dict, config, device)
    load_info = _load_state(model, state_dict, device, strict=strict)

    X_test, y_test = data_loader.get_test_data(task_id, cumulative=True)
    if len(y_test) == 0:
        raise ValueError(f"Empty cumulative test set for task {task_id}")

    metrics_new, _, _ = _evaluate(
        model=model,
        X_test=X_test,
        y_test=y_test,
        device=device,
        batch_size=batch_size,
        seen_classes=seen_classes,
        apply_seen_mask=apply_seen_mask,
    )

    return {
        "algorithm": algorithm,
        "task": task_id,
        "round": round_id,
        "checkpoint": str(checkpoint_path),
        "num_test_samples": int(len(y_test)),
        "train_loss": _metric(metrics_old, "train_loss"),
        "test_loss_old": _metric(metrics_old, "loss", "test_loss"),
        "accuracy_old": _metric(metrics_old, "accuracy"),
        "precision_macro_old": _metric(metrics_old, "precision_macro"),
        "recall_macro_old": _metric(metrics_old, "recall_macro"),
        "f1_macro_old": _metric(metrics_old, "f1_macro"),
        "avg_forgetting_old": _metric(metrics_old, "avg_forgetting"),
        "test_loss": metrics_new["loss"],
        "accuracy": metrics_new["accuracy"],
        "precision_macro": metrics_new["precision_macro"],
        "recall_macro": metrics_new["recall_macro"],
        "f1_macro": metrics_new["f1_macro"],
        "f1_weighted": metrics_new["f1_weighted"],
        "apply_seen_mask": apply_seen_mask,
        "strict_loaded": load_info.get("strict"),
        "load_warning": load_info.get("warning"),
    }


def recompute_batch(
    results_root: Path,
    data_dir: Path,
    output_dir: Path,
    algorithms: Iterable[str],
    task_start: int,
    task_end: int,
    rounds_per_task: int,
    round_ids: Optional[set[int]],
    device_arg: str,
    batch_size: int,
    strict: bool,
    apply_seen_mask: bool,
    resume: bool,
) -> List[Dict[str, Any]]:
    device = _resolve_device(device_arg)
    data_loader = IncrementalDataLoader(str(data_dir))
    fieldnames = [
        "algorithm",
        "task",
        "round",
        "train_loss",
        "test_loss_old",
        "accuracy_old",
        "precision_macro_old",
        "recall_macro_old",
        "f1_macro_old",
        "avg_forgetting_old",
        "test_loss",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "num_test_samples",
        "apply_seen_mask",
        "strict_loaded",
        "load_warning",
        "checkpoint",
    ]

    all_rows: List[Dict[str, Any]] = []
    combined_csv = output_dir / "f1_weighted_recomputed_all.csv"
    existing_combined = _read_existing_rows(combined_csv) if resume else {}

    for algorithm in algorithms:
        algorithm = algorithm.lower()
        checkpoints = _find_checkpoints(results_root, algorithm, task_start, task_end, rounds_per_task, round_ids)
        algo_csv = output_dir / f"f1_weighted_recomputed_{algorithm}.csv"
        existing_algo = _read_existing_rows(algo_csv) if resume else {}
        rows = list(existing_algo.values())
        completed = set(existing_algo) | set(existing_combined)

        print(f"\n[{algorithm.upper()}] {len(checkpoints)} checkpoints found")
        for index, checkpoint_path in enumerate(checkpoints, start=1):
            task_id, round_id = _parse_checkpoint_name(checkpoint_path)
            key = (algorithm, task_id, round_id)
            if resume and key in completed:
                print(f"  skip existing task={task_id} round={round_id}")
                continue

            print(f"  [{index}/{len(checkpoints)}] task={task_id} round={round_id}: {checkpoint_path}")
            row = _evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                data_loader=data_loader,
                device=device,
                batch_size=batch_size,
                strict=strict,
                apply_seen_mask=apply_seen_mask,
            )
            rows.append(row)
            rows.sort(key=lambda r: (int(r["task"]), int(r["round"])))
            _write_csv(algo_csv, rows, fieldnames)
            _write_json(output_dir / f"f1_weighted_recomputed_{algorithm}.json", rows)

        rows.sort(key=lambda r: (str(r["algorithm"]), int(r["task"]), int(r["round"])))
        all_rows.extend(rows)

    all_rows.sort(key=lambda r: (str(r["algorithm"]), int(r["task"]), int(r["round"])))
    _write_csv(combined_csv, all_rows, fieldnames)
    _write_json(output_dir / "f1_weighted_recomputed_all.json", all_rows)
    return all_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch recompute f1_weighted from checkpoint round files.")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--algorithms", nargs="+", default=["der", "ewc", "lwf"], choices=sorted(DEFAULT_ALGORITHM_DIRS))
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--task-end", type=int, default=5)
    parser.add_argument("--rounds-per-task", type=int, default=20)
    parser.add_argument("--round-ids", type=int, nargs="*", default=None, help="Optional explicit round ids to evaluate, e.g. --round-ids 19")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--non-strict", action="store_true")
    parser.add_argument("--apply-seen-mask", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    rows = recompute_batch(
        results_root=Path(args.results_root),
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        algorithms=args.algorithms,
        task_start=args.task_start,
        task_end=args.task_end,
        rounds_per_task=args.rounds_per_task,
        round_ids=set(args.round_ids) if args.round_ids else None,
        device_arg=args.device,
        batch_size=args.batch_size,
        strict=not args.non_strict,
        apply_seen_mask=args.apply_seen_mask,
        resume=not args.no_resume,
    )
    print(f"\nDone. Wrote {len(rows)} rows to {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
