"""Produce the preregistered D3 decision report from a D3 manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.metrics import f1_score


REQUIRED = ("baseline", "class_balanced_batches", "effective_number_ce")


def _read(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def _policy(run: Dict[str, Any], policy: str) -> Dict[str, Any]:
    path = Path(run["evaluation_dir"]) / f"coverage_aware_local_{policy}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing D3 policy artifact: {path}")
    return _read(path)


def _trace(result: Dict[str, Any]) -> Dict[str, Any]:
    trace = result.get("metrics", {}).get("debug", {}).get("prediction_trace")
    if not trace:
        raise ValueError("D3 requires --save-prediction-trace for paired bootstrap.")
    return trace


def _bootstrap_delta(base: Dict[str, Any], candidate: Dict[str, Any], *, seed: int, reps: int) -> Dict[str, float]:
    keys = ("source_test_indices", "client_ids", "targets")
    if any(base.get(key) != candidate.get(key) for key in keys):
        raise ValueError("D3 traces do not share identical source indices, client partitions, and targets.")
    target = np.asarray(base["targets"], dtype=np.int64)
    pred_base = np.asarray(base["predictions"], dtype=np.int64)
    pred_candidate = np.asarray(candidate["predictions"], dtype=np.int64)
    labels = np.unique(target)
    rng = np.random.default_rng(seed)
    deltas = np.empty(reps, dtype=np.float64)
    for index in range(reps):
        sample = rng.integers(0, len(target), size=len(target))
        deltas[index] = f1_score(target[sample], pred_candidate[sample], labels=labels,
                                 average="macro", zero_division=0) - f1_score(
                                     target[sample], pred_base[sample], labels=labels,
                                     average="macro", zero_division=0)
    return {"mean": float(deltas.mean()), "ci95_low": float(np.quantile(deltas, .025)),
            "ci95_high": float(np.quantile(deltas, .975)), "replicates": int(reps)}


def _old_classes(manifest: Dict[str, Any]) -> set[int]:
    metadata = _read(Path(manifest["data_dir"]) / "metadata.json")
    task_classes = metadata["task_structure"]["task_classes"]
    return {int(label) for task, labels in task_classes.items()
            if int(task) < int(manifest["task_end"]) for label in labels}


def _mean_recall(metrics: Dict[str, Any], classes: set[int]) -> float:
    per_class = metrics["debug"]["per_class"]
    values = [float(per_class[str(label)]["accuracy"]) for label in sorted(classes) if str(label) in per_class]
    return float(np.mean(values)) if values else 0.0


def analyze(manifest_path: str | Path, *, bootstrap_replicates: int = 2000) -> Dict[str, Any]:
    manifest = _read(Path(manifest_path))
    variants = manifest.get("variants", {})
    missing = [name for name in REQUIRED if name not in variants]
    if missing:
        raise ValueError(f"D3 manifest lacks variants: {missing}")
    e3 = {name: _policy(variants[name], "e3_oracle_routed_system_ceiling") for name in REQUIRED}
    e4 = {name: _policy(variants[name], "e4_pred_hard") for name in REQUIRED}
    baseline_trace = _trace(e3["baseline"])
    baseline_metrics = e3["baseline"]["metrics"]
    old_classes = _old_classes(manifest)
    baseline_old_recall = _mean_recall(baseline_metrics, old_classes)
    report: Dict[str, Any] = {"manifest": str(manifest_path), "decision": "KEEP_BASELINE",
                              "variants": {}, "gates": {}}
    eligible = []
    for offset, name in enumerate(REQUIRED[1:], start=1):
        candidate_metrics = e3[name]["metrics"]
        bootstrap = _bootstrap_delta(baseline_trace, _trace(e3[name]),
                                     seed=int(manifest.get("seed", 42)) + offset,
                                     reps=bootstrap_replicates)
        old_delta = _mean_recall(candidate_metrics, old_classes) - baseline_old_recall
        report["variants"][name] = {
            "e3_f1_macro_delta": float(candidate_metrics["f1_macro"] - baseline_metrics["f1_macro"]),
            "e4_f1_macro_delta": float(e4[name]["metrics"]["f1_macro"] - e4["baseline"]["metrics"]["f1_macro"]),
            "paired_bootstrap": bootstrap, "old_class_recall_delta": old_delta,
            "minimum_per_class_recall": min(v["accuracy"] for v in candidate_metrics["debug"]["per_class"].values()),
        }
        if bootstrap["ci95_low"] > 0.0 and old_delta >= -0.01:
            eligible.append(name)
    report["gates"]["trace_alignment"] = True
    report["gates"]["bootstrap_positive_candidates"] = eligible
    if eligible:
        report["decision"] = "CANDIDATE_FOR_CONFIRMATION_SEED"
        report["recommended_candidate"] = max(eligible, key=lambda item: report["variants"][item]["e3_f1_macro_delta"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    report = analyze(args.manifest, bootstrap_replicates=args.bootstrap_replicates)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
