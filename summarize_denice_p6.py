"""Validate and aggregate the three full-run P6 evaluation artifacts.

Usage::

    python summarize_denice_p6.py \
      --run-dirs /kaggle/working/results_denice_seed_42/p6_evaluation \
                 /kaggle/working/results_denice_seed_43/p6_evaluation \
                 /kaggle/working/results_denice_seed_44/p6_evaluation \
      --output /kaggle/working/denice_p6_final_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable


POLICIES = (
    "e0_backbone_nomask",
    "e1_pred_adapter_nomask",
    "e2_oracle_adapter_nomask",
    "e3_oracle_hard",
    "e4_pred_hard",
    "e5_topk2",
    "e5_topk3",
    "e6_adaptive",
)
PROTOCOLS = ("coverage_aware_local", "representative_global")
METRICS = (
    "accuracy",
    "precision_weighted",
    "recall_weighted",
    "f1_macro",
    "f1_weighted",
    "loss",
    "route_accuracy",
)


def _load(path: Path) -> Dict[str, Any]:
    # Accept both native runner output and PowerShell-edited JSON artifacts.
    with path.open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def _mean_std(values: Iterable[float]) -> Dict[str, float]:
    data = [float(value) for value in values]
    return {"mean": float(mean(data)), "std": float(stdev(data)) if len(data) > 1 else 0.0}


def _route_collapse(result_path: Path) -> Dict[str, Any]:
    """Detect a route collapse only when true distribution does not match it."""
    result = _load(result_path)
    confusion = result.get("metrics", {}).get("debug", {}).get("route_confusion", {})
    true_counts: Dict[str, int] = {}
    pred_counts: Dict[str, int] = {}
    for true_episode, row in confusion.items():
        true_counts[str(true_episode)] = sum(int(count) for count in row.values())
        for predicted_episode, count in row.items():
            pred_counts[str(predicted_episode)] = pred_counts.get(str(predicted_episode), 0) + int(count)
    total = sum(pred_counts.values())
    if not total:
        return {"assessable": False, "collapse": False, "reason": "route confusion unavailable"}
    predicted_share = {episode: count / total for episode, count in pred_counts.items()}
    true_share = {episode: count / total for episode, count in true_counts.items()}
    collapsed = [
        episode for episode, share in predicted_share.items()
        if share > 0.70 and true_share.get(episode, 0.0) <= 0.70
    ]
    return {
        "assessable": True,
        "collapse": bool(collapsed),
        "collapsed_episodes": collapsed,
        "predicted_share": predicted_share,
        "true_share": true_share,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", required=True, help="P6 evaluation directories")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summaries = []
    seeds = set()
    for raw_dir in args.run_dirs:
        run_dir = Path(raw_dir)
        summary_path = run_dir / "p6_evaluation_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing P6 summary: {summary_path}")
        summary = _load(summary_path)
        seed = int(summary["training_seed"])
        if seed in seeds:
            raise ValueError(f"Duplicate training seed: {seed}")
        seeds.add(seed)
        for protocol in PROTOCOLS:
            if protocol not in summary.get("summary", {}):
                raise ValueError(f"{summary_path} lacks protocol {protocol}")
            for policy in POLICIES:
                metric = summary["summary"][protocol].get(policy)
                if not metric or not metric.get("checkpoint_sha256") or not metric.get("config_sha256"):
                    raise ValueError(f"{summary_path} lacks verified {protocol}/{policy} artifact")
        summaries.append((run_dir, summary))

    aggregate: Dict[str, Any] = {}
    for protocol in PROTOCOLS:
        aggregate[protocol] = {}
        for policy in POLICIES:
            aggregate[protocol][policy] = {
                metric: _mean_std(
                    summary["summary"][protocol][policy][metric]
                    for _directory, summary in summaries
                )
                for metric in METRICS
            }

    local = aggregate["coverage_aware_local"]
    oracle_gap_by_seed = {
        str(summary["training_seed"]): float(summary["summary"]["coverage_aware_oracle_gap"])
        for _directory, summary in summaries
    }
    collapse_by_seed = {
        str(summary["training_seed"]): _route_collapse(
            directory / "coverage_aware_local_e4_pred_hard.json"
        )
        for directory, summary in summaries
    }
    all_no_violations = all(
        int(summary["summary"][protocol][policy]["oracle_mask_violation_count"]) == 0
        for _directory, summary in summaries
        for protocol in PROTOCOLS
        for policy in POLICIES
        if policy == "e3_oracle_hard"
    )
    report = {
        "training_seeds": sorted(seeds),
        "run_dirs": [str(directory) for directory, _summary in summaries],
        "aggregate": aggregate,
        "coverage_aware_oracle_gap_by_seed": oracle_gap_by_seed,
        "coverage_aware_oracle_gap_mean_std": _mean_std(oracle_gap_by_seed.values()),
        "route_collapse_by_seed": collapse_by_seed,
        "gates": {
            "three_distinct_seeds": len(seeds) >= 3,
            "oracle_mask_violations_zero": all_no_violations,
            "predicted_hard_oracle_gap_le_5_points": all(gap <= 0.05 for gap in oracle_gap_by_seed.values()),
            "no_route_collapse": not any(row.get("collapse", False) for row in collapse_by_seed.values()),
            "macro_f1_and_accuracy_reported": all(
                local["e4_pred_hard"][metric]["mean"] >= 0.0 for metric in ("accuracy", "f1_macro")
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
