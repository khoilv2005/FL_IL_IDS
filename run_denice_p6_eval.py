"""Run and summarize the P6 DeNICE evaluation matrix for one final checkpoint.

Example (Kaggle)::

    python run_denice_p6_eval.py \
      --checkpoint /kaggle/working/results_seed_42/checkpoint_task_5_round_19.pt \
      --data-dir /kaggle/input/datasets/khoilv2005/100-clients/100-clients \
      --output-dir /kaggle/working/p6_seed_42 --device cuda --seed 42

This deliberately evaluates every policy on the same checkpoint, test index
seed and requested protocol.  It is an evaluation harness, not a training
shortcut: three independent training seeds still require three full runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from eval_checkpoint import evaluate_checkpoint


POLICIES: Dict[str, Dict[str, Any]] = {
    "e0_backbone_nomask": {"inference_policy": "backbone_nomask"},
    "e1_pred_adapter_nomask": {"inference_policy": "pred_adapter_nomask"},
    "e2_oracle_adapter_nomask": {"inference_policy": "oracle_adapter_nomask"},
    "e3_oracle_hard": {"inference_policy": "oracle_hard"},
    "e4_pred_hard": {"inference_policy": "pred_hard"},
    "e5_topk2": {"route_mode": "topk", "route_topk": 2},
    "e5_topk3": {"route_mode": "topk", "route_topk": 3},
    "e6_adaptive": {"route_mode": "adaptive", "route_topk": 2},
}


def _compact(result: Dict[str, Any]) -> Dict[str, Any]:
    metrics = result["metrics"]
    routing = metrics.get("routing_diagnostics", {})
    return {
        "accuracy": metrics.get("accuracy"),
        "f1_macro": metrics.get("f1_macro"),
        "f1_weighted": metrics.get("f1_weighted"),
        "loss": metrics.get("loss"),
        "route_accuracy": metrics.get("route_accuracy"),
        "route_coverage": metrics.get("route_coverage"),
        "oracle_mask_violation_count": metrics.get("oracle_mask_violation_count", 0),
        "adapter_active_sample_count": routing.get("adapter_active_sample_count"),
        "missing_adapter_sample_count": routing.get("missing_adapter_sample_count"),
        "checkpoint_sha256": result.get("checkpoint_sha256"),
        "config_sha256": result.get("config_sha256"),
        "eval_sample_count": result.get("eval_sample_count"),
        "eval_total_sample_count": result.get("eval_total_sample_count"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, required=True, help="Evaluation/test assignment seed")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--protocols",
        default="coverage_aware_local,representative_global",
        help="Comma-separated evaluation modes",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocols = [item.strip() for item in args.protocols.split(",") if item.strip()]
    results: Dict[str, Dict[str, Any]] = {}
    summary: Dict[str, Dict[str, Any]] = {}

    for protocol in protocols:
        results[protocol] = {}
        summary[protocol] = {}
        for name, overrides in POLICIES.items():
            result = evaluate_checkpoint(
                args.checkpoint,
                device=args.device,
                data_dir=args.data_dir,
                evaluation_mode=protocol,
                eval_seed=args.seed,
                max_samples=args.max_samples,
                router_mode="multiclass",
                route_mode=overrides.get("route_mode", "hard"),
                route_topk=int(overrides.get("route_topk", 1)),
                inference_policy=overrides.get("inference_policy"),
            )
            results[protocol][name] = result
            summary[protocol][name] = _compact(result)
            (output_dir / f"{protocol}_{name}.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )

    e3 = summary.get("coverage_aware_local", {}).get("e3_oracle_hard", {})
    e4 = summary.get("coverage_aware_local", {}).get("e4_pred_hard", {})
    summary["coverage_aware_oracle_gap"] = (
        None
        if e3.get("accuracy") is None or e4.get("accuracy") is None
        else float(e3["accuracy"] - e4["accuracy"])
    )
    payload = {
        "training_seed": int(args.seed),
        "checkpoint": str(args.checkpoint),
        "protocols": protocols,
        "policies": list(POLICIES),
        "summary": summary,
    }
    (output_dir / "p6_evaluation_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
