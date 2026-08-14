"""Fail-closed summary and stop-rule check for a D5 router-memory matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _coverage(summary: Dict[str, Any]) -> Dict[str, Any]:
    protocol = summary["summary"]["coverage_aware_local"]
    e3 = protocol["e3_oracle_routed_system_ceiling"]
    e4 = protocol["e4_pred_hard"]
    coverage = e4.get("coverage_protocol", {})
    if coverage.get("unsupported_sample_count") != 0:
        raise ValueError("D5 requires strict coverage; found unsupported samples")
    if coverage.get("assigned_sample_count") != coverage.get("requested_sample_count"):
        raise ValueError("D5 requires the complete fixed support")
    source_hash = (e4.get("evaluation_sampling") or {}).get("source_index_sha256")
    if not source_hash:
        raise ValueError("D5 evaluation lacks fixed-support source_index_sha256")
    return {"e3": e3, "e4": e4, "source_index_sha256": source_hash}


def analyze(manifest_path: Path) -> Dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    source_model_hash = manifest.get("source_model_state_sha256")
    if not source_model_hash:
        raise ValueError("D5 manifest lacks source_model_state_sha256")
    rows: List[Dict[str, Any]] = []
    support_hashes = set()
    for name, variant in sorted(manifest.get("variants", {}).items()):
        if variant.get("model_state_sha256") != source_model_hash:
            raise ValueError(f"{name} changed model tensors")
        summary_path = Path(variant["evaluation_dir"]) / "p6_evaluation_summary.json"
        coverage = _coverage(json.loads(summary_path.read_text(encoding="utf-8")))
        support_hashes.add(coverage["source_index_sha256"])
        e3_f1 = coverage["e3"].get("f1_macro")
        e4_f1 = coverage["e4"].get("f1_macro")
        if e3_f1 is None or e4_f1 is None:
            raise ValueError(f"{name} lacks E3/E4 macro-F1")
        profiles = variant.get("router_refresh_profiles", {})
        rows.append({
            "variant": name,
            "router_reference_per_class": int(variant["router_reference_per_class"]),
            "e3_macro_f1": float(e3_f1),
            "e4_macro_f1": float(e4_f1),
            "oracle_gap_pp": 100.0 * (float(e3_f1) - float(e4_f1)),
            "route_accuracy": coverage["e4"].get("route_accuracy"),
            "reference_input_mib": float(variant.get("router_reference_input_bytes", 0)) / 2**20,
            "activation_memory_mib": float(variant.get("router_activation_memory_bytes", 0)) / 2**20,
            "router_refresh_seconds": sum(float(profile.get("total_time", 0.0)) for profile in profiles.values()),
            "evaluation_wall_time_seconds": variant.get("evaluation_wall_time_seconds"),
        })
    if len(rows) < 2:
        raise ValueError("D5 needs at least two reference-memory budgets")
    if len(support_hashes) != 1:
        raise ValueError("D5 variants do not share one fixed evaluation support")
    rows.sort(key=lambda row: row["router_reference_per_class"])
    smallest = rows[0]
    better_gap = [row for row in rows[1:] if row["oracle_gap_pp"] < smallest["oracle_gap_pp"]]
    decision = (
        "FOLLOW_UP_FOR_ROUTER_MEMORY_GAIN"
        if better_gap
        else "RETAIN_SMALLEST_BUDGET_NO_ORACLE_GAP_REDUCTION"
    )
    return {
        "source_checkpoint": manifest.get("source_checkpoint"),
        "source_model_state_sha256": source_model_hash,
        "fixed_support_source_index_sha256": support_hashes.pop(),
        "rows": rows,
        "decision": decision,
        "rule": "Do not increase router memory unless E3−E4 decreases on the identical model and fixed support.",
        "note": "This is one seed; do not change the training configuration without the preregistered follow-up.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = analyze(Path(args.manifest))
    output = Path(args.output) if args.output else Path(args.manifest).with_name("d5_decision_report.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
