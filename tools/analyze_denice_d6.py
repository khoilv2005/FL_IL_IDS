"""Validate and summarize the two-branch D6 peer-aggregation ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


E3 = "e3_oracle_routed_system_ceiling"
E3B = "e3b_oracle_hard_no_adapter"
E4 = "e4_pred_hard"
REQUIRED = ("peer", "self_only")


def _read(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def analyze(manifest_path: str | Path) -> Dict[str, Any]:
    manifest_file = Path(manifest_path)
    manifest = _read(manifest_file)
    variants = manifest.get("variants") or {}
    missing = [name for name in REQUIRED if name not in variants]
    if missing:
        raise ValueError(f"D6 manifest missing variants: {missing}")
    rows: Dict[str, Dict[str, Any]] = {}
    support_hashes = set()
    for name in REQUIRED:
        summary = _read(Path(variants[name]["evaluation_summary"]))
        policies = (summary.get("summary") or {}).get("coverage_aware_local") or {}
        for policy in (E3, E3B, E4):
            if policy not in policies:
                raise ValueError(f"D6 {name}: missing {policy}")
        coverage = policies[E4].get("coverage_protocol") or {}
        if coverage.get("unsupported_sample_count") != 0 or coverage.get("assigned_sample_count") != coverage.get("requested_sample_count"):
            raise ValueError(f"D6 {name}: non-strict coverage")
        support_hash = (policies[E4].get("evaluation_sampling") or {}).get("source_index_sha256")
        if not support_hash:
            raise ValueError(f"D6 {name}: missing fixed-support hash")
        support_hashes.add(support_hash)
        rows[name] = {
            "e3_macro_f1": policies[E3].get("f1_macro"),
            "e3b_macro_f1": policies[E3B].get("f1_macro"),
            "e4_macro_f1": policies[E4].get("f1_macro"),
            "route_accuracy": policies[E4].get("route_accuracy"),
            "router_gap_pp": 100.0 * (float(policies[E3]["f1_macro"]) - float(policies[E4]["f1_macro"])),
            "coverage": coverage,
        }
    if len(support_hashes) != 1:
        raise ValueError("D6 variants do not share one fixed evaluation support")
    delta_e3_pp = 100.0 * (float(rows["self_only"]["e3_macro_f1"]) - float(rows["peer"]["e3_macro_f1"]))
    delta_e4_pp = 100.0 * (float(rows["self_only"]["e4_macro_f1"]) - float(rows["peer"]["e4_macro_f1"]))
    return {
        "manifest": str(manifest_file), "seed": manifest.get("seed"), "variants": rows,
        "fixed_support_source_index_sha256": support_hashes.pop(),
        "deltas_self_only_minus_peer_pp": {"e3_macro_f1": delta_e3_pp, "e4_macro_f1": delta_e4_pp},
        "decision": "PEER_HARM_CANDIDATE" if delta_e3_pp >= 1.0 else "NO_MATERIAL_PEER_HARM_ON_E3",
        "rule": "Peer harm requires self-only E3 macro-F1 at least 1.0 pp above peer under this strict matched protocol; confirmation remains required before a method change.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = analyze(args.manifest)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
