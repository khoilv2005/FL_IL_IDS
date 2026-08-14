"""Fail-closed decision report for D2 selective plastic-fc2 aggregation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


E3 = "e3_oracle_routed_system_ceiling"
E4 = "e4_pred_hard"
BASELINE = "peer_default"
CANDIDATE = "peer_supported_fc2"


def _read(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: Iterable[float]) -> float | None:
    rows = [float(value) for value in values if value is not None]
    return None if not rows else float(sum(rows) / len(rows))


def _blocked_classes(history_path: Path) -> Set[int]:
    blocked: Set[int] = set()
    for round_row in _read(history_path):
        protection = round_row.get("selective_fc2_row_protection") or {}
        for rows in (protection.get("clients") or {}).values():
            for row in rows:
                if row.get("blocked"):
                    blocked.add(int(row["class_id"]))
    return blocked


def _policy(summary_path: Path, name: str) -> Dict[str, Any]:
    policies = (_read(summary_path).get("summary") or {}).get("coverage_aware_local") or {}
    if name not in policies:
        raise ValueError(f"{summary_path}: missing {name}")
    return policies[name]


def analyze(manifest_path: str | Path) -> Dict[str, Any]:
    manifest_file = Path(manifest_path)
    manifest = _read(manifest_file)
    variants = manifest.get("variants") or {}
    if set((BASELINE, CANDIDATE)) - set(variants):
        raise ValueError("D2 manifest requires peer_default and peer_supported_fc2")
    rows: Dict[str, Dict[str, Any]] = {}
    support_hashes = set()
    per_class: Dict[str, Dict[int, float]] = {}
    for name in (BASELINE, CANDIDATE):
        e3, e4 = _policy(Path(variants[name]["evaluation_summary"]), E3), _policy(Path(variants[name]["evaluation_summary"]), E4)
        coverage = e4.get("coverage_protocol") or {}
        if coverage.get("unsupported_sample_count") != 0 or coverage.get("assigned_sample_count") != coverage.get("requested_sample_count"):
            raise ValueError(f"D2 {name}: strict coverage failed")
        support_hash = (e4.get("evaluation_sampling") or {}).get("source_index_sha256")
        if not support_hash:
            raise ValueError(f"D2 {name}: missing source-index hash")
        support_hashes.add(support_hash)
        per_class[name] = {int(cls): float(value) for cls, value in (e4.get("per_class_recall") or {}).items() if value is not None}
        rows[name] = {"e3_macro_f1": float(e3["f1_macro"]), "e4_macro_f1": float(e4["f1_macro"]),
                      "route_accuracy": e4.get("route_accuracy"), "coverage": coverage}
    if len(support_hashes) != 1:
        raise ValueError("D2 variants do not share fixed evaluation support")
    blocked = _blocked_classes(Path(variants[CANDIDATE]["cluster_history"]))
    classes = set(per_class[BASELINE]) & set(per_class[CANDIDATE])
    blocked &= classes
    supported = classes - blocked
    old = {class_id for class_id in classes if class_id < 30}
    def delta_for(class_ids: Set[int]) -> float | None:
        return _mean(per_class[CANDIDATE][class_id] - per_class[BASELINE][class_id] for class_id in class_ids)
    deltas = {"e3_macro_f1_pp": 100.0 * (rows[CANDIDATE]["e3_macro_f1"] - rows[BASELINE]["e3_macro_f1"]),
              "e4_macro_f1_pp": 100.0 * (rows[CANDIDATE]["e4_macro_f1"] - rows[BASELINE]["e4_macro_f1"]),
              "blocked_class_recall_pp": None if delta_for(blocked) is None else 100.0 * delta_for(blocked),
              "supported_class_recall_pp": None if delta_for(supported) is None else 100.0 * delta_for(supported),
              "old_class_recall_pp": None if delta_for(old) is None else 100.0 * delta_for(old)}
    gate = {
        "e3_improves": deltas["e3_macro_f1_pp"] > 0.0,
        "blocked_class_recall_improves": deltas["blocked_class_recall_pp"] is not None and deltas["blocked_class_recall_pp"] > 0.0,
        "supported_class_recall_not_worse_than_1pp": deltas["supported_class_recall_pp"] is not None and deltas["supported_class_recall_pp"] >= -1.0,
        "old_class_recall_not_worse_than_1pp": deltas["old_class_recall_pp"] is not None and deltas["old_class_recall_pp"] >= -1.0,
    }
    return {"manifest": str(manifest_file), "seed": manifest.get("seed"), "variants": rows,
            "fixed_support_source_index_sha256": support_hashes.pop(), "blocked_class_ids": sorted(blocked),
            "deltas_candidate_minus_baseline_pp": deltas, "gate": gate,
            "decision": "D2_CANDIDATE_FOR_CONFIRMATION_SEED" if all(gate.values()) else "REJECT_D2_SEED",
            "rule": "D2 needs E3 and blocked-class recall improvement, with no supported/old-class recall regression over 1 pp; a second seed remains mandatory."}


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
