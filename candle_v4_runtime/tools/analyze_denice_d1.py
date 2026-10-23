"""Apply the preregistered D1-to-D2 decision gate to completed artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


REQUIRED_VARIANTS = ("peer_default", "self_only", "peer_self_floor_050")
E3 = "e3_oracle_routed_system_ceiling"
E4 = "e4_pred_hard"


def _read(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _policy(summary_path: Path, policy: str) -> Dict[str, Any]:
    summary = _read(summary_path)
    rows = (summary.get("summary") or {}).get("coverage_aware_local") or {}
    if policy not in rows:
        raise ValueError(f"{summary_path}: missing {policy}")
    return rows[policy]


def _row_drift_evidence(run_dir: Path) -> Dict[str, Any]:
    history_path = run_dir / "cluster_history.json"
    if not history_path.exists():
        return {"available": False, "reason": "missing cluster_history.json"}
    supported_drifts: List[float] = []
    unsupported_drifts: List[float] = []
    unsupported_rows = 0
    for round_row in _read(history_path):
        for rows in (round_row.get("plastic_fc2_row_audit") or {}).values():
            for row in rows:
                drift = float(row.get("row_drift_l2", 0.0))
                if float(row.get("peer_alpha_supported", 0.0)) > 0.0:
                    supported_drifts.append(drift)
                if float(row.get("peer_alpha_unsupported", 0.0)) > 0.0:
                    unsupported_drifts.append(drift)
                    unsupported_rows += 1
    if not unsupported_drifts:
        return {
            "available": bool(supported_drifts),
            "unsupported_row_count": 0,
            "concentrated_harm_evidence": False,
            "reason": "no plastic row received unsupported-peer alpha",
        }
    supported_mean = sum(supported_drifts) / max(1, len(supported_drifts))
    unsupported_mean = sum(unsupported_drifts) / len(unsupported_drifts)
    return {
        "available": True,
        "unsupported_row_count": int(unsupported_rows),
        "supported_row_drift_mean": supported_mean,
        "unsupported_row_drift_mean": unsupported_mean,
        # Drift alone does not prove accuracy harm, but it is a necessary
        # mechanistic indicator for the D2 gate.
        "concentrated_harm_evidence": bool(unsupported_mean > supported_mean),
    }


def _variant_metrics(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    variants = manifest.get("variants") or {}
    missing = [name for name in REQUIRED_VARIANTS if name not in variants]
    if missing:
        raise ValueError(f"D1 manifest missing variants: {missing}")
    result: Dict[str, Dict[str, Any]] = {}
    for name in REQUIRED_VARIANTS:
        item = variants[name]
        summary_path = Path(item["evaluation_summary"])
        e3, e4 = _policy(summary_path, E3), _policy(summary_path, E4)
        coverage = e3.get("coverage_protocol") or {}
        result[name] = {
            "run_dir": str(Path(item["checkpoint"]).parent),
            "e3_accuracy": e3.get("accuracy"),
            "e3_f1_macro": e3.get("f1_macro"),
            "e4_accuracy": e4.get("accuracy"),
            "e4_f1_macro": e4.get("f1_macro"),
            "source_index_sha256": (e3.get("evaluation_sampling") or {}).get("source_index_sha256"),
            "coverage": coverage,
        }
    return result


def analyze_d1(
    manifest_path: str | Path,
    *,
    confirmation_manifest_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Return the audit plan's D2 gate decision; never infer missing evidence."""
    manifest_file = Path(manifest_path)
    manifest = _read(manifest_file)
    metrics = _variant_metrics(manifest)
    errors: List[str] = []
    hashes = {row["source_index_sha256"] for row in metrics.values()}
    if len(hashes) != 1 or None in hashes:
        errors.append("variants do not share one fixed evaluation source-index hash")
    for name, row in metrics.items():
        coverage = row["coverage"]
        if (
            coverage.get("partial_coverage") is not False
            or coverage.get("unsupported_sample_count") != 0
            or coverage.get("assigned_sample_count") != coverage.get("requested_sample_count")
        ):
            errors.append(f"{name}: strict coverage evidence is incomplete")

    peer, self_only = metrics["peer_default"], metrics["self_only"]
    f1_delta = float(self_only["e3_f1_macro"]) - float(peer["e3_f1_macro"])
    material_negative_transfer = f1_delta >= 0.01
    if not material_negative_transfer:
        errors.append("peer-default is not at least 1.0 pp below self-only on E3 macro-F1")

    row_evidence = _row_drift_evidence(Path(peer["run_dir"]))
    if not row_evidence.get("concentrated_harm_evidence", False):
        errors.append("no concentrated unsupported-row drift evidence")

    confirmation: Dict[str, Any] = {"provided": confirmation_manifest_path is not None, "passes": False}
    if confirmation_manifest_path is not None:
        confirm = analyze_d1(confirmation_manifest_path)
        confirmation["seed"] = _read(Path(confirmation_manifest_path)).get("seed")
        confirmation["passes"] = bool(
            confirm["conditions"]["material_negative_transfer"]
            and confirm["conditions"]["strict_protocol"]
        )
        if not confirmation["passes"]:
            errors.append("confirmation manifest does not reproduce protocol-correct negative transfer")
    else:
        errors.append("missing seed-43/bootstrap confirmation")

    return {
        "manifest": str(manifest_file),
        "seed": manifest.get("seed"),
        "variants": metrics,
        "row_drift_evidence": row_evidence,
        "confirmation": confirmation,
        "conditions": {
            "strict_protocol": not any("coverage" in error or "source-index" in error for error in errors),
            "material_negative_transfer": material_negative_transfer,
            "unsupported_row_harm": bool(row_evidence.get("concentrated_harm_evidence", False)),
            "confirmed": bool(confirmation["passes"]),
        },
        "d2_eligible": not errors,
        "decision": "OPEN_D2" if not errors else "KEEP_D2_CLOSED",
        "reasons": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--confirmation-manifest", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = analyze_d1(args.manifest, confirmation_manifest_path=args.confirmation_manifest)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
