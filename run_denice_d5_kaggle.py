"""D5: post-hoc DeNICE router-reference ablation without model retraining.

For every budget this reads one completed D4 checkpoint, reconstructs each
client's local router reference bank from the original training split, refits
only its router, and runs the fixed P6 evaluation.  A manifest proves that the
model tensors are byte-identical to the D4 source checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch

from eval_checkpoint import _make_denice_client_model
from fed_learning.data.incremental_loader import IncrementalDataLoader
from fed_learning.training.checkpoint_state import snapshot_context_detector
from fed_learning.training.denice_delta_checkpoint import load_denice_checkpoint
from fed_learning.training.denice_router_ablation import (
    model_states_sha256,
    rebuild_reference_input_memory,
)


SEED = int(os.environ.get("DENICE_SEED", "42"))
DATA_DIR = os.environ.get("DENICE_DATA_DIR", "/kaggle/input/datasets/khoilv2005/100-clients/100-clients")
REPO_DIR = Path(os.environ.get("DENICE_REPO_DIR", "/kaggle/working/FL_IL_IDS"))
CHECKPOINT = Path(os.environ.get("D5_CHECKPOINT", ""))
OUTPUT_ROOT = Path(os.environ.get("D5_OUTPUT_ROOT", f"/kaggle/working/denice_d5_seed_{SEED}"))
EVAL_DEVICE = os.environ.get("DENICE_EVAL_DEVICE", "cuda")
SAMPLES_PER_CLASS = int(os.environ.get("D5_SAMPLES_PER_CLASS", "100"))
REFERENCE_BUDGETS = tuple(
    int(item.strip()) for item in os.environ.get("D5_REFERENCE_BUDGETS", "20,50,100").split(",") if item.strip()
)


def _algorithm_state(ckpt: Dict[str, Any], client_id: int) -> Dict[str, Any]:
    states = ckpt["client_algorithm_states"]
    key: Any = client_id if client_id in states else str(client_id)
    state = states[key]
    return state.setdefault("denice", state)


def _write_router_checkpoint(source: Path, output: Path, budget: int) -> Dict[str, Any]:
    ckpt = load_denice_checkpoint(str(source))
    if str(ckpt.get("algorithm", "")).lower() != "denice":
        raise ValueError("D5 requires a full DeNICE checkpoint")
    source_model_hash = model_states_sha256(ckpt["client_model_states"])
    data_loader = IncrementalDataLoader(data_dir=str(DATA_DIR))
    task_id = int(ckpt["task_id"])
    round_id = ckpt.get("round_id", ckpt.get("final_round_id"))
    client_ids = [int(cid) for cid in ckpt.get("client_ids", ckpt["client_model_states"].keys())]
    router_rows: Dict[int, Dict[int, Dict[int, int]]] = {}
    refresh_profiles: Dict[int, Dict[str, float]] = {}
    reference_input_bytes = 0
    activation_memory_bytes = 0

    for client_id in client_ids:
        model, detector = _make_denice_client_model(ckpt, client_id, EVAL_DEVICE)
        episode_classes = detector.episode_classes
        if not episode_classes:
            raise ValueError(f"D5 cannot reconstruct router coverage for client {client_id}: missing episode_classes")
        memory, counts = rebuild_reference_input_memory(
            data_loader,
            client_id=client_id,
            episode_classes=episode_classes,
            budget_per_class=budget,
            seed=SEED,
        )
        if not memory:
            raise ValueError(f"D5 found no training references for client {client_id}")
        detector.router_reference_per_class = int(budget)
        detector.reference_input_memory = memory
        # Do not leave old sketches for an episode that has no rebuilt input.
        detector.activation_memory = {}
        profile = detector.refresh_activation_memory(
            model, task_id=task_id, round_id=None if round_id is None else int(round_id)
        )
        state = _algorithm_state(ckpt, client_id)
        state["context_detector"] = snapshot_context_detector(detector)
        router_rows[client_id] = counts
        refresh_profiles[client_id] = {key: float(value) for key, value in profile.items()}
        reference_input_bytes += sum(int(values.nbytes) for values in detector.reference_input_memory.values())
        activation_memory_bytes += sum(int(values.nbytes) for values in detector.activation_memory.values())
        del model

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, output)
    saved = load_denice_checkpoint(str(output))
    saved_model_hash = model_states_sha256(saved["client_model_states"])
    if source_model_hash != saved_model_hash:
        raise RuntimeError("D5 invariant failed: router checkpoint changed model tensors")
    return {
        "checkpoint": str(output),
        "checkpoint_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "model_state_sha256": saved_model_hash,
        "router_reference_per_class": int(budget),
        "reference_rows_by_client_episode_class": router_rows,
        "router_refresh_profiles": refresh_profiles,
        "router_reference_input_bytes": int(reference_input_bytes),
        "router_activation_memory_bytes": int(activation_memory_bytes),
    }


def main() -> None:
    if not REPO_DIR.is_dir() or not Path(DATA_DIR).is_dir() or not CHECKPOINT.is_file():
        raise FileNotFoundError("D5 needs existing DENICE_REPO_DIR, DENICE_DATA_DIR, and D5_CHECKPOINT.")
    if not REFERENCE_BUDGETS:
        raise ValueError("D5_REFERENCE_BUDGETS must contain at least one budget")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    source_sha256 = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    source_payload = load_denice_checkpoint(str(CHECKPOINT))
    source_model_hash = model_states_sha256(source_payload["client_model_states"])
    manifest: Dict[str, Any] = {
        "purpose": "D5 post-hoc router-reference ablation; no model training",
        "seed": SEED,
        "source_checkpoint": str(CHECKPOINT),
        "source_checkpoint_sha256": source_sha256,
        "source_model_state_sha256": source_model_hash,
        "data_dir": DATA_DIR,
        "reference_budgets": list(REFERENCE_BUDGETS),
        "fixed_evaluation": {"protocol": "coverage_aware_local", "samples_per_class": SAMPLES_PER_CLASS},
        "variants": {},
    }
    for budget in REFERENCE_BUDGETS:
        variant_dir = OUTPUT_ROOT / f"reference_{budget:03d}"
        checkpoint = variant_dir / "router_rebuilt_checkpoint.pt"
        evaluation_dir = variant_dir / "d5_evaluation"
        variant_manifest = variant_dir / "d5_variant_manifest.json"
        if checkpoint.is_file() and (evaluation_dir / "p6_evaluation_summary.json").is_file() and variant_manifest.is_file():
            details = json.loads(variant_manifest.read_text(encoding="utf-8"))
            details["reused"] = True
        else:
            details = _write_router_checkpoint(CHECKPOINT, checkpoint, budget)
            evaluation_start = time.perf_counter()
            subprocess.run([
                sys.executable, str(REPO_DIR / "run_denice_p6_eval.py"), "--checkpoint", str(checkpoint),
                "--data-dir", DATA_DIR, "--output-dir", str(evaluation_dir), "--device", EVAL_DEVICE,
                "--seed", str(SEED), "--protocols", "coverage_aware_local", "--samples-per-class",
                str(SAMPLES_PER_CLASS), "--class-balanced-with-replacement",
            ], check=True)
            details["evaluation_wall_time_seconds"] = float(time.perf_counter() - evaluation_start)
        if details["model_state_sha256"] != source_model_hash:
            raise RuntimeError("D5 invariant failed: a variant changed source model tensors")
        summary_path = evaluation_dir / "p6_evaluation_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"D5 missing strict P6 summary: {summary_path}")
        details["evaluation_dir"] = str(evaluation_dir)
        variant_manifest.write_text(json.dumps(details, indent=2), encoding="utf-8")
        manifest["variants"][f"reference_{budget:03d}"] = details
        (OUTPUT_ROOT / "d5_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    subprocess.run([
        sys.executable, str(REPO_DIR / "tools" / "analyze_denice_d5.py"), "--manifest",
        str(OUTPUT_ROOT / "d5_manifest.json"), "--output", str(OUTPUT_ROOT / "d5_decision_report.json"),
    ], check=True)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
