"""
Recompute f1_weighted from a saved checkpoint without retraining.

Example:
    python tools/recompute_f1_weighted_from_checkpoint.py \
        --checkpoint E:/result_of_NCKH/results/results_ewc/results_incremental/checkpoint_task_5.pt \
        --data-dir E:/path/to/100-clients \
        --output-dir E:/result_of_NCKH/results/results_ewc/results_incremental
"""

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fed_learning.data.incremental_loader import IncrementalDataLoader
from fed_learning.models.cnn_gru import CNN_GRU_Model


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _normalize_input_shape(input_shape: Any) -> Any:
    if isinstance(input_shape, list):
        return tuple(input_shape)
    return input_shape


def _infer_input_shape(config: Dict[str, Any], data_loader: IncrementalDataLoader) -> Any:
    return _normalize_input_shape(config.get("input_shape") or data_loader.input_shape)


def _infer_num_classes(config: Dict[str, Any], seen_classes: Iterable[int]) -> int:
    num_classes = (
        config.get("num_classes")
        or config.get("total_classes")
        or config.get("classes")
    )
    if num_classes is not None:
        return int(num_classes)
    seen = [int(c) for c in seen_classes]
    if not seen:
        raise ValueError("Cannot infer num_classes from checkpoint/config.")
    return max(seen) + 1


def _create_model(
    algorithm: str,
    input_shape: Any,
    num_classes: int,
    state_dict: Dict[str, torch.Tensor],
    config: Dict[str, Any],
    device: torch.device,
):
    algo = algorithm.lower()
    if algo == "der":
        from fed_learning.models.der_model import DERModel
        from fed_learning.training.der_worker import _reconstruct_model_structure

        model = DERModel(input_shape, num_classes).to(device)
        _reconstruct_model_structure(model, OrderedDict(state_dict), config)
        return model

    if algo == "nice":
        from fed_learning.models.nice_model import NICEModel

        return NICEModel(input_shape, num_classes).to(device)

    return CNN_GRU_Model(input_shape, num_classes).to(device)


def _load_state(model, state_dict: Dict[str, torch.Tensor], device: torch.device, strict: bool) -> Dict[str, Any]:
    prepared = OrderedDict((k, v.to(device)) for k, v in state_dict.items())
    try:
        model.load_state_dict(prepared, strict=strict)
        return {"strict": strict, "missing_keys": [], "unexpected_keys": []}
    except RuntimeError as exc:
        if strict:
            print("[WARN] strict=True load failed; retrying strict=False.")
            print(f"[WARN] {exc}")
            result = model.load_state_dict(prepared, strict=False)
            return {
                "strict": False,
                "missing_keys": list(result.missing_keys),
                "unexpected_keys": list(result.unexpected_keys),
                "strict_error": str(exc),
            }
        raise


def _apply_seen_class_mask(logits: torch.Tensor, seen_classes: Optional[List[int]]) -> torch.Tensor:
    if not seen_classes:
        return logits
    masked = logits.clone()
    mask = torch.ones(masked.shape[1], dtype=torch.bool, device=masked.device)
    for cls_id in seen_classes:
        if 0 <= int(cls_id) < masked.shape[1]:
            mask[int(cls_id)] = False
    masked[:, mask] = -1e9
    return masked


def _evaluate(
    model,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    device: torch.device,
    batch_size: int,
    seen_classes: Optional[List[int]],
    apply_seen_mask: bool,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    if len(y_test) == 0:
        raise ValueError("Test set is empty; cannot recompute metrics.")

    model.eval()
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for start in range(0, len(y_test), batch_size):
            X_batch = X_test[start : start + batch_size].to(device)
            y_batch = y_test[start : start + batch_size].to(device)
            logits = model(X_batch)
            if apply_seen_mask:
                logits = _apply_seen_class_mask(logits, seen_classes)
            total_loss += criterion(logits, y_batch).item()
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    zero_division = 0
    metrics = {
        "loss": float(total_loss / max(1, len(y_test))),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=zero_division)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=zero_division)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=zero_division)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=zero_division)),
    }
    return metrics, y_true, y_pred


def recompute_f1_weighted(
    checkpoint_path: str,
    data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    device: str = "auto",
    batch_size: Optional[int] = None,
    strict: bool = True,
    apply_seen_mask: bool = True,
) -> Dict[str, Any]:
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    resolved_device = _resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dict.")
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint missing required key: model_state_dict")

    config = dict(checkpoint.get("config") or {})
    algorithm = str(config.get("algorithm", "cnn_gru")).lower()
    task_id = int(checkpoint.get("task_id", config.get("task_id", 0)))
    seen_classes = [int(c) for c in checkpoint.get("seen_classes", [])]
    resolved_data_dir = data_dir or config.get("data_dir")
    if not resolved_data_dir:
        raise ValueError("data_dir was not provided and checkpoint config has no data_dir.")

    data_loader = IncrementalDataLoader(resolved_data_dir)
    input_shape = _infer_input_shape(config, data_loader)
    num_classes = _infer_num_classes(config, seen_classes)
    model_state = checkpoint["model_state_dict"]
    model = _create_model(algorithm, input_shape, num_classes, model_state, config, resolved_device)
    load_info = _load_state(model, model_state, resolved_device, strict=strict)

    X_test, y_test = data_loader.get_test_data(task_id, cumulative=True)
    eval_batch_size = int(batch_size or config.get("batch_size", 2048))
    metrics, y_true, y_pred = _evaluate(
        model=model,
        X_test=X_test,
        y_test=y_test,
        device=resolved_device,
        batch_size=eval_batch_size,
        seen_classes=seen_classes,
        apply_seen_mask=apply_seen_mask,
    )

    metrics_in_checkpoint = checkpoint.get("metrics", {})
    result = {
        "checkpoint": checkpoint_path,
        "algorithm": algorithm,
        "task_id": task_id,
        "seen_classes": seen_classes,
        "num_test_samples": int(len(y_test)),
        "metrics_recomputed": metrics,
        "metrics_in_checkpoint": metrics_in_checkpoint,
        "load_info": load_info,
        "apply_seen_mask": bool(apply_seen_mask),
        "unique_predicted_classes": [int(x) for x in np.unique(y_pred).tolist()],
        "unique_true_classes": [int(x) for x in np.unique(y_true).tolist()],
        "note": "evaluation only, no retraining",
    }

    resolved_output_dir = output_dir or os.path.dirname(checkpoint_path)
    os.makedirs(resolved_output_dir, exist_ok=True)
    out_path = os.path.join(resolved_output_dir, f"f1_weighted_recomputed_task_{task_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    result["output_path"] = out_path
    return result


def _print_summary(result: Dict[str, Any]) -> None:
    old = result.get("metrics_in_checkpoint", {}) or {}
    new = result["metrics_recomputed"]
    print("\n" + "=" * 72)
    print("Recomputed f1_weighted from checkpoint")
    print("=" * 72)
    print(f"Checkpoint: {result['checkpoint']}")
    print(f"Algorithm:  {result['algorithm']}")
    print(f"Task:       {result['task_id']}")
    print(f"Samples:    {result['num_test_samples']}")
    print(f"Output:     {result['output_path']}")
    print("\nCheckpoint metrics:")
    for key in ["loss", "accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]:
        if key in old:
            print(f"  {key}: {old.get(key)}")
    print("\nRecomputed metrics:")
    for key in ["loss", "accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]:
        print(f"  {key}: {new.get(key)}")
    if result.get("load_info", {}).get("strict") is False:
        print("\n[WARN] Model was loaded with strict=False. Inspect load_info in JSON.")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute f1_weighted from checkpoint_task.pt without training.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint_task_X.pt or checkpoint_task_X_round_Y.pt")
    parser.add_argument("--data-dir", default=None, help="Path to federated split data directory. Defaults to checkpoint config.")
    parser.add_argument("--output-dir", default=None, help="Where to save f1_weighted_recomputed_task_<id>.json")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', or CUDA device such as 'cuda:0'")
    parser.add_argument("--batch-size", type=int, default=None, help="Evaluation batch size. Defaults to checkpoint config batch_size.")
    parser.add_argument("--no-seen-mask", action="store_true", help="Do not mask unseen output classes during evaluation.")
    parser.add_argument("--non-strict", action="store_true", help="Load model with strict=False immediately.")
    args = parser.parse_args()

    result = recompute_f1_weighted(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        strict=not args.non_strict,
        apply_seen_mask=not args.no_seen_mask,
    )
    _print_summary(result)


if __name__ == "__main__":
    main()
