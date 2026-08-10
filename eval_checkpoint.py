"""Evaluate a saved training checkpoint.

Examples:
    python eval_checkpoint.py --checkpoint results/checkpoint_task_5.pt
    python eval_checkpoint.py --checkpoint results/checkpoint_task_2_round_19.pt --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict

import torch

from fed_learning.data.incremental_loader import IncrementalDataLoader
from fed_learning.models.cnn_gru import CNN_GRU_Model
from fed_learning.models.der_model import DERModel
from fed_learning.models.denice_model import DeNICEModel
from fed_learning.models.nice_model import NICEModel
from fed_learning.servers.nice_server import ContextDetector
from fed_learning.strategies.incremental.nice import update_freeze_masks
from fed_learning.training.checkpoint_state import restore_context_detector
from fed_learning.training.denice_delta_checkpoint import load_denice_checkpoint
from fed_learning.training.denice_eval import evaluate_denice_ensemble, evaluate_denice_model
from fed_learning.training.der_worker import _reconstruct_model_structure
from fed_learning.training.local_task_loop import _evaluate_model


def _load_checkpoint(path: str) -> Dict[str, Any]:
    checkpoint = load_denice_checkpoint(path)
    # Task-end resume checkpoints predate the delta schema and store ``task``
    # rather than ``task_id``. Normalize at the boundary so every evaluator
    # mode can consume both formats.
    if "task_id" not in checkpoint and "task" in checkpoint:
        checkpoint = dict(checkpoint)
        checkpoint["task_id"] = int(checkpoint["task"])
    return checkpoint

def _dict_get_int(mapping: Dict[Any, Any], key: int, default=None):
    if key in mapping:
        return mapping[key]
    text_key = str(key)
    if text_key in mapping:
        return mapping[text_key]
    return default

def _make_denice_client_model(
    ckpt: Dict[str, Any],
    client_id: int,
    device: str,
    router_mode: str | None = None,
):
    config = dict(ckpt["config"])
    input_shape = config["input_shape"]
    num_classes = config.get("num_classes", config.get("total_classes"))
    model = DeNICEModel(input_shape, num_classes)

    algorithm_states = ckpt.get("client_algorithm_states", {})
    client_alg = _dict_get_int(algorithm_states, int(client_id), {}) or {}
    denice_state = client_alg.get("denice", client_alg)

    adapter_registry = denice_state.get("adapter_registry", {}) or {}
    for meta in adapter_registry.values():
        layer_name = meta.get("layer_name")
        context_id = meta.get("context_id")
        rank = meta.get("rank")
        if layer_name is not None and context_id is not None:
            model.add_adapter(int(context_id), str(layer_name), rank=rank, set_active=False)

    neuron_ages = denice_state.get("neuron_ages")
    if neuron_ages:
        model.set_neuron_ages_state(neuron_ages)
        update_freeze_masks(model)
    freeze_masks = denice_state.get("freeze_masks")
    if freeze_masks:
        model.freeze_masks = freeze_masks
    recycling = denice_state.get("recycling_registry")
    if recycling and hasattr(model, "set_recycling_state"):
        model.set_recycling_state(recycling)

    state_dict = OrderedDict(
        _dict_get_int(ckpt["client_model_states"], int(client_id), {}) or {}
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"warning: client {client_id} missing keys: {len(missing)}")
    if unexpected:
        print(f"warning: client {client_id} unexpected keys: {len(unexpected)}")

    context_detector = ContextDetector(
        memo_per_class=int(config.get("memo_per_class", 50)),
        router_mode=str(router_mode or config.get("denice_router_mode", "chained")),
    )
    restore_context_detector(context_detector, denice_state.get("context_detector"))
    if router_mode is not None and context_detector.router_mode != str(router_mode).lower():
        context_detector.router_mode = str(router_mode).lower()
        if context_detector.activation_memory:
            context_detector.train_models(max(context_detector.activation_memory))

    model.to(device)
    model.eval()
    return model, context_detector


def _make_model(ckpt: Dict[str, Any], device: str):
    config = dict(ckpt["config"])
    algorithm = str(ckpt.get("algorithm") or config.get("algorithm", "")).lower()
    input_shape = config["input_shape"]
    num_classes = config.get("num_classes", config.get("total_classes"))
    state_dict = OrderedDict(ckpt["model_state_dict"])
    algorithm_state = ckpt.get("algorithm_state", {})

    context_detector = None

    if algorithm == "nice":
        model = NICEModel(input_shape, num_classes)
        nice_state = algorithm_state.get("nice", {})
        neuron_ages = nice_state.get("neuron_ages")
        if neuron_ages:
            model.set_neuron_ages_state(neuron_ages)
            update_freeze_masks(model)
        freeze_masks = nice_state.get("freeze_masks")
        if freeze_masks:
            model.freeze_masks = freeze_masks
        context_detector = ContextDetector(
            memo_per_class=int(config.get("memo_per_class", 50))
        )
        restore_context_detector(context_detector, nice_state.get("context_detector"))

    elif algorithm == "der":
        model = DERModel(input_shape, num_classes)
        der_state = algorithm_state.get("der", {})
        task_classes_history = der_state.get("task_classes_history", {})
        recon_config = {
            **config,
            "task_classes_history": task_classes_history,
            "s_max": der_state.get("s_max", config.get("s_max", 15.0)),
        }
        _reconstruct_model_structure(model, state_dict, recon_config)

    else:
        model = CNN_GRU_Model(input_shape, num_classes)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, context_detector


def evaluate_checkpoint(
    checkpoint_path: str,
    device: str | None = None,
    data_dir: str | None = None,
    route_mode: str = "hard",
    route_topk: int = 1,
    router_mode: str | None = None,
    evaluation_mode: str = "local",
    max_samples: int | None = None,
    eval_seed: int = 42,
) -> Dict[str, Any]:
    ckpt = _load_checkpoint(checkpoint_path)
    config = dict(ckpt["config"])
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = data_dir or config["data_dir"]
    config["data_dir"] = data_dir
    ckpt["config"] = config
    task_id = int(ckpt["task_id"])
    data_loader = IncrementalDataLoader(data_dir=data_dir)
    test_X, test_y = data_loader.get_test_data(task_id, cumulative=True)
    total_test_samples = int(len(test_y))
    if max_samples is not None and 0 < int(max_samples) < len(test_y):
        generator = torch.Generator().manual_seed(int(eval_seed) + int(task_id))
        indices = torch.randperm(len(test_y), generator=generator)[: int(max_samples)]
        test_X, test_y = test_X[indices], test_y[indices]

    if ckpt.get("algorithm") == "denice" and "client_model_states" in ckpt:
        client_ids = [
            int(cid) for cid in ckpt.get("client_ids", ckpt["client_model_states"].keys())
        ]
        evaluation_mode = str(evaluation_mode or "local").lower()
        if evaluation_mode not in {"local", "ensemble", "representative"}:
            raise ValueError("evaluation_mode must be local, ensemble, or representative")
        if evaluation_mode in {"ensemble", "representative"}:
            pairs = []
            selected_ids = client_ids
            if evaluation_mode == "representative":
                groups = (ckpt.get("cluster") or {}).get("groups", {})
                seen_groups = set()
                selected_ids = []
                for cid in client_ids:
                    group = tuple(sorted(set(int(x) for x in _dict_get_int(groups, int(cid), [cid]))))
                    if group in seen_groups:
                        continue
                    seen_groups.add(group)
                    selected_ids.append(min(group))
                selected_ids = [cid for cid in selected_ids if cid in client_ids]
            for cid in selected_ids:
                pairs.append(_make_denice_client_model(ckpt, cid, device, router_mode=router_mode))
            metrics = evaluate_denice_ensemble(
                pairs,
                {"X_test": test_X, "y_test": test_y},
                device=device,
                seen_classes=ckpt.get("seen_classes"),
                batch_size=int(config.get("eval_batch_size", 8192)),
                route_mode=route_mode,
                route_topk=route_topk,
            )
            return {
                "checkpoint": str(checkpoint_path),
                "checkpoint_type": ckpt.get("checkpoint_type"),
                "task_id": task_id,
                "round_id": ckpt.get("round_id", ckpt.get("final_round_id")),
                "algorithm": "denice",
                "evaluation_mode": evaluation_mode,
                "representative_client_ids": selected_ids,
                "metrics": metrics,
                "route_mode": route_mode,
                "route_topk": int(route_topk),
                "router_mode": router_mode or config.get("denice_router_mode", "chained"),
                "checkpoint_sha256": hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
                "config_sha256": hashlib.sha256(
                    json.dumps(config, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest(),
                "eval_sample_count": int(len(test_y)),
                "eval_total_sample_count": total_test_samples,
                "eval_seed": int(eval_seed),
            }

        per_client = []
        for cid in client_ids:
            model, context_detector = _make_denice_client_model(
                ckpt, cid, device, router_mode=router_mode
            )
            metrics = evaluate_denice_model(
                model,
                {"X_test": test_X, "y_test": test_y},
                device,
                context_detector=context_detector,
                seen_classes=ckpt.get("seen_classes"),
                batch_size=int(config.get("eval_batch_size", 8192)),
                route_mode=route_mode,
                route_topk=route_topk,
            )
            per_client.append({"client_id": cid, **metrics})
            del model
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()

        metric_keys = [
            "loss",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "f1_weighted",
        ]
        mean_metrics = {
            key: sum(float(row[key]) for row in per_client) / max(1, len(per_client))
            for key in metric_keys
        }
        return {
            "checkpoint": str(checkpoint_path),
            "checkpoint_type": ckpt.get("checkpoint_type"),
            "task_id": task_id,
            "round_id": ckpt.get("round_id", ckpt.get("final_round_id")),
            "algorithm": ckpt.get("algorithm", config.get("algorithm")),
            "mode": ckpt.get("mode", config.get("mode")),
            "eval_client_count": len(per_client),
            "metrics": mean_metrics,
            "per_client_metrics": per_client,
            "route_mode": route_mode,
            "route_topk": int(route_topk),
            "router_mode": router_mode or config.get("denice_router_mode", "chained"),
            "checkpoint_sha256": hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
            "config_sha256": hashlib.sha256(
                json.dumps(config, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest(),
            "eval_sample_count": int(len(test_y)),
            "eval_total_sample_count": total_test_samples,
            "eval_seed": int(eval_seed),
        }

    model, context_detector = _make_model(ckpt, device)
    metrics = _evaluate_model(
        model,
        {"X_test": test_X, "y_test": test_y},
        device,
        context_detector=context_detector,
        seen_classes=ckpt.get("seen_classes"),
    )
    return {
        "checkpoint": str(checkpoint_path),
        "task_id": task_id,
        "round_id": ckpt.get("round_id", ckpt.get("final_round_id")),
        "algorithm": ckpt.get("algorithm", config.get("algorithm")),
        "mode": ckpt.get("mode", config.get("mode")),
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--route-mode", default="hard", choices=["hard", "topk", "nomask"])
    parser.add_argument("--route-topk", type=int, default=1)
    parser.add_argument("--router-mode", default=None, choices=["chained", "multiclass"])
    parser.add_argument(
        "--evaluation-mode",
        default="local",
        choices=["local", "ensemble", "representative"],
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument(
        "--route-modes",
        default=None,
        help="Comma-separated ablation modes; emits one result per mode on identical checkpoint/data.",
    )
    args = parser.parse_args()

    modes = [args.route_mode]
    if args.route_modes:
        modes = [m.strip() for m in args.route_modes.split(",") if m.strip()]
    if any(m not in {"hard", "topk", "nomask"} for m in modes):
        parser.error("--route-modes accepts only hard,topk,nomask")
    results = {
        mode: evaluate_checkpoint(
            args.checkpoint,
            device=args.device,
            data_dir=args.data_dir,
            route_mode=mode,
            route_topk=args.route_topk,
            router_mode=args.router_mode,
            evaluation_mode=args.evaluation_mode,
            max_samples=args.max_samples,
            eval_seed=args.eval_seed,
        )
        for mode in modes
    }
    result = results[modes[0]] if len(modes) == 1 else {"ablations": results}
    print(json.dumps(result, indent=2))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
