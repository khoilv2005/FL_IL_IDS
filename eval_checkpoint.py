"""Evaluate a saved training checkpoint.

Examples:
    python eval_checkpoint.py --checkpoint results/checkpoint_task_5.pt
    python eval_checkpoint.py --checkpoint results/checkpoint_task_2_round_19.pt --device cuda
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict

import torch

from fed_learning.data.incremental_loader import IncrementalDataLoader
from fed_learning.models.cnn_gru import CNN_GRU_Model
from fed_learning.models.der_model import DERModel
from fed_learning.models.nice_model import NICEModel
from fed_learning.servers.nice_server import ContextDetector
from fed_learning.strategies.incremental.nice import update_freeze_masks
from fed_learning.training.checkpoint_state import restore_context_detector
from fed_learning.training.der_worker import _reconstruct_model_structure
from fed_learning.training.local_task_loop import _evaluate_model


def _load_checkpoint(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


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


def evaluate_checkpoint(checkpoint_path: str, device: str | None = None) -> Dict[str, Any]:
    ckpt = _load_checkpoint(checkpoint_path)
    config = dict(ckpt["config"])
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = config["data_dir"]
    task_id = int(ckpt["task_id"])
    data_loader = IncrementalDataLoader(data_dir=data_dir)
    test_X, test_y = data_loader.get_test_data(task_id, cumulative=True)

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
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result = evaluate_checkpoint(args.checkpoint, device=args.device)
    print(json.dumps(result, indent=2))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
