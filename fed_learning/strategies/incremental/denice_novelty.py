"""
DeNICE novelty estimation (plan section 4).

Novelty measures how different the new task is from previously seen tasks, per
layer, using binary activation prototypes - the same activation signature NICE
uses for its context detector.

Procedure (plan section 4)::

    1. Run task data through the model.
    2. Read conv1/conv2/conv3/gru activations.
    3. Binarize each layer activation (threshold = mean_l + std_l from task 0).
    4. Average binary vectors per layer  -> P_new,l
    5. Cosine-compare P_new,l with every stored old prototype P_old,e,l.
    6. novelty_l = 1 - max_e cos(P_new,l, P_old,e,l)
       novelty   = sum_l w_l * novelty_l   (weights renormalized over available layers)

Layer weights (plan)::

    w_conv1 = 0.15, w_conv2 = 0.20, w_conv3 = 0.25, w_gru = 0.25
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

NOVELTY_LAYERS: List[str] = ["conv1", "conv2", "conv3", "gru"]

DEFAULT_LAYER_WEIGHTS: Dict[str, float] = {
    "conv1": 0.15,
    "conv2": 0.20,
    "conv3": 0.25,
    "gru": 0.25,
}


def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class NoveltyEstimator:
    """Tracks per-layer binary prototypes and computes task novelty.

    Prototypes are stored per ``task_id`` (used as the context id in the MVP) and
    per layer. Binarization thresholds are calibrated once from task 0, mirroring
    the NICE context detector (mean + std per layer).
    """

    def __init__(self, layer_weights: Optional[Dict[str, float]] = None):
        self.layer_weights = dict(layer_weights or DEFAULT_LAYER_WEIGHTS)
        self.thresholds: Optional[Dict[str, float]] = None
        # prototypes[task_id][layer] -> np.ndarray
        self.prototypes: Dict[int, Dict[str, np.ndarray]] = {}

    # ------------------------------------------------------------------
    def calibrate_thresholds(self, model, data: torch.Tensor) -> Dict[str, float]:
        """Set per-layer thresholds = mean + std from task-0 activations."""
        acts = model.get_context_activations_per_sample(data)
        self.thresholds = {}
        for name in NOVELTY_LAYERS:
            act = acts[name].detach().cpu()
            self.thresholds[name] = float((act.mean() + act.std()).item())
        return self.thresholds

    # ------------------------------------------------------------------
    def compute_prototype(self, model, data: torch.Tensor) -> Dict[str, np.ndarray]:
        """Compute per-layer binary activation prototype P_new,l for ``data``."""
        acts = model.get_context_activations_per_sample(data)
        proto: Dict[str, np.ndarray] = {}
        for name in NOVELTY_LAYERS:
            act = acts[name].detach().cpu().numpy()  # [n_samples, dim]
            if self.thresholds is not None and name in self.thresholds:
                binary = (act > self.thresholds[name]).astype(np.float32)
            else:
                binary = (act > 0).astype(np.float32)
            proto[name] = binary.mean(axis=0)  # [dim]
        return proto

    # ------------------------------------------------------------------
    def store_prototype(self, task_id: int, proto: Dict[str, np.ndarray]) -> None:
        self.prototypes[int(task_id)] = {k: np.asarray(v) for k, v in proto.items()}

    def has_history(self) -> bool:
        return len(self.prototypes) > 0

    # ------------------------------------------------------------------
    def novelty_from_prototype(
        self, proto: Dict[str, np.ndarray], exclude_task: Optional[int] = None
    ) -> Dict[str, float]:
        """Compute novelty (overall + per-layer) of ``proto`` vs stored history.

        Returns a dict with ``novelty`` and ``novelty_<layer>`` keys. If no
        history exists, novelty is 1.0 (fully novel), matching the plan's task-0
        convention (``nu = 1.0`` / undefined).
        """
        old_tasks = [
            t for t in self.prototypes if exclude_task is None or t != int(exclude_task)
        ]
        if not old_tasks:
            return {"novelty": 1.0, **{f"novelty_{l}": 1.0 for l in NOVELTY_LAYERS}}

        per_layer: Dict[str, float] = {}
        for name in NOVELTY_LAYERS:
            if name not in proto:
                continue
            best_sim = -1.0
            for t in old_tasks:
                old = self.prototypes[t].get(name)
                if old is None or old.shape != proto[name].shape:
                    continue
                best_sim = max(best_sim, _cosine(proto[name], old))
            if best_sim < -0.5:  # no comparable old prototype
                per_layer[name] = 1.0
            else:
                per_layer[name] = float(1.0 - best_sim)

        # Weighted average over available layers (renormalized).
        total_w = 0.0
        acc = 0.0
        for name, nov in per_layer.items():
            w = float(self.layer_weights.get(name, 0.0))
            acc += w * nov
            total_w += w
        novelty = acc / total_w if total_w > 0 else (
            float(np.mean(list(per_layer.values()))) if per_layer else 1.0
        )

        out = {"novelty": float(novelty)}
        for name in NOVELTY_LAYERS:
            out[f"novelty_{name}"] = float(per_layer.get(name, 1.0))
        return out

    def compute_novelty(
        self, model, data: torch.Tensor, exclude_task: Optional[int] = None
    ) -> Dict[str, float]:
        """Convenience: compute prototype then novelty vs stored history."""
        proto = self.compute_prototype(model, data)
        return self.novelty_from_prototype(proto, exclude_task=exclude_task)
