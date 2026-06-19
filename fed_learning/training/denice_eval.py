"""
DeNICE context-aware inference with micro-adapters (plan section 10).

Test flow for each sample ``x``::

    1. Forward x through the (adapter-free) backbone for context activations.
    2. Binarize per layer.
    3. Context detector predicts the episode / context e_hat.
    4. Select the adapter(s) registered for e_hat (top-1 context, top-1 adapter).
    5. Forward with those adapters active.
    6. Mask logits to seen / context classes.
    7. argmax.

Routing uses the adapter-free activation path so the context detector keeps
predicting on the same signal it was trained on.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def _route_episodes(model, X_batch: torch.Tensor, context_detector) -> np.ndarray:
    """Predict the episode of every sample using adapter-free activations."""
    acts = model.get_context_activations_per_sample(X_batch)
    binary = context_detector.binarize_layer_activations(
        {name: act.detach().cpu().numpy() for name, act in acts.items()}
    )
    return context_detector.predict_episodes_batch(binary)


def _seen_unseen_mask(num_classes: int, seen_classes: List[int], device) -> torch.Tensor:
    mask = torch.ones(num_classes, dtype=torch.bool, device=device)
    for cls_id in set(int(c) for c in (seen_classes or [])):
        if 0 <= cls_id < num_classes:
            mask[cls_id] = False
    return mask


def _allowed_classes_for_episode(
    context_detector, episode: int, seen_classes: List[int], num_classes: int
) -> List[int]:
    seen_set = set(int(c) for c in (seen_classes or []))
    allowed = [
        int(c)
        for c in context_detector.episode_classes.get(int(episode), [])
        if int(c) in seen_set and 0 <= int(c) < num_classes
    ]
    if allowed:
        return sorted(set(allowed))
    return sorted(c for c in seen_set if 0 <= c < num_classes)


@torch.no_grad()
def _denice_predict_with_episodes(
    model,
    X_batch: torch.Tensor,
    context_detector,
    seen_classes: List[int],
    device: str,
) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
    """Predict class ids and return the routed episode per sample.

    The episode array is ``None`` when no context detector is available (so
    routing falls back to the plain seen-class mask).
    """
    model.eval()
    num_classes = int(model.num_classes)
    n = X_batch.shape[0]
    preds = torch.full((n,), -1, dtype=torch.long, device=device)

    has_detector = bool(getattr(context_detector, "episode_classes", None))

    if not has_detector:
        model.clear_active_adapters()
        out = model(X_batch)
        unseen = _seen_unseen_mask(num_classes, seen_classes, out.device)
        out[:, unseen] = float("-inf")
        return out.argmax(dim=1), None

    episodes = _route_episodes(model, X_batch, context_detector)
    unseen = _seen_unseen_mask(num_classes, seen_classes, device)

    for ep in np.unique(episodes):
        idx_np = np.where(episodes == ep)[0]
        idx = torch.as_tensor(idx_np, dtype=torch.long, device=device)
        model.set_active_context(int(ep))
        out = model(X_batch.index_select(0, idx))
        out[:, unseen] = float("-inf")
        allowed = _allowed_classes_for_episode(
            context_detector, int(ep), seen_classes, num_classes
        )
        if allowed:
            allowed_t = torch.as_tensor(allowed, dtype=torch.long, device=device)
            out[:, allowed_t] = out[:, allowed_t] + 99999.0
        preds[idx] = out.argmax(dim=1)

    model.clear_active_adapters()
    return preds, np.asarray(episodes)


def denice_predict_batch(
    model,
    X_batch: torch.Tensor,
    context_detector,
    seen_classes: List[int],
    device: str,
) -> torch.Tensor:
    """Return predicted class ids for a batch using context-routed adapters."""
    preds, _episodes = _denice_predict_with_episodes(
        model, X_batch, context_detector, seen_classes, device
    )
    return preds


def _label_to_episode_map(context_detector) -> Dict[int, int]:
    """Map each class id to its (latest) episode from the detector summary."""
    mapping: Dict[int, int] = {}
    for episode, classes in getattr(context_detector, "episode_classes", {}).items():
        for cls_id in classes:
            mapping[int(cls_id)] = int(episode)
    return mapping


def evaluate_denice_model(
    model,
    test_data: Dict[str, torch.Tensor],
    device: str,
    context_detector,
    seen_classes: List[int],
    batch_size: int = 1024,
    progress_label: Optional[str] = None,
    progress_every_batches: int = 0,
) -> Dict[str, float]:
    """Evaluate a DeNICE model with context-routed micro-adapters."""
    model.eval()
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]

    if len(y_test) == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
            "route_accuracy": 0.0,
            "route_coverage": 0.0,
        }

    criterion = nn.CrossEntropyLoss()
    num_classes = int(model.num_classes)
    all_preds: List[int] = []
    all_targets: List[int] = []
    total_loss = 0.0
    n_batches = int(np.ceil(len(y_test) / max(1, batch_size)))
    start_time = time.time()

    # Route accuracy = how often the context detector sends a sample to the
    # episode that actually introduced its class (plan section 12 metric).
    label2episode = _label_to_episode_map(context_detector)
    route_correct = 0
    route_total = 0

    for batch_idx, i in enumerate(range(0, len(y_test), batch_size), start=1):
        X_batch = X_test[i : i + batch_size].to(device)
        y_batch = y_test[i : i + batch_size].to(device)

        # Loss on the assigned-context model (no adapters) with global unseen mask.
        model.clear_active_adapters()
        with torch.no_grad():
            loss_out = model(X_batch)
            unseen = _seen_unseen_mask(num_classes, seen_classes, loss_out.device)
            loss_out[:, unseen] = float("-inf")
            total_loss += criterion(loss_out, y_batch).item() * len(y_batch)

        preds, episodes = _denice_predict_with_episodes(
            model, X_batch, context_detector, seen_classes, device
        )
        all_preds.extend(preds.detach().cpu().tolist())
        all_targets.extend(y_batch.detach().cpu().tolist())

        if episodes is not None and label2episode:
            y_np = y_batch.detach().cpu().numpy()
            true_eps = np.array(
                [label2episode.get(int(c), -1) for c in y_np], dtype=np.int64
            )
            known = true_eps >= 0
            route_total += int(known.sum())
            route_correct += int((episodes[known] == true_eps[known]).sum())

        if progress_label and progress_every_batches > 0:
            should_print = (
                batch_idx == 1
                or batch_idx == n_batches
                or batch_idx % progress_every_batches == 0
            )
            if should_print:
                elapsed = time.time() - start_time
                print(
                    f"      {progress_label}: batch {batch_idx}/{n_batches} "
                    f"({min(i + batch_size, len(y_test))}/{len(y_test)} samples), "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

    y_true = np.asarray(all_targets)
    y_pred = np.asarray(all_preds)
    return {
        "loss": total_loss / max(1, len(y_test)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "route_accuracy": (route_correct / route_total) if route_total > 0 else 0.0,
        "route_coverage": (route_total / len(y_test)) if len(y_test) > 0 else 0.0,
    }
