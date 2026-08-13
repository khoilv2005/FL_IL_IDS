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


def _route_episodes_with_scores(
    model, X_batch: torch.Tensor, context_detector
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Route + return ``(top1_episodes, chain_probs)`` using adapter-free acts.

    ``chain_probs`` is ``[n, n_episodes]`` (episode index == column index) so a
    top-k routing rule can ``argsort`` it directly. Falls back to ``(preds,
    None)`` when the detector cannot produce per-episode scores.
    """
    acts = model.get_context_activations_per_sample(X_batch)
    binary = context_detector.binarize_layer_activations(
        {name: act.detach().cpu().numpy() for name, act in acts.items()}
    )
    if hasattr(context_detector, "predict_episodes_with_scores"):
        preds, probs = context_detector.predict_episodes_with_scores(binary)
        return np.asarray(preds), np.asarray(probs)
    return np.asarray(context_detector.predict_episodes_batch(binary)), None


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


def _mask_logits_to_classes(
    logits: torch.Tensor,
    allowed_classes: List[int],
    fill_value: float = -100.0,
) -> torch.Tensor:
    """Keep only allowed classes active for routed prediction and loss."""
    if not allowed_classes:
        return logits
    mask = torch.ones(logits.shape[1], dtype=torch.bool, device=logits.device)
    allowed_t = torch.as_tensor(allowed_classes, dtype=torch.long, device=logits.device)
    mask[allowed_t] = False
    logits[:, mask] = fill_value
    return logits


INFERENCE_POLICIES = {
    "pred_hard",
    "pred_adapter_nomask",
    "oracle_adapter_nomask",
    "oracle_hard",
    "oracle_hard_no_adapter",
    "backbone_nomask",
}


def _increment_route_diagnostic(
    diagnostics: Optional[Dict[str, int]], key: str, count: int
) -> None:
    """Mutably collect compact routing facts when an evaluator requests them."""
    if diagnostics is not None:
        diagnostics[key] = int(diagnostics.get(key, 0)) + int(count)


def _validate_oracle_episodes(
    oracle_episodes: Optional[np.ndarray], n: int
) -> np.ndarray:
    if oracle_episodes is None:
        raise ValueError("oracle inference policy requires one episode id per sample")
    result = np.asarray(oracle_episodes, dtype=np.int64).reshape(-1)
    if len(result) != n:
        raise ValueError(f"oracle episode count {len(result)} != batch size {n}")
    return result


@torch.no_grad()
def _denice_routed_logits_with_episodes(
    model,
    X_batch: torch.Tensor,
    context_detector,
    seen_classes: List[int],
    device: str,
    route_mode: str = "hard",
    route_topk: int = 1,
    inference_policy: Optional[str] = None,
    oracle_episodes: Optional[np.ndarray] = None,
    routing_diagnostics: Optional[Dict[str, int]] = None,
    adaptive_high_confidence: float = 0.75,
    adaptive_low_confidence: float = 0.45,
) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
    """Return logits from the same routed path used for DeNICE prediction.

    ``route_mode`` controls how routing translates into the class mask:

    - ``"hard"`` (default): top-1 episode; classes outside that episode are
      masked to ``-100`` (original DeNICE behaviour, unchanged).
    - ``"topk"``: keep the union of allowed classes over the sample's top-k
      episodes (``route_topk``). The adapter still follows the top-1 episode.
    - ``"nomask"``: diagnostic upper bound - predict over all seen classes with
      no episode masking (routing is still computed for the metric/adapter).

    In every mode the returned episode array is the top-1 route, so
    ``route_accuracy`` stays comparable across modes.
    """
    policy = str(inference_policy or "").lower()
    if policy and policy not in INFERENCE_POLICIES:
        raise ValueError(f"Unknown DeNICE inference policy: {policy}")

    model.eval()
    num_classes = int(model.num_classes)
    n = X_batch.shape[0]
    routed_logits = torch.full(
        (n, num_classes), -100.0, dtype=torch.float32, device=device
    )

    has_detector = bool(getattr(context_detector, "episode_classes", None))
    if policy == "backbone_nomask" or not has_detector:
        model.clear_active_adapters()
        out = model(X_batch)
        unseen = _seen_unseen_mask(num_classes, seen_classes, out.device)
        out[:, unseen] = -100.0
        return out, None

    if policy == "pred_hard":
        mode, adapter_source, mask_source = "hard", "predicted", "predicted"
    elif policy == "pred_adapter_nomask":
        mode, adapter_source, mask_source = "nomask", "predicted", "none"
    elif policy == "oracle_adapter_nomask":
        mode, adapter_source, mask_source = "nomask", "oracle", "none"
    elif policy == "oracle_hard":
        mode, adapter_source, mask_source = "hard", "oracle", "oracle"
    elif policy == "oracle_hard_no_adapter":
        mode, adapter_source, mask_source = "hard", "none", "oracle"
    else:
        mode, adapter_source, mask_source = str(route_mode or "hard").lower(), "predicted", "predicted"
    if mode not in {"hard", "topk", "nomask", "adaptive"}:
        raise ValueError(f"Unknown DeNICE route mode: {mode}")

    episodes, chain_probs = _route_episodes_with_scores(model, X_batch, context_detector)
    active_episodes = (
        _validate_oracle_episodes(oracle_episodes, n)
        if adapter_source == "oracle"
        else (
            _validate_oracle_episodes(oracle_episodes, n)
            if adapter_source == "none" and mask_source == "oracle"
            else np.asarray(episodes, dtype=np.int64)
        )
    )
    mask_episodes = (
        _validate_oracle_episodes(oracle_episodes, n)
        if mask_source == "oracle"
        else np.asarray(episodes, dtype=np.int64)
    )
    unseen = _seen_unseen_mask(num_classes, seen_classes, device)

    # Per-sample top-k episode indices (only needed for the topk rule).
    topk_eps: Optional[np.ndarray] = None
    if mode == "topk" and chain_probs is not None and chain_probs.ndim == 2:
        k = max(1, int(route_topk))
        topk_eps = np.argsort(-chain_probs, axis=1)[:, :k]
    elif mode == "adaptive" and chain_probs is not None and chain_probs.ndim == 2:
        k = max(2, int(route_topk))
        topk_eps = np.argsort(-chain_probs, axis=1)[:, :k]

    allowed_cache: Dict[int, List[int]] = {}

    def _allowed_for(ep: int) -> List[int]:
        ep = int(ep)
        if ep not in allowed_cache:
            allowed_cache[ep] = _allowed_classes_for_episode(
                context_detector, ep, seen_classes, num_classes
            )
        return allowed_cache[ep]

    for ep in np.unique(active_episodes):
        idx_np = np.where(active_episodes == ep)[0]
        idx = torch.as_tensor(idx_np, dtype=torch.long, device=device)
        if adapter_source == "none":
            model.clear_active_adapters()
        else:
            model.set_active_context(int(ep))  # adapter follows the top-1 route
        if getattr(model, "active_adapters", {}):
            _increment_route_diagnostic(routing_diagnostics, "adapter_active_sample_count", len(idx_np))
        else:
            _increment_route_diagnostic(routing_diagnostics, "missing_adapter_sample_count", len(idx_np))
        out = model(X_batch.index_select(0, idx))
        out[:, unseen] = -100.0

        if mode == "nomask" or mask_source == "none":
            routed_logits[idx] = out
            continue

        if mode == "adaptive":
            # Safe, explicit fallbacks: a confident route gets the original
            # hard mask; uncertain routes use a top-k class union; no scores
            # or low confidence leaves all seen classes available.
            if chain_probs is None or chain_probs.ndim != 2:
                routed_logits[idx] = out
                _increment_route_diagnostic(routing_diagnostics, "adaptive_nomask_sample_count", len(idx_np))
                continue
            confidence = np.max(chain_probs[idx_np], axis=1)
            hard_rows = np.where(confidence >= float(adaptive_high_confidence))[0]
            topk_rows = np.where(
                (confidence >= float(adaptive_low_confidence))
                & (confidence < float(adaptive_high_confidence))
            )[0]
            nomask_rows = np.where(confidence < float(adaptive_low_confidence))[0]
            if hard_rows.size:
                for row in hard_rows:
                    routed_logits[idx[row : row + 1]] = _mask_logits_to_classes(
                        out[row : row + 1], _allowed_for(int(ep))
                    )
            if topk_rows.size:
                allow = np.zeros((len(topk_rows), num_classes), dtype=bool)
                for col in range(topk_eps.shape[1]):
                    sub_eps = topk_eps[idx_np[topk_rows], col]
                    for candidate_ep in np.unique(sub_eps):
                        local_rows = np.where(sub_eps == candidate_ep)[0]
                        allowed = _allowed_for(int(candidate_ep))
                        if local_rows.size and allowed:
                            allow[np.ix_(local_rows, np.asarray(allowed, dtype=np.int64))] = True
                routed_logits[idx[topk_rows]] = torch.where(
                    torch.as_tensor(allow, dtype=torch.bool, device=device),
                    out[topk_rows],
                    torch.full_like(out[topk_rows], -100.0),
                )
            if nomask_rows.size:
                routed_logits[idx[nomask_rows]] = out[nomask_rows]
            _increment_route_diagnostic(routing_diagnostics, "adaptive_hard_sample_count", len(hard_rows))
            _increment_route_diagnostic(routing_diagnostics, "adaptive_topk_sample_count", len(topk_rows))
            _increment_route_diagnostic(routing_diagnostics, "adaptive_nomask_sample_count", len(nomask_rows))
            continue

        if mode == "topk" and topk_eps is not None:
            # Union of allowed classes over each sample's own top-k episodes.
            allow = np.zeros((len(idx_np), num_classes), dtype=bool)
            for col in range(topk_eps.shape[1]):
                sub_eps = topk_eps[idx_np, col]
                for e in np.unique(sub_eps):
                    rows = np.where(sub_eps == e)[0]
                    cols = _allowed_for(int(e))
                    if rows.size and cols:
                        allow[np.ix_(rows, np.asarray(cols, dtype=np.int64))] = True
            allow_t = torch.as_tensor(allow, dtype=torch.bool, device=device)
            routed_logits[idx] = torch.where(
                allow_t, out, torch.full_like(out, -100.0)
            )
            _increment_route_diagnostic(routing_diagnostics, "topk_mask_sample_count", len(idx_np))
            continue

        # Oracle-hard policies mask rows by true task episode rather than by
        # whichever adapter group happened to be active.
        if mask_source == "oracle":
            row_masks = []
            for mask_ep in mask_episodes[idx_np]:
                row_masks.append(_allowed_for(int(mask_ep)))
            mask_allow = torch.zeros_like(out, dtype=torch.bool)
            for row_index, allowed in enumerate(row_masks):
                if allowed:
                    mask_allow[row_index, torch.as_tensor(allowed, device=device)] = True
            routed_logits[idx] = torch.where(mask_allow, out, torch.full_like(out, -100.0))
        else:
            # hard (default): identical to the original top-1 masking path.
            routed_logits[idx] = _mask_logits_to_classes(out, _allowed_for(int(ep)))
        _increment_route_diagnostic(routing_diagnostics, "hard_mask_sample_count", len(idx_np))

    model.clear_active_adapters()
    return routed_logits, np.asarray(episodes)

@torch.no_grad()
def _denice_predict_with_episodes(
    model,
    X_batch: torch.Tensor,
    context_detector,
    seen_classes: List[int],
    device: str,
    route_mode: str = "hard",
    route_topk: int = 1,
) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
    """Predict class ids and return the routed episode per sample.

    The episode array is ``None`` when no context detector is available (so
    routing falls back to the plain seen-class mask).
    """
    logits, episodes = _denice_routed_logits_with_episodes(
        model, X_batch, context_detector, seen_classes, device, route_mode, route_topk
    )
    return logits.argmax(dim=1), episodes


def denice_predict_batch(
    model,
    X_batch: torch.Tensor,
    context_detector,
    seen_classes: List[int],
    device: str,
    route_mode: str = "hard",
    route_topk: int = 1,
) -> torch.Tensor:
    """Return predicted class ids for a batch using context-routed adapters."""
    preds, _episodes = _denice_predict_with_episodes(
        model, X_batch, context_detector, seen_classes, device, route_mode, route_topk
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
    route_mode: str = "hard",
    route_topk: int = 1,
    include_route_diagnostics: bool = False,
) -> Dict[str, Any]:
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
    route_confusion: Dict[str, Dict[str, int]] = {}

    for batch_idx, i in enumerate(range(0, len(y_test), batch_size), start=1):
        X_batch = X_test[i : i + batch_size].to(device)
        y_batch = y_test[i : i + batch_size].to(device)

        routed_logits, episodes = _denice_routed_logits_with_episodes(
            model, X_batch, context_detector, seen_classes, device, route_mode, route_topk
        )
        total_loss += criterion(routed_logits, y_batch).item() * len(y_batch)
        preds = routed_logits.argmax(dim=1)
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
            if include_route_diagnostics:
                for true_ep, pred_ep in zip(true_eps[known], episodes[known]):
                    true_key = str(int(true_ep))
                    pred_key = str(int(pred_ep))
                    row = route_confusion.setdefault(true_key, {})
                    row[pred_key] = int(row.get(pred_key, 0) + 1)

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
    metrics: Dict[str, Any] = {
        "loss": total_loss / max(1, len(y_test)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "route_accuracy": (route_correct / route_total) if route_total > 0 else 0.0,
        "route_coverage": (route_total / len(y_test)) if len(y_test) > 0 else 0.0,
    }
    if include_route_diagnostics:
        metrics["route_confusion"] = route_confusion
    return metrics


@torch.no_grad()
def evaluate_denice_ensemble(
    model_detector_pairs: List[Tuple[Any, Any]],
    test_data: Dict[str, torch.Tensor],
    device: str,
    seen_classes: List[int],
    batch_size: int = 1024,
    route_mode: str = "hard",
    route_topk: int = 1,
    inference_policy: Optional[str] = None,
    class_to_episode: Optional[Dict[int, int]] = None,
) -> Dict[str, Any]:
    """Evaluate an equal-probability ensemble of cluster representatives.

    Every pair supplies its own model and matching context detector, avoiding
    the invalid assumption that a detector calibrated for one model can route
    another model.  This is the protocol-level global metric; it is separate
    from personalized per-client evaluation.
    """
    if not model_detector_pairs:
        raise ValueError("model_detector_pairs must not be empty")
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]
    if len(y_test) == 0:
        return {
            "loss": 0.0, "accuracy": 0.0, "precision_macro": 0.0,
            "recall_macro": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0,
            "route_accuracy": 0.0, "route_coverage": 0.0,
            "ensemble_size": float(len(model_detector_pairs)),
        }

    all_preds: List[int] = []
    all_targets: List[int] = []
    total_loss = 0.0
    route_correct = 0
    route_total = 0
    oracle_mask_violation_count = 0
    for i in range(0, len(y_test), max(1, batch_size)):
        X_batch = X_test[i : i + batch_size].to(device)
        y_batch = y_test[i : i + batch_size].to(device)
        probs = None
        for model, detector in model_detector_pairs:
            oracle_episodes = None
            if inference_policy in {
                "oracle_adapter_nomask", "oracle_hard", "oracle_hard_no_adapter"
            }:
                label_map = class_to_episode or _label_to_episode_map(detector)
                oracle_episodes = np.asarray(
                    [label_map.get(int(label), -1) for label in y_batch.detach().cpu().tolist()],
                    dtype=np.int64,
                )
                if np.any(oracle_episodes < 0):
                    raise ValueError("oracle ensemble evaluation encountered an unmapped test class")
                if inference_policy in {"oracle_hard", "oracle_hard_no_adapter"}:
                    for label, episode in zip(y_batch.detach().cpu().tolist(), oracle_episodes):
                        allowed = detector.episode_classes.get(int(episode), [])
                        if int(label) not in {int(cls) for cls in allowed}:
                            oracle_mask_violation_count += 1
            logits, episodes = _denice_routed_logits_with_episodes(
                model,
                X_batch,
                detector,
                seen_classes,
                device,
                route_mode,
                route_topk,
                inference_policy=inference_policy,
                oracle_episodes=oracle_episodes,
            )
            current = torch.softmax(logits, dim=1)
            probs = current if probs is None else probs + current
            label2episode = _label_to_episode_map(detector)
            if episodes is not None and label2episode:
                true_eps = np.asarray(
                    [label2episode.get(int(c), -1) for c in y_batch.detach().cpu().numpy()],
                    dtype=np.int64,
                )
                known = true_eps >= 0
                route_total += int(known.sum())
                route_correct += int((episodes[known] == true_eps[known]).sum())
        probs = probs / float(len(model_detector_pairs))
        total_loss += (
            -torch.log(probs.clamp_min(1e-12)[
                torch.arange(len(y_batch), device=probs.device), y_batch
            ]).sum().item()
        )
        all_preds.extend(probs.argmax(dim=1).detach().cpu().tolist())
        all_targets.extend(y_batch.detach().cpu().tolist())

    y_true = np.asarray(all_targets)
    y_pred = np.asarray(all_preds)
    return {
        "loss": total_loss / max(1, len(y_test)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "route_accuracy": route_correct / route_total if route_total else 0.0,
        "route_coverage": route_total / max(1, len(y_test) * len(model_detector_pairs)),
        "ensemble_size": float(len(model_detector_pairs)),
        "inference_policy": inference_policy or "routed_default",
        "oracle_mask_violation_count": int(oracle_mask_violation_count),
    }
