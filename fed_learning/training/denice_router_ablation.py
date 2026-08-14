"""Post-hoc, model-invariant router-reference ablations for DeNICE.

The D5 experiment changes only the local raw examples used to fit a client's
context router.  It deliberately rebuilds those examples from the original
client *training* split rather than trying to upsample a checkpoint's old
reference bank.  Model weights, adapters, capacity state, and the evaluation
support therefore remain fixed across reference-memory budgets.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch


def model_states_sha256(client_model_states: Mapping[Any, Mapping[str, torch.Tensor]]) -> str:
    """Hash model tensors only, in a stable order, for D5 invariance checks."""
    digest = hashlib.sha256()
    for client_id in sorted((int(cid) for cid in client_model_states)):
        state = client_model_states.get(client_id, client_model_states.get(str(client_id)))
        if state is None:
            raise KeyError(f"Missing model state for client {client_id}")
        digest.update(f"client:{client_id}\n".encode("utf-8"))
        for name in sorted(state):
            value = state[name].detach().cpu().contiguous()
            digest.update(f"{name}|{value.dtype}|{tuple(value.shape)}\n".encode("utf-8"))
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _sample_episode_reference(
    X: torch.Tensor,
    y: torch.Tensor,
    classes: Iterable[int],
    budget_per_class: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Deterministically sample a nested, class-stratified raw reference bank."""
    selected: List[torch.Tensor] = []
    counts: Dict[int, int] = {}
    labels = y.detach().cpu()
    for class_offset, class_id in enumerate(sorted({int(item) for item in classes})):
        candidates = torch.nonzero(labels == class_id, as_tuple=False).flatten()
        if len(candidates) == 0:
            counts[class_id] = 0
            continue
        generator = torch.Generator().manual_seed(
            int(seed) + 104_729 * int(class_offset + 1)
        )
        order = candidates[torch.randperm(len(candidates), generator=generator)]
        take = min(int(budget_per_class), int(len(order)))
        selected.append(order[:take])
        counts[class_id] = int(take)
    if not selected:
        return np.empty((0, *X.shape[1:]), dtype=np.float32), counts
    rows = torch.cat(selected)
    return X[rows].detach().cpu().numpy().astype(np.float32, copy=True), counts


def rebuild_reference_input_memory(
    data_loader: Any,
    *,
    client_id: int,
    episode_classes: Mapping[Any, Iterable[int]],
    budget_per_class: int,
    seed: int,
) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[int, int]]]:
    """Recreate a client's router bank using only its original train data.

    The sampling seed intentionally does not contain ``budget_per_class``.
    Thus, when a class has enough rows, D5's 20-row bank is a subset of its
    50-row bank, which is a subset of its 100-row bank.
    """
    if int(budget_per_class) < 1:
        raise ValueError("budget_per_class must be positive")
    memory: Dict[int, np.ndarray] = {}
    counts: Dict[int, Dict[int, int]] = {}
    for episode_raw, classes in sorted(episode_classes.items(), key=lambda item: int(item[0])):
        episode = int(episode_raw)
        X, y = data_loader.get_client_data(int(client_id), episode)
        rows, class_counts = _sample_episode_reference(
            X,
            y,
            classes,
            int(budget_per_class),
            seed=int(seed) + 1_000_003 * int(client_id) + 10_007 * episode,
        )
        counts[episode] = class_counts
        if len(rows):
            memory[episode] = rows
    return memory, counts
