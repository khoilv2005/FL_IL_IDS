"""
DeNICE Client - NICE phase-based training with active micro-adapters.

The DeNICE client behaves exactly like :class:`NICEClient` (phase-based
tau-greedy training, connection pruning, mature gradient freezing). The
difference is purely in the model: a :class:`DeNICEModel` applies whichever
micro-adapters are currently *active* during ``forward_output``. Adapter
parameters are part of ``model.parameters()`` so they are optimized together
with the plastic NICE neurons; inactive adapters receive no gradient and stay
frozen automatically.

The CANC decision (which adapters to activate, whether to freeze low layers) is
made by the task loop before ``train`` is called, then applied to the model via
``model.add_adapter`` / ``model.set_active_adapter``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Tuple

import torch

from .nice_client import NICEClient


_BATCH_SAMPLING_MODES = {"natural", "class_balanced"}
_CLASS_WEIGHT_MODES = {"none", "inverse_frequency", "effective_number"}


def normalize_denice_imbalance_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize the opt-in D3 imbalance controls.

    D3-A and D3-B are deliberately mutually exclusive for the initial
    one-factor ablations.  A later confirmation run may opt in to their
    combination explicitly.
    """
    batch_sampling = str(config.get("denice_batch_sampling", "natural")).lower()
    class_weight_mode = str(
        config.get("denice_class_weight_mode", "none")
    ).lower()
    if batch_sampling not in _BATCH_SAMPLING_MODES:
        raise ValueError(
            "denice_batch_sampling must be one of "
            f"{sorted(_BATCH_SAMPLING_MODES)}, got {batch_sampling!r}."
        )
    if class_weight_mode not in _CLASS_WEIGHT_MODES:
        raise ValueError(
            "denice_class_weight_mode must be one of "
            f"{sorted(_CLASS_WEIGHT_MODES)}, got {class_weight_mode!r}."
        )

    smoothing = float(config.get("denice_class_weight_smoothing", 1.0))
    effective_beta = float(config.get("denice_class_weight_effective_beta", 0.999))
    clip_min = float(config.get("denice_class_weight_min", 0.25))
    clip_max = float(config.get("denice_class_weight_max", 4.0))
    if smoothing < 0:
        raise ValueError("denice_class_weight_smoothing must be non-negative.")
    if not 0.0 <= effective_beta < 1.0:
        raise ValueError("denice_class_weight_effective_beta must be in [0, 1).")
    if clip_min <= 0.0 or clip_max < clip_min:
        raise ValueError(
            "denice_class_weight_min must be positive and no greater than "
            "denice_class_weight_max."
        )

    allow_combined = bool(config.get("denice_allow_combined_imbalance_ablation", False))
    if (
        batch_sampling != "natural"
        and class_weight_mode != "none"
        and not allow_combined
    ):
        raise ValueError(
            "D3-A balanced batches and D3-B weighted CE are separate "
            "one-factor ablations. Set denice_allow_combined_imbalance_ablation=true "
            "only for a preregistered combined confirmation run."
        )
    return {
        "denice_batch_sampling": batch_sampling,
        "denice_class_weight_mode": class_weight_mode,
        "denice_class_weight_smoothing": smoothing,
        "denice_class_weight_effective_beta": effective_beta,
        "denice_class_weight_min": clip_min,
        "denice_class_weight_max": clip_max,
        "denice_allow_combined_imbalance_ablation": allow_combined,
    }


def class_balanced_batch_indices(
    labels: torch.Tensor, batch_size: int
) -> Iterator[torch.Tensor]:
    """Yield an epoch-sized, with-replacement class-balanced batch schedule.

    The number of optimization examples stays equal to the natural local
    epoch.  Minority classes are therefore intentionally re-sampled while
    majority examples can be omitted in a given epoch.  Random draws use the
    global torch RNG so a DeNICE continuation restores this sequence exactly.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    labels_cpu = labels.detach().cpu().long().reshape(-1)
    if labels_cpu.numel() == 0:
        return
    classes = torch.unique(labels_cpu, sorted=True)
    class_indices = [torch.nonzero(labels_cpu == cls, as_tuple=False).flatten() for cls in classes]
    num_classes = len(class_indices)
    for start in range(0, int(labels_cpu.numel()), batch_size):
        current_size = min(batch_size, int(labels_cpu.numel()) - start)
        class_order = torch.randperm(num_classes)
        selected: List[torch.Tensor] = []
        for offset in range(current_size):
            source_indices = class_indices[int(class_order[offset % num_classes])]
            draw = torch.randint(len(source_indices), (1,))
            selected.append(source_indices[draw])
        yield torch.cat(selected).long()


def build_denice_class_weights(
    labels: torch.Tensor,
    num_classes: int,
    *,
    mode: str,
    smoothing: float,
    effective_beta: float,
    clip_min: float,
    clip_max: float,
) -> Tuple[torch.Tensor | None, Dict[str, Any]]:
    """Construct clipped, mean-one D3-B class weights and an audit record."""
    mode = str(mode).lower()
    if mode not in _CLASS_WEIGHT_MODES:
        raise ValueError(f"Unsupported DeNICE class-weight mode: {mode!r}.")
    if mode == "none":
        return None, {"mode": mode, "enabled": False}
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")

    labels_cpu = labels.detach().cpu().long().reshape(-1)
    if labels_cpu.numel() and (
        int(labels_cpu.min()) < 0 or int(labels_cpu.max()) >= int(num_classes)
    ):
        raise ValueError("Client labels are outside the model output range.")
    counts = torch.bincount(labels_cpu, minlength=int(num_classes)).float()
    present = counts > 0
    if mode == "inverse_frequency":
        raw = 1.0 / (counts[present] + float(smoothing))
    else:
        adjusted = counts[present] + float(smoothing)
        raw = (1.0 - float(effective_beta)) / (
            1.0 - torch.pow(torch.tensor(float(effective_beta)), adjusted)
        )
    raw = raw / raw.mean().clamp_min(torch.finfo(raw.dtype).eps)
    weights = torch.ones(int(num_classes), dtype=torch.float32)
    weights[present] = raw.clamp(min=float(clip_min), max=float(clip_max))
    present_weights = weights[present]
    return weights, {
        "mode": mode,
        "enabled": True,
        "smoothing": float(smoothing),
        "effective_beta": float(effective_beta),
        "clip_min": float(clip_min),
        "clip_max": float(clip_max),
        "present_class_count": int(present.sum().item()),
        "weight_min": float(present_weights.min().item()),
        "weight_max": float(present_weights.max().item()),
        "class_counts": {
            int(cls): int(count)
            for cls, count in enumerate(counts.tolist())
            if count > 0
        },
    }


class DeNICEClient(NICEClient):
    """NICE client whose model carries capacity-aware micro-adapters."""

    def _create_batches(self, batch_size: int):
        mode = getattr(self, "_denice_batch_sampling", "natural")
        if mode == "natural":
            yield from super()._create_batches(batch_size)
            return

        batches = list(class_balanced_batch_indices(self.y_train, batch_size))
        sampled = torch.cat(batches) if batches else torch.empty(0, dtype=torch.long)
        sampled_labels = self.y_train[sampled] if sampled.numel() else sampled
        sampled_classes, sampled_counts = torch.unique(
            sampled_labels.detach().cpu().long(), return_counts=True
        )
        self._denice_batch_sampling_epochs.append(
            {
                "mode": mode,
                "draw_count": int(sampled.numel()),
                "unique_source_sample_count": int(torch.unique(sampled).numel()),
                "sampled_class_hist": {
                    int(cls): int(count)
                    for cls, count in zip(sampled_classes.tolist(), sampled_counts.tolist())
                },
            }
        )
        for batch_idx in batches:
            X_batch = self.X_train[batch_idx].to(self.device, non_blocking=True)
            y_batch = self.y_train[batch_idx].to(self.device, non_blocking=True)
            yield X_batch, y_batch

    def train(self, *args, **kwargs) -> Dict[str, Any]:
        controls = normalize_denice_imbalance_config(kwargs)
        previous_mode = getattr(self, "_denice_batch_sampling", "natural")
        self._denice_batch_sampling = controls["denice_batch_sampling"]
        self._denice_batch_sampling_epochs: List[Dict[str, Any]] = []
        class_weights, weight_audit = build_denice_class_weights(
            self.y_train,
            int(getattr(self.model, "num_classes")),
            mode=controls["denice_class_weight_mode"],
            smoothing=controls["denice_class_weight_smoothing"],
            effective_beta=controls["denice_class_weight_effective_beta"],
            clip_min=controls["denice_class_weight_min"],
            clip_max=controls["denice_class_weight_max"],
        )
        kwargs["class_weights"] = class_weights
        try:
            result = super().train(*args, **kwargs)
        finally:
            self._denice_batch_sampling = previous_mode

        model = self.model
        if hasattr(model, "get_adapter_registry_state"):
            result["adapter_registry"] = model.get_adapter_registry_state()
            result["adapter_param_count"] = int(model.adapter_param_count())
            result["active_adapters"] = dict(getattr(model, "active_adapters", {}))
        result["imbalance_control"] = {
            "batch_sampling": controls["denice_batch_sampling"],
            "sampling_epochs": list(self._denice_batch_sampling_epochs),
            "class_weights": weight_audit,
        }
        return result

    def get_buffer_stats(self) -> Dict:
        return {
            "buffer_type": "none",
            "has_replay": False,
            "note": "DeNICE is replay-free (NICE + micro-adapters)",
        }
