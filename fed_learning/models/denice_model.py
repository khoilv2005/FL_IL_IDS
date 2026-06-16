"""
DeNICE Model - NICE backbone extended with capacity-aware micro-adapters.

This implements section 2.1 of ``DeNICE_micro_adapter_implementation_plan.md``:

    - Inherits ``NICEModel`` (neuron-age management, weight masks, context
      activations stay unchanged).
    - Adds a small set of *micro-adapters* that can be instantiated per
      context/task and per layer.

Micro-adapter (taken verbatim from the plan / NERVA):

    A_l(h) = U_l sigma(V_l h)
    V_l: d_l -> r_l
    U_l: r_l -> d_l
    r_l = max(4, d_l / 16)

Layer priority for the adapter (plan section 2.1 / Rule 4):

    fc1 -> gru -> conv3 -> conv2 -> conv1

The MVP only enables ``fc1`` by default because the fc1 activation is NOT used
by the NICE context detector (which only looks at conv1/conv2/conv3/gru). That
keeps context routing valid while still letting the classifier head adapt to a
new context. ``conv3`` and ``gru`` adapters are implemented as well so Phase 2
of the plan is code-ready, but they are opt-in.

Adapter identity (plan section 2.1)::

    adapter_id = (context_id, layer_name, rank, architecture_version)

In the MVP ``context_id`` defaults to ``task_id`` until a stable context id is
available. Aggregation/inference code matches adapters by this id.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nice_model import NICEModel


# Layer priority for adapters (highest priority first), plan section 2.1 / Rule 4
ADAPTER_LAYER_PRIORITY: List[str] = ["fc1", "gru", "conv3", "conv2", "conv1"]

# Layers that the MVP supports first (plan section 2.1).
ADAPTER_LAYERS_MVP: List[str] = ["fc1", "gru", "conv3"]

ARCHITECTURE_VERSION = 1


def default_rank(dim: int) -> int:
    """r_l = max(4, d_l / 16) (plan)."""
    return int(max(4, dim // 16))


def adapter_key(
    context_id: int,
    layer_name: str,
    rank: int,
    architecture_version: int = ARCHITECTURE_VERSION,
) -> str:
    """Serialize an ``adapter_id`` tuple into a ``ModuleDict``-safe string key."""
    return f"ctx{int(context_id)}__{layer_name}__r{int(rank)}__v{int(architecture_version)}"


def parse_adapter_key(key: str) -> Tuple[int, str, int, int]:
    """Inverse of :func:`adapter_key`."""
    ctx, layer, rank, ver = key.split("__")
    return (int(ctx[3:]), layer, int(rank[1:]), int(ver[1:]))


class MicroAdapter(nn.Module):
    """A_l(h) = U_l sigma(V_l h) low-rank residual adapter.

    ``U`` is zero-initialized so the adapter starts as a no-op (the residual is
    exactly zero), which means adding an adapter never hurts the model before it
    is trained.
    """

    def __init__(self, dim: int, rank: Optional[int] = None):
        super().__init__()
        self.dim = int(dim)
        self.rank = int(rank) if rank is not None else default_rank(dim)
        self.V = nn.Linear(self.dim, self.rank, bias=False)
        self.U = nn.Linear(self.rank, self.dim, bias=False)
        nn.init.kaiming_uniform_(self.V.weight, a=5 ** 0.5)
        nn.init.zeros_(self.U.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Apply the adapter on the last dimension of ``h`` (size ``dim``)."""
        return self.U(torch.sigmoid(self.V(h)))


class DeNICEModel(NICEModel):
    """NICE backbone + per-context micro-adapter registry.

    Attributes:
        adapters: ``nn.ModuleDict`` of micro-adapters keyed by :func:`adapter_key`.
        adapter_registry: metadata per adapter key (context_id, layer, rank...).
        active_adapters: layer_name -> adapter key currently applied in forward.
        architecture_version: integer architecture id (for aggregation matching).
    """

    def __init__(self, input_shape, num_classes: int = 34):
        super().__init__(input_shape, num_classes)

        self.architecture_version = ARCHITECTURE_VERSION
        self.adapters = nn.ModuleDict()
        self.adapter_registry: Dict[str, Dict] = {}
        self.active_adapters: Dict[str, str] = {}

        # Dimension used by each adapter (the residual operates on these dims).
        self._adapter_dims = {
            "conv1": self._layer_dims["conv1"],
            "conv2": self._layer_dims["conv2"],
            "conv3": self._layer_dims["conv3"],
            "gru": self._layer_dims["gru"],
            "fc1": self._layer_dims["fc1"],
        }

    # ========================================================================
    # Adapter registry management
    # ========================================================================

    def add_adapter(
        self,
        context_id: int,
        layer_name: str,
        rank: Optional[int] = None,
        set_active: bool = True,
    ) -> str:
        """Create (or reuse) the micro-adapter for ``(context_id, layer_name)``.

        Returns the adapter key. Idempotent: if the adapter already exists it is
        returned unchanged.
        """
        if layer_name not in self._adapter_dims:
            raise ValueError(
                f"Adapter not supported for layer '{layer_name}'. "
                f"Supported: {sorted(self._adapter_dims)}"
            )

        dim = self._adapter_dims[layer_name]
        r = int(rank) if rank is not None else default_rank(dim)
        key = adapter_key(context_id, layer_name, r, self.architecture_version)

        if key not in self.adapters:
            device = next(self.parameters()).device
            self.adapters[key] = MicroAdapter(dim, r).to(device)
            self.adapter_registry[key] = {
                "context_id": int(context_id),
                "layer_name": layer_name,
                "rank": r,
                "architecture_version": int(self.architecture_version),
                "dim": int(dim),
                "param_count": int(2 * dim * r),
            }

        if set_active:
            self.active_adapters[layer_name] = key
        return key

    def set_active_adapter(self, layer_name: str, context_id: Optional[int]) -> Optional[str]:
        """Activate the adapter for ``context_id`` on ``layer_name``.

        ``context_id=None`` disables the adapter on that layer. Returns the key
        that became active (or ``None``).
        """
        if context_id is None:
            self.active_adapters.pop(layer_name, None)
            return None

        dim = self._adapter_dims[layer_name]
        r = default_rank(dim)
        key = adapter_key(context_id, layer_name, r, self.architecture_version)
        if key in self.adapters:
            self.active_adapters[layer_name] = key
            return key
        # No adapter for that context -> disable on this layer.
        self.active_adapters.pop(layer_name, None)
        return None

    def set_active_context(self, context_id: Optional[int]) -> None:
        """Activate every adapter that belongs to ``context_id`` across layers.

        Layers without an adapter for that context are disabled. ``None`` clears
        all active adapters. Used at inference once the context detector predicts
        an episode (plan section 10).
        """
        if context_id is None:
            self.clear_active_adapters()
            return
        for layer in self._adapter_dims:
            self.set_active_adapter(layer, context_id)

    def clear_active_adapters(self) -> None:
        self.active_adapters = {}

    def get_active_adapter(self, layer_name: str) -> Optional[MicroAdapter]:
        key = self.active_adapters.get(layer_name)
        if key is None:
            return None
        return self.adapters.get(key)

    def has_adapter(self, context_id: int, layer_name: str, rank: Optional[int] = None) -> bool:
        dim = self._adapter_dims[layer_name]
        r = int(rank) if rank is not None else default_rank(dim)
        key = adapter_key(context_id, layer_name, r, self.architecture_version)
        return key in self.adapters

    def adapter_param_count(self) -> int:
        return int(sum(meta["param_count"] for meta in self.adapter_registry.values()))

    def get_adapter_registry_state(self) -> Dict[str, Dict]:
        return {k: dict(v) for k, v in self.adapter_registry.items()}

    # ========================================================================
    # Adapter-aware forward
    # ========================================================================

    def _apply_conv_channel_adapter(self, x: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Channel-wise residual adapter for conv layers.

        ``x`` is ``[batch, channels, length]``. The adapter acts on the channel
        dimension at every spatial location (1x1-conv style) then adds back.
        """
        adapter = self.get_active_adapter(layer_name)
        if adapter is None:
            return x
        # [B, C, L] -> [B, L, C] -> adapter -> [B, C, L]
        h = x.permute(0, 2, 1)
        residual = adapter(h)
        return x + residual.permute(0, 2, 1)

    def _forward_backbone(self, x):
        """Backbone forward with optional conv3 / gru micro-adapters."""
        if x.ndim == 2:
            x = x.unsqueeze(-1)

        x_cnn = x.permute(0, 2, 1)
        x_cnn = self._apply_masked_conv(x_cnn, self.conv1, self.bn1, self.pool1, "conv1")
        if "conv1" in self.active_adapters:
            x_cnn = self._apply_conv_channel_adapter(x_cnn, "conv1")
        x_cnn = self._apply_masked_conv(x_cnn, self.conv2, self.bn2, self.pool2, "conv2")
        if "conv2" in self.active_adapters:
            x_cnn = self._apply_conv_channel_adapter(x_cnn, "conv2")
        x_cnn = self._apply_masked_conv(x_cnn, self.conv3, self.bn3, self.pool3, "conv3")
        if "conv3" in self.active_adapters:
            x_cnn = self._apply_conv_channel_adapter(x_cnn, "conv3")
        cnn_output = x_cnn.view(x.size(0), -1)

        x_gru, _ = self.gru(x)
        gru_output = x_gru[:, -1, :]
        device = gru_output.device
        gru_mask = self.weight_masks["gru"].to(device)
        gru_output = gru_output * gru_mask

        gru_adapter = self.get_active_adapter("gru")
        if gru_adapter is not None:
            gru_output = gru_output + gru_adapter(gru_output)

        return torch.cat([cnn_output, gru_output], dim=1)

    def _apply_fc1_adapter(self, z: torch.Tensor) -> torch.Tensor:
        adapter = self.get_active_adapter("fc1")
        if adapter is None:
            return z
        return z + adapter(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference forward with active adapters (fc1 residual on penultimate)."""
        features = self._forward_backbone(x)
        z = self.relu(self._apply_masked_linear(features, self.fc1, "fc1"))
        z = self._apply_fc1_adapter(z)
        z = self.dropout(z)
        z = self._apply_masked_linear(z, self.fc2, "fc2")
        return z

    def forward_output(self, x: torch.Tensor) -> torch.Tensor:
        """Training forward (Let_Learner + MaskedOut_Young) with active adapters."""
        from .nice_model import MaskedOutYoung, LetLearner

        features = self._forward_backbone(x)
        z = self.relu(self._apply_masked_linear(features, self.fc1, "fc1"))
        z = self._apply_fc1_adapter(z)

        young_fc1 = torch.as_tensor(
            (self.unit_ranks["fc1"] == 0).tolist(),
            dtype=torch.bool,
            device=z.device,
        )
        if young_fc1.any():
            z = MaskedOutYoung.apply(z, young_fc1)

        z = self.dropout(z)
        z = self._apply_masked_linear(z, self.fc2, "fc2")

        learner_fc2 = torch.as_tensor(
            (self.unit_ranks["fc2"] == 1).tolist(),
            dtype=torch.bool,
            device=z.device,
        )
        if learner_fc2.any():
            z = LetLearner.apply(z, learner_fc2)

        return z

    # NOTE: ``get_context_activations_per_sample`` and
    # ``get_output_and_context_activations`` are intentionally NOT overridden.
    # They inherit the adapter-free NICE path so the context detector keeps
    # routing on stable backbone activations (plan section 4 / 10).
