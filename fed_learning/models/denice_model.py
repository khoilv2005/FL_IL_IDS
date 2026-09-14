"""
DeNICE Model - NICE backbone extended with capacity-aware micro-adapters.

The fresh CANDLE configuration uses ``linear_input`` (architecture v2),
implementing Eq. (23) as U(V(x)) on each adapted block's input. CNN adapters
project adjacent input positions to match stride-two pooling; the GRU adapter
projects the flattened input sequence; fc1 uses concatenated backbone features.
These sequence alignment choices are explicit implementation details, not
hyperparameters specified by the PDF.

Legacy ``legacy_output`` (v1) retains section 2.1 of the earlier implementation plan:

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
import numpy as np

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


class LinearInputAdapter(nn.Module):
    """CANDLE Eq. (23): bias-free U(V(x)) on the layer input."""

    def __init__(self, input_dim: int, dim: int, rank: int):
        super().__init__()
        self.dim, self.rank = int(dim), int(rank)
        self.V = nn.Linear(int(input_dim), self.rank, bias=False)
        self.U = nn.Linear(self.rank, self.dim, bias=False)
        nn.init.kaiming_uniform_(self.V.weight, a=5 ** 0.5)
        nn.init.zeros_(self.U.weight)

    def forward(self, x):
        return self.U(self.V(x))


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
        self.adapter_mode = 'legacy_output'
        self.adapters = nn.ModuleDict()
        self.adapter_registry: Dict[str, Dict] = {}
        self.active_adapters: Dict[str, str] = {}
        self.recycling_registry: Dict[str, Dict[int, Dict[str, int]]] = {}
        # Opt in for fresh runs; legacy checkpoints keep their original graph.
        self.structural_protection = False
        self.fixed_task_allocation = False
        self.allocation_policy = 'class_blocks'
        self.capacity_per_class = {}
        self.candle_state = {}
        self.task_freeze_layers = []
        self.gru_connection_masks = {}
        self.adapter_input_masks = {}
        self.adapter_output_masks = {}

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

    def allocate_task_neurons(self, classes: List[int]) -> Dict[str, int]:
        """Promote reserve units using the configured per-class policy.

        ``fixed_per_class`` implements Eq. (12)'s constant integer budget.
        ``class_blocks`` retains global-coordinate alignment as a legacy ablation.
        Existing learners/mature units are never reset, including on resume.
        """
        classes = sorted(set(int(c) for c in classes))
        if any(c < 0 or c >= self.num_classes for c in classes):
            raise ValueError('Task classes are outside the model output range.')
        allocation = {}
        for layer, ranks in self.unit_ranks.items():
            if layer == 'fc2' or layer in self.task_freeze_layers:
                continue
            selected = np.zeros(len(ranks), dtype=bool)
            if self.allocation_policy in ('legacy_sequential', 'fixed_per_class'):
                free = np.flatnonzero(ranks == 0)
                per_class = self.capacity_per_class.get(layer, int(np.ceil(len(ranks) / self.num_classes)))
                if int(per_class) != per_class or per_class < 1:
                    raise ValueError('capacity_per_class must contain positive integer budgets.')
                budget = int(per_class) * len(classes)
                selected[free[:budget]] = True
            elif self.allocation_policy == 'class_blocks':
                for cls in classes:
                    start = cls * len(ranks) // self.num_classes
                    end = (cls + 1) * len(ranks) // self.num_classes
                    selected[start:end] = True
            else:
                raise ValueError(f'Unknown allocation policy: {self.allocation_policy}')
            selected &= ranks == 0
            ranks[selected] = 1
            allocation[layer] = int(selected.sum())
        return allocation

    def add_adapter(
        self,
        context_id: int,
        layer_name: str,
        rank: Optional[int] = None,
        set_active: bool = True,
        *,
        mode: Optional[str] = None,
        architecture_version: Optional[int] = None,
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

        mode = mode or self.adapter_mode
        if mode not in ('legacy_output', 'linear_input'):
            raise ValueError('Unknown adapter mode.')
        dim = self._adapter_dims[layer_name]
        input_dim = self._adapter_input_dim(layer_name) if mode == 'linear_input' else dim
        r = int(rank) if rank is not None else min(default_rank(dim), input_dim, dim)
        if r < 1:
            raise ValueError('Adapter rank must be positive.')
        version = architecture_version if architecture_version is not None else (2 if mode == 'linear_input' else 1)
        key = adapter_key(context_id, layer_name, r, version)

        if key not in self.adapters:
            device = next(self.parameters()).device
            self.adapters[key] = (LinearInputAdapter(input_dim, dim, r) if mode == 'linear_input'
                                  else MicroAdapter(dim, r)).to(device)
            self.adapter_registry[key] = {
                "context_id": int(context_id),
                "layer_name": layer_name,
                "rank": r,
                "architecture_version": int(version),
                "dim": int(dim),
                "param_count": int((input_dim + dim) * r),
            }
            if mode == 'linear_input':
                self.adapter_registry[key].update(mode=mode, input_dim=input_dim)
        elif self.adapter_registry[key].get('mode', 'legacy_output') != mode:
            raise ValueError('Adapter key already belongs to a different architecture.')

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

        keys = [key for key, meta in self.adapter_registry.items()
                if int(meta['context_id']) == int(context_id)
                and meta['layer_name'] == layer_name and key in self.adapters]
        if len(keys) > 1:
            raise ValueError(f'Ambiguous adapters for context {context_id}, layer {layer_name}: {keys}')
        if keys:
            key = keys[0]
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
        if key not in self.adapters:
            return None
        return self.adapters[key]

    def has_adapter(self, context_id: int, layer_name: str, rank: Optional[int] = None) -> bool:
        dim = self._adapter_dims[layer_name]
        input_dim = self._adapter_input_dim(layer_name) if self.adapter_mode == 'linear_input' else dim
        r = int(rank) if rank is not None else min(default_rank(dim), input_dim, dim)
        key = adapter_key(context_id, layer_name, r, self.architecture_version)
        return key in self.adapters

    def adapter_param_count(self) -> int:
        return int(sum(meta["param_count"] for meta in self.adapter_registry.values()))

    def get_adapter_registry_state(self) -> Dict[str, Dict]:
        return {k: dict(v) for k, v in self.adapter_registry.items()}

    # ========================================================================
    # Graceful recycling (Phase 4)
    # ========================================================================

    def retire_neurons(
        self, layer_name: str, indices: List[int], task_id: int
    ) -> List[int]:
        """Move mature neurons to age=-1 retired state without deleting weights.

        Retired units are masked out of forward/training and can be revived as
        young units after a grace period. This implements the conservative
        Phase-4 recycling rule: mature -> retired -> young, never mature -> young
        directly.
        """
        if layer_name not in self.unit_ranks:
            return []
        ranks = self.unit_ranks[layer_name]
        valid = sorted(
            {
                int(idx)
                for idx in indices
                if 0 <= int(idx) < len(ranks) and int(ranks[int(idx)]) >= 2
            }
        )
        if not valid:
            return []

        layer_state = self.recycling_registry.setdefault(layer_name, {})
        for idx in valid:
            layer_state[idx] = {
                "retired_at_task": int(task_id),
                "previous_age": int(ranks[idx]),
            }
            ranks[idx] = -1
        self._set_recycled_mask(layer_name, valid, enabled=False)
        return valid

    def revive_retired_neurons(self, task_id: int, grace_tasks: int = 1) -> Dict[str, List[int]]:
        """Revive retired neurons as young when the grace period has passed."""
        revived: Dict[str, List[int]] = {}
        for layer_name, layer_state in list(self.recycling_registry.items()):
            ready = []
            for idx, meta in list(layer_state.items()):
                retired_at = int(meta.get("retired_at_task", task_id))
                if int(task_id) - retired_at >= int(grace_tasks):
                    ready.append(int(idx))
            if not ready:
                continue
            ranks = self.unit_ranks.get(layer_name)
            if ranks is None:
                continue
            for idx in ready:
                if 0 <= idx < len(ranks) and ranks[idx] == -1:
                    ranks[idx] = 0
                    layer_state.pop(idx, None)
            self._set_recycled_mask(layer_name, ready, enabled=True)
            revived[layer_name] = sorted(ready)
            if not layer_state:
                self.recycling_registry.pop(layer_name, None)
        return revived

    def get_recycling_state(self) -> Dict[str, Dict[int, Dict[str, int]]]:
        return {
            layer: {int(idx): dict(meta) for idx, meta in state.items()}
            for layer, state in self.recycling_registry.items()
        }

    def set_recycling_state(self, state: Dict[str, Dict[int, Dict[str, int]]]) -> None:
        self.recycling_registry = {
            str(layer): {int(idx): dict(meta) for idx, meta in entries.items()}
            for layer, entries in (state or {}).items()
        }

    def _set_recycled_mask(
        self, layer_name: str, indices: List[int], enabled: bool
    ) -> None:
        if not indices:
            return
        value = 1.0 if enabled else 0.0
        if layer_name == "gru":
            self.weight_masks[layer_name][indices] = value
            self.bias_masks[layer_name][indices] = value
            return
        if layer_name not in self.weight_masks:
            return
        mask = self.weight_masks[layer_name]
        for idx in indices:
            if 0 <= int(idx) < mask.shape[0]:
                mask[int(idx)] = value
        self.bias_masks[layer_name][indices] = value
        if not enabled and layer_name in self.BN_LAYER_MAP:
            frozen = self._bn_frozen_units.get(layer_name)
            if frozen is not None:
                frozen[indices] = False

    # ========================================================================
    # Adapter-aware forward
    # ========================================================================

    def protect_active_adapter_inputs(self):
        """Bind each new adapter to the features consolidated with its task.

        A frozen V/U pair is not stable if V can read reserve neurons trained
        later. Capture support after reserve promotion, once per adapter.
        Legacy adapters without a saved mask retain their original behavior.
        """
        for layer, key in self.active_adapters.items():
            if self.adapter_registry[key].get('mode') == 'linear_input':
                if key not in self.adapter_output_masks:
                    self.adapter_output_masks[key] = torch.as_tensor(
                        self.unit_ranks[layer] == 1, dtype=torch.bool)
                if self.structural_protection and key not in self.adapter_input_masks:
                    self.adapter_input_masks[key] = torch.as_tensor(
                        self._linear_adapter_input_support(layer), dtype=torch.bool)
            elif self.structural_protection and key not in self.adapter_input_masks:
                self.adapter_input_masks[key] = torch.as_tensor(
                    np.asarray(self.unit_ranks[layer]) >= 1, dtype=torch.bool)

    def _linear_adapter_input_support(self, layer):
        """Support in the exact flattening order consumed by the adapter."""
        if layer in ('conv1', 'gru'):
            return np.ones(self._adapter_input_dim(layer), dtype=bool)
        if layer in ('conv2', 'conv3'):
            previous = 'conv1' if layer == 'conv2' else 'conv2'
            return np.repeat(self.unit_ranks[previous] >= 1, 2)
        return np.concatenate([np.repeat(self.unit_ranks['conv3'] >= 1, self.seq_length // 8),
                               self.unit_ranks['gru'] >= 1])

    def remove_recycled_adapter_support(self, layer, chosen):
        """Disconnect recycled features from older adapters and their outputs."""
        for key, meta in self.adapter_registry.items():
            adapter_layer = meta['layer_name']
            if meta.get('mode') == 'linear_input':
                if key in self.adapter_input_masks:
                    self.adapter_input_masks[key] &= torch.as_tensor(
                        self._linear_adapter_input_support(adapter_layer))
                if adapter_layer == layer and key in self.adapter_output_masks:
                    self.adapter_output_masks[key][chosen] = False
            elif adapter_layer == layer and key in self.adapter_input_masks:
                self.adapter_input_masks[key][chosen] = False

    def _adapter_residual(self, h, layer):
        key = self.active_adapters[layer]
        mask = self.adapter_input_masks.get(key)
        if mask is not None:
            h = h * mask.to(device=h.device, dtype=h.dtype)
        residual = self.adapters[key](h)
        output_mask = self.adapter_output_masks.get(key)
        if output_mask is not None:
            residual = residual * output_mask.to(device=h.device, dtype=h.dtype)
        return residual

    def _linear_input_adapter_active(self, layer):
        key = self.active_adapters.get(layer)
        return key is not None and self.adapter_registry[key].get('mode') == 'linear_input'

    def _apply_conv_channel_adapter(self, x: torch.Tensor, layer_name: str,
                                    layer_input=None) -> torch.Tensor:
        """Channel-wise residual adapter for conv layers.

        ``x`` is ``[batch, channels, length]``. The adapter acts on the channel
        dimension at every spatial location (1x1-conv style) then adds back.
        """
        adapter = self.get_active_adapter(layer_name)
        if adapter is None:
            return x
        # [B, C, L] -> [B, L, C] -> adapter -> [B, C, L]
        if self._linear_input_adapter_active(layer_name):
            if layer_input is None:
                raise ValueError('Linear adapter requires its layer input.')
            windows = layer_input.unfold(-1, 2, 2).permute(0, 2, 1, 3)
            h = windows.reshape(windows.shape[0], windows.shape[1], -1)
        else:
            h = x.permute(0, 2, 1)
        residual = self._adapter_residual(h, layer_name)
        return x + residual.permute(0, 2, 1)

    def _forward_backbone(self, x):
        """Backbone forward with optional conv3 / gru micro-adapters."""
        if x.ndim == 2:
            x = x.unsqueeze(-1)

        x_cnn = x.permute(0, 2, 1)
        for index in (1, 2, 3):
            layer = f'conv{index}'
            layer_input = x_cnn
            x_cnn = self._apply_masked_conv(layer_input, getattr(self, layer),
                                           getattr(self, f'bn{index}'), getattr(self, f'pool{index}'), layer)
            if layer in self.active_adapters:
                x_cnn = self._apply_conv_channel_adapter(x_cnn, layer, layer_input)
        cnn_output = x_cnn.view(x.size(0), -1)

        x_gru, _ = self._run_gru(x)
        gru_output = x_gru[:, -1, :]
        device = gru_output.device
        gru_mask = self.weight_masks["gru"].to(device)
        gru_output = gru_output * gru_mask

        gru_adapter = self.get_active_adapter("gru")
        if gru_adapter is not None:
            adapter_input = x.reshape(x.shape[0], -1) if self._linear_input_adapter_active('gru') else gru_output
            gru_output = gru_output + self._adapter_residual(adapter_input, 'gru')

        return torch.cat([cnn_output, gru_output], dim=1)

    def _apply_fc1_adapter(self, z: torch.Tensor, features=None) -> torch.Tensor:
        adapter = self.get_active_adapter("fc1")
        if adapter is None:
            return z
        if self._linear_input_adapter_active('fc1'):
            if features is None:
                raise ValueError('Linear fc1 adapter requires backbone features.')
            return z + self._adapter_residual(features, 'fc1')
        return z + self._adapter_residual(z, 'fc1')

    def penultimate_features(self, x):
        features = self._forward_backbone(x)
        z = self.relu(self._apply_masked_linear(features, self.fc1, 'fc1'))
        return self._apply_fc1_adapter(z, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference forward with active adapters (fc1 residual on penultimate)."""
        z = self.penultimate_features(x)
        z = self.dropout(z)
        z = self._apply_masked_linear(z, self.fc2, "fc2")
        return z

    def forward_output(self, x: torch.Tensor) -> torch.Tensor:
        """Training forward (Let_Learner + MaskedOut_Young) with active adapters."""
        from .nice_model import MaskedOutYoung, LetLearner

        z = self.penultimate_features(x)

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

    def get_output_and_context_activations(self, x):
        """Classifier honors active adapters; routing uses the stable backbone."""
        logits, activations = super().get_output_and_context_activations(x)
        return (self(x) if self.active_adapters else logits), activations

    # get_context_activations_per_sample retains the adapter-free NICE path.
    def _run_gru(self, x):
        if not self.structural_protection or not self.gru_connection_masks:
            return self.gru(x)
        parameters = {
            name: parameter * self.gru_connection_masks[name].to(parameter.device)
            if name in self.gru_connection_masks else parameter
            for name, parameter in self.gru.named_parameters()
        }
        # Keep the optimized GRU implementation and legacy parameter names.
        return torch.func.functional_call(self.gru, parameters, (x,))

    def protect_task_connections(self):
        """Cut only newly forbidden edges; never reopen consolidated inputs."""
        if not self.structural_protection:
            return
        ranks = np.asarray(self.unit_ranks['gru'])
        forbidden = ((ranks[:, None] >= 1) & (ranks[None, :] <= 0)) | (
            (ranks[:, None] >= 2) & (ranks[None, :] < 2)
        )
        for name, parameter in self.gru.named_parameters():
            if name.startswith('weight_hh') or (
                name.startswith('weight_ih') and name != 'weight_ih_l0'
            ):
                keep = torch.as_tensor(~np.tile(forbidden, (3, 1)), dtype=parameter.dtype)
                previous = self.gru_connection_masks.get(name, torch.ones_like(keep))
                self.gru_connection_masks[name] = previous.cpu() * keep

    def get_masks_state(self):
        state = super().get_masks_state()
        state.update({'recurrent_' + k: v.detach().cpu().clone()
                      for k, v in self.gru_connection_masks.items()})
        state.update({'adapter_input_' + k: v.detach().cpu().clone()
                      for k, v in self.adapter_input_masks.items()})
        state.update({'adapter_output_' + k: v.detach().cpu().clone()
                      for k, v in self.adapter_output_masks.items()})
        return state

    def set_masks_state(self, state):
        super().set_masks_state(state)
        self.gru_connection_masks = {
            k[len('recurrent_'):]: v.clone() for k, v in state.items()
            if k.startswith('recurrent_')
        }
        self.adapter_input_masks = {
            k[len('adapter_input_'):]: v.clone().bool() for k, v in state.items()
            if k.startswith('adapter_input_')
        }
        self.adapter_output_masks = {
            k[len('adapter_output_'):]: v.clone().bool() for k, v in state.items()
            if k.startswith('adapter_output_')
        }

    def configure_adapter_mode(self, mode):
        if mode not in ('legacy_output', 'linear_input'):
            raise ValueError('adapter_mode must be legacy_output or linear_input.')
        self.adapter_mode = mode
        self.architecture_version = 2 if mode == 'linear_input' else 1

    def _adapter_input_dim(self, layer):
        if layer.startswith('conv'):
            # A linear projection of adjacent input positions matches the
            # CNN block's stride-two pooling, including odd sequence lengths.
            return 2 * self._layer_in_dims[layer]
        if layer == 'gru':
            return self.seq_length * self.num_features
        return self._layer_in_dims[layer]
