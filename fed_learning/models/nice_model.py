"""
NICE Model - Neurogenesis Inspired Contextual Encoding.

Reference:
    "NICE: Neurogenesis Inspired Contextual Encoding for Replay-free Class Incremental Learning"
    Gurbuz, Moorman, Dovrolis (CVPR 2024)
    GitHub: https://github.com/BurakGurbuz97/NICE

Key Mechanisms:
    - Neuron ages: 0=surplus/young, 1=learner (plastic), >=2=mature (frozen), 999=input
    - Weight masks: Per-layer binary masks applied in forward (weight * mask)
    - MaskedOut_Young: Custom autograd zeroing young neuron outputs (penultimate layer)
    - Let_Learner: Custom autograd zeroing non-learner output logits
    - Phase-based tau-greedy neuron selection
    - Gradient freezing for mature neurons (age > 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
import numpy as np


# ============================================================================
# Custom Autograd Functions (from paper + GitHub)
# ============================================================================

class MaskedOutYoung(Function):
    """Zero out young neuron outputs in both forward and backward pass.

    Applied on penultimate layer activations. Young neurons (age=0) should
    not contribute to output during training of learner neurons.
    """

    @staticmethod
    def forward(ctx, x, young_mask):
        """
        Args:
            x: activations [batch, neurons]
            young_mask: bool tensor [neurons], True for young neurons
        """
        ctx.save_for_backward(young_mask)
        out = x.clone()
        out[:, young_mask] = 0.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        young_mask, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[:, young_mask] = 0.0
        return grad_input, None


class LetLearner(Function):
    """Zero out non-learner output logits in both forward and backward pass.

    Only learner (age=1) output neurons receive gradient. This prevents
    gradient flow into mature neurons through the loss function.
    """

    @staticmethod
    def forward(ctx, x, learner_mask):
        """
        Args:
            x: output logits [batch, num_classes]
            learner_mask: bool tensor [num_classes], True for learner neurons
        """
        ctx.save_for_backward(learner_mask)
        out = x.clone()
        non_learner = ~learner_mask
        out[:, non_learner] = 0.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        learner_mask, = ctx.saved_tensors
        grad_input = grad_output.clone()
        non_learner = ~learner_mask
        grad_input[:, non_learner] = 0.0
        return grad_input, None


# ============================================================================
# NICE Model
# ============================================================================

class NICEModel(nn.Module):
    """
    NICE Model with neuron age tracking, weight masks, and output masking.

    Architecture: Same CNN-GRU backbone as DeepFed (Conv1d x3 + GRU + FC x2).
    NICE adds per-layer weight masks and neuron age management on top.

    Attributes:
        unit_ranks: Dict[str, np.ndarray] - neuron ages per layer
        weight_masks: Dict[str, torch.Tensor] - per-layer weight masks
        bias_masks: Dict[str, torch.Tensor] - per-layer bias masks
        freeze_masks: Dict[str, np.ndarray] - gradient freeze masks for mature neurons
    """

    # Layer names in forward order (used for iteration)
    LAYER_NAMES = ["conv1", "conv2", "conv3", "gru", "fc1", "fc2"]
    BN_LAYER_MAP = {"conv1": "bn1", "conv2": "bn2", "conv3": "bn3"}

    def __init__(self, input_shape, num_classes: int = 34):
        super().__init__()

        if isinstance(input_shape, tuple):
            seq_length = input_shape[0]
            num_features = input_shape[1] if len(input_shape) > 1 else 1
        else:
            seq_length = int(input_shape)
            num_features = 1

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_features = num_features
        self.seq_length = seq_length

        # CNN backbone
        self.conv1 = nn.Conv1d(num_features, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)

        # CNN output size
        cnn_len = seq_length
        for _ in range(3):
            cnn_len = cnn_len // 2
        self.cnn_output_size = 256 * cnn_len

        # GRU
        self.gru = nn.GRU(num_features, 100, num_layers=2, batch_first=True)
        self.gru_output_size = 100

        # Classifier
        concat_size = self.cnn_output_size + self.gru_output_size
        self.fc1 = nn.Linear(concat_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

        # Layer output dimensions
        self._layer_dims = {
            "conv1": 64, "conv2": 128, "conv3": 256,
            "gru": 100, "fc1": 256, "fc2": num_classes,
        }
        # Layer input dimensions (for weight mask shapes)
        self._layer_in_dims = {
            "conv1": num_features, "conv2": 64, "conv3": 128,
            "gru": num_features, "fc1": concat_size, "fc2": 256,
        }

        # NICE-specific: Neuron ages (NOT in state_dict)
        # 0=young/surplus, 1=learner, >=2=mature, 999=input
        self.unit_ranks: Dict[str, np.ndarray] = {}
        self._init_unit_ranks()

        # NICE-specific: Weight masks (NOT registered buffers to avoid state_dict)
        self.weight_masks: Dict[str, torch.Tensor] = {}
        self.bias_masks: Dict[str, torch.Tensor] = {}
        self._init_masks()

        # NICE-specific: Gradient freeze masks for mature neurons
        self.freeze_masks: Dict[str, np.ndarray] = {}
        self._bn_frozen_units: Dict[str, torch.Tensor] = {}
        self._bn_running_mean_frozen: Dict[str, torch.Tensor] = {}
        self._bn_running_var_frozen: Dict[str, torch.Tensor] = {}

    def _init_unit_ranks(self):
        """Initialize all neuron ages to 0 (young/surplus)."""
        for name, dim in self._layer_dims.items():
            self.unit_ranks[name] = np.zeros(dim, dtype=np.int32)

    def _init_masks(self):
        """Initialize all weight/bias masks to ones (all connections enabled)."""
        for name in self.LAYER_NAMES:
            out_dim = self._layer_dims[name]
            if name.startswith("conv"):
                # Conv1d weight shape: [out_channels, in_channels, kernel_size]
                in_dim = self._layer_in_dims[name]
                self.weight_masks[name] = torch.ones(out_dim, in_dim, 3)
            elif name == "gru":
                # GRU has complex weight structure, we mask at output level
                self.weight_masks[name] = torch.ones(out_dim)
            else:
                # Linear weight shape: [out_features, in_features]
                in_dim = self._layer_in_dims[name]
                self.weight_masks[name] = torch.ones(out_dim, in_dim)
            self.bias_masks[name] = torch.ones(out_dim)

    def _move_masks_to_device(self, device):
        """Move masks to the same device as model parameters."""
        for name in list(self.weight_masks.keys()):
            self.weight_masks[name] = self.weight_masks[name].to(device)
            self.bias_masks[name] = self.bias_masks[name].to(device)

    @property
    def current_young_neurons(self) -> Dict[str, np.ndarray]:
        """Get boolean masks for young (age=0) neurons per layer."""
        return {name: (ranks == 0) for name, ranks in self.unit_ranks.items()}

    @property
    def current_learner_neurons(self) -> Dict[str, np.ndarray]:
        """Get boolean masks for learner (age=1) neurons per layer."""
        return {name: (ranks == 1) for name, ranks in self.unit_ranks.items()}

    @property
    def current_mature_neurons(self) -> Dict[str, np.ndarray]:
        """Get boolean masks for mature (age>=2) neurons per layer."""
        return {name: (ranks >= 2) for name, ranks in self.unit_ranks.items()}

    # ========================================================================
    # Forward methods
    # ========================================================================

    def _restore_frozen_bn_stats(self, bn, layer_name: str) -> None:
        frozen = self._bn_frozen_units.get(layer_name)
        mean = self._bn_running_mean_frozen.get(layer_name)
        var = self._bn_running_var_frozen.get(layer_name)
        if frozen is None or mean is None or var is None or not frozen.any():
            return

        frozen = frozen.to(bn.running_mean.device)
        bn.running_mean.data[frozen] = mean.to(bn.running_mean.device)[frozen]
        bn.running_var.data[frozen] = var.to(bn.running_var.device)[frozen]

    def _apply_partial_frozen_bn(self, x, bn, layer_name: str):
        frozen = self._bn_frozen_units.get(layer_name)
        if frozen is None or not frozen.any() or not bn.training:
            return bn(x)

        frozen = frozen.to(x.device)
        self._restore_frozen_bn_stats(bn, layer_name)

        train_out = F.batch_norm(
            x,
            bn.running_mean,
            bn.running_var,
            bn.weight,
            bn.bias,
            training=True,
            momentum=bn.momentum,
            eps=bn.eps,
        )
        self._restore_frozen_bn_stats(bn, layer_name)

        frozen_out = F.batch_norm(
            x,
            bn.running_mean,
            bn.running_var,
            bn.weight,
            bn.bias,
            training=False,
            momentum=0.0,
            eps=bn.eps,
        )

        view_shape = [1, -1] + [1] * (x.dim() - 2)
        frozen_view = frozen.view(*view_shape)
        return torch.where(frozen_view, frozen_out, train_out)

    def _apply_masked_conv(self, x, conv, bn, pool, layer_name):
        """Apply conv layer with weight masking."""
        device = x.device
        w_mask = self.weight_masks[layer_name].to(device)
        b_mask = self.bias_masks[layer_name].to(device)
        masked_weight = conv.weight * w_mask
        masked_bias = conv.bias * b_mask if conv.bias is not None else None
        x = F.conv1d(x, masked_weight, masked_bias, padding=conv.padding[0])
        x = pool(self.relu(self._apply_partial_frozen_bn(x, bn, layer_name)))
        return x

    def _forward_backbone(self, x):
        """Forward through CNN+GRU backbone, return pre-FC1 features."""
        if x.ndim == 2:
            x = x.unsqueeze(-1)

        # CNN pathway with masked weights
        x_cnn = x.permute(0, 2, 1)
        x_cnn = self._apply_masked_conv(x_cnn, self.conv1, self.bn1, self.pool1, "conv1")
        x_cnn = self._apply_masked_conv(x_cnn, self.conv2, self.bn2, self.pool2, "conv2")
        x_cnn = self._apply_masked_conv(x_cnn, self.conv3, self.bn3, self.pool3, "conv3")
        cnn_output = x_cnn.view(x.size(0), -1)

        # GRU pathway
        x_gru, _ = self.gru(x)
        gru_output = x_gru[:, -1, :]

        # Apply GRU output mask
        device = gru_output.device
        gru_mask = self.weight_masks["gru"].to(device)
        gru_output = gru_output * gru_mask

        return torch.cat([cnn_output, gru_output], dim=1)

    def _apply_masked_linear(self, x, linear, layer_name):
        """Apply linear layer with weight masking."""
        device = x.device
        w_mask = self.weight_masks[layer_name].to(device)
        b_mask = self.bias_masks[layer_name].to(device)
        masked_weight = linear.weight * w_mask
        masked_bias = linear.bias * b_mask if linear.bias is not None else None
        return F.linear(x, masked_weight, masked_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass (for inference/evaluation)."""
        features = self._forward_backbone(x)
        z = self.relu(self._apply_masked_linear(features, self.fc1, "fc1"))
        z = self.dropout(z)
        z = self._apply_masked_linear(z, self.fc2, "fc2")
        return z

    def forward_output(self, x: torch.Tensor) -> torch.Tensor:
        """Training forward with output masking (Let_Learner + MaskedOut_Young).

        Used during training to:
        1. Zero young neuron outputs on penultimate layer (MaskedOut_Young)
        2. Zero non-learner output logits (Let_Learner)
        """
        features = self._forward_backbone(x)
        z = self.relu(self._apply_masked_linear(features, self.fc1, "fc1"))

        # MaskedOut_Young on penultimate (fc1) activations
        young_fc1 = torch.as_tensor(
            (self.unit_ranks["fc1"] == 0).tolist(),
            dtype=torch.bool,
            device=z.device,
        )
        if young_fc1.any():
            z = MaskedOutYoung.apply(z, young_fc1)

        z = self.dropout(z)
        z = self._apply_masked_linear(z, self.fc2, "fc2")

        # Let_Learner on output logits
        learner_fc2 = torch.as_tensor(
            (self.unit_ranks["fc2"] == 1).tolist(),
            dtype=torch.bool,
            device=z.device,
        )
        if learner_fc2.any():
            z = LetLearner.apply(z, learner_fc2)

        return z

    def forward_inference(self, x: torch.Tensor) -> torch.Tensor:
        """Inference forward - all neurons active (for context detector masking)."""
        if x.ndim == 2:
            x = x.unsqueeze(-1)

        x_cnn = x.permute(0, 2, 1)
        x_cnn = self.pool1(self.relu(self.bn1(self.conv1(x_cnn))))
        x_cnn = self.pool2(self.relu(self.bn2(self.conv2(x_cnn))))
        x_cnn = self.pool3(self.relu(self.bn3(self.conv3(x_cnn))))
        cnn_output = x_cnn.view(x.size(0), -1)

        x_gru, _ = self.gru(x)
        gru_output = x_gru[:, -1, :]

        z = torch.cat([cnn_output, gru_output], dim=1)
        z = self.relu(self.fc1(z))
        z = self.dropout(z)
        z = self.fc2(z)
        return z

    # ========================================================================
    # Activations for neuron selection and context detection
    # ========================================================================

    def get_activations(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get per-layer activation tensors for neuron selection.

        Returns mean absolute activation per neuron for each layer.
        """
        self.eval()
        activations = {}

        if x.ndim == 2:
            x = x.unsqueeze(-1)

        with torch.no_grad():
            x_cnn = x.permute(0, 2, 1)

            x_cnn = self._apply_masked_conv(x_cnn, self.conv1, self.bn1, self.pool1, "conv1")
            activations["conv1"] = x_cnn.abs().mean(dim=(0, 2))

            x_cnn = self._apply_masked_conv(x_cnn, self.conv2, self.bn2, self.pool2, "conv2")
            activations["conv2"] = x_cnn.abs().mean(dim=(0, 2))

            x_cnn = self._apply_masked_conv(x_cnn, self.conv3, self.bn3, self.pool3, "conv3")
            activations["conv3"] = x_cnn.abs().mean(dim=(0, 2))
            cnn_output = x_cnn.view(x.size(0), -1)

            x_gru, _ = self.gru(x)
            gru_output = x_gru[:, -1, :]
            gru_output = gru_output * self.weight_masks["gru"].to(gru_output.device)
            activations["gru"] = gru_output.abs().mean(dim=0)

            z = torch.cat([cnn_output, gru_output], dim=1)
            z = self.relu(self._apply_masked_linear(z, self.fc1, "fc1"))
            activations["fc1"] = z.abs().mean(dim=0)

        return activations

    def get_context_activations_per_sample(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return masked per-sample activations used by NICE ContextDetector."""
        self.eval()
        if x.ndim == 2:
            x = x.unsqueeze(-1)

        with torch.no_grad():
            x_cnn = x.permute(0, 2, 1)
            x_cnn = self._apply_masked_conv(x_cnn, self.conv1, self.bn1, self.pool1, "conv1")
            conv1 = x_cnn.abs().mean(dim=2)
            x_cnn = self._apply_masked_conv(x_cnn, self.conv2, self.bn2, self.pool2, "conv2")
            conv2 = x_cnn.abs().mean(dim=2)
            x_cnn = self._apply_masked_conv(x_cnn, self.conv3, self.bn3, self.pool3, "conv3")
            conv3 = x_cnn.abs().mean(dim=2)

            x_gru, _ = self.gru(x)
            gru = x_gru[:, -1, :]
            gru = gru * self.weight_masks["gru"].to(gru.device)

        return {"conv1": conv1, "conv2": conv2, "conv3": conv3, "gru": gru.abs()}

    def get_output_and_context_activations(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Official NICE eval path: one forward for logits and context activations."""
        self.eval()
        if x.ndim == 2:
            x = x.unsqueeze(-1)

        x_cnn = x.permute(0, 2, 1)
        x_cnn = self._apply_masked_conv(x_cnn, self.conv1, self.bn1, self.pool1, "conv1")
        conv1 = x_cnn.abs().mean(dim=2)
        x_cnn = self._apply_masked_conv(x_cnn, self.conv2, self.bn2, self.pool2, "conv2")
        conv2 = x_cnn.abs().mean(dim=2)
        x_cnn = self._apply_masked_conv(x_cnn, self.conv3, self.bn3, self.pool3, "conv3")
        conv3 = x_cnn.abs().mean(dim=2)
        cnn_output = x_cnn.view(x.size(0), -1)

        x_gru, _ = self.gru(x)
        gru = x_gru[:, -1, :]
        gru = gru * self.weight_masks["gru"].to(gru.device)

        z = torch.cat([cnn_output, gru], dim=1)
        z = self.relu(self._apply_masked_linear(z, self.fc1, "fc1"))
        z = self.dropout(z)
        logits = self._apply_masked_linear(z, self.fc2, "fc2")

        return logits, {
            "conv1": conv1,
            "conv2": conv2,
            "conv3": conv3,
            "gru": gru.abs(),
        }

    def get_binary_activations(self, x: torch.Tensor, thresholds: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """Get binary activation vector for context detection.

        Returns concatenated binary vector across conv1-3 and gru layers.
        Binarized by threshold (default: > 0).

        Args:
            x: input data
            thresholds: per-layer thresholds (mean+std from episode 1)

        Returns:
            Binary activation tensor [total_neurons_across_layers]
        """
        acts = self.get_activations(x)
        binary_parts = []
        for name in ["conv1", "conv2", "conv3", "gru"]:
            act = acts[name]
            if thresholds is not None and name in thresholds:
                binary = (act > thresholds[name]).float()
            else:
                binary = (act > 0).float()
            binary_parts.append(binary.cpu())
        return torch.cat(binary_parts)

    # ========================================================================
    # Physical weight zeroing (official set_mask behavior)
    # ========================================================================

    def apply_masks_to_weights(self):
        """Physically zero weight.data where mask is 0.

        Official NICE (set_mask): weight.data *= weight_mask.data
        This ensures masked-out weights are truly zero, not just
        ignored in forward pass. Must be called after mask changes
        (drop_young_to_learner, grow_all_to_young).
        """
        layer_map = {
            "conv1": self.conv1, "conv2": self.conv2, "conv3": self.conv3,
            "fc1": self.fc1, "fc2": self.fc2,
        }
        for name, module in layer_map.items():
            if name not in self.weight_masks:
                continue
            w_mask = self.weight_masks[name].to(module.weight.device)
            module.weight.data *= w_mask
            if module.bias is not None and name in self.bias_masks:
                b_mask = self.bias_masks[name].to(module.bias.device)
                module.bias.data *= b_mask

    # ========================================================================
    # Gradient freezing for mature neurons
    # ========================================================================

    def reset_frozen_gradients(self):
        """Zero gradients for mature neurons (age > 1) after backward.

        Must be called after loss.backward() and before optimizer.step().
        This is the core anti-forgetting mechanism in NICE.
        """
        if not self.freeze_masks:
            return

        for name, param in self.named_parameters():
            if param.grad is None:
                continue

            # Extract layer name from param name (e.g., "conv1.weight" -> "conv1")
            layer_name = name.split(".")[0]

            if layer_name == "gru":
                freeze = self.freeze_masks.get("gru")
                if freeze is None or not np.any(freeze):
                    continue
                hidden = int(self._layer_dims["gru"])
                if param.dim() >= 1 and param.shape[0] == 3 * hidden:
                    gate_mask = torch.as_tensor(
                        np.tile(freeze, 3).tolist(),
                        dtype=torch.bool,
                        device=param.device,
                    )
                    param.grad.data[gate_mask] = 0.0
                continue

            conv_layer = None
            for layer, bn_name in self.BN_LAYER_MAP.items():
                if layer_name == bn_name:
                    conv_layer = layer
                    break
            if conv_layer is not None:
                freeze = self.freeze_masks.get(conv_layer)
                if freeze is None or not np.any(freeze):
                    continue
                if len(freeze) == param.shape[0]:
                    mask = torch.as_tensor(
                        freeze.tolist(), dtype=torch.bool, device=param.device
                    )
                    param.grad.data[mask] = 0.0
                continue

            if layer_name not in self.freeze_masks:
                continue

            freeze = self.freeze_masks[layer_name]  # np.ndarray, True=frozen

            if "weight" in name:
                if param.dim() >= 2:
                    # Conv1d: [out_channels, in_channels, kernel_size]
                    # Linear: [out_features, in_features]
                    if len(freeze) == param.shape[0]:
                        mask = torch.as_tensor(
                            freeze.tolist(), dtype=torch.bool, device=param.device
                        )
                        param.grad.data[mask] = 0.0
            elif "bias" in name:
                if len(freeze) == param.shape[0]:
                    mask = torch.as_tensor(
                        freeze.tolist(), dtype=torch.bool, device=param.device
                    )
                    param.grad.data[mask] = 0.0

    def freeze_bn_for_mature(self):
        """Freeze BatchNorm channels whose corresponding conv units are mature."""
        bn_map = {"conv1": self.bn1, "conv2": self.bn2, "conv3": self.bn3}
        for layer_name, bn in bn_map.items():
            if layer_name in self.unit_ranks:
                ranks = self.unit_ranks[layer_name]
                mature = torch.as_tensor(
                    (ranks >= 2).tolist(),
                    dtype=torch.bool,
                    device=bn.running_mean.device,
                )
                if mature.any():
                    prev = self._bn_frozen_units.get(layer_name)
                    if prev is None:
                        prev = torch.zeros_like(mature)
                        self._bn_running_mean_frozen[layer_name] = (
                            bn.running_mean.detach().clone()
                        )
                        self._bn_running_var_frozen[layer_name] = (
                            bn.running_var.detach().clone()
                        )
                    else:
                        prev = prev.to(mature.device)

                    newly_frozen = mature & ~prev
                    if newly_frozen.any():
                        mean = self._bn_running_mean_frozen[layer_name].to(
                            mature.device
                        )
                        var = self._bn_running_var_frozen[layer_name].to(
                            mature.device
                        )
                        mean[newly_frozen] = bn.running_mean.detach()[newly_frozen]
                        var[newly_frozen] = bn.running_var.detach()[newly_frozen]
                        self._bn_running_mean_frozen[layer_name] = mean
                        self._bn_running_var_frozen[layer_name] = var

                    self._bn_frozen_units[layer_name] = mature
                    self._restore_frozen_bn_stats(bn, layer_name)

                if mature.all():
                    bn.eval()

    # ========================================================================
    # State transfer (neuron ages + masks, separate from state_dict)
    # ========================================================================

    def get_neuron_ages_state(self) -> Dict[str, np.ndarray]:
        """Get neuron ages as serializable dict."""
        return {k: v.copy() for k, v in self.unit_ranks.items()}

    def set_neuron_ages_state(self, state: Dict[str, np.ndarray]):
        """Set neuron ages from serialized dict."""
        for k, v in state.items():
            if k in self.unit_ranks:
                self.unit_ranks[k] = np.array(v, dtype=np.int32)

    def get_masks_state(self) -> Dict[str, torch.Tensor]:
        """Get weight masks as serializable dict."""
        state = {}
        for k, v in self.weight_masks.items():
            state[f"weight_{k}"] = v.cpu().clone()
        for k, v in self.bias_masks.items():
            state[f"bias_{k}"] = v.cpu().clone()
        return state

    def set_masks_state(self, state: Dict[str, torch.Tensor]):
        """Set weight masks from serialized dict."""
        for k, v in state.items():
            if k.startswith("weight_"):
                layer = k[7:]  # remove "weight_"
                self.weight_masks[layer] = v.clone()
            elif k.startswith("bias_"):
                layer = k[5:]  # remove "bias_"
                self.bias_masks[layer] = v.clone()
