"""
DER Model - Dynamically Expandable Representation for Class Incremental Learning.

Reference:
    "DER: Dynamically Expandable Representation for Class Incremental Learning"
    Yan, Xie, He (CVPR 2021)

Architecture:
    - Each task adds a new CNNGRUBackbone feature extractor
    - Old extractors are frozen after their task
    - Channel-level mask pruning (HAT-based) keeps model compact
    - Super-feature = concatenation of all masked extractor outputs
    - Unified classifier maps super-feature → all classes
    - Auxiliary classifier regularizes new extractor during training

Paper Equations Implemented:
    Eq.1: Super-feature construction u = [Phi_{t-1}(x), F_t(x)]
    Eq.2: Prediction p = Softmax(H_t(u))
    Eq.5: Mask application f_hat = f ⊙ m
    Eq.6: Mask parameterization m_l = σ(s · e_l)
    Eq.7: Pruned super-feature u_tilde = [F_1^P(x), ..., F_t^P(x)]
    Eq.8: Annealing schedule s = 1/s_max + (s_max - 1/s_max) * (b-1)/(B-1)
    Eq.9: Gradient compensation for mask embeddings
    Eq.10: Sparsity loss L_S
"""

import copy
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .cnn_gru import CNN_GRU_Model


class CNNGRUBackbone(CNN_GRU_Model):
    """
    Feature-only extractor for DER.

    Inherits CNN_GRU_Model backbone (Conv1d×3 + GRU), removes MLP head.
    Adds per-layer channel mask embeddings for HAT-based pruning.

    Architecture:
        Conv1d(in, 64) → BN → ReLU → MaxPool   [layer 0, channels=64]
        Conv1d(64, 128) → BN → ReLU → MaxPool   [layer 1, channels=128]
        Conv1d(128, 256) → BN → ReLU → MaxPool  [layer 2, channels=256]
        GRU(in, 100, num_layers=2)               [output_size=100]
        → concat [cnn_flat, gru_last] → feature vector

    Mask is applied per-layer AFTER activation (Eq.5: f_hat = f ⊙ m):
        mask_0: R^64   (after conv1+bn1+relu+pool1)
        mask_1: R^128  (after conv2+bn2+relu+pool2)
        mask_2: R^256  (after conv3+bn3+relu+pool3)
        mask_3: R^100  (after GRU last hidden state)
    """

    # Channel counts for each maskable layer
    MASK_CHANNELS = [64, 128, 256, 100]  # conv1, conv2, conv3, gru

    def __init__(self, input_shape):
        super().__init__(input_shape, num_classes=2)  # dummy num_classes
        # Remove MLP head — not needed for feature extraction
        del self.fc1, self.fc2, self.dropout
        self.output_dim = self.cnn_output_size + self.gru_output_size

        # Per-layer mask embeddings e_l (Eq.6: m_l = σ(s · e_l))
        # Initialized to 0 → σ(0) = 0.5 → all channels half-active at start
        self.mask_embeds = nn.ParameterList([
            nn.Parameter(torch.zeros(c)) for c in self.MASK_CHANNELS
        ])

    def forward(self, x, s: Optional[float] = None):
        """
        Forward pass with optional channel masking.

        Args:
            x: Input tensor [batch, seq_len] or [batch, seq_len, features]
            s: Mask annealing parameter (Eq.8). None = no masking (inference default).

        Returns:
            Feature vector [batch, output_dim]
        """
        # Input handling (same as CNN_GRU_Model)
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        batch_size = x.size(0)

        # --- CNN pathway with per-layer masking (Eq.5) ---
        x_cnn = x.permute(0, 2, 1)  # [B, features, seq_len]

        # Layer 0: Conv1d(in, 64)
        x_cnn = self.pool1(self.relu(self.bn1(self.conv1(x_cnn))))
        if s is not None:
            mask_0 = torch.sigmoid(s * self.mask_embeds[0])  # [64]
            x_cnn = x_cnn * mask_0.view(1, -1, 1)  # broadcast [B, 64, L]

        # Layer 1: Conv1d(64, 128)
        x_cnn = self.pool2(self.relu(self.bn2(self.conv2(x_cnn))))
        if s is not None:
            mask_1 = torch.sigmoid(s * self.mask_embeds[1])  # [128]
            x_cnn = x_cnn * mask_1.view(1, -1, 1)

        # Layer 2: Conv1d(128, 256)
        x_cnn = self.pool3(self.relu(self.bn3(self.conv3(x_cnn))))
        if s is not None:
            mask_2 = torch.sigmoid(s * self.mask_embeds[2])  # [256]
            x_cnn = x_cnn * mask_2.view(1, -1, 1)

        cnn_output = x_cnn.view(batch_size, -1)  # [B, 256 * cnn_len]

        # --- GRU pathway with masking ---
        x_gru, _ = self.gru(x)
        gru_output = x_gru[:, -1, :]  # [B, 100]
        if s is not None:
            mask_3 = torch.sigmoid(s * self.mask_embeds[3])  # [100]
            gru_output = gru_output * mask_3  # [B, 100]

        return torch.cat([cnn_output, gru_output], dim=1)

    def forward_with_masks(self, x, frozen_masks: List[torch.Tensor]):
        """
        Forward pass using pre-computed frozen (binarized) masks.

        Used for old extractors whose masks are frozen after their task.

        Args:
            x: Input tensor
            frozen_masks: List of 4 mask tensors [mask_0, mask_1, mask_2, mask_3]

        Returns:
            Feature vector [batch, output_dim]
        """
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        batch_size = x.size(0)

        x_cnn = x.permute(0, 2, 1)

        x_cnn = self.pool1(self.relu(self.bn1(self.conv1(x_cnn))))
        x_cnn = x_cnn * frozen_masks[0].view(1, -1, 1)

        x_cnn = self.pool2(self.relu(self.bn2(self.conv2(x_cnn))))
        x_cnn = x_cnn * frozen_masks[1].view(1, -1, 1)

        x_cnn = self.pool3(self.relu(self.bn3(self.conv3(x_cnn))))
        x_cnn = x_cnn * frozen_masks[2].view(1, -1, 1)

        cnn_output = x_cnn.view(batch_size, -1)

        x_gru, _ = self.gru(x)
        gru_output = x_gru[:, -1, :] * frozen_masks[3]

        return torch.cat([cnn_output, gru_output], dim=1)

    def get_masks(self, s: float) -> List[torch.Tensor]:
        """Get current mask values at annealing parameter s."""
        return [torch.sigmoid(s * e) for e in self.mask_embeds]

    def compute_sparsity_loss(self, s: float) -> torch.Tensor:
        """
        Compute sparsity loss L_S for this extractor (Eq.10).

        Paper Eq.10:
            L_S = [ Σ_l K_l · ||m_{l-1}||₁ · ||m_l||₁ ]
                / [ Σ_l K_l · c_{l-1}   · c_l         ]

        Adapted for our CNN-GRU architecture:
            CNN path: input(num_features) → conv1(64) → conv2(128) → conv3(256)
            GRU path: input(num_features) → GRU(100) (parallel, separate term)

        Returns:
            Scalar sparsity loss
        """
        masks = self.get_masks(s)

        # CNN path: consecutive mask product per Eq.10
        channels = [self.num_features, 64, 128, 256]
        kernel_sizes = [3, 3, 3]

        numerator = torch.tensor(0.0, device=masks[0].device)
        denominator = 0.0

        # Input "mask" = all ones (input channels are always active)
        prev_mask_l1 = float(self.num_features)

        for l in range(3):
            K_l = kernel_sizes[l]
            mask_l1 = masks[l].sum()
            c_prev = channels[l]
            c_curr = channels[l + 1]

            numerator = numerator + K_l * prev_mask_l1 * mask_l1
            denominator += K_l * c_prev * c_curr

            prev_mask_l1 = mask_l1

        # GRU path: parallel from input, use same Eq.10 formula
        # input(num_features) → GRU output(100), K=1 (not conv but weight matrix)
        gru_mask = masks[3]  # [100]
        gru_c_in = self.num_features
        gru_c_out = 100
        numerator = numerator + float(gru_c_in) * gru_mask.sum()
        denominator += gru_c_in * gru_c_out

        return numerator / max(denominator, 1.0)

    def compensate_mask_gradients(self, s: float):
        """
        Gradient compensation for mask embeddings (Eq.9).

        Paper Eq.9:
            g_hat = [σ(e) · (1 - σ(e))] / [s · σ(s·e) · (1 - σ(s·e))] · g

        This removes the influence of annealing factor s from gradient magnitude,
        ensuring stable mask learning regardless of the current s value.

        Must be called AFTER loss.backward() and BEFORE optimizer.step().
        """
        for embed in self.mask_embeds:
            if embed.grad is None:
                continue

            e = embed.data
            sig_e = torch.sigmoid(e)
            sig_se = torch.sigmoid(s * e)

            # Numerator: σ(e)(1 - σ(e)) = sigmoid derivative at e
            numer = sig_e * (1.0 - sig_e)
            # Denominator: s · σ(se)(1 - σ(se)) = s × sigmoid derivative at s*e
            # Use 1e-3 (not 1e-12): when mask_embeds go negative, σ(s·e)→0 and
            # denominator→0 causing extreme amplification (positive feedback loop
            # that drives all masks to 0). Larger epsilon prevents this.
            denom = s * sig_se * (1.0 - sig_se) + 1e-3

            # Clamp compensation to [0, 100]: never amplify gradients more than
            # 100x. Without this clamp, compensation can reach 1e8+ for large
            # negative embeds, causing all masks to collapse.
            compensation = (numer / denom).clamp_(max=100.0)
            embed.grad.data.mul_(compensation)


class DERModel(nn.Module):
    """
    Dynamically Expandable Representation model.

    Manages multiple CNNGRUBackbone extractors (one per task),
    channel-level masks, and expandable classifiers.

    Paper Architecture:
        Task t: u = [F_1^P(x), F_2^P(x), ..., F_{t-1}^P(x), F_t(x)]  (Eq.7)
                 y_hat = argmax Softmax(H_t(u))                          (Eq.2)

    Attributes:
        extractors: ModuleList of CNNGRUBackbone (one per task)
        classifier: Linear (super_feat_dim → num_classes), grows with tasks
        aux_classifier: Linear (feat_dim → |Y_t|+1), recreated each task
        frozen_masks: Dict[int, List[Tensor]] — binarized masks for old extractors
        current_task: Current task index (-1 = no task added yet)
    """

    def __init__(self, input_shape, num_classes: int):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes

        # Extractor management
        self.extractors = nn.ModuleList()
        self.classifier: Optional[nn.Linear] = None
        self.aux_classifier: Optional[nn.Linear] = None

        # Feature dimensions (set on first add_task)
        self.feat_dim: Optional[int] = None

        # Task tracking
        self.current_task: int = -1
        # s_max used for inference binarization (paper Eq.7: F_t^P uses binarized masks)
        self._s_max: float = 15.0
        # Frozen masks stored as registered buffers for state_dict serialization
        # Access via _get_frozen_masks(task_id) → List[Tensor]
        self._num_mask_layers = 4  # conv1, conv2, conv3, gru

        # Dropout for classifier (as in original CNN_GRU_Model)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

        # Device sentinel: moves with the module when model.to(device) is called.
        # add_task() uses this to create new submodules on the correct device,
        # since nn.ModuleList.append() does NOT automatically move new modules.
        self.register_buffer('_device_sentinel', torch.zeros(1))

    def _set_frozen_masks(self, task_id: int, masks: List[torch.Tensor]):
        """Store frozen masks as registered buffers."""
        for layer_idx, mask in enumerate(masks):
            self.register_buffer(
                f"frozen_mask_{task_id}_{layer_idx}",
                mask.clone().detach(),
            )

    def _get_frozen_masks(self, task_id: int) -> List[torch.Tensor]:
        """Retrieve frozen masks for a given task from registered buffers."""
        masks = []
        for layer_idx in range(self._num_mask_layers):
            buf = getattr(self, f"frozen_mask_{task_id}_{layer_idx}", None)
            if buf is None:
                raise RuntimeError(f"Frozen mask not found for task {task_id}, layer {layer_idx}")
            masks.append(buf)
        return masks

    def _has_frozen_masks(self, task_id: int) -> bool:
        """Check if frozen masks exist for a task."""
        return hasattr(self, f"frozen_mask_{task_id}_0")

    def _get_device(self) -> torch.device:
        """Get the current device of this module via the device sentinel."""
        return self._device_sentinel.device

    @property
    def num_extractors(self) -> int:
        return len(self.extractors)

    @property
    def super_feat_dim(self) -> int:
        """Current super-feature dimensionality."""
        if self.feat_dim is None:
            return 0
        return self.num_extractors * self.feat_dim

    def add_task(self, new_classes: List[int], s_max: float = 15.0):
        """
        Expand model for a new task.

        Paper Section 3.1:
        1. Freeze old extractor + binarize its masks
        2. Create new extractor (initialized from previous for transfer)
        3. Expand unified classifier (inherit old weights)
        4. Create auxiliary classifier (|Y_t| + 1 classes)

        Args:
            new_classes: List of new class IDs in this task
            s_max: Max annealing parameter for mask binarization
        """
        self.current_task += 1
        self._s_max = s_max  # keep for inference binarization (Eq.7)

        # --- Step 1: Freeze previous extractor + binarize masks ---
        if self.current_task > 0:
            prev_ext = self.extractors[self.current_task - 1]

            # Binarize masks at s_max (Eq.6 with s→∞ gives binary mask)
            with torch.no_grad():
                binarized = [
                    torch.sigmoid(s_max * e.data)
                    for e in prev_ext.mask_embeds
                ]
                self._set_frozen_masks(self.current_task - 1, binarized)

            # Freeze all parameters (weights + mask embeds + BN statistics)
            for param in prev_ext.parameters():
                param.requires_grad = False
            # Freeze BN running stats by setting to eval mode permanently
            prev_ext.eval()

        # --- Step 2: Create new extractor ---
        # Use device sentinel to ensure new backbone is on the same device
        # as the rest of the model (critical for multi-GPU and AMP training).
        device = self._get_device()
        new_backbone = CNNGRUBackbone(self.input_shape).to(device)

        # Paper: "initialize F_t from F_{t-1}" for fast adaptation
        if self.current_task > 0:
            prev_ext = self.extractors[self.current_task - 1]
            # Copy backbone weights (conv, bn, gru) but NOT mask embeds
            src_state = prev_ext.state_dict()
            tgt_state = new_backbone.state_dict()
            for k in tgt_state:
                if 'mask_embeds' not in k and k in src_state:
                    tgt_state[k] = src_state[k].clone()
            new_backbone.load_state_dict(tgt_state)
            # Reset mask embeds to zeros (fresh masks for new task)
            for embed in new_backbone.mask_embeds:
                nn.init.zeros_(embed.data)

        self.extractors.append(new_backbone)

        if self.feat_dim is None:
            self.feat_dim = new_backbone.output_dim

        # --- Step 3: Expand unified classifier ---
        new_super_dim = self.num_extractors * self.feat_dim
        new_classifier = nn.Linear(new_super_dim, self.num_classes).to(device)

        if self.classifier is not None:
            # Inherit old classifier weights (paper Section 3.1)
            with torch.no_grad():
                _, old_in = self.classifier.weight.shape
                # Old feature columns: copy weights
                new_classifier.weight[:, :old_in] = self.classifier.weight.data
                new_classifier.bias[:] = self.classifier.bias.data
                # New feature columns stay randomly initialized

        self.classifier = new_classifier

        # --- Step 4: Auxiliary classifier ---
        # Paper: |Y_t| + 1 classes (new classes + 1 "other" for all old)
        # At task 0, no "other" class needed (λ_a = 0 anyway)
        n_aux_classes = len(new_classes) + (1 if self.current_task > 0 else 0)
        self.aux_classifier = nn.Linear(self.feat_dim, n_aux_classes).to(device)

        print(f"  DERModel: Task {self.current_task} | "
              f"extractors={self.num_extractors} | "
              f"super_feat_dim={new_super_dim} | "
              f"aux_classes={n_aux_classes}")

    def get_super_feature(self, x: torch.Tensor, s: Optional[float] = None) -> torch.Tensor:
        """
        Compute super-feature by forwarding through all extractors (Eq.1, Eq.7).

        Args:
            x: Input tensor [batch, seq_len, ...]
            s: Annealing parameter for current extractor's masks.
               Old extractors always use frozen (binarized) masks.
               If None (inference), current extractor uses binarized masks at s_max
               to match paper's pruned inference: u = Phi_t^P(x) = [F_1^P, ..., F_t^P] (Eq.7).

        Returns:
            Super-feature [batch, num_extractors * feat_dim]
        """
        features = []

        for t, extractor in enumerate(self.extractors):
            if t < self.current_task:
                # Old extractor: frozen weights + frozen (binarized) masks (Eq.7)
                frozen_masks = self._get_frozen_masks(t)
                feat = extractor.forward_with_masks(x, frozen_masks)
            else:
                # Current extractor:
                # - Training (s given): use annealed soft masks (Eq.6, Eq.8)
                # - Inference (s=None): use binarized masks at s_max (paper Eq.7, F_t^P)
                inference_s = s if s is not None else self._s_max
                feat = extractor(x, s=inference_s)

            features.append(feat)

        return torch.cat(features, dim=1)

    def forward(self, x: torch.Tensor, s: Optional[float] = None) -> torch.Tensor:
        """
        Full forward: super-feature → classifier → logits (Eq.2).

        Args:
            x: Input [batch, seq_len, ...]
            s: Annealing parameter for masks

        Returns:
            Logits [batch, num_classes]
        """
        u = self.get_super_feature(x, s)
        u = self.relu(u)
        u = self.dropout(u)
        return self.classifier(u)

    def forward_aux(self, x: torch.Tensor, s: Optional[float] = None) -> torch.Tensor:
        """
        Auxiliary forward: current extractor only → aux_classifier.

        Paper: H_t^a operates on F_t(x) with label space |Y_t| + 1.

        Args:
            x: Input [batch, seq_len, ...]
            s: Annealing parameter

        Returns:
            Auxiliary logits [batch, |Y_t| + 1]
        """
        feat = self.extractors[self.current_task](x, s=s)
        return self.aux_classifier(feat)

    def get_trainable_params(self) -> List[nn.Parameter]:
        """
        Get trainable parameters for Stage 1 (representation learning).

        Trainable: current extractor (weights + mask embeds) + classifier + aux_classifier
        Frozen: all old extractors
        """
        params = []

        # Current extractor (backbone + mask embeds)
        for p in self.extractors[self.current_task].parameters():
            if p.requires_grad:
                params.append(p)

        # Unified classifier
        if self.classifier is not None:
            params.extend(self.classifier.parameters())

        # Auxiliary classifier
        if self.aux_classifier is not None:
            params.extend(self.aux_classifier.parameters())

        return params

    def get_classifier_params(self) -> List[nn.Parameter]:
        """
        Get classifier-only parameters for Stage 2 (balanced finetuning).

        Paper: Stage 2 freezes all extractors, only trains H_t.
        """
        if self.classifier is not None:
            return list(self.classifier.parameters())
        return []

    def weight_align(self, num_new_classes: int):
        """
        Weight Alignment (WA) — rescale new-class classifier weights.

        After Stage 1, new-class weights tend to have larger norms than old-class
        weights (because new data dominates training), causing prediction bias
        toward new classes. WA corrects this by scaling new weights to match
        old weight norms.

        Reference: PyCIL DERNet.weight_align()
            gamma = mean(||w_old||) / mean(||w_new||)
            w_new *= gamma

        Must be called AFTER Stage 1 training, BEFORE Stage 2.

        Args:
            num_new_classes: Number of new classes in current task
        """
        if self.classifier is None or num_new_classes <= 0:
            return
        if self.classifier.out_features <= num_new_classes:
            return  # First task — no old classes to align with

        weights = self.classifier.weight.data
        old_weights = weights[:-num_new_classes, :]
        new_weights = weights[-num_new_classes:, :]

        old_norm = torch.norm(old_weights, p=2, dim=1).mean()
        new_norm = torch.norm(new_weights, p=2, dim=1).mean()

        if new_norm < 1e-8:
            return  # Avoid division by zero

        gamma = old_norm / new_norm
        print(f"  DER Weight Alignment: gamma={gamma:.4f}")
        self.classifier.weight.data[-num_new_classes:, :] *= gamma

    def reset_classifier(self):
        """
        Re-initialize classifier H_t with random weights.

        Paper Section 3.2 (Stage 2):
            "Re-initialize classifier H_t with random weights"

        Must be called on server BEFORE distributing model to clients for Stage 2.
        Ensures Stage 2 starts fresh, not biased from Stage 1 representation training.
        """
        if self.classifier is None:
            return
        nn.init.kaiming_uniform_(self.classifier.weight, a=0, nonlinearity='relu')
        nn.init.zeros_(self.classifier.bias)

    def freeze_all_extractors(self):
        """Freeze ALL extractors (for Stage 2 classifier-only training)."""
        for ext in self.extractors:
            for p in ext.parameters():
                p.requires_grad = False
            ext.eval()

    def unfreeze_current_extractor(self):
        """Restore current extractor to trainable (after Stage 2)."""
        ext = self.extractors[self.current_task]
        for p in ext.parameters():
            p.requires_grad = True
        ext.train()

    def compensate_mask_gradients(self, s: float):
        """
        Apply gradient compensation to current extractor's mask embeds (Eq.9).

        Must be called AFTER loss.backward() and BEFORE optimizer.step().
        """
        self.extractors[self.current_task].compensate_mask_gradients(s)

    def get_sparsity_loss(self, s: float) -> torch.Tensor:
        """
        Compute sparsity loss L_S for current extractor (Eq.10).
        """
        return self.extractors[self.current_task].compute_sparsity_loss(s)

    def get_mask_stats(self) -> Dict:
        """Get mask statistics for logging/debugging."""
        stats = {}
        for t, ext in enumerate(self.extractors):
            if t < self.current_task:
                # Frozen: use stored masks
                masks = self._get_frozen_masks(t)
            else:
                # Current: use raw sigmoid
                masks = [torch.sigmoid(e.data) for e in ext.mask_embeds]

            active_ratio = sum(
                (m > 0.5).float().mean().item() for m in masks
            ) / len(masks)
            stats[f"task_{t}_active_ratio"] = active_ratio

        return stats
