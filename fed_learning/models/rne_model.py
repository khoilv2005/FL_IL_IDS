"""
RNE Model - Recurrent Network Expansion for class-incremental learning.

This module maps the original image-based RNE design to the repo's 1D CNN-GRU
IDS backbone:
- add one task expert per incremental task,
- freeze old experts,
- pass mapped intermediate feature maps from expert t-1 to expert t,
- use causal task-level classifier heads.
"""

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn_gru import CNN_GRU_Model
from .der_model import DERModel


class CRL1D(nn.Module):
    """1D analogue of the original RNE CRL block."""

    def __init__(self, channels: int, length: int):
        super().__init__()
        kernel = 3 if length > 8 else 1
        padding = 1 if kernel == 3 else 0
        self.conv_a = nn.Conv1d(channels, channels, kernel, padding=padding, bias=False)
        self.conv_b = nn.Conv1d(channels, channels, 1, bias=False)
        self.ln_a = nn.LayerNorm(length)
        self.ln_b = nn.LayerNorm(length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.ln_a(self.conv_a(x)), inplace=True)
        out = F.relu(self.ln_b(self.conv_b(out)), inplace=True)
        return x + out


class VectorCRL(nn.Module):
    """Residual mapping for the GRU hidden vector."""

    def __init__(self, dim: int):
        super().__init__()
        self.fc_a = nn.Linear(dim, dim, bias=False)
        self.fc_b = nn.Linear(dim, dim, bias=False)
        self.ln_a = nn.LayerNorm(dim)
        self.ln_b = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.ln_a(self.fc_a(x)), inplace=True)
        out = F.relu(self.ln_b(self.fc_b(out)), inplace=True)
        return x + out


class RNEFeatureMapper(nn.Module):
    """
    Shared mapping modules MM at key layers.

    Original RNE uses one shared module per depth between adjacent experts. For
    CNN-GRU, key layers are the three downsampling CNN blocks and final GRU
    vector.
    """

    def __init__(
        self,
        input_shape,
        channels: Sequence[int] = (64, 128, 256, 100),
    ):
        super().__init__()
        seq_len = input_shape[0] if isinstance(input_shape, tuple) else int(input_shape)
        l1 = max(1, seq_len // 2)
        l2 = max(1, l1 // 2)
        l3 = max(1, l2 // 2)
        c1, c2, c3, c4 = [int(c) for c in channels]
        self.blocks = nn.ModuleList(
            [
                CRL1D(c1, l1),
                CRL1D(c2, l2),
                CRL1D(c3, l3),
                VectorCRL(c4),
            ]
        )

    def forward(self, maps: List[torch.Tensor]) -> List[torch.Tensor]:
        return [block(feature_map) for block, feature_map in zip(self.blocks, maps)]


def _reset_simple_linear(layer: nn.Linear) -> None:
    nn.init.kaiming_uniform_(layer.weight, nonlinearity="linear")
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)

class RNECNNGRUBackbone(CNN_GRU_Model):
    """CNN-GRU expert that can receive recurrent maps from previous expert."""

    def __init__(self, input_shape):
        super().__init__(input_shape, num_classes=2)
        del self.fc1, self.fc2, self.dropout
        self.output_dim = self.cnn_output_size + self.gru_output_size

    def _prepare_input(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        return x, x.size(0)

    def forward_maps(
        self,
        x: torch.Tensor,
        recurrent_maps: Optional[List[torch.Tensor]] = None,
        s: Optional[float] = None,
        frozen_masks: Optional[List[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, List[torch.Tensor]]:
        x, batch_size = self._prepare_input(x)

        x_cnn = x.permute(0, 2, 1)

        x1 = self.pool1(self.relu(self.bn1(self.conv1(x_cnn))))
        if recurrent_maps is not None:
            x1 = F.relu(x1 + recurrent_maps[0], inplace=True)

        x2 = self.pool2(self.relu(self.bn2(self.conv2(x1))))
        if recurrent_maps is not None:
            x2 = F.relu(x2 + recurrent_maps[1], inplace=True)

        x3 = self.pool3(self.relu(self.bn3(self.conv3(x2))))
        if recurrent_maps is not None:
            x3 = F.relu(x3 + recurrent_maps[2], inplace=True)

        x_gru, _ = self.gru(x)
        gru = x_gru[:, -1, :]
        if recurrent_maps is not None:
            gru = F.relu(gru + recurrent_maps[3], inplace=True)

        cnn_output = x3.view(batch_size, -1)
        feature = torch.cat([cnn_output, gru], dim=1)
        return feature, [x1, x2, x3, gru]

    def forward(self, x, s: Optional[float] = None):
        feature, _ = self.forward_maps(x, s=s)
        return feature

    def forward_with_masks(self, x, frozen_masks: List[torch.Tensor]):
        feature, _ = self.forward_maps(x)
        return feature

class RNEMapAdapter(nn.Module):
    """Map full backbone maps to compressed expert map sizes."""

    def __init__(
        self,
        input_shape,
        in_channels: Sequence[int] = (64, 128, 256, 100),
        out_channels: Sequence[int] = (16, 32, 64, 25),
    ):
        super().__init__()
        seq_len = input_shape[0] if isinstance(input_shape, tuple) else int(input_shape)
        lengths = [
            max(1, seq_len // 2),
            max(1, seq_len // 4),
            max(1, seq_len // 8),
        ]
        c_in = [int(c) for c in in_channels]
        c_out = [int(c) for c in out_channels]
        self.conv_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(c_in[i], c_out[i], 1, bias=False),
                    nn.LayerNorm(lengths[i]),
                    nn.ReLU(inplace=True),
                )
                for i in range(3)
            ]
        )
        self.vector = nn.Sequential(
            nn.Linear(c_in[3], c_out[3], bias=False),
            nn.LayerNorm(c_out[3]),
            nn.ReLU(inplace=True),
        )

    def forward(self, maps: List[torch.Tensor]) -> List[torch.Tensor]:
        return [
            self.conv_blocks[0](maps[0]),
            self.conv_blocks[1](maps[1]),
            self.conv_blocks[2](maps[2]),
            self.vector(maps[3]),
        ]

class RNECompressedCNNGRUBackbone(nn.Module):
    """Compressed task expert used by RNE-compress."""

    def __init__(
        self,
        input_shape,
        channels: Sequence[int] = (16, 32, 64),
        gru_hidden: int = 25,
    ):
        super().__init__()
        if isinstance(input_shape, tuple):
            seq_length = int(input_shape[0])
            num_features = int(input_shape[1]) if len(input_shape) > 1 else 1
        else:
            seq_length = int(input_shape)
            num_features = 1

        c1, c2, c3 = [int(c) for c in channels]
        self.input_shape = input_shape
        self.num_features = num_features

        self.conv1 = nn.Conv1d(num_features, c1, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(c1)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(c1, c2, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(c2)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(c2, c3, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(c3)
        self.pool3 = nn.MaxPool1d(2)
        self.relu = nn.ReLU()

        cnn_len = seq_length
        for _ in range(3):
            cnn_len = max(1, cnn_len // 2)
        self.cnn_output_size = c3 * cnn_len
        self.gru = nn.GRU(num_features, int(gru_hidden), num_layers=2, batch_first=True)
        self.gru_output_size = int(gru_hidden)
        self.output_dim = self.cnn_output_size + self.gru_output_size

    def _prepare_input(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        return x, x.size(0)

    def forward_maps(
        self,
        x: torch.Tensor,
        recurrent_maps: Optional[List[torch.Tensor]] = None,
        s: Optional[float] = None,
        frozen_masks: Optional[List[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, List[torch.Tensor]]:
        x, batch_size = self._prepare_input(x)
        x_cnn = x.permute(0, 2, 1)

        x1 = self.pool1(self.relu(self.bn1(self.conv1(x_cnn))))
        if recurrent_maps is not None:
            x1 = F.relu(x1 + recurrent_maps[0], inplace=True)

        x2 = self.pool2(self.relu(self.bn2(self.conv2(x1))))
        if recurrent_maps is not None:
            x2 = F.relu(x2 + recurrent_maps[1], inplace=True)

        x3 = self.pool3(self.relu(self.bn3(self.conv3(x2))))
        if recurrent_maps is not None:
            x3 = F.relu(x3 + recurrent_maps[2], inplace=True)

        x_gru, _ = self.gru(x)
        gru = x_gru[:, -1, :]
        if recurrent_maps is not None:
            gru = F.relu(gru + recurrent_maps[3], inplace=True)

        return torch.cat([x3.view(batch_size, -1), gru], dim=1), [x1, x2, x3, gru]

    def forward(self, x, s: Optional[float] = None):
        feature, _ = self.forward_maps(x, s=s)
        return feature

    def forward_with_masks(self, x, frozen_masks: List[torch.Tensor]):
        feature, _ = self.forward_maps(x)
        return feature

    def compute_sparsity_loss(self, s: float) -> torch.Tensor:
        return torch.tensor(0.0, device=self.conv1.weight.device)

    def compensate_mask_gradients(self, s: float):
        return None


class RNEModel(DERModel):
    """
    Recurrent Network Expansion with causal decoupled classifier heads.

    Head i receives features from experts 0..i; old heads never consume future
    expert features, matching the classifier in Fig. 5(c) of the RNE paper.
    """

    def __init__(self, input_shape, num_classes: int, recurrent_scale: float = 1.0):
        super().__init__(input_shape, num_classes)
        self.classifier = None
        self.classifier_heads = nn.ModuleList()
        self.task_classes_history: List[List[int]] = []
        self.recurrent_mapper = RNEFeatureMapper(input_shape)
        self.recurrent_scale = float(recurrent_scale)
        self.to(self._get_device())

    def _new_backbone(self) -> RNECNNGRUBackbone:
        return RNECNNGRUBackbone(self.input_shape).to(self._get_device())

    def add_task(self, new_classes: List[int], s_max: float = 15.0):
        self.current_task += 1
        self._s_max = s_max
        new_classes = [int(c) for c in new_classes]

        if self.current_task > 0:
            prev_ext = self.extractors[self.current_task - 1]
            for param in prev_ext.parameters():
                param.requires_grad = False
            prev_ext.eval()

        new_backbone = self._new_backbone()
        if self.current_task > 0:
            prev_ext = self.extractors[self.current_task - 1]
            src_state = prev_ext.state_dict()
            tgt_state = new_backbone.state_dict()
            for key in tgt_state:
                if key in src_state:
                    tgt_state[key] = src_state[key].clone()
            new_backbone.load_state_dict(tgt_state)

        self.extractors.append(new_backbone)
        if self.feat_dim is None:
            self.feat_dim = new_backbone.output_dim

        self.task_classes_history.append(new_classes)
        head = nn.Linear(self.num_extractors * self.feat_dim, len(new_classes)).to(
            self._get_device()
        )
        _reset_simple_linear(head)
        self.classifier_heads.append(head)

        n_aux_classes = len(new_classes) + (1 if self.current_task > 0 else 0)
        self.aux_classifier = nn.Linear(self.feat_dim, n_aux_classes).to(self._get_device())
        _reset_simple_linear(self.aux_classifier)

        print(
            f"  RNEModel: Task {self.current_task} | "
            f"experts={self.num_extractors} | "
            f"head_in={self.num_extractors * self.feat_dim} | "
            f"head_out={len(new_classes)} | aux_classes={n_aux_classes}"
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            for task_idx, extractor in enumerate(self.extractors):
                if task_idx < self.current_task:
                    extractor.eval()
        return self

    def get_feature_sequence(
        self, x: torch.Tensor, s: Optional[float] = None
    ) -> List[torch.Tensor]:
        features: List[torch.Tensor] = []
        prev_maps = None
        for task_idx, extractor in enumerate(self.extractors):
            recurrent_maps = None
            if prev_maps is not None:
                recurrent_maps = [
                    self.recurrent_scale * mapped
                    for mapped in self.recurrent_mapper(prev_maps)
                ]

            if task_idx < self.current_task:
                feature, maps = extractor.forward_maps(
                    x,
                    recurrent_maps=recurrent_maps,
                )
            else:
                feature, maps = extractor.forward_maps(
                    x,
                    recurrent_maps=recurrent_maps,
                )
            features.append(feature)
            prev_maps = maps
        return features

    def get_super_feature(self, x: torch.Tensor, s: Optional[float] = None) -> torch.Tensor:
        return torch.cat(self.get_feature_sequence(x, s=s), dim=1)

    def extract_vector(self, x: torch.Tensor, s: Optional[float] = None) -> torch.Tensor:
        return self.get_super_feature(x, s=s)

    def _classify_feature_sequence(self, features: List[torch.Tensor]) -> torch.Tensor:
        if not self.classifier_heads:
            raise RuntimeError("RNEModel has no classifier heads. Call add_task() first.")

        batch_size = features[0].shape[0]
        fill_value = -1e4 if features[0].dtype in (torch.float16, torch.bfloat16) else -1e9
        logits = features[0].new_full((batch_size, self.num_classes), fill_value)
        for task_idx, head in enumerate(self.classifier_heads):
            head_input = torch.cat(features[: task_idx + 1], dim=1)
            head_logits = head(head_input)
            for col_idx, class_id in enumerate(self.task_classes_history[task_idx]):
                if 0 <= class_id < self.num_classes:
                    logits[:, class_id] = head_logits[:, col_idx]
        return logits

    def get_mask_stats(self) -> dict:
        return {}

    def classify_features(self, super_features: torch.Tensor) -> torch.Tensor:
        features = [
            super_features[:, i * self.feat_dim : (i + 1) * self.feat_dim]
            for i in range(self.num_extractors)
        ]
        return self._classify_feature_sequence(features)

    def forward(
        self, x: torch.Tensor, s: Optional[float] = None, mode: Optional[str] = None
    ) -> torch.Tensor:
        if mode == "fc":
            return self.classify_features(x)
        return self._classify_feature_sequence(self.get_feature_sequence(x, s=s))

    def forward_aux(self, x: torch.Tensor, s: Optional[float] = None) -> torch.Tensor:
        feat = self.get_feature_sequence(x, s=s)[-1]
        return self.aux_classifier(feat)

    def get_trainable_params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        if self.current_task >= 0:
            params.extend(
                p for p in self.extractors[self.current_task].parameters() if p.requires_grad
            )
        params.extend(p for p in self.recurrent_mapper.parameters() if p.requires_grad)
        for head in self.classifier_heads:
            params.extend(head.parameters())
        if self.aux_classifier is not None:
            params.extend(self.aux_classifier.parameters())
        return params

    def get_classifier_params(self) -> List[nn.Parameter]:
        return [p for head in self.classifier_heads for p in head.parameters()]

    def scale_old_head_gradients(self, scale: float):
        for head in self.classifier_heads[:-1]:
            for param in head.parameters():
                if param.grad is not None:
                    param.grad.mul_(scale)

    def compensate_mask_gradients(self, s: float):
        return None

    def get_sparsity_loss(self, s: float) -> torch.Tensor:
        return torch.tensor(0.0, device=self._get_device())

    def reset_classifier(self):
        for head in self.classifier_heads:
            _reset_simple_linear(head)

    def freeze_all_extractors(self):
        super().freeze_all_extractors()
        for param in self.recurrent_mapper.parameters():
            param.requires_grad = False
        self.recurrent_mapper.eval()

    def unfreeze_current_extractor(self):
        super().unfreeze_current_extractor()
        for param in self.recurrent_mapper.parameters():
            param.requires_grad = True
        self.recurrent_mapper.train()

    def weight_align(self, num_new_classes: int):
        if len(self.classifier_heads) < 2:
            return
        old_norms = [
            torch.norm(head.weight.data, p=2, dim=1).mean()
            for head in self.classifier_heads[:-1]
        ]
        old_norm = torch.stack(old_norms).mean()
        new_head = self.classifier_heads[-1]
        new_norm = torch.norm(new_head.weight.data, p=2, dim=1).mean()
        if new_norm < 1e-8:
            return
        gamma = old_norm / new_norm
        print(f"  RNE Weight Alignment: gamma={gamma:.4f}")
        new_head.weight.data.mul_(gamma)

    def clone_empty_like(self) -> "RNEModel":
        return RNEModel(
            self.input_shape,
            self.num_classes,
            recurrent_scale=self.recurrent_scale,
        )

class RNECompressModel(RNEModel):
    """
    RNE-compress variant from the original source, adapted to CNN-GRU.

    It keeps a shared full backbone, projects its feature maps into smaller task
    experts, then uses the same recurrent expert chain and causal heads as RNE.
    """

    def __init__(
        self,
        input_shape,
        num_classes: int,
        recurrent_scale: float = 1.0,
        compressed_channels: Sequence[int] = (16, 32, 64, 25),
    ):
        self.compressed_channels = tuple(int(c) for c in compressed_channels)
        super().__init__(input_shape, num_classes, recurrent_scale=recurrent_scale)
        self.shared_backbone = RNECNNGRUBackbone(input_shape).to(self._get_device())
        self.backbone_mapper = RNEMapAdapter(
            input_shape,
            out_channels=self.compressed_channels,
        ).to(self._get_device())
        self.recurrent_mapper = RNEFeatureMapper(
            input_shape,
            channels=self.compressed_channels,
        ).to(self._get_device())

    def _new_backbone(self) -> RNECompressedCNNGRUBackbone:
        return RNECompressedCNNGRUBackbone(
            self.input_shape,
            channels=self.compressed_channels[:3],
            gru_hidden=self.compressed_channels[3],
        ).to(self._get_device())

    def add_task(self, new_classes: List[int], s_max: float = 15.0):
        super().add_task(new_classes, s_max=s_max)
        if self.current_task > 0:
            for param in self.shared_backbone.parameters():
                param.requires_grad = False
            self.shared_backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.current_task > 0:
            self.shared_backbone.eval()
        return self

    def get_feature_sequence(
        self, x: torch.Tensor, s: Optional[float] = None
    ) -> List[torch.Tensor]:
        _, backbone_maps = self.shared_backbone.forward_maps(x)
        features: List[torch.Tensor] = []
        prev_maps = None
        for task_idx, extractor in enumerate(self.extractors):
            if task_idx == 0:
                recurrent_maps = [
                    self.recurrent_scale * mapped
                    for mapped in self.backbone_mapper(backbone_maps)
                ]
            else:
                recurrent_maps = [
                    self.recurrent_scale * mapped
                    for mapped in self.recurrent_mapper(prev_maps)
                ]

            if task_idx < self.current_task:
                feature, maps = extractor.forward_maps(
                    x,
                    recurrent_maps=recurrent_maps,
                )
            else:
                feature, maps = extractor.forward_maps(
                    x,
                    recurrent_maps=recurrent_maps,
                )
            features.append(feature)
            prev_maps = maps
        return features

    def get_trainable_params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        if self.current_task == 0:
            params.extend(
                p for p in self.shared_backbone.parameters() if p.requires_grad
            )
        if self.current_task >= 0:
            params.extend(
                p for p in self.extractors[self.current_task].parameters() if p.requires_grad
            )
        params.extend(p for p in self.backbone_mapper.parameters() if p.requires_grad)
        params.extend(p for p in self.recurrent_mapper.parameters() if p.requires_grad)
        for head in self.classifier_heads:
            params.extend(head.parameters())
        if self.aux_classifier is not None:
            params.extend(self.aux_classifier.parameters())
        return params

    def freeze_all_extractors(self):
        super().freeze_all_extractors()
        for param in self.shared_backbone.parameters():
            param.requires_grad = False
        for param in self.backbone_mapper.parameters():
            param.requires_grad = False
        self.shared_backbone.eval()
        self.backbone_mapper.eval()

    def unfreeze_current_extractor(self):
        super().unfreeze_current_extractor()
        for param in self.backbone_mapper.parameters():
            param.requires_grad = True
        self.backbone_mapper.train()
        if self.current_task == 0:
            for param in self.shared_backbone.parameters():
                param.requires_grad = True
            self.shared_backbone.train()
        else:
            self.shared_backbone.eval()

    def clone_empty_like(self) -> "RNECompressModel":
        return RNECompressModel(
            self.input_shape,
            self.num_classes,
            recurrent_scale=self.recurrent_scale,
            compressed_channels=self.compressed_channels,
        )
