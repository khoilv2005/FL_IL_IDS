"""
RNE Model - Recurrent Network Expansion for class-incremental learning.

This implementation adapts the original RNE idea to the repo's CNN-GRU IDS
backbone:
- one frozen expert per completed task,
- a shared recurrent adapter passes previous expert features to the next expert,
- task-level classifier heads consume causal feature prefixes.
"""

import copy
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .der_model import CNNGRUBackbone, DERModel


class RNEModel(DERModel):
    """
    RNE-style expandable model for IDS features.

    Compared with DERModel, the classifier is decoupled into one head per task.
    Head i only sees features from experts 0..i, so old task classifiers are not
    affected by future experts. A shared feature adapter gives recurrent
    cross-task connection between adjacent experts.
    """

    def __init__(self, input_shape, num_classes: int, recurrent_scale: float = 1.0):
        super().__init__(input_shape, num_classes)
        self.classifier = None
        self.classifier_heads = nn.ModuleList()
        self.task_classes_history: List[List[int]] = []
        self.recurrent_adapter: Optional[nn.Module] = None
        self.recurrent_scale = float(recurrent_scale)

    def _ensure_recurrent_adapter(self, feat_dim: int) -> None:
        if self.recurrent_adapter is not None:
            return
        device = self._get_device()
        self.recurrent_adapter = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        ).to(device)

    def add_task(self, new_classes: List[int], s_max: float = 15.0):
        self.current_task += 1
        self._s_max = s_max
        new_classes = [int(c) for c in new_classes]

        if self.current_task > 0:
            prev_ext = self.extractors[self.current_task - 1]
            with torch.no_grad():
                binarized = [torch.sigmoid(s_max * e.data) for e in prev_ext.mask_embeds]
                self._set_frozen_masks(self.current_task - 1, binarized)
            for param in prev_ext.parameters():
                param.requires_grad = False
            prev_ext.eval()

        device = self._get_device()
        new_backbone = CNNGRUBackbone(self.input_shape).to(device)
        if self.current_task > 0:
            prev_ext = self.extractors[self.current_task - 1]
            src_state = prev_ext.state_dict()
            tgt_state = new_backbone.state_dict()
            for key in tgt_state:
                if "mask_embeds" not in key and key in src_state:
                    tgt_state[key] = src_state[key].clone()
            new_backbone.load_state_dict(tgt_state)
            for embed in new_backbone.mask_embeds:
                nn.init.zeros_(embed.data)

        self.extractors.append(new_backbone)
        if self.feat_dim is None:
            self.feat_dim = new_backbone.output_dim
        self._ensure_recurrent_adapter(self.feat_dim)

        self.task_classes_history.append(new_classes)
        head_in_dim = self.num_extractors * self.feat_dim
        head = nn.Linear(head_in_dim, len(new_classes)).to(device)
        self.classifier_heads.append(head)

        n_aux_classes = len(new_classes) + (1 if self.current_task > 0 else 0)
        self.aux_classifier = nn.Linear(self.feat_dim, n_aux_classes).to(device)

        print(
            f"  RNEModel: Task {self.current_task} | "
            f"experts={self.num_extractors} | "
            f"head_in={head_in_dim} | head_out={len(new_classes)} | "
            f"aux_classes={n_aux_classes}"
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
        for task_idx, extractor in enumerate(self.extractors):
            if task_idx < self.current_task:
                raw = extractor.forward_with_masks(x, self._get_frozen_masks(task_idx))
            else:
                inference_s = s if s is not None else self._s_max
                raw = extractor(x, s=inference_s)

            if task_idx > 0 and self.recurrent_adapter is not None:
                recurrent = self.recurrent_adapter(features[-1].detach())
                raw = raw + self.recurrent_scale * recurrent
            features.append(raw)
        return features

    def get_super_feature(self, x: torch.Tensor, s: Optional[float] = None) -> torch.Tensor:
        return torch.cat(self.get_feature_sequence(x, s=s), dim=1)

    def extract_vector(self, x: torch.Tensor, s: Optional[float] = None) -> torch.Tensor:
        return self.get_super_feature(x, s=s)

    def _classify_feature_sequence(self, features: List[torch.Tensor]) -> torch.Tensor:
        if not self.classifier_heads:
            raise RuntimeError("RNEModel has no classifier heads. Call add_task() first.")

        batch_size = features[0].shape[0]
        logits = features[0].new_full((batch_size, self.num_classes), -1e9)
        for task_idx, head in enumerate(self.classifier_heads):
            head_input = torch.cat(features[: task_idx + 1], dim=1)
            head_input = self.dropout(self.relu(head_input))
            head_logits = head(head_input)
            for col_idx, class_id in enumerate(self.task_classes_history[task_idx]):
                if 0 <= class_id < self.num_classes:
                    logits[:, class_id] = head_logits[:, col_idx]
        return logits

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
            params.extend(p for p in self.extractors[self.current_task].parameters() if p.requires_grad)
        if self.recurrent_adapter is not None:
            params.extend(p for p in self.recurrent_adapter.parameters() if p.requires_grad)
        for head in self.classifier_heads:
            params.extend(head.parameters())
        if self.aux_classifier is not None:
            params.extend(self.aux_classifier.parameters())
        return params

    def get_classifier_params(self) -> List[nn.Parameter]:
        return [p for head in self.classifier_heads for p in head.parameters()]

    def reset_classifier(self):
        for head in self.classifier_heads:
            nn.init.kaiming_uniform_(head.weight, a=0, nonlinearity="relu")
            nn.init.zeros_(head.bias)

    def freeze_all_extractors(self):
        super().freeze_all_extractors()
        if self.recurrent_adapter is not None:
            for param in self.recurrent_adapter.parameters():
                param.requires_grad = False
            self.recurrent_adapter.eval()

    def unfreeze_current_extractor(self):
        super().unfreeze_current_extractor()
        if self.recurrent_adapter is not None:
            for param in self.recurrent_adapter.parameters():
                param.requires_grad = True
            self.recurrent_adapter.train()

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
