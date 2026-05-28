"""RNE local strategy."""

from collections import OrderedDict
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .der import DERTrainer


class RNETrainer(DERTrainer):
    """
    Trainer for Recurrent Network Expansion.

    Stage 1 follows the RNE paper/source weighting:
        Lce = alpha * Lce_new + Lce_old
    where alpha grows with training progress. Stage 2 trains classifier heads only.
    """

    def __init__(
        self,
        lambda_aux: float = 1.0,
        lambda_sparsity: float = 0.0,
        s_max: float = 15.0,
        temperature: float = 2.0,
        buffer_size: int = 500,
        old_head_lr_scale: float = 1.0,
        kd_weight: float = 1.0,
    ):
        super().__init__(
            lambda_aux=lambda_aux,
            lambda_sparsity=lambda_sparsity,
            s_max=s_max,
            temperature=temperature,
            buffer_size=buffer_size,
        )
        self.old_head_lr_scale = old_head_lr_scale
        self.kd_weight = kd_weight
        self.current_epoch = 0
        self.total_epochs = 1

    def set_task(self, task_id: int, new_classes: List[int]):
        self.old_classes = list(self.seen_classes)
        self.new_classes = list(new_classes)
        self.current_task = task_id
        self.seen_classes.update(new_classes)
        self.current_batch = 0
        print(
            f"  RNETrainer Task {task_id}: "
            f"old={len(self.old_classes)}, new={len(new_classes)}"
        )

    def _stage1_loss(
        self,
        model: nn.Module,
        output: torch.Tensor,
        target: torch.Tensor,
        inputs: Optional[torch.Tensor],
        s: Optional[float],
        old_model: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        device = output.device
        old_set = set(int(c) for c in self.old_classes)
        new_set = set(int(c) for c in self.new_classes)

        if self.current_task == 0 or not old_set:
            ce_loss = F.cross_entropy(output, target)
        else:
            old_mask = torch.zeros_like(target, dtype=torch.bool)
            new_mask = torch.zeros_like(target, dtype=torch.bool)
            for cls_id in old_set:
                old_mask |= target == int(cls_id)
            for cls_id in new_set:
                new_mask |= target == int(cls_id)

            old_count = int(old_mask.sum().item())
            new_count = int(new_mask.sum().item())
            progress = min(1.0, self.current_epoch / max(1, self.total_epochs))
            beta = 0.1 + 0.9 * progress

            ce_loss = torch.tensor(0.0, device=device)
            if new_count > 0:
                alpha = old_count / max(1, new_count)
                alpha = alpha / max(1, len(old_set)) * max(1, len(new_set)) * beta
                ce_loss = ce_loss + alpha * F.cross_entropy(
                    output[new_mask], target[new_mask]
                )
            if old_count > 0:
                ce_loss = ce_loss + F.cross_entropy(output[old_mask], target[old_mask])

        aux_loss = torch.tensor(0.0, device=device)
        if (
            self.current_task > 0
            and self.lambda_aux > 0
            and inputs is not None
            and hasattr(model, "forward_aux")
        ):
            aux_output = model.forward_aux(inputs, s=s)
            aux_target = self._remap_aux_targets(target, device)
            aux_loss = F.cross_entropy(aux_output, aux_target)

        self.current_batch += 1
        kd_loss = torch.tensor(0.0, device=device)
        if old_model is not None and inputs is not None and self.current_task > 0:
            old_mask = torch.zeros_like(target, dtype=torch.bool)
            for cls_id in old_set:
                old_mask |= target == int(cls_id)
            if old_mask.any():
                kd_classes = sorted(old_set) + sorted(new_set)
                old_class_count = len(old_set)
                with torch.no_grad():
                    old_logits_full = old_model(inputs[old_mask])
                current_logits = output[old_mask][:, kd_classes]
                old_logits = old_logits_full[:, sorted(old_set)]
                pad = current_logits.new_zeros(
                    old_logits.shape[0],
                    len(kd_classes) - old_class_count,
                )
                soft_logits = torch.cat([old_logits, pad], dim=1)
                kd_loss = self.kd_weight * _kd_loss(
                    current_logits,
                    soft_logits,
                    temperature=2.0,
                )

        return (
            ce_loss
            + self.lambda_aux * aux_loss
            + kd_loss
        )

    def compute_loss(
        self,
        model: nn.Module,
        output: torch.Tensor,
        target: torch.Tensor,
        global_params: Optional[OrderedDict] = None,
        inputs: Optional[torch.Tensor] = None,
        s: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.training_stage == 1:
            return self._stage1_loss(
                model,
                output,
                target,
                inputs,
                s,
                old_model=kwargs.get("old_model"),
            )
        return self._stage2_loss(output, target)

    def pre_step(
        self,
        model: nn.Module,
        global_params: Optional[OrderedDict] = None,
        **kwargs,
    ):
        if (
            self.training_stage == 1
            and self.current_task > 0
            and hasattr(model, "scale_old_head_gradients")
        ):
            model.scale_old_head_gradients(self.old_head_lr_scale)


def _kd_loss(pred: torch.Tensor, soft: torch.Tensor, temperature: float) -> torch.Tensor:
    pred_log = torch.log_softmax(pred / temperature, dim=1)
    soft_prob = torch.softmax(soft / temperature, dim=1)
    return -torch.mul(soft_prob, pred_log).sum() / max(1, pred.shape[0])
