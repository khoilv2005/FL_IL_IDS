"""
LwFLocalClient - Dedicated client for LwF local training.

Extends FederatedClient to always pass `inputs=X_batch` to trainer.compute_loss(),
ensuring Knowledge Distillation loss is computed correctly.

Unlike FedLwFClient which handles task data across multiple tasks,
this client focuses on the local training loop with proper distillation support.
"""

import contextlib
from collections import OrderedDict
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

try:
    from torch.amp import autocast as torch_autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast as torch_autocast, GradScaler

from .client import FederatedClient
from ..core import BaseTrainer


class LwFLocalClient(FederatedClient):
    """
    Client that always passes inputs to compute_loss for proper Knowledge Distillation.

    This client extends FederatedClient to fix the LwF bug where compute_loss()
    was called without `inputs=X_batch`, resulting in only CrossEntropy loss
    instead of CE + KD loss.

    The key difference from FederatedClient.train():
    - Always passes `inputs=X_batch` to trainer.compute_loss()
    """

    def train(
        self,
        trainer: BaseTrainer,
        epochs: int,
        batch_size: int,
        lr: float,
        global_params: Optional[OrderedDict] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train by passing inputs to compute_loss for Knowledge Distillation.

        Args:
            trainer: Training strategy (expects FedLwFTrainer for KD)
            epochs: Number of local epochs
            batch_size: Batch size
            lr: Learning rate
            global_params: Global model parameters (for regularization)
            **kwargs: Additional trainer-specific parameters

        Returns:
            Dict with client_id, num_samples, loss, and params
        """
        self.model.train()

        optimizer_cls = trainer.get_optimizer_class()
        optimizer = optimizer_cls(self.model.parameters(), lr=lr)
        scaler = GradScaler(enabled=self.use_amp)

        trainer.pre_train(self.model, global_params, lr=lr, **kwargs)

        total_loss = 0.0
        total_samples = 0

        for ep in range(epochs):
            for X_batch, y_batch in self._create_batches(batch_size):
                optimizer.zero_grad()

                with self._amp_ctx():
                    out = self.model(X_batch)
                    # Always pass inputs=X_batch so trainer can compute KD loss
                    loss = trainer.compute_loss(
                        self.model, out, y_batch, global_params,
                        inputs=X_batch, **kwargs
                    )

                if self.use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    trainer.pre_step(self.model, global_params, **kwargs)
                    scaler.step(optimizer)
                    scaler.update()
                    trainer.post_step(self.model, global_params, **kwargs)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    trainer.pre_step(self.model, global_params, **kwargs)
                    optimizer.step()
                    trainer.post_step(self.model, global_params, **kwargs)

                bs = len(y_batch)
                total_loss += loss.item() * bs
                total_samples += bs

        trainer.post_train(self.model, global_params, **kwargs)

        return {
            "client_id": self.client_id,
            "num_samples": self.num_samples,
            "loss": total_loss / max(1, total_samples),
            "params": OrderedDict(
                (k, v.cpu().clone()) for k, v in self.model.state_dict().items()
            )
        }
