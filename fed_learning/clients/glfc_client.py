"""
GLFC Client - Client for Global-Local Forgetting Compensation.

Reference:
    Dong et al., "Federated Class-Incremental Learning", CVPR 2022

Extends FederatedClient with:
1. Entropy-based signal detection for global forgetting compensation
2. Exemplar set management (herding-based selection)
3. Prototype gradient sharing for proxy server
4. Knowledge distillation with old model (local forgetting compensation)
"""

import contextlib
import copy
from collections import OrderedDict
from typing import Optional, Dict, Any, List, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    from torch.amp import autocast as torch_autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast as torch_autocast, GradScaler

from .client import FederatedClient
from ..core import BaseTrainer
from ..strategies.fed_incremental.glfc import GLFCTrainer, get_one_hot


class GLFCClient(FederatedClient):
    """
    Client for GLFC algorithm.

    Implements per-client local forgetting compensation:
    - Exemplar memory management (herding-based, from paper)
    - Entropy-based signal detection (global forgetting compensation)
    - Prototype gradient sharing to proxy server
    - Knowledge distillation with old model

    The client maintains its own exemplar set and mixes old exemplars
    with new task data when signal is detected.
    """

    def __init__(
        self,
        client_id: int,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        memory_size: int = 2000,
    ):
        super().__init__(client_id, X_train, y_train)

        # Exemplar memory: list of (X_exemplar, label) per class
        self.exemplar_set: Dict[int, List[torch.Tensor]] = {}
        self.learned_classes: List[int] = []
        self.learned_numclass: int = 0

        # Memory budget
        self.memory_size = memory_size

        # Current/last task class tracking (per author's code)
        self.current_class: Optional[List[int]] = None
        self.last_class: Optional[List[int]] = None
        self.task_id_old: int = -1

        # Entropy tracking
        self.last_entropy: float = 0.0
        self.signal: bool = False

        # Task tracking
        self.current_task: int = 0
        self.seen_classes: Set[int] = set()
        self.old_classes: List[int] = []
        self.new_classes: List[int] = []

        # Old model for distillation
        self.old_model: Optional[nn.Module] = None
        self.old_model_state: Optional[OrderedDict] = None

        # Original data storage (before mixing with exemplars)
        self._original_X: Optional[torch.Tensor] = None
        self._original_y: Optional[torch.Tensor] = None

    def set_task_data(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        task_id: int,
        task_classes: List[int],
    ):
        """
        Set data for current task.

        Implements the beforeTrain logic from GLFC source:
        - Track current/last classes
        - Store original data before mixing
        """
        # Store original data
        self._original_X = X_train.clone()
        self._original_y = y_train.clone()

        self.X_train = X_train
        self.y_train = y_train
        self.num_samples = len(y_train)

        # Update task tracking
        self.old_classes = list(self.seen_classes)
        self.new_classes = task_classes
        self.current_task = task_id
        self.seen_classes.update(task_classes)

        # Track current/last classes (following author's pattern)
        if task_id != self.task_id_old:
            self.task_id_old = task_id
            if self.current_class is not None:
                self.last_class = self.current_class
            self.current_class = list(task_classes)

    def save_model_snapshot(self, model: nn.Module):
        """Save model snapshot for distillation in next task."""
        self.old_model_state = OrderedDict(
            (k, v.cpu().clone()) for k, v in model.state_dict().items()
        )
        self.old_model = None  # Clear cache

    def _load_old_model(self, model_template: nn.Module, device: str):
        """Load old model from saved state for knowledge distillation."""
        if self.old_model_state is None:
            self.old_model = None
            return

        if self.old_model is not None:
            return

        try:
            self.old_model = copy.deepcopy(model_template)
            self.old_model.load_state_dict(
                {k: v.to(device) for k, v in self.old_model_state.items()}
            )
            self.old_model.eval()
            for param in self.old_model.parameters():
                param.requires_grad = False
        except Exception as e:
            print(f"  Client {self.client_id}: Failed to load old model: {e}")
            self.old_model = None

    def compute_entropy_signal(self, model: nn.Module, device: str) -> bool:
        """
        Entropy-based signal detection.

        Paper Section 3.3:
        Compute average entropy on current data.
        If increase > threshold (1.2), signal=True → update exemplar set.
        """
        model.eval()
        all_entropy = []

        # Process data in batches
        batch_size = 128
        with torch.no_grad():
            for i in range(0, self.num_samples, batch_size):
                X_batch = self.X_train[i : i + batch_size].to(device)
                outputs = model(X_batch)
                softmax_out = F.softmax(outputs, dim=1)
                ent = -softmax_out * torch.log(softmax_out + 1e-5)
                ent = torch.sum(ent, dim=1)
                all_entropy.append(ent.cpu())

        if not all_entropy:
            return False

        all_ent = torch.cat(all_entropy, dim=0)
        overall_avg = torch.mean(all_ent).item()

        signal = (overall_avg - self.last_entropy) > 1.2
        self.last_entropy = overall_avg

        model.train()
        return signal

    def update_exemplar_set(self, model: nn.Module, device: str):
        """
        Update exemplar set based on signal detection.

        Paper Section 3.3:
        If signal=True, update learned classes and select exemplars
        using herding-based selection.
        """
        model.eval()

        # Detect signal
        self.signal = self.compute_entropy_signal(model, device)

        if self.signal and self.last_class is not None:
            # Update learned classes
            self.learned_numclass += len(self.last_class)
            self.learned_classes.extend(self.last_class)

            # Budget per class
            m = max(1, self.memory_size // self.learned_numclass)

            # Reduce existing exemplar sets
            for cls_id in list(self.exemplar_set.keys()):
                if len(self.exemplar_set[cls_id]) > m:
                    self.exemplar_set[cls_id] = self.exemplar_set[cls_id][:m]

            # Construct new exemplar sets for last_class
            for cls_id in self.last_class:
                if self._original_X is not None:
                    mask = self._original_y == cls_id
                    if mask.sum() > 0:
                        class_data = self._original_X[mask]
                        exemplars = self._select_exemplars_herding(
                            model, class_data, m, device
                        )
                        self.exemplar_set[cls_id] = exemplars

        model.train()

        # Mix exemplars with current data
        self._mix_data_with_exemplars()

    def _select_exemplars_herding(
        self,
        model: nn.Module,
        class_data: torch.Tensor,
        m: int,
        device: str,
    ) -> List[torch.Tensor]:
        """
        Herding-based exemplar selection.

        Paper Algorithm:
        1. Compute feature mean for the class
        2. Iteratively select sample closest to running mean
        """
        if len(class_data) == 0:
            return []

        model.eval()
        # Extract features
        features = []
        batch_size = 64
        with torch.no_grad():
            for i in range(0, len(class_data), batch_size):
                X_batch = class_data[i : i + batch_size].to(device)
                # Use feature extractor (before classifier)
                feat = self._extract_features(model, X_batch)
                feat = F.normalize(feat, dim=1)
                features.append(feat.cpu())

        features = torch.cat(features, dim=0).numpy()
        class_mean = np.mean(features, axis=0)

        # Herding selection
        exemplars = []
        now_class_mean = np.zeros_like(class_mean)
        selected_indices = set()

        for i in range(min(m, len(class_data))):
            x = class_mean - (now_class_mean + features) / (i + 1)
            x = np.linalg.norm(x, axis=1)
            # Exclude already selected
            for idx in selected_indices:
                x[idx] = float("inf")
            index = np.argmin(x)
            now_class_mean += features[index]
            selected_indices.add(index)
            exemplars.append(class_data[index].cpu())

        return exemplars

    def _extract_features(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Extract features from model's penultimate layer."""
        # Try to use feature_extractor method if available
        if hasattr(model, "feature_extractor"):
            return model.feature_extractor(x)
        elif hasattr(model, "get_fused_representation"):
            return model.get_fused_representation(x)

        # Fallback: hook on penultimate linear layer
        activation = {}

        def hook_fn(module, inp, out):
            if isinstance(inp, tuple) and len(inp) > 0:
                activation["feat"] = inp[0].detach()
            else:
                activation["feat"] = inp.detach()

        # Find penultimate linear layer
        linear_layers = [
            (n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)
        ]
        if len(linear_layers) >= 2:
            handle = linear_layers[-1][1].register_forward_hook(hook_fn)
        elif len(linear_layers) == 1:
            handle = linear_layers[0][1].register_forward_hook(hook_fn)
        else:
            return x.view(x.size(0), -1)

        with torch.no_grad():
            _ = model(x)

        handle.remove()

        if "feat" in activation:
            return activation["feat"]
        return x.view(x.size(0), -1)

    def _mix_data_with_exemplars(self):
        """
        Mix current task data with exemplar data from old classes.

        Following author's code: getTrainData with mix=True.
        """
        if not self.exemplar_set or self._original_X is None:
            return

        all_X = [self._original_X]
        all_y = [self._original_y]

        for cls_id, exemplars in self.exemplar_set.items():
            if len(exemplars) > 0:
                X_exemplar = torch.stack(exemplars)
                y_exemplar = torch.full((len(exemplars),), cls_id, dtype=torch.long)
                all_X.append(X_exemplar)
                all_y.append(y_exemplar)

        self.X_train = torch.cat(all_X, dim=0)
        self.y_train = torch.cat(all_y, dim=0)
        self.num_samples = len(self.y_train)

    def compute_prototype_gradients(
        self, model: nn.Module, trainer: GLFCTrainer, device: str
    ) -> Optional[List]:
        """
        Compute prototype gradients for proxy server.

        Paper Section 3.4:
        1. Find prototype (closest to class mean) for each current class
        2. Optimize prototype via gradient on BCE loss
        3. Compute gradients of encode_model on optimized prototype
        4. Return list of gradient vectors

        Simplified version for our architecture:
        - Use class prototype features instead of raw image reconstruction
        - Share feature-space gradients instead of pixel-space gradients
        """
        if not self.signal:
            return None

        if self.current_class is None:
            return None

        was_training = model.training
        proto_grads = []

        try:
            for cls_id in self.current_class:
                # Find prototype (closest sample to class mean)
                mask = self.y_train == cls_id
                if mask.sum() == 0:
                    continue

                class_data = self.X_train[mask]

                # Use eval() for stable prototype selection without dropout noise.
                model.eval()
                features = []
                batch_size = 64
                with torch.no_grad():
                    for i in range(0, len(class_data), batch_size):
                        X_batch = class_data[i : i + batch_size].to(device)
                        feat = self._extract_features(model, X_batch)
                        feat = F.normalize(feat, dim=1)
                        features.append(feat.cpu())

                if not features:
                    continue

                features = torch.cat(features, dim=0)
                class_mean = features.mean(dim=0, keepdim=True)

                # Find closest sample to mean
                distances = torch.norm(features - class_mean, dim=1)
                proto_idx = torch.argmin(distances).item()
                proto_data = class_data[proto_idx : proto_idx + 1].to(device)
                proto_label = torch.tensor([cls_id], dtype=torch.long, device=device)

                # cuDNN RNN backward requires training mode.
                model.train()
                model.zero_grad()
                for p in model.parameters():
                    p.requires_grad_(True)

                output = model(proto_data)
                # The classifier always emits the full global class space, so the
                # prototype BCE target must match output.size(1), not just seen classes.
                num_class = output.size(1)
                target = get_one_hot(proto_label, num_class, str(device))
                loss = F.binary_cross_entropy_with_logits(output, target)
                grads = torch.autograd.grad(loss, model.parameters(), allow_unused=True)
                grad_list = [
                    g.detach().cpu().clone() if g is not None else None for g in grads
                ]
                proto_grads.append(grad_list)
        finally:
            if was_training:
                model.train()
            else:
                model.eval()

        return proto_grads if proto_grads else None

    def train(
        self,
        trainer: BaseTrainer,
        epochs: int,
        batch_size: int,
        lr: float,
        global_params: Optional[OrderedDict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train with GLFC local forgetting compensation.

        Sequence (following author's code):
        1. Detect entropy signal
        2. Update exemplar set if signal detected
        3. Mix data with exemplars
        4. Train with combined loss (CE + distillation)
        5. Compute prototype gradients for proxy server
        """
        self.model.train()

        # Get optimizer
        optimizer_cls = trainer.get_optimizer_class()
        optimizer = optimizer_cls(self.model.parameters(), lr=lr)
        scaler = GradScaler(enabled=self.use_amp)

        # Load old model for distillation
        if self.current_task > 0:
            # Try trainer's old model first (from proxy server)
            if isinstance(trainer, GLFCTrainer):
                self.old_model = trainer.load_old_model(
                    self.model, self.device, signal=self.signal
                )
            # Fallback to client's own snapshot
            if self.old_model is None and self.old_model_state is not None:
                self._load_old_model(self.model, self.device)

        # Pre-train hook
        trainer.pre_train(self.model, global_params, lr=lr, **kwargs)

        total_loss = 0.0
        total_samples = 0

        for ep in range(epochs):
            for X_batch, y_batch in self._create_batches(batch_size):
                optimizer.zero_grad()

                with self._amp_ctx():
                    out = self.model(X_batch)

                    # Compute GLFC loss
                    if isinstance(trainer, GLFCTrainer):
                        loss = trainer.compute_loss(
                            self.model,
                            out,
                            y_batch,
                            global_params=global_params,
                            inputs=X_batch,
                            old_model=self.old_model,
                            signal=self.signal,
                            **kwargs,
                        )
                    else:
                        loss = trainer.compute_loss(
                            self.model, out, y_batch, global_params, **kwargs
                        )

                if self.use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    trainer.pre_step(self.model, global_params, **kwargs)
                    scaler.step(optimizer)
                    scaler.update()
                    trainer.post_step(self.model, global_params, **kwargs)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    trainer.pre_step(self.model, global_params, **kwargs)
                    optimizer.step()
                    trainer.post_step(self.model, global_params, **kwargs)

                bs = len(y_batch)
                total_loss += loss.item() * bs
                total_samples += bs

        # Post-train hook
        trainer.post_train(self.model, global_params, **kwargs)

        # Compute prototype gradients for proxy server
        proto_grad = None
        if isinstance(trainer, GLFCTrainer) and self.signal:
            proto_grad = self.compute_prototype_gradients(
                self.model, trainer, self.device
            )

        result = {
            "client_id": self.client_id,
            "num_samples": self.num_samples,
            "loss": total_loss / max(1, total_samples),
            "params": OrderedDict(
                (k, v.cpu().clone()) for k, v in self.model.state_dict().items()
            ),
            "signal": self.signal,
            "proto_grad": proto_grad,
        }

        return result
