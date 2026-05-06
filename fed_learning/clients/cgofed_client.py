"""
CGoFed Client - Specialized client for Class Incremental Learning.

Reference:
    "CGoFed: Constrained Gradient Optimization Strategy for Federated Class
    Incremental Learning", IEEE TKDE, 2025

Extends FederatedClient with activation-based representation computation
for cross-task similarity (paper Section 5.2, Eq. 2, 10).
"""

import os
import shutil
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.utils.data import DataLoader, TensorDataset

try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler

from .client import FederatedClient
from ..core import BaseTrainer
from ..strategies.fed_incremental.cgofed import build_representation_artifact


class CGoFedClient(FederatedClient):
    """
    Client for CGoFed algorithm with activation representation computation.

    Inherits all standard FL functionality from FederatedClient,
    adds compute_activation_representation() for cross-task similarity.

    Paper Reference:
    - Representation R is computed as activation vector from last hidden layer (Paper Eq. 2)
    - Used by server to compute similarity between tasks (Paper Eq. 10)
    - Enables personalized aggregation with historical models (Paper Eq. 12)
    """

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
        Train and compute activation representation for CGoFed.

        Paper-faithful split:
        - Eq.6/Eq.8/Eq.9 gradient projection is executed by the client during
          local optimization.
        - Trainer still supplies optimizer/loss hyperparameters and Eq.14 loss.
        - Eq.2 task representation is returned to the server for Eq.10.
        """
        if self.model is None:
            raise RuntimeError("CGoFedClient.train() called before setup_for_gpu().")

        try:
            self._ensure_projection_state()
            self.model.train()

            optimizer_cls = trainer.get_optimizer_class()
            optimizer = optimizer_cls(self.model.parameters(), lr=lr)
            scaler = GradScaler(enabled=self.use_amp)

            trainer.pre_train(self.model, global_params, lr=lr, **kwargs)

            total_loss = 0.0
            total_samples = 0

            for _ep in range(epochs):
                for X_batch, y_batch in self._create_batches(batch_size):
                    optimizer.zero_grad()

                    with self._amp_ctx():
                        output = self.model(X_batch)
                        loss = trainer.compute_loss(
                            self.model, output, y_batch, global_params, **kwargs
                        )

                    if self.use_amp:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=1.0
                        )
                        self._apply_relax_constrained_gradient_update(
                            self.model, trainer
                        )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=1.0
                        )
                        self._apply_relax_constrained_gradient_update(
                            self.model, trainer
                        )
                        optimizer.step()

                    trainer.post_step(self.model, global_params, **kwargs)

                    batch_size_actual = len(y_batch)
                    total_loss += loss.item() * batch_size_actual
                    total_samples += batch_size_actual

            trainer.post_train(self.model, global_params, **kwargs)

            result = {
                "client_id": self.client_id,
                "num_samples": self.num_samples,
                "loss": total_loss / max(1, total_samples),
                "params": OrderedDict(
                    (key, value.cpu().clone())
                    for key, value in self.model.state_dict().items()
                ),
            }
        finally:
            self._clear_projection_cache()

        # Compute activation representation for cross-task similarity (Paper Eq. 2, 10)
        if self.model is not None:
            requested_samples = getattr(trainer, "num_samples_rep", self.num_samples)
            try:
                requested_samples = int(requested_samples)
            except (TypeError, ValueError):
                requested_samples = self.num_samples
            if requested_samples <= 0:
                requested_samples = self.num_samples
            rep = self.compute_activation_representation(
                model=self.model,
                num_samples=requested_samples,
            )
            result["representation"] = build_representation_artifact(rep)

            if kwargs.get("build_projection_space", False):
                self.build_projection_space(
                    model=self.model,
                    trainer=trainer,
                    num_samples=requested_samples,
                )

        return result

    def _ensure_projection_state(self) -> None:
        if not hasattr(self, "projection_layer_bases"):
            self.projection_layer_bases: Dict[str, Dict[str, Dict[str, Any]]] = {}
        if not hasattr(self, "_projection_space_version"):
            self._projection_space_version = 0
        if not hasattr(self, "_projection_cache"):
            self._projection_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        if not hasattr(self, "_projection_logged_tasks"):
            self._projection_logged_tasks = set()

    def _clear_projection_cache(self) -> None:
        self._ensure_projection_state()
        self._projection_cache = {}

    def _cache_projection_matrices(
        self,
        device: str,
        current_task: int,
        max_old_tasks: Optional[int],
    ) -> Dict[str, torch.Tensor]:
        """
        Client-side Eq.5/Eq.9 projector construction from local historical bases.
        """
        self._ensure_projection_state()
        cache_key = f"{device}|task_{current_task}|v{self._projection_space_version}"
        if cache_key in self._projection_cache:
            return self._projection_cache[cache_key]

        task_keys = sorted(
            [k for k in self.projection_layer_bases.keys() if k.startswith("task_")],
            key=lambda item: int(item.split("_")[1]),
        )
        old_task_keys = [
            key for key in task_keys if int(key.split("_")[1]) < current_task
        ]
        if max_old_tasks is not None:
            old_task_keys = old_task_keys[-max_old_tasks:]

        layer_weighted_bases: Dict[str, List[torch.Tensor]] = {}
        for task_key in old_task_keys:
            for layer_name, info in self.projection_layer_bases.get(task_key, {}).items():
                try:
                    basis = torch.load(info["basis"], map_location=device)
                    importance = torch.load(info["importance"], map_location=device)
                    if torch.isnan(basis).any() or torch.isinf(basis).any():
                        continue
                    if torch.isnan(importance).any() or torch.isinf(importance).any():
                        continue
                    weighted_basis = basis * importance
                    layer_weighted_bases.setdefault(layer_name, []).append(weighted_basis)
                except Exception as exc:
                    print(
                        f"    ⚠️ Client {self.client_id}: failed to load "
                        f"projection basis for {layer_name}: {exc}"
                    )

        cached: Dict[str, torch.Tensor] = {}
        for layer_name, bases_list in layer_weighted_bases.items():
            try:
                all_weighted = torch.cat(bases_list, dim=1)
                U, S, _Vh = torch.linalg.svd(all_weighted, full_matrices=False)
                significant = S > 1e-6
                if not torch.any(significant):
                    continue
                U_orth = U[:, significant]
                S_sig = S[significant]
                S_normalized = S_sig / (S_sig.max() + 1e-10)
                cached[layer_name] = torch.mm(U_orth * (S_normalized**2), U_orth.T)
            except Exception as exc:
                print(
                    f"    ⚠️ Client {self.client_id}: failed to build "
                    f"projector for {layer_name}: {exc}"
                )

        self._projection_cache[cache_key] = cached
        return cached

    def _apply_relax_constrained_gradient_update(
        self, model: nn.Module, trainer: BaseTrainer
    ) -> None:
        """
        Client-side Eq.6/Eq.8/Eq.9.

        Runs after backward() and before optimizer.step(), so the modified
        gradients are applied only to this client's local model.
        """
        self._ensure_projection_state()
        current_task = int(getattr(trainer, "current_task", 0))
        if current_task == 0 or not self.projection_layer_bases:
            return

        with torch.no_grad():
            device = next(model.parameters()).device
            device_key = str(device)
            cached = self._cache_projection_matrices(
                device=device_key,
                current_task=current_task,
                max_old_tasks=getattr(trainer, "max_old_tasks", None),
            )
            if not cached:
                return

            module_params = {}
            for layer_name, module in self._get_projection_target_modules(model):
                if isinstance(module, nn.GRU):
                    if hasattr(module, "weight_ih_l0"):
                        module_params[layer_name] = module.weight_ih_l0
                elif hasattr(module, "weight") and module.weight is not None:
                    module_params[layer_name] = module.weight

            log_key = (self.client_id, current_task)
            should_log = log_key not in self._projection_logged_tasks
            per_layer_reduction = {}

            for layer_name, projector in cached.items():
                param = module_params.get(layer_name)
                if param is None or param.grad is None:
                    continue

                grad_shape = param.grad.shape
                if len(grad_shape) == 1:
                    continue

                out_dim = grad_shape[0]
                in_dim = param.grad.numel() // out_dim
                if in_dim != projector.shape[0]:
                    stats = getattr(trainer, "_projection_stats", None)
                    if isinstance(stats, dict):
                        stats["skipped"] = stats.get("skipped", 0) + 1
                    continue

                grad_2d = param.grad.view(out_dim, in_dim)
                try:
                    projected = torch.mm(grad_2d, projector)
                    if torch.isnan(projected).any() or torch.isinf(projected).any():
                        stats = getattr(trainer, "_projection_stats", None)
                        if isinstance(stats, dict):
                            stats["skipped"] = stats.get("skipped", 0) + 1
                        continue

                    mu_t = float(getattr(trainer, "mu_projection", 0.0)) * float(
                        getattr(trainer, "mu_coefficient", 1.0)
                    )
                    grad_new = grad_2d - mu_t * projected

                    orig_norm = grad_2d.norm().item()
                    new_norm = grad_new.norm().item()
                    if orig_norm > 1e-8:
                        reduction = (orig_norm - new_norm) / orig_norm
                        stats = getattr(trainer, "_projection_stats", None)
                        if isinstance(stats, dict):
                            stats["total_reduction"] = stats.get("total_reduction", 0.0) + reduction
                            stats["projected"] = stats.get("projected", 0) + 1
                        if should_log:
                            per_layer_reduction[layer_name] = {
                                "reduction_pct": reduction * 100,
                                "orig_norm": orig_norm,
                                "new_norm": new_norm,
                                "proj_dim": projector.shape[0],
                            }

                    param.grad.copy_(grad_new.view_as(param.grad))
                except RuntimeError as exc:
                    stats = getattr(trainer, "_projection_stats", None)
                    if isinstance(stats, dict):
                        stats["skipped"] = stats.get("skipped", 0) + 1
                    print(
                        f"    ⚠️ Client {self.client_id}: projection error "
                        f"for {layer_name}: {exc}"
                    )

            if should_log:
                print(
                    f"  DEBUG[ClientProjection]: client={self.client_id}, "
                    f"task={current_task}, layers={sorted(cached.keys())}"
                )
                for layer_name, info in per_layer_reduction.items():
                    print(
                        f"    {layer_name}: reduction={info['reduction_pct']:.1f}% | "
                        f"grad_norm {info['orig_norm']:.4f} -> {info['new_norm']:.4f} | "
                        f"projector_dim={info['proj_dim']}"
                    )
                self._projection_logged_tasks.add(log_key)

    def _get_projection_target_modules(
        self, model: nn.Module
    ) -> List[Tuple[str, nn.Module]]:
        modules = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.GRU)):
                if "bn" in name.lower() or "batch" in name.lower():
                    continue
                modules.append((name, module))
        return modules

    @staticmethod
    def _activation_to_features(
        layer_name: str, module: nn.Module, activation: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if isinstance(module, nn.Linear):
            return activation.detach().view(activation.size(0), -1)

        if isinstance(module, nn.GRU):
            # Protect GRU input-hidden weights for layer 0. Recurrent hidden
            # weights require per-timestep hidden-state bases and are not
            # derived here without adding a non-paper hook.
            act = activation.detach()
            if act.dim() == 3:
                return act.contiguous().view(-1, act.shape[-1])
            return act.view(act.size(0), -1)

        if isinstance(module, nn.Conv2d):
            try:
                unfolded = F.unfold(
                    activation.detach(),
                    kernel_size=module.kernel_size,
                    dilation=module.dilation,
                    padding=module.padding,
                    stride=module.stride,
                )
                return unfolded.transpose(1, 2).contiguous().view(-1, unfolded.size(1))
            except Exception as exc:
                print(f"    ⚠️ CGoFed client {layer_name}: Conv2d unfold failed: {exc}")
                return None

        if isinstance(module, nn.Conv1d):
            try:
                activation_4d = activation.detach().unsqueeze(2)
                k = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
                s = module.stride[0] if isinstance(module.stride, tuple) else module.stride
                p = module.padding[0] if isinstance(module.padding, tuple) else module.padding
                d = module.dilation[0] if isinstance(module.dilation, tuple) else module.dilation
                unfolded = F.unfold(
                    activation_4d,
                    kernel_size=(1, k),
                    dilation=(1, d),
                    padding=(0, p),
                    stride=(1, s),
                )
                return unfolded.transpose(1, 2).contiguous().view(-1, unfolded.size(1))
            except Exception as exc:
                print(f"    ⚠️ CGoFed client {layer_name}: Conv1d unfold failed: {exc}")
                return None

        return activation.detach().view(activation.size(0), -1)

    def build_projection_space(
        self,
        model: Optional[nn.Module],
        trainer: BaseTrainer,
        num_samples: Optional[int] = None,
    ) -> None:
        """
        Build client-local Eq. 3-5 projection basis from this client's private data.

        The server/trainer no longer computes SVD over client samples. This method
        keeps only compact basis/importance tensors on disk for this client.
        """
        self._ensure_projection_state()
        if model is None or self.num_samples <= 0:
            return

        current_task = int(getattr(trainer, "current_task", 0))
        task_key = f"task_{current_task}"
        n_available = self.num_samples if not num_samples or num_samples <= 0 else min(num_samples, self.num_samples)
        if n_available <= 0:
            return

        target_modules = self._get_projection_target_modules(model)
        if not target_modules:
            return

        device = next(model.parameters()).device
        was_training = model.training
        model.eval()

        temp_dir = getattr(trainer, "temp_dir", "./temp_svd_storage")
        client_dir = os.path.join(temp_dir, f"client_{self.client_id}")
        task_dir = os.path.join(client_dir, task_key)
        if os.path.isdir(task_dir):
            shutil.rmtree(task_dir, ignore_errors=True)
        os.makedirs(task_dir, exist_ok=True)

        indices = torch.randperm(self.num_samples)[:n_available]
        batch_size = 32
        captured: Dict[str, torch.Tensor] = {}
        grams: Dict[str, torch.Tensor] = {}
        dims: Dict[str, int] = {}
        handles = []

        def make_hook(layer_name: str, module: nn.Module):
            def hook_fn(_module, inp, _out):
                activation = inp[0] if isinstance(inp, tuple) and len(inp) > 0 else inp
                features = self._activation_to_features(layer_name, module, activation)
                if features is not None:
                    captured[layer_name] = features

            return hook_fn

        for layer_name, module in target_modules:
            handles.append(module.register_forward_hook(make_hook(layer_name, module)))

        try:
            with torch.no_grad():
                for i in range(0, len(indices), batch_size):
                    batch_idx = indices[i : i + batch_size]
                    X_batch = self.X_train[batch_idx].to(device)
                    _ = model(X_batch)

                    for layer_name, features in captured.items():
                        chunk = features.detach().cpu().float()
                        if chunk.dim() != 2 or chunk.numel() == 0:
                            continue
                        d = chunk.shape[1]
                        if layer_name not in grams:
                            grams[layer_name] = torch.zeros((d, d), dtype=torch.float64)
                            dims[layer_name] = d
                        if dims[layer_name] != d:
                            raise ValueError(
                                f"Inconsistent activation dimension for {layer_name}: "
                                f"expected {dims[layer_name]}, got {d}"
                            )
                        grams[layer_name] += chunk.T.double().mm(chunk.double())

                    captured.clear()
                    del X_batch
        finally:
            for handle in handles:
                handle.remove()
            model.train(was_training)

        energy_threshold = float(getattr(trainer, "energy_threshold", 0.95))
        beta = float(getattr(trainer, "beta", 1.0))
        task_bases: Dict[str, Dict[str, Any]] = {}

        for layer_name, gram in grams.items():
            try:
                eigvals, eigvecs = torch.linalg.eigh(gram)
                order = torch.argsort(eigvals, descending=True)
                eigvals = torch.clamp(eigvals[order], min=0.0)
                eigvecs = eigvecs[:, order]
                significant = eigvals > 1e-12
                if not torch.any(significant):
                    continue

                eigvals = eigvals[significant]
                eigvecs = eigvecs[:, significant]
                singular_values = torch.sqrt(eigvals.float())

                cumulative = torch.cumsum(eigvals.float(), dim=0)
                ratio = cumulative / (cumulative[-1] + 1e-10)
                rank = int((ratio < energy_threshold).sum().item()) + 1
                rank = min(rank, len(singular_values))

                basis = eigvecs[:, :rank].float()
                importance = torch.sigmoid(beta * singular_values[:rank])

                safe_layer_name = layer_name.replace(".", "_")
                basis_path = os.path.join(task_dir, f"{safe_layer_name}_basis.pt")
                importance_path = os.path.join(task_dir, f"{safe_layer_name}_importance.pt")
                torch.save(basis.detach().cpu().clone(), basis_path)
                torch.save(importance.detach().cpu().clone(), importance_path)

                task_bases[layer_name] = {
                    "basis": basis_path,
                    "importance": importance_path,
                    "shape": (dims[layer_name], rank),
                }
            except Exception as exc:
                print(f"    ⚠️ CGoFed client {self.client_id}: SVD failed for {layer_name}: {exc}")

        if task_bases:
            self.projection_layer_bases[task_key] = task_bases
            self._projection_space_version += 1
            print(
                f"    ✓ Client {self.client_id}: built Eq.3-5 projection space "
                f"for {task_key} ({len(task_bases)} layers)"
            )

    def get_resume_state(self) -> Dict[str, Any]:
        """Materialize client-local CGoFed bases into continuation state."""
        self._ensure_projection_state()
        serialized_bases: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for task_key, layer_dict in self.projection_layer_bases.items():
            serialized_bases[task_key] = {}
            for layer_name, info in layer_dict.items():
                basis_path = info.get("basis")
                importance_path = info.get("importance")
                if not basis_path or not importance_path:
                    continue
                if not os.path.exists(basis_path) or not os.path.exists(importance_path):
                    continue
                serialized_bases[task_key][layer_name] = {
                    "basis_tensor": torch.load(basis_path, map_location="cpu"),
                    "importance_tensor": torch.load(importance_path, map_location="cpu"),
                    "shape": tuple(info.get("shape", ())),
                }
        return {
            "client_id": self.client_id,
            "projection_space_version": self._projection_space_version,
            "projection_layer_bases_state": serialized_bases,
        }

    def load_resume_state(self, state: Dict[str, Any]) -> None:
        """Restore client-local projection bases from continuation state."""
        self._ensure_projection_state()
        self._projection_space_version = int(
            state.get("projection_space_version", self._projection_space_version)
        )

        base_dir = os.path.join("./temp_svd_storage", f"client_{self.client_id}_resume")
        os.makedirs(base_dir, exist_ok=True)
        self.projection_layer_bases = {}
        for task_key, layer_dict in state.get("projection_layer_bases_state", {}).items():
            task_dir = os.path.join(base_dir, task_key)
            os.makedirs(task_dir, exist_ok=True)
            self.projection_layer_bases[task_key] = {}
            for layer_name, info in layer_dict.items():
                basis_tensor = info.get("basis_tensor")
                importance_tensor = info.get("importance_tensor")
                if not isinstance(basis_tensor, torch.Tensor):
                    continue
                if not isinstance(importance_tensor, torch.Tensor):
                    continue

                safe_layer_name = layer_name.replace(".", "_")
                basis_path = os.path.join(task_dir, f"{safe_layer_name}_basis.pt")
                importance_path = os.path.join(task_dir, f"{safe_layer_name}_importance.pt")
                torch.save(basis_tensor.detach().cpu().clone(), basis_path)
                torch.save(importance_tensor.detach().cpu().clone(), importance_path)
                shape = info.get("shape")
                if shape is None or len(shape) != 2:
                    shape = tuple(basis_tensor.shape)
                self.projection_layer_bases[task_key][layer_name] = {
                    "basis": basis_path,
                    "importance": importance_path,
                    "shape": tuple(shape),
                }

    def build_representation_loader(
        self, num_samples: Optional[int], batch_size: int = 32
    ) -> Optional[DataLoader]:
        """
        Build a small local loader for representation-space construction.

        This keeps post-task basis construction client-local instead of
        centralizing raw tensors into a server-side aggregate dataset.
        """
        if self.num_samples <= 0:
            return None

        if num_samples is None or num_samples <= 0:
            n_available = self.num_samples
        else:
            n_available = min(num_samples, self.num_samples)
        if n_available <= 0:
            return None

        indices = torch.randperm(self.num_samples)[:n_available]
        dataset = TensorDataset(self.X_train[indices], self.y_train[indices])
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def compute_activation_representation(
        self, model: Optional[nn.Module], num_samples: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute the task-level representation matrix R^t (Paper Eq. 2).

        Paper CGoFed Section 5.1, Eq. 2:
        R^t = F(Θ^t, X^t) = [z_1, ..., z_n]^T - representation matrix from forward propagation

        For CNN_GRU_Model, R^t is the fused CNN-GRU representation before the MLP
        classifier head, i.e. model.get_fused_representation(x) == input(fc1).
        This representation is used for cross-task similarity (Eq. 10). It is
        intentionally separate from CGoFedTrainer._collect_activations(), which
        collects per-layer inputs for gradient projection.

        Args:
            model: The trained model
            num_samples: Number of samples to use for activation computation

        Returns:
            Activation representation matrix [num_samples, hidden_dim]
        """
        n_default = num_samples if num_samples is not None and num_samples > 0 else 1
        if model is None:
            return torch.zeros(n_default, 256)  # Default hidden dimension

        device = next(model.parameters()).device  # type: ignore
        was_training = model.training
        model.eval()  # type: ignore

        # Sample indices
        if num_samples is None or num_samples <= 0:
            n_available = self.num_samples
        else:
            n_available = min(num_samples, self.num_samples)
        if n_available <= 0:
            return torch.zeros(0, 256)
        indices = torch.randperm(self.num_samples)[:n_available]

        activations = []

        if callable(getattr(model, "get_fused_representation", None)):
            try:
                with torch.no_grad():
                    batch_size = 32
                    for i in range(0, len(indices), batch_size):
                        batch_idx = indices[i : i + batch_size]
                        X_batch = self.X_train[batch_idx].to(device)

                        # R^t for CNN_GRU_Model: fused CNN-GRU representation
                        # before the MLP classifier head.
                        rep = model.get_fused_representation(X_batch)  # type: ignore[attr-defined]
                        if rep.dim() > 2:
                            rep = rep.view(rep.size(0), -1)
                        activations.append(rep.detach().cpu())
                        del X_batch, rep
            finally:
                model.train(was_training)

            if activations:
                return torch.cat(activations, dim=0)

            hidden_dim = getattr(model, "cnn_output_size", 0) + getattr(
                model, "gru_output_size", 0
            )
            hidden_dim = int(hidden_dim) if hidden_dim else 256
            return torch.zeros(n_available, hidden_dim)

        # Fallback for non-CNN_GRU models: capture INPUT to fc1/penultimate Linear.
        # For CNN_GRU_Model this equals get_fused_representation(x), but the direct
        # path above is clearer and avoids depending on layer names.
        activation = {}

        def get_activation(name):
            def hook(module, inp, output):
                if isinstance(inp, tuple) and len(inp) > 0:
                    activation[name] = inp[0].detach()
                else:
                    activation[name] = inp.detach()

            return hook

        # Register hook on fc1 to capture its INPUT, not its output.
        handle = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "fc1" in name:
                handle = module.register_forward_hook(get_activation("fc1"))
                break

        if handle is None:
            # Fallback: use penultimate layer
            linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
            if len(linear_layers) >= 2:
                handle = linear_layers[-2].register_forward_hook(
                    get_activation("hidden")
                )

        try:
            with torch.no_grad():
                # Process in small batches to avoid memory issues
                batch_size = 32
                for i in range(0, len(indices), batch_size):
                    batch_idx = indices[i : i + batch_size]
                    X_batch = self.X_train[batch_idx].to(device)

                    # Forward pass to get activations
                    _ = model(X_batch)

                    # Get activation from hook
                    if activation:
                        key = list(activation.keys())[0]
                        act = activation[key]
                        if act.dim() > 2:
                            # Flatten if needed (e.g., conv output)
                            act = act.view(act.size(0), -1)
                        activations.append(act.cpu())
        finally:
            if handle is not None:
                handle.remove()
            model.train(was_training)

        # Return representation matrix R^t (Paper Eq. 2)
        # R^t has shape [num_samples, hidden_dim] for SVD computation (Eq. 3)
        if activations:
            all_activations = torch.cat(activations, dim=0)
            return all_activations  # Return matrix [n_samples, hidden_dim], not mean vector
        else:
            # Return zero matrix if no activations collected
            # Try to infer hidden dim from model
            hidden_dim = 256  # Default
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    hidden_dim = module.out_features
            return torch.zeros(n_available, hidden_dim)
