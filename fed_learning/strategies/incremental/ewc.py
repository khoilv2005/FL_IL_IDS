"""
EWC (Elastic Weight Consolidation) Strategy for Federated Incremental Learning.

References:
    [1] Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks",
        PNAS 2017  (Standard EWC)
    [2] Huszar, "On Quadratic Penalties in Elastic Weight Consolidation", 2018
        (Corrected EWC - fixes double-counting bias)
    [3] Schwarz et al., "Progress & Compress: A scalable framework for continual
        learning", ICML 2018  (Online EWC)

Implementation:
    Corrected EWC (Huszar 2018) with Online EWC option:
    - Single penalty with accumulated Fisher (no double-counting)
    - Single anchor at latest optimal parameters
    - Online mode: additive Fisher accumulation with decay (Schwarz 2018)
    - Fisher matrices cached in memory for fast compute_loss()

Usage:
    # Combine with any Federated method using Mixin pattern:
    class FedAvgEWCTrainer(EWCMixin, FedAvgTrainer): pass
    class FedProxEWCTrainer(EWCMixin, FedProxTrainer): pass
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Set
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import BaseTrainer


class EWCMixin:
    """
    EWC Mixin - adds Elastic Weight Consolidation to any trainer.

    Corrected EWC Loss (Huszar 2018):
        L = L_base + (λ/2) * Σ_i F_acc_i * (θ_i - θ*_latest_i)²

    Where:
        - F_acc_i: Accumulated Fisher across ALL previous tasks (single penalty)
        - θ*_latest_i: Optimal parameters after the MOST RECENT task (single anchor)
        - λ: EWC regularization strength

    Fix vs original EWC (Kirkpatrick 2017):
        Original uses separate penalty per task → double-counts earlier tasks.
        Corrected uses single accumulated penalty → mathematically correct.

    Online EWC option (Schwarz 2018):
        F_acc = γ * F_acc_prev + F_new  (additive with decay, NOT weighted average)

    This mixin should be placed BEFORE the base trainer in MRO:
        class FedAvgEWCTrainer(EWCMixin, FedAvgTrainer): pass
    """

    def __init__(
        self,
        ewc_lambda: float = 1000.0,
        fisher_samples: int = 200,
        online_ewc: bool = False,
        gamma: float = 0.9,
        temp_dir: str = "./temp_ewc_storage",
    ):
        """
        Args:
            ewc_lambda: Regularization strength (λ in paper)
            fisher_samples: Number of samples for Fisher computation
            online_ewc: If True, use Online EWC with additive Fisher accumulation
            gamma: Decay factor for Online EWC (applied to accumulated Fisher)
            temp_dir: Directory to store Fisher matrices on disk (backup)
        """
        self.ewc_lambda = ewc_lambda
        self.fisher_samples = fisher_samples
        self.online_ewc = online_ewc
        self.gamma = gamma
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)

        # Disk paths for Fisher matrices and optimal params per task
        # {task_id: {"fisher": path, "params": path}}
        self.ewc_data: Dict[int, Dict[str, str]] = {}

        # In-memory cache: accumulated Fisher + latest optimal params
        # Loaded once per task, used for all compute_loss() calls
        self._cached_fisher_acc: Optional[Dict[str, torch.Tensor]] = None
        self._cached_optimal_params: Optional[Dict[str, torch.Tensor]] = None
        self._cache_device: Optional[str] = None

        # Task tracking
        self.current_task: int = 0
        self.seen_classes: Set[int] = set()

        # For AF calculation (compatibility with training script)
        self.mu_coefficient: float = 1.0
        self.best_acc_per_task: Dict[int, float] = {}
        self.current_acc_per_task: Dict[int, float] = {}
        self.last_af: float = 0.0

    def set_task(self, task_id: int, new_classes: List[int]):
        """Called at the beginning of each new task."""
        self.current_task = task_id
        self.seen_classes.update(new_classes)
        # Invalidate cache so it reloads on first compute_loss()
        self._cached_fisher_acc = None
        self._cached_optimal_params = None
        self._cache_device = None

    def compute_fisher_information(
        self, model: nn.Module, data_loader, device: str = "cuda"
    ):
        """
        Compute Fisher Information Matrix (diagonal approximation).

        F_i = E[(∂L/∂θ_i)²]

        Approximated using empirical samples with per-sample gradients.
        """
        # Train mode for RNN backward (cuDNN), but disable Dropout
        model.train()
        for m in model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.training = False

        fisher = {
            name: torch.zeros_like(param)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        sample_count = 0
        criterion = nn.CrossEntropyLoss()

        for X, y in data_loader:
            if sample_count >= self.fisher_samples:
                break

            X, y = X.to(device), y.to(device)

            # Per-sample gradient computation (important for Fisher diagonal)
            for i in range(len(X)):
                if sample_count >= self.fisher_samples:
                    break

                model.zero_grad()
                output = model(X[i : i + 1])
                loss = criterion(output, y[i : i + 1])
                loss.backward()

                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        fisher[name] += param.grad.detach() ** 2

                sample_count += 1

        # Average over samples
        if sample_count > 0:
            for name in fisher:
                fisher[name] /= sample_count

        return fisher

    def consolidate(self, model: nn.Module, data_loader, device: str = "cuda"):
        """
        Consolidate knowledge after completing a task.

        Computes Fisher and accumulates it following Huszar (2018) correction:
        - Standard mode: F_acc = F_acc_prev + F_new
        - Online mode:   F_acc = γ * F_acc_prev + F_new  (Schwarz 2018)

        Always stores the LATEST optimal params as the single anchor point.
        """
        print(
            f"  🔐 Computing Fisher Information for EWC (task {self.current_task})..."
        )

        # Compute Fisher for current task
        fisher_new = self.compute_fisher_information(model, data_loader, device)

        # Store optimal parameters (single anchor = latest task optimum)
        optimal_params = {
            name: param.detach().cpu().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        # Accumulate Fisher (Huszar correction: single accumulated penalty)
        if self.current_task > 0 and self._get_prev_fisher_acc() is not None:
            prev_fisher_acc = self._get_prev_fisher_acc()
            fisher_acc = {}
            for name in fisher_new:
                if name in prev_fisher_acc:
                    prev_f = prev_fisher_acc[name].to(device)
                    if self.online_ewc:
                        # Online EWC (Schwarz 2018): F_acc = γ * F_acc_prev + F_new
                        fisher_acc[name] = self.gamma * prev_f + fisher_new[name]
                    else:
                        # Corrected EWC (Huszar 2018): F_acc = F_acc_prev + F_new
                        fisher_acc[name] = prev_f + fisher_new[name]
                else:
                    fisher_acc[name] = fisher_new[name]
        else:
            # First task: accumulated Fisher = Fisher of task 0
            fisher_acc = fisher_new

        # Save to disk (backup)
        task_key = f"task_{self.current_task}"
        fisher_path = os.path.join(self.temp_dir, f"{task_key}_fisher_acc.pt")
        params_path = os.path.join(self.temp_dir, f"{task_key}_params.pt")

        fisher_cpu = {name: f.cpu() for name, f in fisher_acc.items()}
        torch.save(fisher_cpu, fisher_path)
        torch.save(optimal_params, params_path)

        self.ewc_data[self.current_task] = {
            "fisher": fisher_path,
            "params": params_path,
        }

        # Update in-memory cache immediately
        self._cached_fisher_acc = fisher_cpu
        self._cached_optimal_params = optimal_params
        self._cache_device = None  # Will be loaded to correct device on first use

        n_params = sum(f.numel() for f in fisher_cpu.values())
        fisher_mean = sum(f.mean().item() for f in fisher_cpu.values()) / max(len(fisher_cpu), 1)
        print(
            f"  ✓ Accumulated Fisher for task {self.current_task} "
            f"({n_params} params, mean importance: {fisher_mean:.6f})"
        )

    def _get_prev_fisher_acc(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get the accumulated Fisher from the previous task (from memory or disk)."""
        if self._cached_fisher_acc is not None:
            return self._cached_fisher_acc

        # Load from disk if available
        prev_task = self.current_task - 1
        if prev_task in self.ewc_data:
            return torch.load(
                self.ewc_data[prev_task]["fisher"], map_location="cpu"
            )
        return None

    def _ensure_cache_loaded(self, device: str):
        """Load accumulated Fisher + optimal params into memory cache on correct device."""
        if self._cached_fisher_acc is not None and self._cache_device == device:
            return  # Already cached on correct device

        if not self.ewc_data:
            return

        # Use the LATEST task's accumulated Fisher + optimal params (single penalty)
        latest_task = max(self.ewc_data.keys())
        data = self.ewc_data[latest_task]

        if self._cached_fisher_acc is None:
            self._cached_fisher_acc = torch.load(data["fisher"], map_location="cpu")
        if self._cached_optimal_params is None:
            self._cached_optimal_params = torch.load(data["params"], map_location="cpu")

        # Move to target device
        self._cached_fisher_acc = {
            name: f.to(device) for name, f in self._cached_fisher_acc.items()
        }
        self._cached_optimal_params = {
            name: p.to(device) for name, p in self._cached_optimal_params.items()
        }
        self._cache_device = device

    def compute_loss(
        self,
        model: nn.Module,
        output: torch.Tensor,
        target: torch.Tensor,
        global_params: Optional[OrderedDict] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute loss with Corrected EWC regularization (Huszar 2018).

        L = L_base + (λ/2) * Σ_i F_acc_i * (θ_i - θ*_latest_i)²

        Single penalty term with accumulated Fisher and latest anchor.
        No double-counting of earlier tasks.
        Fisher loaded from memory cache (no disk I/O per batch).
        """
        # Get base loss from parent (FedAvg, FedProx, etc.)
        base_loss = super().compute_loss(model, output, target, global_params, **kwargs)

        # No EWC penalty for first task
        if not self.ewc_data:
            return base_loss

        device = next(model.parameters()).device

        # Load cache if needed (once per task, not per batch)
        self._ensure_cache_loaded(str(device))

        if self._cached_fisher_acc is None or self._cached_optimal_params is None:
            return base_loss

        # Single accumulated penalty (Huszar correction)
        ewc_penalty = torch.tensor(0.0, device=device)

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self._cached_fisher_acc:
                continue
            if name not in self._cached_optimal_params:
                continue

            fisher_val = self._cached_fisher_acc[name]
            optimal_val = self._cached_optimal_params[name]
            diff = param - optimal_val
            ewc_penalty += (fisher_val * diff ** 2).sum()

        return base_loss + (self.ewc_lambda / 2.0) * ewc_penalty

    def update_forgetting(self, task_accuracies: Dict[int, float]):
        """Update Average Forgetting metrics (for compatibility)."""
        self.current_acc_per_task = task_accuracies.copy()

        for tid, acc in task_accuracies.items():
            if tid not in self.best_acc_per_task:
                self.best_acc_per_task[tid] = acc
            else:
                self.best_acc_per_task[tid] = max(self.best_acc_per_task[tid], acc)

        # Calculate AF
        self.last_af = 0.0
        if len(self.best_acc_per_task) > 1:
            forgetting = []
            for tid in range(self.current_task):
                if tid in self.best_acc_per_task and tid in self.current_acc_per_task:
                    f = self.best_acc_per_task[tid] - self.current_acc_per_task[tid]
                    forgetting.append(max(0, f))
            if forgetting:
                self.last_af = sum(forgetting) / len(forgetting)

    def get_current_af(self) -> float:
        """Get the last computed Average Forgetting value."""
        return self.last_af


# Pre-built combined trainers for convenience
from ..federated.fedavg import FedAvgTrainer
from ..federated.fedprox import FedProxTrainer


class FedAvgEWCTrainer(EWCMixin, FedAvgTrainer):
    """FedAvg + EWC: Standard averaging with EWC regularization."""

    def __init__(self, **kwargs):
        ewc_args = {
            "ewc_lambda": kwargs.pop("ewc_lambda", 1000.0),
            "fisher_samples": kwargs.pop("fisher_samples", 200),
            "online_ewc": kwargs.pop("online_ewc", False),
            "gamma": kwargs.pop("gamma", 0.9),
            "temp_dir": kwargs.pop("temp_dir", "./temp_ewc_storage"),
        }
        EWCMixin.__init__(self, **ewc_args)
        FedAvgTrainer.__init__(self)


class FedProxEWCTrainer(EWCMixin, FedProxTrainer):
    """FedProx + EWC: Proximal regularization with EWC."""

    def __init__(self, **kwargs):
        ewc_args = {
            "ewc_lambda": kwargs.pop("ewc_lambda", 1000.0),
            "fisher_samples": kwargs.pop("fisher_samples", 200),
            "online_ewc": kwargs.pop("online_ewc", False),
            "gamma": kwargs.pop("gamma", 0.9),
            "temp_dir": kwargs.pop("temp_dir", "./temp_ewc_storage"),
        }
        mu = kwargs.pop("mu", 0.01)
        EWCMixin.__init__(self, **ewc_args)
        FedProxTrainer.__init__(self, mu=mu)
