"""
GLFC Strategy - Global-Local Forgetting Compensation for Federated Class-Incremental Learning.

Reference:
    Dong et al., "Federated Class-Incremental Learning",
    CVPR 2022

Paper Algorithm Summary:
========================
1. Local Forgetting Compensation (Section 3.2):
   - Knowledge Distillation: L_old = BCE(σ(z_new), σ(z_old)) on old class logits
   - Class-aware gradient compensation: reweight per-sample loss by |pred - target|
   - Combined loss: L = 0.5 * L_cur + 0.5 * L_old (when old model exists)

2. Global Forgetting Compensation (Section 3.3):
   - Entropy-based signal: detect knowledge shift via entropy increase
   - If entropy increase > threshold (1.2), signal=True → update exemplar set
   - Exemplar set management: herding-based selection per class

3. Proxy Server (Section 3.4):
   - Prototype gradient sharing: clients share gradients of class prototypes
   - Server reconstructs pseudo data via gradient inversion (LBFGS optimization)
   - Monitor reconstructed data accuracy → track best model versions
   - Return [best_model_1, best_model_2] to clients for distillation

4. Class-Aware Gradient Compensation (Eq. in paper):
   - w_i = |pred_i - target_i| (prediction error as weight)
   - Separate re-normalization for old-class vs new-class samples
   - Ensures balanced gradient contributions despite class imbalance

5. Federated Aggregation:
   - Standard FedAvg for model aggregation
   - Proxy server maintains best historical models for forgetting compensation
"""

import copy
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ...core import BaseTrainer, BaseAggregator


def get_one_hot(target: torch.Tensor, num_class: int, device: str) -> torch.Tensor:
    """Convert labels to one-hot encoding."""
    one_hot = torch.zeros(target.shape[0], num_class, device=device)
    one_hot.scatter_(dim=1, index=target.long().view(-1, 1), value=1.0)
    return one_hot


def compute_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Compute entropy of probability distribution (per sample)."""
    entropy = -probs * torch.log(probs + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy


class GLFCTrainer(BaseTrainer):
    """
    GLFC Trainer - Local training with forgetting compensation.

    Implements the local forgetting compensation mechanism:
    1. Knowledge Distillation with old model (binary CE on old class logits)
    2. Class-aware gradient compensation (reweight by prediction error)
    3. Entropy-based signal for exemplar set update

    Args:
        memory_size: Total memory budget for exemplar storage
        entropy_threshold: Threshold for entropy change to trigger signal (paper: 1.2)
        distill_weight: Weight for distillation loss (paper: 0.5)
        temp_dir: Directory for temporary storage
    """

    def __init__(
        self,
        memory_size: int = 2000,
        entropy_threshold: float = 1.2,
        distill_weight: float = 0.5,
        temp_dir: str = "./temp_glfc_storage",
        **kwargs,
    ):
        self.memory_size = memory_size
        self.entropy_threshold = entropy_threshold
        self.distill_weight = distill_weight
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)

        # Task tracking
        self.current_task: int = 0
        self.seen_classes: Set[int] = set()
        self.old_classes: List[int] = []
        self.new_classes: List[int] = []
        self.numclass: int = 0  # Total number of classes seen so far

        # Old model for distillation (from proxy server)
        self.old_model_states: Dict[str, OrderedDict] = {}  # "best_1", "best_2"
        self._cached_old_model: Optional[nn.Module] = None
        self._cached_device: Optional[str] = None

        # Exemplar memory: {class_id: List[tensor]}
        # Each tensor is a single sample feature vector
        self.exemplar_set: Dict[int, List[torch.Tensor]] = {}
        self.learned_classes: List[int] = []
        self.learned_numclass: int = 0

        # Entropy tracking for signal detection
        self.last_entropy: float = 0.0

        # Forgetting metrics (for compatibility with main training script)
        self.mu_coefficient: float = 1.0
        self.best_acc_per_task: Dict[int, float] = {}
        self.current_acc_per_task: Dict[int, float] = {}
        self.last_af: float = 0.0

        # Proxy server state
        self.best_model_1: Optional[OrderedDict] = None  # Previous best
        self.best_model_2: Optional[OrderedDict] = None  # Current best
        self.best_perf: float = 0.0

        # Reconstructed proxy data for monitoring
        self.proxy_data: List = []
        self.proxy_labels: List = []

    def set_task(self, task_id: int, new_classes: List[int]):
        """Called at the beginning of each new task."""
        self.old_classes = list(self.seen_classes)
        self.new_classes = new_classes
        self.current_task = task_id
        self.seen_classes.update(new_classes)
        self.numclass = len(self.seen_classes)

        # Invalidate cached old model
        self._cached_old_model = None
        self._cached_device = None

        print(
            f"  GLFC Task {task_id}: old_classes={len(self.old_classes)}, "
            f"new_classes={len(new_classes)}, total={self.numclass}"
        )

    def save_model_snapshot(self, model: nn.Module):
        """Save model snapshot as potential old model for distillation."""
        state_dict = OrderedDict(
            (k, v.cpu().clone()) for k, v in model.state_dict().items()
        )
        self.old_model_states["latest"] = state_dict

        # Save to disk
        path = os.path.join(self.temp_dir, f"task_{self.current_task}_model.pt")
        torch.save(state_dict, path)
        print(f"  Saved GLFC model snapshot for Task {self.current_task}")

    def update_proxy_server_models(self, model: nn.Module, perf: float):
        """
        Update proxy server's best model tracking.

        Paper Section 3.4: Proxy server monitors reconstructed data accuracy
        and maintains best_model_1 (previous best) and best_model_2 (current best).
        """
        state_dict = OrderedDict(
            (k, v.cpu().clone()) for k, v in model.state_dict().items()
        )

        if perf >= self.best_perf:
            self.best_perf = perf
            self.best_model_1 = self.best_model_2
            self.best_model_2 = state_dict

    def get_old_models(self):
        """
        Get old model states for distillation.

        Paper: Proxy server returns [best_model_1, best_model_2]
        Client uses best_model_2 if signal=True, else best_model_1.
        """
        return self.best_model_1, self.best_model_2

    def load_old_model(
        self, model_template: nn.Module, device: str, signal: bool = True
    ) -> Optional[nn.Module]:
        """
        Load old model for knowledge distillation.

        Paper Section 3.4:
        - If signal=True (entropy increase detected): use best_model_2
        - If signal=False: use best_model_1

        Falls back to latest saved model if proxy models not available.
        """
        if self.current_task == 0:
            return None

        # Check cache
        if self._cached_old_model is not None and self._cached_device == device:
            return self._cached_old_model

        # Select old model based on signal
        state_dict = None
        if signal and self.best_model_2 is not None:
            state_dict = self.best_model_2
        elif not signal and self.best_model_1 is not None:
            state_dict = self.best_model_1
        elif signal and self.best_model_1 is not None:
            # Fallback: use best_model_1 if best_model_2 not available
            state_dict = self.best_model_1

        if state_dict is None:
            # Final fallback: use latest saved model
            state_dict = self.old_model_states.get("latest")

        if state_dict is None:
            return None

        try:
            old_model = copy.deepcopy(model_template)
            old_model.load_state_dict(
                {k: v.to(device) for k, v in state_dict.items()}
            )
            old_model.eval()
            for param in old_model.parameters():
                param.requires_grad = False

            self._cached_old_model = old_model
            self._cached_device = device
            return old_model

        except Exception as e:
            print(f"  Warning: Failed to load GLFC old model: {e}")
            return None

    def compute_entropy_signal(
        self, model: nn.Module, data_loader, device: str
    ) -> bool:
        """
        Entropy-based signal detection for global forgetting compensation.

        Paper Section 3.3:
        - Compute average entropy of model predictions on current data
        - If entropy increase > threshold (1.2), signal=True
        - Signal=True means new task knowledge detected → update exemplar set

        Args:
            model: Current model
            data_loader: DataLoader for current task data
            device: Device

        Returns:
            True if entropy signal detected (new knowledge shift)
        """
        model.eval()
        all_entropy = []

        with torch.no_grad():
            for batch in data_loader:
                if len(batch) == 2:
                    X_batch, y_batch = batch
                else:
                    X_batch, y_batch = batch[0], batch[1]

                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                softmax_out = F.softmax(outputs, dim=1)
                ent = compute_entropy(softmax_out)
                all_entropy.append(ent.cpu())

        if not all_entropy:
            return False

        all_ent = torch.cat(all_entropy, dim=0)
        overall_avg = torch.mean(all_ent).item()

        signal = (overall_avg - self.last_entropy) > self.entropy_threshold
        self.last_entropy = overall_avg

        if signal:
            print(f"    Entropy signal detected: avg_entropy={overall_avg:.4f}")

        return signal

    def efficient_old_class_weight(
        self,
        output: torch.Tensor,
        label: torch.Tensor,
        num_class: int,
        device: str,
    ) -> torch.Tensor:
        """
        Class-aware gradient compensation weights.

        Paper mechanism:
        - w_i = |pred_i - target_i| (prediction error as weight)
        - Separate normalization for old-class vs new-class samples
        - Ensures balanced gradient contributions

        Args:
            output: Model logits [batch, num_classes]
            label: Ground truth labels [batch]
            num_class: Total number of classes
            device: Device

        Returns:
            Per-sample weights [batch, 1]
        """
        pred = torch.sigmoid(output)
        N, C = pred.size(0), pred.size(1)

        # Create class mask
        class_mask = pred.data.new(N, C).fill_(0)
        ids = label.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.0)

        # Compute per-sample prediction error
        target = get_one_hot(label, num_class, device)
        g = torch.abs(pred.detach() - target)
        g = (g * class_mask).sum(1).view(-1, 1)

        if len(self.learned_classes) != 0:
            # Separate old-class and new-class samples
            ids_check = ids.clone()
            for i in self.learned_classes:
                ids_check = torch.where(
                    ids_check != i, ids_check, ids_check.clone().fill_(-1)
                )

            # index1: old-class samples (ids became -1)
            index1 = torch.eq(ids_check, -1).float()
            # index2: new-class samples
            index2 = torch.ne(ids_check, -1).float()

            # Normalize within each group
            if index1.sum() != 0:
                w1 = torch.div(g * index1, (g * index1).sum() / index1.sum())
            else:
                w1 = g.clone().fill_(0.0)

            if index2.sum() != 0:
                w2 = torch.div(g * index2, (g * index2).sum() / index2.sum())
            else:
                w2 = g.clone().fill_(0.0)

            w = w1 + w2
        else:
            w = g.clone().fill_(1.0)

        return w

    def compute_loss(
        self,
        model: nn.Module,
        output: torch.Tensor,
        target: torch.Tensor,
        global_params: Optional[OrderedDict] = None,
        inputs: Optional[torch.Tensor] = None,
        old_model: Optional[nn.Module] = None,
        signal: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute GLFC loss with local forgetting compensation.

        Paper Section 3.2:
        L_total = 0.5 * L_cur + 0.5 * L_old  (when old model exists)
        L_cur = weighted BCE(output, target) with class-aware weights
        L_old = BCE(output, distill_target) where distill_target uses old model logits

        Args:
            model: Current model
            output: Model logits
            target: Ground truth labels
            global_params: Not used
            inputs: Input features (needed for old model forward)
            old_model: Old model for distillation
            signal: Entropy signal flag

        Returns:
            Loss tensor
        """
        num_class = self.numclass if self.numclass > 0 else output.size(1)
        device = output.device

        # One-hot target
        one_hot_target = get_one_hot(target, num_class, str(device))

        if old_model is None:
            # First task or no old model: just weighted BCE (LGC only, λ2=0)
            w = self.efficient_old_class_weight(output, target, num_class, str(device))
            loss_cur = torch.mean(
                w
                * F.binary_cross_entropy_with_logits(
                    output, one_hot_target, reduction="none"
                )
            )
            return loss_cur
        else:
            # With old model: LGC + LRD, λ1=λ2=0.5 (Eq. 6)
            w = self.efficient_old_class_weight(output, target, num_class, str(device))
            loss_cur = torch.mean(
                w
                * F.binary_cross_entropy_with_logits(
                    output, one_hot_target, reduction="none"
                )
            )

            # LRD: Class-Semantic Relation Distillation (Eq. 5)
            # Paper: Y^t_l replaces first Cp (old class) dims with old model probabilities.
            # Use len(self.old_classes) as Cp — NOT old_output.shape[1] which is always
            # the full 34-class output and would incorrectly overwrite unseen class dims.
            distill_target = one_hot_target.clone()
            with torch.no_grad():
                if inputs is not None:
                    old_output = torch.sigmoid(old_model(inputs))
                else:
                    old_output = torch.sigmoid(old_model(output))

            # Cp = number of classes seen before this task
            old_task_size = len(self.old_classes) if self.old_classes else 0
            if old_task_size > 0:
                distill_target[..., :old_task_size] = old_output[..., :old_task_size]
            loss_old = F.binary_cross_entropy_with_logits(output, distill_target)

            return self.distill_weight * loss_cur + self.distill_weight * loss_old

    def update_forgetting(self, task_accuracies: Dict[int, float]):
        """Update accuracy tracking for forgetting metrics."""
        self.current_acc_per_task = task_accuracies.copy()

        for task_id, acc in task_accuracies.items():
            if task_id not in self.best_acc_per_task:
                self.best_acc_per_task[task_id] = acc
            else:
                self.best_acc_per_task[task_id] = max(
                    self.best_acc_per_task[task_id], acc
                )

        # Compute Average Forgetting
        if len(self.best_acc_per_task) > 1:
            forgetting_sum = 0.0
            count = 0
            for task_id in self.best_acc_per_task:
                if (
                    task_id != self.current_task
                    and task_id in self.current_acc_per_task
                ):
                    forgetting = (
                        self.best_acc_per_task[task_id]
                        - self.current_acc_per_task[task_id]
                    )
                    forgetting_sum += max(0, forgetting)
                    count += 1
            self.last_af = forgetting_sum / max(1, count)

    def cleanup(self):
        """Clean up temporary files."""
        self._cached_old_model = None
        self.old_model_states.clear()
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


class GLFCAggregator(BaseAggregator):
    """
    GLFC Aggregation - Standard FedAvg weighted average.

    GLFC uses standard FedAvg for model aggregation.
    The forgetting compensation happens at the local training level
    and through the proxy server mechanism.
    """

    def aggregate(
        self,
        results: List[Dict],
        global_params: Optional[OrderedDict] = None,
        **kwargs,
    ) -> OrderedDict:
        """Standard FedAvg aggregation."""
        return self._weighted_average(results)
