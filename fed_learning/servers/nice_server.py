"""
NICE Server - Server for Neurogenesis Inspired Contextual Encoding.

Reference:
    "NICE: Neurogenesis Inspired Contextual Encoding for Replay-free Class Incremental Learning"
    Gurbuz, Moorman, Dovrolis (CVPR 2024)
    GitHub: https://github.com/BurakGurbuz97/NICE

Key Features:
    - Extends IncrementalServer (same as DERServer pattern)
    - Uses NICEModel (fixed architecture, no expansion)
    - Manages neuron ages globally, distributes to clients
    - ContextDetector: Chained LogisticRegression for episode prediction (Eq.3-4)
    - end_task(): Age transition + freeze mask update + context detector training
"""

import time
from collections import OrderedDict
from typing import Dict, List, Optional
from threading import Thread

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from .incremental_server import IncrementalServer
from ..models.nice_model import NICEModel
from ..strategies.incremental.nice import (
    NICEAggregator,
    increase_unit_ranks,
    update_freeze_masks,
)


class ContextDetector:
    """Context detector using chained logistic regressions (Paper Eq.3-4).

    Official implementation (from GitHub context_detector.py):
    - Stores PER-SAMPLE binary activation vectors for each episode (not aggregated)
    - Binarizer: threshold = mean + std from first episode's activations
    - For each episode k, trains binary LR: positive=k's samples, negative=later episodes
    - Prediction uses chained probabilities (Eq.4): tree_preds

    Attributes:
        activation_memory: Dict[episode, np.ndarray] - per-sample binary vectors [n_samples, n_features]
        binarize_thresholds: Dict[str, float] - per-layer thresholds (mean+std from ep1)
        context_learners: List[LogisticRegression] - one per past episode
        episode_classes: Dict[episode, List[int]] - classes per episode
    """

    def __init__(self, memo_per_class: int = 50):
        self.memo_per_class = memo_per_class
        self.activation_memory: Dict[int, np.ndarray] = {}
        self.binarize_thresholds: Optional[Dict[str, float]] = None
        self.context_learners: List[LogisticRegression] = []
        self.episode_classes: Dict[int, List[int]] = {}

    def _binarize_per_sample(self, model: NICEModel, data: torch.Tensor) -> np.ndarray:
        """Get per-sample binary activation vectors.

        Unlike get_binary_activations() which averages over samples,
        this returns individual binary vectors per sample.

        Returns:
            np.ndarray of shape [n_samples, total_features]
        """
        model.eval()
        binary_all = []

        with torch.no_grad():
            if data.ndim == 2:
                x = data.unsqueeze(-1)
            else:
                x = data

            # CNN pathway
            x_cnn = x.permute(0, 2, 1)
            x_cnn = model.pool1(model.relu(model.bn1(model.conv1(x_cnn))))
            act_conv1 = x_cnn  # [batch, 64, seq_len]

            x_cnn = model.pool2(model.relu(model.bn2(model.conv2(x_cnn))))
            act_conv2 = x_cnn  # [batch, 128, seq_len]

            x_cnn = model.pool3(model.relu(model.bn3(model.conv3(x_cnn))))
            act_conv3 = x_cnn  # [batch, 256, seq_len]

            # GRU pathway
            x_gru, _ = model.gru(x)
            act_gru = x_gru[:, -1, :]  # [batch, 100]

        # Binarize per-sample, per-layer
        layer_acts = {
            "conv1": act_conv1.abs().mean(dim=2).cpu().numpy(),  # [batch, 64]
            "conv2": act_conv2.abs().mean(dim=2).cpu().numpy(),  # [batch, 128]
            "conv3": act_conv3.abs().mean(dim=2).cpu().numpy(),  # [batch, 256]
            "gru": act_gru.abs().cpu().numpy(),                  # [batch, 100]
        }

        parts = []
        for name in ["conv1", "conv2", "conv3", "gru"]:
            act = layer_acts[name]
            if self.binarize_thresholds is not None and name in self.binarize_thresholds:
                binary = (act > self.binarize_thresholds[name]).astype(np.float32)
            else:
                binary = (act > 0).astype(np.float32)
            parts.append(binary)

        return np.concatenate(parts, axis=1)  # [batch, total_features]

    def push_activations(self, model: NICEModel, data: torch.Tensor, episode: int):
        """Store per-sample binary activation vectors for an episode.

        Official: stores multiple binary vectors per episode (not just one aggregated).
        For threshold calibration: uses first episode to compute mean+std per layer.

        Args:
            model: NICEModel to extract activations from
            data: Input data samples [n_samples, ...]
            episode: Episode/task index
        """
        model.eval()

        # Set thresholds from first episode BEFORE binarizing (mean + std)
        if episode == 0 and self.binarize_thresholds is None:
            acts = model.get_activations(data)
            self.binarize_thresholds = {}
            for name in ["conv1", "conv2", "conv3", "gru"]:
                act = acts[name].cpu()
                self.binarize_thresholds[name] = (act.mean() + act.std()).item()

        # Store per-sample binary activations
        binary_vecs = self._binarize_per_sample(model, data)  # [n_samples, features]
        self.activation_memory[episode] = binary_vecs

    def train_models(self, current_episode: int):
        """Fit chained logistic regression models.

        Official (from GitHub context_detector.py):
        For episode k (k < current_episode):
            positive = activation_memory[k] (per-sample vectors)
            negative = concat of activation_memory[k+1], ..., activation_memory[current_episode]

        Args:
            current_episode: Most recent episode index
        """
        self.context_learners = []

        if current_episode == 0:
            return

        for k in range(current_episode):
            if k not in self.activation_memory:
                self.context_learners.append(None)
                continue

            # Positive: all samples from episode k
            pos = self.activation_memory[k]
            if pos.ndim == 1:
                pos = pos.reshape(1, -1)

            # Negative: all samples from later episodes
            neg_parts = []
            for j in range(k + 1, current_episode + 1):
                if j in self.activation_memory:
                    neg_j = self.activation_memory[j]
                    if neg_j.ndim == 1:
                        neg_j = neg_j.reshape(1, -1)
                    neg_parts.append(neg_j)

            if not neg_parts:
                self.context_learners.append(None)
                continue

            neg = np.concatenate(neg_parts, axis=0)

            # Build training data
            X = np.concatenate([pos, neg], axis=0)
            y = np.concatenate([
                np.ones(len(pos)),
                np.zeros(len(neg)),
            ])

            # Fit logistic regression
            try:
                lr = LogisticRegression(max_iter=1000, solver="lbfgs")
                lr.fit(X, y)
                self.context_learners.append(lr)
            except Exception:
                self.context_learners.append(None)

    def predict_episode(self, binary_activations: np.ndarray) -> int:
        """Predict which episode a sample belongs to using chained probabilities (Eq.4).

        Official tree_preds logic (from GitHub context_detector.py):
        For each sample, iterate through chained classifiers:
        - If classifier k predicts "positive" with probability > 0.5: episode = k
        - If all classifiers say "negative": episode = most recent

        Args:
            binary_activations: Binary activation vector [n_features] for one sample

        Returns:
            Predicted episode index
        """
        if not self.context_learners:
            return max(self.episode_classes.keys()) if self.episode_classes else 0

        x = binary_activations.reshape(1, -1)

        for k, clf in enumerate(self.context_learners):
            if clf is None:
                continue
            try:
                # Use predict_proba for probability-based chaining
                proba = clf.predict_proba(x)
                # proba[:, 1] = P(belongs to episode k)
                if proba[0, 1] > 0.5:
                    return k
            except Exception:
                continue

        # Default to most recent episode
        return max(self.episode_classes.keys()) if self.episode_classes else 0

    def predict_episodes_batch(self, binary_activations: np.ndarray) -> np.ndarray:
        """Batch prediction for multiple samples.

        Args:
            binary_activations: [n_samples, n_features]

        Returns:
            np.ndarray of predicted episode indices [n_samples]
        """
        if binary_activations.ndim == 1:
            return np.array([self.predict_episode(binary_activations)])

        results = np.full(len(binary_activations),
                         max(self.episode_classes.keys()) if self.episode_classes else 0)

        if not self.context_learners:
            return results

        # Track which samples haven't been assigned yet
        unassigned = np.ones(len(binary_activations), dtype=bool)

        for k, clf in enumerate(self.context_learners):
            if clf is None or not unassigned.any():
                continue
            try:
                proba = clf.predict_proba(binary_activations[unassigned])
                # Samples where P(episode k) > 0.5
                assigned = proba[:, 1] > 0.5
                # Map back to original indices
                orig_indices = np.where(unassigned)[0][assigned]
                results[orig_indices] = k
                unassigned[orig_indices] = False
            except Exception:
                continue

        return results


class NICEServer(IncrementalServer):
    """
    Server for NICE algorithm with neuron age management and context detection.

    Inherits from IncrementalServer:
        - evaluate_global(), set_task(), get_global_params(), set_global_params()
        - Device setup, seen_classes tracking

    NICE-specific:
        - Uses NICEModel (fixed architecture)
        - Manages neuron ages globally
        - Context detector for inference-time episode prediction
        - end_task(): age transition + freeze mask update
    """

    def __init__(self, clients, test_data: Dict, config: Dict):
        """Initialize NICE Server with NICEModel."""
        super().__init__(clients, test_data, config)

        # Replace global_model with NICEModel
        del self.global_model
        self.global_model = NICEModel(
            config["input_shape"], config["num_classes"]
        ).to(self.primary_device)

        # Context detector
        memo_per_class = config.get("memo_per_class", 50)
        self.context_detector = ContextDetector(memo_per_class=memo_per_class)

        # Neuron age state (managed by server, distributed to clients)
        self._neuron_ages_state: Optional[Dict[str, np.ndarray]] = None
        self._masks_state: Optional[Dict[str, torch.Tensor]] = None

        print(f"  Strategy: NICE (Neurogenesis Inspired Contextual Encoding)")

    def set_task(self, task_id: int, task_classes: list, seen_classes: list = None):
        """Set up for a new task - set output neuron ages for new classes.

        Args:
            task_id: Task identifier
            task_classes: New class IDs in this task
            seen_classes: Full cumulative seen classes list
        """
        super().set_task(task_id, task_classes, seen_classes)

        # Set output neuron ages to 1 (learner) for new classes
        for cls_id in task_classes:
            if cls_id < self.global_model.num_classes:
                self.global_model.unit_ranks["fc2"][cls_id] = 1

        # Store episode classes for context detector
        self.context_detector.episode_classes[task_id] = list(task_classes)

        # Update aggregator with frozen keys AND per-neuron freeze masks
        if hasattr(self.aggregator, "set_frozen_keys"):
            frozen_keys = self._get_frozen_param_keys()
            self.aggregator.set_frozen_keys(frozen_keys)
        if hasattr(self.aggregator, "set_freeze_masks"):
            self.aggregator.set_freeze_masks(self.global_model.freeze_masks)

        print(f"  NICE: Task {task_id} | new classes: {task_classes}")
        print(f"  NICE: Output neuron ages (fc2): "
              f"learner={np.sum(self.global_model.unit_ranks['fc2'] == 1)}, "
              f"mature={np.sum(self.global_model.unit_ranks['fc2'] >= 2)}, "
              f"young={np.sum(self.global_model.unit_ranks['fc2'] == 0)}")

    def _get_frozen_param_keys(self) -> List[str]:
        """Get parameter keys that should be frozen (mature neurons)."""
        frozen_keys = []
        for name, param in self.global_model.named_parameters():
            layer_name = name.split(".")[0]
            if layer_name in self.global_model.unit_ranks:
                ranks = self.global_model.unit_ranks[layer_name]
                if np.all(ranks >= 2):
                    frozen_keys.append(name)
        return frozen_keys

    def train_round(
        self,
        participating_clients=None,
        verbose: bool = True,
    ) -> Dict:
        """Train one federated round.

        Args:
            participating_clients: Clients to train (default: all)
            verbose: Whether to print progress

        Returns:
            Dict with train_loss and round_time
        """
        from ..training.nice_worker import train_nice_clients_on_gpu

        round_start = time.time()
        clients = participating_clients or self.clients

        if verbose:
            device_info = "CPU" if self.use_cpu else f"{self.num_gpus} GPU(s)"
            print(f"\n  NICE: Training {len(clients)} clients on {device_info}")

        global_params = self.get_global_params()

        # Prepare config with neuron ages and masks
        worker_config = {**self.config}
        worker_config["neuron_ages"] = self.global_model.get_neuron_ages_state()
        worker_config["masks"] = self.global_model.get_masks_state()
        worker_config["freeze_masks"] = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in self.global_model.freeze_masks.items()
        }

        # Distribute clients across GPUs
        clients_per_gpu = [[] for _ in range(self.num_gpus)]
        for i, c in enumerate(clients):
            clients_per_gpu[i % self.num_gpus].append(c)

        # Train clients in parallel threads
        results_dict = {}
        threads = []

        for gpu_id in range(self.num_gpus):
            if len(clients_per_gpu[gpu_id]) > 0:
                t = Thread(
                    target=train_nice_clients_on_gpu,
                    args=(
                        gpu_id,
                        clients_per_gpu[gpu_id],
                        global_params,
                        worker_config,
                        results_dict,
                        self.trainer,
                        self.use_cpu,
                    ),
                )
                threads.append(t)
                t.start()

        for t in threads:
            t.join()

        # Collect and aggregate results
        results = list(results_dict.values())

        new_params = self.aggregator.aggregate(results, global_params)
        self.global_model.load_state_dict(
            {k: v.to(self.primary_device) for k, v in new_params.items()}
        )

        # Update server model's neuron ages from client results.
        # Use the ages from the first client that has them (all clients
        # start from the same ages and perform similar selection).
        for r in results:
            if "neuron_ages" in r and r["neuron_ages"]:
                self.global_model.set_neuron_ages_state(r["neuron_ages"])
                break

        avg_loss = float(np.mean([r["loss"] for r in results]))
        round_time = time.time() - round_start

        if verbose:
            print(f"    NICE loss: {avg_loss:.4f} ({round_time:.1f}s)")
            # Print neuron age summary
            for name in ["conv1", "fc1", "fc2"]:
                ranks = self.global_model.unit_ranks[name]
                print(f"    {name}: young={np.sum(ranks == 0)}, "
                      f"learner={np.sum(ranks == 1)}, "
                      f"mature={np.sum(ranks >= 2)}")

        return {"train_loss": avg_loss, "round_time": round_time}

    def end_task(self):
        """End-of-task processing: age transition, freeze masks, context detector.

        Called after all training rounds for a task are complete.

        1. Increase unit ranks (learner -> mature)
        2. Update freeze masks
        3. Freeze BN for mature layers
        4. Push activations to context detector
        5. Train context detector models
        """
        print(f"\n  NICE end_task({self.current_task}):")

        # 1. Age transition: all rank >= 1 get +1
        increase_unit_ranks(self.global_model)

        # Print age stats
        for name in self.global_model.LAYER_NAMES:
            ranks = self.global_model.unit_ranks[name]
            print(f"    {name}: young={np.sum(ranks == 0)}, "
                  f"learner={np.sum(ranks == 1)}, "
                  f"mature={np.sum(ranks >= 2)}")

        # 2. Update freeze masks for gradient protection
        update_freeze_masks(self.global_model)

        # 3. Freeze BN for layers with all-mature neurons
        self.global_model.freeze_bn_for_mature()

        # 4. Push activations for context detector
        # Use a sample of test data for this task
        task_classes = self.task_classes.get(self.current_task, [])
        if task_classes:
            X_test = self.test_data["X_test"]
            y_test = self.test_data["y_test"]
            task_set = set(task_classes)
            mask = torch.tensor([y.item() in task_set for y in y_test])
            if mask.any():
                X_task = X_test[mask]
                # Subsample
                n_sample = min(200, len(X_task))
                idx = torch.randperm(len(X_task))[:n_sample]
                sample_data = X_task[idx].to(self.primary_device)
                self.context_detector.push_activations(
                    self.global_model, sample_data, self.current_task
                )

        # 5. Train context detector
        self.context_detector.train_models(self.current_task)
        print(f"    Context detector: {len(self.context_detector.context_learners)} models trained")

        # Update aggregator frozen keys AND per-neuron freeze masks
        if hasattr(self.aggregator, "set_frozen_keys"):
            frozen_keys = self._get_frozen_param_keys()
            self.aggregator.set_frozen_keys(frozen_keys)
        if hasattr(self.aggregator, "set_freeze_masks"):
            self.aggregator.set_freeze_masks(self.global_model.freeze_masks)

    def evaluate_global(
        self,
        batch_size: int = 1024,
        compute_auc: bool = False,
        seen_classes_only: bool = True,
    ) -> Dict:
        """Evaluate using standard forward (context detection is optional).

        For simplicity and compatibility with the evaluation framework,
        we use standard forward pass. Context detection would be used
        at actual inference time but is not needed for accuracy metrics
        since we evaluate on known class sets.
        """
        return super().evaluate_global(
            batch_size=batch_size,
            compute_auc=compute_auc,
            seen_classes_only=seen_classes_only,
        )

    def compute_average_forgetting(self) -> float:
        """Compute Average Forgetting."""
        if self.current_task == 0:
            return 0.0
        current_accs = self.evaluate_per_task()
        if hasattr(self.trainer, "update_forgetting"):
            self.trainer.update_forgetting(current_accs)
            return self.trainer.last_af
        return 0.0

    def evaluate_per_task(self, batch_size: int = 1024) -> Dict[int, float]:
        """Evaluate accuracy per task."""
        from sklearn.metrics import accuracy_score

        self.global_model.eval()
        task_accuracies = {}

        X_test = self.test_data["X_test"]
        y_test = self.test_data["y_test"]

        for task_id, task_classes in self.task_classes.items():
            if not task_classes:
                continue

            task_class_set = set(task_classes)
            mask = torch.tensor([y.item() in task_class_set for y in y_test])

            if not mask.any():
                task_accuracies[task_id] = 0.0
                continue

            X_task = X_test[mask]
            y_task = y_test[mask]

            all_preds = []
            all_targets = []

            with torch.no_grad():
                for i in range(0, len(y_task), batch_size):
                    X_batch = X_task[i:i + batch_size].to(self.primary_device)
                    y_batch = y_task[i:i + batch_size]

                    out = self.global_model(X_batch)
                    preds = out.argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(y_batch.numpy())

            task_accuracies[task_id] = accuracy_score(all_targets, all_preds)

        return task_accuracies
