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
from ..strategies.fed_incremental.nice import (
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
        """Khởi tạo bộ nhớ activation và các bộ phân loại context theo episode."""
        self.memo_per_class = memo_per_class
        self.activation_memory: Dict[int, np.ndarray] = {}
        self.context_masks: Dict[int, np.ndarray] = {}
        self.binarize_thresholds: Optional[Dict[str, float]] = None
        self.context_learners: List[LogisticRegression] = []
        self.episode_classes: Dict[int, List[int]] = {}

    def _binarize_per_sample(self, model: NICEModel, data: torch.Tensor) -> np.ndarray:
        """Get per-sample binary activation vectors.

        Unlike get_binary_activations() which averages over samples,
        this returns individual binary vectors per sample.

        Returns:
            np.ndarray of shape [n_samples, total_features]

        Hàm này biến activation liên tục thành vector nhị phân để context detector
        có thể học episode/task một cách gọn nhẹ hơn.
        """
        model.eval()
        layer_acts = {
            name: act.cpu().numpy()
            for name, act in model.get_context_activations_per_sample(data).items()
        }
        return self.binarize_layer_activations(layer_acts)

    def binarize_layer_activations(self, layer_acts: Dict[str, np.ndarray]) -> np.ndarray:
        """Binarize already-computed per-sample context activations."""
        parts = []
        for name in ["conv1", "conv2", "conv3", "gru"]:
            act = layer_acts[name]
            if (
                self.binarize_thresholds is not None
                and name in self.binarize_thresholds
            ):
                binary = (act > self.binarize_thresholds[name]).astype(np.float32)
            else:
                binary = (act > 0).astype(np.float32)
            parts.append(binary)

        return np.concatenate(parts, axis=1)  # [batch, total_features]

    def _get_context_mask(self, model: NICEModel) -> np.ndarray:
        """Official-style context mask: only units allocated by this episode."""
        parts = []
        for name in ["conv1", "conv2", "conv3", "gru"]:
            ranks = getattr(model, "unit_ranks", {}).get(name)
            if ranks is not None:
                parts.append(np.asarray(ranks) > 0)
        if not parts:
            return np.ones(0, dtype=bool)
        mask = np.concatenate(parts).astype(bool)
        if not mask.any():
            mask[:] = True
        return mask

    def push_activations(self, model: NICEModel, data: torch.Tensor, episode: int):
        """Store per-sample binary activation vectors for an episode.

        Official: stores multiple binary vectors per episode (not just one aggregated).
        For threshold calibration: uses first episode to compute mean+std per layer.

        Args:
            model: NICEModel to extract activations from
            data: Input data samples [n_samples, ...]
            episode: Episode/task index

        Với episode đầu tiên, hàm còn thiết lập ngưỡng nhị phân hóa ban đầu.
        """
        model.eval()

        # Set thresholds from first episode BEFORE binarizing (mean + std)
        if episode == 0 and self.binarize_thresholds is None:
            acts = model.get_context_activations_per_sample(data)
            self.binarize_thresholds = {}
            for name in ["conv1", "conv2", "conv3", "gru"]:
                act = acts[name].cpu()
                self.binarize_thresholds[name] = (act.mean() + act.std()).item()

        # Store per-sample binary activations
        binary_vecs = self._binarize_per_sample(model, data)  # [n_samples, features]
        self.activation_memory[episode] = binary_vecs
        self.context_masks[episode] = self._get_context_mask(model)

    def train_models(self, current_episode: int):
        """Fit chained logistic regression models.

        Official (from GitHub context_detector.py):
        For episode k (k < current_episode):
            positive = activation_memory[k] (per-sample vectors)
            negative = concat of activation_memory[k+1], ..., activation_memory[current_episode]

        Args:
            current_episode: Most recent episode index

        Sau bước này, context detector đã có chuỗi classifier để dự đoán episode.
        """
        self.context_learners = []

        if current_episode == 0:
            return

        for k in range(current_episode):
            if k not in self.activation_memory:
                self.context_learners.append(None)
                continue

            mask = self.context_masks.get(k)
            if mask is None or mask.size == 0:
                mask = np.ones(self.activation_memory[k].shape[1], dtype=bool)
            elif not mask.any():
                mask = np.ones(mask.shape[0], dtype=bool)

            # Positive: all samples from episode k
            pos = self.activation_memory[k][:, mask]
            if pos.ndim == 1:
                pos = pos.reshape(1, -1)

            # Negative: all samples from later episodes
            neg_parts = []
            for j in range(k + 1, current_episode + 1):
                if j in self.activation_memory:
                    neg_j = self.activation_memory[j]
                    if neg_j.ndim == 1:
                        neg_j = neg_j.reshape(1, -1)
                    neg_parts.append(neg_j[:, mask])

            if not neg_parts:
                self.context_learners.append(None)
                continue

            neg = np.concatenate(neg_parts, axis=0)

            # Build training data
            X = np.concatenate([pos, neg], axis=0)
            y = np.concatenate(
                [
                    np.ones(len(pos)),
                    np.zeros(len(neg)),
                ]
            )

            # Fit logistic regression
            try:
                lr = LogisticRegression(max_iter=1000, solver="lbfgs")
                lr.fit(X, y)
                self.context_learners.append(lr)
            except Exception:
                self.context_learners.append(None)

    def _predict_episode_threshold_unused(self, binary_activations: np.ndarray) -> int:
        """Predict which episode a sample belongs to using chained probabilities (Eq.4).

        Official tree_preds logic (from GitHub context_detector.py):
        For each sample, iterate through chained classifiers:
        - If classifier k predicts "positive" with probability > 0.5: episode = k
        - If all classifiers say "negative": episode = most recent

        Args:
            binary_activations: Binary activation vector [n_features] for one sample

        Returns:
            Predicted episode index

        Hàm chạy theo kiểu chained decision: nếu không classifier nào nhận mẫu,
        nó sẽ gán về episode mới nhất.
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

    def _predict_episodes_batch_threshold_unused(self, binary_activations: np.ndarray) -> np.ndarray:
        """Batch prediction for multiple samples.

        Args:
            binary_activations: [n_samples, n_features]

        Returns:
            np.ndarray of predicted episode indices [n_samples]
        """
        if binary_activations.ndim == 1:
            return np.array([self.predict_episode(binary_activations)])

        results = np.full(
            len(binary_activations),
            max(self.episode_classes.keys()) if self.episode_classes else 0,
        )

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

    def _mask_for_episode(self, episode: int, n_features: int) -> np.ndarray:
        """Return stored context mask, falling back when dimensions drift."""
        mask = self.context_masks.get(episode)
        if mask is None or mask.size == 0 or mask.shape[0] != n_features:
            return np.ones(n_features, dtype=bool)
        if not mask.any():
            return np.ones(mask.shape[0], dtype=bool)
        return mask

    def predict_episode(self, binary_activations: np.ndarray) -> int:
        """Predict one episode using official NICE chain-probability argmax."""
        return int(self.predict_episodes_batch(binary_activations.reshape(1, -1))[0])

    def predict_episodes_with_scores(
        self, binary_activations: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return episode predictions and chain probabilities."""
        if binary_activations.ndim == 1:
            binary_activations = binary_activations.reshape(1, -1)

        latest_episode = max(self.episode_classes.keys()) if self.episode_classes else 0
        if not self.context_learners:
            preds = np.full(len(binary_activations), latest_episode, dtype=int)
            probs = np.ones((len(binary_activations), 1), dtype=np.float32)
            return preds, probs

        pos_probs = []
        for k, clf in enumerate(self.context_learners):
            if clf is None:
                pos_probs.append(np.zeros(len(binary_activations), dtype=np.float32))
                continue
            mask = self._mask_for_episode(k, binary_activations.shape[1])
            try:
                proba = clf.predict_proba(binary_activations[:, mask])
                pos_probs.append(proba[:, 1])
            except Exception:
                pos_probs.append(np.zeros(len(binary_activations), dtype=np.float32))

        pos_probs_arr = np.asarray(pos_probs).T
        neg_probs = 1.0 - pos_probs_arr
        chain_probs = np.zeros(
            (len(binary_activations), len(self.context_learners) + 1),
            dtype=np.float32,
        )
        for episode_index in range(len(self.context_learners)):
            if episode_index == 0:
                chain_probs[:, 0] = pos_probs_arr[:, 0]
            else:
                prev_neg_prob = np.prod(neg_probs[:, :episode_index], axis=1)
                chain_probs[:, episode_index] = (
                    prev_neg_prob * pos_probs_arr[:, episode_index]
                )
        chain_probs[:, -1] = np.maximum(0.0, 1.0 - chain_probs.sum(axis=1))
        return chain_probs.argmax(axis=1).astype(int), chain_probs

    def predict_episodes_batch(self, binary_activations: np.ndarray) -> np.ndarray:
        """Batch episode prediction using official NICE tree_preds semantics."""
        preds, _probs = self.predict_episodes_with_scores(binary_activations)
        return preds


class NICEServer(IncrementalServer):
    """
    Server chuyên cho NICE với quản lý tuổi neuron và context detector.

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
        """Khởi tạo NICE server và thay global model mặc định bằng `NICEModel`."""
        super().__init__(clients, test_data, config)

        # Replace global_model with NICEModel
        del self.global_model
        self.global_model = NICEModel(config["input_shape"], config["num_classes"]).to(
            self.primary_device
        )

        # Context detector
        memo_per_class = config.get("memo_per_class", 50)
        self.context_detector = ContextDetector(memo_per_class=memo_per_class)

        # Neuron age state (managed by server, distributed to clients)
        self._neuron_ages_state: Optional[Dict[str, np.ndarray]] = None
        self._masks_state: Optional[Dict[str, torch.Tensor]] = None

        print(f"  Strategy: NICE (Neurogenesis Inspired Contextual Encoding)")

    def _sample_context_data_from_clients(self, task_classes: List[int]) -> torch.Tensor:
        """Sample recently seen training examples for NICE context memory."""
        if not task_classes:
            return torch.empty(0)

        per_class = max(1, int(getattr(self.context_detector, "memo_per_class", 50)))
        samples = []

        for cls_id in task_classes:
            remaining = per_class
            cls_chunks = []
            for client in self.clients:
                X_train = getattr(client, "X_train", None)
                y_train = getattr(client, "y_train", None)
                if X_train is None or y_train is None or len(y_train) == 0:
                    continue

                y_cpu = y_train.detach().cpu() if isinstance(y_train, torch.Tensor) else torch.as_tensor(y_train)
                idx = torch.nonzero(y_cpu == int(cls_id), as_tuple=False).flatten()
                if len(idx) == 0:
                    continue

                take = min(remaining, len(idx))
                if take <= 0:
                    break
                perm = torch.randperm(len(idx))[:take]
                selected = idx[perm]
                cls_chunks.append(X_train[selected].detach().cpu())
                remaining -= take
                if remaining <= 0:
                    break

            if cls_chunks:
                samples.append(torch.cat(cls_chunks, dim=0))

        if not samples:
            return torch.empty(0)
        return torch.cat(samples, dim=0)

    def update_context_detector_memory(self, verbose: bool = False) -> None:
        """
        Update NICE context memory from current task training samples.

        Paper Section 3.4 stores activation memory from recently seen examples
        every p epochs and retrains the chained context detector. In this
        federated simulator the server has access to simulated client tensors,
        so we sample from client train data rather than from test data.
        """
        task_classes = self.task_classes.get(self.current_task, [])
        if not task_classes:
            return

        sample_data = self._sample_context_data_from_clients(task_classes)
        if sample_data.numel() == 0:
            if verbose:
                print("    Context detector: no train samples available")
            return

        self.context_detector.push_activations(
            self.global_model,
            sample_data.to(self.primary_device),
            self.current_task,
        )
        self.context_detector.train_models(self.current_task)
        if verbose:
            print(
                f"    Context detector: {len(self.context_detector.context_learners)} models trained"
            )

    def set_task(self, task_id: int, task_classes: list, seen_classes: list = None):
        """Set up for a new task - set output neuron ages for new classes.

        Ngoài tracking task, hàm này còn đặt output neuron của lớp mới sang
        trạng thái learner và cập nhật freeze information cho aggregator.
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
        print(
            f"  NICE: Output neuron ages (fc2): "
            f"learner={np.sum(self.global_model.unit_ranks['fc2'] == 1)}, "
            f"mature={np.sum(self.global_model.unit_ranks['fc2'] >= 2)}, "
            f"young={np.sum(self.global_model.unit_ranks['fc2'] == 0)}"
        )

    def _get_frozen_param_keys(self) -> List[str]:
        """Tìm các tham số có thể freeze hoàn toàn vì layer tương ứng đã mature."""
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
        phase_offset: int = 0,
        max_phases_override: Optional[int] = None,
    ) -> Dict:
        """Train one federated round.

        Args:
            participating_clients: Clients to train (default: all)
            verbose: Whether to print progress

        Worker NICE sẽ nhận thêm neuron ages, masks và freeze masks để client
        train đồng bộ với trạng thái của server.
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
        worker_config["phase_offset"] = int(phase_offset)
        if max_phases_override is not None:
            worker_config["max_phases_override"] = int(max_phases_override)

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

        # Update server ages using the union/max over client selections rather
        # than an arbitrary first client. Clients start from the same ages but
        # can select different learner neurons on non-IID local data.
        merged_ages = self._merge_client_neuron_ages(results)
        if merged_ages:
            self.global_model.set_neuron_ages_state(merged_ages)

        # Paper Section 3.4 updates context memory every p epochs. This train
        # round is one exposed NICE phase, so refresh memory after aggregation.
        self.update_context_detector_memory(verbose=False)

        avg_loss = float(np.mean([r["loss"] for r in results]))
        round_time = time.time() - round_start

        if verbose:
            print(f"    NICE loss: {avg_loss:.4f} ({round_time:.1f}s)")
            # Print neuron age summary
            for name in ["conv1", "fc1", "fc2"]:
                ranks = self.global_model.unit_ranks[name]
                print(
                    f"    {name}: young={np.sum(ranks == 0)}, "
                    f"learner={np.sum(ranks == 1)}, "
                    f"mature={np.sum(ranks >= 2)}"
                )

        return {"train_loss": avg_loss, "round_time": round_time}

    def _merge_client_neuron_ages(self, results: List[Dict]) -> Dict[str, np.ndarray]:
        """Merge per-client NICE neuron ages by taking the max age per neuron."""
        age_states = [r.get("neuron_ages") for r in results if r.get("neuron_ages")]
        if not age_states:
            return {}

        merged = {}
        for layer_name in self.global_model.LAYER_NAMES:
            layer_arrays = [
                np.asarray(state[layer_name], dtype=np.int32)
                for state in age_states
                if layer_name in state
            ]
            if not layer_arrays:
                continue
            merged[layer_name] = np.maximum.reduce(layer_arrays)
        return merged

    def end_task(self):
        """End-of-task processing: age transition, freeze masks, context detector.

        Called after all training rounds for a task are complete.

        1. Increase unit ranks (learner -> mature)
        2. Update freeze masks
        3. Freeze BN for mature layers
        4. Push activations to context detector
        5. Train context detector models

        Đây là bước update trí nhớ dài hạn quan trọng nhất của NICE sau mỗi task.
        """
        print(f"\n  NICE end_task({self.current_task}):")

        # Ensure training always ends with a final memory/context update before
        # age transition, as described in NICE Section 3.5.
        self.update_context_detector_memory(verbose=True)

        # 1. Age transition: all rank >= 1 get +1
        increase_unit_ranks(self.global_model)

        # Print age stats
        for name in self.global_model.LAYER_NAMES:
            ranks = self.global_model.unit_ranks[name]
            print(
                f"    {name}: young={np.sum(ranks == 0)}, "
                f"learner={np.sum(ranks == 1)}, "
                f"mature={np.sum(ranks >= 2)}"
            )

        # 2. Update freeze masks for gradient protection
        update_freeze_masks(self.global_model)

        # 3. Freeze BN for layers with all-mature neurons
        self.global_model.freeze_bn_for_mature()

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
        """Evaluate with NICE context-aware output masking.

        NICE uses LetLearner during training which blocks gradient flow to
        unseen output neurons, leaving them with random weights. During eval,
        we first mask unseen classes globally for loss stability, then use the
        trained ContextDetector to predict the episode of each sample and mask
        future-episode classes for argmax.

        Nếu bỏ bước này, lớp chưa học có thể thắng argmax chỉ vì trọng số ngẫu nhiên.
        """
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
        )

        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()

        X_test = self.test_data["X_test"]
        y_test = self.test_data["y_test"]

        # Filter test data to seen classes
        if seen_classes_only and self.seen_classes:
            seen_set = set(self.seen_classes)
            mask = torch.tensor([y.item() in seen_set for y in y_test])
            X_test = X_test[mask]
            y_test = y_test[mask]

        n_test = len(y_test)
        if n_test == 0:
            return {"loss": 0.0, "accuracy": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0}

        unseen_mask = self._build_global_unseen_mask()
        use_context_eval = bool(self.config.get("nice_context_eval", False))
        debug_context = bool(self.config.get("nice_debug_context_detector", False))

        all_preds = []
        all_targets = []
        all_true_episodes = []
        all_pred_episodes = []
        all_context_conf = []
        total_loss = 0.0

        with torch.no_grad():
            for i in range(0, n_test, batch_size):
                X_batch = X_test[i : i + batch_size].to(self.primary_device)
                y_batch = y_test[i : i + batch_size].to(self.primary_device)

                if hasattr(self.global_model, "get_output_and_context_activations"):
                    out, context_activations = (
                        self.global_model.get_output_and_context_activations(X_batch)
                    )
                else:
                    out = self.global_model(X_batch)
                    context_activations = None

                loss_out = out.clone()
                loss_out[:, unseen_mask] = float("-inf")
                loss = criterion(loss_out, y_batch)
                total_loss += loss.item() * len(y_batch)

                if debug_context:
                    try:
                        if context_activations is not None:
                            binary_acts = self.context_detector.binarize_layer_activations(
                                {
                                    name: act.detach().cpu().numpy()
                                    for name, act in context_activations.items()
                                }
                            )
                        else:
                            binary_acts = self.context_detector._binarize_per_sample(
                                self.global_model,
                                X_batch,
                            )
                        pred_episodes, chain_probs = (
                            self.context_detector.predict_episodes_with_scores(binary_acts)
                        )
                        true_episodes = self._labels_to_episodes(
                            y_batch.detach().cpu().numpy()
                        )
                        all_true_episodes.extend(true_episodes.tolist())
                        all_pred_episodes.extend(pred_episodes.tolist())
                        all_context_conf.extend(chain_probs.max(axis=1).tolist())
                    except Exception as exc:
                        print(f"  WARNING: NICE context debug failed: {exc}")

                if use_context_eval:
                    pred_out = self._apply_context_mask(
                        out,
                        X_batch,
                        context_activations=context_activations,
                    )
                else:
                    pred_out = loss_out
                preds = pred_out.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        if debug_context and all_true_episodes:
            self._print_context_debug(
                np.array(all_true_episodes),
                np.array(all_pred_episodes),
                np.array(all_context_conf, dtype=np.float32),
            )

        return {
            "loss": total_loss / n_test,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
        }

    def _build_global_unseen_mask(self) -> torch.Tensor:
        """Return a class mask that blocks classes not seen by the server yet."""
        seen_set = (
            set(self.seen_classes)
            if self.seen_classes
            else set(range(self.global_model.num_classes))
        )
        unseen_mask = torch.ones(self.global_model.num_classes, dtype=torch.bool)
        for cls_id in seen_set:
            if 0 <= int(cls_id) < self.global_model.num_classes:
                unseen_mask[int(cls_id)] = False
        return unseen_mask.to(self.primary_device)

    def _allowed_classes_for_episode(self, episode: int) -> List[int]:
        """
        Classes allowed by NICE context prediction.

        If ContextDetector predicts episode k, official NICE boosts classes
        introduced in that episode so argmax is selected from that context.
        """
        seen_set = (
            set(int(c) for c in self.seen_classes)
            if self.seen_classes
            else set(range(self.global_model.num_classes))
        )
        allowed = [
            int(c)
            for c in self.context_detector.episode_classes.get(int(episode), [])
        ]
        allowed = [
            c
            for c in allowed
            if c in seen_set and 0 <= c < self.global_model.num_classes
        ]
        if allowed:
            return sorted(set(allowed))
        return sorted(
            c for c in seen_set if 0 <= int(c) < self.global_model.num_classes
        )

    def _labels_to_episodes(self, labels: np.ndarray) -> np.ndarray:
        """Map class labels to NICE episode ids for context-detector debugging."""
        label_to_episode = {}
        for episode, classes in self.context_detector.episode_classes.items():
            for cls_id in classes:
                label_to_episode[int(cls_id)] = int(episode)
        for episode, classes in self.task_classes.items():
            for cls_id in classes:
                label_to_episode.setdefault(int(cls_id), int(episode))

        labels = np.asarray(labels)
        return np.array(
            [label_to_episode.get(int(label), -1) for label in labels],
            dtype=np.int64,
        )

    def _print_context_debug(
        self,
        true_episodes: np.ndarray,
        pred_episodes: np.ndarray,
        confidences: np.ndarray,
    ) -> None:
        """Print compact diagnostics for NICE context-detector routing."""
        if len(true_episodes) == 0:
            return

        known = true_episodes >= 0
        if known.any():
            route_acc = float(np.mean(true_episodes[known] == pred_episodes[known]))
        else:
            route_acc = 0.0

        def counts(values: np.ndarray) -> Dict[int, int]:
            unique, freq = np.unique(values, return_counts=True)
            return {int(k): int(v) for k, v in zip(unique, freq)}

        print("\n  NICE context detector debug:")
        print(f"    route_acc={route_acc * 100:.2f}%")
        print(f"    true episode counts: {counts(true_episodes)}")
        print(f"    pred episode counts: {counts(pred_episodes)}")
        if len(confidences) > 0:
            q10, q50, q90 = np.quantile(confidences, [0.1, 0.5, 0.9])
            print(
                "    confidence max(chain_probs): "
                f"q10={q10:.4f}, q50={q50:.4f}, q90={q90:.4f}"
            )

        for episode in sorted(counts(true_episodes).keys()):
            if episode < 0:
                continue
            row = pred_episodes[true_episodes == episode]
            print(f"    true {episode} -> pred {counts(row)}")

    def _apply_context_mask(
        self,
        logits: torch.Tensor,
        X_batch: torch.Tensor,
        context_activations: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Apply official NICE inference-time context correction.

        Official test flow forwards once, predicts the context episode from
        activations, then adds a very large bias to classes of that episode.
        """
        masked = logits.clone()
        if not self.context_detector.episode_classes:
            global_unseen = self._build_global_unseen_mask()
            masked[:, global_unseen] = float("-inf")
            return masked

        try:
            if context_activations is not None:
                binary_acts = self.context_detector.binarize_layer_activations(
                    {
                        name: act.detach().cpu().numpy()
                        for name, act in context_activations.items()
                    }
                )
            else:
                binary_acts = self.context_detector._binarize_per_sample(
                    self.global_model,
                    X_batch,
                )
            pred_episodes = self.context_detector.predict_episodes_batch(binary_acts)
        except Exception as exc:
            print(f"  WARNING: NICE context detector failed during eval: {exc}")
            global_unseen = self._build_global_unseen_mask()
            masked[:, global_unseen] = float("-inf")
            return masked

        for row_idx, episode in enumerate(pred_episodes):
            allowed = self._allowed_classes_for_episode(int(episode))
            if allowed:
                masked[row_idx, allowed] = masked[row_idx, allowed] + 99999.0

        return masked

    def compute_average_forgetting(self) -> float:
        """Tính Average Forgetting của NICE dựa trên accuracy từng task."""
        if self.current_task == 0:
            return 0.0
        current_accs = self.evaluate_per_task()
        if hasattr(self.trainer, "update_forgetting"):
            self.trainer.update_forgetting(current_accs)
            return self.trainer.last_af
        return 0.0

    def evaluate_per_task(self, batch_size: int = 1024) -> Dict[int, float]:
        """Đánh giá accuracy riêng cho từng task, vẫn áp output masking của NICE."""
        from sklearn.metrics import accuracy_score

        self.global_model.eval()
        task_accuracies = {}
        use_context_eval = bool(self.config.get("nice_context_eval", False))
        unseen_mask = self._build_global_unseen_mask()

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
                    X_batch = X_task[i : i + batch_size].to(self.primary_device)
                    y_batch = y_task[i : i + batch_size]

                    if hasattr(self.global_model, "get_output_and_context_activations"):
                        out, context_activations = (
                            self.global_model.get_output_and_context_activations(X_batch)
                        )
                    else:
                        out = self.global_model(X_batch)
                        context_activations = None
                    if use_context_eval:
                        pred_out = self._apply_context_mask(
                            out,
                            X_batch,
                            context_activations=context_activations,
                        )
                    else:
                        pred_out = out.clone()
                        pred_out[:, unseen_mask] = float("-inf")
                    preds = pred_out.argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(y_batch.numpy())

            task_accuracies[task_id] = accuracy_score(all_targets, all_preds)

        return task_accuracies
