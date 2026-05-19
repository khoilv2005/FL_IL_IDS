"""
DFCA Server - Simulates the fully decentralized DFCA protocol.

Key design decisions:
- This server does NOT aggregate a single global model (DFCA is decentralized).
- Instead, it orchestrates the round: client assignment -> training ->
  message exchange -> peer aggregation -> representative model for eval.
- The cluster assignment (c(i)) is LOCAL to each client — no server-side selection.

DFCA Round Flow:
    1. Determine active clients (subset participating in this round)
    2. For each active client:
        a. assign_cluster()  — pick cluster with min loss on local data
    3. For each active client:
        b. train_assigned_cluster()  — train only assigned cluster
    4. Build messages: each client -> all neighbors (assigned cluster params)
    5. For each active client:
        c. receive_neighbor_message() + aggregate_received_messages()
    6. Build representative model for evaluation (mean of all active cluster banks)
    7. Evaluate on test set
"""

import contextlib
import random
import time
from collections import Counter, OrderedDict
from threading import Thread, Lock
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

try:
    from torch.amp import autocast as torch_autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast as torch_autocast, GradScaler

from .incremental_server import IncrementalServer
from ..models.cnn_gru import CNN_GRU_Model
from ..strategies.federated.dfca import DFCATrainer, DFCAAggregator


class DFCAServer(IncrementalServer):
    """
    DFCA server that orchestrates the decentralized clustering protocol.

    Unlike standard servers, this one does NOT aggregate a global model.
    Instead it coordinates peer-to-peer communication and maintains a
    representative model for evaluation only.

    State:
        num_clusters: Number of clusters k (fixed at 10 for this experiment)
        graph: Dict[client_id -> List[neighbor_ids]] — communication topology
        _client_order: Fixed deterministic order for nested active client selection
        round_counter: Round number
        cluster_history: List of per-round cluster distribution logs
        trainer: DFCATrainer instance
        dfca_aggregator: DFCAAggregator (sequential running average logic)

    Evaluation Policy:
        "Confidence-based selection" — for each test sample/batch, evaluate
        on all k cluster models (using representative params averaged across
        active clients) and pick the cluster with the highest softmax confidence.
        This avoids label leakage (no ground-truth labels used for cluster selection).
        A separate diagnostic metric reports oracle accuracy (loss-based selection)
        for comparison purposes only.
    """

    def __init__(self, clients, test_data: Dict, config: Dict):
        super().__init__(clients, test_data, config)

        self.num_clusters = config.get("dfca_num_clusters", 10)
        self.init_type = config.get("dfca_init", "global")
        self.graph_type = config.get("dfca_graph", "erdos_renyi")
        self.connectivity = config.get("dfca_connectivity", 0.15)
        self.client_ratios = config.get(
            "dfca_client_ratios",
            [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        self.aggregation = config.get("dfca_aggregation", "sequential_running_average")
        self.dfca_debug_messages = config.get("dfca_debug_messages", False)
        self.dfca_debug_message_limit = config.get("dfca_debug_message_limit", 50)

        self.trainer = DFCATrainer(
            local_epochs=config.get("local_epochs", 1),
            learning_rate=config.get("learning_rate", 0.001),
            batch_size=config.get("batch_size", 2048),
        )
        self.aggregator = DFCAAggregator(num_clusters=self.num_clusters)

        self._round: int = 0
        self.round_counter: int = 0

        # Bug 2 fix: fixed deterministic client order for nested active selection
        seed = self.config.get("seed", 42)
        self._client_order: List[int] = sorted([c.client_id for c in self.clients])
        rng = random.Random(seed)
        rng.shuffle(self._client_order)

        # Graph topology (built on full client population)
        self.graph: Dict[int, List[int]] = {}
        self._build_graph()

        self.cluster_history: List[Dict[str, Any]] = []
        self._prev_assignments: Dict[int, int] = {}

        # P0 fix: MUST initialize representative_cluster_params BEFORE calling
        # _initialize_all_client_cluster_banks(), because the latter calls
        # get_global_params() which the override checks self.representative_cluster_params.
        self.representative_cluster_params: Dict[int, OrderedDict] = {}

        # Initialize cluster banks for all clients
        self._initialize_all_client_cluster_banks()

        print(
            f"DFCA Strategy: k={self.num_clusters}, init={self.init_type}, "
            f"graph={self.graph_type}(p={self.connectivity}), "
            f"agg={self.aggregation}"
        )
        print(
            f"Client ratios: "
            + ", ".join(f"T{i}={r:.0%}" for i, r in enumerate(self.client_ratios))
        )

    def _build_graph(self) -> None:
        """
        Build communication graph on the full client population.

        Uses Erdos-Renyi random graph: each edge (i, j) is included
        with probability p = connectivity. Graph is symmetric (undirected).
        """
        seed = self.config.get("seed", 42)
        rng = random.Random(seed)

        all_ids = self._client_order
        n = len(all_ids)
        self.graph = {cid: [] for cid in all_ids}

        if n <= 1:
            return

        for i in range(n):
            for j in range(i + 1, n):
                cid_i = all_ids[i]
                cid_j = all_ids[j]
                if rng.random() < self.connectivity:
                    self.graph[cid_i].append(cid_j)
                    self.graph[cid_j].append(cid_i)

        for cid in all_ids:
            if not self.graph[cid]:
                other = all_ids[(all_ids.index(cid) + 1) % n]
                self.graph[cid].append(other)
                self.graph[other].append(cid)

        print(
            f"  Graph: {n} clients, "
            f"avg_degree={np.mean([len(nbrs) for nbrs in self.graph.values()]):.1f}, "
            f"min_degree={min(len(nbrs) for nbrs in self.graph.values())}"
        )

    def _initialize_all_client_cluster_banks(self) -> None:
        """
        Initialize cluster banks for all clients.

        Uses super().get_global_params() to avoid depending on
        self.representative_cluster_params during init.
        """
        global_params = super().get_global_params()
        for client in self.clients:
            if hasattr(client, "initialize_cluster_bank"):
                # Skip if already initialized (preserve state across tasks)
                if not getattr(client, "_initialized", False):
                    if self.init_type == "global":
                        client.initialize_cluster_bank(global_params=global_params)
                    else:
                        client.initialize_cluster_bank(template_model=self.global_model)

    # Bug 2 fix: nested prefix selection using fixed client order
    def _get_active_clients_for_task(self, task_id: int) -> List:
        """
        Get the active clients for a specific task.

        Uses FIXED deterministic order (`self._client_order`) and takes
        a nested prefix. Task t's active set is always a superset of task t-1's.
        """
        ratio = self.client_ratios[task_id] if task_id < len(self.client_ratios) else 1.0
        n_total = len(self._client_order)
        num_active = max(1, int(n_total * ratio))

        active_ids = set(self._client_order[:num_active])
        return [c for c in self.clients if c.client_id in active_ids]

    def _get_participating_clients(
        self,
        participating_clients: Optional[List],
        task_id: int
    ) -> List:
        if participating_clients is not None:
            return participating_clients
        return self._get_active_clients_for_task(task_id)

    def train_round(
        self,
        participating_clients: Optional[List] = None,
        task_id: int = 0,
        verbose: bool = True,
        **kwargs
    ) -> Dict:
        """
        Execute one DFCA round.

        DFCA Round:
        1. Select active clients (deterministic nested prefix)
        2. Filter to training_clients (active clients with num_samples > 0)
        3. Each training client assigns itself to a cluster (min local loss)
        4. Each training client trains its assigned cluster model
        5. Build and exchange messages between neighbors (training clients only send)
        6. Each active client aggregates received messages
        7. Build representative cluster models for evaluation
        """
        round_start = time.time()
        self._round += 1
        self.round_counter = self._round

        active_clients = self._get_participating_clients(participating_clients, task_id)

        # P1 fix: distinguish active_clients from training_clients.
        # Only clients with num_samples > 0 participate in assignment/training/message-sending.
        # All active clients still participate in receive/aggregate.
        training_clients = [
            c for c in active_clients
            if getattr(c, "num_samples", 0) > 0
        ]
        skipped_no_data = len(active_clients) - len(training_clients)

        client_map = {c.client_id: c for c in self.clients}

        if verbose:
            print(
                f"\n-> DFCA Round {self._round} [Task {task_id}]: "
                f"{len(active_clients)}/{len(self.clients)} active clients "
                f"({len(training_clients)} training"
                + (f", {skipped_no_data} no-data)" if skipped_no_data else ")")
            )

        # ---- PHASE 1: Cluster Assignment ----
        if verbose:
            print("  [1/4] Cluster Assignment...")
        assignment_results = self._run_cluster_assignment(training_clients, verbose=verbose)

        # ---- PHASE 2: Local Training ----
        if verbose:
            print("  [2/4] Local Training...")
        train_results = self._run_local_training(training_clients, verbose=verbose)

        # ---- PHASE 3: Message Exchange & Aggregation ----
        if verbose:
            print("  [3/4] Message Exchange & Aggregation...")
        aggregation_stats = self._run_decentralized_aggregation(
            active_clients, client_map, training_clients, verbose=verbose
        )

        # ---- PHASE 4: Build Representative Models ----
        self._update_representative_cluster_models(active_clients)

        round_stats = self._log_round_statistics(
            active_clients, training_clients, skipped_no_data,
            assignment_results, train_results,
            aggregation_stats, verbose=verbose
        )

        round_time = time.time() - round_start

        # P1 fix: avg_loss only over training clients (no-data clients have no loss)
        if train_results:
            avg_loss = float(np.mean([r["loss"] for r in train_results if "loss" in r]))
        else:
            avg_loss = 0.0

        if verbose:
            print(f"  -> Avg train loss: {avg_loss:.4f}")
            print(f"  -> Round time: {round_time:.2f}s")

        return {
            "train_loss": avg_loss,
            "round_time": round_time,
            "round_stats": round_stats,
        }

    def _ensure_client_model_on_device(self, client, device: str) -> None:
        """
        Ensure client has its own model instance on the correct device.

        Multi-GPU fix: each client must have a PRIVATE nn.Module instance.
        Sharing a single model object (e.g. self.global_model) across threads/GPU
        causes cuDNN flatten_weight errors, device mismatch, and inplace version
        conflicts when threads concurrently call .to(device) or forward/backward.

        This helper:
        - Detects shared model usage (client.model is self.global_model)
        - Detects wrong device
        - Creates a fresh per-client CNN_GRU_Model instance and loads
          the client's cluster_params or the server's representative params
        - Does NOT mutate self.global_model
        """
        global_model = self.global_model
        model = getattr(client, "model", None)
        client_device = getattr(client, "device", None)

        # Check if model is shared (same object identity as global_model)
        is_shared = (model is global_model) if (model is not None and global_model is not None) else False

        # Check if model is on wrong device
        wrong_device = False
        if model is not None and device.startswith("cuda"):
            try:
                first_param = next(model.parameters(), None)
                if first_param is not None and first_param.device.type != "cuda":
                    wrong_device = True
            except (StopIteration, ValueError):
                wrong_device = True
        elif model is not None and device == "cpu":
            try:
                first_param = next(model.parameters(), None)
                if first_param is not None and first_param.device.type == "cuda":
                    wrong_device = True
            except (StopIteration, ValueError):
                wrong_device = True

        needs_new_model = (model is None) or is_shared or wrong_device

        if needs_new_model:
            num_classes = self.config.get(
                "num_classes", self.config.get("total_classes", 34)
            )
            input_shape = self.config["input_shape"]
            new_model = CNN_GRU_Model(input_shape, num_classes)

            # Load cluster bank params if available (DFCA), otherwise representative params
            if hasattr(client, "cluster_params") and client.cluster_params:
                # Load the cluster params the client is currently using
                assigned = getattr(client, "assigned_cluster", 0)
                if assigned in client.cluster_params:
                    cpu_params = OrderedDict(
                        (k, v.clone().cpu()) for k, v in client.cluster_params[assigned].items()
                    )
                    new_model.load_state_dict(cpu_params)
            elif 0 in self.representative_cluster_params and self.representative_cluster_params[0]:
                # Fallback to representative params
                cpu_params = OrderedDict(
                    (k, v.clone().cpu()) for k, v in self.representative_cluster_params[0].items()
                )
                new_model.load_state_dict(cpu_params)
            elif global_model is not None:
                # Last resort: clone global_model state (NOT the object itself)
                cpu_params = OrderedDict(
                    (k, v.clone().cpu()) for k, v in global_model.state_dict().items()
                )
                new_model.load_state_dict(cpu_params)

            new_model.to(device)
            client.model = new_model
            client.device = device
            client.use_amp = device.startswith("cuda")

    def _run_cluster_assignment(
        self,
        active_clients: List,
        verbose: bool = False
    ) -> Dict[int, int]:
        """Run cluster assignment for all active clients."""
        results_dict = {}
        results_lock = Lock()

        def assign_on_gpu(gpu_id, gpu_clients, results_d, lock):
            device = "cpu" if self.use_cpu else f"cuda:{gpu_id}"
            for client in gpu_clients:
                # Multi-GPU fix: use per-client model, never shared global_model
                self._ensure_client_model_on_device(client, device)
                try:
                    cluster_id = client.assign_cluster(
                        trainer=self.trainer, verbose=verbose
                    )
                    with lock:
                        results_d[client.client_id] = cluster_id
                except Exception as e:
                    if verbose:
                        print(f"    Client {client.client_id}: assign_cluster error: {e}")
                    with lock:
                        results_d[client.client_id] = getattr(client, "assigned_cluster", 0)

        threads = []
        for gpu_id in range(self.num_gpus):
            gpu_clients = [c for i, c in enumerate(active_clients)
                           if i % max(1, self.num_gpus) == gpu_id]
            if gpu_clients:
                t = Thread(target=assign_on_gpu, args=(gpu_id, gpu_clients, results_dict, results_lock))
                threads.append(t)
                t.start()

        for t in threads:
            t.join()

        results = {}
        for client in active_clients:
            results[client.client_id] = results_dict.get(client.client_id, client.assigned_cluster)

        return results

    # Bug 1 fix: delegate to DFCAClient.train_assigned_cluster() — no duplicate training logic
    def _run_local_training(
        self,
        active_clients: List,
        verbose: bool = False
    ) -> List[Dict]:
        """Run local training by delegating to DFCAClient.train_assigned_cluster()."""
        results_dict = {}
        results_lock = Lock()

        def train_on_gpu(gpu_id, gpu_clients, results_d, lock):
            device = "cpu" if self.use_cpu else f"cuda:{gpu_id}"
            lr = self.config.get("learning_rate", 0.001)
            epochs = self.config.get("local_epochs", 1)
            batch_size = self.config.get("batch_size", 2048)

            for client in gpu_clients:
                if not hasattr(client, "cluster_params") or not client.cluster_params:
                    if hasattr(client, "initialize_cluster_bank"):
                        global_params = super().get_global_params()
                        client.initialize_cluster_bank(global_params=global_params)
                    else:
                        continue

                # Multi-GPU fix: use per-client model, never shared global_model
                self._ensure_client_model_on_device(client, device)

                try:
                    result = client.train_assigned_cluster(
                        self.trainer,
                        epochs=epochs,
                        batch_size=batch_size,
                        lr=lr,
                    )
                    with lock:
                        results_d[client.client_id] = result
                except Exception as e:
                    if verbose:
                        print(f"    Client {client.client_id}: train_assigned_cluster error: {e}")
                    with lock:
                        results_d[client.client_id] = {
                            "client_id": client.client_id,
                            "assigned_cluster": getattr(client, "assigned_cluster", 0),
                            "loss": 0.0,
                            "params": {},
                        }

        threads = []
        for gpu_id in range(self.num_gpus):
            gpu_clients = [c for i, c in enumerate(active_clients)
                           if i % max(1, self.num_gpus) == gpu_id]
            if gpu_clients:
                t = Thread(target=train_on_gpu, args=(gpu_id, gpu_clients, results_dict, results_lock))
                threads.append(t)
                t.start()

        for t in threads:
            t.join()

        return list(results_dict.values())

    def _run_decentralized_aggregation(
        self,
        active_clients: List,
        client_map: Dict[int, Any],
        training_clients: List,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Execute decentralized peer-to-peer aggregation.

        P1 fix: Only training_clients export and send messages.
        All active_clients receive messages and aggregate.
        This prevents no-data clients from sending fake cluster-0 messages.
        """
        num_messages = 0
        per_cluster_updates: Dict[int, int] = {c: 0 for c in range(self.num_clusters)}
        changes_count = 0

        # P1 fix: only training_clients export and send messages
        messages: Dict[int, Dict[int, OrderedDict]] = {}
        for client in training_clients:
            msg = client.export_assigned_cluster_message()
            messages[client.client_id] = msg

        # Debug message logging (per-sender: one line per sender showing all recipients)
        logged_count = 0
        limit = self.dfca_debug_message_limit
        unlimited = (limit <= 0)

        if self.dfca_debug_messages:
            for client in training_clients:
                if not unlimited and logged_count >= limit:
                    remaining = len(training_clients) - logged_count
                    if remaining > 0:
                        print(
                            f"[DFCA][messages] hidden {remaining} additional message logs"
                        )
                    break

                sender_cid = client.client_id
                msg = messages.get(sender_cid, {})

                # Check if sender has a valid message to send
                if not msg:
                    print(
                        f"[DFCA][messages] sender=client_{sender_cid} "
                        f"skipped reason=no_assigned_cluster"
                    )
                    logged_count += 1
                    continue

                # Find which cluster this sender is sending
                cluster_ids = list(msg.keys())
                if not cluster_ids:
                    print(
                        f"[DFCA][messages] sender=client_{sender_cid} "
                        f"skipped reason=no_assigned_cluster"
                    )
                    logged_count += 1
                    continue

                sender_cluster = cluster_ids[0]

                # Find all active_clients that have this sender in their neighbor list
                recipients = [
                    c.client_id
                    for c in active_clients
                    if sender_cid in self.graph.get(c.client_id, [])
                ]

                if recipients:
                    recipients_str = ", ".join(f"client_{r}" for r in sorted(recipients))
                    print(
                        f"[DFCA][messages] sender=client_{sender_cid} "
                        f"cluster={sender_cluster} "
                        f"recipients=[{recipients_str}] count={len(recipients)}"
                    )
                else:
                    print(
                        f"[DFCA][messages] sender=client_{sender_cid} "
                        f"cluster={sender_cluster} "
                        f"recipients=[] count=0"
                    )
                logged_count += 1

        for client in active_clients:
            neighbors = self.graph.get(client.client_id, [])
            received: Dict[int, Dict[int, OrderedDict]] = {}

            for neighbor_id in neighbors:
                if neighbor_id not in messages:
                    continue
                received[neighbor_id] = messages[neighbor_id]
                num_messages += 1

            if hasattr(client, "receive_neighbor_message"):
                for sender_id, msg in received.items():
                    client.receive_neighbor_message(sender_id, msg)

            if hasattr(client, "aggregate_received_messages"):
                client.aggregate_received_messages()

            for sender_id, msg in received.items():
                for cluster_id in msg:
                    per_cluster_updates[cluster_id] = per_cluster_updates.get(cluster_id, 0) + 1

        for client in active_clients:
            prev = self._prev_assignments.get(client.client_id, -1)
            curr = client.assigned_cluster
            if prev != -1 and prev != curr:
                changes_count += 1
            self._prev_assignments[client.client_id] = curr

        return {
            "num_messages": num_messages,
            "per_cluster_updates": per_cluster_updates,
            "changes_count": changes_count,
        }

    # Bug 6 fix: divide by actual contributor count per cluster
    def _update_representative_cluster_models(self, active_clients: List) -> None:
        """
        Build representative cluster models for evaluation.

        Representative model for cluster j = average of theta_i,j
        across active clients that actually have cluster j (unweighted mean).
        Bug 6 fix: divide by actual contributor count per cluster,
        not by total active clients (some clients may not have all clusters).
        """
        if not active_clients:
            return

        # Count contributors per cluster
        contributor_count: Dict[int, int] = {c: 0 for c in range(self.num_clusters)}
        for cid in range(self.num_clusters):
            self.representative_cluster_params[cid] = OrderedDict()

        for client in active_clients:
            if not hasattr(client, "cluster_params") or not client.cluster_params:
                continue
            for cluster_id in range(self.num_clusters):
                if cluster_id not in client.cluster_params:
                    continue
                params = client.cluster_params[cluster_id]
                contributor_count[cluster_id] += 1

                if not self.representative_cluster_params[cluster_id]:
                    self.representative_cluster_params[cluster_id] = OrderedDict(
                        (k, v.clone().float()) for k, v in params.items()
                    )
                else:
                    for k in self.representative_cluster_params[cluster_id]:
                        self.representative_cluster_params[cluster_id][k] += params[k].float()

        # Divide by actual contributor count
        for cluster_id in range(self.num_clusters):
            count = contributor_count[cluster_id]
            if count == 0:
                self.representative_cluster_params[cluster_id] = OrderedDict()
                continue
            for k in self.representative_cluster_params[cluster_id]:
                if self.representative_cluster_params[cluster_id][k].dtype.is_floating_point:
                    self.representative_cluster_params[cluster_id][k] /= count
                # Non-floating tensors (e.g., BN counters) are kept as-sum, not averaged

    def _format_cluster_updates(self, per_cluster_updates: Dict[int, int]) -> str:
        """Format per-cluster updates dict into a compact, readable string."""
        if not per_cluster_updates:
            return "none"

        total = sum(per_cluster_updates.values())
        if total == 0:
            parts = []
            for cid in sorted(per_cluster_updates.keys()):
                parts.append(f"c{cid}=0")
            return ", ".join(parts)

        parts = []
        for cid in sorted(per_cluster_updates.keys()):
            n = per_cluster_updates.get(cid, 0)
            parts.append(f"c{cid}={n}")
        return ", ".join(parts)

    def _log_round_statistics(
        self,
        active_clients: List,
        training_clients: List,
        skipped_no_data: int,
        assignment_results: Dict[int, int],
        train_results: List[Dict],
        agg_stats: Dict,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Log and return round statistics."""
        cluster_counts = Counter(assignment_results.values())

        avg_assign_loss = float(np.mean([
            r["loss"] for r in train_results if "loss" in r
        ])) if train_results else 0.0

        stats = {
            "round": self._round,
            "active_clients": len(active_clients),
            "training_clients": len(training_clients),
            "skipped_no_data": skipped_no_data,
            "cluster_distribution": dict(cluster_counts),
            "assignment_changes": agg_stats["changes_count"],
            "avg_assignment_loss": avg_assign_loss,
            "num_messages": agg_stats["num_messages"],
            "per_cluster_updates": agg_stats["per_cluster_updates"],
        }
        self.cluster_history.append(stats)

        if verbose:
            cluster_str = ", ".join(
                f"c{c}={n}" for c, n in sorted(cluster_counts.items())
            )
            cluster_updates_str = self._format_cluster_updates(agg_stats["per_cluster_updates"])
            print(
                f"  Cluster dist: [{cluster_str}], "
                f"changes={agg_stats['changes_count']}, "
                f"messages={agg_stats['num_messages']}, "
                f"avg_loss={avg_assign_loss:.4f}"
                + (f", skipped_no_data={skipped_no_data}" if skipped_no_data else "")
            )
            print(f"  [DFCA] Cluster updates: {cluster_updates_str}")

        return stats

    # P2 fix: ensemble averaging evaluation — code, docstring, and tests are now consistent
    def evaluate_global(
        self,
        batch_size: int = 1024,
        seen_classes_only: bool = True,
        **kwargs
    ) -> Dict:
        """
        Evaluate using "representative cluster ensemble averaging" policy.

        Policy: average softmax probabilities across all k representative cluster models,
        then take argmax per sample. This is an ensemble approach, NOT cluster selection.
        - No ground-truth labels are used for model/prediction selection
        - No loss is used to pick a "best" cluster
        - Loss is computed on the ensemble's averaged probability predictions

        Evaluation steps per sample:
            1. Forward pass each of k cluster models → k probability distributions
            2. Average the k probability distributions → ensemble probs
            3. argmax on ensemble probs → prediction
            4. Cross-entropy between ensemble probs and label → loss (not for selection)
        """
        self.global_model.eval()

        eval_models: Dict[int, nn.Module] = {}
        for cid in range(self.num_clusters):
            if cid not in self.representative_cluster_params:
                continue
            params = self.representative_cluster_params[cid]
            if not params:
                continue
            num_classes = self.config.get("num_classes", self.config.get("total_classes", 34))
            model = CNN_GRU_Model(
                self.config["input_shape"],
                num_classes
            )
            # Move model to device BEFORE loading params so all tensors land on the right device
            model.to(self.primary_device)
            model.load_state_dict(
                {k: v.to(self.primary_device) for k, v in params.items()}
            )
            model.eval()
            eval_models[cid] = model

        X_test = self.test_data["X_test"]
        y_test = self.test_data["y_test"]

        if seen_classes_only and self.seen_classes:
            seen_set = set(self.seen_classes)
            mask = torch.tensor([y.item() in seen_set for y in y_test])
            X_test = X_test[mask]
            y_test = y_test[mask]

        n_test = len(y_test)
        if n_test == 0:
            return {
                "loss": 0.0, "accuracy": 0.0,
                "precision_macro": 0.0, "recall_macro": 0.0,
                "f1_macro": 0.0, "f1_weighted": 0.0,
            }

        if not eval_models:
            return {
                "loss": 0.0, "accuracy": 0.0,
                "precision_macro": 0.0, "recall_macro": 0.0,
                "f1_macro": 0.0, "f1_weighted": 0.0,
            }

        all_preds = []
        all_targets = []
        total_ce_loss = 0.0

        with torch.no_grad():
            for i in range(0, n_test, batch_size):
                X_batch = X_test[i : i + batch_size].to(self.primary_device)
                y_batch = y_test[i : i + batch_size].to(self.primary_device)

                # Ensemble averaging: collect softmax probs from all cluster models
                all_probs: List[torch.Tensor] = []
                for cid, model in eval_models.items():
                    out = model(X_batch)
                    if seen_classes_only and self.seen_classes:
                        out = self._mask_unseen_classes(out)
                    probs = torch.softmax(out, dim=1)
                    all_probs.append(probs)

                # Average probabilities: [batch, num_classes]
                stacked_probs = torch.stack(all_probs, dim=0)
                avg_probs = stacked_probs.mean(dim=0)

                # Per-sample argmax from ensemble probs
                preds = avg_probs.argmax(dim=1)

                # Compute CE loss on ensemble probs (not used for selection)
                ce_loss = -torch.gather(
                    torch.log(avg_probs + 1e-8), dim=1, index=y_batch.unsqueeze(1)
                ).sum()
                total_ce_loss += ce_loss.item()

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        from sklearn.metrics import (
            accuracy_score, precision_score,
            recall_score, f1_score,
        )

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        zero_division: Any = 0

        return {
            "loss": total_ce_loss / n_test,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(
                y_true, y_pred, average="macro", zero_division=zero_division
            ),
            "recall_macro": recall_score(
                y_true, y_pred, average="macro", zero_division=zero_division
            ),
            "f1_macro": f1_score(
                y_true, y_pred, average="macro", zero_division=zero_division
            ),
            "f1_weighted": f1_score(
                y_true, y_pred, average="weighted", zero_division=zero_division
            ),
        }

    def _mask_unseen_classes(self, out: torch.Tensor) -> torch.Tensor:
        """Mask unseen class logits to -inf."""
        if not self.seen_classes:
            return out
        out = out.clone()
        num_classes = out.shape[1]
        seen_mask = torch.ones(num_classes, dtype=torch.bool, device=out.device)
        for cls_id in self.seen_classes:
            if 0 <= int(cls_id) < num_classes:
                seen_mask[int(cls_id)] = False
        out[:, seen_mask] = float("-inf")
        return out

    def set_task(
        self,
        task_id: int,
        task_classes: list,
        seen_classes: Optional[list] = None
    ) -> None:
        """Update task context and sync seen_classes to trainer."""
        super().set_task(task_id, task_classes, seen_classes)

        if hasattr(self.trainer, "set_task"):
            self.trainer.set_task(task_id, task_classes)

        # Sync seen_classes to all clients for assignment masking
        for client in self.clients:
            if hasattr(client, "set_task"):
                client.set_task(task_id, self.seen_classes, task_classes)

    def get_global_params(self) -> OrderedDict:
        """Return params of representative cluster 0 (for compatibility)."""
        if 0 in self.representative_cluster_params and self.representative_cluster_params[0]:
            return self.representative_cluster_params[0]
        return super().get_global_params()

    def set_global_params(self, params: OrderedDict) -> None:
        """Set global params (for compatibility only — not used in DFCA training."""
        pass

    def update_clients(self, clients: List) -> None:
        """
        Update client list.

        Bug 3 fix: ensures the full client population is maintained.
        Only updates the list; graph is built on full population in __init__.
        """
        self.clients = clients
