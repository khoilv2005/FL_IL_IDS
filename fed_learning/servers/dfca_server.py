"""
DFCA Server - Simulates the fully decentralized DFCA protocol.

Key design decisions:
- This server does NOT aggregate a single global model (DFCA is decentralized).
- Instead, it orchestrates the round: client assignment -> training ->
  message exchange -> peer aggregation -> representative model for eval.
- For evaluation, we use a "best-loss selection" policy: each test batch
  is evaluated on all k cluster models, picking the one with lowest loss.
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
from threading import Thread
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
        num_active_clients: Number of active clients this task
        task_client_ratios: List of ratios per task (50%-60%-...-100%)
        round_counter: Round number
        cluster_history: List of per-round cluster distribution logs
        dfca_aggregator: DFCAAggregator (sequential running average logic)
        trainer: DFCATrainer instance

    Evaluation Policy:
        "best-loss selection on test batch" — for each test batch, evaluate
        on all k cluster models (using representative params averaged across
        active clients) and pick the lowest-loss one for prediction.
        This mirrors the paper's evaluation where each client uses the best
        cluster model for its data distribution.
    """

    def __init__(self, clients, test_data: Dict, config: Dict):
        super().__init__(clients, test_data, config)

        # DFCA-specific config
        self.num_clusters = config.get("dfca_num_clusters", 10)
        self.init_type = config.get("dfca_init", "global")
        self.graph_type = config.get("dfca_graph", "erdos_renyi")
        self.connectivity = config.get("dfca_connectivity", 0.15)
        self.client_ratios = config.get(
            "dfca_client_ratios",
            [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        self.round_participation = config.get("dfca_round_participation", 1.0)
        self.aggregation = config.get("dfca_aggregation", "sequential_running_average")

        # Override trainer/aggregator with DFCA-specific ones
        self.trainer = DFCATrainer(
            local_epochs=config.get("local_epochs", 1),
            learning_rate=config.get("learning_rate", 0.001),
            batch_size=config.get("batch_size", 2048),
        )
        self.aggregator = DFCAAggregator(num_clusters=self.num_clusters)

        # Round counter
        self._round: int = 0
        self.round_counter: int = 0

        # Graph topology
        self.graph: Dict[int, List[int]] = {}
        self._build_graph()

        # Cluster history for logging
        self.cluster_history: List[Dict[str, Any]] = []

        # Track assignment changes across rounds
        self._prev_assignments: Dict[int, int] = {}

        # Initialize cluster banks for all clients
        self._initialize_all_client_cluster_banks()

        # For evaluation: maintain representative cluster models
        # (average of all active clients' cluster params)
        self.representative_cluster_params: Dict[int, OrderedDict] = {}

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
        Build communication graph between all clients.

        Uses Erdos-Renyi random graph: each edge (i, j) is included
        with probability p = connectivity.

        Graph is symmetric (undirected): if j is in N_i, then i is in N_j.
        """
        seed = self.config.get("seed", 42)
        rng = random.Random(seed)

        all_ids = [c.client_id for c in self.clients]
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

        # Ensure at least 1 neighbor per node (add closest if isolated)
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

        DFCA-GI: all clients start with the same global params for all k clusters.
        New clients (joining later tasks) are initialized by copying from neighbors
        or using the global model params.
        """
        global_params = self.get_global_params()

        for client in self.clients:
            if hasattr(client, "initialize_cluster_bank"):
                if self.init_type == "global":
                    client.initialize_cluster_bank(global_params=global_params)
                else:
                    client.initialize_cluster_bank(template_model=self.global_model)

    def _get_active_clients_for_task(self, task_id: int) -> List:
        """
        Get the active clients for a specific task.

        Active clients per task: 50%, 60%, 70%, 80%, 90%, 100%
        Uses deterministic nested prefix selection: shuffle all client IDs
        and take the first N clients for each ratio.
        """
        seed = self.config.get("seed", 42)
        task_seed = seed + task_id
        rng = random.Random(task_seed)

        all_ids = sorted([c.client_id for c in self.clients])
        shuffled = all_ids.copy()
        rng.shuffle(shuffled)

        ratio = self.client_ratios[task_id] if task_id < len(self.client_ratios) else 1.0
        num_active = max(1, int(len(all_ids) * ratio))

        active_ids = set(shuffled[:num_active])
        return [c for c in self.clients if c.client_id in active_ids]

    def _get_participating_clients(
        self,
        participating_clients: Optional[List],
        task_id: int
    ) -> List:
        """Resolve the final list of active clients for this round."""
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
        2. Each client assigns itself to a cluster (min local loss)
        3. Each client trains its assigned cluster model
        4. Build and exchange messages between neighbors
        5. Each client aggregates received messages per cluster
        6. Build representative cluster models for evaluation
        """
        round_start = time.time()
        self._round += 1
        self.round_counter = self._round

        active_clients = self._get_participating_clients(participating_clients, task_id)
        client_map = {c.client_id: c for c in self.clients}

        if verbose:
            print(
                f"\n-> DFCA Round {self._round} [Task {task_id}]: "
                f"{len(active_clients)}/{len(self.clients)} active clients"
            )

        global_params = self.get_global_params()

        # ---- PHASE 1: Cluster Assignment ----
        if verbose:
            print("  [1/4] Cluster Assignment...")
        assignment_results = self._run_cluster_assignment(active_clients, verbose=verbose)

        # ---- PHASE 2: Local Training ----
        if verbose:
            print("  [2/4] Local Training...")
        train_results = self._run_local_training(active_clients, verbose=verbose)

        # ---- PHASE 3: Message Exchange & Aggregation ----
        if verbose:
            print("  [3/4] Message Exchange & Aggregation...")
        aggregation_stats = self._run_decentralized_aggregation(
            active_clients, client_map, verbose=verbose
        )

        # ---- PHASE 4: Build Representative Models ----
        self._update_representative_cluster_models(active_clients)

        # ---- Log round statistics ----
        round_stats = self._log_round_statistics(
            active_clients, assignment_results, train_results,
            aggregation_stats, verbose=verbose
        )

        round_time = time.time() - round_start
        avg_loss = float(np.mean([r["loss"] for r in train_results]))

        if verbose:
            print(f"  -> Avg train loss: {avg_loss:.4f}")
            print(f"  -> Round time: {round_time:.2f}s")

        return {
            "train_loss": avg_loss,
            "round_time": round_time,
            "round_stats": round_stats,
        }

    def _run_cluster_assignment(
        self,
        active_clients: List,
        verbose: bool = False
    ) -> Dict[int, int]:
        """Run cluster assignment for all active clients."""
        results = {}

        def assign_on_gpu(gpu_id, gpu_clients, results_dict):
            device = "cpu" if self.use_cpu else f"cuda:{gpu_id}"
            for client in gpu_clients:
                if hasattr(client, "model") and client.model is not None:
                    client.model.to(device)
                    client.device = device
                else:
                    client.setup_for_gpu(self.global_model, device)
                try:
                    cluster_id = client.assign_cluster(verbose=verbose)
                    results_dict[client.client_id] = cluster_id
                except Exception as e:
                    if verbose:
                        print(f"    Client {client.client_id}: assign_cluster error: {e}")
                    results_dict[client.client_id] = 0

        threads = []
        results_dict = {}
        for gpu_id in range(self.num_gpus):
            gpu_clients = [c for i, c in enumerate(active_clients)
                           if i % max(1, self.num_gpus) == gpu_id]
            if gpu_clients:
                t = Thread(target=assign_on_gpu, args=(gpu_id, gpu_clients, results_dict))
                threads.append(t)
                t.start()

        for t in threads:
            t.join()

        for client in active_clients:
            results[client.client_id] = results_dict.get(client.client_id, client.assigned_cluster)

        return results

    def _run_local_training(
        self,
        active_clients: List,
        verbose: bool = False
    ) -> List[Dict]:
        """Run local training for all active clients on their assigned clusters."""
        global_params = self.get_global_params()
        results_dict = {}

        def train_on_gpu(gpu_id, gpu_clients, results_d):
            device = "cpu" if self.use_cpu else f"cuda:{gpu_id}"
            use_amp = not self.use_cpu
            scaler = GradScaler(enabled=use_amp) if use_amp else None
            trainer = self.trainer
            lr = self.config.get("learning_rate", 0.001)
            epochs = self.config.get("local_epochs", 1)
            batch_size = self.config.get("batch_size", 2048)

            for client in gpu_clients:
                if not hasattr(client, "cluster_params") or not client.cluster_params:
                    if hasattr(client, "initialize_cluster_bank"):
                        client.initialize_cluster_bank(global_params=global_params)
                    else:
                        continue

                client.setup_for_gpu(self.global_model, device)
                client.model.train()

                optimizer_cls = trainer.get_optimizer_class()
                optimizer = optimizer_cls(client.model.parameters(), lr=lr)
                cluster_id = client.assigned_cluster

                total_loss = 0.0
                total_samples = 0

                for ep in range(epochs):
                    for X_batch, y_batch in client._create_batches(batch_size):
                        optimizer.zero_grad()

                        with torch_autocast(device_type="cuda", dtype=torch.float16) if use_amp else contextlib.nullcontext():
                            client.model.load_state_dict(
                                {k: v.to(device) for k, v in client.cluster_params[cluster_id].items()}
                            )
                            out = client.model(X_batch)
                            loss = trainer.compute_loss(client.model, out, y_batch, None)

                        if use_amp:
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(client.model.parameters(), max_norm=1.0)
                            trainer.pre_step(client.model, None)
                            scaler.step(optimizer)
                            scaler.update()
                            trainer.post_step(client.model, None)
                        else:
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(client.model.parameters(), max_norm=1.0)
                            trainer.pre_step(client.model, None)
                            optimizer.step()
                            trainer.post_step(client.model, None)

                        bs = len(y_batch)
                        total_loss += loss.item() * bs
                        total_samples += bs

                # Update cluster params with trained params
                client.cluster_params[cluster_id] = OrderedDict(
                    (k, v.cpu().clone()) for k, v in client.model.state_dict().items()
                )

                results_d[client.client_id] = {
                    "client_id": client.client_id,
                    "assigned_cluster": cluster_id,
                    "loss": total_loss / max(1, total_samples),
                    "params": client.cluster_params[cluster_id],
                }

        threads = []
        for gpu_id in range(self.num_gpus):
            gpu_clients = [c for i, c in enumerate(active_clients)
                           if i % max(1, self.num_gpus) == gpu_id]
            if gpu_clients:
                t = Thread(target=train_on_gpu, args=(gpu_id, gpu_clients, results_dict))
                threads.append(t)
                t.start()

        for t in threads:
            t.join()

        return list(results_dict.values())

    def _run_decentralized_aggregation(
        self,
        active_clients: List,
        client_map: Dict[int, Any],
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Execute decentralized peer-to-peer aggregation.

        Each client sends its trained params to all neighbors.
        Each client receives messages and aggregates per cluster using
        sequential running average.

        Communication:
            - Client i sends (client_id, round, assigned_cluster, theta_i,c(i))
              to ALL neighbors N_i
            - Client i receives messages from neighbors
            - Client i groups received params by cluster:
              N_i,j = {m in N_i | c(m) = j}
            - For each cluster j, client i applies running average:
              theta_i,j = seq_avg(theta_i,j, {theta_m,j | m in N_i,j})
        """
        num_messages = 0
        per_cluster_updates: Dict[int, int] = {c: 0 for c in range(self.num_clusters)}
        changes_count = 0

        # Step A: Each client exports its message
        messages: Dict[int, Dict[int, OrderedDict]] = {}
        for client in active_clients:
            msg = client.export_assigned_cluster_message()
            messages[client.client_id] = msg

        # Step B: Route messages to neighbors and aggregate
        for client in active_clients:
            neighbors = self.graph.get(client.client_id, [])
            received: Dict[int, Dict[int, OrderedDict]] = {}

            for neighbor_id in neighbors:
                if neighbor_id not in messages:
                    continue
                received[neighbor_id] = messages[neighbor_id]
                num_messages += 1

            # Pass messages to client for aggregation
            if hasattr(client, "receive_neighbor_message"):
                for sender_id, msg in received.items():
                    client.receive_neighbor_message(sender_id, msg)

            # Aggregate
            if hasattr(client, "aggregate_received_messages"):
                client.aggregate_received_messages()

            # Count per-cluster updates
            for sender_id, msg in received.items():
                for cluster_id in msg:
                    per_cluster_updates[cluster_id] = per_cluster_updates.get(cluster_id, 0) + 1

        # Count assignment changes
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

    def _update_representative_cluster_models(self, active_clients: List) -> None:
        """
        Build representative cluster models for evaluation.

        Representative model for cluster j = average of theta_i,j
        across all active clients (unweighted mean, per paper).

        This is used for evaluation only — it does NOT replace the
        decentralized peer-to-peer aggregation.
        """
        if not active_clients:
            return

        for cid in range(self.num_clusters):
            self.representative_cluster_params[cid] = OrderedDict()

        client_count = 0
        for client in active_clients:
            if not hasattr(client, "cluster_params") or not client.cluster_params:
                continue
            for cluster_id in range(self.num_clusters):
                if cluster_id not in client.cluster_params:
                    continue
                params = client.cluster_params[cluster_id]
                if not self.representative_cluster_params[cluster_id]:
                    self.representative_cluster_params[cluster_id] = OrderedDict(
                        (k, v.clone().float() / len(active_clients))
                        for k, v in params.items()
                    )
                else:
                    for k in self.representative_cluster_params[cluster_id]:
                        self.representative_cluster_params[cluster_id][k] += (
                            params[k].float() / len(active_clients)
                        )
            client_count += 1

    def _log_round_statistics(
        self,
        active_clients: List,
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
            print(
                f"  Cluster dist: [{cluster_str}], "
                f"changes={agg_stats['changes_count']}, "
                f"messages={agg_stats['num_messages']}, "
                f"avg_loss={avg_assign_loss:.4f}"
            )

        return stats

    def evaluate_global(
        self,
        batch_size: int = 1024,
        seen_classes_only: bool = True,
        **kwargs
    ) -> Dict:
        """
        Evaluate using "best-loss selection" policy.

        For each test batch, evaluate on all k representative cluster models
        and pick the one with the lowest loss. This mirrors the paper's
        evaluation where clients use the best cluster model for their data.
        """
        self.global_model.eval()

        eval_models: Dict[int, nn.Module] = {}
        for cid in range(self.num_clusters):
            if cid not in self.representative_cluster_params:
                continue
            model = CNN_GRU_Model(
                self.config["input_shape"],
                self.config.get("num_classes", self.config.get("total_classes", 34))
            )
            model.load_state_dict(
                {k: v.to(self.primary_device) for k, v in
                 self.representative_cluster_params[cid].items()}
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

        all_preds = []
        all_targets = []
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss(reduction="sum")

        with torch.no_grad():
            for i in range(0, n_test, batch_size):
                X_batch = X_test[i : i + batch_size].to(self.primary_device)
                y_batch = y_test[i : i + batch_size].to(self.primary_device)

                best_cluster = None
                best_loss = float("inf")
                best_out = None

                for cid, model in eval_models.items():
                    out = model(X_batch)
                    if seen_classes_only and self.seen_classes:
                        out = self._mask_unseen_classes(out)
                    loss = criterion(out, y_batch).item()
                    if loss < best_loss:
                        best_loss = loss
                        best_cluster = cid
                        best_out = out

                total_loss += best_loss
                preds = best_out.argmax(dim=1) if best_out is not None else torch.zeros(
                    len(y_batch), dtype=torch.long, device=self.primary_device
                )

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
            "loss": total_loss / n_test,
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

    def get_global_params(self) -> OrderedDict:
        """Return params of representative cluster 0 (for compatibility)."""
        if 0 in self.representative_cluster_params:
            return self.representative_cluster_params[0]
        return super().get_global_params()

    def set_global_params(self, params: OrderedDict) -> None:
        """
        Set global params (for compatibility).

        In DFCA, this is not used for training. Only for compatibility
        with the server interface and checkpoint loading.
        """
        pass
