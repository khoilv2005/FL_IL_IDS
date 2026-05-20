"""
Plexus Server - Simulates the decentralized Plexus protocol.

Reference:
    Dhasade et al., "Practical Federated Learning without a Server",
    EuroMLSys 2025 (https://arxiv.org/pdf/2302.13837)

This server simulates the fully decentralized Plexus system within the
existing centralized FL infrastructure.  In each round it:

1. Selects a deterministic sub-sample of clients via hash ordering (Section 3.1).
2. Within that sample, picks the aggregator as the highest-bandwidth node.
3. Trains only the sampled clients (simulating peer availability).
4. Aggregates with success-fraction filtering (Section 3.2).
5. Distributes the aggregated model + population view to sampled clients.

Because the rest of the project expects a server object that holds a global
model, this class wraps the decentralized logic while exposing the same
``train_round()`` / ``evaluate_global()`` interface as ``IncrementalServer``.
"""

import random
import time
from collections import OrderedDict
from math import floor
from threading import Thread
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .incremental_server import IncrementalServer
from ..strategies.federated.plexus import (
    PlexusTrainer,
    PlexusAggregator,
    SampleManager,
    PopulationView,
)


class PlexusServer(IncrementalServer):
    """
    Server that simulates the Plexus decentralized FL protocol.

    Key differences from a standard ``FederatedServer``:

    * **Rotating aggregator**: Each round, one (or more) clients is designated
      as the aggregator based on hash ordering — *not* the server itself.
      In our simulation the server still performs the actual aggregation,
      but it only uses the subset of models that the designated aggregator
      would have access to per the Plexus protocol.

    * **Hash-based sampling**: Only a deterministic subset of clients
      (``sample_size``) participates each round. The subset is derived
      from ``SampleManager.get_ordered_sample_list()``.

    * **Success fraction**: Aggregation proceeds as soon as
      ``success_fraction × sample_size`` models are available (paper Sec. 3.2).
      This is simulated by truncating the list of results.

    * **Population view**: A shared ``PopulationView`` is merged into each
      client's local view after aggregation, mirroring the piggy-backed
      view exchange in the original Plexus networking code.

    Args:
        clients: List of ``PlexusClient`` instances.
        test_data: ``{"X_test": Tensor, "y_test": Tensor}``.
        config: Training configuration dict.  Plexus-specific keys:
            - ``plexus_sample_size``    (default 13)
            - ``plexus_num_aggregators`` (default 1)
            - ``plexus_success_fraction`` (default 0.8)
            - ``plexus_inactivity_threshold`` (default 50)
            - ``plexus_scale_clients`` (default False) - Enable dynamic client scaling per task
            - ``plexus_initial_client_ratio`` (default 0.5) - Initial client ratio (0.5 = 50%)
            - ``plexus_final_client_ratio`` (default 1.0) - Final client ratio (1.0 = 100%)
    """

    def __init__(self, clients, test_data: Dict, config: Dict):
        super().__init__(clients, test_data, config)

        # Override strategy with Plexus-specific instances
        self.sample_size = config.get("plexus_sample_size", 13)
        self.num_aggregators = config.get("plexus_num_aggregators", 1)
        self.success_fraction = config.get("plexus_success_fraction", 0.8)
        self.inactivity_threshold = config.get("plexus_inactivity_threshold", 50)

        # Dynamic client scaling parameters
        self.scale_clients = config.get("plexus_scale_clients", False)
        self.initial_client_ratio = config.get("plexus_initial_client_ratio", 0.5)
        self.final_client_ratio = config.get("plexus_final_client_ratio", 1.0)
        self.total_clients = len(clients)
        self.current_task_id = 0
        self._num_tasks = config.get("num_tasks", 6)  # Total tasks for dynamic scaling

        self.trainer = PlexusTrainer()

        # Simulate client bandwidths for aggregator selection.
        # In real Plexus, each node measures neighbours' upload bandwidth.
        # Here we assign each client a random bandwidth drawn from a
        # log-normal distribution (seeded for reproducibility).
        all_ids = sorted(c.client_id for c in self.clients)
        rng = random.Random(config.get("seed", 42))
        self.client_bandwidths: Dict[int, float] = {
            cid: round(rng.lognormvariate(mu=3.0, sigma=0.8), 2)
            for cid in all_ids
        }

        self.aggregator = PlexusAggregator(
            sample_size=self.sample_size,
            num_aggregators=self.num_aggregators,
            success_fraction=self.success_fraction,
            inactivity_threshold=self.inactivity_threshold,
            client_bandwidths=self.client_bandwidths,
        )

        self.sample_manager = SampleManager(self.sample_size, self.num_aggregators)
        self.population_view = PopulationView()

        # Round counter
        self._round: int = 0

        # Initialize population view with all clients
        all_ids = [c.client_id for c in self.clients]
        for cid in all_ids:
            self.population_view.update(cid, 0, is_online=True)
            self.aggregator.population_view.update(cid, 0, is_online=True)

        print(
            f"📊 Strategy: Plexus (decentralized) — "
            f"sample_size={self.sample_size}, "
            f"num_agg={self.num_aggregators}, "
            f"success_frac={self.success_fraction}"
        )
        
        # Print client scaling info
        if self.scale_clients:
            print(
                f"📈 Dynamic Client Scaling: {self.initial_client_ratio*100:.0f}% → {self.final_client_ratio*100:.0f}% "
                f"(tasks 0 → last)"
            )

    def get_sample_size_for_task(self, task_id: int, num_tasks: int) -> int:
        """
        Calculate dynamic sample size based on task progress.
        
        If scaling is enabled, sample size increases linearly from initial to final
        ratio as more clients join the network.
        
        Example with 100 clients:
        - Task 0 (50%): min(sample_size, 50) clients in sample
        - Task 3 (75%): min(sample_size, 75) clients in sample
        - Task 5 (100%): min(sample_size, 100) clients in sample
        
        Args:
            task_id: Current task ID (0-indexed)
            num_tasks: Total number of tasks
            
        Returns:
            Effective sample size for this task
        """
        if not self.scale_clients:
            return self.sample_size
        
        # Linear interpolation from initial_ratio to final_ratio
        if num_tasks <= 1:
            progress = 1.0
        else:
            progress = task_id / (num_tasks - 1)
        
        # Calculate target ratio
        target_ratio = self.initial_client_ratio + progress * (
            self.final_client_ratio - self.initial_client_ratio
        )
        
        # Calculate effective sample size based on total clients
        # This represents the number of clients we'll select from the pool
        effective_candidates = int(target_ratio * self.total_clients)
        
        # Sample size is the minimum of:
        # - The configured sample_size (max participants per round)
        # - The effective number of candidates available
        effective_sample_size = max(3, min(
            self.sample_size,
            effective_candidates
        ))
        
        return effective_sample_size

    def get_participant_ratio_for_task(self, task_id: int, num_tasks: int) -> float:
        """
        Get the client participation ratio for a given task.
        
        Args:
            task_id: Current task ID
            num_tasks: Total number of tasks
            
        Returns:
            Ratio of clients participating (0.0 to 1.0)
        """
        if not self.scale_clients:
            return 1.0
        
        if num_tasks <= 1:
            return self.final_client_ratio
        
        progress = task_id / (num_tasks - 1)
        return self.initial_client_ratio + progress * (
            self.final_client_ratio - self.initial_client_ratio
        )

    # ------------------------------------------------------------------
    # train_round — the core Plexus simulation
    # ------------------------------------------------------------------

    def train_round(
        self,
        participating_clients=None,
        task_id: int = None,
        verbose: bool = True,
        **kwargs,
    ) -> Dict:
        """
        Execute one Plexus round.

        Protocol (mirrors ``PlexusCommunity.train_in_round_coroutine``):
        1. Increment round counter.
        2. Determine aggregator(s) via hash ordering.
        3. Determine training sample via hash ordering (with dynamic scaling if enabled).
        4. Train only the sampled clients (multi-GPU).
        5. Apply success-fraction filtering.
        6. Aggregate via FedAvg.
        7. Distribute aggregated model + population view.
        """
        from ..training.worker import train_clients_on_gpu

        round_start = time.time()
        self._round += 1
        self.aggregator.current_round = self._round

        # Update current task for dynamic scaling
        if task_id is not None:
            self.current_task_id = task_id

        all_ids = [c.client_id for c in self.clients]
        client_map = {c.client_id: c for c in self.clients}

        # Determine effective sample size based on task progress
        num_tasks = getattr(self, '_num_tasks', 6)  # Default 6 tasks if not set
        if self.scale_clients and task_id is not None:
            effective_sample_size = self.get_sample_size_for_task(task_id, num_tasks)
        else:
            effective_sample_size = self.sample_size

        # Create task-specific sample manager with dynamic sample size
        task_sample_manager = SampleManager(effective_sample_size, self.num_aggregators)

        # --- 1. Determine aggregator(s) (highest bandwidth in sample) ---
        # Filter to participating clients if provided
        candidate_ids = all_ids
        if participating_clients is not None:
            candidate_ids = [c.client_id for c in participating_clients]

        # Plexus Algorithm 1: sample = first K hash-ordered active peers;
        # aggregator = highest-bandwidth peer inside that same sample.
        sample_ids = task_sample_manager.get_sample(
            self._round, candidate_ids, self.client_bandwidths
        )
        aggregator_ids = task_sample_manager.get_aggregators(
            self._round, candidate_ids, self.client_bandwidths
        )

        if not sample_ids:
            if verbose:
                print(f"\n→ Plexus Round {self._round}: no sampled clients")
            return {"train_loss": 0.0, "round_time": time.time() - round_start}

        sampled_clients = [client_map[sid] for sid in sample_ids if sid in client_map]
        n_required = max(1, floor(effective_sample_size * self.success_fraction))
        # Server simulation skips late clients because their updates would be
        # ignored once the Plexus success threshold is reached.
        train_clients = sampled_clients[:n_required]

        if verbose:
            device_info = "CPU" if self.use_cpu else f"{self.num_gpus} GPU(s)"
            agg_bw = {a: self.client_bandwidths.get(a, 0) for a in aggregator_ids}
            
            # Build scaling info string
            scaling_info = ""
            if self.scale_clients and task_id is not None:
                scaling_info = f", sample_size={effective_sample_size}"
            
            print(
                f"\n→ Plexus Round {self._round} [Task {task_id}]: "
                f"aggregator={aggregator_ids} (bw={agg_bw}), "
                f"sample={len(sampled_clients)}/{len(all_ids)} clients, "
                f"training={len(train_clients)} threshold clients{scaling_info}, "
                f"device={device_info}"
            )

        global_params = self.get_global_params()

        # --- 3. Train sampled clients on GPUs ---
        clients_per_gpu = [[] for _ in range(self.num_gpus)]
        for i, c in enumerate(train_clients):
            clients_per_gpu[i % self.num_gpus].append(c)

        results_dict: Dict = {}
        threads = []

        for gpu_id in range(self.num_gpus):
            if len(clients_per_gpu[gpu_id]) > 0:
                t = Thread(
                    target=train_clients_on_gpu,
                    args=(
                        gpu_id,
                        clients_per_gpu[gpu_id],
                        global_params,
                        self.config,
                        results_dict,
                        self.trainer,
                        self.use_cpu,
                    ),
                )
                threads.append(t)
                t.start()

        for t in threads:
            t.join()

        sample_order = {cid: idx for idx, cid in enumerate(sample_ids)}
        results = sorted(
            results_dict.values(),
            key=lambda r: sample_order.get(r.get("client_id", -1), len(sample_order)),
        )

        # --- 4. Success-fraction filtering ---
        used_results = results[:n_required] if len(results) >= n_required else []

        if verbose:
            print(
                f"   Aggregating {len(used_results)}/{len(results)} models "
                f"(threshold={n_required}, success_fraction={self.success_fraction})"
            )

        # --- 5. Aggregate ---
        if used_results:
            new_params = self.aggregator.aggregate(used_results, global_params)
            self.set_global_params(new_params)

        # --- 6. Update population view & distribute ---
        for r in results:
            cid = r.get("client_id", -1)
            self.population_view.update(cid, self._round, is_online=True)
            self.aggregator.population_view.update(cid, self._round, is_online=True)

        # Distribute population view to sampled clients
        for c in sampled_clients:
            if hasattr(c, "receive_aggregated_model"):
                c.receive_aggregated_model(self._round)
            if hasattr(c, "merge_population_view"):
                c.merge_population_view(self.population_view)

        avg_loss = float(np.mean([r["loss"] for r in results])) if results else 0.0
        round_time = time.time() - round_start

        if verbose:
            print(f"→ Train loss: {avg_loss:.4f}")
            print(f"→ Round time: {round_time:.2f}s")

        return {"train_loss": avg_loss, "round_time": round_time}
