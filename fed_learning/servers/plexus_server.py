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
    """

    def __init__(self, clients, test_data: Dict, config: Dict):
        super().__init__(clients, test_data, config)

        # Override strategy with Plexus-specific instances
        self.sample_size = config.get("plexus_sample_size", 13)
        self.num_aggregators = config.get("plexus_num_aggregators", 1)
        self.success_fraction = config.get("plexus_success_fraction", 0.8)
        self.inactivity_threshold = config.get("plexus_inactivity_threshold", 50)

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

    # ------------------------------------------------------------------
    # train_round — the core Plexus simulation
    # ------------------------------------------------------------------

    def train_round(
        self,
        participating_clients=None,
        verbose: bool = True,
        **kwargs,
    ) -> Dict:
        """
        Execute one Plexus round.

        Protocol (mirrors ``PlexusCommunity.train_in_round_coroutine``):
        1. Increment round counter.
        2. Determine aggregator(s) via hash ordering.
        3. Determine training sample via hash ordering.
        4. Train only the sampled clients (multi-GPU).
        5. Apply success-fraction filtering.
        6. Aggregate via FedAvg.
        7. Distribute aggregated model + population view.
        """
        from ..training.plexus_worker import train_plexus_clients_on_gpu

        round_start = time.time()
        self._round += 1
        self.aggregator.current_round = self._round

        all_ids = [c.client_id for c in self.clients]
        client_map = {c.client_id: c for c in self.clients}

        # --- 1. Determine aggregator(s) (highest bandwidth in sample) ---
        aggregator_ids = self.sample_manager.get_aggregators(
            self._round, all_ids, self.client_bandwidths
        )

        # --- 2. Determine training sample ---
        sample_ids = self.sample_manager.get_sample(
            self._round, all_ids, self.client_bandwidths
        )
        # Ensure we have at least 3 peers (liveness)
        if len(sample_ids) < 3:
            sample_ids = all_ids

        # Filter: use participating_clients if provided (e.g., for incremental tasks)
        if participating_clients is not None:
            valid_ids = {c.client_id for c in participating_clients}
            sample_ids = [sid for sid in sample_ids if sid in valid_ids]
            # If filtered sample is too small, use all participating clients
            if len(sample_ids) < 3:
                sample_ids = [c.client_id for c in participating_clients]

        sampled_clients = [client_map[sid] for sid in sample_ids if sid in client_map]

        if verbose:
            device_info = "CPU" if self.use_cpu else f"{self.num_gpus} GPU(s)"
            agg_bw = {a: self.client_bandwidths.get(a, 0) for a in aggregator_ids}
            print(
                f"\n→ Plexus Round {self._round}: "
                f"aggregator={aggregator_ids} (bw={agg_bw}), "
                f"sample={len(sampled_clients)}/{len(all_ids)} clients, "
                f"device={device_info}"
            )

        global_params = self.get_global_params()

        # --- 3. Train sampled clients on GPUs ---
        clients_per_gpu = [[] for _ in range(self.num_gpus)]
        for i, c in enumerate(sampled_clients):
            clients_per_gpu[i % self.num_gpus].append(c)

        results_dict: Dict = {}
        threads = []

        for gpu_id in range(self.num_gpus):
            if len(clients_per_gpu[gpu_id]) > 0:
                t = Thread(
                    target=train_plexus_clients_on_gpu,
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

        results = list(results_dict.values())

        # --- 4. Success-fraction filtering ---
        n_required = max(3, floor(len(results) * self.success_fraction))
        if len(results) > n_required:
            used_results = results[:n_required]
        else:
            used_results = results

        if verbose:
            print(
                f"   Aggregating {len(used_results)}/{len(results)} models "
                f"(success_fraction={self.success_fraction})"
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
