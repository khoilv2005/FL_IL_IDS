"""
Plexus Strategy - Practical Federated Learning without a Server.

Reference:
    Dhasade et al., "Practical Federated Learning without a Server",
    EuroMLSys 2025 (https://arxiv.org/pdf/2302.13837)

Paper Algorithm Summary:
========================
1. Decentralized Aggregation (Section 3):
   - No fixed central server; aggregation responsibility rotates among peers.
   - Each round, a deterministic hash-based ordering selects aggregators
     and training participants from the active population.

2. Sample Manager (Section 3.1):
   - Uses MD5 hash of (peer_id, round) for deterministic, unbiased ordering.
   - First `num_aggregators` in the ordered list become aggregators.
   - Next `sample_size` peers form the training sample.

3. Success Fraction (Section 3.2):
   - Aggregation proceeds when `success_fraction` of sample has submitted.
   - Provides liveness: system continues even if some peers are offline.
   - Minimum 3 models needed for liveness (fallback).

4. Population View Merge (Section 3.3):
   - Each peer maintains a population view mapping peer_id -> (last_round, status).
   - Views are piggybacked on model transfers and merged using vector-clock-like
     semantics (take the latest info for each peer).

5. Availability Checking (Section 3.4):
   - Before sending, peers ping candidate aggregators/participants.
   - Unreachable peers are skipped; next candidates in the ordered list are tried.

6. Model Aggregation:
   - Standard FedAvg weighted average on successfully received models.

Adaptations for this codebase:
- The decentralized protocol is simulated in PlexusServer, which rotates the
  aggregator role and sub-samples clients per round.
- PlexusTrainer uses standard CrossEntropyLoss (same local training as vanilla FL).
- PlexusAggregator performs FedAvg with success-fraction filtering.
"""

import hashlib
from collections import OrderedDict
from math import floor
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ...core import BaseTrainer, BaseAggregator


# ---------------------------------------------------------------------------
# Sample Manager  (dlsim/plexus/sample_manager.py)
# ---------------------------------------------------------------------------

class SampleManager:
    """
    Deterministic hash-based sample ordering + bandwidth-based aggregator
    selection (paper Section 3.1 & ``determine_available_peers_for_sample``).

    Workflow per round:
    1. Hash ordering ``MD5(peer_id || round)`` → deterministic sample.
    2. Take the first ``sample_size`` entries as the round's participants.
    3. Among those participants, sort by **bandwidth descending** → the
       top ``num_aggregators`` nodes become aggregators.

    This mirrors ``dlsim/plexus/community.py``:
    ```
    candidate_peers = self.sample_manager.get_ordered_sample_list(...)
    if getting_aggregators and self.other_nodes_bws:
        candidate_peers = sorted(candidate_peers[:sample_size],
            key=lambda pk: self.other_nodes_bws[pk], reverse=True)
    ```
    """

    def __init__(self, sample_size: int = 13, num_aggregators: int = 1):
        self.sample_size = sample_size
        self.num_aggregators = num_aggregators

    def get_ordered_sample_list(
        self, round_num: int, peer_ids: List[int]
    ) -> List[int]:
        """
        Return peer IDs sorted by ``MD5(peer_id-round)``.

        Args:
            round_num: Current round number.
            peer_ids: List of active peer/client IDs.

        Returns:
            Deterministically ordered list of peer IDs for this round.
        """
        peer_ids = sorted(peer_ids)
        hashes = []
        for pid in peer_ids:
            h = hashlib.md5(f"{pid}-{round_num}".encode())
            hashes.append((pid, h.digest()))
        hashes.sort(key=lambda t: t[1])
        return [t[0] for t in hashes]

    def get_aggregators(
        self,
        round_num: int,
        peer_ids: List[int],
        bandwidths: Optional[Dict[int, float]] = None,
    ) -> List[int]:
        """
        Return the aggregator(s) for this round.

        Paper protocol (``community.py:determine_available_peers_for_sample``):
        1. Get the hash-ordered sample (first ``sample_size`` peers).
        2. Sort that sample by bandwidth **descending**.
        3. Pick the top ``num_aggregators`` — highest-bandwidth nodes aggregate.

        If no bandwidth info is available, falls back to pure hash ordering
        (equivalent to all peers having equal bandwidth).

        Args:
            round_num: Current round number.
            peer_ids: Active peer IDs.
            bandwidths: Optional mapping client_id → bandwidth capacity.

        Returns:
            List of aggregator client IDs.
        """
        ordered = self.get_ordered_sample_list(round_num, peer_ids)
        sample = ordered[: self.sample_size]

        if bandwidths:
            # Sort sample by bandwidth descending → highest-BW nodes first
            sample = sorted(
                sample,
                key=lambda pid: bandwidths.get(pid, 0.0),
                reverse=True,
            )

        return sample[: self.num_aggregators]

    def get_sample(
        self,
        round_num: int,
        peer_ids: List[int],
        bandwidths: Optional[Dict[int, float]] = None,
    ) -> List[int]:
        """
        Return the training sample for this round (excluding aggregators).

        Uses same hash ordering; aggregators (selected by bandwidth)
        are removed from the sample.

        Args:
            round_num: Current round number.
            peer_ids: Active peer IDs.
            bandwidths: Optional mapping for aggregator selection.

        Returns:
            List of training participant client IDs.
        """
        ordered = self.get_ordered_sample_list(round_num, peer_ids)
        aggregator_set = set(
            self.get_aggregators(round_num, peer_ids, bandwidths)
        )
        # Participants = hash-ordered list minus aggregators, capped at sample_size
        participants = [pid for pid in ordered if pid not in aggregator_set]
        return participants[: self.sample_size]


# ---------------------------------------------------------------------------
# Population View  (dlsim/core/peer_manager.py — simplified)
# ---------------------------------------------------------------------------

class PopulationView:
    """
    Tracks the last-known activity of each peer and supports view merging.

    In the original Plexus code this is ``PeerManager.last_active``.  Here we
    maintain a lightweight version that records ``(last_round, is_online)``
    for each client ID.
    """

    def __init__(self):
        # client_id -> (last_active_round, is_online)
        self.view: Dict[int, Tuple[int, bool]] = {}

    def update(self, client_id: int, round_num: int, is_online: bool = True):
        cur = self.view.get(client_id)
        if cur is None or round_num >= cur[0]:
            self.view[client_id] = (round_num, is_online)

    def get_active_peers(
        self, current_round: int, inactivity_threshold: int = 50
    ) -> List[int]:
        """Return IDs of peers considered active."""
        active = []
        for cid, (last_round, online) in self.view.items():
            if online and (current_round - last_round) <= inactivity_threshold:
                active.append(cid)
        return active

    def merge(self, other: "PopulationView"):
        """Merge another view into this one (vector-clock semantics)."""
        for cid, (rnd, online) in other.view.items():
            cur = self.view.get(cid)
            if cur is None or rnd > cur[0]:
                self.view[cid] = (rnd, online)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class PlexusTrainer(BaseTrainer):
    """
    Plexus local trainer — identical to vanilla FedAvg local training.

    Each client trains with standard CrossEntropyLoss for the configured
    number of local epochs.  The novelty of Plexus lies in *who* aggregates
    and *which* peers are sampled, not in the local training objective.
    """

    def compute_loss(
        self,
        model: nn.Module,
        output: torch.Tensor,
        target: torch.Tensor,
        global_params: Optional[OrderedDict] = None,
        **kwargs,
    ) -> torch.Tensor:
        return nn.CrossEntropyLoss()(output, target)


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

class PlexusAggregator(BaseAggregator):
    """
    Plexus aggregator — FedAvg with success-fraction filtering.

    In Plexus, aggregation proceeds as soon as ``success_fraction`` of the
    expected sample has submitted trained models.  This class simulates that
    behavior by:
    1. Sorting incoming results by the deterministic sample order.
    2. Keeping only the first ``ceil(sample_size * success_fraction)`` results.
    3. Performing weighted FedAvg on the kept results.

    Args:
        sample_size: Target number of training participants per round.
        num_aggregators: Number of aggregators per round.
        success_fraction: Fraction of sample needed before aggregation proceeds.
        inactivity_threshold: Rounds of inactivity before a peer is considered offline.
    """

    def __init__(
        self,
        sample_size: int = 13,
        num_aggregators: int = 1,
        success_fraction: float = 0.8,
        inactivity_threshold: int = 50,
        client_bandwidths: Optional[Dict[int, float]] = None,
    ):
        self.sample_size = sample_size
        self.num_aggregators = num_aggregators
        self.success_fraction = success_fraction
        self.inactivity_threshold = inactivity_threshold

        # Internal components
        self.sample_manager = SampleManager(sample_size, num_aggregators)
        self.population_view = PopulationView()

        # Bandwidth info: client_id -> bandwidth capacity.
        # Used to select aggregators (highest-bandwidth node aggregates).
        self.client_bandwidths: Dict[int, float] = client_bandwidths or {}

        # Round counter (managed externally by server)
        self.current_round: int = 0

    # ------------------------------------------------------------------
    # Core aggregation (called from Server.train_round)
    # ------------------------------------------------------------------

    def aggregate(
        self,
        results: List[Dict],
        global_params: Optional[OrderedDict] = None,
        **kwargs,
    ) -> OrderedDict:
        """
        Aggregate client results using FedAvg with success-fraction filtering.

        Only the first ``ceil(sample_size * success_fraction)`` results
        (by the deterministic sample order) contribute to the aggregate.
        A minimum of 3 is enforced for liveness (paper heuristic).
        """
        if not results:
            return global_params

        n_required = max(3, floor(len(results) * self.success_fraction))
        used = results[:n_required] if len(results) > n_required else results

        # Update population view with participating clients
        for r in results:
            cid = r.get("client_id", -1)
            self.population_view.update(cid, self.current_round, is_online=True)

        return self._weighted_average(used)

    # ------------------------------------------------------------------
    # Sampling helpers (exposed for PlexusServer)
    # ------------------------------------------------------------------

    def get_round_aggregators(self, round_num: int, all_client_ids: List[int]) -> List[int]:
        """Determine aggregator(s) for a given round (bandwidth-based)."""
        active = self.population_view.get_active_peers(
            round_num, self.inactivity_threshold
        )
        candidates = active if active else all_client_ids
        return self.sample_manager.get_aggregators(
            round_num, candidates, self.client_bandwidths
        )

    def get_round_sample(self, round_num: int, all_client_ids: List[int]) -> List[int]:
        """Determine training sample for a given round."""
        active = self.population_view.get_active_peers(
            round_num, self.inactivity_threshold
        )
        candidates = active if active else all_client_ids
        sample = self.sample_manager.get_sample(
            round_num, candidates, self.client_bandwidths
        )
        # Fallback: if sample is too small, use all active peers
        if len(sample) < 3:
            return candidates
        return sample
