"""
PlexusSampler - Algorithm 1 (DERIVE_SAMPLE) from Plexus paper.

Reference:
    Dhasade et al., "Practical Federated Learning without a Server",
    EuroMLSys 2025, Algorithm 1

This implements the deterministic, coordination-free peer sampling
using consistent hashing. Every node can independently compute
the same sample for a given round.
"""

import hashlib
from typing import Dict, List, Optional, Tuple


class PlexusSampler:
    """
    Implements Algorithm 1: DERIVE_SAMPLE function.

    Algorithm 1:
        DERIVE_SAMPLE(Nodes, round_r, K):
            scored = [(hash(node_id || round_r), node_id) for node in Nodes]
            scored.sort()            # lexicographic sort on hash
            sample = scored[:K]      # take first K nodes
            aggregator = argmax(sample, key=bandwidth)
            return sample, aggregator

    Key properties:
    - Deterministic: same (Nodes, round_r, K) always produces same sample
    - Coordination-free: every node computes same sample independently
    - Bandwidth-based aggregator selection within sample
    """

    def __init__(
        self,
        node_ids: List[int],
        sample_size: int = 4,
        hash_algorithm: str = "md5",
    ):
        """
        Args:
            node_ids: List of all participating node IDs
            sample_size: K — number of nodes per sample
            hash_algorithm: Hash function (default md5, as in paper)
        """
        self.node_ids = sorted(set(node_ids))
        self.sample_size = min(sample_size, len(self.node_ids))
        self.hash_algorithm = hash_algorithm

    def derive_sample(
        self,
        round_num: int,
    ) -> Tuple[List[int], int]:
        """
        Algorithm 1: DERIVE_SAMPLE using internal node_ids.

        Returns:
            Tuple of (sample_node_ids, aggregator_node_id).
            Aggregator = first node in hash order (no bandwidth info here).
        """
        scored = []
        for nid in self.node_ids:
            h = hashlib.new(self.hash_algorithm)
            h.update(f"{nid}-{round_num}".encode())
            scored.append((h.hexdigest(), nid))

        scored.sort(key=lambda x: x[0])
        sample_ids = [nid for _, nid in scored[:self.sample_size]]
        return sample_ids, sample_ids[0]

    def derive_sample_with_bandwidths(
        self,
        round_num: int,
        bandwidths: Dict[int, float],
    ) -> Tuple[List[int], int]:
        """
        Select sample and aggregator using bandwidth info.

        Args:
            round_num: Current round number
            bandwidths: Dict mapping node_id -> bandwidth capacity

        Returns:
            Tuple of (sample_node_ids, aggregator_node_id) where
            aggregator is the highest-bandwidth node in the sample.
        """
        sample_ids, _ = self.derive_sample(round_num)

        aggregator_id = max(
            sample_ids,
            key=lambda nid: bandwidths.get(nid, 0.0)
        )
        return sample_ids, aggregator_id

    def get_sample_for_next_round(
        self,
        round_num: int,
        bandwidths: Optional[Dict[int, float]] = None,
    ) -> Dict:
        """
        Get complete sample info for next round (round_num + 1).

        Args:
            round_num: Current round number
            bandwidths: Optional bandwidth mapping

        Returns:
            Dict with: sample_ids, aggregator_id, next_sample_ids, next_aggregator_id
        """
        current_sample, current_agg = (
            self.derive_sample_with_bandwidths(round_num, bandwidths)
            if bandwidths
            else self.derive_sample(round_num)
        )
        next_sample, next_agg = (
            self.derive_sample_with_bandwidths(round_num + 1, bandwidths)
            if bandwidths
            else self.derive_sample(round_num + 1)
        )

        return {
            "sample_ids": current_sample,
            "aggregator_id": current_agg,
            "next_sample_ids": next_sample,
            "next_aggregator_id": next_agg,
        }

    def get_aggregator(
        self,
        sample_ids: List[int],
        bandwidths: Dict[int, float],
    ) -> int:
        """
        Select aggregator from a given sample based on bandwidth.

        Args:
            sample_ids: List of node IDs in the sample
            bandwidths: Mapping of node_id -> bandwidth capacity

        Returns:
            Node ID of the selected aggregator (highest bandwidth).
        """
        if not sample_ids:
            raise ValueError("Sample cannot be empty")
        return max(sample_ids, key=lambda nid: bandwidths.get(nid, 0.0))

    def __repr__(self) -> str:
        return (
            f"PlexusSampler("
            f"nodes={len(self.node_ids)}, "
            f"sample_size={self.sample_size}, "
            f"hash={self.hash_algorithm})"
        )