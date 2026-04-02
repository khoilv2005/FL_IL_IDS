"""
PlexusSampler - Consistent-hashing peer sampling.

Reference:
    Dhasade et al., "Practical Federated Learning without a Server",
    EuroMLSys 2025, Algorithm 1 (DERIVE_SAMPLE)

This implements the deterministic, coordination-free peer sampling using
consistent hashing. Every node can independently compute the same sample
for a given round.
"""

import hashlib
from typing import Dict, List, Tuple, Optional


class PlexusSampler:
    """
    Implements Algorithm 1 from the paper: DERIVE_SAMPLE function.

    Uses MD5 hash of (node_id || round) for deterministic, unbiased ordering.
    The first K nodes in sorted hash order form the sample. The node with
    highest bandwidth in the sample becomes the aggregator.

    Key properties:
    - Deterministic: same (node_ids, round) always produces same sample
    - Coordination-free: every node computes same sample independently
    - O(N log N) per round
    """

    def __init__(
        self,
        node_ids: List[int],
        sample_size: int = 13,
        hash_algorithm: str = "md5",
    ):
        """
        Args:
            node_ids: List of all participating node IDs.
            sample_size: K — number of nodes per sample (default 13).
            hash_algorithm: Hash function to use (default md5, as in paper).
        """
        self.node_ids = sorted(set(node_ids))
        self.sample_size = min(sample_size, len(self.node_ids))
        self.hash_algorithm = hash_algorithm

    def _compute_hash(self, node_id: int, round_num: int) -> str:
        """Compute hash for a node in a given round."""
        # Paper uses MD5(peer_id || round) — note the '-' separator
        h = hashlib.new(self.hash_algorithm)
        h.update(f"{node_id}-{round_num}".encode())
        return h.hexdigest()

    def derive_sample(self, round_num: int) -> Tuple[List[int], int]:
        """
        Algorithm 1 from paper: DERIVE_SAMPLE(Nodes, round_r, K).

        Returns:
            Tuple of (sample_node_ids, aggregator_node_id).

        The aggregator is selected as the node with highest bandwidth
        among the sample. Since we don't have bandwidth info here,
        the first node in hash order is returned as default aggregator.
        Use get_aggregator_with_bandwidths() for bandwidth-based selection.
        """
        # Score each node by hash
        scored = []
        for nid in self.node_ids:
            h = self._compute_hash(nid, round_num)
            scored.append((h, nid))

        # Sort by hash (lexicographic)
        scored.sort(key=lambda x: x[0])

        # Take first K nodes as sample
        sample_ids = [nid for _, nid in scored[:self.sample_size]]

        # Default aggregator = first in sample (when no bandwidth info)
        aggregator_id = sample_ids[0]

        return sample_ids, aggregator_id

    def derive_sample_with_bandwidths(
        self,
        round_num: int,
        bandwidths: Dict[int, float],
    ) -> Tuple[List[int], int]:
        """
        Extended version that selects aggregator based on bandwidth.

        Returns:
            Tuple of (sample_node_ids, aggregator_node_id) where
            aggregator is the highest-bandwidth node in the sample.
        """
        sample_ids, _ = self.derive_sample(round_num)

        # Select highest-bandwidth node as aggregator
        aggregator_id = max(
            sample_ids,
            key=lambda nid: bandwidths.get(nid, 0.0)
        )

        return sample_ids, aggregator_id

    def get_aggregator(
        self,
        sample_ids: List[int],
        bandwidths: Dict[int, float],
    ) -> int:
        """
        Select aggregator from a given sample based on bandwidth.

        Args:
            sample_ids: List of node IDs in the sample.
            bandwidths: Mapping of node_id -> bandwidth capacity.

        Returns:
            Node ID of the selected aggregator (highest bandwidth).
        """
        if not sample_ids:
            raise ValueError("Sample cannot be empty")

        return max(
            sample_ids,
            key=lambda nid: bandwidths.get(nid, 0.0)
        )

    def get_sample_for_next_round(
        self,
        round_num: int,
        bandwidths: Optional[Dict[int, float]] = None,
    ) -> Dict:
        """
        Get complete sample info for a round, including next round's sample.

        This supports the push-based protocol where current aggregator
        triggers the next round's sample.

        Args:
            round_num: Current round number.
            bandwidths: Optional bandwidth mapping for aggregator selection.

        Returns:
            Dict with keys: 'sample_ids', 'aggregator_id', 'next_sample_ids', 'next_aggregator_id'.
        """
        current_sample, current_agg = self.derive_sample(round_num)

        if bandwidths:
            current_agg = self.get_aggregator(current_sample, bandwidths)

        # Next round's sample
        next_sample, next_agg = self.derive_sample(round_num + 1)

        if bandwidths:
            next_agg = self.get_aggregator(next_sample, bandwidths)

        return {
            "sample_ids": current_sample,
            "aggregator_id": current_agg,
            "next_sample_ids": next_sample,
            "next_aggregator_id": next_agg,
        }

    def __repr__(self) -> str:
        return (
            f"PlexusSampler("
            f"nodes={len(self.node_ids)}, "
            f"sample_size={self.sample_size}, "
            f"hash={self.hash_algorithm})"
        )