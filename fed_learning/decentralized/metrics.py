"""
PlexusMetrics - Tracking and analysis for decentralized FL.

Reference:
    Dhasade et al., "Practical Federated Learning without a Server",
    EuroMLSys 2025

This module tracks Plexus-specific metrics that are not applicable to
centralized FL, including:
- Communication rounds count
- Participation rate per round
- Sample diversity across rounds
- Aggregator distribution/fairness
- Convergence comparison with centralized baseline
"""

from collections import Counter
from typing import Dict, List, Optional

import numpy as np


class PlexusMetrics:
    """
    Tracks metrics specific to the decentralized Plexus protocol.

    These metrics help evaluate:
    - Efficiency: How well the sampling distributes participation
    - Fairness: How evenly the aggregator role is distributed
    - Convergence: How quickly the protocol reaches acceptable accuracy
    - Liveness: How often success_fraction threshold is met
    """

    def __init__(self):
        """Initialize empty metrics tracking."""
        # Per-round metrics
        self.rounds: List[Dict] = []

        # Aggregator distribution
        self.aggregator_counts: Counter = Counter()

        # Node participation tracking
        self.node_participation: Counter = Counter()

        # Sample diversity
        self.unique_samples: List[int] = []

        # Participation rates per round
        self.participation_rates: List[float] = []

        # Communication efficiency
        self.communication_rounds: int = 0

    def record_round(
        self,
        round_r: int,
        sample_size: int,
        participation: float,
        aggregator_id: int,
        nodes_in_sample: Optional[List[int]] = None,
    ):
        """
        Record metrics for a single round.

        Args:
            round_r: Round number.
            sample_size: Number of nodes in the sample (K).
            participation: Fraction of sample that participated (0-1).
            aggregator_id: ID of the node that aggregated this round.
            nodes_in_sample: List of node IDs in the sample.
        """
        self.rounds.append({
            "round": round_r,
            "sample_size": sample_size,
            "participation": participation,
            "aggregator_id": aggregator_id,
            "nodes_in_sample": nodes_in_sample or [],
        })

        self.aggregator_counts[aggregator_id] += 1
        self.participation_rates.append(participation)
        self.communication_rounds += 1

        if nodes_in_sample:
            for nid in nodes_in_sample:
                self.node_participation[nid] += 1

    def get_aggregator_distribution(self) -> Dict[int, int]:
        """
        Get histogram of which nodes served as aggregator.

        Returns:
            Dict mapping node_id -> number of times that node was aggregator.
        """
        return dict(self.aggregator_counts)

    def get_participation_rate(self) -> float:
        """
        Get average participation rate across all rounds.

        Returns:
            Average fraction of sample that participated per round.
        """
        if not self.participation_rates:
            return 0.0
        return float(np.mean(self.participation_rates))

    def get_sample_diversity(self, total_nodes: int) -> float:
        """
        Calculate sample diversity metric.

        This measures how evenly participation is distributed across nodes.
        Higher is better (more evenly distributed).

        Returns:
            Float between 0 and 1, where 1 means perfect even distribution.
        """
        if not self.node_participation:
            return 0.0

        # Gini coefficient (0 = perfect equality, 1 = perfect inequality)
        participation_values = list(self.node_participation.values())

        # Number of distinct nodes that participated
        unique_participants = len(self.node_participation)

        if unique_participants < 2:
            return 1.0

        # Simple measure: unique participants / total possible (rounds * sample_size)
        max_possible_participation = self.communication_rounds * (
            self.rounds[0]["sample_size"] if self.rounds else 1
        )

        if max_possible_participation == 0:
            return 0.0

        actual_total = sum(participation_values)
        return actual_total / max_possible_participation

    def get_aggregator_fairness(self) -> float:
        """
        Calculate how evenly the aggregator role is distributed.

        Returns:
            Float between 0 and 1, where 1 means perfect fairness.
        """
        if not self.aggregator_counts:
            return 0.0

        num_aggregators = len(self.aggregator_counts)
        expected_per_aggregator = self.communication_rounds / num_aggregators

        if expected_per_aggregator == 0:
            return 1.0

        # Variance from expected
        actual_counts = list(self.aggregator_counts.values())
        variance = np.var(actual_counts) if len(actual_counts) > 1 else 0

        # Normalize: 1 - (variance / max_variance)
        max_variance = expected_per_aggregator ** 2 * (num_aggregators - 1)
        fairness = 1.0 - (variance / max_variance) if max_variance > 0 else 1.0

        return max(0.0, min(1.0, fairness))

    def get_communication_cost(self) -> int:
        """
        Get total communication rounds.

        In Plexus, each round involves:
        - K nodes sending TRAIN results to aggregator (K messages)
        - 1 aggregator sending aggregated model to next sample (~K messages)

        Returns:
            Total number of rounds executed.
        """
        return self.communication_rounds

    def get_liveness_score(self) -> float:
        """
        Calculate liveness: how often success_fraction threshold was met.

        Returns:
            Fraction of rounds where enough nodes participated.
        """
        if not self.participation_rates:
            return 0.0

        # Count rounds where participation >= success_fraction (0.8)
        successful = sum(1 for p in self.participation_rates if p >= 0.8)
        return successful / len(self.participation_rates)

    def get_summary(self) -> Dict:
        """
        Get complete metrics summary.

        Returns:
            Dict with all Plexus metrics.
        """
        return {
            "communication_rounds": self.communication_rounds,
            "avg_participation_rate": self.get_participation_rate(),
            "aggregator_distribution": self.get_aggregator_distribution(),
            "aggregator_fairness": self.get_aggregator_fairness(),
            "liveness_score": self.get_liveness_score(),
            "total_nodes_participated": len(self.node_participation),
            "unique_aggregators": len(self.aggregator_counts),
        }

    def __repr__(self) -> str:
        return (
            f"PlexusMetrics("
            f"rounds={self.communication_rounds}, "
            f"avg_participation={self.get_participation_rate():.2%}, "
            f"liveness={self.get_liveness_score():.2%})"
        )