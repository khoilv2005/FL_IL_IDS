"""
PlexusAggregator - Aggregation logic for Plexus protocol.

Reference:
    Dhasade et al., "Practical Federated Learning without a Server",
    EuroMLSys 2025

This handles:
- Weighted FedAvg of collected models
- Success fraction threshold checking
- Push-to-next-sample trigger
"""

from collections import OrderedDict
from math import floor
from typing import Dict, List, Optional, Callable

import torch.nn as nn


class PlexusAggregator:
    """
    Handles aggregation in the Plexus decentralized protocol.

    In Plexus, aggregation is NOT done by a central server.
    Instead, the elected aggregator for each round:
    1. Collects models from sample members
    2. When threshold reached (K * success_fraction), aggregates
    3. Pushes aggregated model to next round's sample

    This class provides the aggregation logic and threshold checking.
    """

    def __init__(
        self,
        sample_size: int = 4,
        success_fraction: float = 0.8,
    ):
        """
        Args:
            sample_size: K — number of nodes per sample
            success_fraction: Fraction of sample needed before aggregation (default 0.8)
        """
        self.sample_size = sample_size
        self.success_fraction = success_fraction

        # Paper threshold: floor(K * success_fraction). The "3 models" rule is
        # only a timeout/liveness fallback in the original source.
        self.threshold = max(1, floor(sample_size * success_fraction))

    def weighted_average(self, results: List[Dict]) -> OrderedDict:
        """
        Compute weighted average of model parameters (FedAvg).

        Weight is proportional to number of samples per client.

        Args:
            results: List of dicts with 'params' and 'num_samples'

        Returns:
            Aggregated parameters
        """
        if not results:
            return None

        total_samples = sum(r["num_samples"] for r in results)

        agg = None
        for r in results:
            w_i = r["num_samples"] / max(1, total_samples)
            params = r["params"]

            if agg is None:
                agg = OrderedDict(
                    (k, w_i * v.float()) for k, v in params.items()
                )
            else:
                for k in agg.keys():
                    if agg[k].dtype.is_floating_point:
                        agg[k] = agg[k] + w_i * params[k].float()
                    else:
                        agg[k] = params[k]

        return agg

    def can_aggregate(self, num_collected: int) -> bool:
        """
        Check if enough models have been collected for aggregation.

        Args:
            num_collected: Number of models received so far

        Returns:
            True if threshold reached
        """
        return num_collected >= self.threshold

    def get_threshold(self) -> int:
        """Get the threshold number of models needed."""
        return self.threshold

    def __repr__(self) -> str:
        return (
            f"PlexusAggregator("
            f"K={self.sample_size}, "
            f"s_f={self.success_fraction}, "
            f"threshold={self.threshold})"
        )
