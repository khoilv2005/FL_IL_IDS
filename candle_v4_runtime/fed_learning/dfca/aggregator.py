"""
DFCA Aggregator - Pure sequential running average logic.

This aggregator is NOT used for central aggregation (DFCA is fully decentralized).
It provides utility methods for the sequential running average formula.
"""


class DFCAAggregator:
    """
    DFCA aggregator providing sequential running average formula.

    Not used for central aggregation — DFCA is fully decentralized.
    The actual aggregation happens peer-to-peer in DFCANode.aggregate_received_messages().
    """

    def __init__(self, num_clusters: int = 10):
        self.num_clusters = num_clusters

    @staticmethod
    def sequential_running_average(current, incoming, count: int = 0):
        """
        Sequential running average from DFCA paper.

        After incorporating r neighbors (r is count of already-aggregated neighbors):
            theta_new = ((r+1)/(r+2)) * theta_old + (1/(r+2)) * theta_incoming

        This gives equal weight to local model + each received neighbor model:
        - After 1 neighbor: 1/2 local + 1/2 neighbor
        - After 2 neighbors: 1/3 local + 1/3 nbr1 + 1/3 nbr2
        - etc.

        Args:
            current: Current tensor value.
            incoming: Incoming neighbor tensor value.
            count: Number of neighbors already aggregated (r in paper formula).

        Returns:
            Updated tensor value.
        """
        if count < 0:
            count = 0
        alpha = (count + 1.0) / (count + 2.0)
        beta = 1.0 / (count + 2.0)
        return alpha * current + beta * incoming

    def aggregate(self, results, global_params=None, **kwargs):
        """
        Raises: DFCA does not use central aggregation.
        """
        raise RuntimeError(
            "DFCAAggregator.aggregate() should never be called. "
            "DFCA uses decentralized peer-to-peer aggregation. "
            "Each node aggregates received messages via sequential running average."
        )
