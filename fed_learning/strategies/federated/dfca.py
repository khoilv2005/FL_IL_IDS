"""
DFCA Strategy - Decentralized Federated Clustering Algorithm with Incremental Learning.

DFCA (Dhasade et al.) is a fully decentralized clustered FL algorithm where:
- Each client maintains k cluster models (one per cluster)
- Cluster assignment is LOCAL (client picks cluster with minimum loss)
- Aggregation is DECENTRALIZED (peer-to-peer running average, no central server)
- k clusters represent groups of clients with similar data distributions

This implementation extends DFCA for class-incremental learning:
- Model architecture (CNN_GRU) is FIXED: num_classes=34 always
- Incremental learning handled via seen-class masking in compute_loss
- Active clients per task: 50%, 60%, 70%, 80%, 90%, 100%
- k = 10 fixed clusters
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import BaseTrainer, BaseAggregator


class DFCATrainer(BaseTrainer):
    """
    DFCA local trainer with class-incremental output masking.

    This trainer is used for local update steps within DFCA rounds.
    The actual cluster assignment and peer aggregation happen in
    DFCAClient and DFCAServer respectively.

    Key behavior:
    - Standard CrossEntropy loss (inherited from BaseTrainer)
    - Seen-class masking for class-incremental setting
    - No proximal term, no regularization beyond incremental masking
    """

    def __init__(
        self,
        local_epochs: int = 1,
        learning_rate: float = 0.001,
        batch_size: int = 2048,
    ):
        super().__init__()
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.current_task = 0
        self.seen_classes: set = set()
        self.new_classes: list = []

    def set_task(self, task_id: int, new_classes: list):
        """Called when entering a new task to update seen classes."""
        self.current_task = task_id
        self.new_classes = list(new_classes)
        self.seen_classes = self.seen_classes | set(new_classes)

    def compute_loss(
        self,
        model: nn.Module,
        output: torch.Tensor,
        target: torch.Tensor,
        global_params: Optional[OrderedDict] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute CE loss with seen-class masking for incremental learning.

        Uses _seen_class_cross_entropy from BaseTrainer which:
        - Slices logits to only seen classes
        - Remaps targets to the reduced label space
        - Ignores gradients for unseen classes
        """
        return self._seen_class_cross_entropy(output, target)


class DFCAAggregator(BaseAggregator):
    """
    DFCA aggregator - a placeholder that raises on FedAvg-style calls.

    In DFCA, there is NO centralized aggregation. Each client aggregates
    peer models independently via sequential running average.

    This class exists to satisfy the strategy factory pattern but
    MUST NOT be called to aggregate a global model. Any FedAvg-style
    aggregate() call is a programming error in the DFCA pipeline.

    The server orchestrates graph communication, not aggregation.
    """

    def __init__(self, num_clusters: int = 10):
        super().__init__()
        self.num_clusters = num_clusters

    def aggregate(
        self,
        results: List[Dict],
        global_params: Optional[OrderedDict] = None,
        **kwargs
    ) -> OrderedDict:
        """
        NOT USED in DFCA - raises to catch misuse.

        In DFCA, aggregation is fully decentralized. Each client maintains
        its own cluster model bank and aggregates peer updates independently.
        """
        raise RuntimeError(
            "DFCAAggregator.aggregate() should never be called. "
            "DFCA uses decentralized peer-to-peer aggregation via "
            "DFCAClient.aggregate_cluster_models()."
        )

    @staticmethod
    def sequential_running_average(
        current: torch.Tensor,
        incoming: torch.Tensor,
        count: int
    ) -> torch.Tensor:
        """
        Sequential running average formula from DFCA paper.

        The goal is unweighted averaging: after incorporating r+1 values
        (the local model + r neighbors), each should have weight 1/(r+1).

        Verified by induction:
        - r=0: alpha=1, beta=1 -> result = incoming (1st model)
        - r=1: alpha=1/2, beta=1/2 -> (1/2)*incoming + (1/2)*prev
        - After expansion: (1/2)*incoming2 + (1/2)*(1/2)*incoming1 + (1/2)*(1/2)*local
        - Each of 3 models: 1/3, 1/3, 1/3. Verified.

        Formula:
            theta_new = ((r+1)/(r+2)) * theta_old + (1/(r+2)) * theta_incoming

        Args:
            current: Current model params (on this client)
            incoming: Incoming model params from a neighbor
            count: Number of neighbors already incorporated (r)

        Returns:
            Updated params after incorporating the incoming model
        """
        if count < 0:
            count = 0
        alpha = (count + 1.0) / (count + 2.0)
        beta = 1.0 / (count + 2.0)
        return alpha * current + beta * incoming
