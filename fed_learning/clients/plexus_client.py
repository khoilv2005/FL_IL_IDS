"""
Plexus Client - Client for decentralized Federated Learning (Plexus).

Reference:
    Dhasade et al., "Practical Federated Learning without a Server",
    EuroMLSys 2025

Extends FederatedClient with:
1. Population view tracking (each client maintains its own view of the network)
2. Round estimation (local round counter synchronized via model transfers)
3. Online/offline status simulation for availability-aware sampling
4. set_task_data() for compatibility with the incremental task loop
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple

import torch

from .client import FederatedClient
from ..strategies.federated.plexus import PopulationView


class PlexusClient(FederatedClient):
    """
    Federated client with Plexus decentralized protocol state.

    In the full Plexus system, each client/peer maintains:
    - A population view (who is active and at which round).
    - A local round estimate (derived from received aggregated models).
    - An online/offline flag (simulated by the server for availability checks).

    This client adds those minimal state fields on top of the standard
    FederatedClient so the PlexusServer can simulate the decentralized protocol.
    """

    def __init__(
        self,
        client_id: int,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
    ):
        super().__init__(client_id, X_train, y_train)

        # Population view — piggybacked on model transfers
        self.population_view = PopulationView()

        # Local round estimate (updated when receiving aggregated model)
        self.round_estimate: int = 0

        # Availability flag (managed by PlexusServer)
        self.is_online: bool = True

        # Incremental learning support
        self.current_task: int = 0
        self.seen_classes: Set[int] = set()

    def set_task_data(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        task_id: int,
        task_classes: List[int],
    ):
        """Update client data for a new task (incremental setting)."""
        self.X_train = X_train
        self.y_train = y_train
        self.num_samples = len(y_train)
        self.current_task = task_id
        self.seen_classes.update(task_classes)

    def receive_aggregated_model(self, round_num: int):
        """
        Called by PlexusServer when this client receives the aggregated model.

        Updates the local round estimate (mirrors ``received_aggregated_model``
        in ``dlsim/plexus/community.py``).
        """
        if round_num > self.round_estimate:
            self.round_estimate = round_num

    def merge_population_view(self, other_view: PopulationView):
        """Merge the population view received along with a model transfer."""
        self.population_view.merge(other_view)
