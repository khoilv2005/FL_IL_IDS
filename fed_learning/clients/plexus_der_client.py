"""
PlexusDER Client - Decentralized DER client using Plexus mechanisms.

Reference:
    DER: Yan, Xie, He (CVPR 2021)
    Plexus: Dhasade et al. (EuroMLSys 2025)

Inherits from DERClient to get the actual DER training logic:
- Two-stage training (representation + classifier)
- Exemplar replay buffer with herding selection
- _create_combined_batches(), _create_balanced_batches()

Adds Plexus protocol state:
- PopulationView tracking
- Round estimate synchronization
- Online/offline status
- receive_task_history() for model structure sync
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Set

import torch

from .der_client import DERClient
from ..strategies.federated.plexus import PopulationView


class PlexusDERClient(DERClient):
    """
    DER client with Plexus decentralized protocol.

    Inherits from DERClient for:
    - Two-stage training (train())
    - Exemplar replay buffer (update_exemplars())
    - Batch creation (_create_combined_batches, _create_balanced_batches)

    Adds Plexus protocol state:
    - PopulationView tracking
    - Round estimate synchronization
    - receive_task_history() for piggybacked model structure
    """

    def __init__(
        self,
        client_id: int,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        buffer_size: int = 500,
    ):
        super().__init__(client_id, X_train, y_train, buffer_size)

        # Plexus protocol state (from PlexusClient)
        self.population_view = PopulationView()
        self.round_estimate: int = 0
        self.is_online: bool = True

        # DER-specific: model structure tracking (synced via piggyback)
        self._task_classes_history: Dict[int, List[int]] = {}
        self._num_extractors: int = 0

    def receive_aggregated_model(self, round_num: int):
        """
        Called when this client receives the aggregated model.

        Updates local round estimate.
        """
        if round_num > self.round_estimate:
            self.round_estimate = round_num

    def merge_population_view(self, other_view: PopulationView):
        """Merge the population view received along with a model transfer."""
        self.population_view.merge(other_view)

    def receive_task_history(self, task_history: Dict[int, List[int]]):
        """
        Receive task classes history via piggyback from aggregator.

        Used to reconstruct model structure when joining mid-training.
        """
        self._task_classes_history = task_history
        self._num_extractors = len(task_history)
