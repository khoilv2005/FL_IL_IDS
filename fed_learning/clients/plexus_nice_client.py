"""
PlexusNICE Client - Decentralized NICE client using Plexus mechanisms.

Reference:
    NICE: Gurbuz, Moorman, Dovrolis (CVPR 2024)
    Plexus: Dhasade et al. (EuroMLSys 2025)

Inherits from NICEClient to get the actual NICE training logic:
- Phase-based training with tau-greedy neuron selection
- Connection pruning (drop_young_to_learner, grow_all_to_young)
- Output masking via model.forward_output()
- Gradient freezing via reset_frozen_gradients()

Adds Plexus protocol state:
- PopulationView tracking
- Round estimate synchronization
- Online/offline status
- receive_neuron_ages() / receive_freeze_masks() for state sync
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Set

import numpy as np
import torch

from .nice_client import NICEClient
from ..strategies.federated.plexus import PopulationView


class PlexusNICEClient(NICEClient):
    """
    NICE client with Plexus decentralized protocol.

    Inherits from NICEClient for:
    - Phase-based training (train())
    - Neuron selection and pruning
    - Output masking and gradient freezing

    Adds Plexus protocol state:
    - PopulationView tracking
    - Round estimate synchronization
    - receive_neuron_ages() / receive_freeze_masks() for piggybacked state
    """

    def __init__(
        self,
        client_id: int,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        max_phases: int = 5,
        phase_epochs: int = 5,
        tau: float = 0.95,
    ):
        super().__init__(client_id, X_train, y_train, max_phases, phase_epochs, tau)

        # Plexus protocol state (from PlexusClient)
        self.population_view = PopulationView()
        self.round_estimate: int = 0
        self.is_online: bool = True

        # NICE state (piggybacked cache/debug state)
        # NOTE: These are NOT source of truth for training.
        # Source of truth is worker_config sent from server via training worker.
        # These fields exist for protocol symmetry and debugging.
        self._neuron_ages: Optional[Dict[str, np.ndarray]] = None
        self._freeze_masks: Optional[Dict[str, np.ndarray]] = None

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

    def receive_neuron_ages(self, ages: Dict[str, np.ndarray]):
        """
        Receive neuron ages via piggyback from aggregator.

        NOTE: This is piggyback cache/debug state, not source of truth.
        The actual neuron ages used in training come from worker_config
        sent by the server to the training worker.
        """
        self._neuron_ages = ages

    def receive_freeze_masks(self, masks: Dict):
        """
        Receive freeze masks via piggyback from aggregator.

        NOTE: This is piggyback cache/debug state, not source of truth.
        The actual freeze masks used in training come from worker_config
        sent by the server to the training worker.
        """
        self._freeze_masks = {k: np.array(v) for k, v in masks.items()}
