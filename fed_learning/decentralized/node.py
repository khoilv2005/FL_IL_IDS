"""
PlexusNode - Autonomous peer node for decentralized federated learning.

Reference:
    Dhasade et al., "Practical Federated Learning without a Server",
    EuroMLSys 2025, Algorithm 2 (Push-Based Protocol)

Each PlexusNode can:
1. Train locally on its data (upon receiving TRAIN message)
2. Aggregate received models from other nodes (upon receiving AGGREGATE message)
3. Trigger next round by sending TRAIN to next sample (if aggregator)

This replaces the centralized server model with true peer-to-peer communication.
"""

import copy
from collections import OrderedDict
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn

from ..strategies.federated.plexus import PopulationView


class PlexusNode:
    """
    Autonomous peer node that implements the Plexus push-based protocol.

    Each node:
    - Maintains its own local model copy
    - Tracks collected models from other nodes
    - Can act as aggregator when elected
    - Sends TRAIN messages to next sample when aggregation completes

    This mirrors the behavior of peers in the original Plexus system,
    where "upon receive AGGREGATE(round_r, M_i)" triggers the push to next sample.
    """

    def __init__(
        self,
        node_id: int,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        bandwidth: float = 1.0,
        device: str = "cpu",
    ):
        """
        Args:
            node_id: Unique identifier for this node.
            X_train: Training features for this node.
            y_train: Training labels for this node.
            bandwidth: Bandwidth capacity (used for aggregator selection).
            device: Device for training ('cpu' or 'cuda:X').
        """
        self.node_id = node_id
        self.X_train = X_train
        self.y_train = y_train
        self.num_samples = len(y_train)
        self.bandwidth = bandwidth
        self.device = device

        # Local model (set via setup_model)
        self.local_model: Optional[nn.Module] = None

        # Collected models for aggregation: round_r -> List[Dict]
        self.collected_models: Dict[int, List[Dict]] = {}

        # Population view for this node
        self.population_view = PopulationView()

        # Local round estimate (updated when receiving aggregated model)
        self.round_estimate: int = 0

        # Online status
        self.is_online: bool = True

        # Incremental learning state
        self.current_task: int = 0
        self.seen_classes: set = set()

    def setup_model(self, model: nn.Module):
        """Initialize local model copy from a template."""
        self.local_model = copy.deepcopy(model).to(self.device)

    def set_task_data(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        task_id: int,
        task_classes: List[int],
    ):
        """Update data for incremental learning."""
        self.X_train = X_train
        self.y_train = y_train
        self.num_samples = len(y_train)
        self.current_task = task_id
        self.seen_classes.update(task_classes)

    def train_local(
        self,
        global_params: OrderedDict,
        trainer: Any,
        local_epochs: int = 1,
    ) -> Dict:
        """
        Execute local training (Algorithm 2: upon receive TRAIN).

        Loads global params, trains locally, returns updated params.

        Args:
            global_params: Global model parameters to load.
            trainer: Training strategy (must implement BaseTrainer interface).
            local_epochs: Number of local epochs to train.

        Returns:
            Dict with keys: 'client_id', 'params', 'num_samples', 'loss'.
        """
        if self.local_model is None:
            raise RuntimeError("Model not set up. Call setup_model() first.")

        # Load global parameters
        self.local_model.load_state_dict(
            {k: v.to(self.device) for k, v in global_params.items()}
        )

        # Train locally
        self.local_model.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.local_model.parameters(), lr=0.001)

        total_loss = 0.0
        num_batches = 0

        # Simple training loop
        indices = torch.randperm(len(self.X_train))
        X_shuffled = self.X_train[indices]
        y_shuffled = self.y_train[indices]

        for epoch in range(local_epochs):
            for i in range(0, len(X_shuffled), 32):  # batch_size=32
                batch_X = X_shuffled[i:i+32].to(self.device)
                batch_y = y_shuffled[i:i+32].to(self.device)

                optimizer.zero_grad()
                output = self.local_model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        return {
            "client_id": self.node_id,
            "params": OrderedDict(
                (k, v.cpu().clone())
                for k, v in self.local_model.state_dict().items()
            ),
            "num_samples": self.num_samples,
            "loss": avg_loss,
        }

    def receive_for_aggregation(self, round_r: int, model_result: Dict):
        """
        Receive a trained model for aggregation (Algorithm 2: upon receive AGGREGATE).

        Args:
            round_r: The round number this model belongs to.
            model_result: Dict with 'client_id', 'params', 'num_samples', 'loss'.
        """
        if round_r not in self.collected_models:
            self.collected_models[round_r] = []
        self.collected_models[round_r].append(model_result)

    def can_aggregate(self, round_r: int, threshold: int) -> bool:
        """
        Check if enough models have been collected for aggregation.

        Args:
            round_r: Round number.
            threshold: Minimum number of models required (K * success_fraction).

        Returns:
            True if threshold reached, False otherwise.
        """
        return len(self.collected_models.get(round_r, [])) >= threshold

    def aggregate(
        self,
        round_r: int,
        aggregator: Any,
        global_params: Optional[OrderedDict] = None,
    ) -> Optional[OrderedDict]:
        """
        Perform FedAvg aggregation on collected models.

        Args:
            round_r: Round number.
            aggregator: Aggregator object with weighted_average method.
            global_params: Current global params (passed for fallback).

        Returns:
            Aggregated parameters, or None if no models collected.
        """
        results = self.collected_models.pop(round_r, [])

        if not results:
            return global_params

        # Use aggregator's weighted average
        return aggregator.weighted_average(results, global_params)

    def receive_aggregated_model(self, round_num: int):
        """Update local round estimate when receiving aggregated model."""
        if round_num > self.round_estimate:
            self.round_estimate = round_num

    def merge_population_view(self, other_view: PopulationView):
        """Merge another node's population view."""
        self.population_view.merge(other_view)

    def get_population_view(self) -> PopulationView:
        """Get this node's population view."""
        return self.population_view

    def get_local_params(self) -> Optional[OrderedDict]:
        """Get current local model parameters."""
        if self.local_model is None:
            return None
        return OrderedDict(
            (k, v.cpu().clone())
            for k, v in self.local_model.state_dict().items()
        )

    def set_local_params(self, params: OrderedDict):
        """Set local model parameters."""
        if self.local_model is not None:
            self.local_model.load_state_dict(
                {k: v.to(self.device) for k, v in params.items()}
            )

    def is_aggregator_for_round(self, round_r: int, sampler: Any, all_nodes: List["PlexusNode"]) -> bool:
        """
        Check if this node is the aggregator for the given round.

        Args:
            round_r: Round number.
            sampler: PlexusSampler instance.
            all_nodes: List of all PlexusNode instances.

        Returns:
            True if this node is the aggregator for this round.
        """
        node_ids = [n.node_id for n in all_nodes]
        bandwidths = {n.node_id: n.bandwidth for n in all_nodes}

        _, aggregator_id = sampler.derive_sample_with_bandwidths(round_r, bandwidths)
        return self.node_id == aggregator_id

    def __repr__(self) -> str:
        return (
            f"PlexusNode("
            f"id={self.node_id}, "
            f"bandwidth={self.bandwidth}, "
            f"online={self.is_online}, "
            f"round_est={self.round_estimate})"
        )