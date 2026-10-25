"""
PlexusNode - Autonomous peer node implementing Algorithm 2 from Plexus paper.

Reference:
    Dhasade et al., "Practical Federated Learning without a Server",
    EuroMLSys 2025, Algorithm 2 (Push-Based Protocol)

Each PlexusNode implements:
- upon receive TRAIN(round_r, model_M): local_train() -> send to aggregator
- upon receive AGGREGATE(round_r, M_i): collect -> aggregate when threshold -> push to next sample

This is the core of the "without a server" design.
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Callable

import torch
import torch.nn as nn


class PlexusNode:
    """
    Autonomous peer node for the Plexus decentralized protocol.

    Each node can be:
    - A trainer: receives TRAIN, trains locally, sends result to aggregator
    - An aggregator: receives AGGREGATE from sample, collects, aggregates, pushes to next sample

    Key distinction from centralized FL:
    - No central server coordinates
    - Each node knows the sample for any round (via DERIVE_SAMPLE)
    - Aggregator is elected per-round by bandwidth
    - Aggregator pushes to next sample when threshold reached
    """

    def __init__(
        self,
        node_id: int,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        bandwidth: float = 1.0,
        device: str = "cpu",
        batch_size: int = 32,
    ):
        """
        Args:
            node_id: Unique peer identifier
            X_train: Local training features
            y_train: Local training labels
            bandwidth: Upload bandwidth capacity (for aggregator election)
            device: Training device
            batch_size: Batch size for local training
        """
        self.node_id = node_id
        self.X_train = X_train
        self.y_train = y_train
        self.num_samples = len(y_train)
        self.bandwidth = bandwidth
        self.device = device
        self.batch_size = batch_size

        # Local model (set via set_model)
        self.model: Optional[nn.Module] = None

        # Collected models for aggregation: round_r -> List[model_result]
        self.collected_models: Dict[int, List[Dict]] = {}

        # Pending aggregated models to send to next sample: round_r -> params
        self.pending_aggregated: Dict[int, OrderedDict] = {}

        # Callbacks for sending messages (set by orchestrator)
        self.send_to_peer: Optional[Callable] = None
        self.send_to_sample: Optional[Callable] = None

    def set_model(self, model: nn.Module):
        """Set the local model for training."""
        self.model = model.to(self.device)

    def set_send_callbacks(
        self,
        send_to_peer: Callable,
        send_to_sample: Callable,
    ):
        """
        Set callbacks for sending messages.

        Args:
            send_to_peer(peer_id, msg_type, round_r, data): Send to specific peer
            send_to_sample(sample_ids, msg_type, round_r, data): Send to multiple peers
        """
        self.send_to_peer = send_to_peer
        self.send_to_sample = send_to_sample

    # =========================================================================
    # Algorithm 2: upon receive TRAIN(round_r, model_M)
    # =========================================================================
    def receive_train(
        self,
        round_r: int,
        global_params: OrderedDict,
        derive_sample_fn: Callable,
        bandwidths: Dict[int, float],
        local_epochs: int = 1,
        learning_rate: float = 0.001,
    ) -> Dict:
        """
        Handle receiving TRAIN message.

        Algorithm 2 step 1:
            M_local = local_train(M, local_data)
            sample, aggregator = DERIVE_SAMPLE(Nodes, round_r, K)
            send(aggregator, AGGREGATE(round_r, M_local))

        Args:
            round_r: Current round number
            global_params: Global model parameters to train on
            derive_sample_fn: Function(node_ids, round_r, K) -> (sample_ids, aggregator_id)
            bandwidths: Dict of node_id -> bandwidth for aggregator selection
            local_epochs: Number of local epochs
            learning_rate: Learning rate for local training

        Returns:
            Dict with training result
        """
        # Load global params
        self.model.load_state_dict(
            {k: v.to(self.device) for k, v in global_params.items()}
        )

        # Local training
        loss = self._local_train(local_epochs, learning_rate)

        # Get sample and aggregator for this round
        node_ids = list(bandwidths.keys())
        K = len(node_ids)  # Will be clamped by derive_sample_fn
        sample_ids, aggregator_id = derive_sample_fn(node_ids, round_r, K)

        # Prepare result
        result = {
            "client_id": self.node_id,
            "params": OrderedDict(
                (k, v.cpu().clone())
                for k, v in self.model.state_dict().items()
            ),
            "num_samples": self.num_samples,
            "loss": loss,
        }

        # Send to aggregator (NOT to all peers - just the aggregator)
        if self.send_to_peer is not None:
            self.send_to_peer(aggregator_id, "AGGREGATE", round_r, result)

        return result

    # =========================================================================
    # Algorithm 2: upon receive AGGREGATE(round_r, M_i)
    # =========================================================================
    def receive_aggregate(
        self,
        round_r: int,
        model_result: Dict,
        threshold: int,
    ) -> bool:
        """
        Handle receiving AGGREGATE message (collection phase).

        Algorithm 2 step 2:
            collected[round_r].add(M_i)
            if len(collected[round_r]) >= K * success_fraction:
                [trigger aggregation]

        Args:
            round_r: Round number
            model_result: Dict with 'client_id', 'params', 'num_samples', 'loss'
            threshold: Minimum number of models needed (K * success_fraction)

        Returns:
            True if threshold reached and aggregation triggered
        """
        if round_r not in self.collected_models:
            self.collected_models[round_r] = []

        self.collected_models[round_r].append(model_result)

        # Check if threshold reached
        if len(self.collected_models[round_r]) >= threshold:
            return True
        return False

    def try_aggregate_and_push(
        self,
        round_r: int,
        threshold: int,
        derive_sample_fn: Callable,
        bandwidths: Dict[int, float],
        weighted_average_fn: Callable,
        K: int,
    ) -> Optional[OrderedDict]:
        """
        Try to aggregate collected models and push to next sample.

        Algorithm 2 step 2 (continued):
            if len(collected) >= K * success_fraction:
                M_agg = weighted_average(collected)
                sample_next, _ = DERIVE_SAMPLE(Nodes, round_r + 1, K)
                for node in sample_next:
                    send(node, TRAIN(round_r + 1, M_agg))

        Args:
            round_r: Current round
            threshold: Minimum models needed
            derive_sample_fn: Function to derive next sample
            bandwidths: Bandwidth dict for aggregator selection
            weighted_average_fn: Function to aggregate models
            K: Sample size

        Returns:
            Aggregated params if aggregation happened, None otherwise
        """
        if round_r not in self.collected_models:
            return None

        collected = self.collected_models[round_r]
        if len(collected) < threshold:
            return None

        # Aggregate using FedAvg
        aggregated_params = weighted_average_fn(collected)

        # Store pending aggregated model for next round
        self.pending_aggregated[round_r] = aggregated_params

        # Get next sample
        node_ids = list(bandwidths.keys())
        sample_next, _ = derive_sample_fn(node_ids, round_r + 1, K)

        # Push TRAIN to next sample
        if self.send_to_sample is not None:
            self.send_to_sample(
                sample_next,
                "TRAIN",
                round_r + 1,
                aggregated_params,
            )

        return aggregated_params

    # =========================================================================
    # Local training helper
    # =========================================================================
    def _local_train(
        self,
        local_epochs: int,
        learning_rate: float,
    ) -> float:
        """
        Perform local training.

        This is standard SGD - same as FedAvg local training.
        The paper doesn't specify a novel training method, just that
        each node trains locally on its data.

        Args:
            local_epochs: Number of epochs
            learning_rate: Learning rate

        Returns:
            Average training loss
        """
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        total_loss = 0.0
        num_batches = 0

        indices = torch.randperm(len(self.X_train))
        X_shuffled = self.X_train[indices]
        y_shuffled = self.y_train[indices]

        for epoch in range(local_epochs):
            for i in range(0, len(X_shuffled), self.batch_size):
                batch_X = X_shuffled[i:i+self.batch_size].to(self.device)
                batch_y = y_shuffled[i:i+self.batch_size].to(self.device)

                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def get_local_params(self) -> Optional[OrderedDict]:
        """Get current local model parameters."""
        if self.model is None:
            return None
        return OrderedDict(
            (k, v.cpu().clone())
            for k, v in self.model.state_dict().items()
        )

    def __repr__(self) -> str:
        return (
            f"PlexusNode("
            f"id={self.node_id}, "
            f"bw={self.bandwidth:.2f}, "
            f"samples={self.num_samples})"
        )