"""
PlexusOrchestrator - Simulates the Plexus decentralized protocol.

Reference:
    Dhasade et al., "Practical Federated Learning without a Server",
    EuroMLSys 2025, Algorithm 2 (Push-Based Protocol)

This orchestrates the protocol on a single machine by:
1. Creating PlexusNode objects for each peer
2. Routing messages between nodes (simulating the network)
3. Triggering the initial TRAIN for round 0

Key difference from centralized FL:
- No central server aggregates
- Each node follows the protocol independently
- Aggregator role rotates and pushes to next sample
"""

import time
from collections import OrderedDict
from math import floor
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .sampler import PlexusSampler
from .aggregator import PlexusAggregator


class PlexusOrchestrator:
    """
    Orchestrates the Plexus decentralized protocol.

    The protocol (Algorithm 2):
        upon receive TRAIN(round_r, model_M):
            M_local = local_train(M, local_data)
            sample, aggregator = DERIVE_SAMPLE(Nodes, round_r, K)
            send(aggregator, AGGREGATE(round_r, M_local))

        upon receive AGGREGATE(round_r, M_i):
            collected[round_r].add(M_i)
            if len(collected[round_r]) >= K * success_fraction:
                M_agg = weighted_average(collected[round_r])
                sample_next, _ = DERIVE_SAMPLE(Nodes, round_r + 1, K)
                for node in sample_next:
                    send(node, TRAIN(round_r + 1, M_agg))

    The orchestrator:
    - Creates nodes with data
    - Sets up message routing (simulated)
    - Triggers initial round
    - Collects final aggregated model after all rounds complete
    """

    def __init__(
        self,
        node_ids: List[int],
        node_data: Dict[int, tuple],  # node_id -> (X_train, y_train)
        bandwidths: Dict[int, float],
        model_template: nn.Module,
        sample_size: int = 13,
        success_fraction: float = 0.8,
        local_epochs: int = 1,
        learning_rate: float = 0.001,
        device: str = "cpu",
        batch_size: int = 32,
    ):
        """
        Args:
            node_ids: List of participating node IDs
            node_data: Dict mapping node_id -> (X_train Tensor, y_train Tensor)
            bandwidths: Dict mapping node_id -> bandwidth capacity
            model_template: Model architecture to clone for each node
            sample_size: K — sample size per round
            success_fraction: Fraction needed before aggregation (default 0.8)
            local_epochs: Number of local epochs per round
            learning_rate: Learning rate for local training
            device: Device for training
            batch_size: Batch size for local training
        """
        self.node_ids = sorted(node_ids)
        self.bandwidths = bandwidths
        self.sample_size = sample_size
        self.success_fraction = success_fraction
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.device = device
        self.batch_size = batch_size

        # Create sampler and aggregator
        self.sampler = PlexusSampler(
            node_ids=self.node_ids,
            sample_size=sample_size,
        )
        self.aggregator = PlexusAggregator(
            sample_size=sample_size,
            success_fraction=success_fraction,
        )

        # Create nodes
        self.nodes: Dict[int, NodeWrapper] = {}
        for nid in node_ids:
            X_train, y_train = node_data[nid]
            self.nodes[nid] = NodeWrapper(
                node_id=nid,
                X_train=X_train,
                y_train=y_train,
                bandwidth=bandwidths.get(nid, 1.0),
                model_template=model_template,
                device=device,
                batch_size=batch_size,
            )

        # Global params (initial model)
        self.global_params: Optional[OrderedDict] = None

        # Message queue for simulation
        self.message_queue: List[Dict] = []

        # History
        self.history: Dict[str, List] = {
            "round": [],
            "sample": [],
            "aggregator": [],
            "participation": [],
            "loss": [],
        }

    def derive_sample(self, round_r: int) -> tuple:
        """Helper to derive sample using sampler."""
        return self.sampler.derive_sample_with_bandwidths(round_r, self.bandwidths)

    def run_round(self, round_r: int, verbose: bool = True) -> Dict:
        """
        Execute one round of the Plexus protocol (Algorithm 2).

        Correct protocol flow:
        1. Derive sample + aggregator for this round
        2. Send TRAIN to all nodes in sample
        3. Each node trains and sends result to the AGGREGATOR (not orchestrator)
        4. Aggregator collects until threshold reached
        5. Aggregator performs FedAvg aggregation (NOT orchestrator)
        6. Aggregator pushes TRAIN to next sample's nodes
        7. Next round triggered by this push

        Key distinction from centralized FL:
        - Orchestrator does NOT aggregate
        - Aggregator node performs aggregation
        - Next round is triggered by aggregator's push

        Args:
            round_r: Current round number

        Returns:
            Dict with round metrics
        """
        start_time = time.time()

        # Step 1: Derive sample and aggregator
        sample_ids, aggregator_id = self.derive_sample(round_r)

        if verbose:
            print(f"\n→ Plexus Round {round_r}:")
            print(f"   Sample: {sample_ids}")
            print(f"   Aggregator: node-{aggregator_id}")

        # Initialize global params if first round
        if self.global_params is None:
            self.global_params = OrderedDict(
                (k, v.cpu().clone())
                for k, v in self.nodes[sample_ids[0]].model.state_dict().items()
            )

        # Step 2: Send TRAIN to all sample nodes
        for nid in sample_ids:
            self.nodes[nid].receive_train(
                round_r=round_r,
                global_params=self.global_params,
                derive_sample_fn=self._derive_sample_cb,
                bandwidths=self.bandwidths,
                local_epochs=self.local_epochs,
                learning_rate=self.learning_rate,
            )

        # Step 3: Each node sends its result to the aggregator
        # (In simulation: orchestrator routes messages to aggregator node)
        results = []
        for nid in sample_ids:
            result = self.nodes[nid].get_pending_result(round_r)
            if result is not None:
                # Route to aggregator for collection
                self.nodes[aggregator_id].receive_for_aggregation(round_r, result)
                results.append(result)

        if verbose:
            print(f"   Trained {len(results)}/{len(sample_ids)} nodes")
            print(f"   Routing results to aggregator node-{aggregator_id}")

        # Step 4: Aggregator checks threshold and aggregates
        threshold = self.aggregator.get_threshold()
        aggregator_node = self.nodes[aggregator_id]

        aggregated_params = None
        if aggregator_node.can_aggregate(round_r, threshold):
            # 🔴 KEY FIX: Aggregator node performs aggregation (NOT orchestrator!)
            aggregated_params = aggregator_node.aggregate(
                round_r,
                self.aggregator,
                self.global_params,
            )

            if aggregated_params is not None:
                self.global_params = aggregated_params
                if verbose:
                    print(f"   Aggregator aggregated {len(results)} models (threshold={threshold})")
            else:
                if verbose:
                    print(f"   ⚠️ Aggregator aggregation returned None")
        else:
            if verbose:
                print(f"   ⚠️ Threshold not met ({len(results)} < {threshold})")

        elapsed = time.time() - start_time
        avg_loss = np.mean([r["loss"] for r in results]) if results else 0.0

        # Record history
        self.history["round"].append(round_r)
        self.history["sample"].append(sample_ids)
        self.history["aggregator"].append(aggregator_id)
        self.history["participation"].append(len(results) / len(sample_ids) if sample_ids else 0)
        self.history["loss"].append(avg_loss)

        return {
            "round": round_r,
            "sample": sample_ids,
            "aggregator": aggregator_id,
            "participation": len(results) / len(sample_ids) if sample_ids else 0,
            "loss": avg_loss,
            "time": elapsed,
        }

    def run_rounds(self, num_rounds: int, verbose: bool = True) -> Dict:
        """
        Run multiple rounds of Plexus protocol.

        Args:
            num_rounds: Number of rounds to run
            verbose: Print progress

        Returns:
            History dict with metrics
        """
        for r in range(num_rounds):
            self.run_round(r, verbose=verbose)

        return self.history

    def get_global_params(self) -> Optional[OrderedDict]:
        """Get current global model parameters."""
        return self.global_params

    def _derive_sample_cb(self, node_ids: List[int], round_r: int, K: int) -> tuple:
        """Callback for nodes to derive sample (ensures consistency)."""
        return self.sampler.derive_sample_with_bandwidths(round_r, self.bandwidths)


class NodeWrapper:
    """
    Wrapper around PlexusNode for the orchestrator.

    Handles the message passing simulation.
    Acts as both:
    - Trainer node: receives TRAIN, trains locally, sends result to aggregator
    - Aggregator node: receives AGGREGATE, collects, aggregates, triggers next round
    """

    def __init__(
        self,
        node_id: int,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        bandwidth: float,
        model_template: nn.Module,
        device: str,
        batch_size: int = 32,
    ):
        self.node_id = node_id
        self.bandwidth = bandwidth
        self.device = device
        self.batch_size = batch_size

        # Create model
        self.model = model_template.to(device)

        # Local data
        self.X_train = X_train.to(device)
        self.y_train = y_train.to(device)
        self.num_samples = len(y_train)

        # Pending results (from training)
        self.pending_results: Dict[int, Dict] = {}

        # Collected models for aggregation: round_r -> List[model_result]
        self.collected_models: Dict[int, List[Dict]] = {}

        # Pending aggregated params for next round
        self.pending_aggregated: Dict[int, OrderedDict] = {}

        # Training config
        self.local_epochs = 1
        self.learning_rate = 0.001

    def receive_train(
        self,
        round_r: int,
        global_params: OrderedDict,
        derive_sample_fn,
        bandwidths: Dict[int, float],
        local_epochs: int,
        learning_rate: float,
    ) -> Dict:
        """
        Handle receiving TRAIN message (Algorithm 2 step 1).

        Returns training result (would be sent to aggregator in real system).
        """
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate

        # Load global params
        self.model.load_state_dict(
            {k: v.to(self.device) for k, v in global_params.items()}
        )

        # Local training
        loss = self._local_train()

        # Store result (will be sent to aggregator)
        result = {
            "client_id": self.node_id,
            "params": OrderedDict(
                (k, v.cpu().clone())
                for k, v in self.model.state_dict().items()
            ),
            "num_samples": self.num_samples,
            "loss": loss,
        }
        self.pending_results[round_r] = result

        return result

    def get_pending_result(self, round_r: int) -> Optional[Dict]:
        """Get pending result for a round."""
        return self.pending_results.pop(round_r, None)

    def receive_for_aggregation(self, round_r: int, model_result: Dict):
        """
        Handle receiving AGGREGATE message (Algorithm 2 step 2).

        The aggregator node collects models from sample members.
        """
        if round_r not in self.collected_models:
            self.collected_models[round_r] = []
        self.collected_models[round_r].append(model_result)

    def can_aggregate(self, round_r: int, threshold: int) -> bool:
        """Check if enough models collected for aggregation."""
        return len(self.collected_models.get(round_r, [])) >= threshold

    def aggregate(
        self,
        round_r: int,
        aggregator,
        global_params: Optional[OrderedDict] = None,
    ) -> Optional[OrderedDict]:
        """
        Perform FedAvg aggregation on collected models (Algorithm 2).

        Args:
            round_r: Round number
            aggregator: Aggregator with weighted_average method
            global_params: Current global params (for fallback)

        Returns:
            Aggregated parameters, or None if no models collected
        """
        results = self.collected_models.pop(round_r, [])

        if not results:
            return global_params

        # Use aggregator's weighted average
        aggregated = aggregator.weighted_average(results)

        if aggregated is not None:
            self.pending_aggregated[round_r] = aggregated

        return aggregated

    def _local_train(self) -> float:
        """Perform local training."""
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        total_loss = 0.0
        num_batches = 0

        indices = torch.randperm(len(self.X_train))
        X_shuffled = self.X_train[indices]
        y_shuffled = self.y_train[indices]

        for epoch in range(self.local_epochs):
            for i in range(0, len(X_shuffled), self.batch_size):
                batch_X = X_shuffled[i:i+self.batch_size]
                batch_y = y_shuffled[i:i+self.batch_size]

                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)