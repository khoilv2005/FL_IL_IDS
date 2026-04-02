"""
PlexusOrchestrator - Coordinates the decentralized Plexus protocol.

Reference:
    Dhasade et al., "Practical Federated Learning without a Server",
    EuroMLSys 2025, Algorithm 2 (Push-Based Protocol)

The orchestrator simulates the fully decentralized Plexus system where:
1. Each round, a sample of K nodes is selected via consistent hashing
2. Within the sample, the highest-bandwidth node is elected aggregator
3. Only sampled nodes train locally (not all nodes)
4. When aggregator receives K * success_fraction models, it aggregates
5. Aggregator sends TRAIN to the NEXT round's sample (push-based)
6. This triggers the next round without a central coordinator

This replaces the centralized FederatedServer for decentralized mode.
"""

import copy
import time
from collections import OrderedDict
from math import floor
from threading import Thread
from typing import Dict, List, Optional, Any

import numpy as np
import torch

from .sampler import PlexusSampler
from .node import PlexusNode
from .metrics import PlexusMetrics


class PlexusOrchestrator:
    """
    Coordinates the Plexus decentralized protocol across nodes.

    Key differences from FederatedServer:
    - No central aggregation - aggregator role rotates among nodes
    - Only K nodes train per round (hash-based sampling)
    - Push-based: aggregator triggers next round by sending to next sample
    - Success fraction allows aggregation with partial participation

    The orchestrator simulates this on a single machine by:
    - Creating PlexusNode objects for each client
    - Managing the sample selection via PlexusSampler
    - Routing messages between nodes in the simulation
    """

    def __init__(
        self,
        nodes: List[PlexusNode],
        config: Dict,
        trainer: Any,
        aggregator: Any,
        test_data: Optional[Dict] = None,
        model_template: Optional[torch.nn.Module] = None,
    ):
        """
        Args:
            nodes: List of PlexusNode instances.
            config: Training configuration with plexus_* keys.
            trainer: Training strategy (e.g., PlexusTrainer).
            aggregator: Aggregation strategy (e.g., PlexusAggregator).
        """
        self.nodes = nodes
        self.node_map: Dict[int, PlexusNode] = {n.node_id: n for n in nodes}
        self.config = config
        self.trainer = trainer
        self.aggregator = aggregator

        # Plexus parameters from config
        self.sample_size = config.get("plexus_sample_size", 13)
        self.num_aggregators = config.get("plexus_num_aggregators", 1)
        self.success_fraction = config.get("plexus_success_fraction", 0.8)
        self.local_epochs = config.get("local_epochs", 1)

        # Create sampler
        node_ids = [n.node_id for n in nodes]
        self.sampler = PlexusSampler(node_ids, self.sample_size)

        # Bandwidths for aggregator selection
        self.bandwidths = {n.node_id: n.bandwidth for n in nodes}

        # Global parameters (current best model)
        self.global_params: Optional[OrderedDict] = None

        # Round counter
        self.current_round: int = 0

        # Metrics tracking
        self.metrics = PlexusMetrics()

        # Test data for evaluation
        self.test_data = test_data
        self.model_template = model_template

        # History
        self.history: Dict[str, List] = {
            "train_loss": [],
            "sample_sizes": [],
            "aggregator_ids": [],
            "participation_rates": [],
        }

    def set_global_params(self, params: OrderedDict):
        """Set the global model parameters."""
        self.global_params = params

        # Propagate to all nodes so they can load for training
        for node in self.nodes:
            if node.local_model is not None:
                node.set_local_params(params)

    def get_global_params(self) -> Optional[OrderedDict]:
        """Get current global parameters."""
        return self.global_params

    def setup_models(self, model_template: torch.nn.Module):
        """Initialize local models on all nodes."""
        for node in self.nodes:
            node.setup_model(model_template)

    def run_decentralized_round(self, round_r: int, verbose: bool = True) -> Dict:
        """
        Execute one complete Plexus round (Algorithm 2 from paper).

        Protocol:
        1. Derive sample for this round via consistent hashing
        2. Elect aggregator (highest bandwidth in sample)
        3. Train all nodes in sample in parallel
        4. Apply success fraction threshold
        5. Aggregator performs FedAvg
        6. Push aggregated model to next round's sample
        7. Return updated global params

        Args:
            round_r: Round number.
            verbose: Whether to print progress.

        Returns:
            Dict with round metrics.
        """
        round_start = time.time()
        self.current_round = round_r

        # Step 1: Derive sample for this round
        sample_ids, aggregator_id = self.sampler.derive_sample_with_bandwidths(
            round_r, self.bandwidths
        )

        if verbose:
            agg_bw = self.bandwidths.get(aggregator_id, 0)
            print(
                f"\n→ Plexus Round {round_r}: "
                f"sample_size={len(sample_ids)}, "
                f"aggregator={aggregator_id} (bw={agg_bw:.2f})"
            )

        # Get sample nodes
        sample_nodes = [self.node_map[sid] for sid in sample_ids if sid in self.node_map]
        aggregator_node = self.node_map.get(aggregator_id)

        if not sample_nodes or aggregator_node is None:
            if verbose:
                print(f"   ⚠️ No sample nodes or aggregator found, skipping round")
            return {"train_loss": 0.0, "round_time": 0.0}

        # Step 2: Parallel local training
        results = self._parallel_train(sample_nodes, round_r)

        if verbose:
            print(f"   Trained {len(results)}/{len(sample_nodes)} nodes")

        # Step 3: Success fraction filtering
        # Paper: proceed when K * success_fraction models received
        threshold = max(3, floor(len(sample_nodes) * self.success_fraction))

        if len(results) < threshold:
            if verbose:
                print(
                    f"   ⚠️ Not enough results ({len(results)}) for threshold ({threshold}), "
                    f"using all available"
                )

        used_results = results[:threshold] if len(results) > threshold else results

        # Step 4: Send results to aggregator for collection
        for result in used_results:
            if aggregator_node is not None:
                aggregator_node.receive_for_aggregation(round_r, result)

        # Step 5: Aggregator performs aggregation
        if aggregator_node is not None and aggregator_node.can_aggregate(round_r, threshold):
            new_params = aggregator_node.aggregate(
                round_r,
                self.aggregator,
                self.global_params
            )

            if new_params is not None:
                self.global_params = new_params

                # Update population view
                for result in used_results:
                    cid = result.get("client_id", -1)
                    if cid in self.node_map:
                        self.node_map[cid].receive_aggregated_model(round_r)
                        self.node_map[cid].merge_population_view(
                            aggregator_node.get_population_view()
                        )

            if verbose:
                print(f"   Aggregator {aggregator_id} completed aggregation")

        # Step 6: PUSH to next round's sample (key Plexus innovation!)
        # The aggregator sends TRAIN to the next sample to trigger round r+1
        next_sample_info = self.sampler.get_sample_for_next_round(
            round_r, self.bandwidths
        )
        next_sample_ids = next_sample_info["next_sample_ids"]

        if verbose and next_sample_ids != sample_ids:
            print(f"   Next sample: {len(next_sample_ids)} nodes will be triggered")

        # Metrics
        avg_loss = float(np.mean([r["loss"] for r in results])) if results else 0.0
        round_time = time.time() - round_start

        self.history["train_loss"].append(avg_loss)
        self.history["sample_sizes"].append(len(sample_ids))
        self.history["aggregator_ids"].append(aggregator_id)
        self.history["participation_rates"].append(len(results) / len(sample_ids))

        self.metrics.record_round(
            round_r=round_r,
            sample_size=len(sample_ids),
            participation=len(results) / len(sample_ids) if sample_ids else 0,
            aggregator_id=aggregator_id,
        )

        return {
            "train_loss": avg_loss,
            "round_time": round_time,
            "sample_size": len(sample_ids),
            "aggregator_id": aggregator_id,
        }

    def _parallel_train(
        self,
        nodes: List[PlexusNode],
        round_r: int,
    ) -> List[Dict]:
        """
        Train nodes in parallel across available GPUs.

        Uses threading to simulate parallel training on different devices.
        """
        results: Dict[int, Dict] = {}
        threads = []

        num_gpus = self.config.get("num_gpus", 1)

        # Distribute nodes across GPUs
        nodes_per_gpu = [[] for _ in range(num_gpus)]
        for i, node in enumerate(nodes):
            nodes_per_gpu[i % num_gpus].append(node)

        def train_on_gpu(gpu_id: int, gpu_nodes: List[PlexusNode]):
            device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
            for node in gpu_nodes:
                node.device = device
                try:
                    result = node.train_local(
                        global_params=self.global_params,
                        trainer=self.trainer,
                        local_epochs=self.local_epochs,
                    )
                    results[node.node_id] = result
                except Exception as e:
                    print(f"   ⚠️ Node {node.node_id} training failed: {e}")

        # Start threads
        for gpu_id in range(num_gpus):
            if nodes_per_gpu[gpu_id]:
                t = Thread(
                    target=train_on_gpu,
                    args=(gpu_id, nodes_per_gpu[gpu_id])
                )
                threads.append(t)
                t.start()

        # Wait for completion
        for t in threads:
            t.join()

        return list(results.values())

    def run_multi_round(self, num_rounds: int, verbose: bool = True) -> Dict:
        """
        Run multiple Plexus rounds.

        Args:
            num_rounds: Number of rounds to run.
            verbose: Whether to print progress.

        Returns:
            History dict with metrics from all rounds.
        """
        for r in range(num_rounds):
            self.run_decentralized_round(r, verbose=verbose)

            if verbose and (r + 1) % self.config.get("eval_every", 1) == 0:
                print(f"   Round {r + 1}/{num_rounds} completed")

        return self.history

    def evaluate_global(
        self,
        seen_classes: Optional[List[int]] = None,
        compute_auc: bool = False,
    ) -> Dict:
        """
        Evaluate the global model on test data.

        Args:
            seen_classes: List of class IDs that have been seen (for filtering).
            compute_auc: Whether to compute AUC (expensive, done on last task).

        Returns:
            Dict with accuracy, f1_macro, f1_weighted, loss.
        """
        import torch.nn as nn
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        if self.global_params is None:
            return {"accuracy": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0, "loss": 0.0}

        if self.test_data is None:
            return {"accuracy": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0, "loss": 0.0}

        # Create model and load global params
        if self.model_template is None:
            return {"accuracy": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0, "loss": 0.0}

        model = copy.deepcopy(self.model_template)
        model.load_state_dict(
            {k: v.cpu() for k, v in self.global_params.items()}
        )
        model.eval()

        X_test = self.test_data["X_test"]
        y_test = self.test_data["y_test"]

        # Filter to seen classes if provided
        if seen_classes:
            seen_set = set(seen_classes)
            mask = torch.tensor([y.item() in seen_set for y in y_test])
            X_test = X_test[mask]
            y_test = y_test[mask]

        n_test = len(y_test)
        if n_test == 0:
            return {"accuracy": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0, "loss": 0.0}

        # Evaluate
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for i in range(0, len(X_test), 1024):
                batch_X = X_test[i:i+1024]
                batch_y = y_test[i:i+1024]

                output = model(batch_X)
                loss = criterion(output, batch_y)
                total_loss += loss.item() * len(batch_y)

                preds = output.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_y.numpy())

        avg_loss = total_loss / n_test
        accuracy = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

        return {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "loss": avg_loss,
        }

    def get_metrics_summary(self) -> Dict:
        """Get summary of Plexus-specific metrics."""
        return self.metrics.get_summary()

    def __repr__(self) -> str:
        return (
            f"PlexusOrchestrator("
            f"nodes={len(self.nodes)}, "
            f"sample_size={self.sample_size}, "
            f"success_fraction={self.success_fraction})"
        )