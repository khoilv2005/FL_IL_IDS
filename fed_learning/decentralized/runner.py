"""
PlexusIncrementalRunner - Task loop for decentralized incremental learning.

Reference:
    Dhasade et al., "Practical Federated Learning without a Server",
    EuroMLSys 2025

This runner replaces run_incremental_training() in task_loop.py when
mode="decentralized". It manages the complete incremental learning
pipeline using the Plexus decentralized protocol.

Key differences from centralized FL:
- No FederatedServer - uses PlexusOrchestrator instead
- Each task creates a new set of PlexusNodes
- Persistent state across tasks (knowledge retention)
- Push-based protocol for round-to-round communication
"""

import os
import gc
import json
import random
from collections import OrderedDict
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn

from fed_learning.data.incremental_loader import IncrementalDataLoader
from fed_learning.strategies import get_strategy
from fed_learning.utils.seed import set_seed
from fed_learning.decentralized.node import PlexusNode
from fed_learning.decentralized.orchestrator import PlexusOrchestrator
from fed_learning.decentralized.metrics import PlexusMetrics


class PlexusIncrementalRunner:
    """
    Runner for decentralized incremental learning using Plexus protocol.

    This replaces the centralized FederatedServer + task_loop for
    mode="decentralized". It creates PlexusNode objects for each client
    and coordinates training via PlexusOrchestrator.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Training configuration dict. Required keys:
                - "algorithm": Algorithm name (plexus, fedavg, etc.)
                - "data_dir": Path to data directory
                - "output_dir": Base output directory
                - "total_classes", "base_classes", "classes_per_task": Task structure
                - "rounds_per_task", "local_epochs", "learning_rate", "batch_size"
                - "plexus_sample_size": K (default 13)
                - "plexus_success_fraction": s_f (default 0.8)
        """
        self.config = config
        self.data_loader = IncrementalDataLoader(data_dir=config["data_dir"])

        # Update config with data-derived params
        config["input_shape"] = self.data_loader.input_shape
        config["num_classes"] = config["total_classes"]

        # Strategy (trainer + aggregator)
        self.trainer, self.aggregator = get_strategy(**config)

        # Persistent nodes across tasks
        self.persistent_nodes: Dict[int, PlexusNode] = {}

        # Global model params (persistent across tasks)
        self.global_params: OrderedDict = None

        # Metrics
        self.metrics = PlexusMetrics()

        # Output directory
        self.output_dir = None

    def run(self) -> Dict:
        """
        Execute the complete decentralized incremental learning pipeline.

        Returns:
            Dict with task_accuracies, task_forgetting, and other metrics.
        """
        # Setup
        set_seed(self.config.get("random_seed", 42))

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"{self.config['output_dir']}_decentralized_{self.config['algorithm']}_{ts}"
        os.makedirs(self.output_dir, exist_ok=True)

        # Save config
        with open(f"{self.output_dir}/config.json", "w") as f:
            json.dump(self.config, f, indent=2, default=str)

        print("\n" + "=" * 80)
        print(f"🚀 DECENTRALIZED INCREMENTAL LEARNING - {self.config['algorithm'].upper()}")
        print("=" * 80)
        print(f"📊 Plexus: sample_size={self.config.get('plexus_sample_size', 13)}, "
              f"success_fraction={self.config.get('plexus_success_fraction', 0.8)}")

        all_history = {
            "task_accuracies": [],
            "task_forgetting": [],
            "rounds_info": [],
        }

        num_tasks = self.data_loader.get_num_tasks()

        # Initialize model template for node setup
        model_template = self._create_model_template()

        # Task Loop
        for task_id in range(num_tasks):
            print(
                f"\n{'=' * 80}\n📚 TASK {task_id}/{num_tasks}\n{'=' * 80}"
            )

            # Prepare data for this task
            new_classes = self.data_loader.get_task_classes(task_id)
            seen_classes = []
            for t in range(task_id + 1):
                seen_classes.extend(self.data_loader.get_task_classes(t))

            print(f"  New classes: {new_classes}")
            print(f"  Total seen classes: {len(seen_classes)}")

            # Get client data for this task
            client_data_map = {}
            for cid in self.data_loader.get_all_client_ids():
                X, y = self.data_loader.get_client_data(cid, task_id)
                if len(y) > 0:
                    client_data_map[cid] = {"X_train": X, "y_train": y}

            print(f"  Clients with data: {len(client_data_map)}")

            if not client_data_map:
                print("  ⚠️ No data for this task, skipping.")
                continue

            # Prepare test data for evaluation
            test_X, test_y = self.data_loader.get_test_data(task_id, cumulative=True)
            test_data = {"X_test": test_X, "y_test": test_y}

            # Create/update PlexusNodes for participating clients
            nodes = self._prepare_nodes(
                client_data_map, task_id, new_classes, model_template
            )

            # Create orchestrator for this task
            orchestrator = PlexusOrchestrator(
                nodes=nodes,
                config=self.config,
                trainer=self.trainer,
                aggregator=self.aggregator,
                test_data=test_data,
                model_template=model_template,
            )

            # Initialize global params if not set
            if self.global_params is None:
                orchestrator.setup_models(model_template)
                self.global_params = orchestrator.get_global_params()
            else:
                orchestrator.set_global_params(self.global_params)

            # Set task info on trainer/aggregator
            if hasattr(self.trainer, "set_task"):
                self.trainer.set_task(task_id, new_classes)
            if hasattr(self.aggregator, "set_task"):
                self.aggregator.set_task(task_id)

            # Run Plexus rounds
            print(f"\n🎯 Training on {len(new_classes)} new classes...")
            plexus_rounds = self.config.get("rounds_per_task", 5)
            print(f"  === Plexus Decentralized Training ({plexus_rounds} rounds) ===")

            for r in range(plexus_rounds):
                round_result = orchestrator.run_decentralized_round(r, verbose=True)

                if (r + 1) % self.config.get("eval_every", 1) == 0:
                    eval_metrics = orchestrator.evaluate_global(
                        seen_classes=seen_classes
                    )
                    print(
                        f"    Round {r + 1}/{plexus_rounds} -> "
                        f"Acc: {eval_metrics.get('accuracy', 0) * 100:.2f}%"
                    )

            # Update global params
            self.global_params = orchestrator.get_global_params()

            # Evaluate
            print(f"\n📊 Evaluation:")
            eval_metrics = orchestrator.evaluate_global(
                seen_classes=seen_classes,
                compute_auc=(task_id == num_tasks - 1),
            )
            print(
                f"  Accuracy: {eval_metrics['accuracy'] * 100:.2f}% | "
                f"F1: {eval_metrics['f1_macro'] * 100:.2f}%"
            )

            all_history["task_accuracies"].append(eval_metrics)

            # Track forgetting (requires previous task accuracy)
            if task_id > 0 and len(all_history["task_accuracies"]) > 1:
                # Forgetting = drop in accuracy on previous tasks
                # Simplified: compare current task's accuracy on seen classes
                pass  # Forgetting metric computed post-hoc

            # Save model checkpoint
            self._save_checkpoint(task_id, self.global_params)

            # Cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Final summary
        self._print_summary(all_history)

        # Save results
        with open(f"{self.output_dir}/results.json", "w") as f:
            json.dump(
                {
                    "task_accuracies": [
                        {"accuracy": a["accuracy"], "f1_macro": a["f1_macro"]}
                        for a in all_history["task_accuracies"]
                    ],
                    "plexus_metrics": orchestrator.get_metrics_summary(),
                },
                f,
                indent=2,
            )

        return all_history

    def _create_model_template(self) -> nn.Module:
        """Create a model template for initializing node models."""
        from fed_learning.models.cnn_gru import CNN_GRU_Model

        model = CNN_GRU_Model(
            input_shape=self.config["input_shape"],
            num_classes=self.config["total_classes"],
        )
        return model

    def _prepare_nodes(
        self,
        client_data_map: Dict,
        task_id: int,
        new_classes: List[int],
        model_template: nn.Module,
    ) -> List[PlexusNode]:
        """
        Create or update PlexusNodes for participating clients.

        Args:
            client_data_map: Dict mapping client_id -> {"X_train": Tensor, "y_train": Tensor}.
            task_id: Current task ID.
            new_classes: Classes in this task.
            model_template: Model to use for initializing node local models.

        Returns:
            List of PlexusNode instances.
        """
        nodes = []

        for cid, data in client_data_map.items():
            if cid in self.persistent_nodes:
                # Update existing node with new task data
                node = self.persistent_nodes[cid]
                node.set_task_data(
                    data["X_train"],
                    data["y_train"],
                    task_id,
                    new_classes,
                )
            else:
                # Create new node
                bandwidth = self._generate_bandwidth(cid)

                node = PlexusNode(
                    node_id=cid,
                    X_train=data["X_train"],
                    y_train=data["y_train"],
                    bandwidth=bandwidth,
                )

                # Setup model
                node.setup_model(model_template)

                # Set initial global params if available
                if self.global_params is not None:
                    node.set_local_params(self.global_params)

                self.persistent_nodes[cid] = node

            nodes.append(self.persistent_nodes[cid])

        return nodes

    def _generate_bandwidth(self, client_id: int) -> float:
        """
        Generate simulated bandwidth for a client.

        Uses log-normal distribution as in original Plexus paper.
        """
        rng = random.Random(self.config.get("seed", 42) + client_id)
        return round(rng.lognormvariate(mu=3.0, sigma=0.8), 2)

    def _save_checkpoint(self, task_id: int, params: OrderedDict):
        """Save model checkpoint for a task."""
        checkpoint_path = f"{self.output_dir}/task_{task_id}_model.pt"
        torch.save(params, checkpoint_path)

    def _print_summary(self, history: Dict):
        """Print final summary of training."""
        print("\n" + "=" * 80)
        print("📋 FINAL SUMMARY - DECENTRALIZED PLEXUS")
        print("=" * 80)

        print("\n📊 Task Accuracies:")
        for i, metrics in enumerate(history["task_accuracies"]):
            print(f"  Task {i}: Acc={metrics['accuracy']*100:.2f}%, "
                  f"F1={metrics['f1_macro']*100:.2f}%")

        if len(history["task_accuracies"]) > 1:
            avg_acc = np.mean([m["accuracy"] for m in history["task_accuracies"]])
            print(f"\n  Average Accuracy: {avg_acc*100:.2f}%")

        print(f"\n📁 Results saved to: {self.output_dir}")


def run_decentralized_incremental_training(config: Dict[str, Any]) -> Dict:
    """
    Entry point for decentralized incremental training.

    This function is called from task_loop.py when mode="decentralized".

    Args:
        config: Training configuration dict.

    Returns:
        Dict with training history.
    """
    runner = PlexusIncrementalRunner(config)
    return runner.run()