"""
PlexusNICE Server - Decentralized NICE using Plexus mechanisms.

Reference:
    NICE: Gurbuz, Moorman, Dovrolis (CVPR 2024)
    Plexus: Dhasade et al. (EuroMLSys 2025)

Key adaptations from NICEServer -> PlexusNICEServer:
1. Uses SampleManager for peer sampling
2. Neuron ages synchronized via piggybacked state exchange
3. Context detection replaced with distributed approach (round-counting)
4. end_task() logic replaced with protocol-based phase transition
"""

import time
from collections import OrderedDict
from math import floor
from threading import Thread
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .plexus_server import PlexusServer
from ..models.nice_model import NICEModel
from ..strategies.federated.plexus_nice import (
    PlexusNICETrainer,
    PlexusNICEAggregator,
    DistributedContextDetector,
)
from ..strategies.incremental.nice import increase_unit_ranks, update_freeze_masks


class PlexusNICEServer(PlexusServer):
    """
    Server that simulates decentralized NICE using Plexus protocol.

    Inherits from PlexusServer:
    - SampleManager, PopulationView, bandwidth-based aggregator selection
    - Success-fraction filtering

    NICE-specific:
    - Uses NICEModel (fixed architecture, no expansion)
    - Neuron age state piggybacked on model transfers
    - Freeze mask state piggybacked on model transfers
    - DistributedContextDetector using round-counting proxy

    Unlike centralized NICEServer:
    - No centralized context detector training
    - Neuron ages managed via piggyback protocol
    - end_task() called by simulation (not automatically by training loop)
    """

    def __init__(self, clients, test_data: Dict, config: Dict):
        # Initialize PlexusServer first (sets up sampling infrastructure)
        super().__init__(clients, test_data, config)

        # Replace Plexus global model with NICEModel
        del self.global_model
        self.global_model = NICEModel(config["input_shape"], config["num_classes"]).to(
            self.primary_device
        )

        # Replace Plexus trainer/aggregator with NICE-specific
        self.trainer = PlexusNICETrainer(
            tau=config.get("tau", 0.95),
            max_phases=config.get("nice_max_phases", 5),
            phase_epochs=config.get("nice_phase_epochs", 5),
            memo_per_class=config.get("memo_per_class", 50),
            rounds_per_task=config.get("rounds_per_task", 5),
        )

        self.aggregator = PlexusNICEAggregator(
            sample_size=self.sample_size,
            num_aggregators=self.num_aggregators,
            success_fraction=self.success_fraction,
            inactivity_threshold=self.inactivity_threshold,
            client_bandwidths=self.client_bandwidths,
        )

        # Distributed context detector (uses round-counting proxy)
        self.context_detector = DistributedContextDetector(
            rounds_per_task=config.get("rounds_per_task", 5)
        )

        print(f"📊 Strategy: PlexusNICE (Decentralized NICE)")
        print(f"  sample_size={self.sample_size}, success_fraction={self.success_fraction}")

    def _get_frozen_param_keys(self) -> List[str]:
        """
        Find parameter keys that can be fully frozen because all neurons in
        the layer are mature (age >= 2).

        Returns actual parameter keys like 'conv1.weight', not layer names.
        """
        frozen_keys = []
        for name, param in self.global_model.named_parameters():
            layer_name = name.split(".")[0]
            if layer_name in self.global_model.unit_ranks:
                ranks = self.global_model.unit_ranks[layer_name]
                if np.all(ranks >= 2):
                    frozen_keys.append(name)
        return frozen_keys

    def set_task(self, task_id: int, task_classes: list, seen_classes: list = None):
        """Set up for new task - set output neuron ages for new classes."""
        super().set_task(task_id, task_classes, seen_classes)

        # Set output neuron ages to 1 (learner) for new classes
        for cls_id in task_classes:
            if cls_id < self.global_model.num_classes:
                self.global_model.unit_ranks["fc2"][cls_id] = 1

        # Store episode classes for context detector
        self.context_detector.set_episode_classes(task_id, list(task_classes))

        # Update aggregator with neuron ages and freeze masks
        self.aggregator.set_neuron_ages(self.global_model.get_neuron_ages_state())
        self.aggregator.set_freeze_masks(self.global_model.freeze_masks)
        self.aggregator.set_frozen_keys(self._get_frozen_param_keys())

        # Update trainer task state
        if hasattr(self.trainer, "set_task"):
            self.trainer.set_task(task_id, task_classes)

        print(f"  PlexusNICE: Task {task_id} | new classes: {task_classes}")
        for name in ["conv1", "fc1", "fc2"]:
            ranks = self.global_model.unit_ranks[name]
            print(
                f"    {name}: young={np.sum(ranks == 0)}, "
                f"learner={np.sum(ranks == 1)}, "
                f"mature={np.sum(ranks >= 2)}"
            )

    def train_round(
        self,
        participating_clients=None,
        verbose: bool = True,
        **kwargs,
    ) -> Dict:
        """
        Execute one PlexusNICE round.

        This is a SAMPLED DECENTRALIZED SIMULATION: the server orchestrates
        the protocol but the aggregator role is determined by the Plexus protocol
        (highest-bandwidth node in the hash-ordered sample).

        Protocol:
        1. Determine sample and aggregators (via SampleManager)
        2. Train sampled clients (multi-GPU)
        3. Apply success-fraction filtering
        4. Aggregate (with NICE mature-neuron restoration)
        5. Distribute model + population view + NICE state
        """
        from ..training.plexus_nice_worker import train_plexus_nice_clients_on_gpu

        round_start = time.time()
        self._round += 1
        self.aggregator.current_round = self._round

        all_ids = [c.client_id for c in self.clients]
        client_map = {c.client_id: c for c in self.clients}

        # Determine candidate_ids: use participating_clients if provided, else all_ids
        if participating_clients is not None:
            candidate_ids = [c.client_id for c in participating_clients]
        else:
            candidate_ids = all_ids

        # Determine sample and aggregators from candidate_ids (Plexus mechanism)
        aggregator_ids = self.aggregator.get_round_aggregators(self._round, candidate_ids)
        sample_ids = self.aggregator.get_round_sample(self._round, candidate_ids)

        # Fallback if sample too small
        if len(sample_ids) < 3:
            sample_ids = candidate_ids

        # Ensure aggregator is in final sample; if not, pick highest-bandwidth from sample
        self.selected_aggregator_id = aggregator_ids[0] if aggregator_ids else None
        if self.selected_aggregator_id is not None and self.selected_aggregator_id not in sample_ids:
            # Pick highest-bandwidth node from sample_ids as aggregator
            sample_bandwidths = {sid: self.client_bandwidths.get(sid, 0) for sid in sample_ids}
            self.selected_aggregator_id = max(sample_bandwidths, key=sample_bandwidths.get)

        sampled_clients = [client_map[sid] for sid in sample_ids if sid in client_map]

        is_last_task = kwargs.get("is_last_task", False)

        if verbose:
            device_info = "CPU" if self.use_cpu else f"{self.num_gpus} GPU(s)"
            print(
                f"\n→ PlexusNICE Round {self._round}: "
                f"sample={len(sampled_clients)}/{len(all_ids)}, "
                f"aggregator=node-{self.selected_aggregator_id}, "
                f"device={device_info}"
            )

        global_params = self.get_global_params()

        # Prepare config with neuron ages and masks
        worker_config = {**self.config}
        worker_config["neuron_ages"] = self.global_model.get_neuron_ages_state()
        worker_config["masks"] = self.global_model.get_masks_state()
        worker_config["freeze_masks"] = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in self.global_model.freeze_masks.items()
        }
        worker_config["is_last_task"] = is_last_task

        # Distribute clients across GPUs
        clients_per_gpu = [[] for _ in range(self.num_gpus)]
        for i, c in enumerate(sampled_clients):
            clients_per_gpu[i % self.num_gpus].append(c)

        # Train clients
        results_dict: Dict = {}
        threads = []

        for gpu_id in range(self.num_gpus):
            if len(clients_per_gpu[gpu_id]) > 0:
                t = Thread(
                    target=train_plexus_nice_clients_on_gpu,
                    args=(
                        gpu_id,
                        clients_per_gpu[gpu_id],
                        global_params,
                        worker_config,
                        results_dict,
                        self.trainer,
                        self.use_cpu,
                    ),
                )
                threads.append(t)
                t.start()

        for t in threads:
            t.join()

        results = list(results_dict.values())

        # Success-fraction filtering
        n_required = max(3, floor(len(results) * self.success_fraction))
        used_results = results[:n_required] if len(results) > n_required else results

        if verbose:
            print(
                f"   Aggregating {len(used_results)}/{len(results)} models "
                f"(success_fraction={self.success_fraction})"
            )

        # Aggregate
        if used_results:
            new_params = self.aggregator.aggregate(used_results, global_params)
            self.global_model.load_state_dict(
                {k: v.to(self.primary_device) for k, v in new_params.items()}
            )

        # Update server model's neuron ages from used_results (same clients that contributed weights)
        # Must use used_results to keep weight state and age state in sync
        if used_results:
            for r in used_results:
                if "neuron_ages" in r and r["neuron_ages"]:
                    self.global_model.set_neuron_ages_state(r["neuron_ages"])
                    break
        # Refresh freeze-related state in aggregator to keep it consistent
        self.aggregator.set_neuron_ages(self.global_model.get_neuron_ages_state())
        self.aggregator.set_freeze_masks(self.global_model.freeze_masks)
        self.aggregator.set_frozen_keys(self._get_frozen_param_keys())

        # Update population view
        for r in results:
            cid = r.get("client_id", -1)
            self.population_view.update(cid, self._round, is_online=True)
            self.aggregator.population_view.update(cid, self._round, is_online=True)

        # Distribute to sampled clients
        for c in sampled_clients:
            if hasattr(c, "receive_aggregated_model"):
                c.receive_aggregated_model(self._round)
            if hasattr(c, "merge_population_view"):
                c.merge_population_view(self.population_view)
            # NICE-specific: piggyback neuron ages and freeze masks
            if hasattr(c, "receive_neuron_ages"):
                c.receive_neuron_ages(self.global_model.get_neuron_ages_state())
            if hasattr(c, "receive_freeze_masks"):
                c.receive_freeze_masks(self.global_model.freeze_masks)

        avg_loss = float(np.mean([r["loss"] for r in results])) if results else 0.0
        round_time = time.time() - round_start

        if verbose:
            print(f"  → NICE loss: {avg_loss:.4f} ({round_time:.1f}s)")
            for name in ["conv1", "fc1", "fc2"]:
                ranks = self.global_model.unit_ranks[name]
                print(
                    f"    {name}: young={np.sum(ranks == 0)}, "
                    f"learner={np.sum(ranks == 1)}, "
                    f"mature={np.sum(ranks >= 2)}"
                )

        return {"train_loss": avg_loss, "round_time": round_time}

    def end_task(self):
        """
        End-of-task processing via protocol.

        In decentralized setting, this is called by the simulation
        after all training rounds for a task are complete.

        Operations:
        1. Increase unit ranks (learner -> mature)
        2. Update freeze masks
        3. Freeze BN for mature layers
        """
        print(f"\n  PlexusNICE end_task({self.current_task}):")

        # Age transition
        increase_unit_ranks(self.global_model)

        # Print age stats
        for name in self.global_model.LAYER_NAMES:
            ranks = self.global_model.unit_ranks[name]
            print(
                f"    {name}: young={np.sum(ranks == 0)}, "
                f"learner={np.sum(ranks == 1)}, "
                f"mature={np.sum(ranks >= 2)}"
            )

        # Update freeze masks
        update_freeze_masks(self.global_model)

        # Freeze BN for mature layers
        self.global_model.freeze_bn_for_mature()

        # Update aggregator state
        self.aggregator.set_neuron_ages(self.global_model.get_neuron_ages_state())
        self.aggregator.set_freeze_masks(self.global_model.freeze_masks)
        self.aggregator.set_frozen_keys(self._get_frozen_param_keys())

    def evaluate_global(
        self,
        batch_size: int = 1024,
        compute_auc: bool = False,
        seen_classes_only: bool = True,
    ) -> Dict:
        """Evaluate with output masking for unseen classes (same as NICEServer)."""
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
        )

        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()

        X_test = self.test_data["X_test"]
        y_test = self.test_data["y_test"]

        # Filter test data to seen classes
        if seen_classes_only and self.seen_classes:
            seen_set = set(self.seen_classes)
            mask = torch.tensor([y.item() in seen_set for y in y_test])
            X_test = X_test[mask]
            y_test = y_test[mask]

        n_test = len(y_test)
        if n_test == 0:
            return {"loss": 0.0, "accuracy": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0}

        # Build unseen class mask for output masking
        seen_set = (
            set(self.seen_classes)
            if self.seen_classes
            else set(range(self.global_model.num_classes))
        )
        unseen_mask = torch.ones(self.global_model.num_classes, dtype=torch.bool)
        for c in seen_set:
            unseen_mask[c] = False
        unseen_mask = unseen_mask.to(self.primary_device)

        all_preds = []
        all_targets = []
        total_loss = 0.0

        with torch.no_grad():
            for i in range(0, n_test, batch_size):
                X_batch = X_test[i : i + batch_size].to(self.primary_device)
                y_batch = y_test[i : i + batch_size].to(self.primary_device)

                out = self.global_model(X_batch)
                # Mask unseen class logits to -inf
                out[:, unseen_mask] = float("-inf")

                loss = criterion(out, y_batch)
                total_loss += loss.item() * len(y_batch)

                preds = out.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)

        return {
            "loss": total_loss / n_test,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
        }

    def compute_average_forgetting(self) -> float:
        """Compute Average Forgetting of NICE based on per-task accuracy."""
        if self.current_task == 0:
            return 0.0
        current_accs = self.evaluate_per_task()
        if hasattr(self.trainer, "update_forgetting"):
            self.trainer.update_forgetting(current_accs)
            return self.trainer.last_af
        return 0.0

    def evaluate_per_task(self, batch_size: int = 1024) -> Dict[int, float]:
        """Evaluate accuracy per task with output masking."""
        from sklearn.metrics import accuracy_score

        self.global_model.eval()
        task_accuracies = {}

        X_test = self.test_data["X_test"]
        y_test = self.test_data["y_test"]

        # Mask unseen class logits
        seen_set = (
            set(self.seen_classes)
            if self.seen_classes
            else set(range(self.global_model.num_classes))
        )
        unseen_mask = torch.ones(self.global_model.num_classes, dtype=torch.bool)
        for c in seen_set:
            unseen_mask[c] = False
        unseen_mask = unseen_mask.to(self.primary_device)

        for task_id, task_classes in self.task_classes.items():
            if not task_classes:
                continue

            task_class_set = set(task_classes)
            mask = torch.tensor([y.item() in task_class_set for y in y_test])

            if not mask.any():
                task_accuracies[task_id] = 0.0
                continue

            X_task = X_test[mask]
            y_task = y_test[mask]

            all_preds = []
            all_targets = []

            with torch.no_grad():
                for i in range(0, len(y_task), batch_size):
                    X_batch = X_task[i : i + batch_size].to(self.primary_device)
                    y_batch = y_task[i : i + batch_size]

                    out = self.global_model(X_batch)
                    out[:, unseen_mask] = float("-inf")
                    preds = out.argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(y_batch.numpy())

            task_accuracies[task_id] = accuracy_score(all_targets, all_preds)

        return task_accuracies
