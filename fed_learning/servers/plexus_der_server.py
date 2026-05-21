"""
PlexusDER Server - Decentralized DER using Plexus mechanisms.

Reference:
    DER: Yan, Xie, He (CVPR 2021)
    Plexus: Dhasade et al. (EuroMLSys 2025)

Key adaptations from DERServer -> PlexusDERServer:
1. Uses SampleManager for peer sampling (no central coordination)
2. Uses PopulationView for peer state tracking (piggybacked)
3. Task-boundary protocol: model expansion via piggybacked task_classes_history
4. Exemplar coordination via piggybacked metadata
5. Success-fraction filtering for aggregation liveness
"""

import random
import time
from collections import OrderedDict
from math import floor
from threading import Thread
from typing import Dict, List, Optional

import numpy as np
import torch

from .plexus_server import PlexusServer
from ..models.der_model import DERModel
from ..training.der_worker import _reconstruct_model_structure
from ..strategies.federated.plexus_der import PlexusDERTrainer, PlexusDERAggregator


class PlexusDERServer(PlexusServer):
    """
    Server that simulates decentralized DER using Plexus protocol.

    Inherits from PlexusServer:
    - SampleManager, PopulationView, bandwidth-based aggregator selection
    - Hash-ordered sample ordering
    - Success-fraction filtering

    DER-specific overrides:
    - set_task(): model expansion via piggybacked task_classes_history
    - train_round(): two-stage training support
    - Exemplar coordination via piggybacked metadata
    - Uses DERModel instead of CNN_GRU_Model

    Task-Boundary Protocol:
    - Model structure (num_extractors, task_classes_history) is piggybacked
      on model transfers, allowing late-joining peers to reconstruct
    - All peers expand model at the same protocol-defined rounds
    """

    def __init__(self, clients, test_data: Dict, config: Dict):
        # Initialize PlexusServer first (sets up sampling infrastructure)
        super().__init__(clients, test_data, config)

        # Replace Plexus global model with DERModel
        del self.global_model
        self.global_model = DERModel(config["input_shape"], config["num_classes"]).to(
            self.primary_device
        )

        # Replace Plexus trainer/aggregator with DER-specific
        self.trainer = PlexusDERTrainer(
            lambda_aux=config.get("lambda_aux", 1.0),
            lambda_sparsity=config.get("lambda_sparsity", 0.5),
            s_max=config.get("s_max", 15.0),
            temperature=config.get("der_temperature", 2.0),
            buffer_size=config.get("buffer_size", 500),
        )

        self.aggregator = PlexusDERAggregator(
            sample_size=self.sample_size,
            num_aggregators=self.num_aggregators,
            success_fraction=self.success_fraction,
            inactivity_threshold=self.inactivity_threshold,
            client_bandwidths=self.client_bandwidths,
        )

        # DER-specific: task classes history for model reconstruction
        self._task_classes_history: Dict[int, List[int]] = {}

        print(f"📊 Strategy: PlexusDER (Decentralized DER)")
        print(f"  sample_size={self.sample_size}, success_fraction={self.success_fraction}")

    def set_task(self, task_id: int, task_classes: list, seen_classes: list = None):
        """Set up for new task using piggybacked model structure."""
        # Parent handles: task tracking, seen_classes, population view init
        super().set_task(task_id, task_classes, seen_classes)

        # Store task classes history for model reconstruction
        self._task_classes_history[task_id] = list(task_classes)

        # Expand DERModel: add new extractor, expand classifier
        s_max = self.config.get("s_max", 15.0)
        self.global_model.add_task(task_classes, s_max=s_max)

        # Update aggregator with model structure
        self.aggregator.set_model_structure(
            self.global_model.num_extractors,
            self._task_classes_history
        )

        # Derive trainable keys from current model structure
        trainable_keys = self.aggregator.derive_trainable_keys(
            self.global_model.state_dict()
        )

        print(f"  PlexusDER: Task {task_id} | extractors={self.global_model.num_extractors}")

    def set_global_params(self, params: OrderedDict):
        """Load params with model structure reconstruction (same as DERServer)."""
        # Count how many extractors are in params
        extractor_indices = {
            int(k.split(".")[1])
            for k in params
            if k.startswith("extractors.") and k.split(".")[1].isdigit()
        }
        num_tasks_in_params = len(extractor_indices)

        # Reconstruct model structure if it doesn't match params
        if self.global_model.num_extractors != num_tasks_in_params:
            config_for_recon = {
                "input_shape": self.config["input_shape"],
                "num_classes": self.config["num_classes"],
                "s_max": self.config.get("s_max", 15.0),
                "task_classes_history": self._task_classes_history,
            }
            _reconstruct_model_structure(
                self.global_model, OrderedDict(params), config_for_recon
            )

        # Now safe to load
        self.global_model.load_state_dict(
            {k: v.to(self.primary_device) for k, v in params.items()}
        )

    def train_round(
        self,
        participating_clients=None,
        stage: int = 1,
        verbose: bool = True,
        **kwargs,
    ) -> Dict:
        """
        Execute one PlexusDER round with two-stage support.

        This is a SAMPLED DECENTRALIZED SIMULATION: the server orchestrates
        the protocol but the aggregator role is determined by the Plexus protocol
        (highest-bandwidth node in the hash-ordered sample).

        Protocol:
        1. Determine sample and aggregators (via SampleManager)
        2. Train sampled clients (multi-GPU)
        3. Apply success-fraction filtering
        4. Aggregate (with DER frozen-param restoration)
        5. Distribute model + population view + DER state
        """
        from ..training.plexus_der_worker import train_plexus_der_clients_on_gpu

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

        # --- 1. Determine sample and aggregators from candidate_ids (Plexus mechanism) ---
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

        if verbose:
            stage_name = "Representation" if stage == 1 else "Classifier"
            device_info = "CPU" if self.use_cpu else f"{self.num_gpus} GPU(s)"
            print(
                f"\n→ PlexusDER Round {self._round} (Stage {stage}): "
                f"sample={len(sampled_clients)}/{len(all_ids)}, "
                f"aggregator=node-{self.selected_aggregator_id}, "
                f"stage={stage_name}, device={device_info}"
            )

        global_params = self.get_global_params()

        # Prepare config with task history for model reconstruction
        worker_config = {**self.config}
        worker_config["task_classes_history"] = self._task_classes_history

        # Distribute clients across GPUs
        clients_per_gpu = [[] for _ in range(self.num_gpus)]
        for i, c in enumerate(sampled_clients):
            clients_per_gpu[i % self.num_gpus].append(c)

        # Train clients in parallel
        results_dict: Dict = {}
        threads = []

        for gpu_id in range(self.num_gpus):
            if len(clients_per_gpu[gpu_id]) > 0:
                t = Thread(
                    target=train_plexus_der_clients_on_gpu,
                    args=(
                        gpu_id,
                        clients_per_gpu[gpu_id],
                        global_params,
                        worker_config,
                        results_dict,
                        self.trainer,
                        self.use_cpu,
                        stage,
                    ),
                )
                threads.append(t)
                t.start()

        for t in threads:
            t.join()

        results = list(results_dict.values())

        # --- 2. Success-fraction filtering ---
        n_required = max(3, floor(len(results) * self.success_fraction))
        used_results = results[:n_required] if len(results) > n_required else results

        # --- 3. Aggregate (with DER frozen-param restoration) ---
        if used_results:
            new_params = self.aggregator.aggregate(used_results, global_params)
            self.set_global_params(new_params)

        # --- 4. Update population view ---
        for r in results:
            cid = r.get("client_id", -1)
            self.population_view.update(cid, self._round, is_online=True)
            self.aggregator.population_view.update(cid, self._round, is_online=True)

        # --- 5. Distribute to sampled clients ---
        for c in sampled_clients:
            if hasattr(c, "receive_aggregated_model"):
                c.receive_aggregated_model(self._round)
            if hasattr(c, "merge_population_view"):
                c.merge_population_view(self.population_view)
            # DER-specific: piggyback task classes history
            if hasattr(c, "receive_task_history"):
                c.receive_task_history(self._task_classes_history)

        avg_loss = float(np.mean([r["loss"] for r in results])) if results else 0.0
        round_time = time.time() - round_start

        if verbose:
            print(f"  → Stage {stage} loss: {avg_loss:.4f} ({round_time:.1f}s)")
            total_replay = sum(r.get("replay_samples", 0) for r in results)
            if total_replay > 0:
                print(f"  → Replay samples: {total_replay}")

        return {"train_loss": avg_loss, "round_time": round_time}

    def coordinate_exemplar_update(
        self,
        participating_clients=None,
        verbose: bool = True,
    ):
        """
        Coordinate exemplar buffer updates via piggybacked metadata exchange.

        In decentralized setting, each client independently updates its buffer
        based on the global model (received via aggregation).
        """
        clients = participating_clients or self.clients

        if verbose:
            print(f"\n📸 PlexusDER: Exemplar update coordination")

        for idx, client in enumerate(clients):
            if not hasattr(client, "update_exemplars"):
                continue
            if client.num_samples == 0:
                continue

            client.update_exemplars(self.global_model)

        if verbose:
            total_buffer = sum(
                c.replay_buffer.total_samples
                for c in clients
                if hasattr(c, "replay_buffer")
            )
            print(f"   Exemplar update complete. Total buffer: {total_buffer}")

    def compute_average_forgetting(self) -> float:
        """Compute Average Forgetting of DER based on per-task accuracy."""
        if self.current_task == 0:
            return 0.0
        current_accs = self.evaluate_per_task()
        if hasattr(self.trainer, "update_forgetting"):
            self.trainer.update_forgetting(current_accs)
            return self.trainer.last_af
        return 0.0

    def evaluate_per_task(self, batch_size: int = 8192) -> Dict[int, float]:
        """Evaluate accuracy per task."""
        from sklearn.metrics import accuracy_score

        self.global_model.eval()
        task_accuracies = {}

        X_test = self.test_data["X_test"]
        y_test = self.test_data["y_test"]

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
                    preds = out.argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(y_batch.numpy())

            task_accuracies[task_id] = accuracy_score(all_targets, all_preds)

        return task_accuracies
