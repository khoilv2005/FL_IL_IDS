"""
CGoFed Server - Specialized server for Class Incremental Learning.

Implements paper Section 5.2 with proper timing for Eq.14 local regularization
and Eq.12 personalized aggregation per client.
"""

import time
from collections import OrderedDict
from typing import Dict, List, Optional
from threading import Thread

import numpy as np
import torch
import torch.nn.functional as F

from .incremental_server import IncrementalServer
from ..training.cgofed_worker import train_cgofed_clients_on_gpu
from ..clients.cgofed_client import CGoFedClient
from ..strategies.incremental.cgofed import CGoFedAggregator


class CGoFedServer(IncrementalServer):
    """
    Server for CGoFed algorithm with proper Eq.14 and Eq.12 implementation.

    Key differences from base server:
    1. Uses CGoFedWorker instead of generic worker
    2. Computes similarity at START of task (not after aggregate)
    3. Persists historical data across tasks
    4. Creates per-client similarity weights
    5. Passes local regularization info to clients BEFORE training
    """

    def __init__(self, clients: List[CGoFedClient], test_data: Dict, config: Dict):
        super().__init__(clients, test_data, config)

        # Ensure we have CGoFed strategy
        if config["algorithm"].lower() != "cgofed":
            raise ValueError("CGoFedServer only supports 'cgofed' algorithm")

        # Historical data persisted across tasks
        # These are shared between tasks via the server instance
        self._task_global_models: Dict[int, OrderedDict] = {}
        self._task_representation_matrices: Dict[int, torch.Tensor] = {}
        self._task_representations: Dict[int, torch.Tensor] = {}
        self._client_task_representations: Dict[int, Dict[int, torch.Tensor]] = {}
        self._client_task_models: Dict[int, Dict[int, OrderedDict]] = {}

        # Per-client regularization info for current task
        self._client_reg_info: Dict[int, Dict] = {}

        # Paper Eq. 12: Per-client personalized models for next round initialization
        self._personalized_round_models: Dict[int, OrderedDict] = {}

        # Ratio between own model and weighted sum of other-client models (Eq. 12 note)
        self.eq12_self_weight = float(config.get("eq12_self_weight", 0.5))
        self.eq12_self_weight = max(0.0, min(1.0, self.eq12_self_weight))

    def _sync_history_from_aggregator(self) -> None:
        """
        Sync historical task data from persistent aggregator instance.

        train_incremental_kaggle.py creates a new server each task, but trainer/aggregator
        are persistent objects. This method restores server-side history on each task.
        """
        if not isinstance(self.aggregator, CGoFedAggregator):
            return

        if hasattr(self.aggregator, "task_global_models"):
            self._task_global_models = {
                tid: OrderedDict((k, v.cpu().clone()) for k, v in params.items())
                for tid, params in self.aggregator.task_global_models.items()
            }

        if hasattr(self.aggregator, "task_representation_matrices"):
            self._task_representation_matrices = {
                tid: rep.cpu().clone()
                for tid, rep in self.aggregator.task_representation_matrices.items()
            }

        if hasattr(self.aggregator, "task_representations"):
            self._task_representations = {
                tid: rep.cpu().clone()
                for tid, rep in self.aggregator.task_representations.items()
            }

        if hasattr(self.aggregator, "client_representations"):
            self._client_task_representations = {}
            for cid, task_map in self.aggregator.client_representations.items():
                self._client_task_representations[cid] = {}
                for tid, rep in task_map.items():
                    self._client_task_representations[cid][tid] = rep.cpu().clone()

        if hasattr(self.aggregator, "client_historical_models"):
            self._client_task_models = {}
            for cid, task_map in self.aggregator.client_historical_models.items():
                self._client_task_models[cid] = {}
                for tid, params in task_map.items():
                    self._client_task_models[cid][tid] = OrderedDict(
                        (k, v.cpu().clone()) for k, v in params.items()
                    )

    def set_task(
        self, task_id: int, task_classes: List[int], seen_classes: Optional[List[int]] = None
    ):
        """
        Set current task with historical data loading.

        Paper Eq.14: At start of task t+1:
        1. Load historical models from tasks 0...t
        2. Compute similarity between current and historical tasks
        3. Create per-client similarity weights
        4. Prepare local regularization info
        """
        super().set_task(task_id, task_classes, seen_classes)
        self._sync_history_from_aggregator()

        # Reset per-round personalized models for new task
        self._personalized_round_models = {}

        if task_id == 0:
            # First task: no historical data
            self._client_reg_info = {}
            return

        # Tasks > 0: Compute per-client cross-task similarities using client training data
        print(f"📊 CGoFed: Computing per-client cross-task similarities for task {task_id}...")
        self._client_reg_info = {}
        top_k = getattr(self.aggregator, "top_k", 2)

        num_configured = 0
        for client in self.clients:
            current_rep = self._compute_client_task_representation(client)
            if current_rep is None:
                continue

            selected = self._select_historical_models_for_client(
                client_id=client.client_id,
                current_rep=current_rep,
                current_task=task_id,
                top_k=top_k,
            )
            if not selected:
                continue

            sim_scores = torch.tensor([s["similarity"] for s in selected], dtype=torch.float32)
            sim_scores = sim_scores - sim_scores.max()
            sim_scores = torch.clamp(sim_scores, min=-20.0, max=0.0)
            weights = F.softmax(sim_scores, dim=0)

            historical_models = {
                s["key"]: self._clone_params(s["params"]) for s in selected
            }
            similarity_weights = {
                s["key"]: float(weights[i]) for i, s in enumerate(selected)
            }

            self._client_reg_info[client.client_id] = {
                "historical_models": historical_models,
                "similarity_weights": similarity_weights,
            }
            num_configured += 1

        if num_configured == 0:
            print("  ⚠️ No valid per-client historical matches found")
        else:
            print(f"  ✓ Prepared Eq.14 regularization info for {num_configured}/{len(self.clients)} clients")

    def train_round(self, verbose: bool = True) -> Dict:
        """
        Train round with CGoFed-specific worker and per-client regularization.
        """
        round_start = time.time()

        if verbose:
            device_info = "CPU" if self.use_cpu else f"{self.num_gpus} GPU(s)"
            print(f"\n→ CGOFED: Training {len(self.clients)} clients on {device_info}")

        global_params = self.get_global_params()

        # Distribute clients across GPUs
        clients_per_gpu = [[] for _ in range(self.num_gpus)]
        for i, c in enumerate(self.clients):
            clients_per_gpu[i % self.num_gpus].append(c)

        if verbose:
            for gpu_id, clients in enumerate(clients_per_gpu):
                device_label = "CPU" if self.use_cpu else f"GPU {gpu_id}"
                print(f"   {device_label}: {len(clients)} clients")

        # Shared results dict
        results_dict = {}

        # Create threads for each GPU using CGoFed worker
        threads = []
        for gpu_id in range(self.num_gpus):
            if len(clients_per_gpu[gpu_id]) > 0:
                # Get per-client regularization info for this GPU's clients
                client_reg_info = {
                    c.client_id: self._client_reg_info.get(c.client_id, {})
                    for c in clients_per_gpu[gpu_id]
                }
                client_init_models = {
                    c.client_id: self._personalized_round_models.get(c.client_id)
                    for c in clients_per_gpu[gpu_id]
                }

                t = Thread(
                    target=train_cgofed_clients_on_gpu,
                    args=(
                        gpu_id,
                        clients_per_gpu[gpu_id],
                        global_params,
                        self.config,
                        results_dict,
                        self.trainer,
                        self.use_cpu,
                        client_reg_info,  # Per-client regularization info
                        client_init_models,  # Eq.12 personalized init models
                    ),
                )
                threads.append(t)
                t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Collect results
        results = list(results_dict.values())

        # Aggregate using CGoFed aggregator
        new_params = self.aggregator.aggregate(results, global_params)
        self.set_global_params(new_params)

        # Paper Eq. 12: Build per-client personalized models for next round
        self._personalized_round_models = self._compute_personalized_models(results)

        # Keep server cache synchronized with aggregator history
        self._sync_history_from_aggregator()

        avg_loss = float(np.mean([r["loss"] for r in results]))
        round_time = time.time() - round_start

        if verbose:
            print(f"\n→ Train loss: {avg_loss:.4f}")
            print(f"→ Round time: {round_time:.2f}s")

        return {"train_loss": avg_loss, "round_time": round_time}

    def _compute_task_representation(self) -> torch.Tensor:
        """
        Compute an aggregate representation for current task from client TRAIN data.

        Kept as a utility for debugging/fallback paths.
        """
        all_reps = []
        for client in self.clients:
            rep = self._compute_client_task_representation(client, num_samples=64)
            if rep is not None:
                all_reps.append(rep)
        if all_reps:
            return torch.cat(all_reps, dim=0)
        return None

    def _compute_client_task_representation(
        self, client: CGoFedClient, num_samples: int = 100
    ) -> Optional[torch.Tensor]:
        """
        Compute current-task representation for a specific client from its local TRAIN data.

        This avoids using global test data for training-related regularization setup.
        """
        try:
            rep = client.compute_activation_representation(
                model=self.global_model,
                num_samples=num_samples,
            )
        except Exception:
            return None

        if rep is None or not isinstance(rep, torch.Tensor):
            return None
        if torch.isnan(rep).any() or torch.isinf(rep).any():
            return None
        return rep.cpu()

    @staticmethod
    def _compute_similarity(R1: torch.Tensor, R2: torch.Tensor) -> float:
        """Compute negative Frobenius norm distance between representations."""
        if torch.isnan(R1).any() or torch.isinf(R1).any():
            return -1e6
        if torch.isnan(R2).any() or torch.isinf(R2).any():
            return -1e6

        # Use full-matrix distance whenever feature dimension matches.
        # If row count differs, align by shared sample count.
        if R1.dim() == 2 and R2.dim() == 2 and R1.shape[1] == R2.shape[1]:
            n = min(R1.shape[0], R2.shape[0])
            if n > 0:
                l2_dist = torch.norm(R1[:n] - R2[:n], p="fro").item()
            else:
                l2_dist = 1e6
        elif R1.shape == R2.shape:
            l2_dist = torch.norm(R1 - R2, p="fro").item()
        else:
            # Last resort for incompatible shapes: prototype distance.
            r1_mean = R1.view(R1.shape[0], -1).mean(dim=0) if R1.dim() > 1 else R1
            r2_mean = R2.view(R2.shape[0], -1).mean(dim=0) if R2.dim() > 1 else R2
            if r1_mean.shape != r2_mean.shape:
                return -1e6
            l2_dist = torch.norm(r1_mean - r2_mean, p=2).item()

        return -l2_dist

    def _select_historical_models_for_client(
        self,
        client_id: int,
        current_rep: torch.Tensor,
        current_task: int,
        top_k: int,
    ) -> List[Dict]:
        """
        Build candidate historical models for one client using Eq.10-style similarity.

        Priority:
        1) Historical task models from OTHER clients (paper-consistent)
        2) Fallback to task-global historical models (robustness)
        """
        candidates: List[Dict] = []

        # Paper Section 5.2: compare with tasks from other clients.
        for hist_client_id, task_rep_map in self._client_task_representations.items():
            if hist_client_id == client_id:
                continue

            task_model_map = self._client_task_models.get(hist_client_id, {})
            for hist_task_id, hist_rep in task_rep_map.items():
                if hist_task_id >= current_task:
                    continue
                hist_params = task_model_map.get(hist_task_id)
                if hist_params is None:
                    continue
                sim = self._compute_similarity(current_rep, hist_rep)
                if not np.isfinite(sim):
                    continue
                key = f"c{hist_client_id}_t{hist_task_id}"
                candidates.append(
                    {
                        "key": key,
                        "similarity": sim,
                        "params": hist_params,
                    }
                )

        # Fallback if no per-client history exists yet
        if not candidates:
            for hist_task_id in range(current_task):
                hist_rep = self._task_representation_matrices.get(
                    hist_task_id, self._task_representations.get(hist_task_id)
                )
                hist_params = self._task_global_models.get(hist_task_id)
                if hist_rep is None or hist_params is None:
                    continue
                sim = self._compute_similarity(current_rep, hist_rep)
                if not np.isfinite(sim):
                    continue
                key = f"g_t{hist_task_id}"
                candidates.append(
                    {
                        "key": key,
                        "similarity": sim,
                        "params": hist_params,
                    }
                )

        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        return candidates[:top_k]

    @staticmethod
    def _clone_params(params: OrderedDict) -> OrderedDict:
        """Deep-copy model params to CPU tensors."""
        return OrderedDict((k, v.cpu().clone()) for k, v in params.items())

    @staticmethod
    def _to_rep_matrix(rep: torch.Tensor) -> Optional[torch.Tensor]:
        """Normalize representation into a stable 2D matrix."""
        if rep is None:
            return None
        if not isinstance(rep, torch.Tensor):
            return None
        if torch.isnan(rep).any() or torch.isinf(rep).any():
            return None
        if rep.dim() == 1:
            return rep.detach().cpu().float().unsqueeze(0)
        if rep.dim() == 2:
            return rep.detach().cpu().float()
        if rep.dim() > 2:
            return rep.detach().cpu().float().view(rep.shape[0], -1)
        return None

    @staticmethod
    def _weighted_average_models(
        model_params: List[OrderedDict], weights: torch.Tensor
    ) -> OrderedDict:
        """Compute weighted average of model parameters."""
        if not model_params:
            return OrderedDict()

        result = OrderedDict()
        base = model_params[0]
        for name, tensor in base.items():
            if tensor.dtype.is_floating_point:
                acc = torch.zeros_like(tensor, dtype=torch.float32)
                for w, params in zip(weights.tolist(), model_params):
                    acc += float(w) * params[name].float()
                result[name] = acc.to(dtype=tensor.dtype)
            else:
                result[name] = tensor.clone()
        return result

    @staticmethod
    def _blend_models(
        own_params: OrderedDict, others_params: OrderedDict, self_weight: float
    ) -> OrderedDict:
        """
        Paper Eq. 12: Θ^{t,g}_k = β * Θ^t_k + (1-β) * Σ w_i Θ^t_i.

        Convex combination (interpolation) of own model and weighted others.
        self_weight is β (eq12_self_weight). When β=1 the result is pure own model;
        when β=0 the result is pure weighted sum of others.
        """
        blend = OrderedDict()
        for name, own_tensor in own_params.items():
            if own_tensor.dtype.is_floating_point and name in others_params:
                other_tensor = others_params[name].to(dtype=torch.float32)
                mixed = self_weight * own_tensor.float() + (1.0 - self_weight) * other_tensor
                blend[name] = mixed.to(dtype=own_tensor.dtype)
            else:
                blend[name] = own_tensor.clone()
        return blend

    def _compute_personalized_models(self, results: List[Dict]) -> Dict[int, OrderedDict]:
        """
        Paper Eq. 12: personalized aggregation for each client k.

        Θ_k^{t,g} = β * Θ_k^t + (1 - β) * Σ_{i != k}(w_{k,i} * Θ_i^t),
        where w_{k,i} is derived from Eq.10 similarities on current-task representations
        and β = eq12_self_weight controls the contribution of others.
        """
        if len(results) < 2:
            return {}

        client_params: Dict[int, OrderedDict] = {}
        rep_matrices: Dict[int, torch.Tensor] = {}

        for r in results:
            client_id = r.get("client_id")
            params = r.get("params")
            if client_id is None or params is None:
                continue

            client_params[client_id] = self._clone_params(params)
            rep_mat = self._to_rep_matrix(r.get("representation"))
            if rep_mat is not None:
                rep_matrices[client_id] = rep_mat

        if len(client_params) < 2:
            return {}

        personalized_models: Dict[int, OrderedDict] = {}

        for client_id, own_params in client_params.items():
            if client_id not in rep_matrices:
                personalized_models[client_id] = self._clone_params(own_params)
                continue

            other_ids = [
                oid
                for oid in client_params.keys()
                if oid != client_id and oid in rep_matrices
            ]
            if not other_ids:
                personalized_models[client_id] = self._clone_params(own_params)
                continue

            # Eq.10 uses representation distance; convert to softmax logits via -distance.
            own_rep = rep_matrices[client_id]
            neg_distances = []
            valid_other_ids = []
            for oid in other_ids:
                other_rep = rep_matrices[oid]
                sim = self._compute_similarity(own_rep, other_rep)
                dist = -sim
                if not np.isfinite(dist):
                    continue
                neg_distances.append(-dist)
                valid_other_ids.append(oid)

            if not valid_other_ids:
                personalized_models[client_id] = self._clone_params(own_params)
                continue

            sim_scores = torch.tensor(neg_distances, dtype=torch.float32)
            sim_scores = sim_scores - sim_scores.max()
            sim_scores = torch.clamp(sim_scores, min=-20.0, max=0.0)
            weights = F.softmax(sim_scores, dim=0)

            other_models = [client_params[oid] for oid in valid_other_ids]
            others_agg = self._weighted_average_models(other_models, weights)

            personalized_models[client_id] = self._blend_models(
                own_params,
                others_agg,
                self_weight=self.eq12_self_weight,
            )

        return personalized_models
