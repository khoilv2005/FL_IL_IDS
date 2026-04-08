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
from ..strategies.fed_incremental.cgofed import (
    build_representation_artifact,
    CGoFedAggregator,
    clone_representation_state,
    clone_model_reference,
    coerce_representation_matrix,
    compute_representation_similarity,
    load_model_state,
)


class CGoFedServer(IncrementalServer):
    """
    Server chuyên cho CGoFed với regularization theo lịch sử task và personalized models.

    Key differences from base server:
    1. Uses CGoFedWorker instead of generic worker
    2. Computes next-round similarity state from the PREVIOUS round
    3. Persists historical data across tasks
    4. Creates per-client similarity weights
    5. Passes local regularization info to clients BEFORE training
    """

    def __init__(self, clients: List[CGoFedClient], test_data: Dict, config: Dict):
        """Khởi tạo server và các cache lịch sử cần cho Eq.12 / Eq.14 của CGoFed."""
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
        Đồng bộ lịch sử task từ aggregator về server hiện tại.

        Việc này cần thiết vì aggregator của CGoFed giữ state xuyên task,
        còn server có thể được tạo lại ở mỗi task.
        """
        if not isinstance(self.aggregator, CGoFedAggregator):
            return

        if hasattr(self.aggregator, "task_global_models"):
            self._task_global_models = {
                tid: clone_model_reference(params)
                for tid, params in self.aggregator.task_global_models.items()
            }

        if hasattr(self.aggregator, "task_representation_matrices"):
            self._task_representation_matrices = {
                tid: clone_representation_state(rep)
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
                    self._client_task_representations[cid][tid] = clone_representation_state(rep)

        if hasattr(self.aggregator, "client_historical_models"):
            self._client_task_models = {}
            for cid, task_map in self.aggregator.client_historical_models.items():
                self._client_task_models[cid] = {}
                for tid, params in task_map.items():
                    self._client_task_models[cid][tid] = clone_model_reference(params)

        # DEBUG: Print sync status (once per task)
        print(f"  DEBUG[Sync]: task_global_models={len(self._task_global_models)}, task_reps={len(self._task_representation_matrices)}, "
              f"client_reps={len(self._client_task_representations)}, client_models={len(self._client_task_models)}")

    def set_task(
        self,
        task_id: int,
        task_classes: List[int],
        seen_classes: Optional[List[int]] = None,
    ):
        """
        Chuẩn bị task mới cho CGoFed.

        Đây là bước server nạp history cũ và reset state cho task mới.
        Theo paper, Eq.12 / Eq.14 dùng similarity từ previous training round,
        nên task mới luôn bắt đầu với global init chuẩn và chưa có reg info.
        """
        super().set_task(task_id, task_classes, seen_classes)
        self._sync_history_from_aggregator()

        self._personalized_round_models = {}
        self._client_reg_info = {}

    def train_round(self, verbose: bool = True) -> Dict:
        """
        Chạy một round của CGoFed với worker riêng và regularization riêng theo client.
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

        # Keep server cache synchronized with aggregator history
        self._sync_history_from_aggregator()

        is_last_round = (
            getattr(self.aggregator, "_round_in_task", 0)
            >= getattr(self.aggregator, "rounds_per_task", 1)
        )
        if not is_last_round:
            if verbose:
                print("  → Preparing Eq.12 personalized models for next round...")
            self._personalized_round_models = self._compute_personalized_models(results)
            if verbose:
                print("  → Preparing Eq.14 regularization info for next round...")
            self._client_reg_info = self._prepare_next_round_reg_info(results)
        else:
            self._personalized_round_models = {}
            self._client_reg_info = {}

        avg_loss = float(np.mean([r["loss"] for r in results]))
        round_time = time.time() - round_start

        if verbose:
            print(f"\n→ Train loss: {avg_loss:.4f}")
            print(f"→ Round time: {round_time:.2f}s")

        return {"train_loss": avg_loss, "round_time": round_time}

    def _compute_task_representation(self) -> torch.Tensor:
        """
        Tính representation tổng hợp của task hiện tại từ train data của client.

        Chủ yếu dùng cho debug hoặc fallback path.
        """
        num_samples = self.config.get("num_samples_rep")
        if num_samples is not None:
            num_samples = int(num_samples)
            if num_samples <= 0:
                num_samples = None
        all_reps = []
        for client in self.clients:
            rep = self._compute_client_task_representation(client, num_samples=num_samples)
            if rep is not None:
                all_reps.append(rep)
        if all_reps:
            return torch.cat(all_reps, dim=0)
        return None

    def _compute_client_task_representation(
        self, client: CGoFedClient, num_samples: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        """
        Tính representation của một client ở task hiện tại từ train data local.

        Đây là đầu vào để server so sánh similarity với history cũ.
        """
        if num_samples is None:
            num_samples = self.config.get("num_samples_rep")
            if num_samples is not None:
                num_samples = int(num_samples)
                if num_samples <= 0:
                    num_samples = None
        try:
            rep = client.compute_activation_representation(
                model=self.global_model,
                num_samples=num_samples,
            )
        except Exception:
            return None

        return coerce_representation_matrix(rep)

    @staticmethod
    def _compute_similarity(R1: torch.Tensor, R2: torch.Tensor) -> float:
        """Compute a higher-is-better similarity score between representations."""
        return compute_representation_similarity(R1, R2)

    def _select_peer_clients_for_current_round(
        self,
        client_id: int,
        current_rep: torch.Tensor,
        current_round_reps: Dict[int, torch.Tensor],
        top_k: int,
    ) -> List[int]:
        """
        Select the most similar peer clients using current-task representations
        from the previous training round (paper Eq.12 note).
        """
        peer_scores = []
        for other_id, other_rep in current_round_reps.items():
            if other_id == client_id:
                continue
            sim = self._compute_similarity(current_rep, other_rep)
            if np.isfinite(sim):
                peer_scores.append((other_id, sim))

        peer_scores.sort(key=lambda x: x[1], reverse=True)
        return [other_id for other_id, _score in peer_scores[:top_k]]

    def _prepare_next_round_reg_info(self, results: List[Dict]) -> Dict[int, Dict]:
        """
        Prepare Eq.14 regularization info for the NEXT round of the current task.

        Paper-faithful behavior:
        - choose the most similar peer clients using current-task representations
          from the previous round
        - regularize against the FULL historical task set of those selected clients
        """
        if self.current_task == 0:
            return {}

        current_round_reps: Dict[int, object] = {}
        for result in results:
            client_id = result.get("client_id")
            rep = build_representation_artifact(result.get("representation"))
            if client_id is None or rep is None:
                continue
            current_round_reps[client_id] = rep

        if len(current_round_reps) < 2:
            return {}

        top_k = getattr(self.aggregator, "top_k", 2)
        next_round_reg_info: Dict[int, Dict] = {}

        for client_id, current_rep in current_round_reps.items():
            selected_peer_ids = self._select_peer_clients_for_current_round(
                client_id=client_id,
                current_rep=current_rep,
                current_round_reps=current_round_reps,
                top_k=top_k,
            )
            if not selected_peer_ids:
                continue

            historical_entries: List[Dict] = []
            for hist_client_id in selected_peer_ids:
                task_rep_map = self._client_task_representations.get(hist_client_id, {})
                task_model_map = self._client_task_models.get(hist_client_id, {})
                for hist_task_id, hist_rep in task_rep_map.items():
                    if hist_task_id >= self.current_task:
                        continue
                    hist_params = task_model_map.get(hist_task_id)
                    if hist_params is None:
                        continue
                    sim = self._compute_similarity(current_rep, hist_rep)
                    if not np.isfinite(sim):
                        continue
                    historical_entries.append(
                        {
                            "key": f"c{hist_client_id}_t{hist_task_id}",
                            "similarity": sim,
                            "params": hist_params,
                        }
                    )

            if not historical_entries:
                continue

            sim_scores = torch.tensor(
                [entry["similarity"] for entry in historical_entries],
                dtype=torch.float32,
            )
            sim_scores = sim_scores - sim_scores.max()
            weights = F.softmax(sim_scores, dim=0)

            next_round_reg_info[client_id] = {
                "historical_models": {
                    entry["key"]: clone_model_reference(entry["params"])
                    for entry in historical_entries
                },
                "similarity_weights": {
                    entry["key"]: float(weights[i])
                    for i, entry in enumerate(historical_entries)
                },
            }

        if next_round_reg_info:
            print(
                f"  ✓ Prepared Eq.14 next-round regularization for "
                f"{len(next_round_reg_info)}/{len(current_round_reps)} clients"
            )
        else:
            print("  ⚠️ No valid next-round Eq.14 matches found")

        return next_round_reg_info

    @staticmethod
    def _clone_params(params) -> OrderedDict:
        """Materialize model params to a fresh CPU OrderedDict."""
        return load_model_state(params)

    @staticmethod
    def _to_rep_matrix(rep: torch.Tensor) -> Optional[torch.Tensor]:
        """Normalize representation into a stable 2D matrix."""
        return coerce_representation_matrix(rep)

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
                mixed = (
                    self_weight * own_tensor.float()
                    + (1.0 - self_weight) * other_tensor
                )
                blend[name] = mixed.to(dtype=own_tensor.dtype)
            else:
                blend[name] = own_tensor.clone()
        return blend

    def _compute_personalized_models(
        self, results: List[Dict]
    ) -> Dict[int, OrderedDict]:
        """
        Tạo personalized model cho từng client theo Eq.12.

        Mục tiêu là round tiếp theo mỗi client không chỉ nhận global model thuần,
        mà nhận model đã pha trộn với thông tin lịch sử phù hợp hơn với chính nó.
        """
        if len(results) < 2:
            return {}

        client_params: Dict[int, OrderedDict] = {}
        rep_states: Dict[int, object] = {}

        for r in results:
            client_id = r.get("client_id")
            params = r.get("params")
            if client_id is None or params is None:
                continue

            client_params[client_id] = self._clone_params(params)
            rep_state = build_representation_artifact(r.get("representation"))
            if rep_state is not None:
                rep_states[client_id] = rep_state

        if len(client_params) < 2:
            return {}

        personalized_models: Dict[int, OrderedDict] = {}

        for client_id, own_params in client_params.items():
            if client_id not in rep_states:
                personalized_models[client_id] = self._clone_params(own_params)
                continue

            other_ids = [
                oid
                for oid in client_params.keys()
                if oid != client_id and oid in rep_states
            ]
            if not other_ids:
                personalized_models[client_id] = self._clone_params(own_params)
                continue

            # Eq.10 uses representation distance; convert to softmax logits via -distance.
            own_rep = rep_states[client_id]
            neg_distances = []
            valid_other_ids = []
            for oid in other_ids:
                other_rep = rep_states[oid]
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
            weights = F.softmax(sim_scores, dim=0)

            other_models = [client_params[oid] for oid in valid_other_ids]
            others_agg = self._weighted_average_models(other_models, weights)

            personalized_models[client_id] = self._blend_models(
                own_params,
                others_agg,
                self_weight=self.eq12_self_weight,
            )

        return personalized_models
