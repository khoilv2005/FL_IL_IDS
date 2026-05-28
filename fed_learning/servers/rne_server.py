"""RNE federated server."""

import time
from collections import OrderedDict
from threading import Thread
from typing import Dict, List

import numpy as np
import torch

from ..models.rne_model import RNEModel
from ..training.rne_worker import _reconstruct_model_structure
from .der_server import DERServer
from .incremental_server import IncrementalServer


class RNEServer(DERServer):
    """Server for Recurrent Network Expansion."""

    def __init__(self, clients, test_data: Dict, config: Dict):
        IncrementalServer.__init__(self, clients, test_data, config)
        del self.global_model
        self.global_model = RNEModel(
            config["input_shape"],
            config["num_classes"],
            recurrent_scale=config.get("rne_recurrent_scale", 1.0),
        ).to(self.primary_device)
        self._task_classes_history: Dict[int, List[int]] = {}
        print("📊 Strategy: RNE (Recurrent Network Expansion)")

    def set_global_params(self, params: OrderedDict):
        extractor_indices = {
            int(k.split(".")[1])
            for k in params
            if k.startswith("extractors.") and k.split(".")[1].isdigit()
        }
        num_tasks_in_params = len(extractor_indices)
        if self.global_model.num_extractors != num_tasks_in_params:
            recon_config = {
                "input_shape": self.config["input_shape"],
                "num_classes": self.config["num_classes"],
                "s_max": self.config.get("s_max", 15.0),
                "rne_recurrent_scale": self.config.get("rne_recurrent_scale", 1.0),
                "task_classes_history": self._task_classes_history,
            }
            _reconstruct_model_structure(self.global_model, OrderedDict(params), recon_config)

        self.global_model.load_state_dict(
            {k: v.to(self.primary_device) for k, v in params.items()}
        )

    def set_task(self, task_id: int, task_classes: list, seen_classes: list = None):
        super().set_task(task_id, task_classes, seen_classes)
        self._task_classes_history[task_id] = list(task_classes)

    def train_round(
        self,
        participating_clients=None,
        stage: int = 1,
        verbose: bool = True,
    ) -> Dict:
        from ..training.rne_worker import train_rne_clients_on_gpu

        round_start = time.time()
        clients = participating_clients or self.clients

        if verbose:
            stage_name = "Representation" if stage == 1 else "Classifier"
            device_info = "CPU" if self.use_cpu else f"{self.num_gpus} GPU(s)"
            print(
                f"\n→ RNE Stage {stage} ({stage_name}): "
                f"Training {len(clients)} clients on {device_info}"
            )

        global_params = self.get_global_params()
        worker_config = {**self.config}
        worker_config["task_classes_history"] = self._task_classes_history

        clients_per_gpu = [[] for _ in range(self.num_gpus)]
        for idx, client in enumerate(clients):
            clients_per_gpu[idx % self.num_gpus].append(client)

        results_dict = {}
        threads = []
        for gpu_id in range(self.num_gpus):
            if clients_per_gpu[gpu_id]:
                thread = Thread(
                    target=train_rne_clients_on_gpu,
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
                threads.append(thread)
                thread.start()

        for thread in threads:
            thread.join()

        results = list(results_dict.values())
        new_params = self.aggregator.aggregate(results, global_params)
        self.set_global_params(new_params)

        avg_loss = float(np.mean([r["loss"] for r in results])) if results else 0.0
        round_time = time.time() - round_start

        if verbose:
            print(f"  → RNE Stage {stage} loss: {avg_loss:.4f} ({round_time:.1f}s)")
            total_replay = sum(r.get("replay_samples", 0) for r in results)
            if total_replay > 0:
                print(f"  → Replay samples: {total_replay}")
            if stage == 1 and hasattr(self.global_model, "get_mask_stats"):
                for key, value in self.global_model.get_mask_stats().items():
                    print(f"  → {key}: {value:.2%}")

        return {"train_loss": avg_loss, "round_time": round_time}
