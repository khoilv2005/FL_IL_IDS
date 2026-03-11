"""
DER Worker - Multi-GPU training worker for DER clients.

Handles parallel training of DER clients on GPU with two-stage support.
Uses DERModel instead of CNN_GRU_Model.

Inherits from BaseGPUWorker, overriding:
- create_model(): uses DERModel
- prepare_model(): reconstructs task structure
- get_train_kwargs(): passes stage and replay_ratio
- prepare_client(): sets up annealing schedule
"""

from collections import OrderedDict
from typing import Dict, List

import torch

from ..models.der_model import DERModel
from .base_worker import BaseGPUWorker


class DERWorker(BaseGPUWorker):
    """Worker for DER algorithm with two-stage training and DERModel."""

    def __init__(
        self,
        gpu_id: int,
        clients: list,
        global_params: OrderedDict,
        config: Dict,
        results_dict: Dict,
        trainer,
        use_cpu: bool = False,
        stage: int = 1,
    ):
        super().__init__(gpu_id, clients, global_params, config, results_dict, trainer, use_cpu)
        self.stage = stage
        self.replay_ratio = config.get("replay_ratio", 0.5)

    def create_model(self):
        """Create DERModel instead of CNN_GRU_Model."""
        return DERModel(
            self.config["input_shape"], self.config["num_classes"]
        ).to(self.device)

    def prepare_model(self, model):
        """Reconstruct DERModel task structure to match global_params."""
        _reconstruct_model_structure(model, self.global_params, self.config)
        # Ensure all newly added modules are on the correct device
        model.to(self.device)

    def prepare_client(self, client, model, idx: int):
        """Set up DER-specific annealing schedule per client."""
        # Compute total batches PER EPOCH for annealing schedule (Eq.8)
        if hasattr(self.trainer, 'total_batches'):
            self.trainer.total_batches = max(1, client.num_samples // self.batch_size)
        # Reset batch counter per client
        if hasattr(self.trainer, 'current_batch'):
            self.trainer.current_batch = 0

    def get_train_kwargs(self, client, idx: int) -> Dict:
        """Pass stage and replay_ratio to client.train()."""
        return {
            "global_params": self.global_params,
            "stage": self.stage,
            "replay_ratio": self.replay_ratio,
        }

    def run(self):
        """Override run to customize logging message."""
        import time
        gpu_start = time.time()

        model = self.create_model()
        self.prepare_model(model)

        for idx, client in enumerate(self.clients):
            init_params = self.get_init_params(client)
            self.load_params(model, init_params)

            client.setup_for_gpu(model, self.device)
            self.prepare_client(client, model, idx)

            train_kwargs = self.get_train_kwargs(client, idx)
            result = client.train(
                trainer=self.trainer,
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                **train_kwargs,
            )

            self.results_dict[client.client_id] = result

        gpu_time = time.time() - gpu_start
        print(f"      [{self.device_name}] Stage {self.stage}: "
              f"{len(self.clients)} clients done in {gpu_time:.1f}s")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def train_der_clients_on_gpu(
    gpu_id: int,
    clients: list,
    global_params: OrderedDict,
    config: Dict,
    results_dict: Dict,
    trainer,
    use_cpu: bool = False,
    stage: int = 1,
):
    """
    Train DER clients on a specific GPU.

    Args:
        gpu_id: GPU ID (0, 1, 2, ...)
        clients: List of DERClient instances
        global_params: Global DERModel parameters
        config: Training configuration
        results_dict: Shared dict to store results (thread-safe by GIL)
        trainer: DERTrainer instance
        use_cpu: Whether to use CPU instead of GPU
        stage: Training stage (1=representation, 2=classifier)
    """
    worker = DERWorker(
        gpu_id, clients, global_params, config, results_dict,
        trainer, use_cpu, stage,
    )
    worker.run()


def _reconstruct_model_structure(
    model: DERModel,
    global_params: OrderedDict,
    config: Dict,
):
    """
    Reconstruct DERModel task structure to match global_params.

    DERModel is dynamic — add_task() must be called the correct number
    of times before load_state_dict() can work.

    We infer the number of tasks from the extractor keys in global_params.

    Args:
        model: Empty DERModel (no tasks added yet)
        global_params: State dict from server's global model
        config: Config with task_classes_history or fallback info
    """
    # Count extractors from state_dict keys
    # Keys look like: extractors.0.conv1.weight, extractors.1.gru.weight, ...
    extractor_indices = set()
    for key in global_params.keys():
        if key.startswith("extractors."):
            parts = key.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                extractor_indices.add(int(parts[1]))

    num_tasks = len(extractor_indices)
    if num_tasks == 0:
        return

    # Get task classes history from config (set by server)
    task_classes_history = config.get("task_classes_history", {})
    s_max = config.get("s_max", 15.0)

    for t in range(num_tasks):
        # Get new_classes for this task (fallback to dummy if not available)
        if t in task_classes_history:
            new_classes = task_classes_history[t]
        elif str(t) in task_classes_history:
            new_classes = task_classes_history[str(t)]
        else:
            # Fallback: infer aux_classifier output size from state_dict
            aux_key = f"aux_classifier.weight"
            if aux_key in global_params:
                n_aux = global_params[aux_key].shape[0]
                # aux has |Y_t|+1 classes for t>0, |Y_t| for t=0
                new_classes = list(range(n_aux if t == 0 else n_aux - 1))
            else:
                new_classes = [0]  # minimal fallback

        model.add_task(new_classes, s_max=s_max)
