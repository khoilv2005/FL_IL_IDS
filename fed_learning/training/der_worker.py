"""
DER Worker - Multi-GPU training worker for DER clients.

Handles parallel training of DER clients on GPU with two-stage support.
Uses DERModel instead of CNN_GRU_Model.
"""

import time
from collections import OrderedDict
from typing import Dict, List

import torch

from ..models.der_model import DERModel


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

    This function runs in a separate thread for multi-GPU parallelism.
    Same pattern as fedcbdr_worker but uses DERModel and passes stage.

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
    if use_cpu:
        device = "cpu"
        device_name = "CPU"
    else:
        device = f"cuda:{gpu_id}"
        device_name = f"GPU {gpu_id}"

    gpu_start = time.time()

    # Create DERModel for this device (NOT CNN_GRU_Model)
    model = DERModel(config["input_shape"], config["num_classes"]).to(device)

    # DERModel needs task structure before loading state_dict.
    # The global_params already contains the correct architecture
    # (extractors.0.*, extractors.1.*, etc.) from server's global model.
    # We need to add_task() the correct number of times to match the structure.
    #
    # Approach: infer num_tasks from global_params keys and reconstruct.
    _reconstruct_model_structure(model, global_params, config)

    # Belt-and-suspenders: ensure all newly added modules are on the correct device.
    # DERModel uses _device_sentinel to auto-detect device in add_task(), but
    # an explicit .to(device) here guarantees correctness regardless.
    model.to(device)

    # Training hyperparameters
    epochs = config.get("local_epochs", 3)
    batch_size = config.get("batch_size", 128)
    lr = config.get("learning_rate", 0.001)
    replay_ratio = config.get("replay_ratio", 0.5)

    for idx, client in enumerate(clients):
        # Load global params into model
        model.load_state_dict(
            {k: v.to(device) for k, v in global_params.items()},
            strict=True,
        )

        # Setup client for this GPU (inherited from FederatedClient)
        client.setup_for_gpu(model, device)

        # Compute total batches PER EPOCH for annealing schedule (Eq.8).
        # Paper: b is the batch index WITHIN one epoch (resets each epoch),
        #        B is total batches IN ONE EPOCH.
        # Must be per-client since each client has different data sizes.
        if hasattr(trainer, 'total_batches'):
            trainer.total_batches = max(1, client.num_samples // batch_size)

        # Reset batch counter per client
        if hasattr(trainer, 'current_batch'):
            trainer.current_batch = 0

        # Train
        result = client.train(
            trainer=trainer,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            global_params=global_params,
            stage=stage,
            replay_ratio=replay_ratio,
        )

        results_dict[client.client_id] = result

    gpu_time = time.time() - gpu_start
    print(f"      [{device_name}] Stage {stage}: "
          f"{len(clients)} clients done in {gpu_time:.1f}s")

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
