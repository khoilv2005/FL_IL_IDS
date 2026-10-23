"""
Base Worker - Common GPU training logic for all federated workers.

All algorithm-specific workers inherit from this base to eliminate
code duplication. Each worker only overrides the algorithm-specific
parts (model creation, pre/post training hooks).

Inheritance:
    base_worker.BaseGPUWorker  (this file)
    ├── worker.py              (standard FedAvg/FedProx/etc.)
    ├── cgofed_worker.py       (CGoFed with per-client reg)
    ├── der_worker.py          (DER with two-stage + DERModel)
    ├── fedcbdr_worker.py      (FedCBDR with replay)
    ├── fedlwf_worker.py       (FedLwF with distillation)
    ├── glfc_worker.py         (GLFC with exemplar management)
    ├── nice_worker.py         (NICE with NICEModel + ages/masks)
    └── refed_worker.py        (Re-Fed with PIM caching)
"""

import time
from collections import OrderedDict
from typing import Dict, List, Callable, Optional

import torch


class BaseGPUWorker:
    """
    Base class encapsulating the common GPU training loop.

    Subclasses override:
    - create_model(): to use algorithm-specific model (DERModel, NICEModel, etc.)
    - prepare_model(): to set up model structure before loading state_dict
    - prepare_client(): to transfer algorithm-specific state (ages, masks, etc.)
    - get_train_kwargs(): to pass algorithm-specific kwargs to client.train()
    """

    def __init__(
        self,
        gpu_id: int,
        clients: list,
        global_params: OrderedDict,
        config: Dict,
        results_dict: Dict,
        trainer,
        use_cpu: bool = False,
    ):
        self.gpu_id = gpu_id
        self.clients = clients
        self.global_params = global_params
        self.config = config
        self.results_dict = results_dict
        self.trainer = trainer
        self.use_cpu = use_cpu

        # Resolve device
        if use_cpu:
            self.device = "cpu"
            self.device_name = "CPU"
        else:
            self.device = f"cuda:{gpu_id}"
            self.device_name = f"GPU {gpu_id}"

        # Common training hyperparameters
        self.epochs = config.get("local_epochs", 3)
        self.batch_size = config.get("batch_size", 128)
        self.lr = config.get("learning_rate", 0.001)

    def create_model(self):
        """
        Create the model for this device.

        Override for algorithm-specific models (DERModel, NICEModel).
        Default: CNN_GRU_Model.
        """
        from ..models.cnn_gru import CNN_GRU_Model
        return CNN_GRU_Model(
            self.config["input_shape"], self.config["num_classes"]
        ).to(self.device)

    def prepare_model(self, model):
        """
        Prepare model structure before loading state_dict.

        Override for models that need structural setup (e.g., DERModel.add_task()).
        Default: no-op.
        """
        pass

    def load_params(self, model, params: Optional[OrderedDict] = None):
        """Load parameters into model, moving to correct device."""
        if params is None:
            params = self.global_params
        model.load_state_dict(
            {k: v.to(self.device) for k, v in params.items()},
            strict=True,
        )

    def prepare_client(self, client, model, idx: int):
        """
        Per-client preparation before training.

        Override to transfer algorithm-specific state (neuron ages, masks, etc.).
        Default: no-op.
        """
        pass

    def get_train_kwargs(self, client, idx: int) -> Dict:
        """
        Get algorithm-specific kwargs for client.train().

        Override to pass extra arguments like stage, replay_ratio, etc.
        Default: just global_params.
        """
        return {"global_params": self.global_params}

    def post_client_train(self, client, result, idx: int):
        """
        Post-training hook for each client.

        Override for algorithm-specific post-processing.
        Default: no-op.
        """
        pass

    def get_init_params(self, client) -> Optional[OrderedDict]:
        """
        Get per-client initialization parameters.

        Override for algorithms with personalized initialization (e.g., CGoFed Eq.12).
        Default: returns global_params.
        """
        return self.global_params

    def should_log_progress(self, idx: int) -> bool:
        """Whether to log progress at this client index."""
        return (idx + 1) % 50 == 0 or idx == len(self.clients) - 1

    def run(self):
        """
        Execute the training loop for all clients on this GPU.

        This is the main template method that orchestrates the training.
        """
        gpu_start = time.time()

        # Create and prepare model
        model = self.create_model()
        self.prepare_model(model)

        algo_name = self.__class__.__name__.replace("Worker", "")
        print(f"      [{self.device_name}] Starting {len(self.clients)} {algo_name} clients...")

        for idx, client in enumerate(self.clients):
            # Get per-client init params
            init_params = self.get_init_params(client)
            self.load_params(model, init_params)

            # Setup client for this GPU
            client.setup_for_gpu(model, self.device)

            # Algorithm-specific client preparation
            self.prepare_client(client, model, idx)

            # Get training kwargs
            train_kwargs = self.get_train_kwargs(client, idx)

            # Train
            result = client.train(
                trainer=self.trainer,
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                **train_kwargs,
            )

            # Post-training hook
            self.post_client_train(client, result, idx)

            # Log progress
            if self.should_log_progress(idx):
                print(
                    f"      [{self.device_name}] Progress: {idx + 1}/{len(self.clients)} clients done"
                )

            self.results_dict[client.client_id] = result

        gpu_time = time.time() - gpu_start
        print(f"      [{self.device_name}] ✓ All {len(self.clients)} clients done in {gpu_time:.2f}s")

        # Cleanup
        del model
        if not self.use_cpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
