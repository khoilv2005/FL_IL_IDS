"""
Training module - utilities for running federated learning.

Hierarchy:
    base_worker.BaseGPUWorker       - Base class with common GPU training loop
    ├── worker.StandardWorker       - FedAvg/FedProx/FedPlus/EWC
    ├── cgofed_worker.CGoFedWorker  - CGoFed with per-client regularization
    ├── der_worker.DERWorker        - DER with two-stage + DERModel
    ├── fedcbdr_worker.FedCBDRWorker- FedCBDR with replay buffer
    ├── fedlwf_worker.FedLwFWorker  - FedLwF with knowledge distillation
    ├── glfc_worker.GLFCWorker      - GLFC with exemplar management
    ├── nice_worker.NICEWorker      - NICE with NICEModel + ages/masks
    └── refed_worker.ReFedWorker    - Re-Fed with PIM caching

    task_loop   - Main FCIL training orchestration (import directly)
    post_task   - Algorithm-specific post-task processing hooks (import directly)

Note: task_loop and post_task are NOT eagerly imported here to avoid
circular imports (they depend on factories/ which depends on servers/).
Import them directly when needed:
    from fed_learning.training.task_loop import run_incremental_training
    from fed_learning.training.post_task import post_task_processing
"""

from .base_worker import BaseGPUWorker
from .runner import train_federated_multigpu
from .worker import train_clients_on_gpu
from .cgofed_worker import train_cgofed_clients_on_gpu
from .glfc_worker import train_glfc_clients_on_gpu
from .refed_worker import train_refed_clients_on_gpu

__all__ = [
    "BaseGPUWorker",
    "train_federated_multigpu",
    "train_clients_on_gpu",
    "train_cgofed_clients_on_gpu",
    "train_glfc_clients_on_gpu",
    "train_refed_clients_on_gpu",
]
