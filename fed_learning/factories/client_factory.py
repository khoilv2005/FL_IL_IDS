"""
Client Factory - Create algorithm-specific federated clients.

Maps algorithm names to their corresponding client classes,
handling both initial creation and persistent client management.
"""

from typing import Dict, List, Any, Optional

import numpy as np

from fed_learning.clients import (
    FederatedClient,
    CGoFedClient,
    DERClient,
    NICEClient,
    GLFCClient,
    ReFedClient,
    PlexusDERClient,
    PlexusNICEClient,
)
from fed_learning.clients.fedcbdr_client import FedCBDRClient
from fed_learning.clients.fedlwf_client import FedLwFClient


# Registry: algorithm name -> (ClientClass, extra_config_keys)
# extra_config_keys maps config key -> client constructor kwarg
_CLIENT_REGISTRY = {
    "ewc": (
        FederatedClient,
        {},
    ),
    "fedavg_ewc": (
        FederatedClient,
        {},
    ),
    "fedprox_ewc": (
        FederatedClient,
        {},
    ),
    "fedcbdr": (
        FedCBDRClient,
        {
            "buffer_size": ("buffer_size", 500),
            "leverage_rank": ("leverage_rank", 50),
        },
    ),
    "der": (
        DERClient,
        {
            "buffer_size": ("buffer_size", 500),
        },
    ),
    "nice": (
        NICEClient,
        {
            "max_phases": ("nice_max_phases", 5),
            "phase_epochs": ("nice_phase_epochs", 5),
            "tau": ("tau", 0.95),
        },
    ),
    "glfc": (
        GLFCClient,
        {
            "memory_size": ("glfc_memory_size", 2000),
        },
    ),
    "refed": (
        ReFedClient,
        {
            "memory_size": ("refed_memory_size", 2000),
            "lambda_pim": ("refed_lambda_pim", 0.5),
            "pim_iterations": ("refed_pim_iterations", 5),
        },
    ),
    "plexus_der": (
        PlexusDERClient,
        {
            "buffer_size": ("buffer_size", 500),
        },
    ),
    "plexus_nice": (
        PlexusNICEClient,
        {
            "max_phases": ("nice_max_phases", 5),
            "phase_epochs": ("nice_phase_epochs", 5),
            "tau": ("tau", 0.95),
        },
    ),
}

# These algorithms use FedLwFClient
_LWF_ALGORITHMS = {"fedavg_lwf", "fedprox_lwf", "lwf"}


def _resolve_client_class(algo: str):
    """
    Resolve the client class and constructor kwargs for an algorithm.

    Returns:
        (ClientClass, extra_kwargs_spec) where extra_kwargs_spec maps
        constructor kwarg name -> (config_key, default_value).
    """
    if algo in _CLIENT_REGISTRY:
        return _CLIENT_REGISTRY[algo]
    if algo in _LWF_ALGORITHMS:
        return (FedLwFClient, {})
    # Default: CGoFedClient (works for cgofed, fedavg, fedprox, fedavg_ewc, etc.)
    return (CGoFedClient, {})


def _build_extra_kwargs(
    config: Dict[str, Any], spec: Dict[str, tuple]
) -> Dict[str, Any]:
    """Extract constructor kwargs from config using the spec mapping."""
    kwargs = {}
    for kwarg_name, (config_key, default) in spec.items():
        kwargs[kwarg_name] = config.get(config_key, default)
    return kwargs


def create_client(
    cid: int,
    X_train,
    y_train,
    config: Dict[str, Any],
):
    """
    Create a single federated client for the given algorithm.

    Args:
        cid: Client ID
        X_train: Training features tensor
        y_train: Training labels tensor
        config: Full training configuration dict (must contain "algorithm" key)

    Returns:
        A FederatedClient subclass instance
    """
    algo = config["algorithm"].lower()
    client_cls, extra_spec = _resolve_client_class(algo)
    extra_kwargs = _build_extra_kwargs(config, extra_spec)
    return client_cls(cid, X_train, y_train, **extra_kwargs)


def create_clients(
    client_data: Dict[int, Dict],
    config: Dict[str, Any],
    task_id: int = 0,
    new_classes: Optional[List[int]] = None,
) -> list:
    """
    Factory to create clients based on algorithm.

    Args:
        client_data: Dict mapping client_id -> {"X_train": tensor, "y_train": tensor}
        config: Training configuration dict with "algorithm" key
        task_id: Current task ID (for logging)
        new_classes: List of new class indices (for logging)

    Returns:
        List of algorithm-specific client instances
    """
    clients = []

    for cid in sorted(client_data.keys()):
        data = client_data[cid]
        X, y = data["X_train"], data["y_train"]

        # Debug: Data distribution per client
        unique, counts = np.unique(y.numpy(), return_counts=True)
        dist_str = ", ".join([f"cls{c}:{n}" for c, n in zip(unique, counts)])
        print(
            f"  DEBUG[3]: Client {cid} | n_samples={len(y)} | distribution: {dist_str}"
        )

        client = create_client(cid, X, y, config)
        clients.append(client)

    return clients


def get_or_create_persistent_client(
    cid: int,
    data: Dict[str, Any],
    config: Dict[str, Any],
    persistent_clients: Dict[int, Any],
) -> Any:
    """
    Get an existing persistent client or create a new one.

    This handles the persistent client management pattern where
    clients maintain state across tasks (e.g., replay buffers, snapshots).

    Args:
        cid: Client ID
        data: {"X_train": tensor, "y_train": tensor} for this client
        config: Training configuration
        persistent_clients: Mutable dict of existing clients (will be updated in-place)

    Returns:
        The client instance (either existing or newly created)
    """
    if cid not in persistent_clients:
        persistent_clients[cid] = create_client(
            cid, data["X_train"], data["y_train"], config
        )
    return persistent_clients[cid]


def update_client_data(
    client,
    data: Dict[str, Any],
    task_id: int,
    new_classes: List[int],
):
    """
    Update a persistent client with new task data.

    Clients with `set_task_data()` (e.g., FedCBDR, FedLwF) use that method.
    Stateless clients get their data replaced directly.

    Args:
        client: The client instance to update
        data: {"X_train": tensor, "y_train": tensor}
        task_id: Current task ID
        new_classes: List of new class indices for this task
    """
    if hasattr(client, "set_task_data"):
        client.set_task_data(data["X_train"], data["y_train"], task_id, new_classes)
    else:
        # Stateless clients: replace data directly
        client.X_train = data["X_train"]
        client.y_train = data["y_train"]
        client.num_samples = len(data["y_train"])
