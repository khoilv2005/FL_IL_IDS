"""
Server Factory - Create algorithm-specific federated servers.

Maps algorithm names to their corresponding server classes.
"""

from typing import Dict, Any, List

from fed_learning.servers import (
    IncrementalServer,
    CGoFedServer,
    FedCBDRServer,
    FedLwFServer,
    DERServer,
    RNEServer,
    NICEServer,
    GLFCServer,
    ReFedServer,
)
from fed_learning.servers.dfca_server import DFCAServer


# Registry: algorithm name -> ServerClass
_SERVER_REGISTRY = {
    "fedcbdr": FedCBDRServer,
    "der": DERServer,
    "rne": RNEServer,
    "cgofed": CGoFedServer,
    "nice": NICEServer,
    "glfc": GLFCServer,
    "refed": ReFedServer,
    "dfca_il": DFCAServer,
}

# These algorithms use FedLwFServer
_LWF_ALGORITHMS = {"fedavg_lwf", "fedprox_lwf"}


def create_server(
    config: Dict[str, Any],
    clients: list,
    test_data: Dict,
    task_config: Dict[str, Any],
):
    """
    Factory to create the appropriate server for the given algorithm.

    Args:
        config: Training configuration with "algorithm" key
        clients: List of participating client instances
        test_data: {"X_test": tensor, "y_test": tensor}
        task_config: Task-specific configuration (merged from CONFIG + overrides)

    Returns:
        An IncrementalServer (or subclass) instance
    """
    algo = config["algorithm"].lower()

    if algo in _SERVER_REGISTRY:
        return _SERVER_REGISTRY[algo](clients, test_data, task_config)
    elif algo in _LWF_ALGORITHMS:
        return FedLwFServer(clients, test_data, task_config)
    else:
        # Default: IncrementalServer (for fedavg, fedprox, fedplus, ewc variants)
        return IncrementalServer(clients, test_data, task_config)
