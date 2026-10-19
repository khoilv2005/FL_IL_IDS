"""
Plexus - Decentralized Federated Learning without a Server.

Reference:
    Dhasade et al., "Practical Federated Learning without a Server",
    EuroMLSys 2025

Implementation of Algorithm 1 (DERIVE_SAMPLE) and Algorithm 2 (Push-Based Protocol).

Components:
- PlexusSampler: Consistent hashing peer sampling (Algorithm 1)
- PlexusAggregator: FedAvg with success fraction threshold
- PlexusOrchestrator: Protocol orchestration
- NodeWrapper: Peer node wrapper for simulation
- PlexusNode: Autonomous peer node for the protocol

Usage:
    from fed_learning.plexus import PlexusOrchestrator

    orchestrator = PlexusOrchestrator(
        node_ids=[0, 1, 2, 3, 4],
        node_data={i: (X_train, y_train) for i in range(5)},
        bandwidths={i: 1.0 for i in range(5)},
        model_template=model,
        sample_size=3,
        success_fraction=0.8,
    )

    orchestrator.run_rounds(num_rounds=10)
    global_params = orchestrator.get_global_params()
"""

from .sampler import PlexusSampler
from .aggregator import PlexusAggregator
from .orchestrator import PlexusOrchestrator, NodeWrapper
from .node import PlexusNode
from .runner import run_plexus_training

__all__ = [
    "PlexusSampler",
    "PlexusAggregator",
    "PlexusOrchestrator",
    "NodeWrapper",
    "PlexusNode",
    "run_plexus_training",
]