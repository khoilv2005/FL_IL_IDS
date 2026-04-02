"""
Plexus Decentralized Federated Learning Package.

Implements the truly decentralized Plexus protocol from:
    Dhasade et al., "Practical Federated Learning without a Server", EuroMLSys 2025

This package provides:
- PlexusSampler: Consistent-hashing peer sampling (Algorithm 1)
- PlexusNode: Autonomous peer node that can train and aggregate
- PlexusOrchestrator: Simulates the push-based protocol (Algorithm 2)
- PlexusIncrementalRunner: Task loop for decentralized incremental learning

Usage:
    from fed_learning.decentralized import PlexusOrchestrator, PlexusSampler

    # In task_loop.py with mode="decentralized":
    runner = PlexusIncrementalRunner(config)
    results = runner.run()
"""

from .sampler import PlexusSampler
from .node import PlexusNode
from .orchestrator import PlexusOrchestrator
from .runner import PlexusIncrementalRunner
from .metrics import PlexusMetrics

__all__ = [
    "PlexusSampler",
    "PlexusNode",
    "PlexusOrchestrator",
    "PlexusIncrementalRunner",
    "PlexusMetrics",
]