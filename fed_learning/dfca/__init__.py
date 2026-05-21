"""
DFCA - Pure Decentralized Federated Clustering Algorithm.

Based on: Dhasade et al., "DFCA: Decentralized Federated Clustering Algorithm"
A fully decentralized clustered FL where each node maintains k cluster models
and peer-to-peer aggregation via sequential running average.

NO incremental learning. NO task loop. NO central server.
"""

from .client import DFCANode
from .aggregator import DFCAAggregator
from .runner import (
    run_dfca_training,
    build_dfca_checkpoint,
    find_latest_checkpoint,
)
from .graph import build_erdos_renyi_graph, build_graph_summary
from .evaluation import evaluate_ensemble_average, evaluate_representative_clusters, evaluate_oracle

__all__ = [
    "DFCANode",
    "DFCAAggregator",
    "run_dfca_training",
    "build_dfca_checkpoint",
    "find_latest_checkpoint",
    "build_erdos_renyi_graph",
    "build_graph_summary",
    "evaluate_ensemble_average",
    "evaluate_representative_clusters",
    "evaluate_oracle",
]
