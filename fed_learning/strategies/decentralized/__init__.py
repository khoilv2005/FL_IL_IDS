"""
DeNICE decentralized strategy components (plan sections 2.3 - 2.5).

These modules implement the decentralized half of the DeNICE plan:

    - ``denice_capsule``     : NICE Context Capsule construction (section 2.3).
    - ``denice_clustering``  : context-aware similarity + Dynamic-K AP clustering
                               (section 2.4 + ``ap_nice_fl_pseudocode.md`` Alg. 1).
    - ``denice_aggregation`` : age-aware decentralized aggregation + adapter
                               aggregation (section 2.5 + Alg. 2 phase 6).

They are framework-agnostic helpers (operate on capsules / numpy / torch state
dicts) so they can be driven by any coordinator/runner without a central server.
"""

from .denice_capsule import ContextCapsule, build_context_capsule
from .denice_clustering import (
    ClusteringConfig,
    SimilarityWeights,
    class_prototype_similarity,
    context_similarity,
    label_overlap,
    build_similarity_matrix,
    affinity_propagation,
    dynamic_ap_cluster,
    collaboration_group,
    silhouette_score,
)
from .denice_aggregation import (
    AggregationConfig,
    aggregation_weights,
    age_aware_aggregate,
    aggregate_adapters,
    build_compatible_mask,
    merge_neuron_ages,
)

__all__ = [
    "ContextCapsule",
    "build_context_capsule",
    "ClusteringConfig",
    "SimilarityWeights",
    "class_prototype_similarity",
    "context_similarity",
    "label_overlap",
    "build_similarity_matrix",
    "affinity_propagation",
    "dynamic_ap_cluster",
    "collaboration_group",
    "silhouette_score",
    "AggregationConfig",
    "aggregation_weights",
    "age_aware_aggregate",
    "aggregate_adapters",
    "build_compatible_mask",
    "merge_neuron_ages",
]
