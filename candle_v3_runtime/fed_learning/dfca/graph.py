"""
DFCA Graph - Communication topology building.
"""

import random
from typing import Dict, List, Any


def build_erdos_renyi_graph(
    node_ids: List[int],
    connectivity: float = 0.15,
    seed: int = 42,
    ensure_connectivity: bool = True,
) -> Dict[int, List[int]]:
    """
    Build Erdos-Renyi random graph for DFCA communication.

    Args:
        node_ids: List of node IDs.
        connectivity: Edge probability p (default 0.15 per paper).
        seed: Random seed for reproducibility.
        ensure_connectivity: If True, connect isolated nodes to prevent no-neighbor nodes.

    Returns:
        Dict[node_id -> List[neighbor_ids]] — undirected adjacency list.
    """
    rng = random.Random(seed)
    n = len(node_ids)

    neighbors: Dict[int, List[int]] = {cid: [] for cid in node_ids}

    if n <= 1:
        return neighbors

    for i in range(n):
        for j in range(i + 1, n):
            cid_i = node_ids[i]
            cid_j = node_ids[j]
            if rng.random() < connectivity:
                neighbors[cid_i].append(cid_j)
                neighbors[cid_j].append(cid_i)

    if ensure_connectivity:
        for cid in node_ids:
            if not neighbors[cid]:
                other = node_ids[(node_ids.index(cid) + 1) % n]
                neighbors[cid].append(other)
                neighbors[other].append(cid)

    for cid in neighbors:
        neighbors[cid] = sorted(neighbors[cid])

    return neighbors


def build_graph_summary(
    neighbors: Dict[int, List[int]],
    node_ids: List[int],
) -> Dict[str, Any]:
    """
    Build graph statistics summary for logging.

    Returns:
        Dict with num_nodes, num_edges, avg_degree, min_degree, max_degree,
        isolated_count.
    """
    degrees = [len(neighbors.get(cid, [])) for cid in node_ids]
    num_edges = sum(len(neighbors.get(cid, [])) for cid in node_ids) // 2

    return {
        "num_nodes": len(node_ids),
        "num_edges": num_edges,
        "avg_degree": sum(degrees) / max(1, len(degrees)),
        "min_degree": min(degrees) if degrees else 0,
        "max_degree": max(degrees) if degrees else 0,
        "isolated_count": sum(1 for d in degrees if d == 0),
    }
