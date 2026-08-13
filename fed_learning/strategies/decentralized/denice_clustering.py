"""Context-aware dynamic clustering for DeNICE.

The proposal clusters clients from NICE Context Capsules:

    s_ij = l1*cos(P_i,c,P_j,c) + l2*J(M_i,M_j) + l3*O(Y_i,Y_j)
         + l4*C(H_i,H_j) + l5*R_j - l6*D(Delta_i,Delta_j)

Root-cause fixes implemented here:

* activation prototypes stay class-aware; class prototypes are not averaged
  across unrelated classes;
* saturated terms such as age/capacity are down-weighted when they have almost
  no variance across client pairs;
* the collaboration graph is sparse by default via adaptive threshold + mutual
  top-k, matching E_ij = 1 iff clients are context-compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .denice_capsule import CAPSULE_LAYERS, ContextCapsule

LARGE_NEG = -1e9


@dataclass
class SimilarityWeights:
    """Weights for the context-aware similarity (proposal section 6)."""

    proto: float = 0.35
    age: float = 0.10
    label: float = 0.25
    capacity: float = 0.05
    reliability: float = 0.10
    update: float = 0.10
    importance: float = 0.15


@dataclass
class ClusteringConfig:
    """AP + sparse context-graph parameters."""

    damping: float = 0.7
    t_max: int = 100
    conv_iter: int = 15
    theta_s: float = 0.5
    delta_sim: float = 0.0  # <= 0 derives adaptive threshold from similarities.
    beta: float = 0.0
    epsilon: float = 1e-8
    edge_top_k: int = 20
    edge_quantile: float = 0.40
    min_signal_std: float = 0.02


def _safe_l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(x))
    if n < eps:
        return np.zeros_like(x)
    return x / n


def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return 0.0
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _jaccard_binary(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return 0.0
    a = a > 0.5
    b = b > 0.5
    union = float((a | b).sum())
    if union == 0:
        return 0.0
    return float((a & b).sum()) / union


def label_overlap(y_i: List[int], y_j: List[int]) -> float:
    si, sj = set(int(c) for c in y_i), set(int(c) for c in y_j)
    union = si | sj
    if not union:
        return 0.0
    return len(si & sj) / len(union)


def capacity_compatibility(h_i: np.ndarray, h_j: np.ndarray, eps: float = 1e-8) -> float:
    if h_i.size == 0 or h_j.size == 0 or h_i.shape != h_j.shape:
        return 0.0
    dist = float(np.linalg.norm(h_i - h_j))
    denom = float(np.linalg.norm(h_i) + np.linalg.norm(h_j)) + eps
    return float(1.0 - dist / denom)


def update_distance(d_i: Optional[np.ndarray], d_j: Optional[np.ndarray]) -> float:
    if d_i is None or d_j is None:
        return 0.0
    d_i = _safe_l2_normalize(np.asarray(d_i).ravel())
    d_j = _safe_l2_normalize(np.asarray(d_j).ravel())
    if d_i.shape != d_j.shape:
        return 0.0
    return float(np.linalg.norm(d_i - d_j))


def class_prototype_similarity(cap_i: ContextCapsule, cap_j: ContextCapsule) -> float:
    """Compare P_i,c and P_j,c only on shared class slots."""
    pi = getattr(cap_i, "class_activation_prototypes", {}) or {}
    pj = getattr(cap_j, "class_activation_prototypes", {}) or {}
    if not pi or not pj:
        return _cosine(cap_i.proto_vector(), cap_j.proto_vector())

    shared = sorted(set(int(c) for c in pi) & set(int(c) for c in pj))
    if not shared:
        return 0.0

    sims: List[float] = []
    for cls_id in shared:
        li = pi.get(cls_id, {}) or {}
        lj = pj.get(cls_id, {}) or {}
        layer_sims = [
            _cosine(np.asarray(li[name]).ravel(), np.asarray(lj[name]).ravel())
            for name in CAPSULE_LAYERS
            if name in li and name in lj
        ]
        if layer_sims:
            sims.append(float(np.mean(layer_sims)))
    return float(np.mean(sims)) if sims else 0.0


def _similarity_components(cap_i: ContextCapsule, cap_j: ContextCapsule) -> Dict[str, float]:
    return {
        "proto": class_prototype_similarity(cap_i, cap_j),
        "age": _jaccard_binary(cap_i.age_mask_vector(), cap_j.age_mask_vector()),
        "importance": _cosine(cap_i.importance_vector(), cap_j.importance_vector()),
        "label": label_overlap(cap_i.label_set, cap_j.label_set),
        "capacity": capacity_compatibility(cap_i.capacity_vector(), cap_j.capacity_vector()),
        "reliability": float(cap_j.reliability),
        "update": update_distance(cap_i.update_summary, cap_j.update_summary),
    }


def _score_components(components: Dict[str, float], weights: SimilarityWeights) -> float:
    return (
        weights.proto * components["proto"]
        + weights.age * components["age"]
        + weights.importance * components["importance"]
        + weights.label * components["label"]
        + weights.capacity * components["capacity"]
        + weights.reliability * components["reliability"]
        - weights.update * components["update"]
    )


def context_similarity(
    cap_i: ContextCapsule, cap_j: ContextCapsule, weights: SimilarityWeights
) -> float:
    """s_ij (proposal section 6)."""
    return _score_components(_similarity_components(cap_i, cap_j), weights)


def _adaptive_weights(
    component_values: Dict[str, List[float]],
    weights: SimilarityWeights,
    min_signal_std: float,
) -> SimilarityWeights:
    raw = {
        "proto": weights.proto,
        "age": weights.age,
        "importance": weights.importance,
        "label": weights.label,
        "capacity": weights.capacity,
        "reliability": weights.reliability,
    }
    kept: Dict[str, float] = {}
    for name, weight in raw.items():
        vals = np.asarray(component_values.get(name, []), dtype=np.float64)
        if vals.size == 0 or float(np.std(vals)) >= float(min_signal_std):
            kept[name] = float(weight)
        else:
            kept[name] = 0.0

    total_raw = sum(raw.values())
    total_kept = sum(kept.values())
    if total_kept <= 1e-12:
        kept = raw
        total_kept = total_raw
    scale = total_raw / total_kept if total_kept > 0 else 1.0
    update_vals = np.asarray(component_values.get("update", []), dtype=np.float64)
    update_weight = (
        weights.update
        if update_vals.size == 0 or float(np.std(update_vals)) >= float(min_signal_std)
        else 0.0
    )
    return SimilarityWeights(
        proto=kept["proto"] * scale,
        age=kept["age"] * scale,
        label=kept["label"] * scale,
        capacity=kept["capacity"] * scale,
        reliability=kept["reliability"] * scale,
        update=update_weight,
        importance=kept["importance"] * scale,
    )


def _sparse_edges(scores: np.ndarray, threshold: float, top_k: int) -> np.ndarray:
    n = scores.shape[0]
    finite = np.isfinite(scores) & (scores > LARGE_NEG / 2)
    E = (scores > threshold) & finite
    np.fill_diagonal(E, False)
    if top_k and top_k > 0 and n > 1:
        k = min(int(top_k), n - 1)
        top = np.zeros((n, n), dtype=bool)
        for i in range(n):
            row = np.where(finite[i], scores[i], -np.inf)
            row[i] = -np.inf
            if np.isfinite(row).any():
                idx = np.argpartition(row, -k)[-k:]
                top[i, idx[np.isfinite(row[idx])]] = True
        E = E & top & top.T
    return E.astype(np.int8)


def build_similarity_matrix(
    capsules: List[ContextCapsule],
    weights: SimilarityWeights,
    delta_sim: float,
    config: Optional[ClusteringConfig] = None,
    *,
    return_details: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return sparse similarity matrix ``S`` and context graph ``E``.

    ``return_details=True`` additionally returns the dense, *adaptive-weighted*
    score matrix and its provenance.  The sparse matrix is deliberately not a
    suitable aggregation score source: non-edges are encoded as ``LARGE_NEG``.
    Keeping the dense score matrix alongside it lets aggregation use exactly
    the same adaptive similarity definition that formed the graph.
    """
    config = config or ClusteringConfig(delta_sim=delta_sim)
    n = len(capsules)
    raw_scores = np.full((n, n), LARGE_NEG, dtype=np.float64)
    component_values: Dict[str, List[float]] = {
        "proto": [],
        "age": [],
        "importance": [],
        "label": [],
        "capacity": [],
        "reliability": [],
        "update": [],
    }
    pair_components: Dict[Tuple[int, int], Dict[str, float]] = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            comps = _similarity_components(capsules[i], capsules[j])
            pair_components[(i, j)] = comps
            for key, value in comps.items():
                component_values[key].append(float(value))

    effective = _adaptive_weights(component_values, weights, config.min_signal_std)
    for (i, j), comps in pair_components.items():
        raw_scores[i, j] = _score_components(comps, effective)

    finite_scores = raw_scores[(raw_scores > LARGE_NEG / 2) & np.isfinite(raw_scores)]
    if finite_scores.size == 0:
        threshold = None
        E = np.zeros((n, n), dtype=np.int8)
    else:
        if delta_sim > 0:
            threshold = float(delta_sim)
        else:
            q = min(0.99, max(0.0, float(config.edge_quantile)))
            threshold = float(np.quantile(finite_scores, q))
        E = _sparse_edges(raw_scores, threshold, int(config.edge_top_k))

    S = np.full((n, n), LARGE_NEG, dtype=np.float64)
    S[E.astype(bool)] = raw_scores[E.astype(bool)]
    if not return_details:
        return S, E
    return S, E, {
        "effective_similarity": raw_scores,
        "effective_weights": {
            name: float(getattr(effective, name))
            for name in (
                "proto", "age", "importance", "label", "capacity",
                "reliability", "update",
            )
        },
        "similarity_threshold": threshold,
    }


def affinity_propagation(
    S: np.ndarray,
    damping: float = 0.7,
    t_max: int = 200,
    conv_iter: int = 15,
    seed: int = 0,
) -> Tuple[np.ndarray, List[int]]:
    """Run AP on a preference-filled similarity matrix."""
    n = S.shape[0]
    S = S.astype(np.float64).copy()
    rng = np.random.default_rng(seed)
    noise = (np.finfo(np.float64).eps * np.abs(S) + 1e-12) * rng.standard_normal((n, n))
    S = S + noise

    R = np.zeros((n, n))
    A = np.zeros((n, n))
    prev_exemplars = None
    stable = 0
    for _ in range(t_max):
        AS = A + S
        max1 = np.max(AS, axis=1, keepdims=True)
        idx1 = np.argmax(AS, axis=1)
        AS_masked = AS.copy()
        AS_masked[np.arange(n), idx1] = LARGE_NEG
        max2 = np.max(AS_masked, axis=1, keepdims=True)
        R_new = S - max1
        R_new[np.arange(n), idx1] = (S - max2)[np.arange(n), idx1]
        R = damping * R + (1 - damping) * R_new

        Rp = np.maximum(R, 0)
        np.fill_diagonal(Rp, np.diag(R))
        col_sum = np.sum(Rp, axis=0, keepdims=True)
        A_new = np.minimum(0, col_sum - Rp)
        diag_A = np.sum(np.maximum(R, 0), axis=0) - np.maximum(np.diag(R), 0)
        A_new[np.arange(n), np.arange(n)] = diag_A
        A = damping * A + (1 - damping) * A_new

        e_diag = np.diag(A) + np.diag(R)
        exemplars = np.where(e_diag > 0)[0]
        if prev_exemplars is not None and np.array_equal(exemplars, prev_exemplars):
            stable += 1
        else:
            stable = 0
        prev_exemplars = exemplars
        if stable >= conv_iter and len(exemplars) > 0:
            break

    e_diag = np.diag(A) + np.diag(R)
    exemplar_idx = np.where(e_diag > 0)[0]
    if len(exemplar_idx) == 0:
        return np.zeros(n, dtype=int), [int(np.argmax(e_diag))]

    assign = np.argmax(S[:, exemplar_idx], axis=1)
    for c, k in enumerate(exemplar_idx):
        assign[k] = c
    return assign.astype(int), [int(k) for k in exemplar_idx]


def silhouette_score(features: np.ndarray, labels: np.ndarray, eps: float = 1e-8) -> float:
    """Euclidean silhouette kept for backward-compatible tests/debug."""
    unique = np.unique(labels)
    n = len(labels)
    if len(unique) < 2 or n < 2:
        return float("nan")
    diff = features[:, None, :] - features[None, :, :]
    dist = np.sqrt(np.maximum(0.0, (diff ** 2).sum(axis=2)))
    return _silhouette_from_distance(dist, labels, eps)


def _silhouette_from_distance(dist: np.ndarray, labels: np.ndarray, eps: float = 1e-8) -> float:
    unique = np.unique(labels)
    s_vals = []
    for i in range(len(labels)):
        same = labels == labels[i]
        same[i] = False
        a_i = float(dist[i, same].mean()) if same.any() else 0.0
        b_i = np.inf
        for c in unique:
            if c == labels[i]:
                continue
            mask = labels == c
            if mask.any():
                b_i = min(b_i, float(dist[i, mask].mean()))
        if not np.isfinite(b_i):
            b_i = 0.0
        denom = max(a_i, b_i)
        s_vals.append(0.0 if denom < eps else (b_i - a_i) / denom)
    return float(np.mean(s_vals))


def silhouette_score_from_similarity(S: np.ndarray, labels: np.ndarray, eps: float = 1e-8) -> float:
    """Silhouette using the same context graph used by clustering."""
    unique = np.unique(labels)
    n = len(labels)
    if len(unique) < 2 or n < 2:
        return float("nan")
    finite = (S > LARGE_NEG / 2) & np.isfinite(S)
    vals = S[finite]
    if vals.size == 0:
        return float("nan")
    lo, hi = float(vals.min()), float(vals.max())
    sim = np.zeros_like(S, dtype=np.float64)
    if hi - lo < eps:
        sim[finite] = 1.0
    else:
        sim[finite] = (S[finite] - lo) / (hi - lo)
    sim = np.maximum(sim, sim.T)
    np.fill_diagonal(sim, 1.0)
    return _silhouette_from_distance(1.0 - sim, labels, eps)


def _client_features(capsules: List[ContextCapsule]) -> np.ndarray:
    rows = []
    for cap in capsules:
        rows.append(
            np.concatenate(
                [
                    _safe_l2_normalize(cap.proto_vector()),
                    _safe_l2_normalize(cap.age_mask_vector()),
                    _safe_l2_normalize(cap.importance_vector()),
                    _safe_l2_normalize(cap.capacity_vector()),
                ]
            )
        )
    if not rows:
        return np.zeros((0, 0))
    width = max(r.size for r in rows)
    return np.stack([np.pad(r, (0, width - r.size)) for r in rows])


def dynamic_ap_cluster(
    capsules: List[ContextCapsule],
    config: Optional[ClusteringConfig] = None,
    weights: Optional[SimilarityWeights] = None,
) -> Dict:
    """Dynamic-K AP clustering over NICE Context Capsules."""
    config = config or ClusteringConfig()
    weights = weights or SimilarityWeights()
    n = len(capsules)
    if n < 2:
        return {
            "labels": np.zeros(n, dtype=int),
            "exemplars": [0] if n == 1 else [],
            "K_t": 1 if n == 1 else 0,
            "silhouette": float("nan"),
            "similarity": np.zeros((n, n)),
            "edges": np.zeros((n, n), dtype=np.int8),
            "valid": False,
        }

    built = build_similarity_matrix(
        capsules, weights, config.delta_sim, config=config, return_details=True
    )
    # Preserve compatibility with callers/tests that monkeypatch the legacy
    # two-value builder.  Real clustering always returns the provenance below.
    if len(built) == 3:
        S, E, similarity_details = built
    else:
        S, E = built
        similarity_details = {
            "effective_similarity": S.copy(),
            "effective_weights": {
                name: float(getattr(weights, name))
                for name in (
                    "proto", "age", "importance", "label", "capacity",
                    "reliability", "update",
                )
            },
            "similarity_threshold": None,
        }
    valid = S[E.astype(bool)]
    p0 = float(np.median(valid)) if valid.size > 0 else 0.0
    counts = np.array([max(1, c.sample_count) for c in capsules], dtype=np.float64)
    rels = np.array([max(0.0, c.reliability) for c in capsules], dtype=np.float64)
    q = counts * rels
    q = q / (q.max() + config.epsilon) if q.max() > 0 else q
    S_ap = S.copy()
    for i in range(n):
        S_ap[i, i] = p0 + config.beta * q[i]

    labels, exemplars = affinity_propagation(
        S_ap, config.damping, config.t_max, config.conv_iter
    )
    k_t = len(set(labels.tolist()))
    sil = silhouette_score_from_similarity(S, labels) if k_t >= 2 else float("nan")
    is_valid = k_t >= 2 and np.isfinite(sil) and sil >= config.theta_s

    return {
        "labels": labels,
        "exemplars": exemplars,
        "K_t": k_t,
        "silhouette": sil,
        "similarity": S,
        "effective_similarity": similarity_details["effective_similarity"],
        "effective_weights": similarity_details["effective_weights"],
        "similarity_threshold": similarity_details["similarity_threshold"],
        "edges": E,
        "valid": bool(is_valid),
    }


def collaboration_group(
    client_idx: int,
    labels: np.ndarray,
    neighbors: Optional[List[int]] = None,
) -> List[int]:
    """G_i = {j | same cluster and context-compatible}, always including i."""
    same = [int(j) for j in range(len(labels)) if labels[j] == labels[client_idx]]
    if neighbors is not None:
        nb = set(int(j) for j in neighbors) | {int(client_idx)}
        same = [j for j in same if j in nb]
    if client_idx not in same:
        same.append(int(client_idx))
    return sorted(set(same))
