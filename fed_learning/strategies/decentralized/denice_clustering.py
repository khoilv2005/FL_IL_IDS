"""
Context-aware Dynamic-K clustering for DeNICE.

Implements plan section 2.4 and ``ap_nice_fl_pseudocode.md`` Algorithm 1.

Similarity between client i and neighbor j (plan / Đề xuất section 6)::

    s_ij = l1*cos(P_i,P_j) + l2*J(M_i,M_j) + l3*O(Y_i,Y_j)
         + l4*C(H_i,H_j)   + l5*R_j        - l6*D(Delta_i,Delta_j)
         (+ l_imp*cos(A_i,A_j))   # extra importance term from the AP pseudocode

Affinity Propagation is used so the number of clusters ``K_t`` emerges
dynamically. Silhouette score validates the clustering; low quality -> fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .denice_capsule import ContextCapsule, CAPSULE_LAYERS

LARGE_NEG = -1e9


@dataclass
class SimilarityWeights:
    """Weights for the context-aware similarity (plan section 2.4)."""

    proto: float = 0.30        # lambda1 - cos(P_i, P_j)
    age: float = 0.20          # lambda2 - Jaccard(M_i, M_j)
    label: float = 0.20        # lambda3 - label overlap
    capacity: float = 0.10     # lambda4 - capacity compatibility
    reliability: float = 0.10  # lambda5 - neighbor reliability
    update: float = 0.10       # lambda6 - update distance (subtracted)
    importance: float = 0.10   # extra - cos(A_i, A_j)


@dataclass
class ClusteringConfig:
    """AP + silhouette parameters (Algorithm 1)."""

    damping: float = 0.7
    t_max: int = 100
    conv_iter: int = 15
    theta_s: float = 0.5       # silhouette threshold
    delta_sim: float = 0.0      # min context similarity for a collaboration edge
    beta: float = 0.0           # preference weight on data-rich/reliable clients
    epsilon: float = 1e-8


# ----------------------------------------------------------------------------
# Similarity primitives
# ----------------------------------------------------------------------------
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


def context_similarity(
    cap_i: ContextCapsule, cap_j: ContextCapsule, weights: SimilarityWeights
) -> float:
    """s_ij (plan section 2.4)."""
    proto_sim = _cosine(cap_i.proto_vector(), cap_j.proto_vector())
    age_sim = _jaccard_binary(cap_i.age_mask_vector(), cap_j.age_mask_vector())
    imp_sim = _cosine(cap_i.importance_vector(), cap_j.importance_vector())
    lab_sim = label_overlap(cap_i.label_set, cap_j.label_set)
    cap_sim = capacity_compatibility(cap_i.capacity_vector(), cap_j.capacity_vector())
    rel = float(cap_j.reliability)
    upd = update_distance(cap_i.update_summary, cap_j.update_summary)

    return (
        weights.proto * proto_sim
        + weights.age * age_sim
        + weights.importance * imp_sim
        + weights.label * lab_sim
        + weights.capacity * cap_sim
        + weights.reliability * rel
        - weights.update * upd
    )


def build_similarity_matrix(
    capsules: List[ContextCapsule],
    weights: SimilarityWeights,
    delta_sim: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (S, E_context). Off-diagonal entries below ``delta_sim`` are masked."""
    n = len(capsules)
    S = np.full((n, n), LARGE_NEG, dtype=np.float64)
    E = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            s = context_similarity(capsules[i], capsules[j], weights)
            if s > delta_sim:
                S[i, j] = s
                E[i, j] = 1
    return S, E


# ----------------------------------------------------------------------------
# Affinity Propagation (Algorithm 1, steps 3-6)
# ----------------------------------------------------------------------------
def affinity_propagation(
    S: np.ndarray,
    damping: float = 0.7,
    t_max: int = 200,
    conv_iter: int = 15,
    seed: int = 0,
) -> Tuple[np.ndarray, List[int]]:
    """Run AP on a (preference-filled) similarity matrix.

    Returns ``(labels, exemplars)`` where ``labels[i]`` is the cluster index of
    ``i`` (0..K-1) and ``exemplars`` are the chosen exemplar point indices.

    Follows the standard AP convergence: exemplars are points whose
    ``diag(A+R) > 0``; each point is then assigned to the exemplar with the
    highest similarity. A tiny noise is added to break symmetric degeneracies
    (same trick as scikit-learn).
    """
    n = S.shape[0]
    S = S.astype(np.float64).copy()
    # Break ties / symmetric oscillation.
    rng = np.random.default_rng(seed)
    noise = (np.finfo(np.float64).eps * np.abs(S) + 1e-12) * rng.standard_normal((n, n))
    S = S + noise

    R = np.zeros((n, n))
    A = np.zeros((n, n))
    prev_exemplars = None
    stable = 0

    for _ in range(t_max):
        # Responsibilities
        AS = A + S
        max1 = np.max(AS, axis=1, keepdims=True)
        idx1 = np.argmax(AS, axis=1)
        AS_masked = AS.copy()
        AS_masked[np.arange(n), idx1] = LARGE_NEG
        max2 = np.max(AS_masked, axis=1, keepdims=True)
        R_new = S - max1
        R_new[np.arange(n), idx1] = (S - max2)[np.arange(n), idx1]
        R = damping * R + (1 - damping) * R_new

        # Availabilities
        Rp = np.maximum(R, 0)
        np.fill_diagonal(Rp, np.diag(R))
        col_sum = np.sum(Rp, axis=0, keepdims=True)
        A_new = np.minimum(0, col_sum - Rp)
        diag_A = np.sum(np.maximum(R, 0), axis=0) - np.maximum(np.diag(R), 0)
        A_new[np.arange(n), np.arange(n)] = diag_A
        A = damping * A + (1 - damping) * A_new

        E_diag = np.diag(A) + np.diag(R)
        exemplars = np.where(E_diag > 0)[0]
        if prev_exemplars is not None and np.array_equal(exemplars, prev_exemplars):
            stable += 1
        else:
            stable = 0
        prev_exemplars = exemplars
        if stable >= conv_iter and len(exemplars) > 0:
            break

    E_diag = np.diag(A) + np.diag(R)
    exemplar_idx = np.where(E_diag > 0)[0]

    if len(exemplar_idx) == 0:
        # Degenerate: every point in one cluster.
        return np.zeros(n, dtype=int), [int(np.argmax(E_diag))]

    # Assign each point to the exemplar with the highest similarity.
    assign = np.argmax(S[:, exemplar_idx], axis=1)
    # Exemplars point to themselves.
    for c, k in enumerate(exemplar_idx):
        assign[k] = c
    labels = assign.astype(int)
    return labels, [int(k) for k in exemplar_idx]


def silhouette_score(features: np.ndarray, labels: np.ndarray, eps: float = 1e-8) -> float:
    """Average silhouette (Algorithm 1 step 7). Invalid (<2 clusters) -> nan."""
    unique = np.unique(labels)
    n = len(labels)
    if len(unique) < 2 or n < 2:
        return float("nan")

    # Pairwise euclidean distances.
    diff = features[:, None, :] - features[None, :, :]
    dist = np.sqrt(np.maximum(0.0, (diff ** 2).sum(axis=2)))

    s_vals = []
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        if same.any():
            a_i = float(dist[i, same].mean())
        else:
            a_i = 0.0
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


def _client_features(capsules: List[ContextCapsule]) -> np.ndarray:
    """f_i = concat(normalized proto/age/importance/capacity) (Algorithm 1 step 1)."""
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
    """Dynamic-K AP clustering over capsules (Algorithm 1).

    Returns dict with keys: ``labels``, ``exemplars``, ``K_t``, ``silhouette``,
    ``similarity``, ``edges``, ``valid``.
    """
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

    S, E = build_similarity_matrix(capsules, weights, config.delta_sim)

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

    features = _client_features(capsules)
    sil = silhouette_score(features, labels) if k_t >= 2 else float("nan")

    is_valid = k_t >= 2 and np.isfinite(sil) and sil >= config.theta_s

    return {
        "labels": labels,
        "exemplars": exemplars,
        "K_t": k_t,
        "silhouette": sil,
        "similarity": S,
        "edges": E,
        "valid": bool(is_valid),
    }


def collaboration_group(
    client_idx: int,
    labels: np.ndarray,
    neighbors: Optional[List[int]] = None,
) -> List[int]:
    """G_i^t = {i} ∪ {j | C[j]=C[i] and j in N_i} (Algorithm 2 phase 5)."""
    same = [int(j) for j in range(len(labels)) if labels[j] == labels[client_idx]]
    if neighbors is not None:
        nb = set(int(j) for j in neighbors) | {int(client_idx)}
        same = [j for j in same if j in nb]
    if client_idx not in same:
        same.append(int(client_idx))
    return sorted(set(same))
