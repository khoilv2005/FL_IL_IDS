"""Deterministic complete linkage; K is an output, not an input."""
from dataclasses import dataclass

import numpy as np

from .contracts import state_hash


@dataclass(frozen=True)
class SimilaritySnapshot:
    client_ids: tuple[int, ...]
    scores: tuple[tuple[float, ...], ...]
    task: int
    data_task: int
    source: str
    reference_hash: str

    def validate(self):
        ids, values = self.client_ids, np.asarray(self.scores, dtype=np.float64)
        if not ids or len(set(ids)) != len(ids) or any(type(c) is not int or c < 0 for c in ids):
            raise ValueError('Invalid active client population')
        if self.task < 0 or self.data_task != self.task or not self.reference_hash:
            raise ValueError('Similarity must bind current data task and reference')
        if self.source not in {'task0_base_class_overlap', 'current_base_context_capsules'}:
            raise ValueError('Unauthorized similarity source')
        if self.source == 'task0_base_class_overlap' and self.task != 0:
            raise ValueError('Bootstrap overlap is only defined for Task 0')
        if values.shape != (len(ids), len(ids)) or not np.isfinite(values).all():
            raise ValueError('Invalid similarity matrix shape/values')
        if not np.allclose(values, values.T, rtol=0, atol=1e-12) or not np.all(np.diag(values) == 1):
            raise ValueError('Similarity must be symmetric with unit diagonal')
        if np.any(values < -1) or np.any(values > 1):
            raise ValueError('Similarity outside [-1,1]')
        return values


@dataclass(frozen=True)
class PartitionResult:
    task: int
    clusters: tuple[tuple[int, ...], ...]
    threshold: float
    source: str
    rejected_merges: tuple[tuple[tuple[int, ...], tuple[int, ...], str], ...]
    reference_hash: str
    similarity_hash: str

    @property
    def K(self):
        return len(self.clusters)

    def validate(self, active_clients):
        flat = [c for group in self.clusters for c in group]
        if self.task < 0 or not np.isfinite(self.threshold) or not -1 <= self.threshold <= 1:
            raise ValueError('Invalid partition time/threshold')
        if not self.reference_hash or not self.similarity_hash:
            raise ValueError('Missing partition provenance')
        if tuple(sorted(self.clusters)) != self.clusters or any(c < 0 for c in flat):
            raise ValueError('Partition must have canonical ordering and valid IDs')
        if not self.clusters or any(not group or tuple(sorted(set(group))) != group for group in self.clusters):
            raise ValueError('Invalid group')
        if len(flat) != len(set(flat)) or sorted(flat) != sorted(active_clients):
            raise ValueError('Partition must be disjoint and exhaustive')


def bootstrap_similarity(current_base_counts, *, data_task, task_classes, reference_hash):
    """Task0 fallback uses current BASE metadata, never future ownership/prototypes."""
    if data_task != 0:
        raise ValueError('Task0 bootstrap cannot read another task')
    ids = tuple(sorted(current_base_counts))
    supports = []
    for cid in ids:
        counts = current_base_counts[cid]
        if any(c not in task_classes or n < 0 for c, n in counts.items()):
            raise ValueError('Class/count outside authorized Task0 BASE metadata')
        supports.append({c for c, n in counts.items() if n > 0})
        if not supports[-1]:
            raise ValueError('Active training client must have usable BASE data')
    scores = np.eye(len(ids))
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            scores[i, j] = scores[j, i] = len(supports[i] & supports[j]) / len(supports[i] | supports[j])
    result = SimilaritySnapshot(ids, tuple(map(tuple, scores.tolist())), 0, 0,
                                'task0_base_class_overlap', reference_hash)
    result.validate()
    return result


def partition_clients(snapshot, *, threshold=0.8, merge_check=None):
    values = snapshot.validate()
    if not np.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError('Invalid threshold')
    index = {cid: i for i, cid in enumerate(snapshot.client_ids)}
    groups = [(cid,) for cid in sorted(index)]
    rejected = set()
    while True:
        candidates = []
        for i, left in enumerate(groups):
            for right in groups[i + 1:]:
                score = min(values[index[a], index[b]] for a in left for b in right)
                if score >= threshold:
                    allowed, reason = merge_check(left, right) if merge_check else (True, '')
                    if allowed:
                        candidates.append((-score, left, right))
                    else:
                        if not reason:
                            raise ValueError('Rejected merge requires a reason')
                        rejected.add((left, right, reason))
        if not candidates:
            break
        _, left, right = min(candidates)
        groups = sorted([g for g in groups if g not in (left, right)] + [tuple(sorted(left + right))])
    canonical_ids = tuple(sorted(index))
    canonical_scores = tuple(tuple(float(values[index[a], index[b]]) for b in canonical_ids) for a in canonical_ids)
    similarity_hash = state_hash(dict(client_ids=canonical_ids, scores=canonical_scores,
        task=snapshot.task, source=snapshot.source, reference_hash=snapshot.reference_hash))
    result = PartitionResult(snapshot.task, tuple(groups), float(threshold), snapshot.source, tuple(sorted(rejected)),
                             snapshot.reference_hash, similarity_hash)
    result.validate(snapshot.client_ids)
    return result


class PartitionBook:
    """One immutable partition per task; no reclustering during a task."""
    def __init__(self):
        self.partitions = {}

    def freeze(self, result, active_clients):
        result.validate(active_clients)
        previous = self.partitions.get(result.task)
        if previous is not None and previous != result:
            raise ValueError('Membership/partition is already locked for this task')
        self.partitions[result.task] = result

    def snapshot(self):
        from dataclasses import asdict
        return [asdict(self.partitions[t]) for t in sorted(self.partitions)]

    @classmethod
    def restore(cls, payload):
        book = cls()
        for item in payload:
            result = PartitionResult(**item)
            if result.task in book.partitions:
                raise ValueError('Duplicate task partition')
            book.freeze(result, [cid for group in result.clusters for cid in group])
        return book
