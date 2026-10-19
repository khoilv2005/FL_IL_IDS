"""Coordinator election has no effect on aggregation weights."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ElectionResult:
    leader: int
    scores: tuple[tuple[int, int], ...]
    reason: str


def elect_leader(members, ledger, *, task, round=-1, incumbent=None, unavailable=()):
    ids = tuple(sorted(members))
    if not ids or len(set(ids)) != len(ids) or any(c < 0 for c in ids):
        raise ValueError('Members must be unique nonnegative client IDs')
    if incumbent is not None and incumbent not in ids:
        raise ValueError('Incumbent is outside cluster')
    scores = tuple((cid, len(ledger.learned_classes(cid, task=task, round=round))) for cid in ids)
    eligible = [(cid, score) for cid, score in scores if cid not in unavailable]
    if not eligible:
        raise ValueError('No available coordinator')
    if incumbent is not None and incumbent not in unavailable:
        return ElectionResult(incumbent, scores, 'incumbent_retained_within_task')
    leader = min(eligible, key=lambda item: (-item[1], item[0]))[0]
    return ElectionResult(leader, scores, 'failover' if incumbent is not None else 'local_class_count')
