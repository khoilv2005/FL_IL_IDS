"""Complementary lookup on an explicit teaching graph; no K=1 sentinel labels."""
from dataclasses import dataclass


@dataclass(frozen=True)
class TeacherSelection:
    receiver: int
    donors: tuple
    targets: tuple
    covered_by: tuple


def select_teachers(receiver, announcements, allowed_peers, eligible_targets, *,
                    max_donors=2, max_targets=3, quality=None, reference_bytes=None):
    """Greedy marginal coverage, deterministic ties. Evidence checks follow lookup.

    quality, when supplied, must come from current non-test evidence. Absence is
    unmeasured, not competence=PASS. No native similarity threshold is bypassed
    in native aggregation; this function operates only on the teaching graph.
    """
    own = announcements[receiver]
    if max_donors < 1 or max_targets < 1:
        raise ValueError('Positive selection budgets required')
    quality, reference_bytes = quality or {}, reference_bytes or {}
    for value in quality.values():
        if not 0 <= value <= 1:
            raise ValueError('Invalid evidence quality')
    candidates = []
    missing = set(int(c) for c in eligible_targets)
    for donor in sorted(set(int(c) for c in allowed_peers) - {receiver}):
        peer = announcements.get(donor)
        if peer is None or peer.task != own.task:
            continue
        if (peer.preprocessing_sha256 != own.preprocessing_sha256 or
                peer.architecture != own.architecture):
            continue
        if peer.current_classes & missing:
            candidates.append(donor)
    selected, covered = [], set()
    while candidates and len(selected) < max_donors and len(covered) < max_targets:
        def rank(donor):
            new = (announcements[donor].current_classes & missing) - covered
            q = sum(quality.get((donor, c), 0.) for c in new)
            return (-len(new), -q, reference_bytes.get(donor, 0), donor)
        donor = min(candidates, key=rank)
        candidates.remove(donor)
        new = sorted((announcements[donor].current_classes & missing) - covered)
        if not new:
            break
        selected.append(donor)
        covered.update(new[:max_targets - len(covered)])
    mapping = tuple((c, tuple(d for d in selected if c in announcements[d].current_classes))
                    for c in sorted(covered))
    return TeacherSelection(receiver, tuple(selected), tuple(sorted(covered)), mapping)
