"""Local ownership is never inferred from rank, router, or a broadcast model."""
from copy import deepcopy
import numpy as np
from .contracts import SCHEMA, PeerAnnouncement


def state(model, owner):
    current = getattr(model, 'candle_v3_state', None)
    if current is None or current.get('owner') != int(owner):
        # Representative bootstrap may clone weights, but never local ownership.
        inherited = deepcopy((current or {}).get('peer_acquired', {}))
        current = dict(schema=SCHEMA, owner=int(owner), local_seen={}, peer_acquired=inherited,
                       evidence={}, completed_tasks=[], wire_events=[])
        model.candle_v3_state = current
    if current.get('schema') != SCHEMA:
        raise ValueError('Unknown CANDLE V3 knowledge schema')
    return current


def label_counts(labels):
    if hasattr(labels, 'detach'):
        labels = labels.detach().cpu().numpy()
    values, counts = np.unique(np.asarray(labels, dtype=np.int64), return_counts=True)
    return {int(c): int(n) for c, n in zip(values, counts)}


def record_local_update(model, owner, task, labels, optimizer_steps, *, successful_classes):
    record = state(model, owner)
    if int(optimizer_steps) <= 0:
        return
    counts = label_counts(labels)
    task_record = record['local_seen'].setdefault(str(int(task)), {})
    for c, n in counts.items():
        if c not in successful_classes:
            continue
        entry = task_record.setdefault(str(c), dict(base_support_count=n, positive_optimizer_rounds=0))
        if entry['base_support_count'] != n:
            raise ValueError('Local task support changed during the same task')
        entry['positive_optimizer_rounds'] += 1


def announce(model, detector, owner, task, current_labels, preprocessing_sha256):
    record = state(model, owner)
    seen = sorted({int(c) for entries in record['local_seen'].values() for c in entries})
    registered = sorted({int(c) for entries in detector.episode_classes.values() for c in entries})
    acquired = sorted(int(c) for c in record['peer_acquired'])
    shape = tuple(model.fc2.weight.shape)
    architecture = f'denice:{getattr(model,"architecture_version",1)}:fc2={shape}'
    return PeerAnnouncement(int(owner), int(task), tuple(sorted(label_counts(current_labels).items())),
                            tuple(seen), tuple(registered), tuple(acquired),
                            preprocessing_sha256, architecture)
