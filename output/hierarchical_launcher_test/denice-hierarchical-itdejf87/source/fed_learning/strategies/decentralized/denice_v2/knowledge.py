"""Time-bounded local-learning provenance, independent of broadcasts."""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class LocalLearningEvent:
    event_id: str
    client_id: int
    task: int
    round: int
    successful_steps: int
    successful_class_counts: tuple[tuple[int, int], ...]
    data_task: int
    role: str = 'base'
    completed: bool = True


class ClassKnowledgeLedger:
    def __init__(self, task_classes):
        if sorted(task_classes) != list(range(len(task_classes))) or any(type(t) is not int for t in task_classes):
            raise ValueError('Task mapping must use contiguous integer task IDs')
        self.task_classes = {int(t): tuple(sorted(cs)) for t, cs in task_classes.items()}
        labels = [c for cs in self.task_classes.values() for c in cs]
        if not labels or any(type(c) is not int for c in labels) or sorted(labels) != list(range(len(labels))):
            raise ValueError('Requires disjoint contiguous class mapping')
        self.num_classes = len(labels)
        self.events = {}

    def record(self, event, *, at_task, at_round):
        if not isinstance(event, LocalLearningEvent):
            raise TypeError('Expected a local-learning event, not availability/broadcast state')
        if any(type(n) is not int for n in (event.client_id, event.task, event.round,
                event.successful_steps, event.data_task, at_task, at_round)) or type(event.completed) is not bool:
            raise ValueError('Training event IDs/times/counts must be integers')
        if not isinstance(event.event_id, str) or not event.event_id or event.client_id < 0 or event.task not in self.task_classes or event.round < 0:
            raise ValueError('Invalid event identity/time')
        if event.role != 'base' or event.data_task != event.task:
            raise ValueError('Only current-task BASE training can establish local ownership')
        if (event.task, event.round) > (at_task, at_round):
            raise ValueError('Future event is not authorized')
        counts = dict(event.successful_class_counts)
        if len(counts) != len(event.successful_class_counts) or event.successful_steps < 0:
            raise ValueError('Invalid training counts')
        if any(type(c) is not int or type(n) is not int or c not in self.task_classes[event.task] or n <= 0 for c, n in counts.items()):
            raise ValueError('Counts must be positive and belong to current task')
        if (event.successful_steps == 0) != (not counts):
            raise ValueError('Class evidence requires successful optimizer steps')
        previous = self.events.get(event.event_id)
        if previous is not None and previous != event:
            raise ValueError('Event ID reused with different evidence')
        self.events[event.event_id] = event

    def learned_classes(self, client_id, *, task, round=-1):
        return tuple(sorted({c for e in self.events.values()
            if e.client_id == client_id and e.completed and e.successful_steps > 0
            and (e.task, e.round) <= (task, round)
            for c, _ in e.successful_class_counts}))

    def matrix(self, client_ids, *, task, round=-1):
        ids = tuple(sorted(client_ids))
        if len(set(ids)) != len(ids) or any(type(cid) is not int or cid < 0 for cid in ids):
            raise ValueError('Duplicate client ID')
        matrix = np.zeros((len(ids), self.num_classes), dtype=np.uint8)
        for row, cid in enumerate(ids):
            matrix[row, list(self.learned_classes(cid, task=task, round=round))] = 1
        return ids, matrix

    def snapshot(self, *, task, round=-1):
        return {'task_classes': self.task_classes, 'events': [asdict(e) for e in
            sorted(self.events.values(), key=lambda e: e.event_id)
            if (e.task, e.round) <= (task, round)]}

    @classmethod
    def restore(cls, payload):
        ledger = cls(payload['task_classes'])
        for item in payload['events']:
            item = dict(item)
            item['successful_class_counts'] = tuple(tuple(pair) for pair in item['successful_class_counts'])
            event = LocalLearningEvent(**item)
            ledger.record(event, at_task=event.task, at_round=event.round)
        return ledger
