"""RNE local strategy."""

from typing import List

from .der import DERTrainer


class RNETrainer(DERTrainer):
    """
    Trainer for Recurrent Network Expansion.

    Loss schedule follows DER's two-stage training, while RNEModel changes the
    representation path to recurrent experts and decoupled classifier heads.
    """

    def set_task(self, task_id: int, new_classes: List[int]):
        self.old_classes = list(self.seen_classes)
        self.new_classes = list(new_classes)
        self.current_task = task_id
        self.seen_classes.update(new_classes)
        self.current_batch = 0
        print(
            f"  RNETrainer Task {task_id}: "
            f"old={len(self.old_classes)}, new={len(new_classes)}"
        )
