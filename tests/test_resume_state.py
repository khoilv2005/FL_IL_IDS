from collections import OrderedDict

import torch

from fed_learning.strategies.incremental.ewc import EWCTrainer
from fed_learning.training.resume_state import (
    restore_client_state,
    snapshot_client_state,
)


class _DummyBuffer:
    def __init__(self):
        self.total_samples = 0
        self.class_buffers = {}


class _DummyClient:
    def __init__(self):
        self.client_id = 0
        self.X_train = torch.randn(4, 2)
        self.y_train = torch.tensor([0, 1, 0, 1])
        self.num_samples = 4
        self.model = None
        self.device = None
        self.use_amp = False
        self.current_task = 0
        self.seen_classes = set()
        self.old_model = None
        self.old_model_state = OrderedDict()
        self.replay_buffer = _DummyBuffer()


def test_resume_state_roundtrip_for_nested_client_objects():
    source = _DummyClient()
    source.current_task = 2
    source.seen_classes = {0, 1, 2}
    source.old_model_state = OrderedDict(weight=torch.tensor([1.0, 2.0]))
    source.replay_buffer.total_samples = 7
    source.replay_buffer.class_buffers = {
        0: {"X": torch.ones(2, 3), "y": torch.zeros(2, dtype=torch.long)}
    }

    state = snapshot_client_state(source)

    target = _DummyClient()
    target.current_task = 99
    restore_client_state(target, state)

    assert target.current_task == 2
    assert target.seen_classes == {0, 1, 2}
    assert torch.equal(target.old_model_state["weight"], torch.tensor([1.0, 2.0]))
    assert target.replay_buffer.total_samples == 7
    assert 0 in target.replay_buffer.class_buffers
    assert torch.equal(
        target.replay_buffer.class_buffers[0]["X"], torch.ones(2, 3)
    )


def test_ewc_resume_state_rebuilds_latest_fisher(tmp_path):
    first_temp = tmp_path / "ewc_phase1"
    second_temp = tmp_path / "ewc_phase2"

    trainer = EWCTrainer(temp_dir=str(first_temp))
    trainer.current_task = 2
    trainer.seen_classes = {0, 1, 2}
    trainer.best_acc_per_task = {0: 0.9, 1: 0.8}
    trainer.current_acc_per_task = {0: 0.85, 1: 0.75}
    trainer.last_af = 0.05
    trainer._cached_fisher_acc = {"layer.weight": torch.ones(3)}
    trainer._cached_optimal_params = {"layer.weight": torch.zeros(3)}

    state = trainer.get_resume_state()

    restored = EWCTrainer(temp_dir=str(second_temp))
    restored.load_resume_state(state)
    restored.set_task(3, [3])

    fisher = restored._get_prev_fisher_acc()

    assert fisher is not None
    assert torch.equal(fisher["layer.weight"], torch.ones(3))
    assert 2 in restored.ewc_data
    assert restored.last_af == 0.05
    assert 3 in restored.seen_classes
