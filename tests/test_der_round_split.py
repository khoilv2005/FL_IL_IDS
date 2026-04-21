from fed_learning.training.local_task_loop import _resolve_der_round_split as resolve_local
from fed_learning.training.task_loop import _resolve_der_round_split as resolve_fed


def test_der_rounds_use_total_budget_when_stage_rounds_not_explicit():
    config = {"rounds_per_task": 20}
    assert resolve_local(config) == (12, 8)
    assert resolve_fed(config) == (12, 8)


def test_der_rounds_keep_explicit_stage_config():
    config = {"rounds_per_task": 20, "der_stage1_rounds": 3, "der_stage2_rounds": 2}
    assert resolve_local(config) == (3, 2)
    assert resolve_fed(config) == (3, 2)


def test_der_rounds_keep_partial_explicit_config():
    config = {"rounds_per_task": 20, "der_stage1_rounds": 7}
    assert resolve_local(config) == (7, 3)
    assert resolve_fed(config) == (7, 3)
