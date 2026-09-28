import pytest
from fed_learning.training.denice_retention import retention_summary


def metrics(cid, task, value):
    return {'per_client': {str(cid): {'per_task': {str(task): {'accuracy': value, 'sample_count': 10}}}}}


def test_match_clients_and_take_previous_best_not_last():
    history = [{'task': 0, **metrics(1, 0, .9)}, {'task': 1, **metrics(1, 0, .8)}]
    current = metrics(1, 0, .7)
    current['per_client'].update(metrics(2, 0, .2)['per_client'])
    result = retention_summary(history, current, 2)
    assert result['avg_forgetting'] == pytest.approx(.2)
    assert result['forgetting_matched_client_task_count'] == 1
    assert result['per_task_accuracy'][0] == pytest.approx(.45)


def test_missing_history_is_unknown_and_backward_transfer_can_be_negative():
    assert retention_summary([], metrics(1, 0, .9), 0)['avg_forgetting'] is None
    result = retention_summary([{'task': 0, **metrics(1, 0, .8)}], metrics(1, 0, .9), 1)
    assert result['avg_forgetting'] == pytest.approx(-.1)
