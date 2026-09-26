from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fed_learning.strategies.decentralized.denice_transfer import transfer_config, select_peer_transfer


def controls():
    return transfer_config({'denice_transfer_enabled': True, 'denice_aggregation_update_mode': 'local_delta'})


def problem():
    model = nn.Linear(1, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[-1.], [1.]]))
    x, y = torch.tensor([[-1.], [1.]]), torch.tensor([0, 1])
    return model, deepcopy(model.state_dict()), x, y


def test_rejects_harmful_peer_and_preserves_rng_and_model():
    model, local, x, y = problem()
    peer = {'weight': -local['weight']}
    rng = torch.get_rng_state().clone()
    selected, audit = select_peer_transfer(model, local, peer, x, y, None, controls())
    assert audit['weight'] == 0
    assert model.training
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(selected['weight'], local['weight'])
    assert torch.equal(model.weight, local['weight'])


def test_accepts_helpful_peer_and_keeps_local_buffers():
    model, _, x, y = problem()
    model.register_buffer('counter', torch.tensor(7))
    local = deepcopy(model.state_dict())
    peer = {'weight': 2 * local['weight'], 'counter': torch.tensor(99)}
    selected, audit = select_peer_transfer(model, local, peer, x, y, None, controls())
    assert audit['weight'] == 1
    assert selected['counter'].item() == 7
    assert audit['scores'][-1]['metrics']['validation']['ce'] < audit['scores'][0]['metrics']['validation']['ce']


def test_old_memory_can_veto_current_validation_gain():
    model, local, x, y = problem()
    memory = SimpleNamespace(entries={0: {'x': torch.tensor([[1.]]), 'y': torch.tensor([0])}})
    selected, audit = select_peer_transfer(model, local, {'weight': 2*local['weight']}, x, y, memory, controls())
    assert audit['weight'] == 0
    assert not audit['scores'][-1]['eligible']
    assert torch.equal(selected['weight'], local['weight'])


def test_natural_validation_counts_are_not_class_rebalanced():
    model, local, _, _ = problem()
    x, y = torch.ones(10, 1), torch.tensor([1]*9 + [0])
    _, audit = select_peer_transfer(model, local, local, x, y, None, controls())
    metric = audit['scores'][0]['metrics']['validation']
    assert metric['correct'] == 9 and metric['count'] == 10
    assert audit['weight'] == 0  # ties remain local


def test_missing_validation_does_not_silently_accept_peer():
    model, local, _, _ = problem()
    selected, audit = select_peer_transfer(model, local, local, None, None, None, controls())
    assert audit['reason'] == 'no_local_validation' and audit['weight'] == 0


@pytest.mark.parametrize('overrides', [
    {'denice_transfer_batch_size': 0}, {'denice_transfer_memory_per_class': 1.5},
    {'denice_transfer_enabled': True},
    {'denice_transfer_enabled': True, 'denice_aggregation_update_mode': 'local_delta', 'denice_age_merge_policy': 'max'},
])
def test_invalid_configuration(overrides):
    with pytest.raises(ValueError):
        transfer_config(overrides)


def test_exception_restores_local_parameters_and_mode():
    model, local, x, y = problem()
    model.eval()
    with pytest.raises(RuntimeError):
        select_peer_transfer(model, local, {'weight': torch.ones(3, 1)}, x, y, None, controls())
    assert not model.training
    assert torch.equal(model.weight, local['weight'])
