from collections import OrderedDict
from copy import deepcopy
import numpy as np
import pytest
import torch
from torch import nn
from fed_learning.strategies.incremental.denice_der import DERReplay
from fed_learning.strategies.incremental.denice_replay import ReplayConfig
from fed_learning.strategies.incremental.denice_ewc import consolidate_ewc, ewc_loss_factory
from fed_learning.strategies.incremental.denice_variants import variant_config, variant_preset, preserve_local_mature


@pytest.fixture(autouse=True)
def threads():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def controls(method='derpp', **overrides):
    return variant_config({**variant_preset(method), **overrides})


def test_reservoir_is_stream_uniform_and_targets_never_rewritten():
    counts = torch.zeros(20)
    for seed in range(200):
        torch.manual_seed(seed)
        memory = DERReplay(ReplayConfig(capacity=5), controls())
        x = torch.arange(20.).reshape(-1, 1)
        memory.observe(x, torch.zeros(20).long(), x.repeat(1, 2), torch.ones(2).bool())
        assert len(memory) == 5 and memory.seen == 20
        for row in memory.rows:
            counts[int(row['x'][0])] += 1
        saved = deepcopy(memory.state_dict())
        memory.commit(None, None, None, 0)
        for a, b in zip(memory.rows, saved['rows']):
            torch.testing.assert_close(a['logits'], b['logits'])
    # Deterministic seeds, broad check detecting recent-only or prefix retention.
    assert (counts > 25).all() and (counts < 80).all()


def test_derpp_independent_draws_and_paper_squared_norm(monkeypatch):
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    memory = DERReplay(ReplayConfig(capacity=4), controls())
    memory.observe(torch.eye(2), torch.tensor([0, 1]), torch.zeros(2, 2), torch.ones(2).bool())
    calls = []
    def sample(device):
        calls.append(1)
        i = len(calls) - 1
        row = memory.rows[i]
        return {k: v.unsqueeze(0) for k, v in row.items()}
    monkeypatch.setattr(memory, 'sample', sample)
    loss, audit = memory.loss(model, torch.eye(2), torch.tensor([0, 1]), [0, 1])
    expected = .5 + .5 * torch.nn.functional.cross_entropy(torch.tensor([[0., 1.]]), torch.tensor([1]))
    torch.testing.assert_close(loss, expected)
    assert len(calls) == 2 and audit['calibration_ce'] == 0
    memory.controls['method'] = 'der'
    calls.clear()
    loss, _ = memory.loss(model, torch.eye(2), torch.tensor([0, 1]), [0, 1])
    assert len(calls) == 1 and loss.item() == .5


def test_reservoir_continuation_keeps_next_sampling_and_insertion():
    torch.manual_seed(19)
    config = ReplayConfig(capacity=5, batch_size=3)
    memory = DERReplay(config, controls())
    x = torch.arange(12.).reshape(6, 2)
    memory.observe(x, torch.arange(6) % 2, x, torch.ones(2).bool())
    restored = DERReplay.from_state(config, controls(), memory.state_dict())
    rng = torch.get_rng_state()
    a = memory.sample('cpu')
    memory.observe(x, torch.arange(6) % 2, x + 1, torch.ones(2).bool())
    torch.set_rng_state(rng)
    b = restored.sample('cpu')
    restored.observe(x, torch.arange(6) % 2, x + 1, torch.ones(2).bool())
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
    for first, second in zip(memory.rows, restored.rows):
        for key in first:
            torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)


def test_seen_logit_mean_variant_ignores_unseen_coordinates():
    model = nn.Linear(1, 3)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.copy_(torch.tensor([2., 1., 100.]))
    c = controls('der', denice_cl_logit_scope='seen', denice_der_reduction='mean')
    memory = DERReplay(ReplayConfig(capacity=2), c)
    memory.observe(torch.ones(1, 1), torch.zeros(1).long(), torch.zeros(1, 3), torch.tensor([True, True, False]))
    loss, _ = memory.loss(model, torch.ones(1, 1), torch.zeros(1).long(), [0, 1])
    assert loss.item() == 1.25


def test_ewc_fisher_is_mean_of_individual_squared_gradients_and_restores_modes():
    model = nn.Linear(2, 2, bias=False).train()
    x = torch.tensor([[1., 2.], [-2., 1.]])
    y = torch.tensor([0, 1])
    expected = torch.zeros_like(model.weight)
    for i in range(2):
        gradient, = torch.autograd.grad(model(x[i:i+1]).log_softmax(1)[0, y[i]], [model.weight])
        expected += gradient.square() / 2
    rng = torch.get_rng_state().clone()
    consolidate_ewc(model, x, y, 0, controls('ewc', denice_ewc_fisher_samples=2, denice_ewc_fisher_labels='empirical'))
    torch.testing.assert_close(model.ewc_state['banks'][0]['fisher']['weight'], expected)
    assert model.training and model.weight.grad is None and torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize('mode,banks', [('separate', 2), ('online', 1)])
def test_ewc_task_banks_formula_and_nonzero_restoring_gradient(mode, banks):
    model = nn.Linear(1, 2, bias=False)
    c = controls('ewc', denice_ewc_mode=mode, denice_ewc_lambda=4., denice_ewc_fisher_samples=2)
    for task in range(2):
        consolidate_ewc(model, torch.ones(2, 1), torch.zeros(2).long(), task, c)
    assert len(model.ewc_state['banks']) == banks
    penalty = ewc_loss_factory(model, c)
    assert penalty().item() == 0
    with torch.no_grad():
        model.weight.add_(.1)
    expected = 2 * sum((bank['fisher']['weight'] * (model.weight - bank['anchor']['weight']).square()).sum()
                       for bank in model.ewc_state['banks'])
    torch.testing.assert_close(penalty(), expected)
    penalty().backward()
    assert (model.weight.grad > 0).all()


def test_local_mature_update_survives_without_taking_peer_value():
    local = OrderedDict({'fc2.weight': torch.tensor([[3.], [4.]])})
    aggregated = OrderedDict({'fc2.weight': torch.tensor([[1.], [8.]])})
    result = preserve_local_mature(aggregated, local, {'fc2': np.array([2, 1])})
    torch.testing.assert_close(result['fc2.weight'], torch.tensor([[3.], [8.]]))


def test_validation_fold_reconstructs_same_ids_without_train_overlap():
    from fed_learning.training.decentralized_denice_il import _split_local_validation
    x = torch.arange(40).reshape(20, 2)
    y = torch.arange(20) % 2
    train_x, _, val_x, _ = _split_local_validation(x, y, .2, 17)
    assert len(val_x) == 4 and len(train_x) == 16
    assert not set(train_x[:, 0].tolist()).intersection(val_x[:, 0].tolist())
    assert torch.equal(val_x, _split_local_validation(x, y, .2, 17)[2])


@pytest.mark.parametrize('override', [dict(denice_replay_capacity=12), dict(denice_plasticity_enabled=True),
                                      dict(denice_ewc_lambda=-1), dict(denice_ewc_fisher_samples=0)])
def test_reject_mixed_or_invalid_ewc_configuration(override):
    with pytest.raises(ValueError):
        controls('ewc', **override)


@pytest.mark.parametrize('method', ['derpp', 'ewc'])
@pytest.mark.parametrize('device,amp', [('cpu', False), ('cuda', True)])
def test_actual_client_updates_mature_rows_and_captures_pre_step_logits(method, device, amp):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from fed_learning.models.denice_model import DeNICEModel
    from fed_learning.clients.denice_client import DeNICEClient
    from fed_learning.strategies.incremental.denice import DeNICETrainer
    from fed_learning.strategies.incremental.nice import update_freeze_masks
    torch.manual_seed(17)
    model = DeNICEModel((16, 1), 4).to(device)
    model.fixed_task_allocation = True
    for ranks in model.unit_ranks.values():
        ranks[:] = 1
        ranks[0] = 2
    update_freeze_masks(model)
    x, y = torch.randn(24, 16, 1), torch.ones(24).long()
    c = controls(method)
    memory = DERReplay(ReplayConfig(capacity=32, batch_size=4), c) if method == 'derpp' else None
    if method == 'ewc':
        consolidate_ewc(model, x, y, 0, c)
    before = model.fc2.weight[0].detach().clone()
    client = DeNICEClient(0, x, y, max_phases=1, phase_epochs=1)
    client.setup_for_gpu(model, device)
    client.use_amp = amp
    captured = []
    hook = model.register_forward_hook(lambda m, args, output: captured.append(output.detach().float().cpu().clone()))
    try:
        result = client.train(DeNICETrainer(max_phases=1, phase_epochs=1), 1, 4, .001,
                              local_replay=memory, continual_controls=c)
    finally:
        hook.remove()
    assert torch.isfinite(torch.tensor(result['optimization_loss']))
    assert not torch.equal(before, model.fc2.weight[0])
    assert model.freeze_masks['fc2'][0]  # Local relaxation did not erase peer mask.
    if memory is not None:
        assert memory.seen == 4 * result['optimizer_steps']
        targets = torch.stack([row['logits'] for row in memory.rows[:4]])
        assert any(torch.equal(targets, value) for value in captured)
        if not amp:
            torch.testing.assert_close(targets, captured[0], atol=0, rtol=0)
        assert result['replay']['steps'] == 6
    else:
        assert result['continual']['regularization_loss'] > 0
