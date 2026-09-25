"""Behavioral tests for bounded local memory, protected learning and resume."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from fed_learning.strategies.incremental.denice_replay import LocalReplay, ReplayConfig, inference_statistics
from fed_learning.models.denice_model import DeNICEModel
from fed_learning.clients.denice_client import DeNICEClient
from fed_learning.strategies.incremental.denice import DeNICETrainer
from fed_learning.strategies.incremental.nice import update_freeze_masks


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(23)
    yield
    torch.set_num_threads(before)


def toy():
    model = nn.Sequential(nn.BatchNorm1d(3), nn.Linear(3, 4))
    model.num_classes = 4
    return model


@pytest.mark.parametrize('rnn_class', [nn.RNN, nn.GRU, nn.LSTM])
def test_replay_recurrent_modes_and_dropout_restored_on_error(rnn_class):
    model = nn.Sequential(rnn_class(3, 4, num_layers=2, dropout=.4), nn.Dropout(.5))
    model.train()
    model[0].eval()
    with pytest.raises(RuntimeError, match='injected'):
        with inference_statistics(model):
            assert model[0].training
            assert model[0].dropout == 0
            assert not model[1].training
            raise RuntimeError('injected')
    assert model.training and not model[0].training and model[1].training
    assert model[0].dropout == .4
    with torch.no_grad(), inference_statistics(model):
        assert not model[0].training
        assert model[0].dropout == .4


@pytest.mark.parametrize('config', [
    {'denice_replay_capacity': -1}, {'denice_replay_capacity': 2.1},
    {'denice_replay_batch_size': 0}, {'denice_replay_logit_weight': float('nan')},
    {'denice_replay_ce_weight': -1},
])
def test_invalid_configuration(config):
    with pytest.raises(ValueError):
        ReplayConfig.from_dict(config)


def test_memory_is_balanced_bounded_idempotent_and_not_model_state():
    model = toy()
    memory = LocalReplay(ReplayConfig(capacity=8))
    x = torch.randn(100, 3)
    y = torch.tensor([0] * 95 + [1] * 5)
    memory.commit(model, x, y, 0)
    assert memory.stats()['class_counts'] == {0: 4, 1: 4}
    state = memory.state_dict()
    memory.commit(model, x, y, 0)
    for c in memory.entries:
        torch.testing.assert_close(state['entries'][c]['x'], memory.entries[c]['x'])
    memory.commit(model, torch.randn(20, 3), torch.tensor([2]*10+[3]*10), 1)
    assert memory.stats()['class_counts'] == {0: 2, 1: 2, 2: 2, 3: 2}
    assert all(not e['logits'].requires_grad for e in memory.entries.values())
    assert not any('replay' in name for name in model.state_dict())
    assert not memory.entries[0]['valid'][:, 2:].any()
    assert memory.entries[2]['valid'].all()


def test_loss_preserves_batchnorm_and_modes_and_masks_unseen_logits():
    model = toy()
    memory = LocalReplay(ReplayConfig(capacity=4, ce_weight=0, calibration_weight=0))
    x, y = torch.randn(8, 3), torch.tensor([0, 1]*4)
    model[0].eval()  # mixed train/eval must be restored exactly
    memory.commit(model, x, y, 0)
    bn = deepcopy(model[0].state_dict())
    with torch.no_grad():
        model[1].bias[3] += 100
    loss, audit = memory.loss(model, x, y, [2])
    assert audit['dark_mse'] == pytest.approx(0., abs=1e-12)
    loss.backward()
    assert model[1].bias.grad[3] == 0
    assert model.training and not model[0].training and model[1].training
    for k, value in bn.items():
        torch.testing.assert_close(model[0].state_dict()[k], value, atol=0, rtol=0)


def test_replay_reduces_new_class_intrusion_on_old_examples():
    model = nn.Linear(3, 4)
    model.num_classes = 4
    memory = LocalReplay(ReplayConfig(capacity=16, logit_weight=0, calibration_weight=0))
    x = torch.ones(16, 3)
    y = torch.zeros(16, dtype=torch.long)
    memory.commit(model, x, y, 0)
    with torch.no_grad():
        model.bias[1] = 4
    before = torch.nn.functional.cross_entropy(model(x)[:, :2], y).item()
    opt = torch.optim.SGD(model.parameters(), lr=.1)
    for _ in range(10):
        opt.zero_grad()
        loss, _ = memory.loss(model, x, torch.ones_like(y), [1])
        loss.backward()
        opt.step()
    assert torch.nn.functional.cross_entropy(model(x)[:, :2], y).item() < before * .5


@pytest.mark.parametrize('device,amp', [('cpu', False), ('cuda', False), ('cuda', True)])
def test_real_training_preserves_mature_parameters_and_has_replay_gradients(device, amp):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required for cuDNN replay backward regression')
    model = DeNICEModel((16, 1), 4).to(device)
    model.structural_protection = True
    model.fixed_task_allocation = True
    for layer in model.LAYER_NAMES:
        model.unit_ranks[layer][:] = 1
        model.unit_ranks[layer][0] = 2
    model.protect_task_connections()
    update_freeze_masks(model)
    x = torch.randn(8, 16, 1)
    memory = LocalReplay(ReplayConfig(capacity=4))
    memory.commit(model, x, torch.zeros(8, dtype=torch.long), 0)
    before = model.fc2.weight[0].detach().clone()
    client = DeNICEClient(0, x, torch.ones(8, dtype=torch.long), max_phases=1, phase_epochs=1)
    client.setup_for_gpu(model, device)
    client.use_amp = amp
    result = client.train(DeNICETrainer(max_phases=1, phase_epochs=1),
                          1, 4, .001, local_replay=memory)
    torch.testing.assert_close(model.fc2.weight[0], before, atol=0, rtol=0)
    assert result['replay']['losses']['replay_ce'] > 0
    assert result['replay']['losses']['calibration_ce'] > 0
    assert result['optimization_loss'] > result['loss']
    assert torch.isfinite(torch.tensor(result['loss']))


def test_state_restores_sampling_exactly_and_rejects_config_change():
    memory = LocalReplay(ReplayConfig(capacity=4))
    memory.commit(toy(), torch.randn(8, 3), torch.tensor([0, 1]*4), 0)
    restored = LocalReplay.from_state(memory.config, memory.state_dict())
    rng = torch.get_rng_state()
    first = memory.sample('cpu')
    torch.set_rng_state(rng)
    second = restored.sample('cpu')
    for k in first:
        torch.testing.assert_close(first[k], second[k], atol=0, rtol=0)
    with pytest.raises(ValueError):
        LocalReplay.from_state(ReplayConfig(capacity=8), memory.state_dict())


def test_two_client_three_task_resume_and_private_memory(tmp_path, monkeypatch):
    from fed_learning.training.decentralized_denice_il import run_decentralized_denice_il
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr('fed_learning.strategies.incremental.denice_capacity.candle_prototype_drift',
                        lambda previous, current: {'value': 1., 'defined': True, 'shared_classes': []})
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'metadata.json').write_text(json.dumps({
        'task_structure': {'total_classes': 6, 'task_classes': {'0': [0,1], '1': [2,3], '2': [4,5]}},
        'client_allocation': {'task_active_clients': {'0': [0,1], '1': [0], '2': [0,1]}},
    }))
    rng = np.random.default_rng(31)
    y = np.repeat(np.arange(6), 6)
    for cid in range(2):
        x = rng.normal(size=(36,16,1)).astype('float32') + cid * 10
        # Runner participation is data-driven; metadata alone does not remove
        # a client's samples. Client 1 has no task-1 data, then rejoins task 2.
        keep = np.ones(len(y), dtype=bool) if cid == 0 else ~np.isin(y, [2,3])
        np.savez(data / f'client_{cid}_train.npz', X_train=x[keep], y_train=y[keep])
    np.savez(data / 'global_test_data.npz', X_test=x + 100, y_test=y)
    config = dict(algorithm='denice', mode='decentralized', data_dir=str(data),
                  total_classes=6, num_clients=2, denice_max_clients=2,
                  task_end=2, rounds_per_task=1, batch_size=8, nice_phase_epochs=1,
                  learning_rate=.001, seed=31, denice_post_task_eval=False,
                  round_checkpoint_every=1, denice_checkpoint_format='full',
                  save_continuation_every_task=True, denice_structural_protection=True,
                  denice_fixed_task_allocation=True, denice_allocation_policy='fixed_per_class',
                  denice_memory_policy='local_replay', denice_router_mode='binary_cosine',
                  denice_replay_capacity=12, denice_replay_batch_size=4,
                  denice_router_update_schedule='every_round',
                  denice_refresh_router_memory_after_aggregation=False,
                  denice_shared_context_eval=False, denice_capsule_mode='paper',
                  denice_clustering_mode='paper', denice_validation_fraction=.25,
                  denice_aggregation_update_mode='local_delta', denice_aggregation_rho='reserve',
                  denice_fisher_samples=2, denice_canc_schedule='task_end', denice_canc_mode='paper',
                  denice_canc_theta1=.1, denice_collaboration_guard_mode='off',
                  denice_adapter_mode='linear_input')
    full = run_decentralized_denice_il({**config, 'output_dir': str(tmp_path/'full')})
    split = run_decentralized_denice_il({**config, 'task_end': 0, 'output_dir': str(tmp_path/'split')})
    first = Path(split['output_dir'])/'continuation_state_task_0.pt'
    resumed = run_decentralized_denice_il({**config, 'resume_state_path': str(first),
                                          'output_dir': str(tmp_path/'resumed'),
                                          'resume_output_dir': str(tmp_path/'resumed')})
    states = [torch.load(Path(run['output_dir'])/'continuation_state_task_2.pt', weights_only=False)
              for run in [full, resumed]]
    for cid in range(2):
        for name, value in states[0]['client_model_states'][cid].items():
            torch.testing.assert_close(states[1]['client_model_states'][cid][name], value, atol=0, rtol=0)
        memory = states[0]['local_replay_states'][cid]
        other = states[1]['local_replay_states'][cid]
        assert sum(len(e['y']) for e in memory['entries'].values()) <= 12
        assert memory['completed_tasks'] == ([0,1,2] if cid == 0 else [0,2])
        for c, entry in memory['entries'].items():
            for k in entry:
                torch.testing.assert_close(other['entries'][c][k], entry[k], atol=0, rtol=0)
            assert abs(float(entry['x'].mean()) - cid * 10) < 2  # no peer/test samples
        assert not states[0]['old_ref_banks'][cid]
        algorithm = states[0]['client_algorithm_states'][cid]
        algorithm = algorithm.get('denice', algorithm)
        assert not algorithm['context_detector']['reference_input_memory']
        assert 'local_replay_states' not in algorithm
