"""Behavioral invariants from CANDLE equations 12, 15, 19, 20 and 21."""
from collections import OrderedDict
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fed_learning.models.denice_model import DeNICEModel
from fed_learning.strategies.decentralized.denice_aggregation import age_aware_aggregate
from fed_learning.strategies.incremental.nice import increase_unit_ranks, update_freeze_masks


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(31)
    yield
    torch.set_num_threads(previous)


def test_eq12_fixed_integer_budget_does_not_depend_on_global_label_number():
    for classes in ([0, 1, 2], [20, 21, 22]):
        model = DeNICEModel((16, 1), 34)
        model.allocation_policy = 'fixed_per_class'
        model.capacity_per_class = {'conv1': 2, 'conv2': 3, 'conv3': 4, 'gru': 2, 'fc1': 5}
        allocated = model.allocate_task_neurons(classes)
        assert allocated == {'conv1': 6, 'conv2': 9, 'conv3': 12, 'gru': 6, 'fc1': 15}


def test_eq12_reobserved_class_does_not_unfreeze_or_allocate_new_capacity():
    from fed_learning.training.decentralized_denice_il import _prepare_client_task
    from fed_learning.strategies.incremental.denice import DeNICETrainer
    from fed_learning.strategies.incremental.denice_novelty import NoveltyEstimator
    from fed_learning.servers.nice_server import ContextDetector
    model = DeNICEModel((16, 1), 4)
    model.fixed_task_allocation = True
    model.allocation_policy = 'fixed_per_class'
    model.unit_ranks['fc2'][0] = 2
    model.pending_canc_plan = {'action': 'Expand', 'novelty': 1., 'layers': {},
                               'adapters_to_add': ['fc1'], 'reserve_to_promote': {'fc1': 5},
                               'recycle_layers': [], 'freeze_low_layers': False}
    ages = model.get_neuron_ages_state()
    detector = ContextDetector()
    result = _prepare_client_task(
        cid=0, task_id=1, num_tasks=2, new_classes=[0, 1], model=model,
        client=SimpleNamespace(X_train=torch.randn(4,16,1), y_train=torch.zeros(4,dtype=torch.long)),
        trainer=DeNICETrainer(), config={'denice_canc_mode': 'paper', 'denice_canc_schedule': 'task_end'},
        device=torch.device('cpu'), context_detector=detector, novelty_estimator=NoveltyEstimator(),
        prev_ages=None)
    for layer in ages:
        np.testing.assert_array_equal(model.unit_ranks[layer], ages[layer])
    assert detector.episode_classes[1] == [0]
    assert result['plan']['new_local_classes'] == []
    assert not model.adapters


@pytest.mark.parametrize('method', ['weighted_mean', 'coordinate_median', 'trimmed_mean'])
def test_eq19_disjoint_labels_block_all_backbone_updates(method):
    from fed_learning.strategies.decentralized.denice_aggregation import AggregationConfig
    params = OrderedDict({'fc1.weight': torch.ones(2, 2), 'fc2.weight': torch.ones(2, 2)})
    ages = {'fc1': np.ones(2, dtype=int), 'fc2': np.ones(2, dtype=int)}
    deltas = [OrderedDict((name, torch.ones_like(p)) for name, p in params.items())]
    result = age_aware_aggregate(params, ages, deltas, np.array([1.]), AggregationConfig(method=method),
                                 neighbor_ages=[ages], neighbor_labels=[[1]], target_labels=[0])
    for name in params:
        torch.testing.assert_close(result[name], params[name])


def test_eq19_self_only_preserves_local_bn_buffers_and_parameters():
    from fed_learning.training.decentralized_denice_il import _aggregate_round
    from fed_learning.strategies.decentralized.denice_capsule import build_context_capsule
    model = DeNICEModel((16, 1), 4)
    model.fixed_task_allocation = True
    for ages in model.unit_ranks.values():
        ages[:] = 1
    update_freeze_masks(model)
    model.bn1.momentum = None  # A reset counter corrupts cumulative BN statistics.
    before = deepcopy(model.state_dict())
    x, y = torch.randn(8, 16, 1), torch.arange(8) % 4
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(2):
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(model.forward_output(x), y).backward()
        model.reset_frozen_gradients()
        optimizer.step()
    after = deepcopy(model.state_dict())
    capsule = build_context_capsule(model, x, client_id=0, task_id=0, round_id=0,
                                    label_histogram={}, label_set=[0,1,2,3], sample_count=8,
                                    reliability=1., labels=y, capsule_mode='paper', fisher_samples=0)
    _aggregate_round(client_ids=[0], models={0: model}, capsules={0: capsule}, config={
        'denice_clustering_mode': 'paper', 'denice_aggregation_mode': 'self_only',
        'denice_aggregation_update_mode': 'local_delta', 'denice_age_merge_policy': 'none',
    }, device=torch.device('cpu'), before_local_states={0: before})
    for name, value in after.items():
        torch.testing.assert_close(model.state_dict()[name], value, atol=1e-7, rtol=1e-6)
        assert model.state_dict()[name].dtype == value.dtype


def test_eq20_drift_uses_shared_class_embedding_means_and_euclidean_distance():
    from fed_learning.strategies.incremental.denice_capacity import candle_prototype_drift
    before = {0: np.array([0., 0.]), 1: np.array([2., 0.]), 9: np.array([999., 999.])}
    after = {0: np.array([3., 4.]), 1: np.array([5., 4.]), 8: np.array([-999., -999.])}
    result = candle_prototype_drift(before, after)
    assert result['value'] == 5.
    assert result['shared_classes'] == [0, 1]
    assert result['defined']
    absent = candle_prototype_drift(before, {7: np.array([100., 100.])})
    assert not absent['defined'] and absent['value'] == 0.


def test_eq16_capsule_rho_is_reserve_fraction_and_does_not_update_bn():
    from fed_learning.training.decentralized_denice_il import _build_round_capsule
    model = DeNICEModel((16, 1), 4)
    for ages in model.unit_ranks.values():
        ages[:len(ages)//2] = 1
    model.train()
    before = deepcopy(dict(model.named_buffers()))
    x, y = torch.randn(4, 16, 1), torch.arange(4)
    capsule = _build_round_capsule(cid=0, task_id=0, round_id=0, model=model,
        client=SimpleNamespace(X_train=x, y_train=y, X_validation=x, y_validation=y),
        context_detector=None, ref_data=x, ref_labels=y, loss=99.,
        config={'denice_capsule_mode': 'paper', 'denice_fisher_samples': 0})
    assert capsule.reliability == .5
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, before[name])


def test_eq19_keeps_disjoint_neighbors_in_denominator_but_masks_their_deltas():
    from fed_learning.training.decentralized_denice_il import _aggregate_round
    from fed_learning.strategies.decentralized.denice_capsule import build_context_capsule
    models = {cid: DeNICEModel((16, 1), 4) for cid in (0, 1)}
    before, capsules = {}, {}
    x = torch.randn(4, 16, 1)
    for cid, model in models.items():
        for ages in model.unit_ranks.values():
            ages[:len(ages)//2] = 1
        before[cid] = deepcopy(model.state_dict())
        with torch.no_grad():
            model.fc1.weight[:128] += 2. if cid == 0 else 20.
        capsules[cid] = build_context_capsule(model, x, client_id=cid, task_id=0, round_id=0,
            label_histogram={cid: 1.}, label_set=[cid], sample_count=4,
            reliability=.5, labels=torch.full((4,), cid), capsule_mode='paper', fisher_samples=0)
    result = _aggregate_round(client_ids=[0, 1], models=models, capsules=capsules,
        config={'denice_clustering_mode': 'paper', 'denice_similarity_threshold': .5,
                'denice_aggregation_update_mode': 'local_delta', 'denice_aggregation_rho': 'reserve',
                'denice_age_merge_policy': 'none', 'denice_require_label_overlap': True},
        device=torch.device('cpu'), before_local_states=before)
    assert result['groups'][0] == [0, 1]
    np.testing.assert_allclose(result['alpha_debug'][0]['alphas'], [.5, .5])
    torch.testing.assert_close(models[0].fc1.weight[:128], before[0]['fc1.weight'][:128] + 1.)


@pytest.mark.parametrize('free,drift,defined,action,adapter', [
    (90, 0., True, 'Reuse', False), (10, 0., True, 'Expand', False),
    (90, 2., True, 'Expand', True), (90, 0., False, 'Reuse', False),
    (0, 2., True, 'Recycle', False),
])
def test_eq21_three_branches_and_domain_shift_adapter(free, drift, defined, action, adapter):
    from fed_learning.strategies.incremental.denice_capacity import candle_capacity_plan
    capacity = {'fc1': {'free': free, 'total': 100., 'rho0': free / 100,
                        'rhom': 0.1, 'learner': 0., 'mature': 100-free}}
    plan = candle_capacity_plan(capacity, {'value': drift, 'defined': defined, 'shared_classes': []},
                                {'fc1': .2}, {}, previous_consumption={'fc1': .1})
    assert plan['action'] == action
    assert bool(plan['adapters_to_add']) == adapter
    assert plan['freeze_low_layers'] is False
    assert plan['layers']['fc1']['pressure_consumption'] == .025


def test_fisher_sampling_visits_each_class_before_repeating():
    from fed_learning.strategies.decentralized.denice_capsule import fisher_sample_indices
    labels = torch.tensor([0]*50 + [1]*50 + [2]*3)
    selected = fisher_sample_indices(labels, 8)
    assert labels[selected[:3]].tolist() == [0, 1, 2]
    assert len(selected) == len(set(selected)) == 8
    assert set(labels[fisher_sample_indices(labels, 1, round_id=2)].tolist()) == {2}


def test_finalization_preserves_old_fisher_and_roundtrips_controller_state():
    from fed_learning.training.decentralized_denice_il import _finalize_candle_task
    from fed_learning.training.checkpoint_state import snapshot_denice_state, restore_denice_state
    model = DeNICEModel((16, 1), 4)
    model.unit_ranks['fc1'][:128] = 2
    model.unit_ranks['fc1'][128:] = 1
    start = model.get_neuron_ages_state()
    start['fc1'][128:] = 0
    model.candle_state = {'prototypes': {1: np.array([0., 0.])},
                          'fisher': {'fc1.bias': np.full(256, 9., dtype=np.float32)}}
    capsule = SimpleNamespace(penultimate_prototypes={1: np.array([3., 4.])},
                              parameter_fisher={'fc1.bias': np.full(256, 2., dtype=np.float32)})
    plan = _finalize_candle_task(model, capsule, start, 1, {})
    assert plan['drift']['value'] == 5.
    np.testing.assert_array_equal(model.candle_state['fisher']['fc1.bias'][:128], 9.)
    np.testing.assert_array_equal(model.candle_state['fisher']['fc1.bias'][128:], 2.)
    restored = DeNICEModel((16,1), 4)
    restore_denice_state(restored, None, snapshot_denice_state(model))
    np.testing.assert_array_equal(restored.candle_state['fisher']['fc1.bias'],
                                  model.candle_state['fisher']['fc1.bias'])
    assert restored.pending_canc_plan['controller'] == 'candle_eq21'


def test_recycling_uses_fisher_and_keeps_router_anchor():
    from fed_learning.strategies.incremental.denice_recycling import apply_candle_recycling
    model = DeNICEModel((16, 1), 4)
    for ranks in model.unit_ranks.values():
        ranks[:] = 2
    before = deepcopy(model.state_dict())
    model.candle_state = {'fisher': {name: np.ones(tuple(p.shape), dtype=np.float32)
                                     for name, p in model.named_parameters()}}
    model.candle_state['fisher']['conv1.weight'][1] = 0.
    mask = np.zeros(64+128+256+100, dtype=bool)
    mask[0] = True
    summary = apply_candle_recycling(model, {'recycle_layers': ['conv1']}, mask, percentile=1.)
    assert summary['recycled']['conv1'] == [1]
    assert model.unit_ranks['conv1'][0] == 2 and model.unit_ranks['conv1'][1] == 0
    torch.testing.assert_close(model.conv1.weight[0], before['conv1.weight'][0])
    assert not torch.equal(model.conv1.weight[1], before['conv1.weight'][1])
    assert not model.candle_state['fisher']['conv1.weight'][1].any()


@pytest.mark.parametrize('force_adapter_branch', [False, True])
def test_three_task_paper_run_matches_split_continuation(tmp_path, monkeypatch, force_adapter_branch):
    import json
    from pathlib import Path
    from fed_learning.training.decentralized_denice_il import run_decentralized_denice_il
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    if force_adapter_branch:
        # Exercise adapter training/aggregation/checkpoint wiring independently
        # of drift estimation (these synthetic tasks have disjoint labels).
        monkeypatch.setattr('fed_learning.strategies.incremental.denice_capacity.candle_prototype_drift',
                            lambda previous, current: {'value': 1., 'defined': True, 'shared_classes': []})
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'metadata.json').write_text(json.dumps({
        'task_structure': {'total_classes': 6, 'task_classes': {'0': [0,1], '1': [2,3], '2': [4,5]}},
        'client_allocation': {'task_active_clients': {'0': [0], '1': [0], '2': [0]}},
    }))
    rng = np.random.default_rng(31)
    y = np.repeat(np.arange(6), 6)
    x = rng.normal(size=(36,16,1)).astype('float32')
    np.savez(data / 'client_0_train.npz', X_train=x, y_train=y)
    np.savez(data / 'global_test_data.npz', X_test=x + .1, y_test=y)
    config = dict(algorithm='denice', mode='decentralized', data_dir=str(data),
                  total_classes=6, num_clients=1, denice_max_clients=1,
                  task_end=2, rounds_per_task=1, batch_size=8, nice_phase_epochs=1,
                  learning_rate=.001, seed=31, denice_post_task_eval=False,
                  round_checkpoint_every=1, denice_checkpoint_format='full',
                  save_continuation_every_task=True, denice_structural_protection=True,
                  denice_fixed_task_allocation=True, denice_allocation_policy='fixed_per_class',
                  denice_memory_policy='sketches', denice_router_mode='binary_cosine',
                  denice_router_update_schedule='every_round',
                  denice_refresh_router_memory_after_aggregation=False,
                  denice_shared_context_eval=False, denice_capsule_mode='paper',
                  denice_clustering_mode='paper', denice_validation_fraction=.25,
                  denice_aggregation_update_mode='local_delta', denice_aggregation_rho='reserve',
                  denice_fisher_samples=2, denice_canc_schedule='task_end', denice_canc_mode='paper',
                  denice_canc_theta1=.1, denice_collaboration_guard_mode='off')
    config.update(denice_adapter_mode='linear_input', denice_adapter_layers=['fc1', 'gru', 'conv3'])
    full = run_decentralized_denice_il({**config, 'output_dir': str(tmp_path/'full')})
    split = run_decentralized_denice_il({**config, 'task_end': 0, 'output_dir': str(tmp_path/'split')})
    first = Path(split['output_dir'])/'continuation_state_task_0.pt'
    resumed = run_decentralized_denice_il({**config, 'task_start': 1, 'resume_state_path': str(first),
                                          'output_dir': str(tmp_path/'resumed'),
                                          'resume_output_dir': str(tmp_path/'resumed')})
    checkpoints = [torch.load(Path(run['output_dir'])/'continuation_state_task_2.pt', weights_only=False)
                   for run in [full, resumed]]
    for name, value in checkpoints[0]['client_model_states'][0].items():
        torch.testing.assert_close(checkpoints[1]['client_model_states'][0][name], value, rtol=0, atol=0)
    for checkpoint in checkpoints:
        state = checkpoint['client_algorithm_states'][0]
        state = state.get('denice', state)
        assert state['candle_state']['task_id'] == 2
        assert state['pending_canc_plan']['controller'] == 'candle_eq21'
        assert not state['context_detector']['reference_input_memory']
        if force_adapter_branch:
            assert len(state['adapter_registry']) == 6
            assert all(meta['mode'] == 'linear_input' for meta in state['adapter_registry'].values())
            weights = checkpoint['client_model_states'][0]
            assert any(value.abs().sum() > 0 for name, value in weights.items()
                       if name.startswith('adapters.') and name.endswith('U.weight'))
