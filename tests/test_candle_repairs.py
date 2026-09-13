"""Behavioral checks for the PDF audit repairs; no external dataset required."""
import ast
import json
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from fed_learning.models.denice_model import DeNICEModel
from fed_learning.servers.nice_server import ContextDetector
from fed_learning.strategies.incremental.nice import update_freeze_masks, drop_young_to_learner
from fed_learning.strategies.decentralized.denice_aggregation import age_aware_aggregate, AggregationConfig
from fed_learning.training.checkpoint_state import snapshot_denice_state, restore_denice_state


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(19)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('method', ['weighted_mean', 'coordinate_median', 'trimmed_mean'])
def test_both_young_and_class_support_required(method):
    target = OrderedDict({'fc2.weight': torch.ones(4, 1)})
    delta = OrderedDict({'fc2.weight': torch.full((4, 1), 5.)})
    result = age_aware_aggregate(
        target, {'fc2': np.array([0, 1, 1, 1])}, [delta], np.array([1.]),
        AggregationConfig(method=method),
        neighbor_ages=[{'fc2': np.array([1, 0, 1, 1])}],
        neighbor_labels=[[0, 1, 2]], target_labels=[0, 1, 2, 3],
    )
    torch.testing.assert_close(result['fc2.weight'].flatten(), torch.tensor([1., 1., 6., 1.]))


def test_gru_mature_features_and_masks_survive_updates_and_restore():
    model = DeNICEModel((16, 1), 4).eval()
    model.structural_protection = True
    model.fixed_task_allocation = True
    model.unit_ranks['gru'][:50] = 2
    model.unit_ranks['gru'][50:] = 1
    model.protect_task_connections()
    x = torch.randn(5, 16, 1)
    before = model._run_gru(x)[0][:, :, :50].detach().clone()
    with torch.no_grad():
        for parameter in model.gru.parameters():
            plastic = torch.tensor(([False]*50 + [True]*50)*3)
            parameter[plastic] += 0.2
    after = model._run_gru(x)[0][:, :, :50]
    torch.testing.assert_close(after, before, atol=1e-7, rtol=0)
    restored = DeNICEModel((16, 1), 4).eval()
    restore_denice_state(restored, None, snapshot_denice_state(model))
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(x), model(x), atol=0, rtol=0)


def test_task_freeze_override_survives_age_refresh():
    model = DeNICEModel((16, 1), 4)
    model.unit_ranks['conv1'][:] = 1
    model.task_freeze_layers = ['conv1']
    update_freeze_masks(model)
    assert model.freeze_masks['conv1'].all()
    update_freeze_masks(model)
    assert model.freeze_masks['conv1'].all()


def test_binary_router_sparse_episode_ids_and_no_raw_references():
    detector = ContextDetector(router_mode='binary_cosine')
    detector.retain_reference_inputs = False
    detector.episode_classes = {0: [0], 3: [1]}
    detector.activation_memory = {0: np.array([[1., 0.]]), 3: np.array([[0., 1.]])}
    detector.train_models(3)
    pred, scores = detector.predict_episodes_with_scores(np.eye(2))
    assert pred.tolist() == [0, 3]
    assert scores.shape == (2, 4)
    assert detector.reference_input_memory == {}


def test_delta_metadata_keeps_fractional_statistics():
    from fed_learning.training.denice_delta_checkpoint import _compact_metadata
    values = np.array([0.01, 0.3], dtype=np.float32)
    np.testing.assert_array_equal(_compact_metadata(values), values)


def test_notebook_uses_shared_helper_and_all_client_confusion():
    notebook = json.loads(Path('eval_selective.ipynb').read_text(encoding='utf-8'))
    source = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
    ast.parse(source)
    assert 'best_idx = max(' not in source
    assert 'summed_confusion +=' in source
    assert 'require_compatible_calibration=bool(config.get(' in source


@pytest.mark.parametrize('controller', ['legacy', 'paper'])
def test_two_task_candle_continuation_without_raw_bank(tmp_path, controller):
    from fed_learning.training.decentralized_denice_il import run_decentralized_denice_il
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'metadata.json').write_text(json.dumps({
        'task_structure': {'total_classes': 4, 'task_classes': {'0':[0,1], '1':[2,3]}},
        'client_allocation': {'task_active_clients': {'0':[0,1], '1':[0,1]}},
    }))
    rng = np.random.default_rng(19)
    y = np.repeat(np.arange(4), 4)
    for cid in [0, 1]:
        np.savez(data / f'client_{cid}_train.npz', X_train=rng.normal(size=(16,16,1)).astype('float32'), y_train=y)
    np.savez(data / 'global_test_data.npz', X_test=rng.normal(size=(16,16,1)).astype('float32'), y_test=y)
    config = dict(algorithm='denice', mode='decentralized', data_dir=str(data),
                  output_dir=str(tmp_path/'run'), total_classes=4, num_clients=2,
                  task_end=1, rounds_per_task=2, batch_size=4, nice_phase_epochs=1,
                  learning_rate=0.001, seed=19, denice_max_clients=2,
                  denice_post_task_eval=False, round_checkpoint_every=2,
                  denice_checkpoint_format='full', save_continuation_every_task=True,
                  denice_structural_protection=True, denice_fixed_task_allocation=True,
                  denice_memory_policy='sketches', denice_router_mode='binary_cosine',
                  denice_router_update_schedule='every_round',
                  denice_refresh_router_memory_after_aggregation=False,
                  denice_shared_context_eval=False, denice_capsule_mode='paper',
                  denice_clustering_mode='paper', denice_validation_fraction=0.25,
                  denice_aggregation_update_mode='local_delta', denice_aggregation_rho='reserve',
                  denice_fisher_samples=2, denice_canc_schedule='task_end',
                  denice_collaboration_guard_mode='warn')
    if controller == 'paper':
        config.update(denice_canc_mode='paper', denice_allocation_policy='fixed_per_class')
    result = run_decentralized_denice_il(config)
    checkpoint = torch.load(Path(result['output_dir'])/'continuation_state_task_1.pt', weights_only=False)
    assert all(not bank for bank in checkpoint['old_ref_banks'].values())
    for state in checkpoint['client_algorithm_states'].values():
        state = state.get('denice', state)
        assert state['structural_protection']
        assert state['pending_canc_plan']['measured_at_task'] == 1
        if controller == 'paper':
            assert state['pending_canc_plan']['controller'] == 'candle_eq21'
            assert state['candle_state']['task_id'] == 1
            assert state['candle_state']['fisher']
            assert not state['pending_canc_plan']['drift']['defined']
        detector = state['context_detector']
        assert detector['reference_input_memory'] == {}
        assert detector['router_state_fresh']
        assert detector['stable_feature_mask'] is not None
    assert checkpoint['config']['source_sha256']
    from eval_checkpoint import _make_denice_client_model
    first = torch.load(Path(result['output_dir'])/'checkpoint_task_0_round_1.pt', weights_only=False)
    final = torch.load(Path(result['output_dir'])/'checkpoint_task_1_round_1.pt', weights_only=False)
    old_model, old_router = _make_denice_client_model(first, 0, 'cpu')
    new_model, new_router = _make_denice_client_model(final, 0, 'cpu')
    probe = torch.randn(8, 16, 1)
    before = old_router._binarize_per_sample(old_model, probe)
    after = new_router._binarize_per_sample(new_model, probe)
    np.testing.assert_array_equal(after, before)
    # All-client confusion describes the same predictions as reported accuracy.
    from fed_learning.training.denice_eval import evaluate_denice_model
    metrics = evaluate_denice_model(new_model, {'X_test': probe, 'y_test': torch.arange(8)%4},
                                   'cpu', new_router, [0,1,2,3], include_confusion_matrix=True)
    matrix = np.asarray(metrics['confusion_matrix'])
    assert matrix.sum() == 8
    assert matrix.trace()/matrix.sum() == metrics['accuracy']
