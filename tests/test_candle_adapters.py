"""CANDLE Eq. (23) and adapter continuation/structural invariants."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from fed_learning.models.denice_model import DeNICEModel, LinearInputAdapter
from fed_learning.training.checkpoint_state import snapshot_denice_state, restore_denice_state


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(73)
    yield
    torch.set_num_threads(previous)


def model_with_adapter(layer, mode='linear_input'):
    model = DeNICEModel((17, 2), 4).eval()
    model.configure_adapter_mode(mode)
    model.structural_protection = True
    model.fixed_task_allocation = True
    for ages in model.unit_ranks.values():
        ages[:len(ages)//2] = 2
        ages[len(ages)//2:3*len(ages)//4] = 1
    key = model.add_adapter(1, layer)
    model.protect_active_adapter_inputs()
    return model, key


def test_eq23_adapter_is_linear_and_uses_distinct_input_output_dimensions():
    adapter = LinearInputAdapter(11, 7, 3)
    with torch.no_grad():
        adapter.U.weight.normal_()
    x, y = torch.randn(4, 11), torch.randn(4, 11)
    torch.testing.assert_close(adapter(2*x - 3*y), 2*adapter(x) - 3*adapter(y))
    torch.testing.assert_close(adapter(torch.zeros_like(x)), torch.zeros(4, 7))


@pytest.mark.parametrize('layer', ['conv1', 'conv2', 'conv3', 'gru', 'fc1'])
def test_adapter_noop_then_gradients_and_checkpoint_roundtrip(layer):
    model, key = model_with_adapter(layer)
    x = torch.randn(4, 17, 2)
    with_adapter = model(x)
    model.clear_active_adapters()
    torch.testing.assert_close(model(x), with_adapter, rtol=0, atol=0)
    model.set_active_context(1)
    loss = torch.nn.functional.cross_entropy(model.forward_output(x), torch.full((4,), 2))
    loss.backward()
    assert model.adapters[key].U.weight.grad.abs().sum() > 0
    with torch.no_grad():
        model.adapters[key].U.weight.normal_(std=.1)
    assert not torch.equal(model(x), with_adapter)
    restored = DeNICEModel((17, 2), 4).eval()
    restore_denice_state(restored, None, snapshot_denice_state(model))
    restored.load_state_dict(model.state_dict(), strict=True)
    assert restored.adapter_mode == 'linear_input'
    assert restored.architecture_version == 2
    assert restored.get_adapter_registry_state() == model.get_adapter_registry_state()
    torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)
    torch.testing.assert_close(restored.adapter_output_masks[key], model.adapter_output_masks[key])


def test_fc1_residual_uses_input_even_when_layer_output_is_zero_and_preserves_mature_rows():
    model, key = model_with_adapter('fc1')
    with torch.no_grad():
        model.fc1.weight.zero_()
        model.fc1.bias.zero_()
        model.adapters[key].V.weight.fill_(.1)
        model.adapters[key].U.weight.fill_(.2)
    features = torch.ones(3, model.fc1.in_features)
    output = model._apply_fc1_adapter(torch.zeros(3, 256), features)
    assert output[:, 128:192].min() > 0
    assert not output[:, :128].any() and not output[:, 192:].any()
    # Consolidation must not turn off a trained adapter's saved output support.
    model.unit_ranks['fc1'][128:192] = 2
    torch.testing.assert_close(model._apply_fc1_adapter(torch.zeros(3, 256), features), output)


def test_conv_input_projection_matches_stride_two_order_and_blocks_reserve_sources():
    model, key = model_with_adapter('conv3')
    with torch.no_grad():
        model.adapters[key].V.weight.fill_(1.)
        model.adapters[key].U.weight.fill_(1.)
    raw = torch.zeros(2, 128, 5)
    raw[:, 0, :2] = torch.tensor([1., 2.])
    raw[:, 127] = 1000.  # reserve input is structurally masked
    result = model._apply_conv_channel_adapter(torch.zeros(2, 256, 2), 'conv3', raw)
    assert torch.all(result[:, 128:192, 0] == 3 * model.adapters[key].rank)
    assert not result[:, :, 1].any()


def test_penultimate_capsule_uses_same_embedding_as_classifier():
    from fed_learning.strategies.decentralized.denice_capsule import class_penultimate_prototypes
    model, key = model_with_adapter('fc1')
    with torch.no_grad():
        model.adapters[key].U.weight.normal_(std=.1)
    x, y = torch.randn(6, 17, 2), torch.tensor([0, 1, 0, 1, 0, 1])
    prototypes = class_penultimate_prototypes(model, x, y)
    expected = model.penultimate_features(x).detach().numpy()
    for cls in (0, 1):
        np.testing.assert_allclose(prototypes[cls], expected[y.numpy() == cls].mean(0), atol=1e-6)


def test_recycling_cuts_old_input_support_with_correct_source_layer():
    model, key = model_with_adapter('fc1')
    assert model.adapter_input_masks[key][2:4].all()  # conv3 neuron 1, two positions
    assert model.adapter_input_masks[key][512 + 1]
    model.unit_ranks['conv3'][1] = 0
    model.remove_recycled_adapter_support('conv3', [1])
    assert not model.adapter_input_masks[key][2:4].any()
    assert model.adapter_input_masks[key][512 + 1]
    model.unit_ranks['gru'][1] = 0
    model.remove_recycled_adapter_support('gru', [1])
    assert not model.adapter_input_masks[key][512 + 1]


def test_legacy_checkpoint_without_mode_keeps_original_adapter_and_logits():
    model, key = model_with_adapter('fc1', 'legacy_output')
    with torch.no_grad():
        model.adapters[key].U.weight.normal_(std=.1)
    state = snapshot_denice_state(model)
    state.pop('adapter_mode')
    restored = DeNICEModel((17, 2), 4).eval()
    restored.configure_adapter_mode('linear_input')
    restore_denice_state(restored, None, state)
    restored.load_state_dict(model.state_dict(), strict=True)
    x = torch.randn(3, 17, 2)
    assert restored.adapter_mode == 'legacy_output'
    assert key.endswith('__v1')
    torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)


def test_combined_eval_returns_adapter_logits_and_unchanged_router_activations():
    model, key = model_with_adapter('conv3')
    x = torch.randn(4, 17, 2)
    baseline = model.get_context_activations_per_sample(x)
    with torch.no_grad():
        model.adapters[key].U.weight.normal_(std=.5)
    logits, activations = model.get_output_and_context_activations(x)
    torch.testing.assert_close(logits, model(x), rtol=0, atol=0)
    for layer in activations:
        torch.testing.assert_close(activations[layer], baseline[layer], rtol=0, atol=0)


@pytest.mark.parametrize('policy', ['backbone_nomask', 'pred_hard', 'oracle_hard'])
def test_routed_eval_restores_training_adapter_even_on_early_return_or_error(policy, monkeypatch):
    from fed_learning.training.denice_eval import _denice_routed_logits_with_episodes
    from types import SimpleNamespace
    model, key = model_with_adapter('fc1')
    model.train()
    detector = SimpleNamespace(episode_classes={0: [0, 1], 1: [2, 3]})
    monkeypatch.setattr('fed_learning.training.denice_eval._route_episodes_with_scores',
                        lambda *args: (np.zeros(3, dtype=int), None))
    arguments = dict(model=model, X_batch=torch.randn(3, 17, 2), context_detector=detector,
                     seen_classes=[0, 1, 2, 3], device='cpu', inference_policy=policy)
    if policy == 'oracle_hard':
        with pytest.raises(ValueError, match='requires one episode'):
            _denice_routed_logits_with_episodes(**arguments)
    else:
        _denice_routed_logits_with_episodes(**arguments)
    assert model.active_adapters == {'fc1': key}
    assert model.training and model.dropout.training
    model.zero_grad(set_to_none=True)
    torch.nn.functional.cross_entropy(model.forward_output(arguments['X_batch']),
                                     torch.full((3,), 2)).backward()
    assert model.adapters[key].U.weight.grad.abs().sum() > 0


def test_new_task_adapter_training_preserves_previous_context_logits():
    from fed_learning.strategies.incremental.nice import (
        increase_unit_ranks, update_freeze_masks, drop_young_to_learner)
    model = DeNICEModel((16, 1), 4).eval()
    model.configure_adapter_mode('linear_input')
    model.fixed_task_allocation = model.structural_protection = True
    for ages in model.unit_ranks.values():
        ages[:len(ages)//2] = 1
    for layer in ('conv3', 'gru', 'fc1'):
        key = model.add_adapter(0, layer)
        with torch.no_grad():
            model.adapters[key].U.weight.normal_(std=.1)
    drop_young_to_learner(model)
    model.protect_task_connections()
    model.protect_active_adapter_inputs()
    increase_unit_ranks(model)
    update_freeze_masks(model)
    model.freeze_bn_for_mature()
    x = torch.randn(5, 16, 1)
    old_logits = model(x)[:, :2].detach().clone()
    old_weights = deepcopy(model.adapters.state_dict())
    model.clear_active_adapters()
    for ages in model.unit_ranks.values():
        ages[ages == 0] = 1
    for layer in ('conv3', 'gru', 'fc1'):
        model.add_adapter(1, layer)
    drop_young_to_learner(model)
    model.protect_task_connections()
    model.protect_active_adapter_inputs()
    update_freeze_masks(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    model.train()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model.forward_output(x), torch.tensor([2,3,2,3,2])).backward()
        model.reset_frozen_gradients()
        optimizer.step()
    model.eval()
    model.set_active_context(0)
    for name, value in old_weights.items():
        torch.testing.assert_close(model.adapters.state_dict()[name], value, rtol=0, atol=0)
    torch.testing.assert_close(model(x)[:, :2], old_logits, rtol=1e-6, atol=1e-7)


def test_peer_adapter_with_different_output_support_cannot_change_receiver():
    from fed_learning.training.decentralized_denice_il import _aggregate_round
    from fed_learning.strategies.decentralized.denice_capsule import build_context_capsule
    left, key = model_with_adapter('fc1')
    models = {0: left, 1: deepcopy(left)}
    models[1].adapter_output_masks[key][128] = False
    before = {cid: deepcopy(model.state_dict()) for cid, model in models.items()}
    x = torch.randn(4, 17, 2)
    capsules = {}
    for cid, model in models.items():
        with torch.no_grad():
            model.adapters[key].U.weight += 1. if cid == 0 else 8.
        capsules[cid] = build_context_capsule(model, x, client_id=cid, task_id=1, round_id=0,
            label_histogram={2: 1.}, label_set=[2], sample_count=4, reliability=1.,
            labels=torch.full((4,), 2), capsule_mode='paper', fisher_samples=0)
    result = _aggregate_round(client_ids=[0, 1], models=models, capsules=capsules,
        config={'denice_clustering_mode': 'paper', 'denice_similarity_threshold': -1.,
                'denice_aggregation_update_mode': 'local_delta', 'denice_aggregation_rho': 'reserve',
                'denice_age_merge_policy': 'none'}, device=torch.device('cpu'), before_local_states=before)
    np.testing.assert_allclose(result['alpha_debug'][0]['alphas'], [.5, .5])
    torch.testing.assert_close(left.adapters[key].U.weight,
                               before[0][f'adapters.{key}.U.weight'] + .5)


def test_whole_evaluation_preserves_active_adapter_and_training_mode():
    from fed_learning.training.denice_eval import evaluate_denice_model
    model, key = model_with_adapter('fc1')
    model.train()
    evaluate_denice_model(model, {'X_test': torch.randn(4, 17, 2), 'y_test': torch.tensor([0,1,2,3])},
                          'cpu', None, [0,1,2,3], batch_size=2)
    assert model.active_adapters == {'fc1': key}
    assert model.training and model.dropout.training
