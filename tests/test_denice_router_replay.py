from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn

from fed_learning.servers.nice_server import ContextDetector
from fed_learning.training.checkpoint_state import snapshot_context_detector, restore_context_detector
from fed_learning.strategies.incremental.denice_router_replay import refresh_replay_router, router_replay_config


class Features(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.))
        self.unit_ranks = {layer: np.array([2, 1]) for layer in ('conv1', 'conv2', 'conv3', 'gru')}

    def get_context_activations_per_sample(self, x):
        return {layer: x * self.weight for layer in self.unit_ranks}


def setup():
    model = Features()
    detector = ContextDetector(router_mode='binary_cosine')
    detector.stable_feature_mask = np.tile([True, False], 4)
    detector.episode_classes = {0: [0], 1: [1]}
    detector.activation_memory = {0: np.zeros((2, 8)), 1: np.zeros((2, 8))}
    memory = SimpleNamespace(entries={0: {'x': torch.zeros(2, 2)}})
    return model, detector, memory


def test_new_units_can_disambiguate_old_and_new_episodes_without_raw_bank():
    model, detector, memory = setup()
    original_mask = detector.stable_feature_mask.copy()
    controls = {'per_class': 2, 'batch_size': 1}
    x, y = torch.tensor([[0., 3.], [0., 3.]]), torch.ones(2, dtype=torch.long)
    rng = torch.get_rng_state().clone()
    audit = refresh_replay_router(detector, model, memory, x, y, 1, controls)
    assert audit['updated'] and audit['feature_count'] == 8
    assert model.training and torch.equal(rng, torch.get_rng_state())
    assert np.array_equal(detector.stable_feature_mask, original_mask)
    assert not detector.reference_input_memory and not detector.retain_reference_inputs
    queries = torch.tensor([[0., 0.], [0., 3.]])
    with torch.no_grad():
        vectors = detector._binarize_per_sample(model, queries)
    assert detector.predict_episodes_batch(vectors).tolist() == [0, 1]
    restored = ContextDetector()
    restore_context_detector(restored, snapshot_context_detector(detector))
    assert restored.predict_episodes_batch(vectors).tolist() == [0, 1]
    assert np.array_equal(restored.routing_feature_mask, detector.routing_feature_mask)


def test_missing_old_raw_memory_keeps_previous_coordinate_system():
    model, detector, _ = setup()
    original = deepcopy(detector.activation_memory)
    audit = refresh_replay_router(detector, model, None, torch.ones(2, 2), torch.ones(2).long(),
                                 1, {'per_class': 2, 'batch_size': 1})
    assert audit['reason'] == 'missing_episode_replay'
    assert not hasattr(detector, 'routing_feature_mask')
    for key in original:
        np.testing.assert_array_equal(detector.activation_memory[key], original[key])


def test_nonfinite_features_do_not_publish_partial_refresh():
    model, detector, memory = setup()
    with pytest.raises(ValueError, match='Nonfinite'):
        refresh_replay_router(detector, model, memory, torch.full((2, 2), float('nan')),
                              torch.ones(2).long(), 1, {'per_class': 2, 'batch_size': 1})
    assert model.training
    assert not hasattr(detector, 'routing_feature_mask')


def test_reject_shared_context_and_no_replay():
    with pytest.raises(ValueError):
        router_replay_config({'denice_router_replay_enabled': True})
    with pytest.raises(ValueError):
        router_replay_config({'denice_router_replay_enabled': True, 'denice_memory_policy': 'local_replay',
                              'denice_replay_capacity': 128, 'denice_shared_context_eval': True})
