"""Checks for the local incremental readout, beyond testing code syntax."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from fed_learning.strategies.incremental.denice_classifier import (
    fit_balanced_lda, fit_local_classifier, local_classifier_logits,
    herding_indices, classifier_config,
)
from fed_learning.strategies.incremental.denice_replay import LocalReplay, ReplayConfig
from fed_learning.models.denice_model import DeNICEModel
from fed_learning.training.checkpoint_state import snapshot_denice_state, restore_denice_state
from fed_learning.training.denice_eval import _denice_routed_logits_with_episodes
from fed_learning.training.denice_delta_checkpoint import compact_algorithm_states


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    yield
    torch.set_num_threads(before)


class ToyFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)
        self.linear.weight.data.copy_(torch.eye(4))
        self.num_classes = 8
        self.active_adapters = {'fc1': 'original'}

    def clear_active_adapters(self):
        self.active_adapters = {}

    def penultimate_features(self, x):
        return self.linear(x)

    def forward(self, x):
        return torch.cat([self.linear(x), self.linear(x)], dim=1)


def clusters():
    # Train and held-out noise are independent; labels are deliberately sparse.
    x = torch.eye(4).repeat_interleave(16, 0) + .02 * torch.randn(64, 4)
    y = torch.tensor([0, 2, 5, 7]).repeat_interleave(16)
    heldout = torch.eye(4).repeat_interleave(16, 0) + .02 * torch.randn(64, 4)
    return x, y, heldout


def test_incremental_head_retains_old_support_and_corrects_forced_wrong_router():
    model = ToyFeatures()
    x, y, test = clusters()
    memory = LocalReplay(ReplayConfig(capacity=16))
    cfg = {'denice_classifier_enabled': True, 'denice_classifier_per_class': 16}
    fit_local_classifier(model, memory, x[:32], y[:32], cfg, task_id=0, client_id=0)
    memory.commit(model, x[:32], y[:32], 0)
    fit_local_classifier(model, memory, x[32:], y[32:], cfg, task_id=1, client_id=0)
    before = deepcopy(model.state_dict())
    logits = local_classifier_logits(model, test, [0, 2, 5, 7])
    assert (logits.argmax(1) == y).float().mean() == 1
    assert (logits[:, [1,3,4,6]] < -1e8).all()
    assert model.active_adapters == {'fc1': 'original'}
    for key in before:
        torch.testing.assert_close(before[key], model.state_dict()[key], atol=0, rtol=0)


def test_equal_priors_do_not_favor_the_majority_class():
    x = torch.tensor([[1.,0.], [1.,.01], [0.,1.], [.01,1.]])
    y = torch.tensor([0,0,1,1])
    balanced = fit_balanced_lda(x, y)
    imbalanced = fit_balanced_lda(torch.cat([x[:2].repeat(50,1), x[2:]]),
                                torch.tensor([0]*100+[1]*2))
    # Query at each class mean must keep both classes, not collapse to the head class.
    query = torch.eye(2)
    assert (query @ imbalanced['weight'] + imbalanced['bias']).argmax(1).tolist() == [0,1]
    assert balanced['classes'].tolist() == imbalanced['classes'].tolist() == [0,1]


def test_lda_singular_covariance_and_singletons_are_finite():
    state = fit_balanced_lda(torch.ones(3, 16), torch.tensor([1,1,7]))
    assert torch.isfinite(state['weight']).all()
    assert torch.isfinite(state['bias']).all()
    one = fit_balanced_lda(torch.ones(1,16), torch.tensor([3]))
    assert one['classes'].tolist() == [3]


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_real_readout_preserves_backbone_rng_bn_and_checkpoint(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    model = DeNICEModel((16,1), 4).to(device).train()
    x, y = torch.randn(8,16,1), torch.tensor([0,2]*4)
    memory = LocalReplay(ReplayConfig(capacity=4))
    before = deepcopy(model.state_dict())
    rng = torch.get_rng_state()
    audit = fit_local_classifier(model, memory, x, y, {'denice_classifier_enabled': True},
                                 task_id=0, client_id=7)
    assert audit['classes'] == [0,2]
    assert model.training
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    for key in before:
        torch.testing.assert_close(model.state_dict()[key], before[key], rtol=0, atol=0)
    assert not any('classifier' in key for key in model.state_dict())
    state = snapshot_denice_state(model)
    compact = compact_algorithm_states({7: {'denice': state}})[7]['denice']
    assert compact['local_classifier']['weight'].dtype == torch.float32
    restored = DeNICEModel((16,1),4).to(device)
    restore_denice_state(restored, None, compact)
    restored.load_state_dict(before)
    a = local_classifier_logits(model, x.to(device), [0,2])
    b = local_classifier_logits(restored, x.to(device), [0,2])
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_local_route_ignores_wrong_episode_mask_but_keeps_router_diagnostic(monkeypatch):
    model = ToyFeatures()
    x, y, test = clusters()
    fit_local_classifier(model, None, x, y, {'denice_classifier_enabled': True}, task_id=1, client_id=0)
    detector = SimpleNamespace(episode_classes={0:[0,2], 1:[5,7]})
    monkeypatch.setattr('fed_learning.training.denice_eval._route_episodes',
                        lambda *args: np.zeros(len(test), dtype=np.int64))
    output, episodes = _denice_routed_logits_with_episodes(
        model, test, detector, [0,2,5,7], 'cpu', route_mode='local_lda')
    assert (output.argmax(1) == y).all()
    assert (episodes == 0).all()  # Do not fabricate improved route accuracy.
    assert model.active_adapters == {'fc1':'original'}


def test_missing_head_fails_instead_of_silently_evaluating_baseline():
    with pytest.raises(ValueError, match='fitted'):
        local_classifier_logits(ToyFeatures(), torch.randn(2,4), [0,2])


def test_peer_bootstrap_does_not_copy_private_classifier():
    from fed_learning.training.decentralized_denice_il import _bootstrap_denice_model
    source = DeNICEModel((16,1), 4)
    source.local_classifier = fit_balanced_lda(torch.randn(8,256), torch.tensor([0,1]*4))
    target = _bootstrap_denice_model(source, {'input_shape':(16,1), 'num_classes':4}, torch.device('cpu'))
    assert target.local_classifier is None
    assert source.local_classifier is not None


def test_validation_without_holdout_preserves_original_path():
    model = ToyFeatures()
    fit_local_classifier(model, None, torch.randn(4,4), torch.tensor([0,0,2,2]),
                         {'denice_classifier_enabled':True, 'denice_classifier_validation_select':True},
                         task_id=0, client_id=0, detector=SimpleNamespace())
    assert model.local_classifier['blend_weight'] == 0


def test_herding_represents_mean_without_duplicate_selection_and_replay_is_bounded():
    features = torch.tensor([[1.,0.], [0.,1.], [1.,1.], [1.,1.]])
    index = herding_indices(features, 4)
    assert len(index.unique()) == 4
    assert index[0] in (2,3)
    model = ToyFeatures()
    memory = LocalReplay(ReplayConfig(capacity=8, selection='herding', candidate_limit=16))
    x, y, _ = clusters()
    memory.commit(model, x[:32], y[:32], 0)
    old = deepcopy(memory.entries)
    memory.commit(model, x[32:], y[32:], 1)
    assert len(memory) == 8
    assert memory.stats()['class_counts'] == {0:2,2:2,5:2,7:2}
    for c in (0,2):
        for row, target in zip(memory.entries[c]['x'], memory.entries[c]['logits']):
            source = torch.nonzero((old[c]['x'] == row).all(1)).flatten()[0]
            torch.testing.assert_close(target, old[c]['logits'][source])
    assert model.active_adapters == {'fc1':'original'}


def test_legacy_memory_configuration_loads_without_inventing_herding():
    memory = LocalReplay(ReplayConfig(capacity=8))
    state = memory.state_dict()
    state['config'].pop('selection')
    state['config'].pop('candidate_limit')
    assert LocalReplay.from_state(memory.config, state).config.selection == 'priority'
    with pytest.raises(ValueError):
        LocalReplay.from_state(ReplayConfig(capacity=8, selection='herding'), state)


def test_validation_old_samples_are_disjoint_from_classifier_fit(monkeypatch):
    import fed_learning.strategies.incremental.denice_classifier as module
    model = ToyFeatures()
    x, y, heldout = clusters()
    memory = LocalReplay(ReplayConfig(capacity=16))
    memory.commit(model, x[:32], y[:32], 0)
    fitted, validated = [], []
    original = module.encode_features
    def capture_fit(model, inputs, batch_size):
        fitted.extend(inputs.clone())
        return original(model, inputs, batch_size)
    def capture_validation(model, detector, inputs, labels, classes, batch_size):
        validated.extend(inputs.clone())
        return {'selected_weight': .5, 'reason': 'test'}
    monkeypatch.setattr(module, 'encode_features', capture_fit)
    monkeypatch.setattr(module, 'select_readout_weight', capture_validation)
    fit_local_classifier(model, memory, x[32:], y[32:],
                         {'denice_classifier_enabled':True, 'denice_classifier_validation_select':True},
                         task_id=1, client_id=0, detector=SimpleNamespace(),
                         validation_inputs=heldout[32:], validation_labels=y[32:])
    assert validated and fitted
    a, b = torch.stack(fitted), torch.stack(validated)
    assert not (a[:,None,:] == b[None,:,:]).all(-1).any()
    assert model.local_classifier['blend_weight'] == .5


def test_validation_mixing_selects_predictive_head_without_test_labels(monkeypatch):
    import fed_learning.strategies.incremental.denice_classifier as module
    model = ToyFeatures()
    x, y, heldout = clusters()
    fit_local_classifier(model, None, x, y, {'denice_classifier_enabled':True}, task_id=0, client_id=0)
    def wrong_baseline(model, batch, *args, **kwargs):
        logits = torch.full((len(batch), 8), -100.)
        logits[:,0] = 100.
        return logits, np.zeros(len(batch), dtype=np.int64)
    monkeypatch.setattr('fed_learning.training.denice_eval._denice_routed_logits_with_episodes', wrong_baseline)
    selected = module.select_readout_weight(model, None, heldout, y, [0,2,5,7], 16)
    assert selected['selected_weight'] > 0
    assert selected['scores'][0]['balanced_accuracy'] == .25
    assert max(s['balanced_accuracy'] for s in selected['scores']) == 1.


@pytest.mark.parametrize('config', [{'denice_classifier_per_class':0},
                                  {'denice_classifier_shrinkage':0},
                                  {'denice_classifier_temperature':float('nan')}])
def test_invalid_classifier_config(config):
    with pytest.raises(ValueError):
        classifier_config(config)
