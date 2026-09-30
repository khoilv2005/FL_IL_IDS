from collections import OrderedDict
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from fed_learning.models.denice_model import DeNICEModel
from fed_learning.training.checkpoint_state import snapshot_denice_state, restore_denice_state
from fed_learning.strategies.incremental.nice import update_freeze_masks
from fed_learning.strategies.decentralized.denice_aggregation import age_aware_aggregate
from fed_learning.strategies.incremental.denice_normalization import calibrate_plastic_bn


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_zero_initial_correction_preserves_logits_and_rng():
    model = DeNICEModel((16, 1), 4).eval()
    x = torch.randn(4, 16, 1)
    with torch.no_grad():
        expected = model(x)
    rng = torch.get_rng_state().clone()
    model.configure_continual_head(32)
    assert torch.equal(rng, torch.get_rng_state())
    with torch.no_grad():
        torch.testing.assert_close(model(x), expected, atol=0, rtol=0)
        logits, _ = model.get_output_and_context_activations(x)
        torch.testing.assert_close(logits, model(x))


def test_global_ce_updates_old_residual_rows_but_not_mature_core():
    model = DeNICEModel((16, 1), 4)
    model.configure_continual_head(32)
    for layer in model.unit_ranks:
        model.unit_ranks[layer][:] = 2
    model.unit_ranks['fc2'][:] = [2, 1, 0, 0]
    update_freeze_masks(model)
    logits = model.forward_output(torch.randn(4, 16, 1))
    assert (logits[:, 2:] == -1e4).all()
    F.cross_entropy(logits, torch.ones(4, dtype=torch.long)).backward()
    model.reset_frozen_gradients()
    assert model.continual_head[-1].bias.grad[0] > 0
    assert torch.count_nonzero(model.fc1.weight.grad) == 0
    assert torch.count_nonzero(model.fc2.weight.grad[0]) == 0


def test_checkpoint_reconstructs_branch_before_loading_weights():
    model = DeNICEModel((16, 1), 4).eval()
    model.configure_continual_head(16)
    with torch.no_grad():
        model.continual_head[-1].bias.add_(1)
    restored = DeNICEModel((16, 1), 4).eval()
    restore_denice_state(restored, None, snapshot_denice_state(model))
    restored.load_state_dict(model.state_dict(), strict=True)
    x = torch.randn(3, 16, 1)
    with torch.no_grad():
        torch.testing.assert_close(restored(x), model(x), atol=0, rtol=0)


def test_continual_parameters_receive_peer_deltas():
    local = OrderedDict({'continual_head.5.bias': torch.zeros(4)})
    deltas = [OrderedDict({'continual_head.5.bias': torch.ones(4)})]
    result = age_aware_aggregate(local, {}, deltas, np.array([1.]))
    torch.testing.assert_close(result['continual_head.5.bias'], torch.ones(4))


@pytest.mark.parametrize('width', [-1, 1.5, True])
def test_invalid_architecture(width):
    with pytest.raises(ValueError):
        DeNICEModel((16, 1), 4).configure_continual_head(width)


def test_population_bn_matches_moments_and_preserves_mature_channels():
    model = DeNICEModel((16, 1), 4)
    model.unit_ranks['conv1'][:] = 1
    model.unit_ranks['conv1'][0] = 2
    x = torch.randn(8, 16, 1) + 4
    old_mean, old_var = model.bn1.running_mean.clone(), model.bn1.running_var.clone()
    with torch.no_grad():
        raw = F.conv1d(x.permute(0, 2, 1), model.conv1.weight, model.conv1.bias, padding=1)
    rng = torch.get_rng_state().clone()
    audit = calibrate_plastic_bn(model, x, batch_size=3)
    assert model.training and torch.equal(rng, torch.get_rng_state())
    assert audit['updated_channels']['conv1'] == 63
    torch.testing.assert_close(model.bn1.running_mean[1:], raw.mean((0, 2))[1:])
    torch.testing.assert_close(model.bn1.running_var[1:], raw.var((0, 2), unbiased=True)[1:])
    assert model.bn1.running_mean[0] == old_mean[0]
    assert model.bn1.running_var[0] == old_var[0]


def test_bn_calibration_failure_rolls_back_buffers_and_hooks():
    model = DeNICEModel((16, 1), 4)
    model.unit_ranks['conv1'][:] = 1
    old = {name: value.clone() for name, value in model.named_buffers()}
    with pytest.raises(ValueError, match='Nonfinite'):
        calibrate_plastic_bn(model, torch.full((3, 16, 1), float('nan')))
    assert model.training and not model.bn1._forward_pre_hooks
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, old[name], atol=0, rtol=0)
