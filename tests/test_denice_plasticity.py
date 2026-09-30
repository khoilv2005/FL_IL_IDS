import numpy as np
import pytest
import torch
from fed_learning.models.denice_model import DeNICEModel
from fed_learning.strategies.incremental.denice_plasticity import (
    plasticity_config, capacity_plan, allocate_capacity, consolidate, elastic_loss_factory)
from fed_learning.training.checkpoint_state import snapshot_denice_state, restore_denice_state


def test_capacity_horizon_ignores_run_stop_and_protects_existing_units():
    model = DeNICEModel((16, 1), 34)
    for ranks in model.unit_ranks.values():
        ranks[:] = 0
        ranks[0] = 4
        ranks[1] = 1
    controls = plasticity_config({})
    plan = capacity_plan(model, [0, 1], 0, 6, controls)
    allocate_capacity(model, plan)
    for layer, ranks in model.unit_ranks.items():
        assert ranks[0] == 4 and ranks[1] == 1
        if layer != 'fc2':
            assert plan['adaptive_allocation'][layer] == int(np.ceil((len(ranks) - 2) / 4))
    last = capacity_plan(model, [2], 5, 6, controls)
    allocate_capacity(model, last)
    assert all(not (r == 0).any() for n, r in model.unit_ranks.items() if n != 'fc2')
    assert capacity_plan(model, [3], 5, 6, controls)['adapters_to_add'] == ['conv3', 'gru', 'fc1']
    assert not capacity_plan(model, [], 5, 6, controls)['adapters_to_add']


def test_importance_maturation_elastic_gradient_and_checkpoint():
    model = DeNICEModel((16, 1), 4)
    for ranks in model.unit_ranks.values():
        ranks[:] = 0
        ranks[:10] = 1
    fisher = {name: torch.ones_like(p) for name, p in model.named_parameters()}
    controls = plasticity_config({})
    audit = consolidate(model, fisher, controls)
    assert audit['fc1'] == dict(graduated=8, retained_plastic=2)
    assert (model.unit_ranks['fc1'][:8] == 2).all()
    assert (model.unit_ranks['fc2'] == 2).all()
    elastic = elastic_loss_factory(model, 100.)
    assert elastic().item() == 0
    with torch.no_grad():
        model.fc1.weight.add_(.1)
    assert elastic().item() > 0
    elastic().backward()
    assert model.fc1.weight.grad[8:10].abs().sum() > 0
    assert model.fc1.weight.grad[:8].abs().sum() == 0
    clone = DeNICEModel((16, 1), 4)
    restore_denice_state(clone, None, snapshot_denice_state(model))
    torch.testing.assert_close(clone.elastic_state['anchor']['fc1.weight'], model.elastic_state['anchor']['fc1.weight'])
    assert clone.elastic_state['anchor']['fc1.weight'].data_ptr() != model.elastic_state['anchor']['fc1.weight'].data_ptr()


@pytest.mark.parametrize('key,value', [('denice_capacity_frontload', .5), ('denice_mature_fraction', 0),
                                      ('denice_elastic_strength', float('nan')), ('denice_elastic_decay', 2)])
def test_invalid_controls(key, value):
    with pytest.raises(ValueError):
        plasticity_config({key: value})


def test_no_fisher_evidence_falls_back_to_hard_maturation():
    model = DeNICEModel((16, 1), 4)
    for ranks in model.unit_ranks.values():
        ranks[:] = 1
    audit = consolidate(model, {}, plasticity_config({}))
    assert all(row['retained_plastic'] == 0 for row in audit.values())
    assert elastic_loss_factory(model, 100.) is None


def test_age_transition_keeps_old_recurrent_dependencies_but_blocks_new_ones():
    model = DeNICEModel((16, 1), 4)
    model.structural_protection = True
    model.unit_ranks['gru'][:] = 0
    model.unit_ranks['gru'][:3] = [2, 1, 1]
    model.protect_task_connections(provisional_gru=[1])
    mask = model.gru_connection_masks['weight_hh_l0']
    assert mask[0, 1] == 1  # Prior task's provisional input survives.
    assert mask[0, 2] == 0  # Newly allocated learner cannot alter mature state.
    model.protect_task_connections(provisional_gru=[1, 2])
    assert model.gru_connection_masks['weight_hh_l0'][0, 2] == 0  # Never reopen.


def test_balanced_bn_calibration_includes_private_old_classes_without_mutation():
    from types import SimpleNamespace
    from fed_learning.strategies.incremental.denice_normalization import balanced_calibration_inputs
    x = torch.ones(20, 16, 1)
    memory = SimpleNamespace(entries={0: {'x': torch.zeros(3, 16, 1)}})
    rng = torch.get_rng_state().clone()
    selected = balanced_calibration_inputs(x, torch.ones(20).long(), memory, limit=20)
    assert (selected[:, 0, 0] == 0).sum() == 10
    assert (selected[:, 0, 0] == 1).sum() == 10
    assert torch.equal(rng, torch.get_rng_state())
    assert memory.entries[0]['x'].shape[0] == 3
