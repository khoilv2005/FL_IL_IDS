from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from fed_learning.strategies.decentralized.denice_appliance import ApplianceProtocol
from fed_learning.training.denice_overlay_resume import restore, snapshot
from fed_learning.training.decentralized_denice_il import (
    _split_local_validation as split, _stratified_limit_client_task_data as limit)


def fixture():
    cfg = json.loads((Path(__file__).parents[1] / 'configs/denice_appliance.json').read_text())
    cfg.update(seed=42, random_seed=42, denice_overlay_rebuild_private_state=True,
               denice_max_clients=1, denice_max_train_samples_per_client=12)
    class Loader:
        def get_all_client_ids(self):
            return [0, 1, 2]
        def get_client_data(self, cid, task):
            if cid == 0:
                return torch.empty(0, 1), torch.empty(0, dtype=torch.long)
            return torch.arange(20).float().reshape(-1, 1) + 100 * task, torch.tensor([task] * 10 + [task + 1] * 10)
    loader = Loader()
    protocol = ApplianceProtocol(deepcopy(cfg))
    audits = []
    for task in range(2):
        x, y = loader.get_client_data(1, task)
        x, y = limit(x, y, 12, seed=42 + 10000 * task + 1)
        labels, counts = y.unique(return_counts=True)
        audits.append({'type': 'task_start', 'task': task, 'client_data': {1: {
            'num_samples': len(y), 'class_hist': {int(c): int(n) for c, n in zip(labels, counts)}}}})
        _, train_y, val_x, val_y = split(x, y, cfg['denice_validation_fraction'], 42 + 10000 * task + 1)
        protocol.add_local_task(1, val_x, val_y, train_y)
    state = {'config': deepcopy(cfg), 'meta': {'boundary': 'task', 'completed_task': 1}, 'debug_history': audits}
    return cfg, loader, protocol, state


def assert_equal(a, b):
    assert a.knowledge == b.knowledge
    assert a.counts == b.counts
    assert a.validators.keys() == b.validators.keys()
    for cid in a.validators:
        assert a.validators[cid].bank.keys() == b.validators[cid].bank.keys()
        for label, pair in a.validators[cid].bank.items():
            assert all(torch.equal(x, y) for x, y in zip(pair, b.validators[cid].bank[label]))


def test_reconstruction_matches_uninterrupted_and_preserves_rng():
    cfg, loader, original, state = fixture()
    resumed = ApplianceProtocol(cfg)
    rng = torch.get_rng_state().clone()
    restore(resumed, state, loader, split, limit)
    assert_equal(original, resumed)
    assert torch.equal(rng, torch.get_rng_state())


def test_saved_state_restores_without_reading_data():
    cfg, _, original, state = fixture()
    state['denice_appliance_state'] = snapshot(original)
    resumed = ApplianceProtocol(cfg)
    assert restore(resumed, state, None, None, None) == 'checkpoint'
    assert_equal(original, resumed)
    resumed.validators[1].bank[0][0].add_(99)
    assert not torch.equal(resumed.validators[1].bank[0][0], original.validators[1].bank[0][0])


@pytest.mark.parametrize('failure', ['seed', 'missing_audit', 'wrong_hist', 'non_overlay', 'round', 'no_opt_in'])
def test_reject_unsafe_resume(failure):
    cfg, loader, _, state = fixture()
    if failure == 'seed': cfg['seed'] = 43
    if failure == 'missing_audit': state['debug_history'] = []
    if failure == 'wrong_hist': state['debug_history'][0]['client_data'][1]['num_samples'] = 1
    if failure == 'non_overlay': state['config']['denice_appliance_enabled'] = False
    if failure == 'round': state['meta']['boundary'] = 'round'
    if failure == 'no_opt_in': cfg['denice_overlay_rebuild_private_state'] = False
    with pytest.raises(ValueError):
        restore(ApplianceProtocol(cfg), state, loader, split, limit)
