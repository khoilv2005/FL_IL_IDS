from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from fed_learning.strategies.decentralized import denice_appliance as ap


def packet(source=1, cluster=0, count=1, row=(2., 4.), mask=(1., 1.)):
    return ap.Packet(0, 2, source, cluster, 1, 'schema', 'features', 'v1', count,
                     torch.tensor(row), torch.tensor(2.), torch.tensor(mask), torch.tensor(1.))


def test_election_failover_and_empty():
    labels, known = {1: 0, 2: 0, 3: 1}, {1: {0}, 2: {0, 1}, 3: {2}}
    assert ap.elect_leaders(labels, known, {1, 2, 3}) == {0: 2, 1: 3}
    assert ap.elect_leaders(labels, known, {1, 2}, {2}) == {0: 1, 1: None}
    assert ap.elect_leaders({}, {}, set()) == {}
    assert ap.elect_leaders({1: 0, 2: 0}, {1: {0}, 2: {0}}, {1, 2}) == {0: 1}


def test_two_stage_weighting_missing_coordinates():
    packets = [packet(count=1), packet(2, 1, 3, (6., 8.), (1., 0.))]
    row, bias = ap.aggregate_rows(packets, torch.tensor([0., 9.]), torch.tensor(0.),
                                  torch.ones(2), torch.tensor(0.))
    assert torch.equal(row, torch.tensor([5., 4.]))
    assert bias == 0  # receiver mask is protected
    row, _ = ap.aggregate_rows([], torch.tensor([7., 9.]), torch.tensor(1.), torch.ones(2), torch.tensor(1.))
    assert torch.equal(row, torch.tensor([7., 9.]))


def test_packet_freshness_nonfinite_and_version():
    p = packet()
    assert ap.valid_packet(p, 0, 2, 'schema', {1: 'v1'})
    for bad in [replace(p, task=1), replace(p, round=1), replace(p, schema='other'),
                replace(p, version='old'), replace(p, count=0), replace(p, row=torch.tensor([float('nan'), 0.]))]:
        assert not ap.valid_packet(bad, 0, 2, 'schema', {1: 'v1'})


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc2 = nn.Linear(1, 2)
        self.active_adapters = {}
        self.weight_masks = {'fc2': torch.ones(2, 1)}
        self.bias_masks = {'fc2': torch.ones(2)}
        self.ages = {'fc2': np.ones(2, dtype=int)}
        self.register_buffer('counter', torch.tensor(5))
        with torch.no_grad():
            self.fc2.weight.copy_(torch.tensor([[-1.], [-2.]]))
            self.fc2.bias.zero_()

    def forward(self, x):
        return self.fc2(x)

    def get_masks_state(self):
        return {'weight_fc2': self.weight_masks['fc2'], 'bias_fc2': self.bias_masks['fc2']}

    def get_neuron_ages_state(self):
        return self.ages


@pytest.fixture
def problem(monkeypatch):
    def logits(model, x, detector, seen, device, **kwargs):
        result = model(x)
        allowed = set(sum(detector.episode_classes.values(), []))
        for c in range(2):
            if c not in allowed:
                result[:, c] = -100
        return result, None
    monkeypatch.setattr(ap, '_denice_routed_logits_with_episodes', logits)
    model = Tiny()
    detector = SimpleNamespace(episode_classes={0: [0, 1]})
    validator = ap.LocalValidator({'appliance_min_evidence': 2})
    validator.add(torch.tensor([[-1.], [-1.], [1.], [1.]]), torch.tensor([0, 0, 1, 1]))
    return model, detector, validator


def test_private_shadow_accept_reject_restore(problem):
    model, detector, validator = problem
    state = deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    accepted, _ = ap.shadow_check(model, detector, validator, [0, 1], 1, 0,
                                 torch.tensor([-3.]), torch.tensor(0.))
    assert not accepted
    assert all(torch.equal(model.state_dict()[k], v) for k, v in state.items())
    accepted, _ = ap.shadow_check(model, detector, validator, [0, 1], 1, 0,
                                 torch.tensor([1.]), torch.tensor(0.))
    assert accepted and model.fc2.weight[1] == 1
    assert model.training and model.counter == 5
    assert torch.equal(rng, torch.get_rng_state())


def test_unknown_no_evidence_and_historical_bank(problem):
    model, detector, validator = problem
    assert not validator.sufficient(2)
    assert ap.shadow_check(model, detector, validator, [0, 1], 2, 0) == (False, 'unknown_gap')
    validator.add(torch.ones(100, 1), torch.full((100,), 2))
    assert len(validator.bank[2][1]) == 64
    assert set(validator.bank) == {0, 1, 2}


def test_overlay_transfer_protection_drop_and_failure(problem):
    model, detector, validator = problem
    cfg = json.loads((Path(__file__).parents[1] / 'configs/denice_appliance.json').read_text())
    cfg.update(appliance_min_evidence=2, appliance_eta=1.)
    protocol = ap.ApplianceProtocol(cfg)
    donor = deepcopy(model)
    with torch.no_grad():
        donor.fc2.weight[1] = 1.
    models, detectors = {0: model, 1: donor}, {0: detector, 1: deepcopy(detector)}
    for cid in models:
        protocol.validators[cid] = deepcopy(validator)
        protocol.knowledge[cid] = {0, 1}
        protocol.counts[cid] = {0: 8, 1: 8}
    audit = protocol.run_round(models, detectors, {0: 0, 1: 1}, 0, 0, {0: [0, 1]})
    assert audit['accepted'] == 1
    assert audit['leader_deliveries'] == 2 * audit['packets']
    assert audit['inter_leader_deliveries'] == audit['packets']
    assert model.fc2.weight[1] == 1
    with torch.no_grad():
        model.fc2.weight[1] = -2.
    model.ages['fc2'][1] = 2
    audit = protocol.run_round(models, detectors, {0: 0, 1: 1}, 0, 1, {0: [0, 1]})
    assert model.fc2.weight[1] == -2.
    assert any(v['reason'] == 'protected_row' for v in audit['decisions'])
    cfg['appliance_drop_probability'] = 1.
    assert protocol.run_round(models, detectors, {0: 0, 1: 1}, 0, 2, {0: [0, 1]})['packets'] == 0
    cfg['appliance_failed_servers'] = [0, 1]
    assert protocol.run_round(models, detectors, {0: 0, 1: 1}, 0, 3, {0: [0, 1]})['leaders'] == {0: None, 1: None}


def test_bad_configuration():
    cfg = json.loads((Path(__file__).parents[1] / 'configs/denice_appliance.json').read_text())
    ap.validate_config(cfg)
    for changes in [{'resume_state_path': 'checkpoint.pt'}, {'appliance_eta': 2},
                    {'denice_capsule_mode': 'paper'}, {'appliance_min_evidence': 0}]:
        with pytest.raises(ValueError):
            ap.validate_config({**cfg, **changes})


def test_registration_only_and_exception_do_not_change_weights(problem, monkeypatch):
    model, detector, validator = problem
    with torch.no_grad():
        model.fc2.weight[0] = -20.
        model.fc2.weight[1] = 20.
    detector.episode_classes = {0: [0]}
    state = deepcopy(model.state_dict())
    accepted, reason = ap.shadow_check(model, detector, validator, [0, 1], 1, 0)
    assert accepted and reason == 'registration'
    assert detector.episode_classes == {0: [0, 1]}
    assert all(torch.equal(model.state_dict()[k], v) for k, v in state.items())
    def broken(*args, **kwargs):
        raise RuntimeError('injected scoring failure')
    monkeypatch.setattr(ap, '_denice_routed_logits_with_episodes', broken)
    with pytest.raises(RuntimeError):
        ap.shadow_check(model, detector, validator, [0, 1], 1, 0)
    assert model.training
    assert all(torch.equal(model.state_dict()[k], v) for k, v in state.items())


def test_split_reproducible_disjoint_and_singleton_retained():
    from fed_learning.training.decentralized_denice_il import _split_local_validation
    x = torch.arange(21).reshape(-1, 1).float()
    y = torch.tensor([0] * 10 + [1] * 10 + [2])
    train_x, train_y, val_x, val_y = _split_local_validation(x, y, .2, 43)
    assert not set(train_x.flatten().tolist()) & set(val_x.flatten().tolist())
    assert len(train_y) + len(val_y) == 21
    assert 2 in train_y and 2 not in val_y
    again = _split_local_validation(x, y, .2, 43)
    assert all(torch.equal(a, b) for a, b in zip((train_x, train_y, val_x, val_y), again))


def test_fingerprint_detects_feature_state_and_structure(problem):
    model, _, _ = problem
    donor = deepcopy(model)
    donor.counter += 1
    assert ap.fingerprint(model) == ap.fingerprint(donor)
    assert ap.fingerprint(model, True) != ap.fingerprint(donor, True)
    donor.ages['fc1'] = np.array([1])
    assert ap.fingerprint(model) != ap.fingerprint(donor)
