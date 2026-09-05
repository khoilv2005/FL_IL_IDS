"""Small causal invariants from the September 2026 forensic audit."""
from collections import OrderedDict
from copy import deepcopy
from io import BytesIO

import numpy as np
import pytest
import torch

from fed_learning.models.denice_model import DeNICEModel
from fed_learning.servers.nice_server import ContextDetector
from fed_learning.strategies.incremental.nice import update_freeze_masks
from fed_learning.strategies.decentralized.denice_aggregation import (
    age_aware_aggregate, aggregation_weights, build_compatible_mask,
)
from fed_learning.training.checkpoint_state import snapshot_denice_state, restore_denice_state
from fed_learning.training.denice_eval import _denice_routed_logits_with_episodes


@pytest.fixture(autouse=True)
def small_cpu_test():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(73)
    yield
    torch.set_num_threads(previous)


def model_detector():
    model = DeNICEModel((39, 1), 4).eval()
    for ranks in model.unit_ranks.values():
        ranks[:] = 2
    update_freeze_masks(model)
    model.add_adapter(0, 'fc1', set_active=False)
    model.add_adapter(1, 'gru', set_active=False)
    for adapter in model.adapters.values():
        torch.nn.init.normal_(adapter.U.weight, std=0.1)
    detector = ContextDetector(router_mode='multiclass')
    detector.episode_classes = {0: [0, 1], 1: [2, 3]}
    # Distinct train-only reference banks, unrelated to the evaluation batch.
    for ep in range(2):
        inputs = torch.randn(12, 39, 1) + ep*2
        detector.push_activations(model, inputs, ep, reference_data=inputs)
    detector.train_models(1)
    return model, detector


@pytest.mark.parametrize('policy', ['backbone_nomask', 'oracle_adapter_nomask', 'oracle_hard', 'pred_hard'])
def test_masks_and_router_roundtrip_preserves_logits(policy):
    model, detector = model_detector()
    # A legal masked state after aggregation: stored parameter is nonzero but
    # its connection mask is zero. Restoring dense weights must not reopen it.
    model.weight_masks['fc2'][0, :] = 0
    model.fc2.weight.data[0, :] = 2
    x = torch.randn(7, 39, 1)
    oracle = np.zeros(len(x), dtype=np.int64)
    before, route_before = _denice_routed_logits_with_episodes(
        model, x, detector, list(range(4)), 'cpu', inference_policy=policy, oracle_episodes=oracle)
    stream = BytesIO()
    torch.save({'weights': model.state_dict(), 'algorithm': snapshot_denice_state(model, detector)}, stream)
    stream.seek(0)
    state = torch.load(stream, weights_only=False)
    restored = DeNICEModel((39, 1), 4).eval()
    router = ContextDetector()
    restore_denice_state(restored, router, state['algorithm'])
    restored.load_state_dict(state['weights'])
    after, route_after = _denice_routed_logits_with_episodes(
        restored, x, router, list(range(4)), 'cpu', inference_policy=policy, oracle_episodes=oracle)
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    np.testing.assert_array_equal(route_before, route_after)
    for ep in detector.reference_input_memory:
        np.testing.assert_array_equal(router.reference_input_memory[ep], detector.reference_input_memory[ep])


def test_standalone_evaluator_restores_connection_masks():
    from eval_checkpoint import _make_denice_client_model
    model, detector = model_detector()
    model.weight_masks['fc2'][0] = 0
    checkpoint = {'config': {'input_shape': (39,1), 'num_classes': 4},
                  'client_model_states': {17: model.state_dict()},
                  'client_algorithm_states': {17: {'denice': snapshot_denice_state(model, detector)}}}
    restored, _ = _make_denice_client_model(checkpoint, 17, 'cpu')
    assert torch.equal(model.weight_masks['fc2'], restored.weight_masks['fc2'])


def test_mature_parameters_local_step_and_aggregation_are_protected():
    model = DeNICEModel((39,1), 4)
    for ranks in model.unit_ranks.values():
        ranks[:] = 1
        ranks[:len(ranks)//2] = 2
    update_freeze_masks(model)
    model.freeze_bn_for_mature()
    model.add_adapter(1, 'fc1')
    before = deepcopy(model.state_dict())
    keep = build_compatible_mask(before, model.unit_ranks)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss = torch.nn.functional.cross_entropy(model.forward_output(torch.randn(8,39,1)), torch.full((8,), 3))
    loss.backward()
    model.reset_frozen_gradients()
    assert model.adapters[next(iter(model.adapters))].U.weight.grad.abs().sum() > 0
    optimizer.step()
    changed = False
    for name, value in model.state_dict().items():
        forbidden = keep[name] == 0
        assert torch.equal(value[forbidden], before[name][forbidden]), name
        if value.is_floating_point():
            changed |= not torch.equal(value, before[name])
    assert changed
    state = OrderedDict((k,v.detach().clone()) for k,v in model.state_dict().items())
    delta = OrderedDict((k,torch.ones_like(v)) for k,v in state.items())
    after = age_aware_aggregate(state, model.unit_ranks, [delta], np.array([1.0]))
    for name in state:
        forbidden = keep[name] == 0
        assert torch.equal(after[name][forbidden], state[name][forbidden]), name


def test_hand_computed_aggregation_and_self_floor():
    alpha = aggregation_weights([1,1], [1,3], [1,1], self_index=0, count_transform='raw')
    np.testing.assert_allclose(alpha, [0.25, 0.75])
    target = OrderedDict({'fc2.weight': torch.tensor([[10.], [20.]])})
    delta = OrderedDict({'fc2.weight': torch.tensor([[4.], [8.]])})
    zero = OrderedDict({'fc2.weight': torch.zeros(2,1)})
    result = age_aware_aggregate(target, {'fc2': np.array([2,1])}, [zero,delta], alpha)
    torch.testing.assert_close(result['fc2.weight'], torch.tensor([[10.],[26.]]))
    floor = aggregation_weights([1,1], [1,1000], [1,1], 0, 'raw', 0.25)
    np.testing.assert_allclose(floor, [0.25,0.75])


def test_p6_records_training_seed_from_checkpoint_not_evaluation_rng(monkeypatch, tmp_path):
    import run_denice_p6_eval as p6
    import json
    import sys
    monkeypatch.setattr(p6, 'evaluate_checkpoint', lambda *a, **k: {
        'training_seed': 42, 'eval_seed': 99,
        'metrics': {'accuracy': 0.5, 'f1_macro': 0.4}})
    monkeypatch.setattr(sys, 'argv', ['p6', '--checkpoint', 'fake.pt', '--data-dir', 'fake',
        '--output-dir', str(tmp_path), '--seed', '99', '--protocols', 'coverage_aware_local'])
    p6.main()
    saved = json.loads((tmp_path/'p6_evaluation_summary.json').read_text())
    assert saved['training_seed'] == 42
    assert saved['evaluation_seed'] == 99
    assert saved['training_seed_source'] == 'checkpoint_config'


def test_aggregation_sweep_uses_consistent_peer_snapshots(monkeypatch):
    import fed_learning.training.decentralized_denice_il as runner
    from types import SimpleNamespace
    class Toy(torch.nn.Module):
        LAYER_NAMES = ['fc2']
        def __init__(self, value):
            super().__init__()
            self.fc2 = torch.nn.Linear(1,2)
            self.fc2.weight.data.fill_(value)
            self.fc2.bias.data.fill_(value)
            self.unit_ranks = {'fc2': np.array([1,1])}
            self.adapters = torch.nn.ModuleDict()
        def get_neuron_ages_state(self): return deepcopy(self.unit_ranks)
        def set_neuron_ages_state(self, ages): self.unit_ranks = deepcopy(ages)
    monkeypatch.setattr(runner, 'dynamic_ap_cluster', lambda *a,**k: {
        'labels': np.array([0,0]), 'K_t': 1, 'valid': True, 'silhouette': 0.5,
        'edges': np.ones((2,2)), 'similarity': np.ones((2,2)), 'effective_similarity': np.ones((2,2))})
    caps = {cid: SimpleNamespace(label_set=[0,1], sample_count=1, reliability=1,
                                proto_vector=lambda: np.ones(2)) for cid in [7,19]}
    config = {'denice_age_merge_policy':'none'}
    models = {7:Toy(2), 19:Toy(6)}
    runner._aggregate_round(client_ids=[7,19], models=models, capsules=caps, config=config, device=torch.device('cpu'))
    for m in models.values():
        torch.testing.assert_close(m.fc2.weight, torch.full((2,1),4.))
    reverse = {7:Toy(2), 19:Toy(6)}
    runner._aggregate_round(client_ids=[19,7], models=reverse, capsules=caps, config=config, device=torch.device('cpu'))
    for cid in models:
        torch.testing.assert_close(models[cid].fc2.weight, reverse[cid].fc2.weight)
    runner._aggregate_round(client_ids=[7,19], models=models, capsules=caps, config=config, device=torch.device('cpu'))
    for m in models.values():
        torch.testing.assert_close(m.fc2.weight, torch.full((2,1),4.))


def test_mature_gru_features_need_structural_isolation_not_only_frozen_rows():
    from tools.audit_denice_invariants import recurrent_isolation_probe
    probe = recurrent_isolation_probe()
    assert probe['dense']['mature_gate_parameters_unchanged']
    assert probe['dense']['mature_feature_max_abs_change'] > 1e-3
    assert probe['isolated_control']['mature_gate_parameters_unchanged']
    assert probe['isolated_control']['mature_feature_max_abs_change'] < 1e-6


def test_p6_validator_accepts_current_local_policies_and_rejects_false_provenance():
    from summarize_denice_p6 import _validate_run, POLICIES
    metrics = {p: {'checkpoint_sha256': 'actual-bytes', 'config_sha256': 'cfg',
                   'training_seed': 42} for p in POLICIES}
    summary = {'training_seed': 42, 'training_seed_source': 'checkpoint_config',
               'summary': {'coverage_aware_local': metrics}}
    assert _validate_run(summary, ['coverage_aware_local']) == 'actual-bytes'
    summary['training_seed'] = 99
    with pytest.raises(ValueError, match='disagrees'):
        _validate_run(summary, ['coverage_aware_local'])
    del summary['training_seed_source']
    with pytest.raises(ValueError, match='provenance'):
        _validate_run(summary, ['coverage_aware_local'])


def test_two_task_local_lifecycle_ages_only_at_task_boundaries():
    from tools.audit_denice_invariants import two_task_client_probe
    records = two_task_client_probe()['records']
    assert records[0]['output_ages'] == [2, 2, 0, 0]
    assert records[1]['output_ages'] == [3, 3, 2, 2]
    assert all(np.isfinite(row['losses']).all() for row in records)


def test_late_join_bootstrap_preserves_source_connection_topology():
    from fed_learning.training.decentralized_denice_il import _bootstrap_denice_model
    model, _ = model_detector()
    model.clear_active_adapters()
    model.weight_masks['fc2'][0] = 0
    model.fc2.weight.data[0] = 2
    clone = _bootstrap_denice_model(model, {'input_shape': (39,1), 'num_classes': 4}, torch.device('cpu')).eval()
    assert torch.equal(model.weight_masks['fc2'], clone.weight_masks['fc2'])
    x = torch.randn(4,39,1)
    torch.testing.assert_close(model(x), clone(x), rtol=0, atol=0)
