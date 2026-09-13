"""Causal regressions for capacity, sketch calibration, and routed exclusion."""
import numpy as np
import pytest
import torch

from fed_learning.models.denice_model import DeNICEModel
from fed_learning.servers.nice_server import ContextDetector
from fed_learning.strategies.incremental.nice import increase_unit_ranks
from fed_learning.training.denice_eval import _denice_routed_logits_with_episodes
from fed_learning.training.checkpoint_state import snapshot_denice_state, restore_denice_state


def test_fixed_allocation_keeps_proportional_capacity_for_all_six_tasks():
    model = DeNICEModel((16, 1), 34)
    allocations = []
    for classes in [list(range(t, min(t + 6, 34))) for t in range(0, 34, 6)]:
        allocations.append(model.allocate_task_neurons(classes))
        increase_unit_ranks(model)
    for layer in model.unit_ranks:
        if layer == 'fc2':
            continue
        width = len(model.unit_ranks[layer])
        assert sum(a[layer] for a in allocations) == width
        assert allocations[-1][layer] >= int(np.floor(width * 4 / 34))


def test_fixed_allocation_aligns_late_clients_and_is_idempotent():
    early = DeNICEModel((16, 1), 34)
    late = DeNICEModel((16, 1), 34)
    early.allocate_task_neurons(list(range(6)))
    increase_unit_ranks(early)
    classes = [6, 7, 9, 11]
    early.allocate_task_neurons(classes)
    late.allocate_task_neurons(classes)
    for layer in early.unit_ranks:
        if layer != 'fc2':
            np.testing.assert_array_equal(early.unit_ranks[layer] == 1, late.unit_ranks[layer] == 1)
    assert all(n == 0 for n in late.allocate_task_neurons(classes).values())


def test_allocation_policy_roundtrip_and_legacy_continuation():
    source = DeNICEModel((16, 1), 34)
    state = snapshot_denice_state(source)
    restored = DeNICEModel((16, 1), 34)
    restore_denice_state(restored, None, state)
    assert restored.allocation_policy == 'class_blocks'
    del state['allocation_policy']
    restore_denice_state(restored, None, state)
    assert restored.allocation_policy == 'legacy_sequential'
    assert restored.allocate_task_neurons(list(range(6)))['conv1'] == 12


class SketchEncoder:
    """Two useful frozen features and one arbitrary reserve feature per layer."""
    def eval(self):
        return self

    def get_context_activations_per_sample(self, data):
        return {name: data for name in ('conv1', 'conv2', 'conv3', 'gru')}


def test_sketch_calibration_ignores_features_excluded_from_routing():
    encoder = SketchEncoder()
    detectors = []
    for reserve in [0., 1000.]:
        detector = ContextDetector(router_mode='binary_cosine')
        detector.stable_feature_mask = np.tile([True, True, False], 4)
        detector.episode_classes = {0: [0], 1: [1]}
        detector.push_activations(encoder, torch.tensor([[4., 0., reserve], [0., 0., reserve]]), 0)
        detector.push_activations(encoder, torch.tensor([[0., 4., reserve], [0., 0., reserve]]), 1)
        detectors.append(detector)
    assert detectors[0].binarize_thresholds == detectors[1].binarize_thresholds
    for detector in detectors:
        query = detector._binarize_per_sample(encoder, torch.tensor([[4., 0., 999.], [0., 4., 999.]]))
        np.testing.assert_array_equal(detector.predict_episodes_batch(query), [0, 1])


def test_d3_rejects_oracle_gain_when_predicted_routing_regresses(monkeypatch):
    from tools import analyze_denice_d3 as analyzer
    from sklearn.metrics import f1_score
    baseline = [0, 0, 0, 1, 1, 1, 1, 0]
    targets = [0, 0, 0, 0, 1, 1, 1, 1]
    bad = [1 - y for y in targets]
    manifest = {'seed': 3, 'variants': {name: {'name': name} for name in analyzer.REQUIRED}}
    def artifact(run, policy):
        pred = baseline if run['name'] == 'baseline' else (
            targets if policy.startswith('e3') else bad)
        trace = {'source_test_indices': list(range(8)), 'client_ids': [0] * 8,
                 'targets': targets, 'predictions': pred}
        return {'metrics': {'f1_macro': f1_score(targets, pred, average='macro'),
                'debug': {'prediction_trace': trace, 'per_class': {
                    str(c): {'accuracy': np.mean([p == y for p, y in zip(pred, targets) if y == c])}
                    for c in [0, 1]}}}}
    monkeypatch.setattr(analyzer, '_read', lambda path: manifest)
    monkeypatch.setattr(analyzer, '_old_classes', lambda data: {0})
    monkeypatch.setattr(analyzer, '_policy', artifact)
    report = analyzer.analyze('unused.json', bootstrap_replicates=40)
    assert report['selection_policy'] == 'e4_pred_hard'
    assert report['decision'] == 'KEEP_BASELINE'
    assert report['variants']['class_balanced_batches']['e3_f1_macro_delta'] > 0
    assert report['variants']['class_balanced_batches']['e4_f1_macro_delta'] < 0


@pytest.mark.parametrize('policy', ['backbone_nomask', 'pred_adapter_nomask', 'pred_hard'])
def test_excluded_classes_cannot_win_when_allowed_logits_are_below_minus_100(policy):
    model = DeNICEModel((16, 1), 4).eval()
    with torch.no_grad():
        model.fc2.weight.zero_()
        model.fc2.bias.copy_(torch.tensor([-200., -201., 1000., 1000.]))
    detector = ContextDetector(router_mode='binary_cosine')
    detector.episode_classes = {0: [0, 1]}
    logits, _ = _denice_routed_logits_with_episodes(
        model, torch.zeros(2, 16, 1), detector, [0, 1], 'cpu', inference_policy=policy)
    assert logits.argmax(1).tolist() == [0, 0]
    assert torch.isfinite(logits).all()
    assert torch.softmax(logits, dim=1)[:, 2:].sum() < 1e-20
