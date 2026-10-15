import numpy as np
import pytest

from fed_learning.training.denice_run_audit import method_manifest, optimizer_health
from tools.run_denice_recovery import build_config
from fed_learning.servers.nice_server import ContextDetector
from fed_learning.training.checkpoint_state import snapshot_context_detector, restore_context_detector


def test_method_guard_detects_wrong_launch_and_ambiguous_flags():
    with pytest.raises(ValueError, match='Expected method'):
        method_manifest({'denice_expected_method': 'row_overlay'})
    with pytest.raises(ValueError, match='only one'):
        method_manifest({'denice_appliance_enabled': True, 'appliance_enabled': True})
    assert method_manifest({'denice_appliance_enabled': True})['effective_method'] == 'row_overlay'


def test_health_does_not_treat_partial_skips_as_no_training():
    assert not optimizer_health({'optimizer_steps': 2, 'skipped_optimizer_steps': 1}, fail_on_zero=True)['zero_update']
    with pytest.raises(FloatingPointError):
        optimizer_health({'optimizer_steps': 0, 'skipped_optimizer_steps': 1}, fail_on_zero=True)
    assert not optimizer_health({})['zero_update']


def test_ablation_pairs_and_guard_are_consistent():
    arms = {arm: build_config(arm, 'data', 'out') for arm in
            ('amp_control', 'fp32', 'balanced_router', 'cluster_control', 'row_overlay')}
    for config in arms.values():
        method_manifest(config)
        assert config['denice_eval_require_full_coverage'] is False
    def differences(a, b):
        return {key for key in a.keys() | b.keys() if a.get(key) != b.get(key)}
    assert differences(arms['fp32'], arms['balanced_router']) == {'denice_router_mode'}
    assert differences(arms['cluster_control'], arms['row_overlay']) == {'denice_appliance_enabled', 'denice_expected_method'}


def test_balanced_router_handles_unequal_memory_and_checkpoint():
    detector = ContextDetector(router_mode='multiclass_balanced')
    detector.activation_memory = {0: np.tile([1., 0.], (100, 1)),
                                  1: np.tile([0., 1.], (4, 1))}
    detector.episode_classes = {0: [0], 1: [1]}
    detector.train_models(1)
    queries = np.array([[1., 0.], [0., 1.]])
    assert detector.predict_episodes_batch(queries).tolist() == [0, 1]
    restored = ContextDetector()
    restore_context_detector(restored, snapshot_context_detector(detector))
    assert restored.predict_episodes_batch(queries).tolist() == [0, 1]
