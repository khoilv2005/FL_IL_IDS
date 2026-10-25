"""Pure P1 aggregation/moment/transaction tests. No native training campaign."""
from dataclasses import replace
import unittest

import numpy as np
import torch

from test_denice_v2_p0 import fixture, event
from fed_learning.strategies.decentralized.denice_v2.aggregation import (
    aggregate_round, finish_worker, prepare_round, reference_hash, start_worker)
from fed_learning.strategies.decentralized.denice_v2.contracts import ClusterState, state_hash
from fed_learning.strategies.decentralized.denice_v2.normalization import BNMoment, model_hash
from fed_learning.strategies.decentralized.denice_v2.state import snapshot, restore


def canonical(*, mutable_bn=False, extra_bn=False):
    state = fixture(task=1, round=-1)
    p = snapshot(state)
    alg = p['algorithm_state']
    alg['unit_ranks']['fc2'] = np.array([2, 2, 1, 1], np.int32)
    alg['freeze_masks']['fc2'] = np.array([True, True, False, False])
    for suffix in ('weight', 'bias'):
        p['protected_parameters']['fc2.' + suffix][2:] = False
    p['model_state'].update({'conv1.weight': torch.ones(2, 2, 1), 'conv1.bias': torch.zeros(2)})
    alg['unit_ranks']['conv1'] = np.array([1, 2], np.int32)
    alg['weight_masks']['conv1'] = torch.ones(2, 2, 1)
    alg['bias_masks']['conv1'] = torch.ones(2)
    alg['freeze_masks']['conv1'] = np.array([False, True])
    p['protected_parameters']['conv1.weight'] = torch.tensor([False, True]).reshape(2, 1, 1).expand(2, 2, 1).clone()
    p['protected_parameters']['conv1.bias'] = torch.tensor([False, True])
    if mutable_bn:
        p['routing_dependencies'] = ('fc1.weight', 'fc1.bias')
    if extra_bn:
        p['model_state'].update({'conv2.weight': torch.ones(2, 2, 1), 'conv2.bias': torch.zeros(2),
            'bn2.running_mean': torch.zeros(2), 'bn2.running_var': torch.ones(2), 'bn2.num_batches_tracked': torch.tensor(0)})
        alg['unit_ranks']['conv2'] = np.array([1, 1], np.int32)
        alg['weight_masks']['conv2'] = torch.ones(2, 2, 1)
        alg['bias_masks']['conv2'] = torch.ones(2)
        alg['freeze_masks']['conv2'] = np.zeros(2, bool)
    p['bn_state'] = {k: v for k, v in p['model_state'].items() if k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
    return ClusterState.create(**{k: v for k, v in p.items() if k not in ('model_spec', 'routing_space_hash')})


def plan(state=None):
    return prepare_round(state or canonical(), current_classes=(2, 3), base_counts={0: {2: 2}, 1: {2: 6, 3: 2}}, base_role_hash='locked-BASE')


def contribution(p, cid, changes, classes=(2,), *, steps=1, masks=None):
    worker = start_worker(p, cid)
    for key, delta in changes.items():
        worker.state.model_state[key] += torch.as_tensor(delta, dtype=worker.state.model_state[key].dtype)
    return finish_worker(p, worker, event(cid, classes, task=1, round=0, successful_steps=steps), contribution_masks=masks)


def moment(stage, layer, p, cid, *, rows, observations):
    x = np.asarray(observations, dtype=np.float64)
    return BNMoment(cid, p.reference.task, p.target_round, layer, model_hash(stage), p.hash, rows,
        np.full(x.shape[1], len(x), dtype=np.int64), x.sum(0), (x * x).sum(0), p.reference.task)


class AggregationTests(unittest.TestCase):
    def test_hierarchical_reduction_matches_unequal_sample_flat_aggregation(self):
        p = plan()
        a = contribution(p, 0, {'fc2.bias': [0, 0, 2, 0]})
        b = contribution(p, 1, {'fc2.bias': [0, 0, 6, 10]}, (2, 3))
        flat = aggregate_round(p, [a, b])
        grouped = aggregate_round(p, [a, b], groups=((0,), (1,)))
        self.assertEqual(state_hash(flat.state.model_state), state_hash(grouped.state.model_state))
        self.assertEqual(float(grouped.state.model_state['fc2.bias'][2]), 5.)
        self.assertFalse(grouped.receipt['raw_training_rows_shared'])
        for groups in (((0,),), ((0,), (0, 1)), ((), (0, 1))):
            with self.assertRaisesRegex(ValueError, 'Hierarchy'):
                aggregate_round(p, [a, b], groups=groups)

    def test_shared_reference_clones_and_hashes(self):
        p = plan()
        workers = [start_worker(p, c) for c in (0, 1)]
        self.assertEqual(workers[0].state.fingerprint(), workers[1].state.fingerprint())
        self.assertEqual(workers[0].initial_reference_hash, p.reference_hash)
        workers[0].state.model_state['fc2.weight'][2, 0] += 1
        self.assertNotEqual(state_hash(workers[0].state.model_state), state_hash(workers[1].state.model_state))

    def test_fc2_class_specific_sample_weights_and_bias(self):
        p = plan()
        a = contribution(p, 0, {'fc2.weight': [[0, 0], [0, 0], [2, 2], [0, 0]], 'fc2.bias': [0, 0, 2, 0]})
        b = contribution(p, 1, {'fc2.weight': [[0, 0], [0, 0], [6, 6], [10, 10]], 'fc2.bias': [0, 0, 6, 10]}, (2, 3))
        result = aggregate_round(p, [a, b])
        expected = p.reference.model_state['fc2.weight'].clone()
        expected[2] += 5  # (2*2 + 6*6)/(2+6)
        expected[3] += 10  # No dilution by client 0.
        torch.testing.assert_close(result.state.model_state['fc2.weight'], expected, rtol=0, atol=0)
        torch.testing.assert_close(result.state.model_state['fc2.bias'], torch.tensor([0., 0., 5., 10.]))

    def test_backbone_sample_count_weights(self):
        p = plan()
        a = contribution(p, 0, {'conv1.weight': [[[2], [2]], [[0], [0]]]})
        b = contribution(p, 1, {'conv1.weight': [[[6], [6]], [[0], [0]]]}, (2, 3))
        result = aggregate_round(p, [a, b])
        self.assertAlmostEqual(float(result.state.model_state['conv1.weight'][0, 0, 0]), 6.2, places=5)  # 1+(2*2+8*6)/10
        self.assertEqual(float(result.state.model_state['conv1.weight'][1, 0, 0]), 1)

    def test_parameter_contributor_normalization(self):
        p = plan()
        masks = {k: v.copy() for k, v in p.parameter_permissions.items()}
        masks['conv1.weight'][0, 1, 0] = False
        a = contribution(p, 0, {'conv1.weight': [[[2], [2]], [[0], [0]]]})
        b = contribution(p, 1, {'conv1.weight': [[[6], [6]], [[0], [0]]]}, masks=masks)
        result = aggregate_round(p, [a, b])
        self.assertEqual(float(result.state.model_state['conv1.weight'][0, 1, 0]), 3)
        self.assertEqual(float(result.state.model_state['conv1.weight'][0, 0, 0]), 6)

    def test_zero_contributor_identity_preserves_weights_and_masks(self):
        p = plan()
        result = aggregate_round(p, [])
        self.assertEqual(state_hash(result.state.model_state), state_hash(p.reference.model_state))
        self.assertEqual(state_hash(result.state.algorithm_state), state_hash(p.reference.algorithm_state))
        self.assertEqual(result.state.round, 0)

    def test_unused_fc2_classes_do_not_upload_negative_only_updates(self):
        p = plan()
        packet = contribution(p, 0, {'fc2.bias': [0, 0, 1, 99]})
        self.assertFalse(packet.masks['fc2.bias'][3])
        result = aggregate_round(p, [packet])
        self.assertEqual(float(result.state.model_state['fc2.bias'][3]), 0)

    def test_eta_applies_delta_not_absolute_average(self):
        p = plan()
        a = contribution(p, 0, {'fc2.bias': [0, 0, 4, 0]})
        self.assertEqual(float(aggregate_round(p, [a], eta=.25).state.model_state['fc2.bias'][2]), 1)
        self.assertEqual(state_hash(aggregate_round(p, [a], eta=0).state.model_state), state_hash(p.reference.model_state))

    def test_order_and_leader_do_not_change_aggregation(self):
        original = canonical()
        other = restore(snapshot(original)); other.coordinator_id = 1
        p, q = plan(original), plan(other)
        self.assertEqual(reference_hash(original), reference_hash(other))
        self.assertEqual(p.hash, q.hash)
        packets = [contribution(p, c, {'fc2.bias': [0, 0, c + 1, 0]}) for c in (0, 1)]
        a, b = aggregate_round(p, packets), aggregate_round(q, list(reversed(packets)))
        self.assertEqual(state_hash(a.state.model_state), state_hash(b.state.model_state))
        self.assertEqual(state_hash(a.state.algorithm_state), state_hash(b.state.algorithm_state))
        self.assertFalse(a.receipt['leader_weight_bonus'])

    def test_mature_and_routing_mutation_rejected(self):
        p = plan()
        for name, index in [('conv1.weight', (1, 0, 0)), ('fc1.weight', (0, 0)), ('fc2.bias', 0)]:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Protected'):
                worker = start_worker(p, 0); worker.state.model_state[name][index] += 1
                finish_worker(p, worker, event(0, (2,), task=1))

    def test_topology_or_version_change_rejected(self):
        p = plan()
        for change in [lambda w: w.state.algorithm_state['unit_ranks']['fc2'].__setitem__(2, 2),
                       lambda w: w.state.algorithm_state['weight_masks']['conv1'].__setitem__((0, 0, 0), 0),
                       lambda w: w.state.versions.__setitem__('feature', 'different')]:
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'topology'):
                w = start_worker(p, 0); change(w)
                finish_worker(p, w, event(0, (2,), task=1))

    def test_stale_reference_packet_rejected_without_mutation(self):
        p = plan()
        packet = contribution(p, 0, {'fc2.bias': [0, 0, 1, 0]})
        before = p.reference.fingerprint()
        packet.initial_reference_hash = 'stale'
        with self.assertRaisesRegex(ValueError, 'reference'):
            aggregate_round(p, [packet])
        self.assertEqual(before, p.reference.fingerprint())

    def test_no_local_steps_no_ownership_or_weights(self):
        p = plan()
        packet = contribution(p, 0, {}, (), steps=0)
        self.assertEqual(state_hash(aggregate_round(p, [packet]).state.model_state), state_hash(p.reference.model_state))
        with self.assertRaisesRegex(ValueError, 'optimizer step'):
            contribution(p, 0, {'fc2.bias': [0, 0, 1, 0]}, (), steps=0)

    def test_broadcast_and_rollback(self):
        p = plan()
        packet = contribution(p, 0, {'fc2.bias': [0, 0, 1, 0]})
        before = p.reference.fingerprint()
        with self.assertRaisesRegex(ValueError, 'ROLLBACK'):
            aggregate_round(p, [packet], acceptance=lambda s: (False, 'controlled failure'))
        self.assertEqual(before, p.reference.fingerprint())
        result = aggregate_round(p, [packet])
        copies = result.broadcast()
        self.assertEqual(copies[0].fingerprint(), copies[1].fingerprint())
        copies[0].model_state['fc2.bias'][2] += 10
        self.assertEqual(copies[1].fingerprint(), result.state.fingerprint())

    def test_invalid_packet_masks_shapes_values_and_duplicates(self):
        p = plan()
        mutations = [lambda a: a.masks['fc2.bias'].__setitem__(0, True),
                     lambda a: a.deltas.__setitem__('fc2.bias', np.zeros(5)),
                     lambda a: a.deltas['fc2.bias'].__setitem__(2, np.nan),
                     lambda a: a.deltas['fc2.bias'].__setitem__(3, 10)]
        for change in mutations:
            with self.subTest(change=change), self.assertRaises(ValueError):
                packet = contribution(p, 0, {}); change(packet)
                aggregate_round(p, [packet])
        a = contribution(p, 0, {})
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            aggregate_round(p, [a, a])

    def test_unauthorized_roles_classes_incomplete_receipt(self):
        p = plan()
        w = start_worker(p, 0)
        receipts = [event(0, (0,), task=1), event(0, (2,), task=1, role='validation'),
                    event(0, (2,), task=1, data_task=0), event(0, (2,), task=1, completed=False)]
        for e in receipts:
            with self.subTest(event=e), self.assertRaises(ValueError):
                finish_worker(p, w, e)

    def test_reference_authority_tamper_rejected(self):
        p = plan()
        p.base_counts[0][2] = 1000
        with self.assertRaisesRegex(ValueError, 'authority'):
            start_worker(p, 0)

    def test_aggregation_does_not_age_neurons_per_member(self):
        p = plan()
        packets = [contribution(p, c, {}) for c in (0, 1)]
        after = aggregate_round(p, packets).state
        self.assertEqual(state_hash(after.algorithm_state), state_hash(p.reference.algorithm_state))
        self.assertEqual(state_hash(after.router_state), state_hash(p.reference.router_state))

    def test_gru_gate_masks_both_layers_and_recurrent_connection_masks(self):
        payload = snapshot(canonical())
        alg = payload['algorithm_state']
        alg['unit_ranks']['gru'] = np.array([1, 2], np.int32)
        alg['weight_masks']['gru'] = torch.ones(2)
        alg['bias_masks']['gru'] = torch.ones(2)
        alg['freeze_masks']['gru'] = np.array([False, True])
        alg['gru_connection_masks'] = {'weight_hh_l1': torch.ones(6, 2)}
        alg['gru_connection_masks']['weight_hh_l1'][0, 1] = 0
        for layer in (0, 1):
            for family in ('weight_ih', 'weight_hh', 'bias_ih', 'bias_hh'):
                name = f'gru.{family}_l{layer}'
                shape = (6, 2) if family.startswith('weight') else (6,)
                payload['model_state'][name] = torch.zeros(shape)
                rows = torch.tensor([False, True] * 3)
                payload['protected_parameters'][name] = rows if len(shape) == 1 else rows[:, None].expand(shape).clone()
        state = ClusterState.create(**{k: v for k, v in payload.items() if k not in ('model_spec', 'routing_space_hash')})
        p = plan(state)
        mask = p.parameter_permissions['gru.weight_hh_l1']
        self.assertTrue(mask[0, 0]); self.assertFalse(mask[0, 1]); self.assertFalse(mask[1, 0])
        w = start_worker(p, 0)
        w.state.model_state['gru.weight_hh_l1'][0, 0] = 2
        packet = finish_worker(p, w, event(0, (2,), task=1))
        result = aggregate_round(p, [packet])
        self.assertEqual(float(result.state.model_state['gru.weight_hh_l1'][0, 0]), 2)
        with self.assertRaisesRegex(ValueError, 'Protected'):
            w.state.model_state['gru.weight_ih_l0'][1, 0] = 1
            finish_worker(p, w, event(0, (2,), task=1))

    def test_invalid_eta_and_mask_expansion_rejected(self):
        p = plan()
        for eta in (-.1, 1.1, np.nan):
            with self.subTest(eta=eta), self.assertRaises(ValueError):
                aggregate_round(p, [], eta=eta)
        masks = {k: v.copy() for k, v in p.parameter_permissions.items()}
        masks['conv1.bias'][1] = True
        with self.assertRaisesRegex(ValueError, 'permissions'):
            contribution(p, 0, {}, masks=masks)

    def test_acceptance_cannot_mutate_shadow_candidate(self):
        p = plan()
        def acceptance(state):
            state.model_state['fc2.bias'][2] = 999
            return True, 'pass'
        with self.assertRaisesRegex(ValueError, 'mutated'):
            aggregate_round(p, [], acceptance=acceptance)

    def test_native_denice_tensor_state_aggregates_and_loads(self):
        from fed_learning.models.denice_model import DeNICEModel
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            model = DeNICEModel((16, 3), num_classes=4)
        model.unit_ranks['fc1'][0] = 1
        model.unit_ranks['fc2'][2:] = 1
        weights = model.state_dict()
        layers = ('conv1', 'conv2', 'conv3', 'gru')
        width = sum(len(model.unit_ranks[k]) for k in layers)
        reference = ClusterState.create(cluster_uid='native', task=1, round=-1, members=(0, 1), coordinator_id=0,
            versions=dict(architecture='legacy-cnn-gru', feature='adapter-free', class_mapping='four-classes', class_ids=(0, 1, 2, 3)),
            model_state=weights, algorithm_state=dict(unit_ranks=model.unit_ranks, weight_masks=model.weight_masks,
                bias_masks=model.bias_masks, freeze_masks=model.freeze_masks, adapter_registry=model.adapter_registry),
            router_state=dict(router_mode='binary_cosine', context_layers=layers, stable_feature_mask=np.ones(width, bool),
                binarize_thresholds=None, activation_memory={}, episode_classes={}),
            bn_state={k: v for k, v in weights.items() if k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))},
            protected_parameters={}, routing_dependencies=tuple(k for k in weights if not k.startswith(('fc1.', 'fc2.'))))
        p = plan(reference)
        w = start_worker(p, 0)
        w.state.model_state['fc1.weight'][0] += .1
        w.state.model_state['fc2.bias'][2] += 1
        packet = finish_worker(p, w, event(0, (2,), task=1))
        result = aggregate_round(p, [packet])
        model.load_state_dict(result.state.model_state, strict=True)
        self.assertEqual(state_hash(model.state_dict()), state_hash(result.state.model_state))
        for name in reference.routing_dependencies:
            self.assertEqual(state_hash(reference.model_state[name]), state_hash(result.state.model_state[name]))


class PartitionDryRunTests(unittest.TestCase):
    def test_real_format_class_gaps_and_no_future_data_use(self):
        from tools.dry_run_denice_v2_partition import inspect_manifest
        manifest = dict(completed=True, content_disjoint=True, eligible_base_experts={'0': [0, 1], '1': [1], '2': [0]},
            clients={'0': {'role_class_counts': {'base': {'0': 2, '2': 99}}},
                     '1': {'role_class_counts': {'base': {'0': 3, '1': 4}}}})
        a = inspect_manifest(manifest, current_classes=(0, 1), manifest_sha256='locked', threshold=.5)
        self.assertEqual(a['K'], 1)
        self.assertEqual(a['peer_fillable_data_gap_pairs'], 1)
        manifest['clients']['0']['role_class_counts']['base']['2'] = 1000000
        b = inspect_manifest(manifest, current_classes=(0, 1), manifest_sha256='locked', threshold=.5)
        self.assertEqual(a, b)
        self.assertEqual(inspect_manifest(manifest, current_classes=(0, 1), manifest_sha256='locked', threshold=.8)['K'], 2)

    def test_incomplete_manifest_rejected(self):
        from tools.dry_run_denice_v2_partition import inspect_manifest
        with self.assertRaises(ValueError):
            inspect_manifest({'completed': False}, current_classes=(0,), manifest_sha256='x')


class NormalizationTests(unittest.TestCase):
    def test_bn_global_moments_and_mature_channel_protection(self):
        p = plan(canonical(mutable_bn=True))
        old = p.reference.fingerprint()
        def provider(stage, layer):
            return [moment(stage, layer, p, 0, rows=2, observations=[[1, 100], [3, 200]]),
                    moment(stage, layer, p, 1, rows=2, observations=[[5, 300], [7, 400]])]
        result = aggregate_round(p, [], bn_provider=provider)
        torch.testing.assert_close(result.state.model_state['bn1.running_mean'], torch.tensor([4., 0.]))
        torch.testing.assert_close(result.state.model_state['bn1.running_var'], torch.tensor([20 / 3, 1.]))
        self.assertEqual(old, p.reference.fingerprint())
        self.assertEqual(state_hash(result.state.bn_state['bn1.num_batches_tracked']), state_hash(p.reference.bn_state['bn1.num_batches_tracked']))

    def test_mutable_bn_without_moments_rejected(self):
        p = plan(canonical(mutable_bn=True))
        with self.assertRaisesRegex(ValueError, 'moments required'):
            aggregate_round(p, [])

    def test_local_bn_not_averaged(self):
        p = plan(canonical(mutable_bn=True))
        w = start_worker(p, 0)
        w.state.model_state['bn1.running_mean'][0] = 999
        packet = finish_worker(p, w, event(0, (2,), task=1))
        def provider(stage, layer):
            return [moment(stage, layer, p, 0, rows=2, observations=[[1, 0], [3, 0]])]
        result = aggregate_round(p, [packet], bn_provider=provider)
        self.assertEqual(float(result.state.model_state['bn1.running_mean'][0]), 2)

    def test_mature_or_routing_bn_local_mutation_rejected(self):
        for mutable, index in [(True, 1), (False, 0)]:
            with self.subTest(mutable=mutable), self.assertRaisesRegex(ValueError, 'Protected'):
                p = plan(canonical(mutable_bn=mutable)); w = start_worker(p, 0)
                w.state.model_state['bn1.running_mean'][index] = 999
                finish_worker(p, w, event(0, (2,), task=1))

    def test_sequential_bn_requires_new_post_upstream_hash(self):
        p = plan(canonical(mutable_bn=True, extra_bn=True))
        seen = []
        def provider(stage, layer):
            seen.append((layer, model_hash(stage)))
            return [moment(stage, layer, p, 0, rows=2, observations=[[1, 0], [3, 2]])]
        aggregate_round(p, [], bn_provider=provider)
        self.assertEqual([x[0] for x in seen], ['bn1', 'bn2'])
        self.assertNotEqual(seen[0][1], seen[1][1])
        stale = moment(p.reference, 'bn2', p, 0, rows=2, observations=[[1, 0], [3, 2]])
        with self.assertRaisesRegex(ValueError, 'Stale'):
            aggregate_round(p, [], bn_provider=lambda s, l: provider(s, l) if l == 'bn1' else [stale])

    def test_bn_current_role_version_duplicate_and_counts_gates(self):
        p = plan(canonical(mutable_bn=True))
        good = moment(p.reference, 'bn1', p, 0, rows=2, observations=[[1, 0], [3, 2]])
        variants = [replace(good, role='calibration'), replace(good, data_task=0), replace(good, plan_hash='bad'),
                    replace(good, sample_rows=100), replace(good, squared_sums=np.zeros(2)), replace(good, counts=np.zeros(2, np.int64))]
        for packet in variants:
            with self.subTest(packet=packet), self.assertRaises(ValueError):
                aggregate_round(p, [], bn_provider=lambda s, l: [packet])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            aggregate_round(p, [], bn_provider=lambda s, l: [good, good])

    def test_protected_bn_never_calls_provider(self):
        p = plan()
        def forbidden(stage, layer):
            raise AssertionError('Protected BN must retain reference buffers')
        result = aggregate_round(p, [], bn_provider=forbidden)
        self.assertEqual(state_hash(result.state.bn_state), state_hash(p.reference.bn_state))


if __name__ == '__main__':
    unittest.main()
