"""Real native optimizer/BN smoke fixtures; no TEST or historical raw data."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fed_learning.models.denice_model import DeNICEModel
from fed_learning.clients.denice_client import DeNICEClient
from fed_learning.strategies.incremental.denice import DeNICETrainer
from fed_learning.strategies.decentralized.denice_v2.contracts import state_hash
from fed_learning.strategies.decentralized.denice_v2.native import (
    CurrentBase, NativeBNCollector, allocate_cluster, capture_native, load_native,
    prepare_native_round, run_native_round, train_native_worker)
from fed_learning.strategies.decentralized.denice_v2.normalization import protected_bn_channels


def fixture(mature=True):
    torch.manual_seed(42)
    model = DeNICEModel((16, 3), 4)
    if mature:
        for layer in ('conv1', 'fc1', 'fc2'):
            model.unit_ranks[layer][-1] = 2
    router = dict(router_mode='binary_cosine', context_layers=('conv1', 'conv2', 'conv3', 'gru'),
        stable_feature_mask=np.zeros(548, bool), binarize_thresholds=None, activation_memory={}, episode_classes={})
    state = capture_native(model, members=(0, 1), coordinator_id=0, cluster_uid='native-test',
                           router_state=router, routing_dependencies=('gru.weight_hh_l1',))
    bases = {}
    for cid, labels in [(0, [0, 1] * 4), (1, [1, 2] * 4)]:
        bases[cid] = CurrentBase(cid, 0, torch.randn(8, 16, 3), torch.tensor(labels),
                                 np.arange(8, dtype=np.int64), 'synthetic-current-BASE-fixture')
    return state, bases


class NativeP1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.state, cls.bases = fixture()
        cls.before = cls.state.fingerprint()
        cls.result, cls.plan, cls.packets, cls.receipt = run_native_round(cls.state, cls.bases, (0, 1, 2),
            batch_size=3, epochs=1, optimizer_family='sgd', weight_decay=.1, momentum=.9)

    def test_shared_allocation_union_and_optimizer_changes(self):
        ranks = self.plan.reference.algorithm_state['unit_ranks']['fc2']
        np.testing.assert_array_equal(ranks, [1, 1, 1, 2])
        for worker in self.receipt['workers']:
            self.assertEqual(worker['steps'], 3)
            self.assertGreater(worker['changed_coordinates']['fc1.weight'], 0)
            self.assertGreater(worker['changed_coordinates']['conv2.weight'], 0)
            self.assertGreater(worker['changed_coordinates']['gru.weight_ih_l0'], 0)
            self.assertEqual(worker['initial_reference_hash'], self.plan.reference_hash)

    def test_unique_class_row_only_contributor(self):
        self.assertEqual(self.receipt['aggregation']['fc2_contributors'][2], (1,))
        delta = self.packets[1].deltas['fc2.weight'][2]
        self.assertTrue(np.any(delta))
        expected = self.plan.reference.model_state['fc2.weight'][2].numpy().astype(np.float64) + delta
        np.testing.assert_array_equal(self.result.state.model_state['fc2.weight'][2].numpy(), expected.astype(np.float32))
        self.assertFalse(self.packets[0].masks['fc2.weight'][2].any())

    def test_mature_and_routing_weights_byte_exact_with_decay_momentum(self):
        for key, mask in self.state.protected_parameters.items():
            torch.testing.assert_close(self.state.model_state[key][mask], self.result.state.model_state[key][mask], rtol=0, atol=0)
        self.assertTrue(torch.equal(self.state.model_state['gru.weight_hh_l1'], self.result.state.model_state['gru.weight_hh_l1']))
        self.assertEqual(self.state.fingerprint(), self.before)

    def test_protected_bn_and_sequential_collector_chain(self):
        self.assertEqual([r['layer'] for r in self.receipt['aggregation']['bn']], ['bn1', 'bn2', 'bn3'])
        mask = torch.tensor(protected_bn_channels(self.state, 'bn1'))
        for suffix in ('running_mean', 'running_var', 'weight', 'bias'):
            name = 'bn1.' + suffix
            self.assertTrue(torch.equal(self.state.model_state[name][mask], self.result.state.model_state[name][mask]))
        for previous, following in zip(self.receipt['aggregation']['bn'], self.receipt['aggregation']['bn'][1:]):
            self.assertEqual(previous['output_model_hash'], following['input_model_hash'])

    def test_real_bn_moments_independent_conv_math(self):
        collector = NativeBNCollector(self.plan, self.bases, batch_size=3)
        packets = collector(self.result.state, 'bn1')
        model = load_native(self.result.state).eval()
        for packet in packets:
            base = self.bases[packet.client_id]
            with torch.no_grad():
                z = torch.nn.functional.conv1d(base.x.permute(0, 2, 1), model.conv1.weight * model.weight_masks['conv1'],
                    model.conv1.bias * model.bias_masks['conv1'], padding=1).double().movedim(1, 0).reshape(64, -1)
            np.testing.assert_allclose(packet.sums, z.sum(1).numpy(), atol=1e-5, rtol=1e-6)
            np.testing.assert_array_equal(packet.counts, np.full(64, len(base.y)*16))

    def test_broadcast_state_and_logits(self):
        self.assertTrue(self.receipt['broadcast_state_identical'])
        self.assertTrue(self.receipt['broadcast_logits_identical'])
        self.assertFalse(self.receipt['router_fitted'])

    def test_locked_native_loop_does_not_call_local_selection(self):
        with (patch('fed_learning.clients.nice_client.select_learner_units', side_effect=AssertionError('local selection')),
             patch('fed_learning.clients.nice_client.drop_young_to_learner', side_effect=AssertionError('local pruning')),
             patch('fed_learning.clients.nice_client.grow_all_to_young', side_effect=AssertionError('local regrowth'))):
            train_native_worker(self.plan, self.bases[0], batch_size=8)

    def test_v1_default_still_calls_local_preparation(self):
        model = load_native(self.plan.reference)
        b = self.bases[0]
        client = DeNICEClient(0, b.x, b.y)
        client.setup_for_gpu(model, 'cpu')
        with (patch('fed_learning.clients.nice_client.select_learner_units') as select,
             patch('fed_learning.clients.nice_client.drop_young_to_learner') as drop,
             patch('fed_learning.clients.nice_client.grow_all_to_young') as grow):
            client.train(DeNICETrainer(max_phases=1, phase_epochs=1), epochs=1, batch_size=8, lr=.001)
            select.assert_called_once()
            drop.assert_called_once()
            grow.assert_called_once()

    def test_singleton_minibatch_tail(self):
        # 8 = 7 + 1: merge the tail without dropping any sample.
        packet, receipt = train_native_worker(self.plan, self.bases[0], batch_size=7)
        self.assertEqual(receipt['steps'], 1)
        self.assertEqual(sum(dict(packet.event.successful_class_counts).values()), 8)

    def test_allocation_preserves_parameters_maturity_and_router(self):
        allocated, _ = allocate_cluster(self.state, (0, 1, 2))
        self.assertEqual(state_hash(allocated.model_state), state_hash(self.state.model_state))
        self.assertEqual(state_hash(allocated.router_state), state_hash(self.state.router_state))
        self.assertEqual(allocated.algorithm_state['unit_ranks']['fc1'][-1], 2)

    def test_historical_role_rejected(self):
        state, bases = fixture()
        base = bases[0]
        bases[0] = CurrentBase(0, 1, base.x, base.y, base.row_ids, base.source_hash)
        with self.assertRaisesRegex(ValueError, 'Historical'):
            prepare_native_round(state, bases, (0, 1, 2))

    def test_mutated_base_content_rejected(self):
        state, bases = fixture()
        plan, _ = prepare_native_round(state, bases, (0, 1, 2))
        bases[0].x[0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'content differs'):
            NativeBNCollector(plan, bases)

    def test_base_mutation_after_collector_creation_rejected(self):
        state, bases = fixture()
        plan, _ = prepare_native_round(state, bases, (0, 1, 2))
        collector = NativeBNCollector(plan, bases)
        bases[0].x[0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'mutated during round'):
            collector(plan.reference, 'bn1')

    def test_base_shape_mismatch_rejected(self):
        state, bases = fixture()
        b = bases[0]
        bases[0] = CurrentBase(0, 0, b.x[:, :, :1], b.y, b.row_ids, b.source_hash)
        with self.assertRaisesRegex(ValueError, 'input shape'):
            prepare_native_round(state, bases, (0, 1, 2))

    def test_no_trainable_capacity_rejected(self):
        state, bases = fixture()
        model = load_native(state)
        for ranks in model.unit_ranks.values():
            ranks[:] = 2
        state = capture_native(model, members=(0, 1), coordinator_id=0, cluster_uid='no-capacity',
            router_state=state.router_state, routing_dependencies=state.routing_dependencies)
        plan, _ = prepare_native_round(state, bases, (0, 1, 2))
        with self.assertRaisesRegex(ValueError, 'NO_TRAINABLE_CAPACITY'):
            train_native_worker(plan, bases[0])


if __name__ == '__main__':
    unittest.main()
