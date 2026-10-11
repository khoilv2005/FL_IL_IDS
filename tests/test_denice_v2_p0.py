"""P0 structural checks; no training or claim of router functional stability."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from fed_learning.strategies.decentralized.denice_v2.contracts import ClusterState, decode, encode, state_hash
from fed_learning.strategies.decentralized.denice_v2.election import elect_leader
from fed_learning.strategies.decentralized.denice_v2.knowledge import ClassKnowledgeLedger, LocalLearningEvent
from fed_learning.strategies.decentralized.denice_v2.partition import (
    PartitionBook, SimilaritySnapshot, bootstrap_similarity, partition_clients)
from fed_learning.strategies.decentralized.denice_v2.state import (
    ClusterRegistry, P0Checkpoint, broadcast, compatibility, failover, handoff,
    initialize_clusters, load_checkpoint, merge_gate, restore, save_checkpoint, snapshot, split, transition_clusters)


TASK_CLASSES = {0: (0, 1), 1: (2, 3)}


def ledger():
    return ClassKnowledgeLedger(TASK_CLASSES)


def event(cid=0, classes=(0,), task=0, round=0, **kwargs):
    values = dict(event_id=f'{cid}/{task}/{round}', client_id=cid, task=task, round=round,
                  successful_steps=1, successful_class_counts=tuple((c, 2) for c in classes), data_task=task)
    values.update(kwargs)
    return LocalLearningEvent(**values)


def similarities(ids=(0, 1, 2), scores=None, task=0):
    if scores is None:
        scores = np.ones((len(ids), len(ids)))
    return SimilaritySnapshot(tuple(ids), tuple(map(tuple, np.asarray(scores).tolist())),
        task, task, 'current_base_context_capsules', 'reference-v1')


def fixture(uid='A', members=(0, 1), task=0, round=0):
    model = {'fc1.weight': torch.tensor([[1., 2.], [3., 4.]]), 'fc1.bias': torch.zeros(2),
        'fc2.weight': torch.arange(8, dtype=torch.float32).reshape(4, 2), 'fc2.bias': torch.zeros(4),
        'bn1.running_mean': torch.zeros(2), 'bn1.running_var': torch.ones(2),
        'bn1.num_batches_tracked': torch.tensor(3)}
    ranks = dict(fc1=np.array([2, 2], np.int32), fc2=np.array([2, 2, 0, 0], np.int32))
    algorithms = dict(unit_ranks=ranks,
        weight_masks={k: torch.ones_like(model[k + '.weight']) for k in ranks},
        bias_masks={k: torch.ones(len(v)) for k, v in ranks.items()},
        freeze_masks={k: v >= 2 for k, v in ranks.items()},
        allocation_state={'policy': 'fixture'}, adapter_state={'registry': {}, 'active': {}},
        local_optimizer_scaler_policy='reset_each_round')
    protected = {k: torch.ones_like(v, dtype=torch.bool) for k, v in model.items()
                 if k.startswith('fc1.') or k.startswith('fc2.')}
    router = dict(router_mode='binary_cosine', context_layers=('fc1',), stable_feature_mask=np.ones(2, bool),
        binarize_thresholds={'fc1': 0.1}, activation_memory={0: np.array([[1, 0], [0, 1]], np.uint8)},
        episode_classes={0: (0, 1)})
    return ClusterState.create(cluster_uid=uid, task=task, round=round, members=members, coordinator_id=min(members),
        versions=dict(architecture='fixture-v1', feature='route-v1', class_mapping='classes-v1', class_ids=(0, 1, 2, 3)),
        model_state=model, algorithm_state=algorithms, router_state=router,
        bn_state={k: v for k, v in model.items() if k.startswith('bn1.')}, protected_parameters=protected,
        routing_dependencies=('fc1.weight', 'fc1.bias', 'bn1.running_mean', 'bn1.running_var'),
        training_rng_state={'torch_cpu': torch.get_rng_state(), 'numpy_generator': np.random.default_rng(42).bit_generator.state})


class PartitionTests(unittest.TestCase):
    def test_singleton(self):
        result = partition_clients(similarities((7,)))
        self.assertEqual(result.clusters, ((7,),))
        self.assertEqual(result.K, 1)

    def test_k_one_is_output(self):
        self.assertEqual(partition_clients(similarities()).clusters, ((0, 1, 2),))

    def test_no_edges_all_singletons(self):
        self.assertEqual(partition_clients(similarities(scores=np.eye(3))).K, 3)

    def test_complete_linkage_does_not_merge_transitive_chain(self):
        values = [[1, .9, .1], [.9, 1, .9], [.1, .9, 1]]
        self.assertEqual(partition_clients(similarities(scores=values)).clusters, ((0, 1), (2,)))

    def test_deterministic_replay_and_shuffled_input(self):
        values = np.array([[1, .9, .1], [.9, 1, .9], [.1, .9, 1]])
        baseline = partition_clients(similarities(scores=values))
        order = [2, 0, 1]
        changed = partition_clients(similarities(tuple(order), values[np.ix_(order, order)]))
        self.assertEqual(baseline, changed)
        self.assertEqual(baseline, partition_clients(similarities(scores=values)))

    def test_threshold_inclusive(self):
        self.assertEqual(partition_clients(similarities((0, 1), [[1, .8], [.8, 1]])).K, 1)

    def test_invalid_matrix(self):
        cases = [similarities((), []), similarities((1, 1)),
            similarities((0, 1), [[1, .8], [.9, 1]]), similarities(scores=np.full((3, 3), np.nan)),
            similarities(scores=np.ones((2, 2))), replace(similarities(), data_task=1),
            replace(similarities(), source='test'), replace(similarities(), reference_hash='')]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                partition_clients(case)

    def test_invalid_threshold(self):
        for value in [np.nan, 1.1, -1.1]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                partition_clients(similarities(), threshold=value)

    def test_task0_bootstrap_uses_current_metadata(self):
        current = {3: {0: 10}, 1: {0: 2}, 2: {1: 4}}
        result = partition_clients(bootstrap_similarity(current, data_task=0, task_classes=(0, 1), reference_hash='init'))
        self.assertEqual(result.clusters, ((1, 3), (2,)))
        self.assertEqual(ledger().learned_classes(3, task=0), ())

    def test_bootstrap_rejects_future_or_empty_support(self):
        for counts, task in [({0: {2: 2}}, 0), ({0: {0: 2}}, 1), ({0: {0: 0}}, 0), ({}, 0)]:
            with self.subTest(counts=counts, task=task), self.assertRaises(ValueError):
                bootstrap_similarity(counts, data_task=task, task_classes=(0, 1), reference_hash='init')

    def test_partition_is_locked_and_restorable(self):
        book = PartitionBook()
        partition = partition_clients(similarities())
        book.freeze(partition, (0, 1, 2))
        book.freeze(partition, (0, 1, 2))
        with self.assertRaises(ValueError):
            book.freeze(partition_clients(similarities(scores=np.eye(3))), (0, 1, 2))
        self.assertEqual(book.snapshot(), PartitionBook.restore(book.snapshot()).snapshot())


class KnowledgeElectionTests(unittest.TestCase):
    def test_locally_trained_classes_and_matrix(self):
        book = ledger()
        book.record(event(1, (0, 1)), at_task=0, at_round=0)
        ids, matrix = book.matrix([1, 0], task=1)
        self.assertEqual(ids, (0, 1))
        np.testing.assert_array_equal(matrix, [[0, 0, 0, 0], [1, 1, 0, 0]])

    def test_skipped_or_incomplete_steps_do_not_establish_ownership(self):
        book = ledger()
        book.record(event(0, (), successful_steps=0), at_task=0, at_round=0)
        book.record(event(1, (0,), completed=False), at_task=0, at_round=0)
        self.assertEqual(book.learned_classes(0, task=1), ())
        self.assertEqual(book.learned_classes(1, task=1), ())

    def test_broadcast_and_historical_replay_are_not_ownership(self):
        book = ledger()
        for e in [event(role='broadcast'), event(data_task=1), event(successful_steps=0), event(classes=(2,))]:
            with self.subTest(event=e), self.assertRaises(ValueError):
                book.record(e, at_task=0, at_round=0)
        with self.assertRaises(TypeError):
            book.record({'availability': [0, 1]}, at_task=0, at_round=0)

    def test_future_evidence_is_excluded_at_decision_time(self):
        book = ledger()
        with self.assertRaises(ValueError):
            book.record(event(task=1, classes=(2,)), at_task=0, at_round=19)
        book.record(event(1, classes=(2,), task=1), at_task=1, at_round=0)
        self.assertEqual(book.learned_classes(1, task=1), ())
        self.assertEqual(elect_leader((0, 1), book, task=1).leader, 0)
        self.assertEqual(book.snapshot(task=0, round=19)['events'], [])

    def test_event_replay_idempotent_and_collision_rejected(self):
        book = ledger()
        e = event()
        book.record(e, at_task=0, at_round=0)
        book.record(e, at_task=0, at_round=0)
        self.assertEqual(len(book.events), 1)
        with self.assertRaises(ValueError):
            book.record(replace(e, successful_steps=2), at_task=0, at_round=0)

    def test_invalid_training_counts(self):
        for e in [event(successful_class_counts=((0, 1), (0, 2))), event(successful_steps=1.5),
                  event(successful_class_counts=((0, .5),))]:
            with self.subTest(event=e), self.assertRaises(ValueError):
                ledger().record(e, at_task=0, at_round=0)

    def test_ledger_restore_preserves_provenance(self):
        book = ledger()
        book.record(event(), at_task=0, at_round=0)
        recovered = ClassKnowledgeLedger.restore(decode(encode(book.snapshot(task=0, round=0))))
        self.assertEqual(book.events, recovered.events)

    def test_election_unique_local_class_count_and_tie(self):
        book = ledger()
        self.assertEqual(elect_leader((3, 2, 1), book, task=0).leader, 1)
        book.record(event(2, (0, 1)), at_task=0, at_round=0)
        book.record(event(3, (0,)), at_task=0, at_round=0)
        result = elect_leader((3, 2, 1), book, task=1)
        self.assertEqual(result.leader, 2)
        self.assertEqual(result.scores, ((1, 0), (2, 2), (3, 1)))

    def test_incumbent_retained_and_failover(self):
        book = ledger()
        book.record(event(1, (0, 1)), at_task=0, at_round=0)
        self.assertEqual(elect_leader((0, 1), book, task=1, incumbent=0).leader, 0)
        self.assertEqual(elect_leader((0, 1), book, task=1, incumbent=0, unavailable=(0,)).leader, 1)
        with self.assertRaises(ValueError):
            elect_leader((0, 1), book, task=1, unavailable=(0, 1))


class ContractStateTests(unittest.TestCase):
    def test_snapshot_restore_and_broadcast_have_no_alias(self):
        original = fixture()
        copied = restore(snapshot(original))
        self.assertEqual(original.fingerprint(), copied.fingerprint())
        clones = broadcast(original)
        self.assertEqual(clones[0].fingerprint(), clones[1].fingerprint())
        clones[0].model_state['fc2.weight'][2, 0] += 10
        clones[0].router_state['activation_memory'][0][0, 0] = 0
        self.assertEqual(original.fingerprint(), clones[1].fingerprint())
        self.assertNotEqual(original.fingerprint(), clones[0].fingerprint())
        self.assertEqual(ledger().learned_classes(1, task=1), ())

    def test_lossless_codec_preserves_dtypes_shapes_and_rng(self):
        values = dict(scalar=torch.tensor(4), empty=torch.empty(0), bf16=torch.tensor([1., 2.], dtype=torch.bfloat16),
            array=np.array(3, np.int32), endian=np.array([1, 2], dtype='>i4'), tuple=(2, [False, None]),
            rng=np.random.default_rng(42).bit_generator.state, integer_key={0: torch.get_rng_state()})
        recovered = decode(encode(values))
        self.assertEqual(state_hash(values), state_hash(recovered))
        self.assertEqual(recovered['array'].shape, ())
        self.assertEqual(recovered['bf16'].dtype, torch.bfloat16)

    def test_missing_or_malformed_versions_model_masks(self):
        operations = [lambda s: s.versions.pop('feature'),
            lambda s: s.model_state.__setitem__('fc2.weight', torch.zeros(5, 2)),
            lambda s: s.algorithm_state['weight_masks'].__setitem__('fc1', torch.full((2, 2), .5)),
            lambda s: s.algorithm_state['freeze_masks'].__setitem__('fc1', np.zeros(2, bool)),
            lambda s: s.protected_parameters.pop('fc1.weight'),
            lambda s: s.model_state['fc2.bias'].__setitem__(0, float('nan'))]
        for operation in operations:
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                state = fixture()
                operation(state)
                state.validate()

    def test_bn_state_must_match_model_buffers(self):
        state = fixture()
        state.bn_state['bn1.running_mean'][0] = 1
        with self.assertRaises(ValueError):
            state.validate()

    def test_router_rejects_wrong_width_future_history_and_raw_data(self):
        operations = [lambda s: s.router_state.__setitem__('stable_feature_mask', np.ones(3, bool)),
            lambda s: s.router_state['episode_classes'].__setitem__(1, (2,)),
            lambda s: s.router_state.__setitem__('reference_input_memory', {0: np.ones((2, 39))}),
            lambda s: s.router_state.__setitem__('context_layers', ('fc1', 'fc1'))]
        for operation in operations:
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                state = fixture()
                operation(state)
                state.validate()

    def test_router_hash_binds_dependencies_thresholds_and_mask(self):
        operations = [lambda s: s.model_state['fc1.weight'].__setitem__((0, 0), 8),
            lambda s: s.router_state['binarize_thresholds'].__setitem__('fc1', .4)]
        for operation in operations:
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                state = fixture()
                operation(state)
                state.validate()

    def test_fc2_changes_do_not_change_routing_space_hash(self):
        state = fixture()
        before, routing = state.fingerprint(), state.routing_space_hash
        state.model_state['fc2.weight'][2, 0] += 1
        state.validate()
        self.assertEqual(routing, state.routing_fingerprint())
        self.assertNotEqual(before, state.fingerprint())

    def test_split_preserves_history_protection_and_ownership(self):
        parent = fixture()
        parent.ownership_metadata = {0: {'local_classes': (0,)}}
        before = parent.fingerprint()
        children = split(parent, {'A/0': (0,), 'A/1': (1,)}, task=1, ledger=ledger())
        for child in children:
            self.assertEqual(child.parent_lineage, ('A',))
            self.assertEqual(child.task, 1)
            self.assertEqual(child.round, -1)
            self.assertEqual(state_hash(child.router_state), state_hash(parent.router_state))
            self.assertEqual(state_hash(child.protected_parameters), state_hash(parent.protected_parameters))
            self.assertEqual(child.routing_space_hash, parent.routing_space_hash)
        self.assertEqual(before, parent.fingerprint())
        with self.assertRaises(ValueError):
            split(parent, {'a': (0,), 'b': (0, 1)}, task=1, ledger=ledger())

    def test_compatible_merge_copies_history_once(self):
        left, right = fixture('A', (0,)), fixture('B', (1,))
        self.assertTrue(compatibility(left, right)[0])
        merged = handoff([left, right], members=(0, 1), cluster_uid='AB', task=1, ledger=ledger())
        self.assertEqual(merged.parent_lineage, ('A', 'B'))
        self.assertEqual(merged.router_state['activation_memory'][0].shape, (2, 2))
        self.assertEqual(state_hash(left.model_state), state_hash(merged.model_state))

    def test_rejected_merge_leaves_parent_state_untouched(self):
        left, right = fixture('A', (0,)), fixture('B', (1,))
        right.model_state['fc2.weight'][2, 0] += 2
        before = (left.fingerprint(), right.fingerprint())
        with self.assertRaisesRegex(ValueError, 'MODEL_STATE_MISMATCH'):
            handoff([left, right], members=(0, 1), cluster_uid='AB', task=1, ledger=ledger())
        self.assertEqual(before, (left.fingerprint(), right.fingerprint()))
        partition = partition_clients(similarities((0, 1), task=1), merge_check=merge_gate({0: left, 1: right}))
        self.assertEqual(partition.clusters, ((0,), (1,)))
        self.assertEqual(partition.rejected_merges, (((0,), (1,), 'MODEL_STATE_MISMATCH'),))

    def test_history_or_algorithm_mismatch_rejects_merge(self):
        left, right = fixture('A', (0,)), fixture('B', (1,))
        right.router_state['activation_memory'][0][0, 0] = 0
        self.assertEqual(compatibility(left, right), (False, 'ROUTER_HISTORY_MISMATCH'))
        right = fixture('B', (1,))
        right.algorithm_state['allocation_state']['policy'] = 'other'
        self.assertEqual(compatibility(left, right), (False, 'ALGORITHM_STATE_MISMATCH'))

    def test_merge_gate_caches_isolated_parent_snapshot(self):
        a, b = fixture('A', (0,)), fixture('B', (1,))
        gate = merge_gate({0: a, 1: b})
        a.model_state['fc2.bias'][2] = 1
        self.assertTrue(gate((0,), (1,))[0])
        self.assertFalse(merge_gate({0: a, 1: b})((0,), (1,))[0])

    def test_metadata_conflict_is_rejected_before_handoff(self):
        a, b = fixture('A', (0,)), fixture('B', (1,))
        a.ownership_metadata = {0: {'count': 2}}
        b.ownership_metadata = {0: {'count': 3}}
        self.assertEqual(compatibility(a, b), (False, 'OWNERSHIP_METADATA_CONFLICT'))
        partition = partition_clients(similarities((0, 1), task=1), merge_check=merge_gate({0: a, 1: b}))
        self.assertEqual(partition.K, 2)

    def test_failover_preserves_model_router_masks_history(self):
        parent = fixture()
        recovered, result = failover(parent, ledger(), unavailable=(0,))
        self.assertEqual(recovered.coordinator_id, 1)
        self.assertEqual(result.reason, 'failover')
        a, b = snapshot(parent), snapshot(recovered)
        a.pop('coordinator_id'); b.pop('coordinator_id')
        self.assertEqual(state_hash(a), state_hash(b))

    def test_registry_rejects_overlap_atomically(self):
        registry = ClusterRegistry()
        registry.install_task([fixture()], active_clients=(0, 1))
        before = state_hash(registry.snapshot())
        with self.assertRaises(ValueError):
            registry.install_task([fixture('A', (0,)), fixture('B', (0, 1))], active_clients=(0, 1))
        self.assertEqual(before, state_hash(registry.snapshot()))

    def test_registry_task_transition_retains_dormant_parents(self):
        registry = ClusterRegistry()
        parent = fixture()
        registry.install_task([parent], active_clients=(0, 1))
        children = split(parent, {'a': (0,), 'b': (1,)}, task=1, ledger=ledger())
        registry.install_task(children, active_clients=(0, 1))
        self.assertEqual(registry.active_uids, ('a', 'b'))
        self.assertIn('A', registry.states)
        self.assertEqual(registry.states['A'].fingerprint(), parent.fingerprint())
        self.assertEqual(state_hash(registry.snapshot()), state_hash(ClusterRegistry.restore(registry.snapshot()).snapshot()))

    def test_two_task_controlled_state_integration_and_replay(self):
        learned = ledger()
        template = fixture('common-init', (0, 1), round=-1)
        template.algorithm_state['unit_ranks'] = {k: np.zeros_like(v) for k, v in template.algorithm_state['unit_ranks'].items()}
        template.algorithm_state['freeze_masks'] = {k: np.zeros_like(v, dtype=bool) for k, v in template.algorithm_state['unit_ranks'].items()}
        template.protected_parameters = {}
        template.router_state.update(activation_memory={}, episode_classes={}, binarize_thresholds=None)
        template.routing_space_hash = template.routing_fingerprint()
        current = {0: {0: 3}, 1: {1: 5}}
        initial_partition = partition_clients(bootstrap_similarity(current, data_task=0,
            task_classes=TASK_CLASSES[0], reference_hash=template.fingerprint()))
        initial = initialize_clusters(template, initial_partition, learned)
        registry = ClusterRegistry()
        registry.install_task(initial, active_clients=(0, 1))
        self.assertEqual(len(initial), 2)
        mapping = {cid: state for state in initial for cid in state.members}
        next_partition = partition_clients(similarities((0, 1), task=1), merge_check=merge_gate(mapping))
        children = transition_clusters(next_partition, mapping, learned)
        replayed = transition_clusters(next_partition, mapping, learned)
        self.assertEqual(tuple(s.fingerprint() for s in children), tuple(s.fingerprint() for s in replayed))
        registry.install_task(children, active_clients=(0, 1))
        self.assertEqual(len(registry.active_uids), 1)
        self.assertEqual(len(registry.states), 3)
        self.assertEqual(learned.learned_classes(1, task=1), ())

    def test_task_transition_rejects_divergent_parent_without_reset(self):
        a, b = fixture('A', (0,)), fixture('B', (1,))
        b.model_state['fc2.bias'][2] = 1
        before = (a.fingerprint(), b.fingerprint())
        candidate = partition_clients(similarities((0, 1), task=1))
        with self.assertRaisesRegex(ValueError, 'MERGE_REJECTED'):
            transition_clusters(candidate, {0: a, 1: b}, ledger())
        self.assertEqual(before, (a.fingerprint(), b.fingerprint()))

    def test_registry_rejects_membership_change_in_task(self):
        registry = ClusterRegistry()
        registry.install_task([fixture()], active_clients=(0, 1))
        with self.assertRaisesRegex(ValueError, 'locked'):
            registry.install_task([fixture('B', (0, 1))], active_clients=(0, 1))

    def test_initialization_rejects_mature_or_foreign_reference(self):
        template = fixture('init', (0, 1), round=-1)
        template.router_state.update(activation_memory={}, episode_classes={}, binarize_thresholds=None)
        template.routing_space_hash = template.routing_fingerprint()
        partition = partition_clients(replace(similarities((0, 1)), reference_hash=template.fingerprint()))
        with self.assertRaisesRegex(ValueError, 'untrained'):
            initialize_clusters(template, partition, ledger())
        template.algorithm_state['unit_ranks'] = {k: np.zeros_like(v) for k, v in template.algorithm_state['unit_ranks'].items()}
        template.algorithm_state['freeze_masks'] = {k: np.zeros_like(v, dtype=bool) for k, v in template.algorithm_state['unit_ranks'].items()}
        with self.assertRaisesRegex(ValueError, 'reference'):
            initialize_clusters(template, partition, ledger())

    def test_checkpoint_after_split_restores_dormant_and_history(self):
        registry, book, learned = ClusterRegistry(), PartitionBook(), ledger()
        parent = fixture()
        registry.install_task([parent], active_clients=(0, 1))
        book.freeze(partition_clients(similarities((0, 1))), (0, 1))
        children = split(parent, {'a': (0,), 'b': (1,)}, task=1, ledger=learned)
        registry.install_task(children, active_clients=(0, 1))
        book.freeze(partition_clients(similarities((0, 1), scores=np.eye(2), task=1)), (0, 1))
        checkpoint = P0Checkpoint(registry, learned, book)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'split.json'
            before = save_checkpoint(path, checkpoint)
            resumed = load_checkpoint(path)
            self.assertEqual(before, state_hash(resumed.payload()))
            self.assertEqual(resumed.registry.states['A'].fingerprint(), parent.fingerprint())

    def test_checkpoint_restore_replay_and_integrity(self):
        registry, book, learned = ClusterRegistry(), PartitionBook(), ledger()
        registry.install_task([fixture()], active_clients=(0, 1))
        book.freeze(partition_clients(similarities((0, 1))), (0, 1))
        learned.record(event(), at_task=0, at_round=0)
        checkpoint = P0Checkpoint(registry, learned, book)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.json'
            before = save_checkpoint(path, checkpoint)
            resumed = load_checkpoint(path)
            self.assertEqual(before, state_hash(resumed.payload()))
            self.assertEqual(before, save_checkpoint(path, resumed))
            self.assertEqual(elect_leader((0, 1), learned, task=1), elect_leader((0, 1), resumed.ledger, task=1))
            envelope = json.loads(path.read_text())
            envelope['sha256'] = 'bad'
            path.write_text(json.dumps(envelope))
            with self.assertRaisesRegex(ValueError, 'integrity'):
                load_checkpoint(path)

    def test_checkpoint_rejects_future_or_wrong_partition(self):
        registry, book, learned = ClusterRegistry(), PartitionBook(), ledger()
        registry.install_task([fixture()], active_clients=(0, 1))
        book.freeze(partition_clients(similarities((0, 1), scores=np.eye(2))), (0, 1))
        with self.assertRaises(ValueError):
            P0Checkpoint(registry, learned, book).payload()
        book = PartitionBook()
        book.freeze(partition_clients(similarities((0, 1))), (0, 1))
        learned.record(event(1, (2,), task=1), at_task=1, at_round=0)
        with self.assertRaises(ValueError):
            P0Checkpoint(registry, learned, book).payload()

    def test_native_denice_model_state_shape_contract(self):
        from fed_learning.models.denice_model import DeNICEModel
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            model = DeNICEModel((16, 3), num_classes=4)
        weights = model.state_dict()
        layers = ('conv1', 'conv2', 'conv3', 'gru')
        width = sum(len(model.unit_ranks[k]) for k in layers)
        state = ClusterState.create(cluster_uid='native', task=0, round=-1, members=(0,), coordinator_id=0,
            versions=dict(architecture='legacy-cnn-gru', feature='native-adapter-free', class_mapping='fixture-4', class_ids=(0, 1, 2, 3)),
            model_state=weights, algorithm_state=dict(unit_ranks=model.unit_ranks, weight_masks=model.weight_masks,
                bias_masks=model.bias_masks, freeze_masks=model.freeze_masks, adapter_registry=model.adapter_registry),
            router_state=dict(router_mode='binary_cosine', context_layers=layers, stable_feature_mask=np.ones(width, bool),
                binarize_thresholds=None, activation_memory={}, episode_classes={}),
            bn_state={k: v for k, v in weights.items() if k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))},
            protected_parameters={}, routing_dependencies=tuple(k for k in weights if not k.startswith(('fc1.', 'fc2.'))))
        self.assertEqual(width, 548)
        self.assertEqual(state.fingerprint(), restore(snapshot(state)).fingerprint())


if __name__ == '__main__':
    unittest.main()
