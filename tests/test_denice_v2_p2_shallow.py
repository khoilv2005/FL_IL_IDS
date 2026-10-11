"""Versioned shallow contract, actual native optimizers and task handoff."""
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch
from fed_learning.models.denice_model import DeNICEModel
from fed_learning.strategies.decentralized.denice_v2.contracts import ClusterState, encode, decode, state_hash
from fed_learning.strategies.decentralized.denice_v2.knowledge import ClassKnowledgeLedger
from fed_learning.strategies.decentralized.denice_v2.native import (
    CurrentBase, capture_native, consolidate_native_task, load_native, run_native_round)
from fed_learning.strategies.decentralized.denice_v2.router import (
    ClusterRouter, allocated_feature_mask, capacity_receipt, probe_router_capacity, stage_cluster_router)
from fed_learning.strategies.decentralized.denice_v2.state import handoff, snapshot, restore


def recreate(payload):
    return ClusterState.create(**{k:v for k,v in payload.items() if k not in ('model_spec','routing_space_hash')})


class ShallowP2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        router = dict(router_mode='binary_cosine',context_layers=('conv1','conv2','conv3','gru'),
            stable_feature_mask=np.zeros(548,bool),binarize_thresholds=None,activation_memory={},episode_classes={})
        initial = capture_native(DeNICEModel((16,3),4),members=(0,1),coordinator_id=0,
            cluster_uid='shallow0',router_state=router,routing_dependencies=('bn1.num_batches_tracked',))
        data0 = {cid:CurrentBase(cid,0,torch.randn(8,16,3),torch.full((8,),cid,dtype=torch.int64),
            np.arange(8,dtype=np.int64),'synthetic-current-base') for cid in (0,1)}
        cls.round0,cls.plan0,cls.packets0,cls.receipt0 = run_native_round(initial,data0,(0,1),batch_size=4)
        cls.data0 = data0
        cls.p0 = stage_cluster_router(cls.round0.state,cls.plan0,cls.packets0,data0,feature_policy='conv1_only_v1')
        cls.state0 = cls.p0.commit()
        cls.mature = consolidate_native_task(cls.state0)
        ledger = ClassKnowledgeLedger({0:(0,1),1:(2,3)})
        for packet in cls.packets0:
            ledger.record(packet.event,at_task=0,at_round=0)
        cls.next_state = handoff([cls.mature],members=(0,1),cluster_uid='shallow1',task=1,ledger=ledger)
        cls.data1 = {
            0:CurrentBase(0,1,torch.empty(0,16,3),torch.empty(0,dtype=torch.int64),np.empty(0,dtype=np.int64),'synthetic-current-base'),
            1:CurrentBase(1,1,torch.randn(8,16,3),torch.tensor([2,3]*4),np.arange(8,dtype=np.int64),'synthetic-current-base')}
        cls.round1,cls.plan1,cls.packets1,cls.receipt1 = run_native_round(cls.next_state,cls.data1,(2,3),batch_size=4)
        with patch('fed_learning.strategies.decentralized.denice_v2.router.collect_routing_moments',side_effect=AssertionError('recalibrated old thresholds')):
            cls.p1 = stage_cluster_router(cls.round1.state,cls.plan1,cls.packets1,cls.data1,feature_policy='conv1_only_v1')
        cls.state1 = cls.p1.commit()

    def test_shallow_mask_real_builder(self):
        mask = self.state0.router_state['stable_feature_mask']
        self.assertTrue(mask[:64].any())
        self.assertFalse(mask[64:].any())
        self.assertEqual(self.state0.algorithm_state['routing_contract_version'],'conv1_only_v1')

    def test_capacity_shallow_and_old_baseline(self):
        shallow = probe_router_capacity(self.round0.state,'conv1_only_v1')['capacity']
        whole = probe_router_capacity(self.round0.state)['capacity']
        self.assertTrue(shallow['passed'])
        self.assertFalse(whole['passed'])
        self.assertEqual(shallow['per_layer']['conv2']['closure_locked'],0)

    def test_preexisting_gru_dependency_preserved_and_counted(self):
        payload = snapshot(self.round0.state)
        payload['routing_dependencies'] += ('gru.weight_hh_l1',)
        inherited = recreate(payload)
        probe = probe_router_capacity(inherited,'conv1_only_v1')
        self.assertIn('gru.weight_hh_l1',probe['dependencies'])
        self.assertEqual(probe['capacity']['per_layer']['gru']['closure_locked'],30000)
        # Actual committed protections are tested through a compatible plan.
        result,plan,packets,_ = run_native_round(inherited,self.data0,(0,1),batch_size=8)
        candidate = stage_cluster_router(result.state,plan,packets,self.data0,feature_policy='conv1_only_v1').commit()
        self.assertIn('gru.weight_hh_l1',candidate.routing_dependencies)
        self.assertTrue(candidate.protected_parameters['gru.weight_hh_l1'].all())

    def test_actual_mature_protection_in_capacity(self):
        cap = capacity_receipt(self.mature,self.mature.routing_dependencies)
        self.assertGreater(cap['per_layer']['conv2']['protected'],0)
        self.assertEqual(cap['per_layer']['conv2']['closure_locked'],0)
        self.assertGreater(cap['locked_backbone_parameters'],self.p0.receipt['capacity']['locked_backbone_parameters'])

    def test_capacity_counts_preexisting_freeze_without_maturity(self):
        payload = snapshot(self.round0.state)
        payload['algorithm_state']['freeze_masks']['conv2'][0] = True
        cap = probe_router_capacity(recreate(payload),'conv1_only_v1')['capacity']
        self.assertEqual(cap['per_layer']['conv2']['protected'],193)

    def test_real_optimizer_both_tasks(self):
        self.assertGreater(self.receipt0['workers'][0]['steps'],0)
        self.assertGreater(self.receipt1['workers'][1]['steps'],0)
        for layer in ('conv2.weight','conv3.weight','gru.weight_ih_l0'):
            self.assertGreater(self.receipt1['workers'][1]['changed_coordinates'][layer],0)

    def test_empty_client_no_optimizer_or_bn_sketch(self):
        self.assertEqual(self.receipt1['workers'][0]['steps'],0)
        self.assertEqual(self.packets1[0].event.successful_class_counts,())
        self.assertEqual(self.p1.receipt['sketch_contributors'],(1,))
        self.assertTrue(all(not np.any(m) for m in self.packets1[0].masks.values()))

    def test_empty_tensor_codec(self):
        for value in (torch.empty(0,dtype=torch.int64),torch.empty(0,16,3)):
            actual = decode(encode(value))
            self.assertEqual(value.shape,actual.shape)
            self.assertEqual(value.dtype,actual.dtype)

    def test_no_all_empty_cluster_training(self):
        empty = {cid:replace(self.data1[0],client_id=cid) for cid in (0,1)}
        with self.assertRaisesRegex(ValueError,'No current BASE'):
            run_native_round(self.next_state,empty,(2,3))

    def test_singleton_base_row_rejected(self):
        base = replace(self.data1[1],x=self.data1[1].x[:1],y=self.data1[1].y[:1],row_ids=np.array([0]))
        with self.assertRaisesRegex(ValueError,'Invalid current-task BASE'):
            base.validate((2,3))

    def test_historical_memory_threshold_fingerprint_exact(self):
        self.assertEqual(self.state0.routing_space_hash,self.state1.routing_space_hash)
        self.assertEqual(state_hash(self.state0.router_state['activation_memory'][0]),state_hash(self.state1.router_state['activation_memory'][0]))
        self.assertEqual(self.state0.router_state['binarize_thresholds'],self.state1.router_state['binarize_thresholds'])
        self.assertEqual(set(self.state1.router_state['activation_memory']),{0,1})

    def test_actual_current_witness_conv1_stable(self):
        x = self.data1[1].x
        before = load_native(self.next_state).eval().get_context_activations_per_sample(x)['conv1']
        after = load_native(self.state1).eval().get_context_activations_per_sample(x)['conv1']
        self.assertTrue(torch.equal(before,after))

    def test_irrelevant_deep_masks_do_not_bind_shallow_space(self):
        payload = snapshot(self.state1)
        payload['algorithm_state']['weight_masks']['conv2'][0,0,0] *= 0
        self.assertEqual(recreate(payload).routing_space_hash,self.state1.routing_space_hash)

    def test_conv1_masks_bind_space(self):
        payload = snapshot(self.state1)
        payload['algorithm_state']['weight_masks']['conv1'][0,0,0] *= 0
        self.assertNotEqual(recreate(payload).routing_space_hash,self.state1.routing_space_hash)

    def test_conv1_bn_buffers_bound_and_protected(self):
        for name in self.state0.routing_dependencies:
            self.assertTrue(torch.equal(self.state0.model_state[name],self.state1.model_state[name]))
        payload = snapshot(self.state1)
        payload['model_state']['bn1.running_mean'][0] += .1
        payload['bn_state']['bn1.running_mean'] = payload['model_state']['bn1.running_mean'].clone()
        self.assertNotEqual(recreate(payload).routing_space_hash,self.state1.routing_space_hash)

    def test_implicit_policy_migration_rejected(self):
        with self.assertRaisesRegex(ValueError,'Cannot migrate'):
            probe_router_capacity(self.state1)
        with self.assertRaisesRegex(ValueError,'Unknown'):
            allocated_feature_mask(self.state0,'unknown')

    def test_consolidation_is_single_and_weights_unchanged(self):
        self.assertEqual(state_hash(self.state0.model_state),state_hash(self.mature.model_state))
        with self.assertRaisesRegex(ValueError,'already consolidated'):
            consolidate_native_task(self.mature)

    def test_consolidation_retains_preexisting_freeze(self):
        payload = snapshot(self.state0)
        payload['algorithm_state']['freeze_masks']['conv2'][-1] = True
        state = recreate(payload)
        consolidated = consolidate_native_task(state)
        self.assertTrue(consolidated.algorithm_state['freeze_masks']['conv2'][-1])

    def test_protected_mature_state_survives_real_task1(self):
        for name,mask in self.mature.protected_parameters.items():
            self.assertTrue(torch.equal(self.mature.model_state[name][mask],self.state1.model_state[name][mask]))

    def test_restore_prediction_bits_scores_exact(self):
        state = restore(decode(json.loads(json.dumps(encode(snapshot(self.state1))))))
        a = ClusterRouter(self.state1).predict(self.data1[1].x)
        b = ClusterRouter(state).predict(self.data1[1].x)
        self.assertTrue(all(np.array_equal(x,y) for x,y in zip(a,b)))
        self.assertEqual(state.fingerprint(),self.state1.fingerprint())


if __name__ == '__main__':
    unittest.main()
