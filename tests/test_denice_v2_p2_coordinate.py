"""Coordinate closure, actual optimizer/BN survival, sparse upstream proof."""
import json
import unittest
from unittest.mock import patch
import numpy as np
import torch
from fed_learning.models.denice_model import DeNICEModel
from fed_learning.strategies.decentralized.denice_v2.contracts import encode,decode,state_hash
from fed_learning.strategies.decentralized.denice_v2.coordinate_router import (
    fresh_coordinate_state,cnn_closure,protect_coordinate_closure,recreate)
from fed_learning.strategies.decentralized.denice_v2.native import (
    CurrentBase,run_native_round,consolidate_native_task,load_native)
from fed_learning.strategies.decentralized.denice_v2.knowledge import ClassKnowledgeLedger
from fed_learning.strategies.decentralized.denice_v2.normalization import protected_bn_channels
from fed_learning.strategies.decentralized.denice_v2.aggregation import permissions
from fed_learning.strategies.decentralized.denice_v2.router import (
    ClusterRouter,stage_cluster_router,allocated_feature_mask,probe_router_capacity)
from fed_learning.strategies.decentralized.denice_v2.state import snapshot,restore,handoff


class CoordinateRouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);torch.manual_seed(42)
        cls.initial=fresh_coordinate_state(DeNICEModel((16,3),4),members=(0,1),coordinator_id=0,cluster_uid='coord0')
        cls.data0={cid:CurrentBase(cid,0,torch.randn(8,16,3),torch.full((8,),cid,dtype=torch.int64),
            np.arange(8,dtype=np.int64),'synthetic-current') for cid in (0,1)}
        cls.r0,cls.plan0,cls.packets0,cls.receipt0=run_native_round(cls.initial,cls.data0,(0,1),batch_size=4)
        cls.state0=stage_cluster_router(cls.r0.state,cls.plan0,cls.packets0,cls.data0,feature_policy='cnn_coordinate_router_v1').commit()
        ledger=ClassKnowledgeLedger({0:(0,1),1:(2,3)})
        for p in cls.packets0:ledger.record(p.event,at_task=0,at_round=0)
        cls.next_state=handoff([consolidate_native_task(cls.state0)],members=(0,1),cluster_uid='coord1',task=1,ledger=ledger)
        cls.data1={cid:CurrentBase(cid,1,torch.randn(8,16,3),torch.full((8,),2+cid,dtype=torch.int64),
            np.arange(8,dtype=np.int64),'synthetic-current') for cid in (0,1)}
        cls.r1,cls.plan1,cls.packets1,cls.receipt1=run_native_round(cls.next_state,cls.data1,(2,3),batch_size=4)
        with patch('fed_learning.strategies.decentralized.denice_v2.router.collect_routing_moments',side_effect=AssertionError('refresh old thresholds')):
            cls.state1=stage_cluster_router(cls.r1.state,cls.plan1,cls.packets1,cls.data1,feature_policy='cnn_coordinate_router_v1').commit()

    def test_policy_selects_conv2_only(self):
        mask=self.state0.router_state['stable_feature_mask']
        self.assertFalse(mask[:64].any());self.assertTrue(mask[64:192].any());self.assertFalse(mask[192:].any())

    def test_dense_upstream_requires_all_conv1(self):
        required=cnn_closure(self.state0,self.state0.router_state['stable_feature_mask'])
        self.assertTrue(required['channels']['conv1'].all())
        self.assertEqual(required['channels']['conv2'].sum(),64)
        self.assertNotIn('conv3.weight',required['parameters'])
        self.assertFalse(any(k.startswith('gru.') for k in required['parameters']))

    def test_partial_parameter_permissions_not_full_tensor(self):
        allowed=permissions(self.plan1.reference)
        self.assertFalse(allowed['conv2.weight'][:64].any())
        self.assertTrue(allowed['conv2.weight'][64:].all())
        self.assertTrue(allowed['conv3.weight'].any());self.assertTrue(allowed['gru.weight_ih_l0'].any())

    def test_bn_only_selected_channels_protected(self):
        mask=protected_bn_channels(self.state0,'bn2')
        self.assertTrue(mask[:64].all());self.assertFalse(mask[64:].any())
        self.assertTrue(protected_bn_channels(self.state0,'bn1').all())

    def test_bn_mutable_channels_recalibrate(self):
        self.assertTrue(torch.equal(self.state0.model_state['bn2.running_mean'][:64],self.state1.model_state['bn2.running_mean'][:64]))
        self.assertFalse(torch.equal(self.state0.model_state['bn2.running_mean'][64:],self.state1.model_state['bn2.running_mean'][64:]))

    def test_real_optimizer_updates_unprotected_rows(self):
        self.assertGreater(self.receipt1['workers'][0]['steps'],0)
        self.assertGreater(self.receipt1['workers'][0]['changed_coordinates']['conv2.weight'],0)
        self.assertGreater(self.receipt1['workers'][0]['changed_coordinates']['gru.weight_ih_l0'],0)
        self.assertEqual(self.receipt1['workers'][0]['changed_coordinates']['conv1.weight'],0)

    def test_coordinate_weights_and_bn_survive(self):
        for name,mask in self.state0.algorithm_state['routing_coordinate_masks'].items():
            mask=np.asarray(mask,bool)
            np.testing.assert_array_equal(self.state0.model_state[name].numpy()[mask],self.state1.model_state[name].numpy()[mask])

    def test_current_input_features_stable(self):
        x=self.data1[0].x
        a=load_native(self.state0).eval().get_context_activations_per_sample(x)['conv2'][:,:64]
        b=load_native(self.state1).eval().get_context_activations_per_sample(x)['conv2'][:,:64]
        torch.testing.assert_close(a,b,atol=1e-6,rtol=0)

    def test_memory_threshold_hash_unchanged(self):
        self.assertEqual(self.state0.routing_space_hash,self.state1.routing_space_hash)
        self.assertEqual(state_hash(self.state0.router_state['activation_memory'][0]),state_hash(self.state1.router_state['activation_memory'][0]))
        self.assertEqual(self.state0.router_state['binarize_thresholds'],self.state1.router_state['binarize_thresholds'])

    def test_unrelated_conv2_coordinate_not_in_fingerprint(self):
        payload=snapshot(self.state1);payload['model_state']['conv2.weight'][127,0,0]+=.1
        self.assertEqual(recreate(payload).routing_space_hash,self.state1.routing_space_hash)

    def test_selected_coordinate_changes_hash(self):
        payload=snapshot(self.state1);payload['model_state']['conv2.weight'][0,0,0]+=.1
        self.assertNotEqual(recreate(payload).routing_space_hash,self.state1.routing_space_hash)

    def test_connection_masks_including_inactive_bits_bound(self):
        payload=snapshot(self.state1);payload['algorithm_state']['weight_masks']['conv2'][0,0,0]=0
        self.assertNotEqual(recreate(payload).routing_space_hash,self.state1.routing_space_hash)
        payload=snapshot(self.state1);payload['algorithm_state']['weight_masks']['conv2'][127,0,0]=0
        self.assertEqual(recreate(payload).routing_space_hash,self.state1.routing_space_hash)

    def test_missing_upstream_protection_rejected(self):
        payload=snapshot(self.state1);payload['algorithm_state']['routing_coordinate_masks']['conv1.weight'][0]=False
        with self.assertRaisesRegex(ValueError,'Incomplete upstream'):
            recreate(payload)

    def test_missing_mask_binding_rejected(self):
        payload=snapshot(self.state1);payload['algorithm_state']['routing_weight_mask_coordinates']['conv2'][0]=False
        with self.assertRaisesRegex(ValueError,'Incomplete connection'):
            recreate(payload)

    def test_scalar_counter_codec_restore(self):
        state=restore(decode(json.loads(json.dumps(encode(snapshot(self.state1))))))
        self.assertEqual(state.routing_space_hash,self.state1.routing_space_hash)

    def test_restore_predictions_and_bits(self):
        state=restore(decode(json.loads(json.dumps(encode(snapshot(self.state1))))))
        a=ClusterRouter(self.state1).predict(self.data1[0].x);b=ClusterRouter(state).predict(self.data1[0].x)
        self.assertTrue(all(np.array_equal(x,y) for x,y in zip(a,b)))

    def test_class_registry_and_cumulative_ce(self):
        self.assertEqual(self.state1.algorithm_state['seen_classes'],(0,1,2,3))
        self.assertEqual(self.receipt1['workers'][0]['softmax_classes'],(0,1,2,3))
        self.assertEqual(self.state1.router_state['episode_classes'],{0:(0,1),1:(2,3)})

    def test_sparse_connections_limit_upstream_channels(self):
        payload=snapshot(self.initial)
        payload['algorithm_state']['unit_ranks']['conv2'][0]=1
        payload['algorithm_state']['weight_masks']['conv2'][0]=0
        payload['algorithm_state']['weight_masks']['conv2'][0,3,:]=1
        state=recreate(payload)
        mask=np.zeros(548,bool);mask[64]=True
        closed,required=protect_coordinate_closure(state,mask)
        self.assertEqual(np.flatnonzero(required['channels']['conv1']).tolist(),[3])
        self.assertTrue(required['weight_masks']['conv2'][0].all())
        # Changing a truly disconnected upstream output cannot change root.
        x=torch.randn(8,16,3)
        a=load_native(closed).eval().get_context_activations_per_sample(x)['conv2'][:,0]
        payload=snapshot(closed);payload['model_state']['conv1.weight'][2]+=10
        changed=recreate(payload)
        b=load_native(changed).eval().get_context_activations_per_sample(x)['conv2'][:,0]
        torch.testing.assert_close(a,b,atol=1e-6,rtol=0)
        self.assertEqual(changed.routing_space_hash,closed.routing_space_hash)

    def test_conv3_closure_includes_both_upstream_convs(self):
        model=DeNICEModel((16,3),4);model.unit_ranks['conv3'][0]=1
        state=fresh_coordinate_state(model,feature_layer='conv3',members=(0,1),coordinator_id=0,cluster_uid='conv3-fixture')
        mask=allocated_feature_mask(state,'cnn_coordinate_router_v1')
        required=cnn_closure(state,mask)
        self.assertTrue(required['channels']['conv1'].all());self.assertTrue(required['channels']['conv2'].all())
        self.assertEqual(required['channels']['conv3'].sum(),1)

    def test_existing_protection_is_never_narrowed(self):
        payload=snapshot(self.r0.state)
        payload['protected_parameters']['conv3.weight'][0]=True
        state=recreate(payload)
        proposed,_=protect_coordinate_closure(state,allocated_feature_mask(state,'cnn_coordinate_router_v1'))
        self.assertTrue(proposed.protected_parameters['conv3.weight'][0].all())

    def test_capacity_counts_coordinates_not_named_tensors(self):
        probe=probe_router_capacity(self.r0.state,'cnn_coordinate_router_v1')
        cap=probe['capacity'];self.assertTrue(cap['passed'])
        self.assertLess(cap['per_layer']['conv2']['closure_locked'],cap['per_layer']['conv2']['total'])
        self.assertEqual(cap['per_layer']['gru']['closure_locked'],0)
        self.assertGreater(cap['currently_trainable_before'],cap['currently_trainable_after'])

    def test_no_implicit_feature_policy_migration(self):
        with self.assertRaisesRegex(ValueError,'Cannot migrate'):
            probe_router_capacity(self.state1,'conv1_only_v1')

    def test_committed_feature_mask_cannot_expand(self):
        mask=self.state1.router_state['stable_feature_mask'].copy();mask[64+100]=True
        with self.assertRaisesRegex(ValueError,'Cannot change committed'):
            protect_coordinate_closure(self.state1,mask)

    def test_unallocated_feature_rejected(self):
        mask=np.zeros(548,bool);mask[64]=True
        with self.assertRaisesRegex(ValueError,'unallocated'):
            cnn_closure(self.initial,mask)

    def test_disconnected_root_needs_no_upstream_rows(self):
        payload=snapshot(self.initial)
        payload['algorithm_state']['unit_ranks']['conv2'][0]=1
        payload['algorithm_state']['weight_masks']['conv2'][0]=0
        state=recreate(payload);mask=np.zeros(548,bool);mask[64]=True
        _,required=protect_coordinate_closure(state,mask)
        self.assertNotIn('conv1.weight',required['parameters'])

    def test_gru_mutation_not_in_cnn_fingerprint(self):
        payload=snapshot(self.state1);payload['model_state']['gru.weight_ih_l0'][0,0]+=.1
        self.assertEqual(recreate(payload).routing_space_hash,self.state1.routing_space_hash)


if __name__=='__main__':unittest.main()
