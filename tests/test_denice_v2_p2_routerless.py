"""Opt-in routerless loss, class receipts, protection, label-blind predictor."""
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F
from fed_learning.models.denice_model import DeNICEModel
from fed_learning.strategies.decentralized.denice_v2.contracts import ClusterState,encode,decode,state_hash
from fed_learning.strategies.decentralized.denice_v2.knowledge import ClassKnowledgeLedger
from fed_learning.strategies.decentralized.denice_v2.native import CurrentBase,consolidate_native_task,run_native_round
from fed_learning.strategies.decentralized.denice_v2.routerless import (
    capture_routerless,register_learned_classes,RouterlessPredictor,CumulativeClassTrainer,masked_logits,training_classes)
from fed_learning.strategies.decentralized.denice_v2.state import snapshot,restore,handoff,broadcast
from tools.run_denice_v2_routerless_smoke import metrics,protection_exact


def recreate(payload):
    return ClusterState.create(**{k:v for k,v in payload.items() if k not in ('model_spec','routing_space_hash')})


class RouterlessP2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);torch.manual_seed(42)
        cls.initial=capture_routerless(DeNICEModel((16,3),4),members=(0,1),coordinator_id=0,cluster_uid='no-route0')
        cls.data0={cid:CurrentBase(cid,0,torch.randn(8,16,3),torch.full((8,),cid,dtype=torch.int64),
            np.arange(8,dtype=np.int64),'synthetic-current') for cid in (0,1)}
        cls.r0,cls.plan0,cls.packets0,cls.receipt0=run_native_round(cls.initial,cls.data0,(0,1),batch_size=4)
        cls.state0,cls.reg0=register_learned_classes(cls.r0.state,cls.plan0,cls.packets0)
        cls.mature=consolidate_native_task(cls.state0)
        ledger=ClassKnowledgeLedger({0:(0,1),1:(2,3)})
        for p in cls.packets0:
            ledger.record(p.event,at_task=0,at_round=0)
        cls.next_state=handoff([cls.mature],members=(0,1),cluster_uid='no-route1',task=1,ledger=ledger)
        cls.data1={
            0:CurrentBase(0,1,torch.empty(0,16,3),torch.empty(0,dtype=torch.int64),np.empty(0,dtype=np.int64),'synthetic-current'),
            1:CurrentBase(1,1,torch.randn(8,16,3),torch.full((8,),2,dtype=torch.int64),np.arange(8,dtype=np.int64),'synthetic-current')}
        cls.r1,cls.plan1,cls.packets1,cls.receipt1=run_native_round(cls.next_state,cls.data1,(2,3),batch_size=4)
        cls.state1,cls.reg1=register_learned_classes(cls.r1.state,cls.plan1,cls.packets1)

    def test_fresh_state_no_task_router_or_dependencies(self):
        self.assertEqual(self.initial.routing_dependencies,())
        self.assertEqual(self.initial.router_state['context_layers'],())
        self.assertEqual(self.initial.router_state['activation_memory'],{})
        self.assertIsNone(self.initial.router_state['binarize_thresholds'])
        self.assertEqual(self.initial.algorithm_state['seen_classes'],())

    def test_class_registration_actual_receipts(self):
        self.assertEqual(self.reg0['trained_classes'],(0,1))
        self.assertEqual(self.state1.algorithm_state['seen_classes'],(0,1,2))
        self.assertEqual(self.state1.algorithm_state['learned_classes_by_task'],{0:(0,1),1:(2,)})
        self.assertEqual(self.reg1['contributors'][0],())
        self.assertEqual(self.state1.algorithm_state['unit_ranks']['fc2'][3],0)

    def test_registration_no_weight_changes(self):
        self.assertEqual(state_hash(self.state1.model_state),state_hash(self.r1.state.model_state))
        self.assertTrue(self.reg1['weights_unchanged'])

    def test_cluster_ce_authority_not_local_minibatch_support(self):
        self.assertEqual(training_classes(self.plan0),(0,1))
        self.assertEqual(training_classes(self.plan1),(0,1,2))
        self.assertEqual(self.receipt0['workers'][0]['softmax_classes'],(0,1))
        self.assertEqual(self.receipt1['workers'][1]['softmax_classes'],(0,1,2))

    def test_real_native_optimizers_finite_loss(self):
        for w in (*self.receipt0['workers'],self.receipt1['workers'][1]):
            self.assertGreater(w['steps'],0)
            self.assertTrue(np.isfinite(w['loss_mean']))
            self.assertEqual(w['classification_objective'],'cumulative_registered_ce_v1')
        self.assertEqual(self.receipt1['workers'][0]['steps'],0)

    def test_future_logits_excluded_from_loss_and_gradient(self):
        raw=torch.tensor([[1.,1000.,3.,1000.]],requires_grad=True)
        trainer=CumulativeClassTrainer((0,2))
        loss=trainer.compute_loss(None,masked_logits(raw,(0,2)),torch.tensor([2]))
        expected=F.cross_entropy(raw[:,[0,2]],torch.tensor([1]))
        torch.testing.assert_close(loss,expected)
        loss.backward()
        self.assertEqual(raw.grad[0,1].item(),0)
        self.assertEqual(raw.grad[0,3].item(),0)
        self.assertNotEqual(raw.grad[0,0].item(),0)

    def test_old_registered_logits_compete_with_current(self):
        trainer=CumulativeClassTrainer((0,2))
        loss=trainer.compute_loss(None,torch.tensor([[5.,0.,1.,0.]]),torch.tensor([2]))
        self.assertGreater(loss.item(),4)

    def test_unregistered_target_rejected(self):
        with self.assertRaisesRegex(ValueError,'Target outside'):
            CumulativeClassTrainer((0,2)).compute_loss(None,torch.zeros(1,4),torch.tensor([3]))

    def test_invalid_class_mask_rejected(self):
        for classes in ((),(2,0),(1,1),(9,)):
            with self.assertRaisesRegex(ValueError,'Invalid active'):
                masked_logits(torch.zeros(1,4),classes)

    def test_routerless_inference_never_calls_context_apis(self):
        with (patch('fed_learning.servers.nice_server.ContextDetector',side_effect=AssertionError('router')),
              patch.object(DeNICEModel,'get_context_activations_per_sample',side_effect=AssertionError('activation')),
              patch.object(DeNICEModel,'get_output_and_context_activations',side_effect=AssertionError('task route'))):
            pred,logits=RouterlessPredictor(self.state1).predict(self.data1[1].x)
        self.assertTrue(set(pred.tolist())<=set((0,1,2)))
        self.assertTrue(np.isneginf(logits[:,3]).all())

    def test_prediction_argmax_cumulative_logits(self):
        predictor=RouterlessPredictor(self.state1)
        x=self.data1[1].x
        with torch.no_grad():
            raw=predictor.model(x).numpy()
        actual,_=predictor.predict(x)
        np.testing.assert_array_equal(actual,raw[:,:3].argmax(1))

    def test_prediction_cannot_take_labels_or_true_task(self):
        predictor=RouterlessPredictor(self.state1)
        with self.assertRaises(TypeError):
            predictor.predict(self.data1[1].x,self.data1[1].y)
        with self.assertRaises(TypeError):
            predictor.predict(self.data1[1].x,task=1)

    def test_mature_parameters_and_bn_retained(self):
        self.assertTrue(protection_exact(self.next_state,self.state1))
        self.assertFalse(self.state1.routing_dependencies)
        self.assertEqual(self.receipt1['workers'][1]['changed_coordinates']['fc2.weight']>0,True)
        self.assertTrue(np.all(self.packets1[1].deltas['fc2.weight'][:2]==0))

    def test_consolidation_and_handoff_no_memory(self):
        self.assertEqual(state_hash(self.state0.model_state),state_hash(self.mature.model_state))
        self.assertEqual(self.next_state.algorithm_state['seen_classes'],(0,1))
        self.assertFalse(self.next_state.router_state['episode_classes'])

    def test_registration_rejects_missing_duplicate_stale_receipts(self):
        for packets in (self.packets1[:1],[self.packets1[0],self.packets1[0]],self.packets0):
            with self.assertRaises(ValueError):
                register_learned_classes(self.r1.state,self.plan1,packets)

    def test_no_predictor_for_untrained_state(self):
        with self.assertRaisesRegex(ValueError,'requires learned'):
            RouterlessPredictor(self.initial)

    def test_routerless_cannot_silently_add_dependencies(self):
        payload=snapshot(self.state1)
        payload['routing_dependencies']=('conv1.weight',)
        with self.assertRaisesRegex(ValueError,'cannot carry'):
            recreate(payload)

    def test_routerless_cannot_store_activation_memory(self):
        payload=snapshot(self.state1)
        payload['router_state']['activation_memory'][0]=np.zeros((1,0),bool)
        with self.assertRaisesRegex(ValueError,'cannot carry'):
            recreate(payload)

    def test_forged_seen_classes_invalid(self):
        payload=snapshot(self.state1)
        payload['algorithm_state']['seen_classes']=(0,1,2,3)
        with self.assertRaisesRegex(ValueError,'Invalid routerless'):
            recreate(payload)

    def test_future_class_registration_invalid(self):
        payload=snapshot(self.state1)
        payload['algorithm_state']['learned_classes_by_task'][2]=(3,)
        with self.assertRaisesRegex(ValueError,'Invalid routerless'):
            recreate(payload)

    def test_checkpoint_prediction_and_class_registry_exact(self):
        state=restore(decode(json.loads(json.dumps(encode(snapshot(self.state1))))))
        a=RouterlessPredictor(state).predict(self.data1[1].x)
        b=RouterlessPredictor(self.state1).predict(self.data1[1].x)
        self.assertTrue(all(np.array_equal(x,y) for x,y in zip(a,b)))
        self.assertEqual(state.fingerprint(),self.state1.fingerprint())

    def test_broadcast_shared_predictor(self):
        copies=broadcast(self.state1)
        a=RouterlessPredictor(copies[0]).predict(self.data1[1].x)
        b=RouterlessPredictor(copies[1]).predict(self.data1[1].x)
        self.assertTrue(all(np.array_equal(x,y) for x,y in zip(a,b)))

    def test_metrics_include_absent_classes_in_macro_scope(self):
        result=metrics(np.array([0,0]),np.array([0,0]),4)
        self.assertEqual(result['accuracy'],1)
        self.assertEqual(result['macro_f1'],.25)
        self.assertEqual(result['missing_validation_classes'],[1,2,3])


if __name__=='__main__':
    unittest.main()
