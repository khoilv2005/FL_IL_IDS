"""P2 shadow contract fixtures. A NO-GO shadow is never production committed.

Two-task fixtures use synthetic delta receipts, not native survival training.
Current-task BASE tensors are the only raw inputs accepted by each builder.
"""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fed_learning.models.denice_model import DeNICEModel
from fed_learning.strategies.decentralized.denice_v2.aggregation import aggregate_round, start_worker, finish_worker
from fed_learning.strategies.decentralized.denice_v2.contracts import ClusterState, encode, decode, state_hash
from fed_learning.strategies.decentralized.denice_v2.knowledge import ClassKnowledgeLedger, LocalLearningEvent
from fed_learning.strategies.decentralized.denice_v2.native import CurrentBase, NativeBNCollector, capture_native, prepare_native_round
from fed_learning.strategies.decentralized.denice_v2.router import (
    ClusterRouter, aggregate_thresholds, collect_routing_moments, collect_sketch,
    dependency_closure, merge_sketches, probe_router_capacity, stage_cluster_router)
from fed_learning.strategies.decentralized.denice_v2.state import broadcast, handoff, restore, snapshot


def bases(task):
    generator = torch.Generator().manual_seed(2026 + task)
    return {cid: CurrentBase(cid, task, torch.randn(8, 16, 3, generator=generator),
        torch.full((8,), task*2+cid, dtype=torch.int64), np.arange(8, dtype=np.int64), 'synthetic-current-BASE') for cid in (0,1)}


def controlled_round(state, data):
    plan, _ = prepare_native_round(state, data, (state.task*2, state.task*2+1))
    packets = []
    for cid in state.members:
        worker = start_worker(plan, cid)
        label = state.task*2+cid
        worker.state.model_state['fc2.weight'][label, 0] += .01
        event = LocalLearningEvent(f'synthetic/{state.task}/{cid}', cid, state.task, plan.target_round, 1, ((label,8),), state.task)
        packets.append(finish_worker(plan, worker, event))
    result = aggregate_round(plan, packets, bn_provider=NativeBNCollector(plan, data))
    return result.state, plan, packets


class RouterP2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        model = DeNICEModel((16,3),4)
        router = dict(router_mode='binary_cosine',context_layers=('conv1','conv2','conv3','gru'),
            stable_feature_mask=np.zeros(548,bool),binarize_thresholds=None,activation_memory={},episode_classes={})
        initial = capture_native(model,members=(0,1),coordinator_id=0,cluster_uid='p2-fixture',
            router_state=router,routing_dependencies=('gru.weight_hh_l1',))
        cls.data0=bases(0)
        cls.round0,cls.plan0,cls.packets0=controlled_round(initial,cls.data0)
        cls.proposal0=stage_cluster_router(cls.round0,cls.plan0,cls.packets0,cls.data0,memo_per_class=4)
        # Shadow-only handoff; this exercises structural branches without
        # bypassing the production commit capacity gate.
        ledger=ClassKnowledgeLedger({0:(0,1),1:(2,3)})
        next_state=handoff([cls.proposal0.state],members=(0,1),cluster_uid='p2-next-fixture',task=1,ledger=ledger)
        cls.data1=bases(1)
        cls.round1,cls.plan1,cls.packets1=controlled_round(next_state,cls.data1)
        with patch('fed_learning.strategies.decentralized.denice_v2.router.collect_routing_moments', side_effect=AssertionError('historical threshold refresh')):
            cls.proposal1=stage_cluster_router(cls.round1,cls.plan1,cls.packets1,cls.data1,memo_per_class=4)

    def test_capacity_no_go_full_backbone_and_commit_reject(self):
        probe=probe_router_capacity(self.round0)
        self.assertEqual(probe['capacity']['locked_backbone_fraction'],1)
        self.assertEqual(probe['capacity']['remaining_backbone_parameters'],0)
        self.assertGreater(probe['capacity']['currently_trainable_after'],0)
        with self.assertRaisesRegex(ValueError,'P2_NO_GO'):
            self.proposal0.commit()

    def test_closure_upstream_bn_and_both_gru_layers(self):
        mask=np.zeros(548,bool);mask[64+128]=True;mask[-1]=True
        deps,_=dependency_closure(self.round0,mask)
        for k in ('conv1.weight','conv2.weight','conv3.weight','bn1.running_var','bn2.weight','bn3.running_mean',
                  'gru.weight_ih_l0','gru.weight_hh_l0','gru.weight_ih_l1','gru.weight_hh_l1','gru.bias_hh_l1'):
            self.assertIn(k,deps)
        self.assertNotIn('fc1.weight',deps)

    def test_empty_feature_mask_rejected(self):
        with self.assertRaisesRegex(ValueError,'Empty'):
            dependency_closure(self.round0,np.zeros(548,bool))

    def test_shared_thresholds_equal_direct_mean_std(self):
        state=self.proposal0.state
        from fed_learning.strategies.decentralized.denice_v2.native import load_native
        model=load_native(state)
        values=model.get_context_activations_per_sample(torch.cat([b.x for b in self.data0.values()]))
        offset=0
        for k in state.router_state['context_layers']:
            width=values[k].shape[1]
            mask=state.router_state['stable_feature_mask'][offset:offset+width]
            z=values[k][:,torch.tensor(mask)].double()
            expected=float(z.mean()+z.std()) if z.numel()>1 else 0.
            self.assertAlmostEqual(state.router_state['binarize_thresholds'][k],expected,places=6)
            offset+=width

    def test_stale_moment_and_missing_contributor_rejected(self):
        state=self.proposal0.state;mask=state.router_state['stable_feature_mask']
        packets=[collect_routing_moments(state,self.plan0,b,mask) for b in self.data0.values()]
        with self.assertRaisesRegex(ValueError,'Stale'):
            aggregate_thresholds([replace(packets[0],model_hash='stale'),packets[1]],state,self.plan0,mask)
        with self.assertRaisesRegex(ValueError,'Missing'):
            aggregate_thresholds(packets[:1],state,self.plan0,mask)

    def test_compatible_sketches_current_class_registration(self):
        r=self.proposal0.state.router_state
        self.assertEqual(r['activation_memory'][0].shape,(8,548))
        self.assertEqual(r['episode_classes'],{0:(0,1)})
        self.assertFalse(np.any(r['activation_memory'][0][:,~r['stable_feature_mask']]))
        self.assertFalse(self.proposal0.receipt['raw_reference_bank'])

    def test_sketch_header_classes_and_degenerate_bank_rejected(self):
        state=self.proposal0.state
        packets=[collect_sketch(state,self.plan0,self.data0[cid],(cid,),memo_per_class=4) for cid in (0,1)]
        actual={0:(0,),1:(1,)}
        with self.assertRaisesRegex(ValueError,'Incompatible'):
            merge_sketches(state,self.plan0,[replace(packets[0],routing_hash='wrong'),packets[1]],actual,4)
        with self.assertRaisesRegex(ValueError,'Incompatible'):
            merge_sketches(state,self.plan0,[replace(packets[0],class_counts={2:4}),packets[1]],actual,4)
        with self.assertRaisesRegex(ValueError,'DEGENERATE'):
            merge_sketches(state,self.plan0,[replace(p,bits=np.zeros_like(p.bits)) for p in packets],actual,4)

    def test_two_task_memory_thresholds_preserved_no_refresh(self):
        a,b=self.proposal0.state,self.proposal1.state
        self.assertEqual(set(b.router_state['activation_memory']),{0,1})
        self.assertEqual(state_hash(a.router_state['activation_memory'][0]),state_hash(b.router_state['activation_memory'][0]))
        self.assertEqual(a.router_state['binarize_thresholds'],b.router_state['binarize_thresholds'])
        self.assertEqual(a.routing_space_hash,b.routing_space_hash)
        self.assertFalse(self.proposal1.receipt['threshold_calibrated'])
        self.assertTrue(self.proposal1.receipt['historical_memory_unchanged'])
        self.assertEqual(self.proposal1.receipt['threshold_wire_bytes'],0)

    def test_label_blind_two_task_inference_and_broadcast(self):
        copies=broadcast(self.proposal1.state)
        x=self.data1[0].x
        outputs=[ClusterRouter(s).predict(x) for s in copies.values()]
        for first,second in zip(outputs[0],outputs[1]):
            np.testing.assert_array_equal(first,second)
        y,tasks,scores,bits=outputs[0]
        self.assertEqual(scores.shape,(8,2))
        for label,task in zip(y,tasks):
            self.assertIn(int(label),self.proposal1.state.router_state['episode_classes'][int(task)])
        with self.assertRaises(TypeError):
            ClusterRouter(self.proposal1.state).predict(x,y_true=torch.zeros(8))

    def test_snapshot_restore_same_routing_without_raw_bank(self):
        state=self.proposal1.state
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'state.json'
            path.write_text(json.dumps(encode(snapshot(state))))
            recovered=restore(decode(json.loads(path.read_text())))
        self.assertEqual(state.fingerprint(),recovered.fingerprint())
        for a,b in zip(ClusterRouter(state).predict(self.data1[1].x),ClusterRouter(recovered).predict(self.data1[1].x)):
            np.testing.assert_array_equal(a,b)
        self.assertFalse(ClusterRouter(recovered).detector.reference_input_memory)

    def test_protected_weights_and_function_metadata_reject_drift(self):
        state=restore(snapshot(self.proposal1.state))
        state.model_state['conv1.weight'][0,0,0]+=.01
        with self.assertRaisesRegex(ValueError,'Routing dependency'):
            state.validate()
        state=restore(snapshot(self.proposal1.state))
        state.algorithm_state['structural_protection']=True
        with self.assertRaisesRegex(ValueError,'Routing dependency'):
            state.validate()

    def test_mutated_proposal_cannot_commit(self):
        proposal=type(self.proposal0)(restore(snapshot(self.proposal0.state)),dict(self.proposal0.receipt),self.proposal0.binding)
        proposal.receipt['capacity']=dict(proposal.receipt['capacity'],passed=True)
        with self.assertRaisesRegex(ValueError,'mutated'):
            proposal.commit()

    def test_historical_base_rejected(self):
        bad=dict(self.data1);bad[0]=self.data0[0]
        with self.assertRaisesRegex(ValueError,'BASE content'):
            stage_cluster_router(self.round1,self.plan1,self.packets1,bad)

    def test_builder_does_not_mutate_reference(self):
        self.assertEqual(self.round0.fingerprint(),self.proposal0.receipt['reference_state_hash'])
        self.assertEqual(state_hash(self.round0.model_state),state_hash(self.proposal0.state.model_state))


if __name__=='__main__':
    unittest.main()
