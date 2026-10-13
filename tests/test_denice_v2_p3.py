"""P3 archive, evaluation, causal membership and correctness regression tests."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from fed_learning.strategies.decentralized.denice_v2.archive import TaskArchive
from fed_learning.strategies.decentralized.denice_v2.contracts import state_hash
from fed_learning.strategies.decentralized.denice_v2.evaluation import evaluate_cluster,aggregate_evaluations,metrics
from fed_learning.strategies.decentralized.denice_v2.native import CurrentBase,prepare_native_round,train_native_worker
from fed_learning.strategies.decentralized.denice_v2.runner import Runner,RunConfig
from tools.run_denice_v2_end_to_end_smoke import FixtureData


class P3ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.archive=TaskArchive(self.root,0)
        self.payload={1:torch.arange(12,dtype=torch.float32).reshape(3,4),
            'mask':np.ones(12,bool),'tuple':(3,np.array(True)),'empty':torch.empty(0)}
    def tearDown(self):self.tmp.cleanup()
    def test_exact_readback_typed_state(self):
        self.archive.save(0,self.payload)
        self.assertEqual(state_hash(self.payload),state_hash(self.archive.load()))
    def test_identical_tensors_deduplicated_across_rounds(self):
        a=self.archive.save(0,self.payload);b=self.archive.save(1,{**self.payload,'round':1})
        self.assertGreater(a['new_blobs'],0);self.assertEqual(b['new_blobs'],0)
    def test_mutation_different_blob(self):
        self.archive.save(0,self.payload)
        p={**self.payload,1:self.payload[1]+1}
        self.assertEqual(self.archive.save(1,p)['new_blobs'],1)
    def test_single_coordinate_change_does_not_duplicate_full_tensor(self):
        torch.manual_seed(99)
        original=torch.randn(8192)
        self.archive.save(0,{'large':original})
        changed=original.clone();changed[4000]+=1
        receipt=self.archive.save(1,{'large':changed})
        self.assertEqual(receipt['new_blobs'],1)
        self.assertLess(receipt['bytes'],original.numel()*original.element_size()/2)
        self.assertTrue(torch.equal(self.archive.load(1)['large'],changed))
    def test_seal_self_contained_all_rounds(self):
        self.archive.save(0,self.payload);self.archive.save(1,{**self.payload,'round':1})
        self.archive.seal()
        self.assertEqual(len(self.archive.sources()),1)
        self.assertEqual(state_hash(self.archive.load(0)),state_hash(self.payload))
        self.assertEqual(self.archive.load(1)['round'],1)
        self.assertFalse(list(self.root.glob('*round_*.zip')))
    def test_round_overwrite_rejected(self):
        self.archive.save(0,self.payload)
        with self.assertRaises(ValueError):self.archive.save(0,self.payload)
    def test_sealed_append_rejected(self):
        self.archive.save(0,self.payload);self.archive.seal()
        with self.assertRaises(ValueError):self.archive.save(1,self.payload)
    def test_partial_not_resume_source(self):
        (self.root/'checkpoint_task_0_round_000.zip.partial').write_bytes(b'incomplete')
        with self.assertRaises(FileNotFoundError):self.archive.load()
    def test_corrupted_blob_rejected(self):
        self.archive.save(0,self.payload)
        target=self.archive.sources()[0]
        with zipfile.ZipFile(target) as z:items={n:z.read(n) for n in z.namelist()}
        address=next(n for n in items if n.startswith('blobs/') and len(items[n]))
        items[address]=b'corrupt'
        with zipfile.ZipFile(target,'w') as z:
            for n,v in items.items():z.writestr(n,v)
        with self.assertRaisesRegex(ValueError,'checksum'):self.archive.load()
    def test_nonfinite_metadata_rejected(self):
        with self.assertRaises(ValueError):self.archive.save(0,{'bad':float('nan')})
    def test_mounted_resume_prefix_retains_earlier_rounds(self):
        self.archive.save(0,self.payload);self.archive.save(1,{**self.payload,'round':1});self.archive.seal()
        destination=TaskArchive(self.root/'continuation',0)
        destination.import_prefix(self.archive,0)
        self.assertEqual(state_hash(destination.load(0)),state_hash(self.payload))
        destination.save(1,{**self.payload,'round':1});destination.seal()
        self.assertEqual(destination.load(1)['round'],1)
    def test_budget_preserves_previous_checkpoint(self):
        self.archive.save(0,self.payload)
        self.archive.budget=1
        with self.assertRaises(RuntimeError):self.archive.save(1,self.payload)
        self.assertEqual(state_hash(self.archive.load(0)),state_hash(self.payload))


class P3EvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.data=FixtureData();self.config=RunConfig(smoke=True,tasks=2,rounds_per_task=1)
        self.runner=Runner(self.data,self.root/'run',self.config)
        self.runner.initialize();self.data.prepare(0);self.runner.start_task(0)
        self.state=next(iter(self.runner.unique_states().values()))
    def tearDown(self):self.tmp.cleanup()
    def test_unknown_unbuilt_router_counts_all_test_rows(self):
        x,y,binding=self.data.test()
        r=evaluate_cluster(self.state,x,y,binding,self.root/'eval.json',batch_size=17)
        self.assertEqual(r['metrics']['rows'],18)
        self.assertEqual(r['metrics']['accuracy'],0)
        self.assertEqual(r['metrics']['unregistered_prediction_rows'],18)
    def test_completed_evaluation_resumes_without_prediction(self):
        x,y,b=self.data.test();p=self.root/'eval.json'
        a=evaluate_cluster(self.state,x,y,b,p)
        with patch('fed_learning.strategies.decentralized.denice_v2.evaluation.ClusterRouter',side_effect=AssertionError):
            self.assertEqual(a,evaluate_cluster(self.state,x,y,b,p))
    def test_test_binding_mismatch_rejected(self):
        x,y,b=self.data.test();p=self.root/'eval.json';evaluate_cluster(self.state,x,y,b,p)
        with self.assertRaises(ValueError):evaluate_cluster(self.state,x,y,'changed',p)
    def test_fixed_class_macro_includes_missing_classes(self):
        cm=np.zeros((34,35),np.int64);cm[0,0]=5
        self.assertEqual(metrics(cm,range(6))['macro_f1'],1/6)
    def test_headline_is_mean_of_predictors_not_ensemble(self):
        rows=[dict(completed=True,cluster_uid=str(i),metrics=dict(rows=10,accuracy=a,macro_f1=f)) for i,a,f in [(1,.8,.6),(2,.2,.1)]]
        r=aggregate_evaluations(rows)
        self.assertEqual(r['mean_cluster_accuracy'],.5);self.assertEqual(r['unique_test_rows'],10)
        self.assertEqual(r['prediction_evaluations'],20)
    def test_unequal_test_pool_rejected(self):
        rows=[dict(completed=True,cluster_uid=str(i),metrics=dict(rows=i,accuracy=.5,macro_f1=.5)) for i in [1,2]]
        with self.assertRaises(ValueError):aggregate_evaluations(rows)
    def test_unknown_prediction_does_not_drop_classes(self):
        cm=np.zeros((34,35),np.int64);cm[2,34]=10
        self.assertEqual(metrics(cm,range(6))['per_class'][2]['support'],10)
    def test_full_configuration_cannot_be_capped_smoke(self):
        with self.assertRaises(ValueError):RunConfig(tasks=2).validate()
        RunConfig().validate()
    def test_task0_multi_cluster_automatic_not_fixed_K(self):
        self.assertEqual(self.runner.book.partitions[0].K,2)
        self.assertEqual(len(self.runner.active),2)
    def test_zero_update_single_class_is_report_only(self):
        cid=min(self.state.members);x=torch.randn(4,39,1);y=torch.zeros(4,dtype=torch.int64)
        bases={c:CurrentBase(c,0,x,y,np.arange(4),'one_class') for c in self.state.members}
        plan,_=prepare_native_round(self.state,bases,tuple(range(6)))
        _,receipt=train_native_worker(plan,bases[cid],batch_size=4,engineering_baseline=True)
        self.assertGreater(receipt['steps'],0);self.assertTrue(receipt['zero_parameter_update'])
    def test_singleton_opt_in_uses_current_row_only(self):
        cid=min(self.state.members)
        one=CurrentBase(cid,0,torch.randn(1,39,1),torch.tensor([0]),np.array([9]),'one_row',allow_singleton=True)
        self.assertEqual(one.validate(range(6)),{0:1})
        with self.assertRaises(ValueError):replace(one,allow_singleton=False).validate(range(6))
    def test_nonfinite_BASE_is_correctness_failure(self):
        bad=CurrentBase(0,0,torch.full((2,39,1),float('nan')),torch.tensor([0,1]),np.arange(2),'nan')
        with self.assertRaises(ValueError):bad.validate(range(6))
    def test_dormant_then_rejoining_client_retains_state(self):
        before=self.runner.states[0]
        self.data.prepare(1);self.data.counts[0]={}
        self.runner.start_task(1)
        dormant=self.runner.states[0]
        self.assertEqual(dormant.members,(0,))
        self.assertEqual(state_hash(dormant.model_state),state_hash(before.model_state))
        self.data.prepare(2);self.runner.start_task(2)
        self.assertIn(0,self.runner.states[0].members)
        self.assertEqual(self.runner.states[0].task,2)
    def test_eval_checksum_rejects_modified_counters(self):
        x,y,b=self.data.test();p=self.root/'eval.json';evaluate_cluster(self.state,x,y,b,p)
        value=json.loads(p.read_text());value['confusion'][0][0]=999;p.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'checksum'):evaluate_cluster(self.state,x,y,b,p)


if __name__=='__main__':unittest.main()
