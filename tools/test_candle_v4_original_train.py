"""Original-TRAIN integration regression, synthetic NPZs only."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'candle_v4_runtime'))
import numpy as np
import torch
from fed_learning.data.incremental_loader import IncrementalDataLoader
from fed_learning.training.decentralized_denice_il import _split_local_validation
from fed_learning.training import candle_v4_original_train as original
from fed_learning.training.candle_v4_shadow import digest,native_fingerprint


class OriginalTrainTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name);self.data=self.path/'data';self.data.mkdir()
        tasks={str(t):list(range(t*6,min(t*6+6,34))) for t in range(6)}
        (self.data/'metadata.json').write_text(json.dumps(dict(task_structure=dict(task_classes=tasks))),encoding='utf-8')
        for owner,classes in enumerate(((2,),(0,2),(1,2),(2,))):
            y=np.concatenate((np.repeat(classes,5000),np.repeat(6,400)))
            x=np.random.default_rng(owner).normal(size=(len(y),39)).astype(np.float32);x[:,0]=y
            np.savez(self.data/f'client_{owner}_train.npz',X_train=x,y_train=y)
        self.spec=dict(version=original.VERSION,seed=42,input_sha256='synthetic-metadata',
            data_dir=str(self.data),output_dir=str(self.path/'out'),arms=['coverage'],fixed_threshold=.5,
            per_expert_bytes=128*1024**2,per_arm_bytes=64*1024**3)
        self.loader=IncrementalDataLoader(str(self.data));self.clients=self.load_task(0)
        self.models={o:torch.nn.Linear(39,34) for o in self.clients};self.detectors={o:None for o in self.clients}
        self.graph=dict(alpha_debug={o:dict(group_ids=list(range(4)),alphas=[.25]*4) for o in range(4)})

    def load_task(self,task):
        result={}
        for owner in range(4):
            x,y=self.loader.get_client_data(owner,task)
            x,y,xv,yv=_split_local_validation(x,y,.1,42+task*10000+owner)
            result[owner]=SimpleNamespace(X_train=x,y_train=y,X_validation=xv,y_validation=yv)
        return result

    def new(self,saved=None):return original.OriginalTrainService(self.spec,saved)
    def run_round(self,s,r=0,task=0):return s.after_round(task,r,self.models,self.detectors,self.clients,list(self.clients),self.graph)

    def test_original_loader_no_roles_no_test_and_full_current_train_view(self):
        s=self.new();s.bind_clients(0,self.clients,list(self.clients))
        self.assertEqual(s.counts(1,'train'),{0:4500,2:4500})
        entry=dict(task=0,receiver=0,donor=1,target=0)
        engine=s._engine(entry,'coverage');x,y=s._data(1,self.clients,engine)
        self.assertEqual(len(y),9000);self.assertEqual(int((y==0).sum()),4500)
        self.assertGreater(int((y==0).sum()),4096)
        self.assertTrue(set(y.tolist())<={0,2})
        self.assertFalse(list(self.path.rglob('role*')))
        with self.assertRaises(ValueError):s.role(1,0,'calibration')
        with self.assertRaises(ValueError):s.counts(1,'fit')
        self.assertFalse((self.data/'global_test_data.npz').exists())

    def test_single_bank_actual_gradients_native_rng_isolation_resume(self):
        s=self.new();before=native_fingerprint(self.models,self.detectors);rng=torch.random.get_rng_state().clone()
        self.run_round(s)
        self.assertEqual(tuple(s.engines),('coverage',));self.assertGreater(len(s.entries),1)
        self.assertEqual(before,native_fingerprint(self.models,self.detectors));self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        self.assertEqual(tuple(s.engines['coverage'][0,0].branches),('joint_class_balanced',))
        self.assertGreater(sum(t['extra_negative_gradient_norm'] for t in s.engines['coverage'][0,0].trace),0)
        restored=self.new(s.state_dict());self.run_round(s,1);self.run_round(restored,1)
        for key in s.entries:self.assertEqual(digest(s.engines['coverage'][key].state_dict()),digest(restored.engines['coverage'][key].state_dict()))
        self.assertEqual(s.bytes,restored.bytes)

    def test_fixed_gate_native_validation_only_no_train_overlap(self):
        s=self.new();self.run_round(s)
        # Deliberately contaminate one validation row; it must not certify anything.
        self.clients[1].X_validation[0]=self.clients[0].X_train[0]
        contaminated=self.clients[1].X_validation[0].clone()
        s.prepare_validation()
        self.assertFalse(bool((s.eligible_validation[1][0]==contaminated).all(1).any()))
        def scores(engines,classes,x):
            return np.column_stack([np.where(x[:,0].numpy()==c,.99,.01) for c in classes]) if classes else np.empty((len(x),0))
        with patch.object(original,'score_matrix',scores),patch.object(original,'base_predictions',side_effect=lambda m,d,x,seen:np.full(len(x),2,dtype=np.int64)),patch('fed_learning.training.candle_v4_service.base_predictions',side_effect=lambda m,d,x,seen:np.full(len(x),2,dtype=np.int64)):
            # The deployed bank uses the same fixed synthetic scores during equality/readout.
            with patch('fed_learning.training.candle_v4_service.score_matrix',scores):
                s.finish_task(0,self.models,self.detectors,self.clients,list(self.clients),self.graph)
        self.assertEqual(s.status['coverage'][0,0],'ACTIVE')
        self.assertEqual(s.thresholds['coverage'][0,0],.5)
        self.assertTrue(all(not r.get('CAL_opened',False) for r in s.acceptances))
        self.assertFalse(any('roles' in str(p) for p in self.path.rglob('*')))

    def test_absent_evidence_defers_and_task_transition_never_replays(self):
        s=self.new();self.run_round(s)
        for client in self.clients.values():
            client.X_validation=client.X_validation[:0];client.y_validation=client.y_validation[:0]
        with patch('fed_learning.training.candle_v4_service.base_predictions',side_effect=lambda m,d,x,seen:np.full(len(x),2,dtype=np.int64)):
            s.finish_task(0,self.models,self.detectors,self.clients,list(self.clients),self.graph)
        self.assertNotIn('ACTIVE',s.status['coverage'].values())
        before={k:digest(e.branches['joint_class_balanced'][0].state_dict()) for k,e in s.engines['coverage'].items()}
        with self.assertRaises(ValueError):s.bind_clients(1,self.clients,list(self.clients))
        self.clients=self.load_task(1);self.run_round(s,0,1)
        for key,value in before.items():self.assertEqual(value,digest(s.engines['coverage'][key].branches['joint_class_balanced'][0].state_dict()))
        self.assertTrue(all(set(client.y_train.tolist())=={6} for client in self.clients.values()))

    def test_native_only_variant_and_old_checkpoint_rejected(self):
        spec=deepcopy(self.spec);spec['arms']=[]
        s=original.OriginalTrainService(spec);self.run_round(s)
        self.assertEqual(s.entries,{});self.assertEqual(s.engines,{})
        legacy=deepcopy(s.state_dict());legacy['version']='candle_v4_native_coverage_full_dev_v1'
        with self.assertRaises(ValueError):original.OriginalTrainService(spec,legacy)
        launcher=(ROOT/'candle_v4_runtime/tools/train_candle_v4.py').read_text(encoding='utf-8')
        for forbidden in ('prepare_denice_clean_roles','CleanRoleData','--roles','prepare(str(data)'):
            self.assertNotIn(forbidden,launcher)
        self.assertIn('denice_clean_roles_dir=None',launcher)

    def test_no_disjoint_validation_reports_unknown_without_stopping_training(self):
        s=self.new();self.run_round(s)
        for client in self.clients.values():
            client.X_validation=client.X_train[:40].clone();client.y_validation=client.y_train[:40].clone()
        s.finish_task(0,self.models,self.detectors,self.clients,list(self.clients),self.graph)
        self.assertEqual(s.evaluations[-1]['rows'],0)
        self.assertIsNone(s.evaluations[-1]['metrics']['coverage']['accuracy'])
        self.assertNotIn('ACTIVE',s.status['coverage'].values())

    def test_config_rejects_external_roles_pilot_caps_and_early_test(self):
        config=json.loads((ROOT/'candle_v4_runtime/configs/candle_v1.json').read_text(encoding='utf-8'))
        config.update(denice_clean_roles_dir=None,candle_v4_original_train=True,candle_v4_full_service={'version':original.VERSION})
        original.validate_original_config(config)
        for key,value in (('denice_clean_roles_dir','FORBIDDEN'),('denice_post_task_eval',True),
                          ('denice_eval_terminal_state_only',False),('denice_max_train_samples_per_client',32)):
            with self.subTest(key=key),self.assertRaises(ValueError):
                original.validate_original_config(dict(config,**{key:value}))
        with self.assertRaises(ValueError):original.OriginalTrainService(dict(self.spec,fixed_threshold=.7))

    def test_current_validation_metrics_do_not_oracle_mask_inference(self):
        s=self.new();self.clients=self.load_task(1);s.bind_clients(1,self.clients,list(self.clients));s.prepare_validation()
        seen=[]
        def predict(model,detector,x,classes):
            seen.append(classes);return np.full(len(x),2,dtype=np.int64)
        with patch('fed_learning.training.candle_v4_service.base_predictions',predict):
            s._evaluate(1,self.models,self.detectors)
        self.assertTrue(all(classes==list(range(12)) for classes in seen))
        self.assertEqual(s.evaluations[-1]['metric_labels'],list(range(6,12)))
        self.assertEqual(s.evaluations[-1]['metrics']['coverage']['accuracy'],0.)

    def small_validation(self,s):
        for client in self.clients.values():
            indices=torch.cat([torch.nonzero(client.y_validation==c).flatten()[:10] for c in torch.unique(client.y_validation)])
            client.X_validation=client.X_validation[indices];client.y_validation=client.y_validation[indices]
        s.prepare_validation()

    def controlled_gate(self,s,mode='correct'):
        def scores(engines,classes,x):
            values=np.column_stack([np.where(x[:,0].numpy()==c,.99,.01) for c in classes])
            if mode=='miss' and 0 in classes:
                rows=np.flatnonzero(x[:,0].numpy()==0)
                if len(rows):values[rows[0],classes.index(0)]=.01
            if mode=='fp' and 0 in classes:
                rows=np.flatnonzero(x[:,0].numpy()==2)
                if len(rows):values[rows[0],classes.index(0)]=.99
            return values
        with patch.object(original,'score_matrix',scores),patch.object(original,'base_predictions',side_effect=lambda m,d,x,seen:np.full(len(x),2,dtype=np.int64)):
            return s._gate(0,'coverage',0,self.models,self.detectors,list(self.clients),self.graph)

    def test_ten_positives_pass_and_empty_owner_is_unknown_not_zero_far(self):
        s=self.new();self.run_round(s);self.small_validation(s)
        s.eligible_validation[3]=(torch.empty(0,39),np.empty(0,dtype=np.int64))
        receipt=self.controlled_gate(s)
        self.assertEqual(receipt['outcome'],'PASS');self.assertEqual(s.status['coverage'][0,0],'ACTIVE')
        self.assertEqual(receipt['owners'][1]['targets'][0]['positive'],10)
        self.assertEqual(receipt['owners'][1]['targets'][0]['hits'],10)
        self.assertIsNone(receipt['owners'][3]['targets'][0]['observed_far'])
        self.assertEqual(receipt['owners'][3]['targets'][0]['negative_evidence_status'],'UNKNOWN_NO_SAMPLES')
        self.assertFalse(s.certificates['coverage'][0,0]['statistical_safety_certificate'])
        self.assertIn(5,s.certificates['coverage'][0,0]['unobserved_negative_classes'])

    def test_small_panel_recall_and_false_activation_still_reject(self):
        s=self.new();self.run_round(s);self.small_validation(s)
        receipt=self.controlled_gate(s,'miss')
        self.assertEqual(receipt['outcome'],'REJECT')
        self.assertEqual(receipt['owners'][1]['targets'][0]['hits'],9)
        self.assertIn('CLASS_0_RECALL',receipt['failures'])
        # A separate fixed fixture, not a retry on evidence after a decision.
        self.spec['output_dir']=str(self.path/'fp_fixture')
        s=self.new();self.run_round(s);s.prepare_validation()
        receipt=self.controlled_gate(s,'fp')
        self.assertEqual(receipt['outcome'],'REJECT')
        self.assertTrue(any(reason.endswith('_FAR') for reason in receipt['failures']))
        self.assertGreater(receipt['break_count'],0)

    def test_zero_positive_or_negative_defers_and_v1_resume_is_rejected(self):
        s=self.new();self.run_round(s);self.small_validation(s)
        x,y=s.eligible_validation[1];keep=y!=0;s.eligible_validation[1]=(x[keep],y[keep])
        self.assertEqual(self.controlled_gate(s)['outcome'],'DEFERRED')
        self.spec['output_dir']=str(self.path/'zero_negative_fixture')
        s=self.new();self.run_round(s)
        source=self.clients[1];keep=source.y_validation==0
        x=source.X_validation[keep];y=source.y_validation[keep].numpy()
        s.eligible_validation={o:(x,y) for o in self.clients}
        receipt=self.controlled_gate(s)
        self.assertIn('CLASS_0_NO_NEGATIVE_VALIDATION',receipt['failures'])
        self.assertEqual(receipt['outcome'],'DEFERRED')
        old=s.state_dict();old['version']='candle_v4_original_train_v1'
        with self.assertRaises(ValueError):self.new(old)

    def test_donor_train_minimum_still_32(self):
        x=self.clients[1].X_train;y=self.clients[1].y_train
        def set_positive(n):
            keep=torch.cat((torch.nonzero(y==0).flatten()[:n],torch.nonzero(y!=0).flatten()))
            self.clients[1].X_train=x[keep];self.clients[1].y_train=y[keep]
        set_positive(31);s=self.new();s.bind_clients(0,self.clients,list(self.clients));s.boundary=[0,0]
        s._discover(0,self.clients,list(self.clients),self.graph)
        self.assertNotIn((0,0),s.entries)
        set_positive(32);s.count_cache.clear();s._discover(0,self.clients,list(self.clients),self.graph)
        self.assertEqual(s.entries[0,0]['donor'],1)

    def test_inactive_receiver_is_deferred_without_missing_pool_error(self):
        s=self.new();self.run_round(s);self.small_validation(s)
        receipt=s._gate(0,'coverage',0,self.models,self.detectors,[1,2,3],self.graph)
        self.assertEqual(receipt['outcome'],'DEFERRED')
        self.assertIn('INACTIVE_RECEIVER',receipt['failures'])


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(OriginalTrainTests))
    receipt=dict(protocol_version=original.VERSION,tests=result.testsRun,passed=result.wasSuccessful(),errors=len(result.errors),failures=len(result.failures),
                 fixture='synthetic original TRAIN NPZ only',roles_created=False,real_TEST_opened=False,real_CAL_opened=False,full_training_started=False)
    (ROOT/'artifacts/candle_v4_original_train_regression_20261023.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    sys.exit(not result.wasSuccessful())
