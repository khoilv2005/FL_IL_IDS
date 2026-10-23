"""Controlled technical regression. Synthetic roles only; never opens real CAL."""
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
from fed_learning.training import candle_v4_service as service
from fed_learning.training.candle_v4_shadow import digest


class Authority:
    def __init__(self):
        self.calls=[];self.arrays={};self.manifest=dict(content_disjoint=True,clients={})
        for owner,classes in enumerate(((2,),(0,2),(1,2),(2,))):
            counts={str(c):64 for c in classes}
            self.manifest['clients'][str(owner)]=dict(role_class_counts={r:counts.copy() for r in ('base','fit','calibration','validation')})
            for index,role in enumerate(('base','fit','calibration','validation')):
                y=np.repeat(classes,64);x=np.zeros((len(y),39),np.float32);x[:,0]=y
                x[:,1]=np.arange(len(y))+index*1000+owner*10000
                x[:,2:]=np.random.default_rng(owner*4+index).normal(size=(len(y),37))
                self.arrays[owner,role]=(x,y,np.arange(len(y)))

    def client_role(self,owner,role):
        self.calls.append((owner,role));return self.arrays[owner,role]


class FullServiceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.authority=Authority()
        self.spec=dict(version=service.VERSION,seed=42,input_sha256='synthetic-content-authority',
            roles_dir='SYNTHETIC',data_dir='SYNTHETIC',output_dir=self.temp.name,
            per_expert_bytes=128*1024**2,per_arm_bytes=64*1024**3)
        self.clients={o:SimpleNamespace(X_train=torch.from_numpy(self.authority.arrays[o,'base'][0]),
                                       y_train=torch.from_numpy(self.authority.arrays[o,'base'][1])) for o in range(4)}
        self.models={o:torch.nn.Linear(39,3) for o in range(4)};self.detectors={o:None for o in range(4)}
        self.graph=dict(alpha_debug={o:dict(group_ids=list(range(4)),alphas=[.25]*4) for o in range(4)})
        self.addCleanup(patch.stopall)

    def new(self,saved=None):return service.FullCoverageService(self.spec,saved,self.authority)
    def run_round(self,s,round=0,graph=None):
        return s.after_round(0,round,self.models,self.detectors,self.clients,[0,1,2,3],graph or self.graph)

    def test_actual_gradient_coverage_and_native_isolation(self):
        before=digest({k:m.state_dict() for k,m in self.models.items()});rng=torch.random.get_rng_state().clone()
        torch.set_num_threads(2)
        s=self.new();self.run_round(s)
        self.assertEqual(torch.get_num_threads(),2)
        torch.set_num_threads(1)
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        self.assertEqual(before,digest({k:m.state_dict() for k,m in self.models.items()}))
        self.assertGreaterEqual(len({k[0] for k in s.entries}),2)
        e=s.engines['coverage'][0,0];n=s.engines['no_aux'][0,0]
        self.assertEqual(e.steps,44);self.assertEqual(n.steps,e.steps)
        self.assertGreater(sum(r['extra_negative_samples'] for r in e.trace),0)
        self.assertGreater(sum(r['extra_negative_gradient_norm'] for r in e.trace),0)
        self.assertNotEqual(digest(e.branches['joint_class_balanced'][0].state_dict()),digest(n.branches['joint_class_balanced'][0].state_dict()))
        self.assertFalse(any(r=='calibration' for _,r in self.authority.calls))

    def test_checkpoint_resume_exact_with_actual_adam(self):
        s=self.new();self.run_round(s);saved=s.state_dict()
        path=Path(self.temp.name)/'state.pt';torch.save(saved,path)
        restored=self.new(torch.load(path,weights_only=False))
        self.run_round(s,1);self.run_round(restored,1)
        for arm in service.ARMS:
            for key in s.entries:
                self.assertEqual(digest(s.engines[arm][key].state_dict()),digest(restored.engines[arm][key].state_dict()))
        self.assertEqual(s.bytes,restored.bytes)
        with self.assertRaises(ValueError):self.run_round(restored,1)
        changed=deepcopy(self.spec);changed['seed']=43
        with self.assertRaises(ValueError):service.FullCoverageService(changed,saved,self.authority)

    def test_current_graph_blocks_and_catchup(self):
        s=self.new();self.run_round(s)
        graph=dict(alpha_debug={o:dict(group_ids=[o],alphas=[1.]) for o in range(4)})
        self.run_round(s,1,graph)
        self.assertEqual(s.engines['coverage'][0,0].steps,44)
        self.run_round(s,2)
        self.assertEqual(s.engines['coverage'][0,0].steps,132)

    def test_budget_matched_all_arms_and_no_partial_update(self):
        s=self.new();self.run_round(s);before={a:digest(s.engines[a][0,0].state_dict()['branches']) for a in service.ARMS}
        s.spec['per_expert_bytes']=1
        self.run_round(s,1)
        for a in service.ARMS:self.assertEqual(before[a],digest(s.engines[a][0,0].state_dict()['branches']))

    def test_controlled_active_arbitration_transition_and_restore(self):
        s=self.new();self.run_round(s)
        def scores(engines,classes,x):
            return np.column_stack([np.where(x[:,0].numpy()==c,.99,.01) for c in classes]) if classes else np.empty((len(x),0))
        patch.object(service,'score_matrix',scores).start()
        patch.object(service,'base_predictions',lambda model,detector,x,seen:np.full(len(x),2,dtype=np.int64)).start()
        s.finish_task(0,self.models,self.detectors,self.clients,list(range(4)),self.graph)
        self.assertEqual(s.status['coverage'][0,0],'ACTIVE');self.assertEqual(s.status['coverage'][0,1],'ACTIVE')
        x=torch.zeros(3,39);x[:,0]=torch.tensor([0,1,2])
        self.assertEqual(s.infer(0,'coverage',np.array([2,2,2]),x)[0].tolist(),[0,1,2])
        restored=self.new(s.state_dict())
        self.assertTrue(np.array_equal(s.infer(0,'coverage',np.array([2]*3),x)[0],restored.infer(0,'coverage',np.array([2]*3),x)[0]))
        before=digest(s.engines['coverage'][0,0].state_dict()['branches'])
        s.begin_task(1)
        self.assertEqual(s.status['coverage'][0,0],'PENDING_NEW_TASK')
        self.assertEqual(before,digest(s.engines['coverage'][0,0].state_dict()['branches']))
        with self.assertRaises(ValueError):s.role(1,0,'calibration')
        self.assertEqual(s.infer(0,'coverage',np.array([2]*3),x)[0].tolist(),[2,2,2])
        s.after_round(1,0,self.models,self.detectors,self.clients,list(range(4)),self.graph)
        s.finish_task(1,self.models,self.detectors,self.clients,list(range(4)),self.graph)
        self.assertEqual(s.status['coverage'][0,0],'SUSPENDED')
        self.assertEqual(before,digest(s.engines['coverage'][0,0].state_dict()['branches']))

    def test_controlled_collision_reject_is_not_training_failure(self):
        s=self.new();self.run_round(s)
        patch.object(service,'score_matrix',lambda engines,classes,x:np.full((len(x),len(classes)),.99)).start()
        patch.object(service,'base_predictions',lambda model,detector,x,seen:np.full(len(x),2,dtype=np.int64)).start()
        s.finish_task(0,self.models,self.detectors,self.clients,list(range(4)),self.graph)
        self.assertEqual(s.status['coverage'][0,0],'REJECTED')
        self.assertTrue(any(r.get('collisions',0)>0 for r in s.acceptances))
        self.assertEqual(s.gated,[0]);s.begin_task(1)
        with self.assertRaises(ValueError):s.finish_task(0,self.models,self.detectors,self.clients,list(range(4)),self.graph)

    def test_actual_nice_model_router_fingerprint_unchanged(self):
        from fed_learning.models.nice_model import NICEModel
        from fed_learning.servers.nice_server import ContextDetector
        self.models={o:NICEModel((39,1),34) for o in range(4)}
        self.detectors={o:ContextDetector() for o in range(4)}
        before=service.native_fingerprint(self.models,self.detectors)
        s=self.new();self.run_round(s)
        self.assertEqual(before,service.native_fingerprint(self.models,self.detectors))
        self.assertNotIn('candle_v4',str(list(self.models[0].state_dict())))

    def test_gate_tamper_and_historical_access_fail_closed(self):
        s=self.new();self.run_round(s)
        patch.object(service,'score_matrix',lambda engines,classes,x:np.column_stack([np.where(x[:,0].numpy()==c,.99,.01) for c in classes]) if classes else np.empty((len(x),0))).start()
        patch.object(service,'base_predictions',lambda model,detector,x,seen:np.full(len(x),2,dtype=np.int64)).start()
        s.finish_task(0,self.models,self.detectors,self.clients,list(range(4)),self.graph)
        s.thresholds['coverage'][0,0]=.5
        with self.assertRaises(ValueError):s.infer(0,'coverage',np.array([2]),torch.zeros(1,39))

    def test_completed_gate_journal_resumes_without_reopening_cal(self):
        s=self.new();self.run_round(s);before_gate=s.state_dict()
        patch.object(service,'score_matrix',lambda engines,classes,x:np.column_stack([np.where(x[:,0].numpy()==c,.99,.01) for c in classes]) if classes else np.empty((len(x),0))).start()
        patch.object(service,'base_predictions',lambda model,detector,x,seen:np.full(len(x),2,dtype=np.int64)).start()
        s.finish_task(0,self.models,self.detectors,self.clients,list(range(4)),self.graph)
        self.authority.calls.clear();restored=self.new(before_gate)
        restored.finish_task(0,self.models,self.detectors,self.clients,list(range(4)),self.graph)
        self.assertFalse(any(role=='calibration' for _,role in self.authority.calls))
        self.assertEqual(s.status,restored.status);self.assertEqual(s.bytes,restored.bytes)

    def test_native_round_hook_and_full_profile(self):
        code=(ROOT/'candle_v4_runtime/fed_learning/training/decentralized_denice_il.py').read_text(encoding='utf-8')
        self.assertIn("if config.get('candle_v4_full_service'):",code)
        self.assertIn('if appliance_service or v4_shadow or',code)
        self.assertIn("state.get('candle_v4_shadow_state')",code)
        launcher=(ROOT/'candle_v4_runtime/tools/train_candle_v4.py').read_text(encoding='utf-8')
        self.assertIn('denice_validation_fraction=0.0',launcher)
        self.assertNotIn('run_legacy_self(',launcher)

    def test_task_archive_contains_expert_continuation_and_locks(self):
        import zipfile
        from fed_learning.training.denice_checkpoint_archive import compress_checkpoint,seal_task_archive
        root=Path(self.temp.name);s=self.new()
        torch.save(dict(candle_v4_shadow_state=s.state_dict()),root/'continuation_state_task_0.pt')
        torch.save(dict(synthetic=True),root/'checkpoint_task_0_round_0.pt')
        (root/'candle_v4_run_lock.json').write_text('{}',encoding='utf-8')
        (root/'v4_threshold_locks').mkdir()
        (root/'v4_threshold_locks/t0_r0_coverage.json').write_text('{}',encoding='utf-8')
        compress_checkpoint(root/'checkpoint_task_0_round_0.pt')
        archive=seal_task_archive(root,0)
        with zipfile.ZipFile(archive) as z:
            self.assertIn('continuation_state_task_0.pt',z.namelist())
            self.assertIn('candle_v4_run_lock.json',z.namelist())
            self.assertIn('v4_threshold_locks/t0_r0_coverage.json',z.namelist())
            state=torch.load(z.open('continuation_state_task_0.pt'),weights_only=False)
            self.assertEqual(state['candle_v4_shadow_state']['version'],service.VERSION)
        self.assertTrue((root/'candle_v4_run_lock.json').exists())


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(FullServiceTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    receipt=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                 real_CAL_opened=False,full_training_started=False,fixture='synthetic-only',
                 passed=result.wasSuccessful())
    path=ROOT/'artifacts/candle_v4_full_integration_regression_20261023.json'
    path.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    sys.exit(not result.wasSuccessful())
