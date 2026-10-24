"""Full-TRAIN provisional activation regression; synthetic NPZs only."""
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
from fed_learning.training import decentralized_denice_il as native
from fed_learning.training import candle_v4_original_train as original
from fed_learning.training.candle_v4_shadow import digest,native_fingerprint


class TrainOnlyClient:
    def __init__(self,x,y):self.X_train=x;self.y_train=y
    @property
    def X_validation(self):raise AssertionError('Validation must never be read')
    @property
    def y_validation(self):raise AssertionError('Validation must never be read')


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
            activation_policy='training_only_unverified',per_expert_bytes=128*1024**2,per_arm_bytes=64*1024**3)
        self.loader=IncrementalDataLoader(str(self.data));self.clients=self.load_task(0)
        self.models={o:torch.nn.Linear(39,34) for o in self.clients};self.detectors={o:None for o in self.clients}
        self.graph=dict(alpha_debug={o:dict(group_ids=list(range(4)),alphas=[.25]*4) for o in range(4)})

    def load_task(self,task):
        result={}
        for owner in range(4):
            x,y=self.loader.get_client_data(owner,task)
            xt,yt,xv,yv=native._split_local_validation(x,y,0.0,42+task*10000+owner)
            self.assertEqual(len(yt),len(y));self.assertEqual(len(yv),0)
            result[owner]=TrainOnlyClient(xt,yt)
        return result

    def new(self,saved=None):return original.OriginalTrainService(self.spec,saved)
    def run_round(self,s,r=0,task=0,graph=None):
        return s.after_round(task,r,self.models,self.detectors,self.clients,list(self.clients),graph or self.graph)
    def finish(self,s,task=0):return s.finish_task(task,self.models,self.detectors,self.clients,list(self.clients),self.graph)

    def test_all_original_current_samples_no_validation_or_roles(self):
        s=self.new();s.bind_clients(0,self.clients,list(self.clients))
        self.assertEqual(s.counts(1,'train'),{0:5000,2:5000})
        e=s._engine(dict(task=0,receiver=0,donor=1,target=0),'coverage')
        self.assertEqual(len(s._data(1,self.clients,e)[1]),10000)
        self.assertFalse(list(self.path.rglob('role*')))
        with self.assertRaises(ValueError):s.role(1,0,'calibration')
        with self.assertRaises(ValueError):s.counts(1,'native_validation')
        self.assertFalse((self.data/'global_test_data.npz').exists())

    def test_real_updates_activate_unverified_without_quality_data(self):
        s=self.new();before=native_fingerprint(self.models,self.detectors);rng=torch.random.get_rng_state().clone()
        with patch('fed_learning.training.candle_v4_service.FullCoverageService._gate',side_effect=AssertionError('No acceptance gate')):
            self.run_round(s)
        self.assertEqual(s.status['coverage'][0,0],'ACTIVE')
        self.assertNotIn('_gate',original.OriginalTrainService.__dict__)
        self.assertNotIn('prepare_validation',original.OriginalTrainService.__dict__)
        self.assertEqual(tuple(s.engines),('coverage',));self.assertEqual(s.engines['coverage'][0,0].steps,44)
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        with patch('fed_learning.training.candle_v4_service.base_predictions',side_effect=AssertionError('No quality prediction')),patch('fed_learning.training.candle_v4_service.FullCoverageService._gate',side_effect=AssertionError('No acceptance gate')):
            self.finish(s)
        self.assertEqual(before,native_fingerprint(self.models,self.detectors))
        self.assertEqual(s.status['coverage'][0,0],'ACTIVE')
        cert=s.certificates['coverage'][0,0]
        self.assertEqual(cert['quality_status'],'UNVERIFIED');self.assertEqual(cert['acceptance_type'],'training_only_unverified')
        self.assertEqual(cert['threshold'],.5);self.assertIsNone(cert['far']);self.assertIsNone(cert['recall'])
        self.assertFalse(cert['statistical_safety_certificate']);self.assertEqual(s.evaluations[-1]['status'],'NOT_MEASURED')
        report=json.loads((Path(self.spec['output_dir'])/'candle_v4_development_report.json').read_text())
        self.assertEqual(report['local_validation_fraction'],0.0);self.assertTrue(report['activation_receipts'])

    def test_no_updates_bad_provenance_and_nonfinite_defer(self):
        s=self.new();s.bind_clients(0,self.clients,list(self.clients));s.boundary=[0,0]
        s._discover(0,self.clients,list(self.clients),self.graph);self.finish(s)
        self.assertNotIn('ACTIVE',s.status['coverage'].values())
        self.spec['output_dir']=str(self.path/'invalid');s=self.new();self.run_round(s)
        for row in s.history:
            for item in row['experts']:
                if item['receiver']==0 and item['target']==0:item['eligible']=False
        with torch.no_grad():next(s.engines['coverage'][0,1].branches['joint_class_balanced'][0].parameters()).fill_(float('nan'))
        self.finish(s)
        self.assertEqual(s.status['coverage'][0,0],'DEFERRED');self.assertEqual(s.status['coverage'][0,1],'DEFERRED')

    def test_round_optimizer_and_active_checkpoint_resume(self):
        s=self.new();self.run_round(s);path=self.path/'state.pt';torch.save(s.state_dict(),path)
        restored=self.new(torch.load(path,weights_only=False));self.run_round(s,1);self.run_round(restored,1)
        for k in s.entries:self.assertEqual(digest(s.engines['coverage'][k].state_dict()),digest(restored.engines['coverage'][k].state_dict()))
        self.assertEqual(s.bytes,restored.bytes);self.finish(s);restored=self.new(s.state_dict())
        self.assertEqual(s.status,restored.status);self.assertEqual(s.certificates,restored.certificates)
        for version in ('candle_v4_original_train_v1','candle_v4_original_train_empirical_gate_v2','candle_v4_full_train_provisional_v3','candle_v4_direct_prediction_fusion_v4','candle_v4_predicted_task_scoped_v5'):
            saved=s.state_dict();saved['version']=version
            with self.assertRaises(ValueError):self.new(saved)

    def test_old_active_experts_stay_frozen_and_available(self):
        s=self.new();self.run_round(s);self.finish(s)
        hashes={k:digest(e.branches['joint_class_balanced'][0].state_dict()) for k,e in s.engines['coverage'].items()}
        s.begin_task(1);self.assertTrue(all(v=='ACTIVE' for v in s.status['coverage'].values()))
        self.clients=self.load_task(1);self.run_round(s,0,1);self.finish(s,1)
        for k,sha in hashes.items():
            self.assertEqual(sha,digest(s.engines['coverage'][k].branches['joint_class_balanced'][0].state_dict()))
            self.assertEqual(s.status['coverage'][k],'ACTIVE');self.assertEqual(s.engines['coverage'][k].steps,44)

    def test_provisional_inference_and_native_fallback(self):
        s=self.new();self.run_round(s)  # Inference works before task-end callbacks.
        with patch('fed_learning.training.candle_v4_service.score_matrix',return_value=np.array([[.9,.1],[.1,.1],[.1,.9]])):
            self.assertEqual(s.infer(0,'coverage',np.array([2,2,2]),torch.zeros(3,39),predicted_tasks=np.zeros(3,dtype=np.int64))[0].tolist(),[0,2,1])
        s.certificates['coverage'][0,0]['quality_status']='VERIFIED'
        with self.assertRaises(ValueError):s.infer(0,'coverage',np.array([2]),torch.zeros(1,39),predicted_tasks=np.zeros(1,dtype=np.int64))

    def test_receipt_refreshes_as_active_expert_keeps_learning(self):
        s=self.new();self.run_round(s);old=s.certificates['coverage'][0,0]['sha256']
        with patch('builtins.print') as output:
            self.run_round(s,1)
        self.assertFalse(any('CANDLE V4 T' in str(c) for c in output.call_args_list))
        cert=s.certificates['coverage'][0,0]
        self.assertNotEqual(old,cert['sha256']);self.assertEqual(cert['activation_round'],1)
        self.assertEqual(cert['training_proof']['updates'],88)
        self.assertEqual(cert['sha256'],digest(s.engines['coverage'][0,0].branches['joint_class_balanced'][0].state_dict()))
        s.infer(0,'coverage',np.array([2]),torch.zeros(1,39),predicted_tasks=np.zeros(1,dtype=np.int64))

    def test_scope_blocks_high_score_and_requires_real_routes(self):
        s=self.new();self.run_round(s)
        with patch('fed_learning.training.candle_v4_service.score_matrix',return_value=np.array([[1.,.99],[.9,.1]])):
            before=digest(s.state_dict())
            p,n=s.infer(0,'coverage',np.array([7,2]),torch.zeros(2,39),predicted_tasks=np.array([1,0]))
            self.assertEqual(p.tolist(),[7,0]);self.assertEqual(n.tolist(),[0,1])
            self.assertEqual(before,digest(s.state_dict()))
            with self.assertRaises(ValueError):s.infer(0,'coverage',np.array([7,2]),torch.zeros(2,39))
        s.entries[0,0]['task']=1
        with self.assertRaises(ValueError):s.infer(0,'coverage',np.array([2]),torch.zeros(1,39),np.array([0]))

    def test_scope_arbitration_invalid_routes_and_empty_bank(self):
        from fed_learning.training.candle_v4_bank import fuse_scoped,base_predictions
        native=np.array([2,8,14,3]);scores=np.array([[.9,.9,.99],[1.,.6,1.],[1.,1.,1.],[.1,.2,.9]])
        args=(native,scores,[.5]*3,[0,1,6],np.array([0,0,1]))
        p,n=fuse_scoped(*args,np.array([0,1,2,0]))
        self.assertEqual(p.tolist(),[0,6,14,3]);self.assertEqual(n.tolist(),[2,1,0,0])
        p,n=fuse_scoped(*args,np.full(4,-1,dtype=np.int64))
        self.assertTrue(np.array_equal(p,native));self.assertFalse(n.any())
        for routes in (np.array([0.]*4),np.array([-2,0,1,2]),np.array([0,1,2,6]),np.array([0])):
            with self.assertRaises(ValueError):fuse_scoped(*args,routes)
        s=self.new();p,n=s.infer(0,'coverage',native,torch.zeros(4,39))
        self.assertTrue(np.array_equal(p,native));self.assertFalse(n.any())
        model=torch.nn.Linear(39,34);model.train()
        logits=torch.zeros(4,34);logits[:,2]=1
        with patch('fed_learning.training.denice_eval._denice_routed_logits_with_episodes',return_value=(logits,np.array([0,1,2,3]))) as forward:
            p,routes=base_predictions(model,None,torch.zeros(4,39),list(range(34)),return_episodes=True)
        self.assertEqual(forward.call_count,1);self.assertTrue(model.training)
        self.assertEqual(p.tolist(),[2]*4);self.assertEqual(routes.tolist(),[0,1,2,3])

    def test_causal_graph_and_native_only_baseline(self):
        s=self.new();self.run_round(s)
        graph=dict(alpha_debug={o:dict(group_ids=[o],alphas=[1.]) for o in self.clients})
        self.run_round(s,1,graph=graph);self.assertEqual(s.engines['coverage'][0,0].steps,44)
        s.finish_task(0,self.models,self.detectors,self.clients,list(self.clients),graph)
        self.assertEqual(s.status['coverage'][0,0],'ACTIVE')
        self.spec['arms']=[];self.spec['output_dir']=str(self.path/'v1');s=self.new();self.run_round(s);self.finish(s)
        self.assertEqual(s.engines,{});self.assertEqual(s.entries,{})
        self.assertEqual(set(s.evaluations[0]['metrics']),{'V1'})

    def test_zero_validation_config_and_native_capsule_fallback(self):
        config=json.loads((ROOT/'candle_v4_runtime/configs/candle_v1.json').read_text())
        self.assertEqual(config['denice_validation_fraction'],0.0);self.assertIsNone(config['denice_clean_roles_dir'])
        config.update(candle_v4_original_train=True,candle_v4_full_service={'version':original.VERSION})
        original.validate_original_config(config)
        for key,value in (('denice_validation_fraction',.1),('denice_clean_roles_dir','FORBIDDEN'),('denice_post_task_eval',True)):
            with self.assertRaises(ValueError):original.validate_original_config(dict(config,**{key:value}))
        c=self.clients[0];ref=c.X_train[:8];labels=c.y_train[:8]
        client=SimpleNamespace(X_train=c.X_train,y_train=c.y_train,y_validation=torch.empty(0,dtype=torch.long))
        with patch.object(native,'build_context_capsule',return_value='train capsule') as build:
            result=native._build_round_capsule(cid=0,task_id=0,round_id=0,model=self.models[0],client=client,
                context_detector=SimpleNamespace(binarize_thresholds=None),ref_data=ref,ref_labels=labels,loss=.1,config=config)
        self.assertEqual(result,'train capsule');self.assertTrue(torch.equal(build.call_args.args[1],ref))
        self.assertEqual(build.call_args.kwargs['sample_count'],5000)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(OriginalTrainTests))
    receipt=dict(protocol_version=original.VERSION,tests=result.testsRun,passed=result.wasSuccessful(),
        errors=len(result.errors),failures=len(result.failures),fixture='synthetic full current-task TRAIN NPZ only',
        validation_read=False,roles_created=False,real_TEST_opened=False,full_training_started=False)
    (ROOT/'artifacts/candle_v4_handoff_original_regression_20261024.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    sys.exit(not result.wasSuccessful())
