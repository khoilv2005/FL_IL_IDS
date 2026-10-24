"""V4-only continuous router: synthetic TRAIN, native V1 paths unchanged."""
from copy import deepcopy
import json
from pathlib import Path
import sys,tempfile,unittest
import numpy as np
import torch
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'candle_v4_runtime'))
from fed_learning.servers.nice_server import ContextDetector
from fed_learning.servers.continuous_class_router import ContinuousClassMemory
from fed_learning.training.checkpoint_state import snapshot_context_detector,restore_context_detector
from fed_learning.training.local_task_loop import _update_local_nice_context_memory
from fed_learning.training.candle_v4_continuous_runtime import update_memory,predictions,ContinuousRoutingView
from fed_learning.training.candle_v4_original_train import OriginalTrainService,CONTINUOUS_VERSION,VERSION,validate_original_config
from fed_learning.training.candle_v4_shadow import digest


class ActivationModel(torch.nn.Module):
    def __init__(self):super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(()))
    def get_context_activations_per_sample(self,x):
        z=x.reshape(len(x),-1)[:,0:1]
        return {k:z*(i+1) for i,k in enumerate(('conv1','conv2','conv3','gru'))}


class ContinuousTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);self.model=ActivationModel().eval()
    def spec(self):
        return dict(version=CONTINUOUS_VERSION,seed=42,arms=[],router_mode='continuous_class_v1',fixed_threshold=.5,
            activation_policy='training_only_unverified',input_sha256='synthetic',data_dir='synthetic',
            output_dir='synthetic',per_expert_bytes=2**27,per_arm_bytes=2**36)
    def detector(self,cap=5):
        d=ContextDetector(router_mode='multiclass_balanced',router_reference_per_class=cap)
        d.stable_feature_mask=np.array([1,0,1,0],dtype=bool);d.retain_reference_inputs=False
        d.binarize_thresholds={k:.1 for k in ('conv1','conv2','conv3','gru')};return d
    def inputs(self,values):
        x=torch.full((len(values),39),99.);x[:,0]=torch.as_tensor(values,dtype=torch.float32);return x
    def push(self,s,d,values,labels,task):
        return update_memory(s,0,d,self.model,self.inputs(values),torch.tensor(labels),task,sorted(set(labels)),'cpu')
    def fitted(self,cap=5):
        s=OriginalTrainService(self.spec());d=self.detector(cap)
        self.push(s,d,[1.]*10+[3.]*10,[0]*10+[1]*10,0);self.push(s,d,[2.]*10,[6]*10,1);return s,d

    def test_class_identity_amplitude_and_native_binary_bank(self):
        s,d=self.fitted();x=self.inputs([1.,2.,3.]);acts={k:v.numpy() for k,v in self.model.get_context_activations_per_sample(x).items()}
        bits=d.binarize_layer_activations(acts);self.assertTrue(np.array_equal(bits[0],bits[1]));self.assertTrue(np.array_equal(bits[1],bits[2]))
        view=ContinuousRoutingView(d,s.continuous_routers[0]);pred,scores=view.predict_episodes_with_scores(view.binarize_layer_activations(acts))
        self.assertEqual(pred.tolist(),[0,1,0]);self.assertTrue(np.allclose(scores.sum(1),1))
        self.assertEqual(s.continuous_routers[0].class_to_task,{0:0,1:0,6:1});self.assertEqual(d.router_mode,'multiclass_balanced')

    def test_cap_historical_memory_and_no_raw_retention(self):
        s,d=self.fitted(3);m=s.continuous_routers[0]
        self.assertTrue(all(v.shape==(3,2) for v in m.memory.values()));self.assertEqual(m.stats()['feature_bytes'],72)
        self.assertFalse(d.retain_reference_inputs);self.assertEqual(d.reference_input_memory,{})
        self.assertTrue(np.all(m.memory[0][:,0]==1.));self.assertTrue(np.all(m.memory[1][:,0]==3.))
        with self.assertRaises(RuntimeError):ContinuousClassMemory(3).predict(np.zeros((1,2)))

    def test_rejects_old_labels_nonfinite_and_changed_coordinates(self):
        s,d=self.fitted();m=s.continuous_routers[0];before=digest(m.state_dict());acts={k:v.numpy() for k,v in self.model.get_context_activations_per_sample(self.inputs([1.])).items()}
        for task,labels in [(0,[0]),(1,[2]),(6,[33])]:
            with self.assertRaises(ValueError):m.push(acts,np.array(labels),task,d.stable_feature_mask)
            self.assertEqual(before,digest(m.state_dict()))
        bad=deepcopy(acts);bad['conv1'][0,0]=np.nan
        with self.assertRaises(ValueError):m.push(bad,np.array([6]),1,d.stable_feature_mask)
        with self.assertRaises(ValueError):m.push(acts,np.array([6]),1,np.array([0,1,1,0]))
        self.assertEqual(before,digest(m.state_dict()))
        native_before=digest(snapshot_context_detector(d))
        with self.assertRaises(ValueError):self.push(s,d,[1.],[0],0)
        self.assertEqual(native_before,digest(snapshot_context_detector(d)))

    def test_service_checkpoint_restore_exact_and_no_alias(self):
        s,d=self.fitted();acts={k:v.numpy() for k,v in self.model.get_context_activations_per_sample(self.inputs([1.,2.,3.])).items()}
        view=ContinuousRoutingView(d,s.continuous_routers[0]);expected=view.predict_episodes_with_scores(view.binarize_layer_activations(acts))
        state=s.state_dict()
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'state.pt';torch.save(state,path);state=torch.load(path,weights_only=False)
        restored=OriginalTrainService(s.spec,state);view=ContinuousRoutingView(d,restored.continuous_routers[0]);actual=view.predict_episodes_with_scores(view.binarize_layer_activations(acts))
        self.assertTrue(all(np.array_equal(a,b) for a,b in zip(expected,actual)))
        before=digest(s.state_dict());state['continuous_routers'][0]['memory'][0][0,0]=999.
        self.assertEqual(before,digest(s.state_dict()));self.assertNotIn('continuous_class_state',snapshot_context_detector(d))

    def test_binary_conversion_and_corrupted_state_rejected(self):
        s,d=self.fitted();state=s.state_dict()
        for corrupt in ['version','coef','mask']:
            saved=deepcopy(state)
            if corrupt=='version':saved['continuous_routers'][0]['version']='bad'
            elif corrupt=='coef':saved['continuous_routers'][0]['classifier']['coef_'][0,0]=np.nan
            else:saved['continuous_routers'][0]['indices']=np.array([0,0])
            with self.assertRaises(ValueError):OriginalTrainService(s.spec,saved)
        del state['continuous_routers']
        with self.assertRaises(ValueError):OriginalTrainService(s.spec,state)
        with self.assertRaises(ValueError):predictions(OriginalTrainService(self.spec()),0,self.model,d,self.inputs([1.]),list(range(12)))

    def test_native_sampler_rng_graph_capsules_and_memory_unchanged(self):
        from fed_learning.strategies.decentralized.denice_capsule import _context_detector_summary
        ends=[];states=[];capsules=[];before=digest(self.model.state_dict())
        for enabled in [False,True]:
            s=OriginalTrainService(self.spec()) if enabled else None;d=self.detector(4);torch.manual_seed(123)
            for task,values,labels in [(0,[1.]*20+[3.]*20,[0]*20+[1]*20),(1,[2.]*20,[6]*20)]:
                x=self.inputs(values);y=torch.tensor(labels);classes=sorted(set(labels))
                if enabled:r=update_memory(s,0,d,self.model,x,y,task,classes,'cpu')
                else:r=_update_local_nice_context_memory(d,self.model,x,y,task,classes,'cpu')
                self.assertEqual(r['reference_sample_count'],len(classes)*4)
            ends.append(torch.get_rng_state().clone());states.append(snapshot_context_detector(d));capsules.append(_context_detector_summary(d))
        self.assertTrue(torch.equal(*ends));self.assertEqual(digest(states[0]),digest(states[1]));self.assertEqual(capsules[0],capsules[1]);self.assertEqual(before,digest(self.model.state_dict()))

    def test_real_encoder_two_tasks_inference_and_restore(self):
        from fed_learning.models.denice_model import DeNICEModel
        model=DeNICEModel((39,1),34).eval();d=ContextDetector(router_mode='multiclass_balanced',router_reference_per_class=3);d.retain_reference_inputs=False
        d.stable_feature_mask=np.zeros(548,dtype=bool);d.stable_feature_mask[:16]=True;d.stable_feature_mask[-16:]=True
        s=OriginalTrainService(self.spec());before=digest(model.state_dict());old=None
        for task,classes in [(0,[0,1]),(1,[6,7])]:
            x=torch.randn(12,39,1);y=torch.tensor([classes[0]]*6+[classes[1]]*6)
            update_memory(s,0,d,model,x,y,task,classes,'cpu')
            if task==0:old=digest(s.continuous_routers[0].memory)
        self.assertEqual(old,digest({c:v for c,v in s.continuous_routers[0].memory.items() if c<6}));self.assertEqual(before,digest(model.state_dict()))
        restored=OriginalTrainService(s.spec,s.state_dict());native=ContextDetector();restore_context_detector(native,snapshot_context_detector(d))
        a,ep=predictions(s,0,model,d,x[:4],list(range(12)));b,again=predictions(restored,0,model,native,x[:4],list(range(12)))
        self.assertTrue(np.array_equal(a,b));self.assertTrue(np.array_equal(ep,again));self.assertTrue(set(ep)<={0,1})

    def test_variant_config_and_v1_baseline_dispatch(self):
        config=json.loads((ROOT/'candle_v4_runtime/configs/candle_v1.json').read_text());spec=self.spec()
        config.update(candle_v4_original_train=True,candle_v4_full_service=spec,denice_router_mode='multiclass_balanced');validate_original_config(config)
        with self.assertRaises(ValueError):validate_original_config(dict(config,denice_router_replay_enabled=True))
        s=OriginalTrainService(spec);state=s.state_dict();state['version']=VERSION
        with self.assertRaises(ValueError):OriginalTrainService(spec,state)
        with self.assertRaises(ValueError):validate_original_config(dict(config,candle_v4_full_service=dict(spec,version=VERSION)))
        legacy=OriginalTrainService(dict(spec,version=VERSION,router_mode='multiclass_balanced'));d=self.detector()
        update_memory(legacy,0,d,self.model,self.inputs([1.]*10),torch.zeros(10,dtype=torch.long),0,[0],'cpu')
        self.assertEqual(legacy.continuous_routers,{});self.assertNotIn('continuous_routers',legacy.state_dict())

    def test_terminal_keeps_v1_control_and_same_test_rows(self):
        from types import SimpleNamespace
        from fed_learning.training import candle_v4_terminal_test as terminal
        with tempfile.TemporaryDirectory() as td:
            spec=dict(self.spec(),output_dir=td,arms=['coverage']);s=OriginalTrainService(spec)
            s.boundary=[5,19];s.gated=list(range(6));s.evaluations=[dict(task=t) for t in range(6)]
            x=torch.zeros(102,39);y=torch.arange(34).repeat(3)
            loader=SimpleNamespace(get_full_test_data=lambda:(x,y));models={0:torch.nn.Linear(39,34)}
            def old(m,d,x,seen,return_episodes=False):return np.full(len(x),2),np.zeros(len(x),dtype=np.int64)
            def new(s,c,m,d,x,seen):return np.full(len(x),8),np.ones(len(x),dtype=np.int64)
            with patch.object(terminal,'IncrementalDataLoader',return_value=loader),patch.object(terminal,'base_predictions',side_effect=old),patch('fed_learning.training.candle_v4_continuous_runtime.predictions',side_effect=new):
                r=terminal.evaluate_terminal(s,models,{0:None},5,19)
            self.assertEqual(r['evaluated_rows'],102);self.assertTrue(r['partition']['all_rows_assigned_once'])
            self.assertEqual(r['metrics']['V1']['confusion_matrix'][8][2],3)
            self.assertEqual(r['metrics']['continuous_native']['confusion_matrix'][8][8],3)
            self.assertEqual(r['metrics']['coverage'],r['metrics']['continuous_native'])
            self.assertEqual(r['rescue']['coverage'],3);self.assertEqual(r['break_count']['coverage'],3)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ContinuousTests))
    (ROOT/'artifacts/candle_v4_continuous_router_regression_20261024.json').write_text(json.dumps(dict(
        version='continuous_class_v1',tests=result.testsRun,passed=result.wasSuccessful(),failures=len(result.failures),errors=len(result.errors),
        fixture='synthetic TRAIN, real encoder, unchanged V1 sampler/graph/control path',real_TEST_opened=False,
        old_raw_data_replayed=False,full_training_started=False),indent=2)+'\n');sys.exit(not result.wasSuccessful())
