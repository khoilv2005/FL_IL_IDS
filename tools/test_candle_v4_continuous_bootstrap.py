"""V4-only exact-clone lifecycle regression. Synthetic inputs, no backbone training."""
from copy import deepcopy
import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'candle_v4_runtime'))
from test_candle_continuous_class_router import ContinuousTests
from fed_learning.training.candle_v4_continuous_runtime import (
    BOOTSTRAP_VERSION,clone_continuous_memory,validate_task_support,update_memory,predictions)
from fed_learning.training.candle_v4_original_train import OriginalTrainService
from fed_learning.training.candle_v4_shadow import digest
from fed_learning.training.checkpoint_state import snapshot_context_detector,restore_context_detector
from fed_learning.servers.nice_server import ContextDetector


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.f=ContinuousTests();self.f.setUp();self.model=self.f.model
        self.model.weight_masks={'synthetic':torch.ones(1)}
        self.s=OriginalTrainService(dict(self.f.spec(),version=BOOTSTRAP_VERSION))
        self.d=self.f.detector()
        self.f.push(self.s,self.d,[1.]*10+[3.]*10,[0]*10+[1]*10,0)

    def clone(self,receiver=1,source=0,task=1,model=None,detector=None):
        model=deepcopy(self.model) if model is None else model
        detector=deepcopy(self.d) if detector is None else detector
        receipt=clone_continuous_memory(self.s,receiver,source,task,self.model,model,self.d,detector)
        return model,detector,receipt

    def test_late_join_has_historical_tasks_and_no_alias(self):
        before=digest(self.s.continuous_routers[0].state_dict())
        model,det,receipt=self.clone()
        self.assertEqual(receipt['tasks'],[0]);self.assertGreater(receipt['serialized_bytes'],0)
        update_memory(self.s,1,det,model,self.f.inputs([2.]*10),torch.tensor([6]*10),1,[6],'cpu')
        self.assertEqual(set(self.s.continuous_routers[1].class_to_task.values()),{0,1})
        self.assertEqual(before,digest(self.s.continuous_routers[0].state_dict()))
        self.s.continuous_routers[1].memory[0][0,0]=999
        self.assertNotEqual(self.s.continuous_routers[0].memory[0][0,0],999)

    def test_clone_chain_keeps_all_history(self):
        model,det,_=self.clone()
        update_memory(self.s,1,det,model,self.f.inputs([2.]*10),torch.tensor([6]*10),1,[6],'cpu')
        target=deepcopy(model);native=deepcopy(det)
        clone_continuous_memory(self.s,2,1,2,model,target,det,native)
        update_memory(self.s,2,native,target,self.f.inputs([4.]*10),torch.tensor([12]*10),2,[12],'cpu')
        self.assertEqual(set(self.s.continuous_routers[2].class_to_task.values()),{0,1,2})
        validate_task_support(self.s.continuous_routers[2],native)
        self.assertEqual([(r['source'],r['receiver']) for r in self.s.continuous_bootstrap_receipts],[(0,1),(1,2)])

    def test_checkpoint_resume_exact_receipts_predictions_and_bytes(self):
        model,det,_=self.clone()
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'continuation.pt';torch.save(self.s.state_dict(),path)
            saved=torch.load(path,weights_only=False)
        restored=OriginalTrainService(self.s.spec,saved)
        self.assertEqual(digest(restored.state_dict()),digest(self.s.state_dict()))
        self.assertEqual(restored.bytes['continuous_bootstrap'],sum(r['serialized_bytes'] for r in restored.continuous_bootstrap_receipts))
        acts={k:v.numpy() for k,v in model.get_context_activations_per_sample(self.f.inputs([1.,3.])).items()}
        features=self.s.continuous_routers[1].encode(acts,det.stable_feature_mask)
        a=self.s.continuous_routers[1].predict(features)
        b=restored.continuous_routers[1].predict(features)
        self.assertTrue(all(np.array_equal(u,v) for u,v in zip(a,b)))
        saved['continuous_routers'][1]['memory'][0][:]=777
        self.assertNotEqual(digest(restored.state_dict()),digest(saved))

    def test_reject_partial_or_mismatched_clone_without_state_commit(self):
        for kind in ('weights','masks','coordinates','missing_source','existing_receiver'):
            model=deepcopy(self.model);det=deepcopy(self.d);source=0;receiver=1
            if kind=='weights':model.anchor.data.fill_(1)
            elif kind=='masks':model.weight_masks['synthetic'].zero_()
            elif kind=='coordinates':det.stable_feature_mask=np.array([0,1,1,0])
            elif kind=='missing_source':source=9
            else:receiver=0
            before=digest(self.s.state_dict())
            with self.assertRaises(ValueError):
                clone_continuous_memory(self.s,receiver,source,1,self.model,model,self.d,det)
            self.assertEqual(before,digest(self.s.state_dict()))

    def test_missing_history_cannot_silently_recur(self):
        native=deepcopy(self.d)
        before=digest(snapshot_context_detector(native))
        with self.assertRaises(ValueError):
            update_memory(self.s,1,native,self.model,self.f.inputs([2.]*10),torch.tensor([6]*10),1,[6],'cpu')
        self.assertNotIn(1,self.s.continuous_routers)
        self.assertEqual(before,digest(snapshot_context_detector(native)))
        self.d.activation_memory[1]=self.d.activation_memory[0].copy()
        with self.assertRaises(ValueError):self.clone()

    def test_v1_and_legacy_v7_are_noops_rng_native_unchanged(self):
        before=digest(self.model.state_dict());bank=digest(snapshot_context_detector(self.d));rng=torch.get_rng_state().clone()
        self.clone()
        self.assertEqual(before,digest(self.model.state_dict()));self.assertEqual(bank,digest(snapshot_context_detector(self.d)))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for service in (None,OriginalTrainService(self.f.spec())):
            self.assertIsNone(clone_continuous_memory(service,0,0,0,None,None,None,None))

    def test_budget_failure_is_atomic_and_is_reported(self):
        self.s.spec['per_arm_bytes']=1;self.s.arms=('coverage',)
        before=digest(self.s.state_dict())
        with self.assertRaises(ValueError):self.clone()
        self.assertEqual(before,digest(self.s.state_dict()))
        self.s.spec['per_arm_bytes']=2**30;_,_,receipt=self.clone()
        self.assertEqual(self.s.bytes['coverage'],receipt['serialized_bytes'])

    def test_v7_checkpoint_is_not_relabelled_as_v8(self):
        old=OriginalTrainService(self.f.spec())
        with self.assertRaises(ValueError):OriginalTrainService(self.s.spec,old.state_dict())
        saved=self.s.state_dict();del saved['continuous_bootstrap_receipts']
        with self.assertRaises(ValueError):OriginalTrainService(self.s.spec,saved)
        self.clone();saved=self.s.state_dict()
        saved['continuous_bootstrap_receipts'][0]['serialized_bytes']+=1
        with self.assertRaises(ValueError):OriginalTrainService(self.s.spec,saved)

    def test_v8_config_and_terminal_dispatch_keep_native_control(self):
        from unittest.mock import patch
        from fed_learning.training.candle_v4_original_train import validate_original_config
        cfg=json.loads((ROOT/'candle_v4_runtime/configs/candle_v1.json').read_text())
        cfg.update(candle_v4_original_train=True,candle_v4_full_service=self.s.spec,denice_router_mode='multiclass_balanced')
        validate_original_config(cfg)
        original=self.f.spec
        with patch.object(self.f,'spec',side_effect=lambda:dict(original(),version=BOOTSTRAP_VERSION)):
            self.f.test_terminal_keeps_v1_control_and_same_test_rows()

    def test_real_native_bootstrap_with_early_and_late_task_memory(self):
        from fed_learning.models.denice_model import DeNICEModel
        from fed_learning.training.decentralized_denice_il import _bootstrap_denice_model
        model=DeNICEModel((39,1),34).eval()
        detector=ContextDetector(router_mode='multiclass_balanced',router_reference_per_class=3)
        detector.retain_reference_inputs=False;detector.stable_feature_mask=np.zeros(548,dtype=bool)
        detector.stable_feature_mask[:16]=True;detector.stable_feature_mask[-16:]=True
        service=OriginalTrainService(dict(self.f.spec(),version=BOOTSTRAP_VERSION))
        torch.manual_seed(42);x=torch.randn(12,39,1);y=torch.tensor([0]*6+[1]*6)
        update_memory(service,0,detector,model,x,y,0,[0,1],'cpu')
        cfg=json.loads((ROOT/'candle_v4_runtime/configs/candle_v1.json').read_text())
        cfg['input_shape']=tuple(cfg['input_shape'])
        clone=_bootstrap_denice_model(model,cfg,torch.device('cpu'))
        native=ContextDetector();restore_context_detector(native,snapshot_context_detector(detector))
        before=digest(model.state_dict());original=digest(service.continuous_routers[0].state_dict())
        clone_continuous_memory(service,35,0,5,model,clone,detector,native)
        update_memory(service,35,native,clone,torch.randn(12,39,1),torch.tensor([30]*6+[32]*6),5,[30,32],'cpu')
        self.assertEqual(set(service.continuous_routers[35].class_to_task.values()),{0,5})
        self.assertEqual(before,digest(model.state_dict()));self.assertEqual(original,digest(service.continuous_routers[0].state_dict()))
        restored=OriginalTrainService(service.spec,service.state_dict())
        a=predictions(service,35,clone,native,x[:4],list(range(34)))
        b=predictions(restored,35,clone,native,x[:4],list(range(34)))
        self.assertTrue(all(np.array_equal(u,v) for u,v in zip(a,b)))

    def test_hook_only_in_exact_clone_not_partial_catchup(self):
        source=(ROOT/'candle_v4_runtime/fed_learning/training/decentralized_denice_il.py').read_text(encoding='utf-8')
        tree=ast.parse(source)
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='clone_continuous_memory']
        self.assertEqual(len(calls),1)
        catchup=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_catch_up_rejoining_model')
        self.assertFalse(any(isinstance(n,ast.Name) and n.id=='clone_continuous_memory' for n in ast.walk(catchup)))
        self.assertIn('restore_context_detector',source[:source.index('clone_continuous_memory(\n')])


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BootstrapTests))
    (ROOT/'artifacts/candle_v4_bootstrap_fix_regression_20261026.json').write_text(json.dumps(dict(
        protocol=BOOTSTRAP_VERSION,tests=result.testsRun,passed=result.wasSuccessful(),failures=len(result.failures),errors=len(result.errors),
        synthetic_only=True,backbone_training=False,real_TEST_opened=False),indent=2))
    sys.exit(not result.wasSuccessful())
