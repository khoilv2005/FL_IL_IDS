"""V4-only readout must match V4 predictions without baseline inference."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'candle_v4_runtime'))
from fed_learning.training.candle_v4_original_train import OriginalTrainService,BOOTSTRAP_VERSION
from fed_learning.training import candle_v4_only_eval as only
from fed_learning.training import candle_v4_terminal_test as terminal


class OnlyTests(unittest.TestCase):
    def service(self,path,enabled):
        spec=dict(version=BOOTSTRAP_VERSION,seed=42,arms=['coverage'],router_mode='continuous_class_v1',fixed_threshold=.5,
            activation_policy='training_only_unverified',prediction_fusion='predicted_task_scoped',input_sha256='synthetic',
            data_dir='synthetic',output_dir=str(path),per_expert_bytes=2**27,per_arm_bytes=2**36)
        if enabled:spec['evaluation_mode']='v4_only'
        s=OriginalTrainService(spec);s.boundary=[5,19];s.gated=list(range(6));s.evaluations=[dict(task=t) for t in range(6)]
        return s

    def test_only_v4_has_identical_v4_predictions_and_no_baseline_forward(self):
        x=torch.zeros(102,39);y=torch.arange(34).repeat(3);x[:,0]=y.float()
        loader=SimpleNamespace(get_full_test_data=lambda:(x,y));models={0:torch.nn.Linear(39,34)};detectors={0:None}
        def prediction(service,cid,model,det,inputs,seen):
            pred=inputs[:,0].numpy().astype(np.int64);return pred,pred//6
        with tempfile.TemporaryDirectory() as td:
            normal=Path(td)/'normal';normal.mkdir();single=Path(td)/'only';single.mkdir()
            with patch.object(terminal,'IncrementalDataLoader',return_value=loader),patch.object(terminal,'base_predictions',return_value=(np.zeros(102,dtype=np.int64),np.zeros(102,dtype=np.int64))) as baseline,patch('fed_learning.training.candle_v4_continuous_runtime.predictions',side_effect=prediction):
                old=terminal.evaluate_terminal(self.service(normal,False),models,detectors,5,19)
                self.assertEqual(baseline.call_count,1)
            with patch.object(only,'IncrementalDataLoader',return_value=loader),patch.object(only,'predictions',side_effect=prediction) as routed,patch.object(terminal,'base_predictions',side_effect=AssertionError('baseline must not run')):
                new=terminal.evaluate_terminal(self.service(single,True),models,detectors,5,19)
            self.assertEqual(routed.call_count,1)
            self.assertEqual(new['metrics']['V4'],old['metrics']['coverage'])
            self.assertEqual(set(new['metrics']),{'V4'});self.assertEqual(set(new['per_client'][0]),{'V4','router_metrics'})
            self.assertEqual(new['per_client'][0]['router_metrics']['accuracy'],1.)
            self.assertTrue(new['partition']['all_rows_assigned_once']);self.assertEqual(new['evaluated_rows'],102)
            self.assertEqual(new['router_metrics']['accuracy'],1.);self.assertFalse(new['comparisons_computed'])
            self.assertNotIn('rescue',new);self.assertNotIn('baseline_router_mode',new)

    def test_terminal_boundary_and_overwrite_protected(self):
        with tempfile.TemporaryDirectory() as td:
            s=self.service(td,True)
            with self.assertRaises(ValueError):only.evaluate_v4_only(s,{}, {},4,19)
            (Path(td)/'candle_v4_final_test.json').write_text('{}')
            with self.assertRaises(ValueError):only.evaluate_v4_only(s,{}, {},5,19)

    def test_v4_report_has_no_baseline_or_placeholder_baseline_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            s=self.service(td,True);s.evaluations=[];s._evaluate(0,{},{});s.write_report()
            report=json.loads((Path(td)/'candle_v4_development_report.json').read_text())
            self.assertEqual(set(report['evaluations'][0]['metrics']),{'V4'})
            self.assertNotIn('baseline',report);self.assertNotIn('native_control_router_mode',report)
            self.assertFalse(report['comparisons_computed'])

    def test_separate_config_keeps_training_budget_identical(self):
        root=ROOT/'candle_v4_runtime'
        self.assertEqual((root/'configs/candle_v4.json').read_bytes(),(root/'configs/candle_v1.json').read_bytes())
        self.assertEqual((root/'requirements-candle-v4-kaggle.txt').read_bytes(),(root/'requirements-candle-v1-kaggle.txt').read_bytes())


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(OnlyTests))
    (ROOT/'artifacts/candle_v4_only_eval_tests_20261026.json').write_bytes(json.dumps(dict(tests=result.testsRun,
        passed=result.wasSuccessful(),failures=len(result.failures),errors=len(result.errors),
        synthetic_only=True,no_full_training=True,no_real_TEST_opened=True),indent=2).encode())
    sys.exit(not result.wasSuccessful())
