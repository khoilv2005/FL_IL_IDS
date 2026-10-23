"""Terminal evaluator checks on synthetic TEST only; no training or dataset access."""
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
from fed_learning.training.candle_v4_service import FullCoverageService,VERSION
from fed_learning.training import candle_v4_terminal_test as terminal


class TerminalTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        spec=dict(version=VERSION,seed=42,roles_dir='SYNTHETIC',data_dir='SYNTHETIC',
            output_dir=self.directory.name,input_sha256='synthetic',per_expert_bytes=2**27,per_arm_bytes=2**36)
        self.service=FullCoverageService(spec,authority=SimpleNamespace(manifest={'content_disjoint':True}))
        self.models={o:torch.nn.Linear(39,34) for o in range(3)};self.detectors={o:None for o in self.models}
        self.x=torch.arange(102*39,dtype=torch.float32).reshape(102,39);self.y=torch.arange(34).repeat(3)
        self.loader=SimpleNamespace(get_full_test_data=lambda:(self.x,self.y))

    def prepare_terminal(self):
        self.service.boundary=[5,19];self.service.gated=list(range(6))
        self.service.evaluations=[dict(task=t) for t in range(6)]

    def test_full_test_only_after_terminal_and_identical_complete_rows(self):
        with patch.object(terminal,'IncrementalDataLoader',return_value=self.loader) as loader:
            with self.assertRaises(ValueError):terminal.evaluate_terminal(self.service,self.models,self.detectors,4,19)
            loader.assert_not_called()
            self.prepare_terminal()
            with patch.object(terminal,'base_predictions',side_effect=lambda model,detector,x,seen:np.full(len(x),2,dtype=np.int64)):
                result=terminal.evaluate_terminal(self.service,self.models,self.detectors,5,19)
            self.assertTrue(result['partition']['all_rows_assigned_once'])
            self.assertEqual(result['test_rows'],102);self.assertEqual(result['evaluated_rows'],102)
            for metrics in result['metrics'].values():self.assertEqual(metrics['samples'],102)
            self.assertEqual(result['metrics']['V1'],result['metrics']['coverage'])
            self.assertEqual(result['rescue']['coverage'],0)
            loader.assert_called_once()
            with self.assertRaises(ValueError):terminal.evaluate_terminal(self.service,self.models,self.detectors,5,19)
            loader.assert_called_once()
        self.assertTrue((Path(self.directory.name)/'candle_v4_final_test.json').exists())

    def test_missing_class_rejected_and_native_hook_is_after_consolidation_archive(self):
        self.prepare_terminal();self.y=torch.zeros(102,dtype=torch.long)
        with patch.object(terminal,'IncrementalDataLoader',return_value=self.loader):
            with self.assertRaises(ValueError):terminal.evaluate_terminal(self.service,self.models,self.detectors,5,19)
        code=(ROOT/'candle_v4_runtime/fed_learning/training/decentralized_denice_il.py').read_text(encoding='utf-8')
        self.assertGreater(code.index("if config.get('candle_v4_terminal_full_test'):"),code.index('seal_task_archive(output_dir,task_id'))
        self.assertFalse((Path(self.directory.name)/'candle_v4_final_test.json').exists())

    def test_original_train_single_bank_and_native_only_terminal(self):
        from fed_learning.training.candle_v4_original_train import OriginalTrainService,VERSION
        for index,arms in enumerate((['coverage'],[])):
            spec=dict(self.service.spec,version=VERSION,arms=arms,fixed_threshold=.5,
                      output_dir=str(Path(self.directory.name)/str(index)))
            Path(spec['output_dir']).mkdir()
            self.service=OriginalTrainService(spec);self.prepare_terminal()
            with patch.object(terminal,'IncrementalDataLoader',return_value=self.loader),patch.object(terminal,'base_predictions',side_effect=lambda m,d,x,seen:np.full(len(x),2,dtype=np.int64)):
                result=terminal.evaluate_terminal(self.service,self.models,self.detectors,5,19)
            self.assertEqual(set(result['metrics']),{'V1',*arms})
            self.assertEqual(result['evaluated_rows'],102)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TerminalTests))
    (ROOT/'artifacts/candle_v4_terminal_test_regression_20261023.json').write_text(json.dumps(dict(
        tests=result.testsRun,passed=result.wasSuccessful(),failures=len(result.failures),errors=len(result.errors),
        real_TEST_opened=False,real_CAL_opened=False,full_training_started=False,fixture='synthetic TEST only'),indent=2)+'\n',encoding='utf-8')
    sys.exit(not result.wasSuccessful())
