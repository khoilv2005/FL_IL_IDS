"""Train-only cluster ownership, resumability and separate FULL TEST regression."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fed_learning.strategies.decentralized.denice_v2.runner import Runner, RunConfig
from fed_learning.strategies.decentralized.denice_v2.archive import TaskArchive
from fed_learning.strategies.decentralized.denice_v2.contracts import state_hash
from tools.run_denice_v2_end_to_end_smoke import FixtureData
from tools.eval_cluster_servers import evaluate_servers, completed_payload


class ClusterServerWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.data = FixtureData()
        cls.cfg = RunConfig(smoke=True, tasks=6, rounds_per_task=1, batch_size=8,
            context_rows=16, defer_evaluation=True, server_only_storage=True)
        cls.runner = Runner(cls.data, cls.root/'train', cls.cfg)
        with patch.object(cls.data, 'test', side_effect=AssertionError('TRAIN read TEST')):
            cls.runner.run()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_only_cluster_states_persist_and_no_post_round_broadcast(self):
        p = completed_payload(self.root/'train')
        self.assertEqual(len(p['states']), len(set(p['client_state'].values())))
        self.assertNotIn('client_models', p)
        for state in self.runner.unique_states().values():
            for cid in state.members:
                self.assertIs(self.runner.states[cid], state)
        for item in self.runner.communication:
            if item['phase'] == 'cluster_round':
                self.assertEqual(item['final_state_broadcast_bytes'], 0)
        self.assertEqual(self.runner.evaluations, [])
        self.assertFalse(list((self.root/'train').glob('evaluation_*.json')))
        inventory=json.loads((self.root/'train/temporary_servers_task_5.json').read_text())
        self.assertEqual(len(inventory['servers']),len(p['active']))

    def test_external_eval_uses_every_test_row_for_every_server(self):
        source=self.root/'data';source.mkdir(exist_ok=True)
        np.savez(source/'global_test_data.npz',X_test=self.data.test_x,y_test=self.data.test_y)
        before=state_hash(self.runner.checkpoint_payload())
        summary=evaluate_servers(self.root/'train',source,self.root/'eval',batch_size=17)
        self.assertEqual(summary['unique_test_rows'],len(self.data.test_y))
        self.assertEqual(len(summary['per_temporary_server']),len(self.runner.active))
        self.assertTrue(all(r['test_samples']==len(self.data.test_y) for r in summary['per_temporary_server']))
        self.assertEqual(before,state_hash(self.runner.checkpoint_payload()))
        again=evaluate_servers(self.root/'train',source,self.root/'eval',batch_size=17)
        self.assertEqual(summary,again)

    def test_incomplete_training_rejected_before_test_access(self):
        folder=self.root/'incomplete';folder.mkdir(exist_ok=True)
        (folder/'completion.json').write_text(json.dumps(dict(completed=False,tasks_completed=5)))
        with self.assertRaisesRegex(ValueError,'Finish TRAIN'):
            evaluate_servers(folder,self.root/'missing_test',self.root/'invalid_eval')

    def test_train_only_resume_has_identical_final_state(self):
        resumed=Runner(FixtureData(),self.root/'resumed',self.cfg)
        resumed.resume(4,source_root=self.root/'train')
        with patch.object(resumed.data,'test',side_effect=AssertionError('TRAIN read TEST')):
            resumed.run()
        self.assertEqual({c:s.fingerprint() for c,s in resumed.states.items()},
                         {c:s.fingerprint() for c,s in self.runner.states.items()})
        self.assertEqual(resumed.communication,self.runner.communication)


class ClusterLauncherTests(unittest.TestCase):
    def test_default_launcher_dispatches_before_legacy_resume_download(self):
        import os
        import runpy
        import sys
        from types import ModuleType
        from unittest.mock import Mock
        module=ModuleType('tools.train_cluster_server')
        module.main=Mock()
        root=Path(__file__).resolve().parents[1]
        with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, {'tools.train_cluster_server':module}):
            with patch('subprocess.run',side_effect=AssertionError('Legacy download path executed')):
                with self.assertRaises(SystemExit) as end:
                    runpy.run_path(str(root/'train_incremental_kaggle.py'),run_name='__main__')
        self.assertEqual(end.exception.code,0)
        module.main.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
