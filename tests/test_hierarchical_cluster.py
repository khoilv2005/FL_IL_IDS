"""Real optimizers, cross-cluster class coverage, no TEST during training, resume."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tools.run_denice_v2_end_to_end_smoke import FixtureData
from fed_learning.strategies.decentralized.denice_v2.runner import RunConfig
from fed_learning.strategies.decentralized.denice_v2.hierarchical import HierarchicalRunner
from fed_learning.strategies.decentralized.denice_v2.contracts import state_hash
from fed_learning.strategies.decentralized.denice_v2.evaluation import evaluate_cluster


class CoverageData(FixtureData):
    def __init__(self):
        super().__init__()
        for (task,cid), (x,y,rows) in list(self.raw.items()):
            classes=self.task_classes[task]
            labels=classes[:len(classes)//2] if cid<2 else classes[len(classes)//2:]
            self.raw[task,cid]=(x,np.resize(labels,len(y)).astype(np.int64),rows)


class HierarchyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tmp=tempfile.TemporaryDirectory()
        cls.root=Path(cls.tmp.name)
        cls.cfg=RunConfig(smoke=True,tasks=2,rounds_per_task=2,batch_size=8,context_rows=8,
            defer_evaluation=True,server_only_storage=True,policy='routerless_v1',server_exchange=True)
        cls.runner=HierarchicalRunner(CoverageData(),cls.root/'train',cls.cfg)
        with patch.object(cls.runner.data,'test',side_effect=AssertionError('TEST touched during train')):
            cls.runner.run()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_cluster_learns_union_without_router(self):
        for s in self.runner.unique_states().values():
            self.assertEqual(s.algorithm_state['seen_classes'],tuple(range(12)))
            self.assertFalse(s.router_state['activation_memory'])
            self.assertEqual(s.router_state['router_mode'],'routerless_v1')
            for cid in s.members:
                self.assertIs(self.runner.states[cid],s)
        self.assertEqual(len({state_hash(s.model_state) for s in self.runner.unique_states().values()}),1)

    def test_resume_identical(self):
        resumed=HierarchicalRunner(CoverageData(),self.root/'resumed',self.cfg)
        resumed.resume(0,0,self.root/'train')
        with patch.object(resumed.data,'test',side_effect=AssertionError('TEST touched')):
            resumed.run()
        self.assertEqual({c:s.fingerprint() for c,s in self.runner.states.items()},
                         {c:s.fingerprint() for c,s in resumed.states.items()})
        self.assertEqual(self.runner.communication,resumed.communication)

    def test_compact_uploads_match_dense_training_and_resume(self):
        cfg = replace(self.cfg, compact_uploads=True)
        compact = HierarchicalRunner(CoverageData(), self.root/'compact', cfg)
        with patch.object(compact.data, 'test', side_effect=AssertionError('TEST touched')):
            compact.run()
        self.assertEqual({c:s.fingerprint() for c,s in self.runner.states.items()},
                         {c:s.fingerprint() for c,s in compact.states.items()})
        before = sum(r.get('model_delta_upload_bytes', 0) for r in self.runner.communication)
        after = sum(r.get('model_delta_upload_bytes', 0) for r in compact.communication)
        self.assertLess(after, before)
        resumed = HierarchicalRunner(CoverageData(), self.root/'compact_resumed', cfg)
        resumed.resume(0, 0, self.root/'compact')
        with patch.object(resumed.data, 'test', side_effect=AssertionError('TEST touched')):
            resumed.run()
        self.assertEqual({c:s.fingerprint() for c,s in compact.states.items()},
                         {c:s.fingerprint() for c,s in resumed.states.items()})
        self.assertEqual(compact.communication, resumed.communication)

    def test_eval_no_task_labels_used_for_prediction(self):
        data=self.runner.data
        s=next(iter(self.runner.unique_states().values()))
        with patch('fed_learning.strategies.decentralized.denice_v2.evaluation.ClusterRouter',
                side_effect=AssertionError('Router used')):
            r=evaluate_cluster(s,data.test_x,data.test_y,'fixture',self.root/'eval.json',batch_size=17)
        self.assertFalse(r['inference_uses_true_task'])
        self.assertFalse(r['router_metrics_applicable'])
        self.assertEqual(r['metrics']['rows'],36)

    def test_no_raw_sharing_or_client_final_broadcast(self):
        rounds=[r for r in self.runner.communication if r['phase']=='hierarchical_round']
        self.assertEqual(len(rounds),4)
        self.assertTrue(all(not r['raw_training_rows_shared'] and r['final_state_broadcast_bytes']==0 for r in rounds))
        self.assertTrue(rounds[0]['server_statistic_upload_bytes']>0)


if __name__=='__main__':unittest.main()
