"""Current-graph handoff regression, synthetic TRAIN only."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import torch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.test_candle_v4_original_train import OriginalTrainTests,TrainOnlyClient
from fed_learning.training.candle_v4_shadow import digest,native_fingerprint
from fed_learning.training.candle_v4_original_train import VERSION


class HandoffTests(OriginalTrainTests):
    def alternate(self,count=5000):
        c=self.clients[2];labels=torch.full_like(c.y_train,2);labels[:count]=0
        self.clients[2]=TrainOnlyClient(c.X_train,labels)

    def graph_for(self,peers):
        g=dict(alpha_debug={o:dict(group_ids=[o],alphas=[1.]) for o in self.clients})
        ids=[0,*peers];g['alpha_debug'][0]=dict(group_ids=ids,alphas=[1/len(ids)]*len(ids))
        return g

    def assert_same_continuation(self,left,right):
        a=left.state_dict();b=right.state_dict()
        # Wall-clock accounting is intentionally different across two executions.
        self.assertGreaterEqual(a.pop('runtime'),0);self.assertGreaterEqual(b.pop('runtime'),0)
        self.assertEqual(digest(a),digest(b))

    def test_handoff_preserves_model_adam_and_valid_multi_donor_proof(self):
        self.alternate();s=self.new();self.run_round(s,graph=self.graph_for([1]))
        e=s.engines['coverage'][0,0];model,opt=e.branches['joint_class_balanced']
        state=digest(e.state_dict()['branches']);wire=e.bytes;native=native_fingerprint(self.models,self.detectors)
        select=s._select_training_donor
        def checked(*args):
            result=select(*args)
            self.assertIs(e,s.engines['coverage'][0,0]);self.assertIs(opt,e.branches['joint_class_balanced'][1])
            self.assertEqual(state,digest(e.state_dict()['branches']));self.assertEqual(wire,e.bytes)
            self.assertEqual(e.steps,44);self.assertEqual(e.pending,0)
            return result
        with patch.object(s,'_select_training_donor',side_effect=checked):self.run_round(s,1,graph=self.graph_for([2]))
        self.assertEqual(e.steps,88);self.assertEqual(e.pending,0)
        self.assertEqual(s.entries[0,0]['donor'],2);self.assertEqual(e.spec['donor'],2)
        self.assertEqual(len(s.entries[0,0]['donor_lineage']),1)
        self.assertEqual(s.history[-1]['handoff_count'],1)
        proof=s._training_proof((0,0),'coverage');self.assertTrue(proof['valid']);self.assertEqual(proof['donor_updates'],{1:44,2:44})
        self.assertEqual(float(opt.state[next(model.parameters())]['step']),88.)
        self.assertEqual(native,native_fingerprint(self.models,self.detectors))

    def test_handoff_retains_current_valid_donor_without_churn(self):
        self.alternate();s=self.new();self.run_round(s,graph=self.graph_for([1]))
        self.run_round(s,1,graph=self.graph_for([2]));self.run_round(s,2,graph=self.graph_for([1,2]))
        self.assertEqual(s.entries[0,0]['donor'],2);self.assertEqual(len(s.entries[0,0]['donor_lineage']),1)
        self.assertEqual(s.history[-1]['handoff_count'],0)
        self.assertEqual(s._training_proof((0,0),'coverage')['donor_updates'],{1:44,2:88})

    def test_handoff_no_alternate_or_insufficient_counts_do_not_use_absent_peer(self):
        self.alternate(31);s=self.new();self.run_round(s,graph=self.graph_for([1]))
        self.run_round(s,1,graph=self.graph_for([2]))
        e=s.engines['coverage'][0,0];self.assertEqual(e.steps,44);self.assertEqual(e.pending,44)
        self.assertEqual(s.entries[0,0]['donor'],1);self.assertFalse(s.entries[0,0]['donor_lineage'])
        row=s.history[-1]['experts'][0];self.assertFalse(row['eligible']);self.assertEqual(row['updates'],0)
        self.assertEqual(row['donor_authorization']['reason'],'NO_ELIGIBLE_DONOR')
        self.assertTrue(s._training_proof((0,0),'coverage')['valid'])

    def test_handoff_resume_before_and_after_switch_exact(self):
        self.alternate();s=self.new();self.run_round(s,graph=self.graph_for([1]))
        self.run_round(s,1,graph=self.graph_for([]))
        path=self.path/'handoff.pt';torch.save(s.state_dict(),path);restored=self.new(torch.load(path,weights_only=False))
        for service in (s,restored):self.run_round(service,2,graph=self.graph_for([2]))
        self.assert_same_continuation(s,restored)
        self.assertEqual(s.engines['coverage'][0,0].steps,132);self.assertEqual(s.engines['coverage'][0,0].pending,0)
        restored=self.new(s.state_dict())
        for service in (s,restored):self.run_round(service,3,graph=self.graph_for([1,2]))
        self.assert_same_continuation(s,restored)

    def test_handoff_catchup_cap_and_historical_freeze(self):
        self.alternate();s=self.new();self.run_round(s,graph=self.graph_for([1]))
        self.run_round(s,1,graph=self.graph_for([]));self.run_round(s,2,graph=self.graph_for([]))
        self.run_round(s,3,graph=self.graph_for([2]));e=s.engines['coverage'][0,0]
        self.assertEqual(e.steps,132);self.assertEqual(e.pending,44)
        self.assertEqual(s.history[-1]['experts'][0]['updates'],88)
        self.assertTrue(s._training_proof((0,0),'coverage')['valid'])
        self.finish(s);before=digest(e.state_dict()['branches']);lineage=deepcopy(s.entries[0,0])
        self.clients=self.load_task(1);self.run_round(s,0,1,graph=self.graph_for([1,2]))
        self.assertEqual(before,digest(e.state_dict()['branches']));self.assertEqual(lineage,s.entries[0,0])

    def test_handoff_rejects_tampered_graph_lineage_or_authorization(self):
        self.alternate();s=self.new();self.run_round(s,graph=self.graph_for([1]));self.run_round(s,1,graph=self.graph_for([2]))
        for change in ['donor','graph','from','count']:
            saved=deepcopy(s.state_dict())
            if change=='donor':saved['history'][0]['experts'][0]['donor']=2
            elif change=='graph':saved['entries'][0,0]['donor_lineage'][0]['graph_sha256']='wrong'
            elif change=='from':saved['entries'][0,0]['donor_lineage'][0]['from_donor']=3
            else:saved['history'][1]['experts'][0]['donor_authorization']['target_count']=31
            self.assertFalse(self.new(saved)._training_proof((0,0),'coverage')['valid'],change)

    def test_handoff_router_binding_and_streaming_multiclass_restore(self):
        from fed_learning.training.candle_v4_original_train import validate_original_config
        from fed_learning.servers.nice_server import ContextDetector
        from fed_learning.training.checkpoint_state import snapshot_context_detector,restore_context_detector
        config=json.loads((ROOT/'candle_v4_runtime/configs/candle_v1.json').read_text())
        config.update(candle_v4_original_train=True,candle_v4_full_service=dict(self.spec,router_mode='multiclass_balanced'),
                      denice_router_mode='multiclass_balanced')
        validate_original_config(config)
        with self.assertRaises(ValueError):validate_original_config(dict(config,denice_router_mode='binary_cosine'))
        detector=ContextDetector(router_mode='multiclass_balanced')
        detector.activation_memory={0:np.zeros((4,2))};detector.train_models(0)
        self.assertEqual(detector.predict_episodes_batch(np.zeros((1,2))).tolist(),[0])
        detector.activation_memory[1]=np.ones((4,2));detector.train_models(1)
        inputs=np.array([[0,0],[1,1]])
        expected=detector.predict_episodes_batch(inputs)
        self.assertEqual(expected.tolist(),[0,1])
        restored=ContextDetector();restore_context_detector(restored,snapshot_context_detector(detector))
        self.assertTrue(np.array_equal(expected,restored.predict_episodes_batch(inputs)))


if __name__=='__main__':
    suite=unittest.TestSuite(HandoffTests(n) for n in sorted(dir(HandoffTests)) if n.startswith('test_handoff_'))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    (ROOT/'artifacts/candle_v4_handoff_regression_20261024.json').write_text(json.dumps(dict(
        version=VERSION,tests=result.testsRun,passed=result.wasSuccessful(),failures=len(result.failures),
        errors=len(result.errors),fixture='synthetic current-task TRAIN only',real_TEST_opened=False,
        real_CAL_opened=False,full_training_started=False),indent=2)+'\n')
    sys.exit(not result.wasSuccessful())
