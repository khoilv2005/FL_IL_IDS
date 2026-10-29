"""Historical repair contract tests; synthetic data, no TEST or backbone training."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'candle_v4_runtime'))
import candle_v4_historical_repair as h
from fed_learning.servers.continuous_class_router import ContinuousClassMemory
from fed_learning.training.candle_v4_shadow import digest


def memory(task,value):
    m=ContinuousClassMemory(3)
    acts={k:np.full((3,1),value+i,dtype=np.float32) for i,k in enumerate(('conv1','conv2','conv3','gru'))}
    m.push(acts,np.full(3,task*6,dtype=np.int64),task,np.array([1,0,1,0]));m.fit(task)
    return m.state_dict()


def detector(tasks):
    return dict(activation_memory={t:np.full((3,4),t,dtype=np.float32) for t in tasks},
        episode_classes={t:[t*6] for t in tasks},stable_feature_mask=np.array([1,0,1,0]),routing_feature_mask=None,
        binarize_thresholds=dict(conv1=.1,conv2=.2,conv3=.3,gru=.4))


def fixture(chain=False):
    old=dict(client_algorithm_states={0:dict(context_detector=detector([0]))},
             candle_v4_shadow_state=dict(continuous_routers={0:memory(0,1.)}))
    final=dict(client_model_states={0:{},1:{}},
        client_algorithm_states={0:dict(context_detector=detector([0])),1:dict(context_detector=detector([0,1]))},
        candle_v4_shadow_state=dict(continuous_routers={0:memory(0,1.),1:memory(1,2.)}),
        debug_history=[dict(task=0,client_data={0:dict(num_samples=3)},bootstrap_events=[]),
            dict(task=1,client_data={1:dict(num_samples=3)},bootstrap_events=[dict(client_id=1,bootstrap_policy='representative_clone',bootstrap_source=0,param_distance_to_source=0.)])])
    if chain:
        final['client_model_states'][2]={};final['client_algorithm_states'][2]=dict(context_detector=detector([0,1]))
        final['candle_v4_shadow_state']['continuous_routers'][2]=memory(1,3.)
        final['debug_history'][1]['client_data'][2]=dict(num_samples=3)
        final['debug_history'][1]['bootstrap_events'].append(dict(client_id=2,bootstrap_policy='representative_clone',bootstrap_source=1,param_distance_to_source=0.))
    return final,{0:old}


class RepairTests(unittest.TestCase):
    def test_preserves_unaffected_estimators_and_local_memory_no_alias(self):
        final,snapshots=fixture();before=digest(final)
        staged,records=h.repair_memories(final,snapshots)
        self.assertEqual(set(staged),{1});self.assertEqual(set(staged[1].class_to_task.values()),{0,1})
        self.assertEqual(digest(final),before);self.assertEqual(records[1]['imports'][0]['source_client'],0)
        self.assertTrue(np.array_equal(staged[1].memory[6],final['candle_v4_shadow_state']['continuous_routers'][1]['memory'][6]))
        staged[1].memory[0][:]=999
        self.assertFalse(np.any(final['candle_v4_shadow_state']['continuous_routers'][0]['memory'][0]==999))
        self.assertEqual(before,digest(final))

    def test_same_task_clone_chain_resolves_prior_snapshot_owner(self):
        final,snapshots=fixture(True);staged,records=h.repair_memories(final,snapshots)
        self.assertEqual(set(staged),{1,2});self.assertEqual(records[2]['imports'][0]['clone_chain'],[2,1,0])

    def test_rejects_coordinate_threshold_or_sketch_mismatch_atomically(self):
        for key in ('stable_feature_mask','binarize_thresholds','activation_memory'):
            final,snapshots=fixture();before=digest(final)
            donor=snapshots[0]['client_algorithm_states'][0]['context_detector']
            if key=='stable_feature_mask':donor[key]=np.array([0,1,1,0])
            elif key=='binarize_thresholds':donor[key]['conv1']=99.
            else:donor[key][0][0,0]=99.
            with self.assertRaises(ValueError):h.repair_memories(final,snapshots)
            self.assertEqual(before,digest(final))

    def test_missing_or_cyclic_lineage_fails_instead_of_using_other_source(self):
        final,_=fixture();line=h.lineage_from_state(final)
        line[1]['events'][0]['bootstrap_source']=1
        with self.assertRaises(ValueError):h.find_origin(line,1,0)
        line[1]['events']=[]
        with self.assertRaises(ValueError):h.find_origin(line,1,0)

    def test_exact_member_selection_and_checksum_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            zpath=Path(td)/'x.zip'
            with zipfile.ZipFile(zpath,'w') as z:z.writestr('run/file.pt',b'correct');z.writestr('../unrelated',b'ignore')
            with zipfile.ZipFile(zpath) as z:
                name=h.member(z,'file.pt');target=Path(td)/'out.pt'
                with self.assertRaises(ValueError):h.copy_member(z,name,target,'0'*64)
                self.assertFalse(target.exists())
                h.copy_member(z,name,target,hashlib.sha256(b'correct').hexdigest())
                self.assertEqual(target.read_bytes(),b'correct')
                with self.assertRaises(ValueError):h.member(z,'absent.pt')

    def test_prepare_extracts_only_continuations_and_checks_same_run(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);a=root/'r16.zip';b=root/'r17.zip';lock=dict(source_commit=h.SOURCE)
            with zipfile.ZipFile(a,'w') as outer:
                outer.writestr('run/candle_v4_run_lock.json',json.dumps(lock))
                for task in range(5):
                    data=('task'+str(task)).encode();name=f'continuation_state_task_{task}.pt';buf=io.BytesIO()
                    with zipfile.ZipFile(buf,'w') as z:
                        z.writestr(name,data);z.writestr('round_model.pt',b'never extract')
                        z.writestr('checkpoint_archive_manifest.json',json.dumps(dict(task_id=task,completed_rounds=list(range(20)),continuation_checkpoint=name,checksums={name:hashlib.sha256(data).hexdigest()})))
                    outer.writestr(f'run/checkpoint_task_{task}_all_rounds.zip',buf.getvalue())
            with zipfile.ZipFile(b,'w') as z:
                z.writestr('run/candle_v4_run_lock.json',json.dumps(lock));z.writestr('run/candle_v4_completion.json',json.dumps(dict(completed=True)))
                z.writestr('run/continuation_state_latest.pt',b'task5');z.writestr('run/candle_v4_final_test.json','{}')
            with patch.object(h,'FINAL_SHA',hashlib.sha256(b'task5').hexdigest()):
                paths,report=h.prepare_archives(a,b,root/'cache')
            self.assertEqual(set(paths),set(range(6)));self.assertFalse((root/'cache/round_model.pt').exists())
            self.assertFalse(list((root/'cache').glob('input_task_*.zip')))


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RepairTests))
    (ROOT/'artifacts/candle_v4_historical_repair_tests_20261026.json').write_bytes(json.dumps(dict(
        tests=result.testsRun,passed=result.wasSuccessful(),failures=len(result.failures),errors=len(result.errors),
        synthetic_only=True,raw_TRAIN_or_TEST_opened=False,backbone_or_expert_training=False),indent=2).encode())
    sys.exit(not result.wasSuccessful())
