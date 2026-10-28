"""Top2 union contracts; no real data, training or label-based routing."""
import inspect
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'candle_v4_runtime'))
from eval_candle_v4_task_round19_topk import candidates,fuse_candidates,predict_pair,prepare,SOURCE


class TopkTests(unittest.TestCase):
    def test_prepare_six_boundaries_without_round_weight_extraction(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);r19=root/'19.zip';r21=root/'21.zip';lock=json.dumps({'source_commit':SOURCE})
            for path,tasks in [(r19,range(5)),(r21,[5])]:
                with zipfile.ZipFile(path,'w') as outer:
                    outer.writestr('run/candle_v4_run_lock.json',lock)
                    if path==r21:outer.writestr('run/candle_v4_final_test.json',json.dumps(dict(completed=True,task=5,round=19)))
                    for task in tasks:
                        name=f'continuation_state_task_{task}.pt';data=f'boundary{task}'.encode();buf=io.BytesIO()
                        with zipfile.ZipFile(buf,'w') as inner:
                            inner.writestr(name,data);inner.writestr('unused_round_weights.pt',b'ignore')
                            inner.writestr('checkpoint_archive_manifest.json',json.dumps(dict(task_id=task,completed_rounds=list(range(20)),continuation_checkpoint=name,checksums={name:hashlib.sha256(data).hexdigest()})))
                        outer.writestr(f'run/checkpoint_task_{task}_all_rounds.zip',buf.getvalue())
            paths,_=prepare(r19,r21,root/'cache')
            self.assertEqual(set(paths),set(range(6)))
            self.assertFalse(list((root/'cache').glob('input_task_*.zip')))
            self.assertFalse((root/'cache/unused_round_weights.pt').exists())

    def test_single_task_never_selects_absent_task(self):
        got=candidates([0,0],np.array([[1.,0.,0.],[1.,0.,0.]]),{0})
        np.testing.assert_array_equal(got,[[0,-1],[0,-1]])

    def test_second_only_from_available_preserves_production_top1(self):
        got=candidates([2,0],np.array([[.8,.9,.8,.7],[.9,1.,.1,.4]]),{0,2,3})
        np.testing.assert_array_equal(got,[[2,0],[0,3]])

    def test_second_task_expert_can_override_but_third_cannot(self):
        native=np.array([0,0]);scores=np.array([[.6,.8,.99],[.4,.3,.99]])
        classes=[1,7,13];births=[0,1,2]
        top1,_=fuse_candidates(native,scores,classes,births,np.array([[0],[0]]))
        top2,_=fuse_candidates(native,scores,classes,births,np.array([[0,1],[0,1]]))
        np.testing.assert_array_equal(top1,[1,0]);np.testing.assert_array_equal(top2,[7,0])

    def test_fixed_half_threshold_and_smallest_class_tie(self):
        pred,fire=fuse_candidates([5],np.array([[.5,.5]]),[1,7],[0,1],np.array([[0,1]]))
        np.testing.assert_array_equal(pred,[1]);np.testing.assert_array_equal(fire,[2])

    def test_no_experts_native_fallback_and_task0_equality(self):
        p,n=fuse_candidates([3,1],np.empty((2,0)),[],[],np.array([[0,-1],[0,-1]]))
        np.testing.assert_array_equal(p,[3,1]);np.testing.assert_array_equal(n,[0,0])
        scores=np.array([[.9,.8]])
        a,_=fuse_candidates([2],scores,[0,1],[0,0],np.array([[0]]))
        b,_=fuse_candidates([2],scores,[0,1],[0,0],np.array([[0,-1]]))
        np.testing.assert_array_equal(a,b)

    def test_predictor_has_no_true_label_or_oracle_arguments(self):
        self.assertEqual(list(inspect.signature(predict_pair).parameters),['service','cid','model','det','inputs','seen'])
        self.assertNotIn('oracle',inspect.getsource(predict_pair))


if __name__=='__main__':unittest.main()
