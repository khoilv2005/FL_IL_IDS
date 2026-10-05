import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from tools.denice_peer_supported_gate_data import peer_supported_data,local_training_support
from tools.denice_competence_gate import content_hash
from tools.eval_denice_competence_gate_v2 import run_competence_gate_v2


class GateV2Tests(unittest.TestCase):
    def fixture(self,root):
        ids=list(range(5));classes={0:[0,1],1:[2,3]};data=root/'data';data.mkdir()
        states={};shards={};source_x=None
        for cid in ids:
            rng=np.random.default_rng(cid)
            x=np.column_stack([np.arange(700)+cid*1000,rng.normal(size=700),np.arange(700)%4]).astype('float32')
            y=x[:,-1].astype(int)
            if cid==0:source_x=x.copy()
            if cid==1:x[0]=source_x[2];y[0]=0 # ambiguous training content, valid task0 label
            np.savez(data/f'client_{cid}_train.npz',X_train=x,y_train=y)
            tasks=[1] if cid==0 else ([0] if cid==1 else [0,1])
            masks=classes if cid!=1 else {0:[0,1]}
            states[cid]=dict(denice=dict(cgofed_projection_state=dict(tasks=[dict(task_id=t) for t in tasks]),
                context_detector=dict(episode_classes=masks,router_last_refresh_task=tasks[-1])))
            tx=np.column_stack([np.arange(32)+100000+cid*1000,np.zeros(32),np.arange(32)%4]).astype('float32')
            shards[cid]=dict(X_test=torch.from_numpy(tx),y_test=torch.from_numpy(tx[:,-1].astype(int)),sample_ids=np.arange(32)+cid*32)
        ckpt=dict(config=dict(git_commit='03b9b53test'),client_algorithm_states=states,
            cluster=dict(task=5,round=19,groups={cid:ids for cid in ids},alpha_debug={
                cid:dict(group_ids=ids,alphas=[.2]*5) for cid in ids}))
        return ckpt,classes,ids,data,shards,source_x

    def test_peer_support_provenance_and_content_partitions(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);ckpt,classes,ids,data,shards,source_x=self.fixture(root)
            self.assertEqual(set(local_training_support(ckpt,0,classes)),{2,3}) # inherited task0 is excluded
            forbidden=content_hash(source_x[6]);shards[0]['X_test']=torch.from_numpy(source_x[6:7])
            needed={cid:[cid]+[d for d in ids if d!=cid] for cid in ids}
            pools=peer_supported_data(ckpt,data,classes,ids,shards,root,
                dict(calibration=32,fit=64,validation=32),20261005,needed)
            rows=pd.read_csv(root/'gate_data_provenance.csv');audit=json.loads((root/'gate_split_audit.json').read_text())
            self.assertEqual(audit['training_content_with_conflicting_labels_excluded'],1)
            self.assertFalse((rows.input_sha256==forbidden).any())
            self.assertEqual(int((rows.groupby('input_sha256').role.nunique()>1).sum()),0)
            self.assertTrue(set(rows[rows.origin_client_id==0].label).issubset({2,3}))
            self.assertTrue((rows[(rows.client_id==1)&(~rows.receiver_local_covered)].origin_client_id!=1).all())
            self.assertGreater(int((~pools['validation'][1]['receiver_local_covered']).sum()),0)
            for role in pools:self.assertEqual(set(pools[role][1]['y']),{0,1,2,3})
            again=root/'changed_test_labels';again.mkdir()
            for cid in ids:shards[cid]['y_test']=(shards[cid]['y_test']+1)%4
            changed=peer_supported_data(ckpt,data,classes,ids,shards,again,
                dict(calibration=32,fit=64,validation=32),20261005,needed)
            pd.testing.assert_frame_equal(rows,pd.read_csv(again/'gate_data_provenance.csv'))
            for role in pools:
                for cid in ids:np.testing.assert_array_equal(pools[role][cid]['y'],changed[role][cid]['y'])

    def test_illegal_origin_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);ckpt,classes,ids,data,shards,_=self.fixture(root)
            needed={cid:[cid] for cid in ids};needed[0].append(99)
            with self.assertRaisesRegex(ValueError,'outside fixed legitimate'):
                peer_supported_data(ckpt,data,classes,ids,shards,root,
                    dict(calibration=32,fit=64,validation=32),20261005,needed)

    def test_complete_v2_pipeline_and_validation_baselines(self):
        def record(x,donor):
            values=x.numpy();label=values[:,-1].astype(int);offset=(values[:,0].astype(int)+donor)%3==0
            pred=(label+offset)%4;n=len(pred)
            return dict(pred=pred,task=pred//2,router_confidence=np.full(n,.6),class_confidence=np.where(offset,.3,.9),
                class_margin=np.full(n,.3),class_entropy=np.full(n,.4),mask_count=np.full(n,2),
                task_supported=np.ones(n),class_supported=np.ones(n))
        def collect(ckpt,pools,needed,ids,seen,device,batch_size,manifest,out,stage):
            if stage=='test':self.assertTrue((out/'gate_lock.json').exists())
            return {role:{cid:{d:record(clients[cid]['X'],d) for d in needed[cid]} for cid in ids}
                    for role,clients in pools.items()}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);ckpt,classes,ids,data,shards,_=self.fixture(root)
            reference=root/'reference';reference.mkdir();frames=[]
            for cid in ids:
                own=record(shards[cid]['X_test'],cid)
                frames.append(pd.DataFrame(dict(client_id=cid,sample_in_shard=np.arange(32),
                    global_test_row=shards[cid]['sample_ids'],y_true=shards[cid]['y_test'].numpy(),
                    Multiclass=own['pred'],Multiclass_task=own['task'])))
            pd.concat(frames).to_csv(reference/'predictions.csv',index=False)
            (reference/'profile_manifest.json').write_text('{}')
            with patch('tools.eval_denice_competence_gate.collect',side_effect=collect),warnings.catch_warnings():
                warnings.simplefilter('ignore',ConvergenceWarning)
                result=run_competence_gate_v2(ckpt,shards,classes,ids,root/'out','cpu',reference,data,
                    budgets=(4,),limits=dict(calibration=32,fit=64,validation=32))
            out=root/'out';v=pd.read_csv(out/'gate_validation_metrics.csv')
            baselines={'GlobalDonorPrior','GlobalTaskPrior','ReceiverTaskPrior','majority','self'}
            self.assertTrue(baselines.issubset(set(v.policy)));self.assertTrue(baselines.issubset(set(result.policy)))
            lock=json.loads((out/'gate_lock.json').read_text())
            self.assertTrue(lock['selected_by_validation']['4'].startswith(('LR_','MLP_')))
            self.assertEqual(json.loads((out/'gate_completion.json').read_text())['variant'],'v2_peer_supported')
            pred=pd.read_csv(out/'gate_validation_predictions'/'client_1.csv')
            self.assertTrue('receiver_local_covered' in pred)
            for row in v.itertuples():
                allpred=pd.concat([pd.read_csv(out/'gate_validation_predictions'/f'client_{cid}.csv') for cid in ids])
                self.assertAlmostEqual(float((allpred[f'k4_{row.policy}']==allpred.y_true).mean()),row.accuracy)
            self.assertTrue((out/'gate_role_targets'/'calibration_receiver_1.npz').exists())
            self.assertTrue((out/'gate_validation_coverage_metrics.csv').exists())


if __name__=='__main__':unittest.main()
