import unittest
import numpy as np
from tools.denice_competence_gate import content_hash,content_role,competence_priors,feature_matrix,gate_decision


class CompetenceGateTests(unittest.TestCase):
    def record(self,pred,task):
        n=len(pred)
        return dict(pred=np.asarray(pred),task=np.asarray(task),router_confidence=np.full(n,.6),
            class_confidence=np.full(n,.7),class_margin=np.full(n,.3),class_entropy=np.full(n,.4),
            mask_count=np.full(n,2),task_supported=np.ones(n),class_supported=np.ones(n))

    def test_content_roles_and_shape(self):
        row=np.array([1,2],dtype=np.float32)
        self.assertEqual(content_hash(row),content_hash(row.astype(np.float64)))
        self.assertNotEqual(content_hash(row),content_hash(row[:,None]))
        self.assertEqual(content_role(content_hash(row)),content_role(content_hash(row.copy())))

    def test_priors_only_calibration(self):
        record=self.record([0,1],[0,0]);p=competence_priors({3:{3:record}},{3:np.array([0,0])})
        self.assertEqual(p['global_task'][3,0],(2,1))
        self.assertEqual(p['local_task'][3,3,0],(2,1))

    def test_feature_order_and_batch_invariance(self):
        records={3:self.record([0,1,2],[0,0,1]),7:self.record([1,1,0],[0,0,0])}
        priors=competence_priors({3:records},{3:np.array([0,1,2])})
        x,pred,names=feature_matrix(3,[3,7],records,{3:.1,7:.9},priors,2,3)
        self.assertEqual(x.shape,(6,len(names)))
        np.testing.assert_array_equal(pred,[[0,1],[1,1],[2,0]])
        for i in range(3):
            chunk={d:{k:v[i:i+1] for k,v in r.items()} for d,r in records.items()}
            small,_,_=feature_matrix(3,[3,7],chunk,{3:.1,7:.9},priors,2,3)
            np.testing.assert_allclose(x[i*2:i*2+2],small)

    def test_label_free_selection_and_ties(self):
        pred=np.array([[0,1,2],[0,1,1],[2,0,1]])
        scores=np.array([[.2,.8,.1],[.5,.5,.5],[.1,.8,.8]])
        result,donor=gate_decision(scores,pred,[3,9,7],[0,1,2],1)
        np.testing.assert_array_equal(result,[1,0,1]);np.testing.assert_array_equal(donor,[9,3,7])
        weighted,_=gate_decision(scores,pred,[3,9,7],[0,1,2],2)
        np.testing.assert_array_equal(weighted,[1,0,0])
        zero,_=gate_decision(np.zeros_like(scores),pred,[3,9,7],[0,1,2],2)
        np.testing.assert_array_equal(zero,pred[:,0])

    def test_masked_expert_confidence_batch_invariance(self):
        import torch
        from types import SimpleNamespace
        from unittest.mock import patch
        from tools.eval_denice_competence_gate import expert_features
        detector=SimpleNamespace(episode_classes={0:[0,1],1:[2,3]},activation_memory={0:[],1:[]})
        model=torch.nn.Linear(1,4);x=torch.tensor([[0.],[1.],[0.]])
        def logits(model,xb,*args,**kwargs):
            task=xb[:,0].long().numpy();out=torch.full((len(xb),4),-100.)
            for row,t in enumerate(task):
                out[row,2*t:2*t+2]=torch.tensor([0.,2.]) if t==0 else torch.tensor([2.,2.])
            return out,task
        def route(model,xb,detector):
            task=xb[:,0].long().numpy();q=np.full((len(task),2),.25);q[np.arange(len(task)),task]=.75
            return task,q
        with patch('tools.eval_denice_competence_gate._denice_routed_logits_with_episodes',side_effect=logits),patch(
                'tools.eval_denice_competence_gate._route_episodes_with_scores',side_effect=route):
            a=expert_features(model,detector,x,[0,1,2,3],'cpu',2)
            b=expert_features(model,detector,x,[0,1,2,3],'cpu',1)
        for key in a:np.testing.assert_allclose(a[key],b[key])
        np.testing.assert_array_equal(a['pred'],[1,2,1])
        np.testing.assert_allclose(a['router_confidence'],.75)
        self.assertAlmostEqual(a['class_entropy'][1],1.,places=6)
        self.assertAlmostEqual(a['class_margin'][1],0.)

    def test_pipeline_freezes_before_test_and_test_labels_do_not_fit_gate(self):
        import json,tempfile,warnings
        from pathlib import Path
        from unittest.mock import patch
        import pandas as pd
        import torch
        import joblib
        from sklearn.exceptions import ConvergenceWarning
        from tools.eval_denice_competence_gate import run_competence_gate
        ids=list(range(5));classes={0:[0,1],1:[2,3]}
        def synthetic_record(x,donor):
            values=x.numpy();label=values[:,-1].astype(int)
            offset=(values[:,0].astype(int)+donor)%3==0
            pred=(label+offset)%4;r=self.record(pred,pred//2)
            r['class_confidence']=np.where(offset,.3,.9)
            return r
        def fake_collect(ckpt,pools,needed,ids,seen,device,batch_size,manifest,out,stage):
            if stage=='test':
                lock=json.loads((out/'gate_lock.json').read_text())
                self.assertTrue(lock['locked_before_test_expert_inference'])
                self.assertTrue((out/'frozen_gate.joblib').exists())
            return {role:{cid:{d:synthetic_record(pool[cid]['X'],d) for d in needed[cid]} for cid in ids}
                    for role,pool in pools.items()}
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);reference=root/'reference';reference.mkdir();data=root/'data';data.mkdir()
            shards={};rows=[]
            for cid in ids:
                rng=np.random.default_rng(cid)
                x=np.column_stack([np.arange(1000)+cid*1000,rng.normal(size=1000),np.arange(1000)%4]).astype('float32')
                np.savez(data/f'client_{cid}_train.npz',X_train=x,y_train=x[:,-1].astype(int))
                x_test=np.column_stack([np.arange(32)+100000+cid*1000,np.zeros(32),np.arange(32)%4]).astype('float32')
                shards[cid]=dict(X_test=torch.from_numpy(x_test),y_test=torch.from_numpy(x_test[:,-1].astype(int)),sample_ids=np.arange(32)+cid*32)
                own=synthetic_record(shards[cid]['X_test'],cid)
                rows.append(pd.DataFrame(dict(client_id=cid,sample_in_shard=np.arange(32),global_test_row=shards[cid]['sample_ids'],
                    y_true=x_test[:,-1].astype(int),Multiclass=own['pred'],Multiclass_task=own['task'])))
            pd.concat(rows).to_csv(reference/'predictions.csv',index=False)
            (reference/'profile_manifest.json').write_text('{}')
            ckpt=dict(config=dict(git_commit='03b9b53test'),client_algorithm_states={cid:dict(denice=dict(
                cgofed_projection_state=dict(tasks=[dict(task_id=0),dict(task_id=1)]),
                context_detector=dict(episode_classes=classes,router_last_refresh_task=1))) for cid in ids},
                cluster=dict(task=5,round=19,groups={cid:ids for cid in ids},alpha_debug={
                    cid:dict(group_ids=ids,alphas=[.2]*5) for cid in ids}))
            with patch('tools.eval_denice_competence_gate.collect',side_effect=fake_collect),warnings.catch_warnings():
                warnings.simplefilter('ignore',ConvergenceWarning)
                first=run_competence_gate(ckpt,shards,classes,ids,root/'first','cpu',reference,data,
                    budgets=(4,),limits=dict(calibration=32,fit=64,validation=32))
                original=pd.read_csv(reference/'predictions.csv');original.y_true=(original.y_true+1)%4
                original.to_csv(reference/'predictions.csv',index=False)
                for cid in ids:shards[cid]['y_test']=(shards[cid]['y_test']+1)%4
                second=run_competence_gate(ckpt,shards,classes,ids,root/'second','cpu',reference,data,
                    budgets=(4,),limits=dict(calibration=32,fit=64,validation=32))
            a=joblib.load(root/'first'/'frozen_gate.joblib');b=joblib.load(root/'second'/'frozen_gate.joblib')
            self.assertEqual(a['selected'],b['selected']);self.assertEqual(a['priors'],b['priors'])
            rng=np.random.default_rng(77);probe=rng.normal(size=(20,len(a['feature_names'])))
            for key in a['gates']:
                np.testing.assert_array_equal(a['gates'][key].predict_proba(probe),b['gates'][key].predict_proba(probe))
            self.assertTrue(json.loads((root/'second'/'gate_completion.json').read_text())['completed'])
            for name in ('competence_gate_vs_budget.png','competence_gate_vs_budget.pdf'):
                self.assertGreater((root/'second'/name).stat().st_size,0)


if __name__=='__main__':unittest.main()
