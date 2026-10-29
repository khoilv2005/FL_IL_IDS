"""Frozen bounded Top2 oracle diagnostic. No fitting, tuning or deployed rule."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'candle_v4_runtime'))
from eval_checkpoint import _make_denice_client_model
from fed_learning.training.candle_v4_original_train import OriginalTrainService
from fed_learning.training.candle_v4_continuous_runtime import ContinuousRoutingView
from fed_learning.training.candle_v4_bank import base_predictions
from fed_learning.training.candle_v4_shadow import digest
from fed_learning.training.candle_v4_terminal_test import statistics
from fed_learning.training.decentralized_denice_il import _partition_test_data_by_client
from audit_candle_v4_router_final_preflight import route_scores


class FixedRouteView(ContinuousRoutingView):
    def __init__(self,det,memory,routes):
        super().__init__(det,memory);self.routes=np.asarray(routes,dtype=np.int64)
    def predict_episodes_batch(self,features):
        assert len(features)==len(self.routes)
        return self.routes.copy()
    def predict_episodes_with_scores(self,features):
        return self.predict_episodes_batch(features),None


def main():
    warnings.filterwarnings('ignore',message='Trying to unpickle estimator')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=ROOT/'audit_denice/candle_v4_results20_top2_20261028';out.mkdir(parents=True,exist_ok=True)
    ck=torch.load(ROOT/'audit_denice/candle_v4_results20_20261027/continuation_state_latest.pt',map_location='cpu',weights_only=False,mmap=True)
    assert ck['meta']['completed_task']==5 and ck['meta']['boundary']=='task'
    service=OriginalTrainService(ck['candle_v4_shadow_state']['spec'],ck['candle_v4_shadow_state'])
    expert_before=digest(ck['candle_v4_shadow_state']['engines'])
    with np.load(r'C:\Users\khoak\Downloads\candle_v1_multiclass_audit\global_test_data.npz') as z:
        x=torch.from_numpy(z['X_test']);y=torch.from_numpy(z['y_test']).long()
    ids=sorted(ck['client_model_states'])
    shards,_=_partition_test_data_by_client(x,y,ids,seed=523687,lazy_features=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    result=dict(completed=False,source='results20 final Task5/R19',seed=20261028,max_rows_per_client=2048,
        panel='Uniform deterministic sample without replacement from each official disjoint client shard',
        fitted=False,training=False,threshold_tuning=False,production_rule_changed=False,
        true_labels_used_only_for_diagnostics=True,sklearn_coefficients_not_refit=True,
        sampled_TEST_development_only=True,clients={})
    cms={k:np.zeros((34,34),dtype=np.int64) for k in ('baseline','oracle_top2')}
    counts=dict(rows=0,top1_correct_task=0,top2_contains_true_task=0,true_task_second=0,
                task_absent=0,rescued=0,broken=0,second_task_native_correct=0,second_task_v4_correct=0)
    start=time.perf_counter()
    for position,cid in enumerate(ids,1):
        shard=shards[cid];rng=np.random.default_rng(20261028+cid)
        idx=np.sort(rng.choice(len(shard['y_test']),size=min(2048,len(shard['y_test'])),replace=False))
        inputs=shard['X_test'][idx];truth=shard['y_test'][idx].numpy()
        model,det=_make_denice_client_model(ck,cid,device);model.eval()
        before=digest(model.state_dict());memory=service.continuous_routers[cid]
        raw=memory.state_dict();parts=[]
        with torch.no_grad():
            for lo in range(0,len(truth),512):
                acts=model.get_context_activations_per_sample(inputs[lo:lo+512].to(device))
                features=np.concatenate([acts[k].cpu().numpy() for k in ('conv1','conv2','conv3','gru')],axis=1)[:,raw['indices']]
                scores=route_scores(raw,features)
                order=np.argsort(-scores,axis=1,kind='stable')
                # Verify direct saved-coefficient decoder reproduces the production task choice.
                expected,_=memory.predict(features)
                assert np.array_equal(order[:,0],expected)
                t1=order[:,0];has2=np.isfinite(scores[np.arange(len(scores)),order[:,1]])
                t2=np.where(has2,order[:,1],t1)
                xx=inputs[lo:lo+512].reshape(-1,39).cpu();yy=truth[lo:lo+512]
                native1,routes=base_predictions(model,FixedRouteView(det,memory,t1),xx,list(range(34)),True)
                assert np.array_equal(routes,t1)
                pred1,_=service.infer(cid,'coverage',native1,xx,predicted_tasks=t1)
                native2,routes2=base_predictions(model,FixedRouteView(det,memory,t2),xx,list(range(34)),True)
                assert np.array_equal(routes2,t2)
                pred2,_=service.infer(cid,'coverage',native2,xx,predicted_tasks=t2)
                second=(yy//6==t2)&(yy//6!=t1)&has2
                oracle=np.where(second,pred2,pred1)
                parts.append(dict(y=yy,pred=pred1,oracle=oracle,top1=t1==yy//6,
                    top2=(yy//6==t1)|second,second=second,
                    absent=~np.isin(yy//6,list(set(memory.class_to_task.values()))),
                    native2_ok=(native2==yy)&second,v4second_ok=(pred2==yy)&second,
                    gap=np.where(has2,scores[np.arange(len(scores)),t1]-scores[np.arange(len(scores)),t2],np.inf)))
        assert before==digest(model.state_dict())
        data={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
        local={k:np.zeros((34,34),dtype=np.int64) for k in cms}
        np.add.at(local['baseline'],(data['y'],data['pred']),1)
        np.add.at(local['oracle_top2'],(data['y'],data['oracle']),1)
        cnt=dict(rows=len(truth),top1_correct_task=int(data['top1'].sum()),top2_contains_true_task=int(data['top2'].sum()),
                 true_task_second=int(data['second'].sum()),task_absent=int(data['absent'].sum()),
                 rescued=int(((data['oracle']==truth)&(data['pred']!=truth)).sum()),
                 broken=int(((data['oracle']!=truth)&(data['pred']==truth)).sum()),
                 second_task_native_correct=int(data['native2_ok'].sum()),second_task_v4_correct=int(data['v4second_ok'].sum()))
        for k,v in cnt.items():counts[k]+=v
        for k,v in local.items():cms[k]+=v
        result['clients'][cid]=dict(counts=cnt,metrics={k:statistics(v) for k,v in local.items()},
            rescued_router_gap_quantiles=np.quantile(data['gap'][(data['oracle']==truth)&(data['pred']!=truth)],[0,.25,.5,.75,1]).tolist() if cnt['rescued'] else [],
            native_weights_unchanged=True)
        del model,det,shards[cid]
        if position%10==0:
            print(f'Top2 diagnostic {position}/100: rows={counts["rows"]}, oracle rescues={counts["rescued"]}',flush=True)
    assert expert_before==digest(service.state_dict()['engines'])
    result.update(completed=True,counts=counts,metrics={k:statistics(v) for k,v in cms.items()},
                  wall_seconds=time.perf_counter()-start,expert_weights_and_optimizer_unchanged=True)
    result['oracle_accuracy_gain_pp']=100*(result['metrics']['oracle_top2']['accuracy']-result['metrics']['baseline']['accuracy'])
    result['task_top1_accuracy']=counts['top1_correct_task']/counts['rows']
    result['task_top2_recall']=counts['top2_contains_true_task']/counts['rows']
    (out/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['counts','oracle_accuracy_gain_pp','task_top1_accuracy','task_top2_recall','wall_seconds']},indent=2),flush=True)


if __name__=='__main__':main()
