"""One bounded read-only router audit before fresh V4-only training.

Global decomposition uses saved full TEST confusion matrices. A fixed six-client
panel reads up to128 samples/class, with frozen model and stored LR coefficients.
No fitting, optimizer steps, new threshold or production prediction rule.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'candle_v4_runtime'))
from eval_checkpoint import _make_denice_client_model
from fed_learning.training.candle_v4_shadow import digest
from fed_learning.training.decentralized_denice_il import _partition_test_data_by_client


def class_probabilities(state,x):
    """Decode saved coefficients directly; do not refit with local sklearn."""
    classes=np.asarray(state['fitted_classes']);x=np.asarray(x,dtype=np.float32).copy()
    x-=state['scaler']['mean_'];x/=state['scaler']['scale_']
    if len(classes)==1:return classes,np.ones((len(x),1))
    clf=state['classifier'];scores=x@clf['coef_'].T+clf['intercept_']
    if len(classes)==2:
        v=np.clip(scores[:,0],-700,700);p=1/(1+np.exp(-v));return classes,np.column_stack([1-p,p])
    scores-=scores.max(1,keepdims=True);p=np.exp(scores);p/=p.sum(1,keepdims=True)
    return classes,p


def route_scores(state,x):
    classes,p=class_probabilities(state,x);scores=np.full((len(x),6),-np.inf)
    for j,c in enumerate(classes):scores[:,int(c)//6]=np.maximum(scores[:,int(c)//6],p[:,j])
    return scores


def main():
    out=ROOT/'audit_denice/candle_v4_router_final_preflight_20261026';out.mkdir(parents=True,exist_ok=True)
    warnings.filterwarnings('ignore',message='Trying to unpickle estimator')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ck=torch.load(ROOT/'audit_denice/candle_v4_results17_20261026/continuation_state_latest.pt',map_location='cpu',weights_only=False,mmap=True)
    bundle=torch.load(ROOT/'audit_denice/candle_v4_results18_20261026/historical_router_repair_bundle.pt',map_location='cpu',weights_only=False)
    routers={**ck['candle_v4_shadow_state']['continuous_routers'],**bundle['routers']}
    final=json.loads((ROOT/'audit_denice/candle_v4_results18_20261026/candle_v4_final_test.json').read_text())
    total=final['test_rows'];groups={k:dict(rows=0,route_correct=0,class_correct=0) for k in ('task_missing','class_missing_but_task_present','class_in_memory')}
    owners={};global_tasks=np.zeros((6,6),dtype=np.int64);mismatches=[]
    for cid,state in sorted(routers.items()):
        alg=ck['client_algorithm_states'][cid];native=alg.get('denice',alg)['context_detector']
        tasks={int(t) for t,v in native['activation_memory'].items() if len(v)}
        if tasks!=set(state['class_to_task'].values()):mismatches.append(cid)
        assert all(int(c)//6==int(t) for t,classes in native['episode_classes'].items() for c in classes)
        cm=np.asarray(final['per_client'][str(cid)]['coverage']['confusion_matrix']);n=int(cm.sum());taskcm=np.zeros((6,6),dtype=np.int64)
        local={k:dict(rows=0,route_correct=0,class_correct=0) for k in groups}
        for c in range(34):
            t=c//6;key='task_missing' if t not in tasks else ('class_in_memory' if c in state['memory'] else 'class_missing_but_task_present')
            values=dict(rows=int(cm[c].sum()),route_correct=int(cm[c,t*6:min(34,t*6+6)].sum()),class_correct=int(cm[c,c]))
            for k,v in values.items():groups[key][k]+=v;local[key][k]+=v
            for pred in range(34):taskcm[t,pred//6]+=cm[c,pred]
        global_tasks+=taskcm
        xx=np.concatenate([v for c,v in sorted(state['memory'].items())]);yy=np.concatenate([np.full(len(v),int(c)) for c,v in sorted(state['memory'].items())])
        train_scores=route_scores(state,xx);pred=train_scores.argmax(1)
        owners[str(cid)]=dict(tasks=sorted(tasks),rows=n,route_accuracy=float(np.trace(taskcm)/n),
            v4_accuracy=float(np.trace(cm)/n),task_support_ceiling=1-local['task_missing']['rows']/n,
            groups=local,saved_memory_rows=len(yy),saved_memory_task_accuracy=float((pred==yy//6).mean()),
            class_counts_per_task={str(t):sum(int(c)//6==t for c in state['memory']) for t in sorted(tasks)},
            task_confusion=taskcm.tolist())
    assert not mismatches
    assert np.array_equal(global_tasks,np.asarray(final['router_metrics']['confusion_matrix'])[:,:6])
    route_ok=int(np.trace(global_tasks));class_ok=sum(v['class_correct'] for v in groups.values())
    errors=dict(missing_task=groups['task_missing']['rows'],wrong_route_with_task_present=total-route_ok-groups['task_missing']['rows'],
                correct_route_wrong_class=route_ok-class_ok)
    assert sum(errors.values())==total-class_ok
    for values in groups.values():
        values['route_accuracy']=values['route_correct']/values['rows'] if values['rows'] else None
    result=dict(completed=False,protocol='frozen_results18_router_final_preflight',no_fit=True,no_training=True,
        no_runtime_or_threshold_changes=True,router_support_mismatches=mismatches,total_rows=total,
        global_route_accuracy=route_ok/total,global_class_accuracy=class_ok/total,
        task_support_ceiling=1-groups['task_missing']['rows']/total,
        route_accuracy_when_task_present=route_ok/(total-groups['task_missing']['rows']),
        class_accuracy_when_route_correct=class_ok/route_ok,error_decomposition=errors,groups=groups,
        task_confusion=global_tasks.tolist(),clients=owners,
        panel_selection='Fixed19/38 low-support,1/40 repair regressions,35/43 successful controls; max128/class; seed20261026+cid',panel={})
    (out/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Global error decomposition:',errors,flush=True)
    path=Path(r'C:\Users\khoak\Downloads\candle_v1_multiclass_audit\global_test_data.npz')
    with np.load(path) as z:x=torch.from_numpy(z['X_test']);y=torch.from_numpy(z['y_test']).long()
    shards,_=_partition_test_data_by_client(x,y,sorted(ck['client_ids']),seed=523687,lazy_features=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    for cid in [19,38,1,40,35,43]:
        labels=shards[cid]['y_test'].numpy();rng=np.random.default_rng(20261026+cid)
        idx=np.concatenate([rng.permutation(np.flatnonzero(labels==c))[:128] for c in range(34)])
        inputs=shards[cid]['X_test'][idx];truth=labels[idx];state=routers[cid]
        model,det=_make_denice_client_model(ck,cid,device);before=digest(model.state_dict());parts=[]
        with torch.no_grad():
            for start in range(0,len(inputs),512):
                acts=model.get_context_activations_per_sample(inputs[start:start+512].to(device))
                parts.append(np.concatenate([acts[k].cpu().numpy() for k in ('conv1','conv2','conv3','gru')],axis=1)[:,state['indices']])
        features=np.concatenate(parts);scores=route_scores(state,features);order=np.argsort(-scores,axis=1,kind='stable')
        expected=truth//6;rank=np.argmax(order==expected[:,None],axis=1)+1
        available=np.isin(expected,list(set(state['class_to_task'].values())));owned=np.isin(truth,list(state['memory']))
        top2=np.any(order[:,:2]==expected[:,None],axis=1)&available
        correct=order[:,0]==expected;z=features.astype(np.float32).copy();z-=state['scaler']['mean_'];z/=state['scaler']['scale_']
        panel=dict(rows=len(truth),route_accuracy=float(correct.mean()),top2_task_recall_diagnostic=float(top2.mean()),
            unavailable_task_rows=int((~available).sum()),
            route_on_memory_classes=float(correct[owned].mean()) if owned.any() else None,
            route_on_unrepresented_classes_with_task=float(correct[~owned&available].mean()) if (~owned&available).any() else None,
            true_task_rank_histogram={str(i):int(((rank==i)&available).sum()) for i in range(1,7)},
            standardized_abs_quantiles=np.quantile(np.abs(z),[.5,.95,.99,1]).tolist(),
            finite_scores=bool(np.isfinite(scores[:,list(set(state['class_to_task'].values()))]).all()),
            weights_unchanged=before==digest(model.state_dict()),note='Top2 is diagnostic only, no inference rule changed')
        result['panel'][str(cid)]=panel
        (out/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print('Panel',cid,'rows',len(truth),'route',panel['route_accuracy'],'top2',panel['top2_task_recall_diagnostic'],flush=True)
        del model,det
    result['completed']=True
    (out/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()
