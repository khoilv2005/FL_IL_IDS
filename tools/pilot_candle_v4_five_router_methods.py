"""Five fixed literature-inspired task rerankers, saved TRAIN features only.

Adaptations for routing, not reproductions of the cited full CL algorithms.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import json,sys,time,warnings
from pathlib import Path
import numpy as np
import torch
from sklearn.covariance import LedoitWolf
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'candle_v4_runtime'))
from eval_checkpoint import _make_denice_client_model
from fed_learning.training.candle_v4_original_train import OriginalTrainService
from fed_learning.training.candle_v4_shadow import digest
from fed_learning.training.candle_v4_terminal_test import statistics
from fed_learning.training.decentralized_denice_il import _partition_test_data_by_client
from fed_learning.training.candle_v4_continuous_runtime import predictions
from fed_learning.training.candle_v4_bank import base_predictions
from candle_v4_top2_task_select import FixedChoiceView,choose_task
from eval_candle_v4_task_round19_topk import candidates

METHODS=('ncm_euclidean','slda_shared_covariance','fecam_shrinkage','ranpac_ridge','knn5')

class Scorers:
    def __init__(self,memory):
        self.scaler=memory.scaler;self.classes=np.array(sorted(memory.memory))
        self.x=self.scaler.transform(np.concatenate([memory.memory[c] for c in self.classes]))
        self.y=np.concatenate([np.full(len(memory.memory[c]),i) for i,c in enumerate(self.classes)])
        self.means=np.stack([self.x[self.y==i].mean(0) for i in range(len(self.classes))])
        residual=self.x-self.means[self.y]
        self.precision=LedoitWolf().fit(residual).precision_
        # Class-specific covariance shrinkage; no FeCAM Tukey/correlation transforms.
        self.class_precision=[LedoitWolf().fit(self.x[self.y==i]).precision_ if np.sum(self.y==i)>1 else self.precision.copy() for i in range(len(self.classes))]
        rng=np.random.default_rng(42);self.projection=rng.normal(size=(self.x.shape[1],256))/np.sqrt(self.x.shape[1])
        h=np.maximum(self.x@self.projection,0)
        labels=np.eye(len(self.classes))[self.y]
        self.ridge=np.linalg.solve(h.T@h+np.eye(256),h.T@labels)

    def score(self,z):
        x=self.scaler.transform(z);delta=x[:,None,:]-self.means
        raw={
            'ncm_euclidean':-np.sum(delta**2,axis=2),
            'slda_shared_covariance':-np.einsum('ncd,de,nce->nc',delta,self.precision,delta,optimize=True),
            'fecam_shrinkage':np.column_stack([-np.einsum('nd,de,ne->n',delta[:,i],p,delta[:,i],optimize=True) for i,p in enumerate(self.class_precision)]),
            'ranpac_ridge':np.maximum(x@self.projection,0)@self.ridge,
        }
        dist=np.maximum(np.sum(x*x,1)[:,None]+np.sum(self.x*self.x,1)[None,:]-2*x@self.x.T,0)
        k=min(5,len(self.x));near=np.argsort(dist,axis=1,kind='stable')[:,:k]
        votes=np.zeros((len(x),len(self.classes)))
        for j in range(k):votes[np.arange(len(x)),self.y[near[:,j]]]+=1/k
        raw['knn5']=votes
        result={}
        for method,scores in raw.items():
            tasks=np.full((len(x),6),-np.inf)
            for j,c in enumerate(self.classes):tasks[:,c//6]=np.maximum(tasks[:,c//6],scores[:,j])
            result[method]=tasks
        return result

def main(scorer_type=Scorers,methods=METHODS,output_name='candle_v4_five_router_methods_20261107'):
    warnings.filterwarnings('ignore',message='Trying to unpickle estimator');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ck=torch.load(ROOT/'audit_denice/candle_v4_results20_20261027/continuation_state_latest.pt',map_location='cpu',weights_only=False,mmap=True)
    service=OriginalTrainService(ck['candle_v4_shadow_state']['spec'],ck['candle_v4_shadow_state']);frozen=digest(service.state_dict())
    ids=sorted(ck['client_model_states'])
    # All scorers fitted before opening TEST; no method/parameter selection from TEST.
    scorers={cid:scorer_type(service.continuous_routers[cid]) for cid in ids}
    print(f'{len(methods)} methods fitted on saved TRAIN memory; opening fixed development panel',flush=True)
    with np.load(r'C:/Users/khoak/Downloads/candle_v1_multiclass_audit/global_test_data.npz') as data:
        x=torch.from_numpy(data['X_test']);y=torch.from_numpy(data['y_test']).long()
    shards,_=_partition_test_data_by_client(x,y,ids,seed=523687,lazy_features=True)
    cms={m:np.zeros((34,34),dtype=np.int64) for m in ('baseline',)+methods}
    counts={m:dict(rescue=0,break_count=0,route_correct=0,switched=0) for m in cms};clients={};start=time.perf_counter()
    for pos,cid in enumerate(ids,1):
        shard=shards[cid];idx=np.sort(np.random.default_rng(20261028+cid).choice(len(shard['y_test']),2048,replace=False))
        xx=shard['X_test'][idx];truth=shard['y_test'][idx].numpy()
        model,det=_make_denice_client_model(ck,cid,'cuda');model.eval();before=digest(model.state_dict());mem=service.continuous_routers[cid]
        local={m:np.zeros((34,34),dtype=np.int64) for m in cms}
        for lo in range(0,len(truth),512):
            inputs=xx[lo:lo+512].reshape(-1,39).cpu();yy=truth[lo:lo+512]
            fallback,first=predictions(service,cid,model,det,inputs,list(range(34)))
            baseline=service.infer(cid,'coverage',fallback,inputs,predicted_tasks=first)[0]
            with torch.no_grad():acts=model.get_context_activations_per_sample(inputs.cuda())
            z=np.concatenate([acts[k].cpu().numpy() for k in ('conv1','conv2','conv3','gru')],axis=1)[:,mem.indices]
            check,score=mem.predict(z);assert np.array_equal(check,first)
            top2=candidates(first,score,set(mem.class_to_task.values()))
            choices={m:choose_task(top2,s) for m,s in scorers[cid].score(z).items()}
            # Only two task-specific predictions needed, shared by all five methods.
            second=np.where(top2[:,1]>=0,top2[:,1],first)
            native,routes=base_predictions(model,FixedChoiceView(det,mem,second),inputs,list(range(34)),True)
            assert np.array_equal(second,routes)
            pred2=service.infer(cid,'coverage',native,inputs,predicted_tasks=second)[0]
            choices['baseline']=first
            for method,choice in choices.items():
                pred=np.where(choice!=first,pred2,baseline)
                np.add.at(local[method],(yy,pred),1)
                v=counts[method];v['rescue']+=int(np.sum((pred==yy)&(baseline!=yy)));v['break_count']+=int(np.sum((pred!=yy)&(baseline==yy)))
                v['route_correct']+=int(np.sum(choice==yy//6));v['switched']+=int(np.sum(choice!=first))
        assert before==digest(model.state_dict())
        clients[cid]={m:statistics(cm)['accuracy'] for m,cm in local.items()}
        for m in cms:cms[m]+=local[m]
        del model,det,shards[cid]
        if pos%20==0:print(f'{len(methods)}-method evaluation {pos}/100',flush=True)
    prior=json.loads((ROOT/'audit_denice/candle_v4_results20_top2_20261028/audit.json').read_text())
    assert cms['baseline'].tolist()==prior['metrics']['baseline']['confusion_matrix']
    assert frozen==digest(service.state_dict())
    result=dict(completed=True,rows=204800,metrics={m:statistics(cm) for m,cm in cms.items()},counts=counts,clients=clients,
        fit_source='Saved receiver TRAIN activation memory only; all methods fitted before TEST open',
        panel='Repeated TEST development panel; not independent confirmation',no_parameter_search=True,
        baseline_exact_match=True,service_unchanged=True,native_weights_unchanged=True,
        added_network_bytes=0,seconds=time.perf_counter()-start)
    out=ROOT/'audit_denice'/output_name;out.mkdir(parents=True,exist_ok=True)
    (out/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({m:dict(accuracy=v['accuracy'],macro_f1=v['macro_f1'],**counts[m]) for m,v in result['metrics'].items()},indent=2),flush=True)

if __name__=='__main__':main()
