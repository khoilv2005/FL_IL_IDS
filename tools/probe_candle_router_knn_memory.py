"""One fixed nonlinear probe; fit saved sketches only, never TEST labels."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import pathlib,json,sys
import torch,numpy as np
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits


def main():
    out=pathlib.Path('audit_denice/candle_v4_mask_router_audit_20261024')
    panel=json.loads((out/'router_t23_drift.json').read_text());assert panel['completed']
    ck=torch.load(r'C:\Users\khoak\Downloads\candle_v4_debug\continuation_state_latest.pt',map_location='cpu',weights_only=False,mmap=True)
    result=dict(completed=False,protocol='Fixed k=3, Hamming, distance weights; no hyperparameter search; fit saved memory only',
                no_TEST_fit=True,training_backbone=False,clients={})
    for cid in panel['client_ids']:
        a=ck['client_algorithm_states'][cid];ctx=a.get('denice',a)['context_detector'];memory=ctx['activation_memory']
        x=np.concatenate([np.asarray(v) for t,v in sorted(memory.items()) if len(v)]);y=np.concatenate([np.full(len(v),int(t)) for t,v in sorted(memory.items()) if len(v)])
        train=[];hold=[];rng=np.random.default_rng(42)
        for t in np.unique(y):
            ix=rng.permutation(np.flatnonzero(y==t));n=max(1,len(ix)//5);hold.extend(ix[:n]);train.extend(ix[n:])
        def model():return KNeighborsClassifier(n_neighbors=3,weights='distance',metric='hamming',algorithm='brute',n_jobs=2)
        with threadpool_limits(limits=2):
            fitted=model().fit(x[train],y[train]);hp=fitted.predict(x[hold])
            final=model().fit(x,y)
            with np.load(out/f'router_panel_client_{cid}.npz') as saved:
                bits=saved['bits_T5'];truth=saved['truth']//6;owned=saved['owned'];pred=final.predict(bits)
        result['clients'][str(cid)]=dict(rows=len(truth),correct=int((pred==truth).sum()),owned_rows=int(owned.sum()),
            owned_correct=int(((pred==truth)&owned).sum()),unowned_correct=int(((pred==truth)&~owned).sum()),
            memory_holdout_rows=len(hold),memory_holdout_correct=int((hp==y[hold]).sum()),
            per_task={str(t):dict(rows=int((truth==t).sum()),correct=int(((truth==t)&(pred==t)).sum())) for t in [2,3]})
        print(cid,result['clients'][str(cid)],flush=True)
    c=list(result['clients'].values());n=sum(v['rows'] for v in c);owned=sum(v['owned_rows'] for v in c)
    result['summary']=dict(rows=n,accuracy=sum(v['correct'] for v in c)/n,owned_accuracy=sum(v['owned_correct'] for v in c)/owned,
        unowned_accuracy=sum(v['unowned_correct'] for v in c)/(n-owned),
        memory_holdout_accuracy=sum(v['memory_holdout_correct'] for v in c)/sum(v['memory_holdout_rows'] for v in c))
    result['completed']=True;(out/'knn_memory_probe.json').write_text(json.dumps(result,indent=2));print(result['summary'],flush=True)


if __name__=='__main__':main()
