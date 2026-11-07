"""One metadata-selected current-task donor feature transfer, checkpoint development only."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import copy
import io
import json
import pickle
import sys
import warnings
import zipfile
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'candle_v4_runtime'))
from eval_checkpoint import _make_denice_client_model
from fed_learning.training.candle_v4_original_train import OriginalTrainService
from fed_learning.training.candle_v4_shadow import digest
from fed_learning.training.candle_v4_terminal_test import statistics
from fed_learning.training.candle_v4_continuous_runtime import predictions
from fed_learning.training.candle_v4_bank import base_predictions
from fed_learning.servers.continuous_class_router import ContinuousClassMemory
from candle_v4_top2_task_select import FixedChoiceView, choose_task
from eval_candle_v4_task_round19_topk import candidates


def encode(model, inputs, indices):
    with torch.no_grad():
        acts=model.get_context_activations_per_sample(inputs.to(next(model.parameters()).device))
    return np.concatenate([acts[k].cpu().numpy() for k in ('conv1','conv2','conv3','gru')],axis=1)[:,indices]


def main():
    warnings.filterwarnings('ignore',message='Trying to unpickle estimator')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=ROOT/'audit_denice/candle_v4_receiver_routing_features_20261107';out.mkdir(parents=True,exist_ok=True)
    ck=torch.load(ROOT/'audit_denice/candle_v4_results20_20261027/continuation_state_latest.pt',map_location='cpu',weights_only=False,mmap=True)
    assert ck['meta']['completed_task']==5 and ck['meta']['boundary']=='task'
    service=OriginalTrainService(ck['candle_v4_shadow_state']['spec'],ck['candle_v4_shadow_state'])
    before_service=digest(service.state_dict())
    eligible=[]
    for key,entry in sorted(service.entries.items()):
        r,c=key
        if entry['task']!=5 or service.status['coverage'].get(key)!='ACTIVE':continue
        mem=service.continuous_routers[r]
        if c in mem.memory or 5 not in mem.class_to_task.values():continue
        for row in service.history:
            if row['task']!=5 or row['round']!=19:continue
            for item in row['experts']:
                if item['receiver']==r and item['target']==c and item['arm']=='coverage' and item['updates']>0:
                    auth=item.get('donor_authorization',{})
                    if item['eligible'] and auth.get('authorized') and item['donor']==entry['donor']:
                        eligible.append((r,c,entry['donor'],row['graph_sha256'],auth))
    if not eligible:raise RuntimeError('No current-task metadata-eligible fixture; do not search by TEST score')
    r,c,d,graph_hash,auth=sorted(eligible,key=lambda v:v[:3])[0]
    memory=service.continuous_routers[r]
    fixture=dict(receiver=r,target=c,donor=d,task=5,round=19,graph_hash=graph_hash,
                 authorization=auth,candidate_count=len(eligible),selection='lexicographic receiver,class,donor',
                 feature_budget=memory.per_class,seed=42,test_opened=False,
                 scope='Frozen post-consolidation Task5 diagnostic; no native round integration or retention claim')
    (out/'fixture.json').write_text(json.dumps(fixture,indent=2),encoding='utf-8')
    print('Locked fixture',r,c,d,'feature cap',memory.per_class,flush=True)
    model,det=_make_denice_client_model(ck,r,'cuda');model.eval();before_model=digest(model.state_dict())
    # Application-wire simulation: donor receives an exact receiver function reference.
    capsule=pickle.dumps(dict(model=copy.deepcopy(model).cpu(),indices=memory.indices.copy(),
                              model_sha256=before_model,graph_sha256=graph_hash,task=5),protocol=5)
    donor_reference=pickle.loads(capsule);donor_model=donor_reference['model'].to('cuda').eval()
    assert digest(donor_model.state_dict())==before_model
    with zipfile.ZipFile(r'C:/Users/khoak/Downloads/archive (1).zip') as archive:
        with np.load(io.BytesIO(archive.read(f'100-clients/client_{d}_train.npz'))) as data:
            labels=data['y_train'];target_indices=np.flatnonzero(labels==c)
            assert len(target_indices)>=32 and c//6==5
            selected=np.sort(np.random.default_rng(42).choice(target_indices,min(memory.per_class,len(target_indices)),replace=False))
            donor_inputs=torch.from_numpy(data['X_train'][selected].astype(np.float32))
    features=encode(donor_model,donor_inputs,donor_reference['indices'])
    assert np.allclose(features,encode(model,donor_inputs,memory.indices),rtol=0,atol=0)
    packet=pickle.dumps(dict(features=features,class_id=c,task=5,donor=d,receiver=r,
                            model_sha256=before_model,graph_sha256=graph_hash,train_indices=selected),protocol=5)
    received=pickle.loads(packet)
    assert received['model_sha256']==digest(model.state_dict()) and np.isfinite(received['features']).all()
    control=ContinuousClassMemory.restore(memory.state_dict());control.fit(5)
    augmented=ContinuousClassMemory.restore(memory.state_dict())
    augmented.memory[c]=received['features'].copy();augmented.class_to_task[c]=5;augmented.fit(5)
    # All fitting ends before TEST is opened; no target threshold tuning.
    print('TRAIN features transferred; fitting complete; opening diagnostic TEST panel',flush=True)
    with np.load(r'C:/Users/khoak/Downloads/candle_v1_multiclass_audit/global_test_data.npz') as data:
        all_y=data['y_test'];rng=np.random.default_rng(20261107)
        idx=np.concatenate([rng.choice(np.flatnonzero(all_y==label),min(128,int((all_y==label).sum())),replace=False) for label in range(34)])
        truth=all_y[idx].astype(np.int64);inputs=torch.from_numpy(data['X_test'][idx].astype(np.float32))
    variants=('baseline','refit_control','augmented_top2','augmented_top1')
    preds={v:[] for v in variants};routes={v:[] for v in variants};candidate_hit=[]
    for lo in range(0,len(truth),512):
        xx=inputs[lo:lo+512].reshape(-1,39)
        fallback,first=predictions(service,r,model,det,xx,list(range(34)))
        base=service.infer(r,'coverage',fallback,xx,predicted_tasks=first)[0]
        z=encode(model,xx,memory.indices);check,score=memory.predict(z)
        assert np.array_equal(first,check)
        top2=candidates(first,score,set(memory.class_to_task.values()))
        cand,scores=augmented.predict(z); padded=np.zeros((len(z),6));padded[:,:scores.shape[1]]=scores
        choices=dict(baseline=first,refit_control=control.predict(z)[0],
                     augmented_top2=choose_task(top2,padded),augmented_top1=cand)
        candidate_hit.extend(np.any(top2==truth[lo:lo+512,None]//6,axis=1).tolist())
        for v,choice in choices.items():
            if v=='baseline': pred=base
            else:
                native,actual=base_predictions(model,FixedChoiceView(det,memory,choice),xx,list(range(34)),True)
                assert np.array_equal(actual,choice)
                pred=service.infer(r,'coverage',native,xx,predicted_tasks=choice)[0]
            preds[v].extend(pred.tolist());routes[v].extend(choice.tolist())
    result=dict(fixture=fixture,completed=True,panel='128 samples/class, up to 34 classes; class-balanced global TEST development, not official pooled benchmark',
                rows=len(truth),seed=20261107,metrics={},communication=dict(receiver_capsule_bytes=len(capsule),feature_packet_bytes=len(packet),total_bytes=len(capsule)+len(packet)),
                no_raw_data_sent=True,fit_data='current Task5 donor TRAIN + existing receiver saved TRAIN features',
                model_unchanged=before_model==digest(model.state_dict()),service_unchanged=before_service==digest(service.state_dict()),
                original_top2_target_task_coverage=float(np.mean(np.asarray(candidate_hit)[truth==c])),
                imported_rows=len(features),routing_feature_width=features.shape[1],quality_status='development only; no causal native training/next-task retention evidence')
    base=np.asarray(preds['baseline']);target=truth==c
    for v in variants:
        p=np.asarray(preds[v]);rt=np.asarray(routes[v]);cm=np.zeros((34,34),dtype=np.int64);np.add.at(cm,(truth,p),1)
        result['metrics'][v]=dict(**statistics(cm),route_accuracy=float(np.mean(rt==truth//6)),
            target_recall=float(np.mean(p[target]==c)),target_route_accuracy=float(np.mean(rt[target]==5)),
            target_false_positives=int(np.sum((p==c)&~target)),rescue=int(np.sum((p==truth)&(base!=truth))),
            break_count=int(np.sum((p!=truth)&(base==truth))))
    assert result['model_unchanged'] and result['service_unchanged']
    (out/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({v:{k:result['metrics'][v][k] for k in ('accuracy','macro_f1','route_accuracy','target_recall','target_route_accuracy','target_false_positives','rescue','break_count')} for v in variants},indent=2),flush=True)
    print('Wire bytes',result['communication'],flush=True)


if __name__=='__main__':main()
