"""Lightweight expert competence features and label-free gate decisions."""
import hashlib
import numpy as np
from tools.denice_peer_voting import vote


def content_hash(row):
    value=np.ascontiguousarray(row,dtype=np.float32)
    return hashlib.sha256(str(value.shape).encode()+value.tobytes()).hexdigest()


def content_role(digest):
    # Equal content always has the same role, including across receiver shards.
    bucket=int(digest[:8],16)%10
    return 'calibration' if bucket<2 else ('validation' if bucket<4 else 'fit')


def competence_priors(cache, labels):
    """Estimate priors exclusively from the calibration partition."""
    global_task={};local_task={};global_donor={}
    def add(table,key,correct):
        n,hits=table.get(key,(0,0));table[key]=(n+len(correct),hits+int(correct.sum()))
    for cid,donors in cache.items():
        y=np.asarray(labels[cid])
        for donor,record in donors.items():
            correct=record['pred']==y
            add(global_donor,donor,correct)
            for task in np.unique(record['task']):
                selected=record['task']==task
                add(global_task,(donor,int(task)),correct[selected])
                add(local_task,(cid,donor,int(task)),correct[selected])
    return dict(global_task=global_task,local_task=local_task,global_donor=global_donor)


def feature_matrix(cid,chosen,records,alphas,priors,n_tasks=6,n_classes=34):
    """No sample labels accepted; matrix is sample-major, then expert-major."""
    pred=np.stack([records[d]['pred'] for d in chosen],axis=1)
    task=np.stack([records[d]['task'] for d in chosen],axis=1)
    n,k=pred.shape
    if ((pred<0).any() or (pred>=n_classes).any() or (task<0).any() or (task>=n_tasks).any()):
        raise ValueError('Prediction/task outside declared label space')
    agreement=np.stack([(pred==pred[:,j,None]).mean(1) for j in range(k)],axis=1)
    task_agreement=np.stack([(task==task[:,j,None]).mean(1) for j in range(k)],axis=1)
    scalar_names=['router_confidence','class_confidence','class_margin','class_entropy',
        'mask_count','task_supported','class_supported']
    columns=[np.stack([records[d][name] for d in chosen],axis=1) for name in scalar_names]
    columns.extend([agreement,task_agreement,(pred==pred[:,0,None]).astype(float),
        (task==task[:,0,None]).astype(float),np.broadcast_to(np.asarray([alphas.get(d,0.) for d in chosen]),(n,k)),
        np.broadcast_to(np.asarray([float(d==cid) for d in chosen]),(n,k))])
    overall=np.empty((n,k));global_rate=np.empty((n,k));local_rate=np.empty((n,k))
    for j,d in enumerate(chosen):
        count,hits=priors['global_donor'].get(d,(0,0));base=(hits+1)/(count+2)
        overall[:,j]=base
        for t in np.unique(task[:,j]):
            count,hits=priors['global_task'].get((d,int(t)),(0,0));g=(hits+20*base)/(count+20)
            count,hits=priors['local_task'].get((cid,d,int(t)),(0,0));local=(hits+10*g)/(count+10)
            rows=task[:,j]==t;global_rate[rows,j]=g;local_rate[rows,j]=local
    columns.extend([overall,global_rate,local_rate])
    continuous=np.stack(columns,axis=2).reshape(n*k,-1)
    features=np.concatenate([continuous,np.eye(n_tasks)[task.reshape(-1)],
                              np.eye(n_classes)[pred.reshape(-1)]],axis=1).astype(np.float32)
    if not np.isfinite(features).all():raise ValueError('Non-finite competence features')
    names=scalar_names+['class_agreement','task_agreement','agrees_self_class','agrees_self_task',
        'aggregation_alpha','is_self','calibration_donor_rate','calibration_donor_task_rate',
        'calibration_receiver_donor_task_rate']+[f'predicted_task_{t}' for t in range(n_tasks)]+[
        f'predicted_class_{c}' for c in range(n_classes)]
    return features,pred,names


def gate_decision(scores,pred,chosen,labels,top=1):
    """Choose by learned score; ties favor self then donor ID, never truth."""
    scores=np.asarray(scores,dtype=float);pred=np.asarray(pred,dtype=np.int64)
    if scores.shape!=pred.shape or not np.isfinite(scores).all() or (scores<0).any():
        raise ValueError('Invalid gate scores')
    if top<1 or top>len(chosen):raise ValueError('Invalid gate expert budget')
    priority=np.asarray([0]+[d+1 for d in chosen[1:]])
    order=np.argsort(priority,kind='stable')
    ranking=order[np.argsort(-scores[:,order],axis=1,kind='stable')]
    selected=ranking[:,:top]
    if top==1:
        return pred[np.arange(len(pred)),selected[:,0]],np.asarray(chosen)[selected[:,0]]
    weights=np.zeros_like(scores)
    np.put_along_axis(weights,selected,np.take_along_axis(scores,selected,axis=1),axis=1)
    # Use one-hot sums over all candidates so zero-score fallback remains self.
    return vote(pred.T,weights.T,labels,pred[:,0]),np.asarray(chosen)[selected[:,0]]
