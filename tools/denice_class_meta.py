"""Permutation-invariant class evidence, with a fixed routed-expert action set."""
import numpy as np


def class_features(features,pred,scores,names,labels):
    """Aggregate available predictions; neither input labels nor true tasks enter."""
    pred=np.asarray(pred,dtype=np.int64);scores=np.asarray(scores,dtype=float)
    if pred.ndim!=2 or pred.shape[1]==0 or scores.shape!=pred.shape or not np.isfinite(scores).all() or (scores<0).any():
        raise ValueError('Invalid expert score matrix')
    n,k=pred.shape;expert=np.asarray(features).reshape(n,k,-1)
    labels=np.asarray(labels,dtype=np.int64)
    if len(np.unique(labels))!=len(labels) or not np.isin(pred,labels).all():raise ValueError('Invalid class space')
    raw={name:expert[:,:,names.index(name)].astype(np.float64) for name in [
        'router_confidence','class_confidence','class_margin','class_entropy','mask_count',
        'calibration_donor_rate','calibration_donor_task_rate','calibration_receiver_donor_task_rate','aggregation_alpha']}
    output=[];schema=[];available=np.zeros((n,len(labels)),dtype=bool)
    for c,label in enumerate(labels):
        mask=pred==label;count=mask.sum(1);denom=np.maximum(count,1);available[:,c]=count>0
        def total(value):return np.where(mask,value,0.).sum(1,dtype=np.float64)
        maximum=np.where(mask,scores,-np.inf).max(1)
        values=[count/k,total(scores)/k,np.where(count>0,maximum,0.),total(scores)/denom,
            (pred[:,0]==label).astype(float)]
        stat_names=['vote_fraction','gate_score_sum_per_model','gate_score_max','gate_score_mean','self_predicts']
        for name,value in raw.items():
            if name=='mask_count':value=np.log1p(value)
            values.append(total(value)/denom);stat_names.append(name+'_mean')
        for name in ('router_confidence','class_confidence','class_margin'):
            values.append(np.where(count>0,np.where(mask,raw[name],-np.inf).max(1),0.))
            stat_names.append(name+'_max')
        output.extend(values);schema.extend([f'class_{label}_{name}' for name in stat_names])
    matrix=np.stack(output,axis=1).astype(np.float32)
    if not np.isfinite(matrix).all():raise ValueError('Non-finite class evidence')
    return matrix,available,schema


def masked_class_decision(probabilities,model_classes,labels,available,self_prediction,fallback):
    """Choose a class already predicted by a queried expert; no truth accepted."""
    prob=np.asarray(probabilities,dtype=np.float64);labels=np.asarray(labels,dtype=np.int64)
    mask=np.asarray(available,dtype=bool);own=np.asarray(self_prediction,dtype=np.int64)
    fallback=np.asarray(fallback,dtype=np.int64)
    if (prob.ndim!=2 or own.shape!=(len(prob),) or fallback.shape!=(len(prob),)
        or len(np.unique(labels))!=len(labels) or not np.all(labels[:-1]<labels[1:])
        or not np.isin(own,labels).all() or not np.isin(fallback,labels).all()
        or mask.ndim!=2 or mask.shape!=(len(prob),len(labels)) or not mask.any(1).all()
        or not np.isfinite(prob).all() or (prob<0).any() or prob.shape[1]!=len(model_classes)):
        raise ValueError('Invalid masked class probabilities')
    aligned=np.zeros(mask.shape,dtype=np.float64);lookup={int(c):i for i,c in enumerate(labels)}
    for column,label in enumerate(model_classes):
        if int(label) not in lookup:raise ValueError('Estimator predicts outside declared classes')
        aligned[:,lookup[int(label)]]=prob[:,column]
    aligned=np.where(mask,aligned,-1.)
    chosen=aligned.argmax(1);maximum=aligned[np.arange(len(prob)),chosen]
    own_columns=np.asarray([lookup[int(c)] for c in own])
    own_tied=aligned[np.arange(len(prob)),own_columns]==maximum
    chosen=np.where(own_tied,own_columns,chosen)
    result=labels[chosen];result=np.where(maximum<=0,fallback,result)
    if not mask[np.arange(len(prob)),np.asarray([lookup[int(c)] for c in result])].all():
        raise ValueError('Class prediction outside routed-expert action set')
    return result


def balanced_resample_indices(labels,seed=20261005):
    """Balance reachable fitting classes, keeping approximately the original count."""
    y=np.asarray(labels);classes=np.unique(y);rng=np.random.default_rng(seed)
    if len(classes)==0:raise ValueError('Empty fitting classes')
    target=max(1,len(y)//len(classes));indices=[]
    for label in classes:
        rows=np.flatnonzero(y==label)
        indices.extend(rng.choice(rows,target,replace=len(rows)<target).tolist())
    return rng.permutation(indices)
