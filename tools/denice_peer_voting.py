"""Label-free peer ranking and voting, independent of diagnostic labels."""
import numpy as np


def peer_orders(cid, peers, alphas, random_seeds=(42,43,44,45,46)):
    candidates=sorted(int(p) for p in peers if int(p)!=int(cid))
    top=sorted(candidates,key=lambda p:(-float(alphas.get(p,0.0)),p))
    orders={'alpha':top}
    for seed in random_seeds:
        rng=np.random.default_rng(int(seed)+int(cid)*1009)
        orders[f'random_{seed}']=rng.permutation(candidates).tolist()
    return orders


def vote(predictions, weights, labels, self_prediction):
    """Weighted one-hot class votes. Self wins exact ties, then smallest label.

    predictions: [experts, samples]; weights: [experts] or [experts, samples].
    No labels of the input samples are accepted by this API.
    """
    pred=np.asarray(predictions,dtype=np.int64)
    w=np.asarray(weights,dtype=np.float64)
    labels=np.asarray(sorted(labels),dtype=np.int64)
    self_pred=np.asarray(self_prediction,dtype=np.int64)
    if (pred.ndim!=2 or pred.shape[0]==0 or self_pred.ndim!=1
            or len(self_pred)!=pred.shape[1] or len(labels)==0
            or len(np.unique(labels))!=len(labels)
            or not np.isin(pred,labels).all() or not np.isin(self_pred,labels).all()):
        raise ValueError('Invalid class-vote inputs')
    if w.ndim==1: w=np.broadcast_to(w[:,None],pred.shape)
    if w.shape!=pred.shape or not np.isfinite(w).all() or (w<0).any():
        raise ValueError('Invalid vote weights')
    scores=np.zeros((pred.shape[1],len(labels)),dtype=np.float64)
    for index in range(len(pred)):
        columns=np.searchsorted(labels,pred[index])
        scores[np.arange(len(self_pred)),columns]+=w[index]
    chosen=scores.argmax(1)
    self_col=np.searchsorted(labels,self_pred)
    maxima=scores[np.arange(len(self_pred)),chosen]
    self_tied=scores[np.arange(len(self_pred)),self_col]==maxima
    chosen=np.where(self_tied,self_col,chosen)
    return labels[chosen]
