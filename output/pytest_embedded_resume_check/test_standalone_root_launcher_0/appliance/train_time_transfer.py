"""Class-specific readout training at the donor in receiver coordinates.

Prototype: frozen receiver encoder and old readout rows, one newly trained row.
The donor receives a receiver capsule and trains using its current owned BASE.
Receiver-current BASE provides a second regularized integration step. No raw
examples or per-example features cross the transport interface. There is no
imported route, donor model query at inference, or population safety claim.
"""
import copy
import io

import numpy as np
import torch
import torch.nn.functional as F

from .closure import effective_linear
from .config import Rejected

RULES = dict(version='appliance_train_time_readout_v1', seed=42,
    steps=128, integration_steps=64, learning_rate=.02, parameter_l2=.001,
    old_probability_regularization=1., cap_BASE_per_class=1024,
    cap_development_per_owner_class=256, batch_size=256,
    min_target_recall=.95, max_current_negative_FAR=.001,
    max_current_break=0, max_development_old_accuracy_drop=.01,
    max_development_old_false_positive_rate=.001)


def stratified_cap(pool, classes, cap, seed):
    rng=np.random.default_rng(seed); ids=[]
    for c in sorted(classes):
        ids.extend(rng.permutation(np.flatnonzero(pool['y']==c))[:cap].tolist())
    ids=np.asarray(ids,np.int64)
    return dict(pool,X=pool['X'][ids],y=pool['y'][ids],rows=pool['rows'][ids])


@torch.no_grad()
def receiver_features(model, x, device, batch_size=256):
    model.eval();model.clear_active_adapters()
    if getattr(model,'adapter_registry',{}) or model.continual_head is not None:
        raise Rejected('TRAIN_TIME_PROTOTYPE_ADAPTER_FREE_RECEIVER_REQUIRED')
    h=[]
    for start in range(0,len(x),batch_size):
        values=model.penultimate_features(torch.as_tensor(x[start:start+batch_size],
            dtype=torch.float32,device=device)).detach().cpu()
        h.append(values)
    return torch.cat(h) if h else torch.empty((0,model.fc2.in_features))


def fit_readout(h, labels, weight, bias, target, seen):
    """All-seen CE competition; original receiver readouts stay fixed.

    Integrating with anchor negatives adds KL(old distribution || distribution
    with the target row). Old logits are unchanged, so that KL reduces to the
    probability mass assigned to the new competing class. It cannot certify
    unseen old inputs; it is empirical regularization on current receiver BASE.
    """
    positive=labels==target; negative=~positive
    if not positive.any() or not negative.any():raise Rejected('TRAIN_TIME_BASE_POSITIVE_NEGATIVE_REQUIRED')
    scale=h.square().mean(0).sqrt().clamp_min(1e-3)
    hn=h/scale
    w=torch.nn.Parameter(weight[target].clone()*scale)
    b=torch.nn.Parameter(bias[target].clone())
    initial=w.detach().clone();initial_b=b.detach().clone()
    other=[c for c in seen if c!=target]
    reference=torch.logsumexp(F.linear(h,weight[other],bias[other]),1).detach()
    optimizer=torch.optim.Adam([w,b],lr=RULES['learning_rate'])
    steps=RULES['steps']
    history=[]
    for step in range(steps):
        optimizer.zero_grad()
        score=hn@w+b
        loss=.5*F.softplus(reference[positive]-score[positive]).mean()
        loss+=.5*F.softplus(score[negative]-reference[negative]).mean()
        loss+=RULES['parameter_l2']*((w-initial).square().mean()+(b-initial_b).square())
        if not torch.isfinite(loss):raise Rejected('TRAIN_TIME_NONFINITE_LOSS')
        loss.backward()
        if not torch.isfinite(w.grad).all() or not torch.isfinite(b.grad):
            raise Rejected('TRAIN_TIME_NONFINITE_GRADIENT')
        torch.nn.utils.clip_grad_norm_([w,b],1.)
        optimizer.step()
        if step in (0,steps-1):history.append(dict(step=step,loss=float(loss.detach())))
    return (w.detach()/scale).numpy(),float(b.detach()),history


def update_packet(weight,bias,metadata):
    # Only a parameter update and binding metadata, no training examples.
    buffer=io.BytesIO()
    torch.save(dict(weight=torch.as_tensor(weight),bias=float(bias),metadata=metadata),buffer)
    return buffer.getvalue()


def read_update(packet):
    value=torch.load(io.BytesIO(packet),map_location='cpu',weights_only=True)
    if (set(value)!={'weight','bias','metadata'} or value['weight'].ndim!=1 or
            not torch.isfinite(value['weight']).all() or not np.isfinite(value['bias'])):
        raise Rejected('TRAIN_TIME_UPDATE_SCHEMA')
    return value


def put_readout(model, target, weight, bias):
    if np.shape(weight)!=(model.fc2.in_features,):raise Rejected('TRAIN_TIME_READOUT_SHAPE')
    with torch.no_grad():
        model.fc2.weight[target].copy_(torch.as_tensor(weight,device=model.fc2.weight.device))
        model.fc2.bias[target].fill_(bias)
        model.weight_masks['fc2'][target].fill_(1.)
        model.bias_masks['fc2'][target]=1.
        model.unit_ranks['fc2'][target]=2


def donor_proposal(model, donor_pool, target, seen, device):
    """Donor-local endpoint; returns only trained parameters and aggregates."""
    weight,bias=effective_linear(model,'fc2')
    dh=receiver_features(model,donor_pool['X'],device,RULES['batch_size'])
    w,b,donor_trace=fit_readout(dh,torch.as_tensor(donor_pool['y']),weight,bias,target,seen)
    return dict(weight=w,bias=b,donor_trace=donor_trace,
        raw_examples_transmitted=0,old_readout_rows_changed=0,encoder_updated=False)


def receiver_integration(model, receiver_pool, target, seen, weight_update, bias_update, device):
    """Receiver-local proximal integration, using only own current BASE."""
    if not len(receiver_pool['y']) or np.any(receiver_pool['y']==target):
        raise Rejected('TRAIN_TIME_RECEIVER_CURRENT_NEGATIVES_REQUIRED')
    weight,bias=effective_linear(model,'fc2')
    rh=receiver_features(model,receiver_pool['X'],device,RULES['batch_size'])
    scale=rh.square().mean(0).sqrt().clamp_min(1e-3)
    wn=torch.nn.Parameter(torch.as_tensor(weight_update)*scale)
    bn=torch.nn.Parameter(torch.tensor(bias_update))
    startw=wn.detach().clone();startb=bn.detach().clone()
    other=[c for c in seen if c!=target]
    ref=torch.logsumexp(F.linear(rh,weight[other],bias[other]),1).detach()
    optimizer=torch.optim.Adam([wn,bn],lr=RULES['learning_rate'])
    trace=[]
    for step in range(RULES['integration_steps']):
        optimizer.zero_grad()
        kl=F.softplus((rh/scale)@wn+bn-ref).mean()
        # Proximal regularization preserves the donated row while penalizing
        # probability theft on legal receiver-current anchor samples.
        proximal=(wn-startw).square().mean()+(bn-startb).square()
        loss=RULES['old_probability_regularization']*kl+proximal
        if not torch.isfinite(loss):raise Rejected('TRAIN_TIME_NONFINITE_INTEGRATION')
        loss.backward()
        if not torch.isfinite(wn.grad).all() or not torch.isfinite(bn.grad):
            raise Rejected('TRAIN_TIME_NONFINITE_INTEGRATION_GRADIENT')
        torch.nn.utils.clip_grad_norm_([wn,bn],1.);optimizer.step()
        if step in (0,RULES['integration_steps']-1):trace.append(dict(step=step,loss=float(loss.detach())))
    return dict(weight=(wn.detach()/scale).numpy(),
        bias=float(bn.detach()),receiver_trace=trace,
        raw_examples_transmitted=0,old_readout_rows_changed=0,encoder_updated=False)


def shadow(model,router,target,task,weight,bias):
    candidate=copy.deepcopy(model);detector=copy.deepcopy(router)
    put_readout(candidate,target,weight,bias)
    # Register trained class availability in the existing task router. No
    # signature/threshold/imported route or router refit is created.
    detector.episode_classes[task]=sorted(set(detector.episode_classes.get(task,[]))|{target})
    return candidate,detector
