"""Label-blind routed prediction and separate development-only measurements."""
import copy
import numpy as np
import torch
from .state import complete_hash


class Predictor:
    def __init__(self,seen,device,batch_size):
        self.seen=seen;self.device=device;self.batch_size=batch_size

    @torch.no_grad()
    def records(self,model,router,inputs):
        from fed_learning.training.denice_eval import _denice_routed_logits_with_episodes
        modes=[(m,m.training) for m in model.modules()];active=copy.deepcopy(model.active_adapters)
        predictions=[];episodes=[]
        try:
            model.eval()
            for start in range(0,len(inputs),self.batch_size):
                x=torch.as_tensor(inputs[start:start+self.batch_size],dtype=torch.float32,device=self.device)
                logits,ep=_denice_routed_logits_with_episodes(model,x,router,self.seen,self.device,inference_policy='pred_hard')
                if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite routed logits')
                predictions.append(logits.argmax(1).cpu().numpy());episodes.append(ep)
        finally:
            model.active_adapters=active
            for module,training in modes:module.training=training
        return dict(pred=np.concatenate(predictions) if predictions else np.empty(0,dtype=np.int64),
                    task=np.concatenate(episodes) if episodes else np.empty(0,dtype=np.int64))

    def __call__(self,model,router,inputs):return self.records(model,router,inputs)['pred']


def accuracy(pred,y):return float(np.mean(pred==y)) if len(y) else None


def compare(old,new,y,c,observed,episodes,task):
    y=np.asarray(y);positive=y==c;old_rows=np.isin(y,list(observed));old_rows &= ~positive
    def part(mask):
        return dict(rows=int(mask.sum()),before=accuracy(old[mask],y[mask]),after=accuracy(new[mask],y[mask]))
    return dict(rows=len(y),overall=part(np.ones(len(y),dtype=bool)),missing_class_recall=part(positive),
        old_supported_accuracy=part(old_rows),rescue=int(((old!=y)&(new==y)).sum()),
        break_count=int(((old==y)&(new!=y)).sum()),predicted_import_task_rows=int((episodes==task).sum()),
        positive_predicted_import_task_rows=int(((episodes==task)&positive).sum()))


@torch.no_grad()
def logit_fidelity(candidate,donor,inputs,task,c,protocol,batch_size):
    """Fixed transfer context, never true sample task; diagnostic only."""
    states=[(m,copy.deepcopy(m.active_adapters),[(a,a.training) for a in m.modules()]) for m in (candidate,donor)]
    maxima=0.;passed=True
    try:
        for model,_,_ in states:model.eval();model.set_active_context(task)
        device=next(candidate.parameters()).device
        for start in range(0,len(inputs),batch_size):
            x=torch.as_tensor(inputs[start:start+batch_size],dtype=torch.float32,device=device)
            a=candidate(x)[:,c];b=donor(x)[:,c]
            passed=passed and bool(torch.isfinite(a).all() and torch.isfinite(b).all())
            passed=passed and torch.allclose(a,b,rtol=protocol.fidelity_rtol,atol=protocol.fidelity_atol)
            maxima=max(maxima,float((a-b).abs().max()))
    finally:
        for model,active,modes in states:
            model.active_adapters=active
            for module,training in modes:module.training=training
    return dict(passed=bool(passed and len(inputs)>0),rows=len(inputs),max_absolute_error=maxima,
                policy='fixed imported context diagnostic; no ground-truth task used for primary prediction')
