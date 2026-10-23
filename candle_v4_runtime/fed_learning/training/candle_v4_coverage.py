"""Development variant: add missing negative classes from authorized live peers."""
from collections import Counter
from copy import deepcopy

import torch

from .candle_v4_preflight import positive_neighbors
from .candle_v4_shadow import ShadowService,digest,make_expert

VERSION='candle_v4_coverage_negatives_dev_v1'


def negative_contributors(receiver,donor,target,counts,graph,minimum_count=32):
    """Metadata only: zero-covered negative classes; smallest eligible owner per class."""
    peers=positive_neighbors(graph,receiver)
    if donor not in peers:raise ValueError('Primary positive donor outside current graph')
    parents={receiver,donor}
    covered={c for o in parents for c,n in counts[o].items() if c!=target and n>0}
    available={c for o in peers if o in counts for c,n in counts[o].items() if c!=target and n>0}
    selected={}
    for c in sorted(available-covered):
        eligible=[o for o in peers if o not in parents and counts.get(o,{}).get(c,0)>=minimum_count]
        if eligible:selected[c]=min(eligible)
    return selected


class CoverageService(ShadowService):
    def __init__(self,spec,enabled,saved=None):
        super().__init__(spec,saved)
        self.coverage_enabled=enabled
        self.graph=None
        self.trace=[]

    def train_step(self,name,data,step,task,round_id):
        target=self.spec['target'];donor=self.spec['donor'];receiver=self.spec['receiver']
        sampling=getattr(self,'sampling_cache',None)
        def view(owner):
            y=data[owner][1]
            key=(owner,id(y),len(y))
            if sampling is None:return dict(counts=Counter(y.tolist()))
            if key not in sampling:
                labels,counts=torch.unique(y,sorted=True,return_counts=True)
                sampling[key]=dict(counts=Counter(dict(zip(labels.tolist(),counts.tolist()))),
                    indices={c:torch.nonzero(y==c).flatten() for c in labels.tolist()})
            return sampling[key]
        views={o:view(o) for o in data};all_counts={o:v['counts'] for o,v in views.items()}
        proposed=negative_contributors(receiver,donor,target,all_counts,self.graph)
        selected=proposed if self.coverage_enabled else {}
        parents={donor} if self.spec.get('donor_only_control') else {donor,receiver}
        if self.spec.get('donor_only_control'):selected={}
        counts={o:all_counts[o].copy() for o in sorted(parents)}
        for c,o in selected.items():counts.setdefault(o,Counter())[c]=all_counts[o][c]
        negative=[(o,c,n) for o in sorted(counts) for c,n in sorted(counts[o].items()) if c!=target]
        nneg=sum(n for _,_,n in negative); by_class=Counter()
        if not nneg:raise ValueError('No authorized current negative samples')
        for _,c,n in negative:by_class[c]+=n
        generator=torch.Generator().manual_seed(42+step*100003+202)
        tags=torch.randint(nneg,(128,),generator=generator)
        requests={o:Counter() for o in counts};requests[donor][target]=128;offset=0
        for o,c,n in negative:
            requests[o][c]=int(((tags>=offset)&(tags<offset+n)).sum());offset+=n
        model,optimizer=self.branches[name];reference=deepcopy(model.state_dict());sha=digest(reference)
        identity=dict(task=task,round=round_id,step=step,target=target,receiver=receiver,donor=donor,
            version=VERSION,input_sha256=self.spec['input_sha256'],reference_sha256=sha,
            layout='39-64-1-fp32',mask='all-expert-parameters-young-v1',
            source_plan_sha256=digest(dict(selected=selected,counts=counts)))
        global_gradient=[torch.zeros_like(p) for p in model.parameters()]
        positive=0;negative_samples=0;loss_sum=0.;extra_samples=0;extra_norm=0.;receiver_norm=0.
        for owner in sorted(counts):
            request={c:n for c,n in requests[owner].items() if n}
            if not request:continue
            peer=make_expert(42)
            if owner!=receiver:
                packet=self.send('coverage_reference',dict(identity=identity,contributor=owner,state=reference,requests=request,counts=counts))
                if packet['identity']!=identity or digest(packet['state'])!=sha:raise ValueError('Stale coverage reference')
                peer.load_state_dict(packet['state'])
            else:peer.load_state_dict(reference)
            x,y=data[owner];rows=[];weights=[]
            for c,n in sorted(request.items()):
                g=torch.Generator().manual_seed(42+step*100003+(0 if c==target else owner*1009)+c*17)
                local=views[owner].get('indices',{}).get(c)
                if local is None:local=torch.nonzero(y==c).flatten()
                if not len(local):raise ValueError('Requested class missing in local current BASE')
                rows.append(local[torch.randint(len(local),(n,),generator=g)])
                w=.5/128
                if c!=target:w*=nneg/(len(by_class)*by_class[c])
                weights.extend([w]*n)
            idx=torch.cat(rows);truth=(y[idx]==target).float()
            if owner!=donor and truth.any():raise ValueError('Positive contribution from undeclared donor')
            positive+=int(truth.sum());negative_samples+=int((truth==0).sum())
            loss=(torch.nn.functional.binary_cross_entropy_with_logits(peer(x[idx]).flatten(),truth,reduction='none')*
                  torch.tensor(weights)).sum()
            grads=[g.detach() for g in torch.autograd.grad(loss,list(peer.parameters()))]
            if not all(torch.isfinite(g).all() for g in grads):raise FloatingPointError('Nonfinite coverage gradient')
            if owner!=receiver:
                packet=self.send('coverage_gradient',dict(identity=identity,contributor=owner,gradients=grads))
                if packet['identity']!=identity:raise ValueError('Stale coverage gradient')
                grads=packet['gradients']
            norm=float(sum(g.square().sum() for g in grads).sqrt())
            if owner not in (receiver,donor):extra_samples+=len(idx);extra_norm+=norm
            if owner==receiver:receiver_norm=norm
            for total,g in zip(global_gradient,grads):total.add_(g)
            loss_sum+=float(loss.detach())
        if (positive,negative_samples)!=(128,128):raise ValueError('Matched sample budget violated')
        optimizer.zero_grad(set_to_none=True)
        for p,g in zip(model.parameters(),global_gradient):p.grad=g
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('Nonfinite coverage model')
        result=dict(loss=loss_sum,receiver_negative_gradient_norm=receiver_norm,
            extra_negative_samples=extra_samples,extra_negative_gradient_norm=extra_norm,
            positives=positive,negatives=negative_samples,selected=selected,proposed=proposed)
        self.trace.append(dict(round=round_id,step=step,**result))
        return result
