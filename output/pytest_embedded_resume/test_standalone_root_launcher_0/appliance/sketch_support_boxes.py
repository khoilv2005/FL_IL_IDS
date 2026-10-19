"""Oriented fixed-sketch support boxes and finite signature-hit upper counts.

PCA and boxes summarize current owned observations; no rows survive. Later
prototype queries use a convex box/unit-sphere dual bound, without replay or
backbone dependence. The bound covers enclosed observations, not a population.
"""
import copy
import numpy as np
from .config import Rejected
from .portable_route import SharedSketch
from .state import digest

def sphere_box_support(weights,lower,upper,cap):
    """Conservative dual bound for max w.t, l<=t<=u, ||t||<=cap."""
    w,l,u=map(lambda x:np.asarray(x,np.float64),(weights,lower,upper))
    if (w.ndim!=1 or l.shape!=w.shape or u.shape!=w.shape or
            not np.isfinite(w).all() or not np.isfinite(l).all() or not np.isfinite(u).all() or
            np.any(l>u) or not np.isfinite(cap) or cap<=0):
        raise Rejected('INVALID_SPHERE_BOX_QUERY')
    corner=np.where(w>=0,u,l)
    box=float(w@corner)
    slack=1e-9*(1+float(np.abs(w)@np.maximum(np.abs(l),np.abs(u))))
    if np.linalg.norm(corner)<=cap:
        return box+slack
    if np.linalg.norm(np.clip(np.zeros_like(w),l,u))>cap:
        return box+slack  # Numerical ambiguity: use the looser bound.
    lo,hi=0.,1.
    for _ in range(80):
        trial=np.clip(w/(2*hi),l,u)
        if np.linalg.norm(trial)<=cap:
            break
        hi*=2
    else:
        return box+slack
    for _ in range(64):
        mid=(lo+hi)/2
        trial=np.clip(w/(2*mid),l,u)
        if np.linalg.norm(trial)>cap:
            lo=mid
        else:
            hi=mid
    trial=np.clip(w/(2*hi),l,u)
    terms=w*trial-hi*trial*trial
    dual=float(hi*cap*cap+terms.sum())
    error=256*np.finfo(np.float64).eps*(1+abs(hi*cap*cap)+float(np.abs(terms).sum()))
    return min(box,dual)+slack+error

class SketchSupportBoxes:
    VERSION='appliance_current_fixed_sketch_PCA_boxes_v1'
    BUDGET=8
    MIN_GROUP=8
    NORM_CAP=1.0001
    def __init__(self,sketch,owner):
        if not isinstance(sketch,SharedSketch) or type(owner) is not int or owner<0:
            raise Rejected('INVALID_SKETCH_BOX_OWNER')
        self.sketch=sketch;self.owner=owner;self.task=-1
        self.entries={};self.events=[]

    def observe(self,task,classes,x,y,event):
        y=np.asarray(y)
        if (type(task) is not int or not 0<=task<=5 or task<=self.task or
                type(event) is not str or not event or event in self.events):
            raise Rejected('SKETCH_BOX_PAST_OR_REPLAY')
        if (y.dtype!=np.int64 or y.shape!=(len(x),) or
                not set(y.tolist()).issubset(classes) or
                any(type(c) is not int or not 0<=c<34 for c in classes) or
                any(str(c) in self.entries for c in classes)):
            raise Rejected('SKETCH_BOX_TASK_CLASS_CHANGED')
        z,valid=self.sketch.features(x)
        if not valid.all() or not np.isfinite(z).all() or np.linalg.norm(z.astype(np.float64),axis=1).max(initial=0)>self.NORM_CAP:
            raise Rejected('SKETCH_BOX_UNIT_CONTRACT_CHANGED')
        pending=copy.deepcopy(self.entries)
        for c in sorted(set(y.tolist())):
            values=z[y==c].astype(np.float64)
            centered=values-values.mean(0)
            scatter=centered.T@centered
            eigen,basis=np.linalg.eigh(scatter)
            basis=basis[:,np.argsort(eigen)[::-1]]
            for col in range(basis.shape[1]):
                anchor=np.argmax(np.abs(basis[:,col]))
                if basis[anchor,col]<0:basis[:,col]*=-1
            coordinates=values@basis
            scales=np.maximum(coordinates.std(0),1e-12)
            leaves=[np.arange(len(values))]
            while len(leaves)<self.BUDGET:
                options=[]
                for k,ids in enumerate(leaves):
                    if len(ids)<2*self.MIN_GROUP:continue
                    spread=np.ptp(coordinates[ids],axis=0)/scales
                    axis=int(np.argmax(spread))
                    if spread[axis]>0:options.append((float(spread[axis]),len(ids),-k,axis,k))
                if not options:break
                _,_,_,axis,k=max(options)
                ids=leaves.pop(k)
                order=ids[np.argsort(coordinates[ids,axis],kind='stable')]
                middle=len(order)//2
                leaves.extend([order[:middle],order[middle:]])
            boxes=[]
            for ids in leaves:
                lower=coordinates[ids].min(0)-1e-8
                upper=coordinates[ids].max(0)+1e-8
                boxes.append(dict(count=len(ids),lower=lower.tolist(),upper=upper.tolist()))
            pending[str(c)]=dict(task=task,count=len(values),basis=basis.tolist(),boxes=boxes)
        self.entries,self.task,self.events=pending,task,self.events+[event]

    def projection_bounds(self,prototype,tau,required):
        w=np.asarray(prototype)
        if (w.dtype!=np.float32 or w.shape!=(self.sketch.dimension,) or not np.isfinite(w).all() or
                not np.isclose(np.linalg.norm(w.astype(np.float64)),1,rtol=1e-5,atol=1e-6) or
                not np.isfinite(tau) or not -1<=tau<=1):
            raise Rejected('INVALID_SKETCH_BOX_PROJECTION')
        d=len(w);wf=w.astype(np.float64);eps=np.finfo(np.float32).eps
        gamma=(2*d+2)*eps/(1-(2*d+2)*eps)
        fp_error=gamma*self.NORM_CAP*float(np.abs(wf).sum())
        missing=[];result={}
        for c in sorted(set(required)):
            if type(c) is not int or not 0<=c<34:
                raise Rejected('INVALID_SKETCH_BOX_REQUIRED_CLASS')
            if str(c) not in self.entries:
                missing.append(c);continue
            entry=self.entries[str(c)];basis=np.asarray(entry['basis'],np.float64)
            transformed=basis.T@wf
            gram_error=float(np.linalg.norm(basis.T@basis-np.eye(d)))
            reconstruction=float(np.linalg.norm(basis@basis.T-np.eye(d)))*self.NORM_CAP*np.linalg.norm(wf)+1e-8
            cap=self.NORM_CAP*(1+gram_error)+1e-8
            upper_count=0;bounds=[]
            for box in entry['boxes']:
                score=sphere_box_support(transformed,box['lower'],box['upper'],cap)+reconstruction+fp_error
                ambiguous=bool(tau<1 and score>tau)
                upper_count+=box['count'] if ambiguous else 0
                bounds.append(dict(rows=box['count'],score_upper=float(min(1.,score)),ambiguous=ambiguous))
            result[str(c)]=dict(rows=entry['count'],activation_count_upper=upper_count,
                far_upper=upper_count/entry['count'],boxes=bounds)
        return dict(per_class=result,missing_classes=missing,fp32_dot_error_allowance=fp_error,
            scope='finite enclosed observations; no population probability claim',
            unseen_population_far_certified=False,main_install_authorized=False)

    def veto(self,x,required):
        """Label-blind veto on enclosed support; missing classes reject.

        This development path uses the same FP32 sketch function as summary
        construction. Cross-batch numeric enclosure is audited separately.
        It is not a population-FAR certificate or main-install authority.
        """
        if any(type(c) is not int or not 0<=c<34 for c in required):
            raise Rejected('INVALID_SKETCH_BOX_REQUIRED_CLASS')
        missing=[c for c in required if str(c) not in self.entries]
        if missing:raise Rejected('MISSING_OLD_CLASS_ROUTING_SUMMARY',str(missing))
        z,valid=self.sketch.features(x)
        rejected=~valid
        for c in sorted(set(required)):
            entry=self.entries[str(c)]
            coordinates=z.astype(np.float64)@np.asarray(entry['basis'],np.float64)
            for box in entry['boxes']:
                lower=np.asarray(box['lower'],np.float64)
                upper=np.asarray(box['upper'],np.float64)
                rejected|=((coordinates>=lower)&(coordinates<=upper)).all(1)
        return rejected

    def state(self):
        body=dict(version=self.VERSION,signature=self.sketch.manifest(),owner=self.owner,task=self.task,
            box_budget=self.BUDGET,min_group_rows=self.MIN_GROUP,norm_cap=self.NORM_CAP,
            entries=copy.deepcopy(self.entries),events=list(self.events),
            retained_raw_examples=0,retained_per_example_features=0,main_install_authorized=False)
        return dict(body,state_digest=digest(body))

    @classmethod
    def restore(cls,state):
        fields={'version','signature','owner','task','box_budget','min_group_rows','norm_cap','entries','events',
                'retained_raw_examples','retained_per_example_features','main_install_authorized','state_digest'}
        if set(state)!=fields:
            raise Rejected('SKETCH_BOX_STATE_SCHEMA_CHANGED')
        body={k:copy.deepcopy(v) for k,v in state.items() if k!='state_digest'}
        if (body['version']!=cls.VERSION or body['box_budget']!=cls.BUDGET or body['min_group_rows']!=cls.MIN_GROUP or
                body['norm_cap']!=cls.NORM_CAP or state['state_digest']!=digest(body) or
                body['retained_raw_examples']!=0 or body['retained_per_example_features']!=0 or
                body['main_install_authorized'] is not False or type(body['task']) is not int or not -1<=body['task']<=5):
            raise Rejected('SKETCH_BOX_STATE_CHANGED')
        sig=body['signature']
        obj=cls(SharedSketch(tuple(sig['input_shape']),sig['dimension'],sig['preprocessing_sha256'],sig['seed']),body['owner'])
        if obj.sketch.manifest()!=sig:
            raise Rejected('SKETCH_BOX_FUNCTION_CHANGED')
        obj.task=body['task'];obj.entries=body['entries'];obj.events=body['events']
        if len(set(obj.events))!=len(obj.events) or any(type(e) is not str or not e for e in obj.events):
            raise Rejected('SKETCH_BOX_HISTORY_CHANGED')
        d=obj.sketch.dimension
        for c,e in obj.entries.items():
            if (str(int(c))!=c or not 0<=int(c)<34 or set(e)!={'task','count','basis','boxes'} or
                    type(e['task']) is not int or not 0<=e['task']<=obj.task or
                    type(e['count']) is not int or e['count']<1 or not 1<=len(e['boxes'])<=cls.BUDGET):
                raise Rejected('SKETCH_BOX_ENTRY_CHANGED')
            basis=np.asarray(e['basis'],np.float64)
            if basis.shape!=(d,d) or not np.isfinite(basis).all() or np.linalg.norm(basis.T@basis-np.eye(d))>1e-8:
                raise Rejected('SKETCH_BOX_NONORTHOGONAL_BASIS')
            total=0
            for box in e['boxes']:
                if set(box)!={'count','lower','upper'} or type(box['count']) is not int or box['count']<1:
                    raise Rejected('SKETCH_BOX_INTERVAL_CHANGED')
                lo,hi=np.asarray(box['lower'],np.float64),np.asarray(box['upper'],np.float64)
                if (lo.shape!=(d,) or hi.shape!=(d,) or not np.isfinite(lo).all() or not np.isfinite(hi).all() or
                        np.any(lo>=hi) or e['count']>=cls.MIN_GROUP and box['count']<cls.MIN_GROUP):
                    raise Rejected('SKETCH_BOX_INTERVAL_CHANGED')
                total+=box['count']
            if total!=e['count']:
                raise Rejected('SKETCH_BOX_COUNT_CHANGED')
        return obj
