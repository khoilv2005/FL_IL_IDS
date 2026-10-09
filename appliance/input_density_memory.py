"""Chronological Gaussian input moments for receiver-local routing diagnostics.

Stores counts, means and diagonal variances, not samples. Density preference is
an empirical routing model, not a hard old-population FAR guarantee. The fixed
floor/shrinkage must be declared before any holdout measurement.
"""
import numpy as np
from .config import Rejected

class InputDensityMemory:
    VERSION='appliance_fixed_input_diagonal_density_v1'
    FLOOR_FRACTION=1e-3
    VARIANCE_SHRINKAGE=.1
    ABSOLUTE_FLOOR=1e-8

    def __init__(self,input_shape,preprocessing_sha256,transform="identity"):
        if transform not in ("identity","signed_log1p"):raise ValueError("Unknown fixed input transform")
        self.transform=transform
        self.input_shape=tuple(input_shape);self.width=int(np.prod(input_shape))
        self.preprocessing_sha256=preprocessing_sha256
        self.task=-1;self.entries={};self.events=set()

    def features(self,x):
        x=np.asarray(x,np.float64)
        if tuple(x.shape[1:])!=self.input_shape or not np.isfinite(x).all():raise Rejected('INPUT_DENSITY_PREPROCESSING_CHANGED')
        if self.transform=="signed_log1p":x=np.sign(x)*np.log1p(np.abs(x))
        return x.reshape(len(x),self.width)

    def observe(self,task,classes,x,y,event_id):
        y=np.asarray(y,np.int64);classes=set(map(int,classes));task=int(task)
        if task<self.task:raise Rejected('HISTORICAL_CALIBRATION_REOPENED')
        if str(event_id) in self.events:raise Rejected('CALIBRATION_EVENT_REPLAYED')
        if not 0<=task<6 or any(c<0 or c>=34 for c in classes):raise Rejected('CALIBRATION_CLASS_RANGE')
        if len(x)!=len(y) or not set(map(int,np.unique(y))).issubset(classes):raise Rejected('CALIBRATION_TASK_SCOPE_CHANGED')
        z=self.features(x)
        pending={}
        for c in np.unique(y):
            if int(c) in self.entries:raise Rejected('CLASS_SUMMARY_ALREADY_SEALED')
            v=z[y==c];mean=v.mean(0);variance=v.var(0,ddof=1) if len(v)>1 else np.zeros(self.width)
            pending[int(c)]=dict(task=task,count=len(v),mean=mean,variance=variance)
        self.entries.update(pending);self.task=task;self.events.add(str(event_id))

    def state(self):
        result=dict(version=self.VERSION,input_shape=list(self.input_shape),preprocessing_sha256=self.preprocessing_sha256,
            floor_fraction=self.FLOOR_FRACTION,variance_shrinkage=self.VARIANCE_SHRINKAGE,
            absolute_floor=self.ABSOLUTE_FLOOR,task=self.task,events=sorted(self.events),
            entries={str(c):dict(v,mean=v['mean'].tolist(),variance=v['variance'].tolist()) for c,v in self.entries.items()},
            retained_raw_examples=0,retained_per_sample_features=0,
            limitation='Gaussian diagonal approximation; no hard support coverage or population FAR guarantee')
        if self.transform!='identity':result.update(version='appliance_fixed_log_input_diagonal_density_v2',transform=self.transform)
        return result

    @classmethod
    def restore(cls,state):
        if (state['version'] not in (cls.VERSION,'appliance_fixed_log_input_diagonal_density_v2') or state['floor_fraction']!=cls.FLOOR_FRACTION
            or state['variance_shrinkage']!=cls.VARIANCE_SHRINKAGE or state['absolute_floor']!=cls.ABSOLUTE_FLOOR):
            raise Rejected('INPUT_DENSITY_POLICY_CHANGED')
        transform=state.get('transform','identity')
        if (state['version']=='appliance_fixed_log_input_diagonal_density_v2')!=(transform=='signed_log1p'):
            raise Rejected('INPUT_DENSITY_TRANSFORM_VERSION')
        obj=cls(state['input_shape'],state['preprocessing_sha256'],transform);obj.task=int(state['task']);obj.events=set(state['events'])
        for c,v in state['entries'].items():
            mean=np.asarray(v['mean'],np.float64);var=np.asarray(v['variance'],np.float64)
            if (not 0<=int(c)<34 or not 0<=int(v['task'])<=obj.task or
                    mean.shape!=(obj.width,) or var.shape!=(obj.width,) or not np.isfinite(mean).all()
                    or not np.isfinite(var).all() or (var<0).any() or int(v['count'])<1):
                raise Rejected('INVALID_INPUT_DENSITY_SUMMARY')
            obj.entries[int(c)]=dict(v,mean=mean,variance=var)
        return obj

    def target_preference(self,x,target):
        if self.input_shape!=target.input_shape or self.preprocessing_sha256!=target.preprocessing_sha256 or self.transform!=target.transform or len(target.entries)!=1:
            raise Rejected('INPUT_DENSITY_TARGET_SCOPE_CHANGED')
        entries=list(self.entries.values())+list(target.entries.values())
        z=self.features(x);n=sum(v['count'] for v in entries)
        mean=sum(v['count']*v['mean'] for v in entries)/n
        pooled=sum(max(0,v['count']-1)*v['variance']+v['count']*(v['mean']-mean)**2 for v in entries)/max(1,n-1)
        floor=np.maximum(pooled*self.FLOOR_FRACTION,self.ABSOLUTE_FLOOR)
        def score(v):
            variance=np.maximum((1-self.VARIANCE_SHRINKAGE)*v['variance']+self.VARIANCE_SHRINKAGE*pooled,floor)
            return -.5*np.mean((z-v['mean'])**2/variance+np.log(variance),axis=1)
        target_score=score(next(iter(target.entries.values())))
        old_score=np.max(np.stack([score(v) for v in self.entries.values()]),axis=0) if self.entries else np.full(len(z),-np.inf)
        return target_score>old_score,target_score-old_score
