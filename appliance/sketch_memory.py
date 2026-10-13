"""Streaming, input-free bounds for portable routing on past local classes.

Bounds enclose observed FIT sketches, not the population distribution. They
certify non-activation on those observations without retaining raw examples or
re-evaluating a changing backbone. Missing coverage is reported, never invented.
"""
import hashlib

import numpy as np

from .config import Rejected
from .portable_route import SharedSketch


class SketchMemory:
    VERSION='appliance_streaming_sketch_bounds_v1'
    ROUNDING_SLACK=1e-5

    def __init__(self,sketch,max_classes=34):
        self.sketch=sketch;self.max_classes=int(max_classes);self.task=-1
        self.entries={};self.consumed=set()

    def observe(self,task,classes,x,y,event_id):
        task=int(task);classes=set(map(int,classes));y=np.asarray(y,dtype=np.int64)
        if task<self.task:raise Rejected('HISTORICAL_CALIBRATION_REOPENED')
        if event_id in self.consumed:raise Rejected('CALIBRATION_EVENT_REPLAYED')
        if len(x)!=len(y) or not set(map(int,np.unique(y))).issubset(classes):raise Rejected('CALIBRATION_TASK_SCOPE_CHANGED')
        if any(c<0 or c>=self.max_classes for c in classes):raise Rejected('CALIBRATION_CLASS_RANGE')
        z,valid=self.sketch.features(x)
        for c in np.unique(y):
            values=z[(y==c)&valid]
            if not len(values):continue
            lo,hi=values.min(0),values.max(0)
            old=self.entries.get(int(c))
            self.entries[int(c)]=dict(count=len(values)+(old['count'] if old else 0),
                minimum=np.minimum(lo,old['minimum']) if old else lo.copy(),
                maximum=np.maximum(hi,old['maximum']) if old else hi.copy(),
                task=task)
        self.consumed.add(str(event_id));self.task=task
        return dict(task=task,rows=len(y),usable_rows=int(valid.sum()),classes=sorted(map(int,np.unique(y))))

    def upper_bounds(self,prototype,classes):
        p=np.asarray(prototype,dtype=np.float64)
        if p.shape!=(self.sketch.dimension,) or not np.isfinite(p).all():raise Rejected('ROUTING_PROTOTYPE_SHAPE')
        result={};missing=[]
        for c in sorted(set(map(int,classes))):
            entry=self.entries.get(c)
            if entry is None:missing.append(c);continue
            endpoints=np.where(p>=0,entry['maximum'],entry['minimum']).astype(np.float64)
            result[c]=min(1.,float(endpoints@p)+self.ROUNDING_SLACK)
        return dict(bounds=result,missing_classes=missing,
            minimum_safe_tau=max(result.values(),default=-1.),
            fully_covered=not missing,
            guarantee='no activation on observed FIT sketches inside stored boxes; no population FAR guarantee')

    def state(self):
        return dict(version=self.VERSION,signature=self.sketch.manifest(),max_classes=self.max_classes,
            task=self.task,consumed_events=sorted(self.consumed),
            entries={str(c):dict(v,minimum=v['minimum'].tolist(),maximum=v['maximum'].tolist()) for c,v in self.entries.items()},
            retained_raw_examples=0,retained_per_sample_features=0)

    @classmethod
    def restore(cls,state):
        if state['version']!=cls.VERSION:raise Rejected('SKETCH_MEMORY_VERSION')
        s=state['signature'];result=cls(SharedSketch(tuple(s['input_shape']),s['dimension'],s['preprocessing_sha256'],s['seed']),state['max_classes'])
        if result.sketch.manifest()!=s:raise Rejected('SKETCH_MEMORY_ENCODER_CHANGED')
        result.task=int(state['task']);result.consumed=set(state['consumed_events'])
        result.entries={int(c):dict(v,minimum=np.asarray(v['minimum'],np.float32),maximum=np.asarray(v['maximum'],np.float32)) for c,v in state['entries'].items()}
        return result
