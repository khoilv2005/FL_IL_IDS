"""Current BASE class moments: training support, never calibration evidence.

The fixed preprocessed input coordinate system survives model/router drift.
Only class-level count/mean/scatter and provenance hashes persist. Quality and
future patch safety remain unknown until an appropriate acceptance check.
"""
import copy
import hashlib
import numpy as np
from .config import Rejected
from .current_base_data import CurrentBaseData
from .state import digest

class CurrentBaseSupportMoments:
    VERSION='appliance_current_owned_BASE_support_moments_v1'
    def __init__(self,view):
        if not isinstance(view,CurrentBaseData):
            raise Rejected('SCOPED_BASE_PROVIDER_REQUIRED')
        self.owner=view.client_id
        self.role_sha256=view.store['role_manifest_sha256']
        self.preprocessing_sha256=view.store['metadata_sha256']
        self.input_shape=tuple(view.store['input_shape'])
        self.entries={}
        self.provenance=[]

    def observe_current(self,view):
        if (not isinstance(view,CurrentBaseData) or view.client_id!=self.owner or
                view.store['role_manifest_sha256']!=self.role_sha256 or
                view.store['metadata_sha256']!=self.preprocessing_sha256 or
                tuple(view.store['input_shape'])!=self.input_shape or
                (self.provenance and view.task<=self.provenance[-1]['task'])):
            raise Rejected('BASE_SUPPORT_SCOPE_OR_TASK_CHANGED')
        classes=view.store['task_classes'][str(view.task)]
        if any(str(c) in self.entries for c in classes):
            raise Rejected('BASE_SUPPORT_CLASS_REPLAY')
        pool=view.current_pool(self.owner,'base',classes)
        counts=view.manifest['clients'][str(self.owner)]['role_class_counts']['base']
        expected={str(c):int(counts.get(str(c),0)) for c in classes}
        if expected!={str(c):int((pool['y']==c).sum()) for c in classes}:
            raise Rejected('BASE_SUPPORT_ROLE_COUNTS_CHANGED')
        values=pool['X'].reshape(len(pool['y']),int(np.prod(self.input_shape))).astype(np.float64)
        pending=copy.deepcopy(self.entries)
        for c in sorted(set(pool['y'].tolist())):
            v=values[pool['y']==c]
            mean=v.mean(0)
            centered=v-mean
            scatter=centered.T@centered
            if not np.isfinite(mean).all() or not np.isfinite(scatter).all():
                raise Rejected('NONFINITE_BASE_SUPPORT_MOMENTS')
            pending[str(c)]=dict(task=view.task,count=len(v),mean=mean.tolist(),scatter=scatter.tolist())
        event=dict(owner=self.owner,task=view.task,role='current own BASE',
            role_manifest_sha256=self.role_sha256,preprocessing_sha256=self.preprocessing_sha256,
            partition_sha256=pool['partition_sha256'],
            row_ids_sha256=hashlib.sha256(np.asarray(pool['rows'],dtype='<i8').tobytes()).hexdigest(),
            class_counts=expected)
        event['event_id']=digest(event)
        self.entries,self.provenance=pending,self.provenance+[event]
        return copy.deepcopy(event)

    def state(self):
        body=dict(version=self.VERSION,owner=self.owner,role_manifest_sha256=self.role_sha256,
            preprocessing_sha256=self.preprocessing_sha256,input_shape=list(self.input_shape),
            entries=copy.deepcopy(self.entries),provenance=copy.deepcopy(self.provenance),
            retained_raw_examples=0,retained_per_example_features=0,
            can_substitute_CAL_acceptance=False,old_quality_verified=False,main_install_authorized=False)
        return dict(body,state_digest=digest(body))

    @classmethod
    def restore(cls,state):
        fields={'version','owner','role_manifest_sha256','preprocessing_sha256','input_shape','entries',
            'provenance','retained_raw_examples','retained_per_example_features','can_substitute_CAL_acceptance',
            'old_quality_verified','main_install_authorized','state_digest'}
        if set(state)!=fields:
            raise Rejected('BASE_SUPPORT_STATE_SCHEMA_CHANGED')
        body={k:copy.deepcopy(v) for k,v in state.items() if k!='state_digest'}
        if (body['version']!=cls.VERSION or state['state_digest']!=digest(body) or
                type(body['owner']) is not int or body['owner']<0 or
                any(type(s) is not str or len(s)!=64 for s in (body['role_manifest_sha256'],body['preprocessing_sha256'])) or
                not body['input_shape'] or any(type(i) is not int or i<=0 for i in body['input_shape']) or
                body['retained_raw_examples']!=0 or body['retained_per_example_features']!=0 or
                any(body[k] is not False for k in ('can_substitute_CAL_acceptance','old_quality_verified','main_install_authorized'))):
            raise Rejected('BASE_SUPPORT_STATE_CHANGED')
        obj=cls.__new__(cls)
        obj.owner=body['owner'];obj.role_sha256=body['role_manifest_sha256']
        obj.preprocessing_sha256=body['preprocessing_sha256'];obj.input_shape=tuple(body['input_shape'])
        obj.entries=body['entries'];obj.provenance=body['provenance']
        tasks=[];declared={}
        for p in obj.provenance:
            if set(p)!={'owner','task','role','role_manifest_sha256','preprocessing_sha256',
                        'partition_sha256','row_ids_sha256','class_counts','event_id'}:
                raise Rejected('BASE_SUPPORT_EVENT_SCHEMA_CHANGED')
            if (p['owner']!=obj.owner or p['role']!='current own BASE' or
                    type(p['task']) is not int or not 0<=p['task']<=5 or
                    p['role_manifest_sha256']!=obj.role_sha256 or
                    p['preprocessing_sha256']!=obj.preprocessing_sha256 or
                    any(type(p[k]) is not str or len(p[k])!=64 for k in ('partition_sha256','row_ids_sha256')) or
                    p['event_id']!=digest({k:v for k,v in p.items() if k!='event_id'})):
                raise Rejected('BASE_SUPPORT_EVENT_CHANGED')
            tasks.append(p['task'])
            for c,n in p['class_counts'].items():
                if str(int(c))!=c or not 0<=int(c)<34 or type(n) is not int or n<0:
                    raise Rejected('BASE_SUPPORT_CLASS_COUNTS_CHANGED')
                if n:
                    if c in declared:
                        raise Rejected('BASE_SUPPORT_CLASS_REPLAY')
                    declared[c]=(p['task'],n)
        if tasks!=sorted(set(tasks)) or declared!={c:(v['task'],v['count']) for c,v in obj.entries.items()}:
            raise Rejected('BASE_SUPPORT_HISTORY_CHANGED')
        width=int(np.prod(obj.input_shape))
        for c,e in obj.entries.items():
            if set(e)!={'task','count','mean','scatter'}:
                raise Rejected('BASE_SUPPORT_MOMENT_SCHEMA_CHANGED')
            mean=np.asarray(e['mean'],np.float64);scatter=np.asarray(e['scatter'],np.float64)
            scale=max(1.,float(np.abs(scatter).max(initial=0.)))
            if (type(e['count']) is not int or e['count']<1 or
                    mean.shape!=(width,) or scatter.shape!=(width,width) or
                    not np.isfinite(mean).all() or not np.isfinite(scatter).all() or
                    not np.allclose(scatter,scatter.T,atol=1e-10*scale,rtol=0) or
                    np.linalg.eigvalsh((scatter+scatter.T)/2).min() < -1e-10*scale or
                    (e['count']==1 and np.any(scatter))):
                raise Rejected('INVALID_BASE_SUPPORT_MOMENTS')
        return obj

    def coverage(self,required):
        if any(type(c) is not int or not 0<=c<34 for c in required):
            raise Rejected('BASE_SUPPORT_COVERAGE_SCOPE_CHANGED')
        counts={str(c):self.entries[str(c)]['count'] if str(c) in self.entries else 0 for c in sorted(set(required))}
        return dict(source_role='BASE training support; not CAL evidence',counts=counts,
            missing_classes=[int(c) for c,n in counts.items() if not n],
            classes_below_32_rows=[int(c) for c,n in counts.items() if n<32],
            old_quality_verified=False,main_install_authorized=False)
