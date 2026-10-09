"""Owned current-BASE sketch shield: finite support veto, not CAL evidence.

Current examples are discarded after grouped geometry/provenance is recorded.
Floating-point enclosures cover the fixed FP32 sketch under different batches.
No population, privacy or positive-recall guarantee is inferred from support.
"""
import copy
import hashlib
import numpy as np
from .config import Rejected
from .current_base_data import CurrentBaseData
from .portable_route import SharedSketch
from .sketch_support_boxes import SketchSupportBoxes
from .state import digest

def sketch_with_error(sketch,x):
    """Conservative distance from FP32 output to normalized exact projection.

    Bounds roundoff in dense FP32 projection and normalization. Requires normal
    FP32 arithmetic (no TF32/FP16), finite projection, and the locked matrix.
    Near-zero/cancellation-dominated rows receive a full unit-ball allowance.
    """
    x=np.asarray(x,np.float32)
    z,valid=sketch.features(x)
    flat=x.reshape(len(x),int(np.prod(sketch.input_shape)))
    matrix=sketch.matrix()
    projected=flat@matrix
    absolute=flat.astype(np.float64).__abs__()@np.abs(matrix.astype(np.float64))
    eps=np.finfo(np.float32).eps;n=2*flat.shape[1]+4
    gamma=n*eps/(1-n*eps)
    projection_error=np.linalg.norm(gamma*absolute,axis=1)+1e-12
    norm=np.linalg.norm(projected.astype(np.float64),axis=1)
    unit_norm=np.linalg.norm(z.astype(np.float64),axis=1)
    error=np.full(len(x),2.001,np.float64)
    good=valid & np.isfinite(norm) & (norm>projection_error)
    error[good]=(2*projection_error[good]/(norm[good]-projection_error[good])+
        np.abs(unit_norm[good]-1)+2*eps*unit_norm[good]/(1-eps)+1e-8)
    return z,valid,error

class CurrentBaseSketchShield:
    VERSION='appliance_owned_current_BASE_sketch_shield_v1'
    NUMERICAL_POLICY='FP32 dense projection gamma(2*input_width+4); unit-distance enclosure v1'
    def __init__(self,view,sketch):
        if not isinstance(view,CurrentBaseData) or not isinstance(sketch,SharedSketch):
            raise Rejected('SCOPED_BASE_SKETCH_PROVIDER_REQUIRED')
        if (tuple(view.store['input_shape'])!=sketch.input_shape or
                view.store['metadata_sha256']!=sketch.preprocessing_sha256):
            raise Rejected('BASE_SHIELD_PREPROCESSING_CHANGED')
        self.owner=view.client_id;self.role_sha=view.store['role_manifest_sha256']
        self.pp_sha=view.store['metadata_sha256'];self.memory=SketchSupportBoxes(sketch,self.owner)
        self.provenance=[]

    def observe_current(self,view):
        if (not isinstance(view,CurrentBaseData) or view.client_id!=self.owner or
                view.store['role_manifest_sha256']!=self.role_sha or
                view.store['metadata_sha256']!=self.pp_sha or
                (self.provenance and view.task<=self.provenance[-1]['task'])):
            raise Rejected('BASE_SHIELD_SCOPE_OR_REPLAY_CHANGED')
        classes=view.store['task_classes'][str(view.task)]
        pool=view.current_pool(self.owner,'base',classes)
        expected={str(c):int(view.manifest['clients'][str(self.owner)]['role_class_counts']['base'].get(str(c),0)) for c in classes}
        actual={str(c):int((pool['y']==c).sum()) for c in classes}
        if actual!=expected:raise Rejected('BASE_SHIELD_ROLE_COUNTS_CHANGED')
        event=dict(owner=self.owner,task=view.task,role='current own BASE support; not CAL',
            role_manifest_sha256=self.role_sha,preprocessing_sha256=self.pp_sha,
            partition_sha256=pool['partition_sha256'],
            row_ids_sha256=hashlib.sha256(np.asarray(pool['rows'],dtype='<i8').tobytes()).hexdigest(),
            class_counts=expected,numerical_policy=self.NUMERICAL_POLICY)
        event['event_id']=digest(event)
        pending=SketchSupportBoxes.restore(self.memory.state())
        pending.observe(view.task,classes,pool['X'],pool['y'],event['event_id'])
        _,valid,error=sketch_with_error(pending.sketch,pool['X'])
        if not valid.all():raise Rejected('BASE_SHIELD_INVALID_UNIT_SUPPORT')
        for c in np.unique(pool['y']):
            entry=pending.entries[str(int(c))]
            basis=np.asarray(entry['basis'],np.float64)
            expansion=float(error[pool['y']==c].max())*np.linalg.norm(basis,axis=0)+1e-8
            for box in entry['boxes']:
                box['lower']=(np.asarray(box['lower'])-expansion).tolist()
                box['upper']=(np.asarray(box['upper'])+expansion).tolist()
        candidate=copy.copy(self);candidate.memory=pending
        if len(pool['y']) and not candidate.veto(pool['X'],sorted(map(int,np.unique(pool['y'])))).all():
            raise Rejected('BASE_SHIELD_ENCLOSURE_FAILED')
        self.memory,self.provenance=pending,self.provenance+[event]
        return copy.deepcopy(event)

    def veto(self,x,required):
        if any(type(c) is not int or not 0<=c<34 for c in required):
            raise Rejected('BASE_SHIELD_REQUIRED_CLASS_CHANGED')
        missing=[c for c in required if str(c) not in self.memory.entries]
        if missing:raise Rejected('BASE_SHIELD_MISSING_OLD_SUPPORT',str(missing))
        z,valid,error=sketch_with_error(self.memory.sketch,x)
        veto=~valid
        for c in sorted(set(required)):
            e=self.memory.entries[str(c)];basis=np.asarray(e['basis'],np.float64)
            coordinates=z.astype(np.float64)@basis
            expansion=error[:,None]*np.linalg.norm(basis,axis=0)[None,:]+1e-8
            for box in e['boxes']:
                lo=np.asarray(box['lower'],np.float64);hi=np.asarray(box['upper'],np.float64)
                veto|=((coordinates>=lo-expansion)&(coordinates<=hi+expansion)).all(1)
        return veto

    def coverage(self,required):
        if any(type(c) is not int or not 0<=c<34 for c in required):
            raise Rejected('BASE_SHIELD_REQUIRED_CLASS_CHANGED')
        counts={str(c):self.memory.entries.get(str(c),{}).get('count',0) for c in sorted(set(required))}
        return dict(source_role='finite own BASE support only; not CAL acceptance',counts=counts,
            missing_classes=[int(c) for c,n in counts.items() if not n],
            classes_below_32_BASE_rows=[int(c) for c,n in counts.items() if n<32],
            unseen_population_FAR_certified=False,main_install_authorized=False)

    def state(self):
        body=dict(version=self.VERSION,owner=self.owner,role_manifest_sha256=self.role_sha,
            preprocessing_sha256=self.pp_sha,memory=self.memory.state(),provenance=copy.deepcopy(self.provenance),
            numerical_policy=self.NUMERICAL_POLICY,retained_raw_examples=0,retained_per_example_features=0,
            CAL_acceptance_substitution=False,unseen_population_FAR_certified=False,main_install_authorized=False)
        return dict(body,state_digest=digest(body))

    @classmethod
    def restore(cls,state):
        fields={'version','owner','role_manifest_sha256','preprocessing_sha256','memory','provenance',
            'numerical_policy','retained_raw_examples','retained_per_example_features','CAL_acceptance_substitution',
            'unseen_population_FAR_certified','main_install_authorized','state_digest'}
        if set(state)!=fields:raise Rejected('BASE_SHIELD_STATE_SCHEMA_CHANGED')
        body={k:copy.deepcopy(v) for k,v in state.items() if k!='state_digest'}
        if (body['version']!=cls.VERSION or body['numerical_policy']!=cls.NUMERICAL_POLICY or
                state['state_digest']!=digest(body) or type(body['owner']) is not int or body['owner']<0 or
                any(type(body[k]) is not str or len(body[k])!=64 for k in ('role_manifest_sha256','preprocessing_sha256')) or
                any(body[k] is not False for k in ('CAL_acceptance_substitution','unseen_population_FAR_certified','main_install_authorized')) or
                body['retained_raw_examples']!=0 or body['retained_per_example_features']!=0):
            raise Rejected('BASE_SHIELD_STATE_CHANGED')
        obj=cls.__new__(cls);obj.owner=body['owner'];obj.role_sha=body['role_manifest_sha256']
        obj.pp_sha=body['preprocessing_sha256'];obj.memory=SketchSupportBoxes.restore(body['memory'])
        obj.provenance=body['provenance']
        if obj.memory.owner!=obj.owner or obj.memory.sketch.preprocessing_sha256!=obj.pp_sha:
            raise Rejected('BASE_SHIELD_MEMORY_OWNER_OR_SPACE_CHANGED')
        tasks=[];declared={};events=[]
        for p in obj.provenance:
            if (set(p)!={'owner','task','role','role_manifest_sha256','preprocessing_sha256','partition_sha256',
                    'row_ids_sha256','class_counts','numerical_policy','event_id'} or
                    p['owner']!=obj.owner or type(p['task']) is not int or not 0<=p['task']<=5 or
                    p['role']!='current own BASE support; not CAL' or p['role_manifest_sha256']!=obj.role_sha or
                    p['preprocessing_sha256']!=obj.pp_sha or p['numerical_policy']!=cls.NUMERICAL_POLICY or
                    any(type(p[k]) is not str or len(p[k])!=64 for k in ('partition_sha256','row_ids_sha256')) or
                    p['event_id']!=digest({k:v for k,v in p.items() if k!='event_id'})):
                raise Rejected('BASE_SHIELD_PROVENANCE_CHANGED')
            tasks.append(p['task']);events.append(p['event_id'])
            for c,n in p['class_counts'].items():
                if (not c.isdigit() or str(int(c))!=c or not 0<=int(c)<34 or type(n) is not int or n<0 or c in declared):
                    raise Rejected('BASE_SHIELD_PROVENANCE_COUNTS_CHANGED')
                if n:declared[c]=(p['task'],n)
        if (tasks!=sorted(set(tasks)) or events!=obj.memory.events or
                (tasks[-1] if tasks else -1)!=obj.memory.task or
                declared!={c:(e['task'],e['count']) for c,e in obj.memory.entries.items()}):
            raise Rejected('BASE_SHIELD_HISTORY_CHANGED')
        return obj
