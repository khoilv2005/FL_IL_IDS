"""Aggregated fixed-input support boxes; no backbone-dependent old features.

Boxes summarize at least eight current FIT observations per leaf. Rare classes
keep count-only missing-support records, never a singleton historical vector.
A veto covers the finite FIT observations enclosed at construction, not an
unseen population. This component cannot by itself authorize an installation.
"""
import copy
import numpy as np
from .config import Rejected
from .state import digest

class InputBoxMemory:
    VERSION = 'appliance_current_fit_input_boxes_v1'
    MAX_BOXES = 32
    MIN_LEAF_ROWS = 8

    def __init__(self,input_shape,preprocessing_sha256,owner):
        if (type(owner) is not int or owner<0 or
                not input_shape or any(type(d) is not int or d<=0 for d in input_shape) or
                type(preprocessing_sha256) is not str or len(preprocessing_sha256)!=64):
            raise Rejected('INVALID_INPUT_BOX_OWNER_OR_SPACE')
        self.input_shape=tuple(input_shape)
        self.width=int(np.prod(input_shape))
        self.preprocessing_sha256=preprocessing_sha256
        self.owner=owner
        self.task=-1
        self.entries={}
        self.events=[]

    def _values(self,x):
        values=np.asarray(x,np.float32)
        if tuple(values.shape[1:])!=self.input_shape or not np.isfinite(values).all():
            raise Rejected('INPUT_BOX_PREPROCESSING_CHANGED')
        return values.reshape(len(values),self.width)

    def observe(self,task,classes,x,y,event_id):
        labels=np.asarray(y)
        if (type(task) is not int or not 0<=task<=5 or task<=self.task or
                type(event_id) is not str or not event_id or event_id in self.events):
            raise Rejected('INPUT_BOX_TASK_OR_EVENT_REPLAY')
        if (not classes or any(type(c) is not int or not 0<=c<34 for c in classes) or
                len(classes)!=len(set(classes)) or labels.dtype!=np.int64 or
                labels.shape!=(len(x),) or not set(labels.tolist()).issubset(classes) or
                any(str(c) in self.entries for c in classes)):
            raise Rejected('INPUT_BOX_CLASS_SCOPE_CHANGED')
        values=self._values(x)
        pending=copy.deepcopy(self.entries)
        for c in sorted(set(labels.tolist())):
            data=values[labels==c]
            entry=dict(task=task,count=len(data),boxes=[],
                       insufficient_support=len(data)<self.MIN_LEAF_ROWS)
            if len(data)>=self.MIN_LEAF_ROWS:
                # Fixed class-wide scales used only in construction, not retained.
                scales=np.maximum(np.std(data.astype(np.float64),axis=0),1e-12)
                leaves=[np.arange(len(data))]
                while len(leaves)<self.MAX_BOXES:
                    options=[]
                    for k,ids in enumerate(leaves):
                        if len(ids)<2*self.MIN_LEAF_ROWS:
                            continue
                        spread=np.ptp(data[ids].astype(np.float64),axis=0)/scales
                        axis=int(np.argmax(spread))
                        if spread[axis]>0:
                            options.append((float(spread[axis]),len(ids),-k,axis,k))
                    if not options:
                        break
                    _,_,_,axis,k=max(options)
                    ids=leaves.pop(k)
                    order=ids[np.argsort(data[ids,axis],kind='stable')]
                    middle=len(order)//2
                    leaves.extend([order[:middle],order[middle:]])
                for ids in leaves:
                    lower=np.nextafter(data[ids].min(0),np.float32(-np.inf))
                    upper=np.nextafter(data[ids].max(0),np.float32(np.inf))
                    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
                        raise Rejected('INPUT_BOX_UNBOUNDED_FLOAT_RANGE')
                    entry['boxes'].append(dict(count=len(ids),lower=lower.tolist(),upper=upper.tolist()))
            pending[str(c)]=entry
        self.entries,self.task,self.events=pending,task,self.events+[event_id]
        return dict(task=task,classes=sorted(set(labels.tolist())),rows=len(labels))

    def veto(self,x,classes=None,strict=True):
        values=self._values(x)
        requested=sorted(self.entries) if classes is None else [str(c) for c in sorted(set(classes))]
        unknown=[int(c) for c in requested if c not in self.entries or not self.entries[c]['boxes']]
        if strict and unknown:
            raise Rejected('INSUFFICIENT_OLD_INPUT_BOX_COVERAGE',str(unknown))
        rejected=np.zeros(len(values),bool)
        for c in requested:
            for box in self.entries.get(c,{}).get('boxes',[]):
                lo=np.asarray(box['lower'],np.float32)
                hi=np.asarray(box['upper'],np.float32)
                rejected|=((values>=lo)&(values<=hi)).all(1)
        return rejected,dict(unknown_classes=unknown,
            scope='finite enclosed current-FIT support only; no unseen-population guarantee',
            main_install_authorized=False)

    def state(self):
        body=dict(version=self.VERSION,input_shape=list(self.input_shape),
            preprocessing_sha256=self.preprocessing_sha256,owner=self.owner,task=self.task,
            max_boxes_per_class=self.MAX_BOXES,min_leaf_rows=self.MIN_LEAF_ROWS,
            entries=copy.deepcopy(self.entries),events=list(self.events),
            retained_raw_examples=0,retained_per_sample_features=0,
            old_population_safety_certified=False,main_install_authorized=False)
        return dict(body,state_digest=digest(body))

    @classmethod
    def restore(cls,state):
        fields={'version','input_shape','preprocessing_sha256','owner','task','max_boxes_per_class',
            'min_leaf_rows','entries','events','retained_raw_examples','retained_per_sample_features',
            'old_population_safety_certified','main_install_authorized','state_digest'}
        if set(state)!=fields:
            raise Rejected('INPUT_BOX_STATE_SCHEMA_CHANGED')
        body={k:copy.deepcopy(v) for k,v in state.items() if k!='state_digest'}
        if (body['version']!=cls.VERSION or state['state_digest']!=digest(body) or
                body['max_boxes_per_class']!=cls.MAX_BOXES or body['min_leaf_rows']!=cls.MIN_LEAF_ROWS or
                body['retained_raw_examples']!=0 or body['retained_per_sample_features']!=0 or
                body['old_population_safety_certified'] is not False or body['main_install_authorized'] is not False):
            raise Rejected('INPUT_BOX_STATE_CHANGED')
        obj=cls(body['input_shape'],body['preprocessing_sha256'],body['owner'])
        obj.task=body['task']
        if type(obj.task) is not int or not -1<=obj.task<=5:
            raise Rejected('INVALID_INPUT_BOX_TASK')
        obj.events=body['events']
        if (not isinstance(obj.events,list) or any(type(e) is not str or not e for e in obj.events) or
                len(obj.events)!=len(set(obj.events))):
            raise Rejected('INPUT_BOX_EVENT_CHANGED')
        obj.entries=body['entries']
        for c,entry in obj.entries.items():
            if (str(int(c))!=c or not 0<=int(c)<34 or
                    set(entry)!={'task','count','boxes','insufficient_support'} or
                    type(entry['task']) is not int or not 0<=entry['task']<=obj.task or
                    type(entry['count']) is not int or entry['count']<1 or
                    type(entry['insufficient_support']) is not bool or
                    entry['insufficient_support']!=(entry['count']<cls.MIN_LEAF_ROWS) or
                    not isinstance(entry['boxes'],list) or len(entry['boxes'])>cls.MAX_BOXES):
                raise Rejected('INVALID_INPUT_BOX_ENTRY')
            if entry['insufficient_support'] and entry['boxes']:
                raise Rejected('INPUT_BOX_RARE_CLASS_EXEMPLAR_FORBIDDEN')
            if not entry['insufficient_support'] and not entry['boxes']:
                raise Rejected('INPUT_BOX_SUPPORT_MISSING')
            for box in entry['boxes']:
                if set(box)!={'count','lower','upper'}:
                    raise Rejected('INPUT_BOX_PAYLOAD_SCHEMA_CHANGED')
                lo,hi=np.asarray(box['lower'],np.float32),np.asarray(box['upper'],np.float32)
                if (type(box['count']) is not int or box['count']<cls.MIN_LEAF_ROWS or
                        lo.shape!=(obj.width,) or hi.shape!=(obj.width,) or
                        not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo>=hi)):
                    raise Rejected('INVALID_INPUT_BOX_BOUNDS')
            if entry['boxes'] and sum(b['count'] for b in entry['boxes'])!=entry['count']:
                raise Rejected('INPUT_BOX_COUNT_CHANGED')
        return obj


class CurrentInputBoxProfiles:
    """Current own CAL-FIT staging only; aggregated input support, not quality."""
    VERSION='appliance_owner_current_fit_input_box_profiles_v1'
    def __init__(self,scoped):
        from .current_calibration_data import CurrentCalibrationData
        if not isinstance(scoped,CurrentCalibrationData):
            raise Rejected('CURRENT_INPUT_BOX_SCOPE_REQUIRED')
        self.memory=InputBoxMemory(scoped.store['input_shape'],scoped.store['metadata_sha256'],scoped.client_id)
        self.role_sha256=scoped.store['role_manifest_sha256']
        self.provenance=[]

    def observe_current(self,scoped):
        import hashlib
        from .current_calibration_data import CurrentCalibrationData
        from .imported_route import ROUTE_RULES,stratified_roles
        if (not isinstance(scoped,CurrentCalibrationData) or scoped.client_id!=self.memory.owner or
                scoped.store['role_manifest_sha256']!=self.role_sha256 or
                scoped.store['metadata_sha256']!=self.memory.preprocessing_sha256 or
                tuple(scoped.store['input_shape'])!=self.memory.input_shape or scoped.task<=self.memory.task):
            raise Rejected('CURRENT_INPUT_BOX_OWNER_TASK_CHANGED')
        classes=scoped.store['task_classes'][str(scoped.task)]
        pool=scoped.current_pool(scoped.client_id,'calibration',classes)
        # Preserve the existing campaign role assignment; no HOLDOUT re-split.
        seed=ROUTE_RULES['seed']+scoped.client_id+100003*(scoped.task+1)
        fit=stratified_roles(pool,seed)['fit']
        event=dict(owner=scoped.client_id,task=scoped.task,role='current own CAL FIT',
            role_manifest_sha256=self.role_sha256,partition_sha256=pool['partition_sha256'],
            fit_row_ids_sha256=hashlib.sha256(np.asarray(pool['rows'][fit],dtype='<i8').tobytes()).hexdigest(),
            input_shape=list(self.memory.input_shape),preprocessing_sha256=self.memory.preprocessing_sha256,
            split_seed=seed,class_counts={str(c):int((pool['y'][fit]==c).sum()) for c in classes},
            trained_model_support_claimed=False)
        event['event_id']=digest(event)
        pending=InputBoxMemory.restore(self.memory.state())
        pending.observe(scoped.task,classes,pool['X'][fit],pool['y'][fit],event['event_id'])
        self.memory=pending
        self.provenance.append(event)
        return copy.deepcopy(event)

    def state(self):
        body=dict(version=self.VERSION,memory=self.memory.state(),role_manifest_sha256=self.role_sha256,
                  provenance=copy.deepcopy(self.provenance),main_install_authorized=False)
        return dict(body,state_digest=digest(body))

    @classmethod
    def restore(cls,state):
        from .imported_route import ROUTE_RULES
        if set(state)!={'version','memory','role_manifest_sha256','provenance','main_install_authorized','state_digest'}:
            raise Rejected('CURRENT_INPUT_BOX_STATE_SCHEMA_CHANGED')
        body={k:copy.deepcopy(v) for k,v in state.items() if k!='state_digest'}
        if body['version']!=cls.VERSION or state['state_digest']!=digest(body) or body['main_install_authorized'] is not False:
            raise Rejected('CURRENT_INPUT_BOX_STATE_CHANGED')
        obj=cls.__new__(cls)
        obj.memory=InputBoxMemory.restore(body['memory'])
        obj.role_sha256=body['role_manifest_sha256']
        obj.provenance=body['provenance']
        declared={}
        tasks=[]
        for p in obj.provenance:
            if set(p)!={'owner','task','role','role_manifest_sha256','partition_sha256','fit_row_ids_sha256',
                        'input_shape','preprocessing_sha256','split_seed','class_counts',
                        'trained_model_support_claimed','event_id'}:
                raise Rejected('CURRENT_INPUT_BOX_PROVENANCE_SCHEMA_CHANGED')
            if (p['owner']!=obj.memory.owner or type(p['task']) is not int or not 0<=p['task']<=5 or
                    p['role']!='current own CAL FIT' or p['role_manifest_sha256']!=obj.role_sha256 or
                    p['input_shape']!=list(obj.memory.input_shape) or
                    p['preprocessing_sha256']!=obj.memory.preprocessing_sha256 or
                    p['split_seed']!=ROUTE_RULES['seed']+obj.memory.owner+100003*(p['task']+1) or
                    p['trained_model_support_claimed'] is not False or
                    p['event_id']!=digest({k:v for k,v in p.items() if k!='event_id'})):
                raise Rejected('CURRENT_INPUT_BOX_PROVENANCE_CHANGED')
            tasks.append(p['task'])
            for c,n in p['class_counts'].items():
                if str(int(c))!=c or not 0<=int(c)<34 or type(n) is not int or n<0:
                    raise Rejected('CURRENT_INPUT_BOX_CLASS_COUNT_CHANGED')
                if n:
                    if c in declared:
                        raise Rejected('CURRENT_INPUT_BOX_CLASS_REPLAY')
                    declared[c]=(p['task'],n)
        if (tasks!=sorted(set(tasks)) or (tasks[-1] if tasks else -1)!=obj.memory.task or
                [p['event_id'] for p in obj.provenance]!=obj.memory.events or
                declared!={c:(v['task'],v['count']) for c,v in obj.memory.entries.items()}):
            raise Rejected('CURRENT_INPUT_BOX_HISTORY_CHANGED')
        return obj
