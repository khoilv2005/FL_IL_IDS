"""V4-only adapters; native V1 sampling, detector and inference code stay intact."""
import numpy as np
import torch
import pickle
from threadpoolctl import threadpool_limits
from fed_learning.servers.continuous_class_router import ContinuousClassMemory
from .local_task_loop import _update_local_nice_context_memory
from .candle_v4_bank import base_predictions

VERSION='candle_v4_continuous_class_router_v7'
BOOTSTRAP_VERSION='candle_v4_continuous_bootstrap_v8'
CONTINUOUS_VERSIONS=(VERSION,BOOTSTRAP_VERSION)


def validate_task_support(memory,detector):
    native={int(t) for t,rows in detector.activation_memory.items() if len(rows)}
    actual=set(memory.class_to_task.values()) if memory is not None else set()
    if native!=actual:
        raise ValueError('Continuous/native task memory mismatch; bootstrap state must travel with the model')


def clone_continuous_memory(service,receiver,source,task,source_model,target_model,source_detector,target_detector):
    """V8 only: exact-clone lifecycle hook, never a partial catch-up/aggregation hook."""
    if service is None or getattr(service,'spec',{}).get('version')!=BOOTSTRAP_VERSION:return None
    from .checkpoint_state import snapshot_context_detector
    from .candle_v4_shadow import digest
    if receiver==source or receiver in service.continuous_routers:
        raise ValueError('Continuous bootstrap requires a new receiver')
    memory=service.continuous_routers.get(source)
    if memory is None or memory.fitted_classes is None or memory.latest_task>int(task):
        raise ValueError('Missing/future source continuous memory at exact clone')
    validate_task_support(memory,source_detector)
    if (digest(source_model.state_dict())!=digest(target_model.state_dict()) or
            digest(source_model.weight_masks)!=digest(target_model.weight_masks) or
            digest(snapshot_context_detector(source_detector))!=digest(snapshot_context_detector(target_detector))):
        raise ValueError('Continuous memory can travel only with the exact model/masks/native detector')
    # Native bootstrap preserves these feature coordinates; no extra data/forward/fit.
    for detector in (source_detector,target_detector):
        mask=getattr(detector,'routing_feature_mask',None)
        if mask is None:mask=getattr(detector,'stable_feature_mask',None)
        if not np.array_equal(memory.indices,memory.selected(mask,memory.full_width)):
            raise ValueError('Bootstrap routing coordinates differ')
    payload=memory.state_dict();raw=pickle.dumps(payload,protocol=5)
    for arm in service.arms:
        if service.bytes[arm]+len(raw)>service.spec['per_arm_bytes']:
            raise ValueError('Continuous bootstrap exceeds application-wire budget')
    copied=ContinuousClassMemory.restore(pickle.loads(raw))
    validate_task_support(copied,target_detector)
    receipt=dict(policy='exact_model_clone_v1',task=int(task),source=int(source),receiver=int(receiver),
        source_memory_sha256=digest(payload),serialized_bytes=len(raw),
        tasks=sorted(set(copied.class_to_task.values())),classes=sorted(copied.memory))
    service.continuous_routers[receiver]=copied
    service.continuous_bootstrap_receipts.append(receipt)
    service.bytes['continuous_bootstrap']+=len(raw)
    for arm in service.arms:service.bytes[arm]+=len(raw)
    return receipt


class _CaptureModel:
    def __init__(self,model):self.model=model;self.activations=None
    def __getattr__(self,key):return getattr(self.model,key)
    def eval(self):self.model.eval();return self
    def get_context_activations_per_sample(self,x):
        self.activations=self.model.get_context_activations_per_sample(x)
        return self.activations


class _CaptureDetector:
    def __init__(self,native,memory,labels):self.native=native;self.memory=memory;self.labels=labels
    def __getattr__(self,key):return getattr(self.native,key)
    def push_activations(self,model,data,episode,reference_data=None):
        if len(data)!=len(self.labels):raise ValueError('Native sampler label ordering mismatch')
        capture=_CaptureModel(model)
        self.native.push_activations(capture,data,episode,reference_data=reference_data)
        mask=getattr(self.native,'routing_feature_mask',None)
        if mask is None:mask=getattr(self.native,'stable_feature_mask',None)
        self.memory.push({k:v.detach().cpu().numpy() for k,v in capture.activations.items()},self.labels,episode,mask)
    def train_models(self,task):
        self.native.train_models(task)
        with threadpool_limits(limits=2):self.memory.fit(task)


class ContinuousRoutingView:
    """Inference-only view. Never passed to native training or graph capsules."""
    def __init__(self,native,memory):self.native=native;self.memory=memory
    def __getattr__(self,key):return getattr(self.native,key)
    def binarize_layer_activations(self,activations):
        mask=getattr(self.native,'routing_feature_mask',None)
        if mask is None:mask=getattr(self.native,'stable_feature_mask',None)
        return self.memory.encode(activations,mask)
    def predict_episodes_batch(self,features):return self.memory.predict(features)[0]
    def predict_episodes_with_scores(self,features):return self.memory.predict(features)


def update_memory(service,client_id,detector,model,x,y,task,classes,device,fit_router=True):
    if service is None or getattr(service,'spec',{}).get('version') not in CONTINUOUS_VERSIONS:
        return _update_local_nice_context_memory(detector,model,x,y,task,classes,device,fit_router=fit_router)
    if y is None or not len(y) or not classes:
        return _update_local_nice_context_memory(detector,model,x,y,task,classes,device,fit_router=fit_router)
    yy=y.detach().cpu().numpy()
    if yy.dtype.kind not in 'iu' or np.any(yy<0) or np.any(yy>=34) or np.any(yy//6!=task):
        raise ValueError('Continuous memory only accepts current-task TRAIN labels')
    if len(set(map(int,classes)))!=len(classes) or any(not 0<=int(c)<34 or int(c)//6!=task for c in classes):
        raise ValueError('Invalid current-task class metadata')
    if detector.router_mode!='multiclass_balanced' or getattr(detector,'retain_reference_inputs',True) or detector.reference_input_memory:
        raise ValueError('V4 continuous adapter requires unchanged native multiclass/sketch policy')
    existing=service.continuous_routers.get(client_id)
    if service.spec['version']==BOOTSTRAP_VERSION:
        validate_task_support(existing,detector)
    if existing is not None:
        if task<existing.latest_task:raise ValueError('Historical task writes are forbidden')
        mask=getattr(detector,'routing_feature_mask',None)
        if mask is None:mask=getattr(detector,'stable_feature_mask',None)
        if existing.indices is not None and not np.array_equal(existing.selected(mask,existing.full_width),existing.indices):
            raise ValueError('Continuous routing feature coordinates changed')
    cap=int(detector.router_reference_per_class)
    # Native sampler concatenates one chunk per class in this exact order.
    # Recover class labels by counts, not another sample draw or input lookup.
    labels=np.concatenate([np.full(min(cap,int((yy==c).sum())),int(c),dtype=np.int64) for c in classes])
    memory=service.continuous_routers.setdefault(client_id,ContinuousClassMemory(cap))
    proxy=_CaptureDetector(detector,memory,labels)
    return _update_local_nice_context_memory(proxy,model,x,y,task,classes,device,fit_router=fit_router)


def predictions(service,client_id,model,detector,x,seen):
    memory=service.continuous_routers.get(client_id)
    if service.spec['version']==BOOTSTRAP_VERSION:validate_task_support(memory,detector)
    if memory is None:
        if detector.activation_memory:raise ValueError('Continuous memory missing; binary conversion forbidden')
        return base_predictions(model,detector,x,seen,return_episodes=True)
    return base_predictions(model,ContinuousRoutingView(detector,memory),x,seen,return_episodes=True)
