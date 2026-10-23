"""CNN channel/kernel dependency closure; no GRU feature selection or rewiring.

Selected output rows (including inactive edges) are immutable. Active incoming
mask edges determine upstream channel closure. BN is channel-wise in this model.
"""
import numpy as np
import torch
from .contracts import ClusterState,array,binary,state_hash
from .state import snapshot

POLICY='cnn_coordinate_router_v1'
LAYERS=('conv1','conv2','conv3','gru')


def fresh_coordinate_state(model,*,feature_layer='conv2',**identity):
    from .native import capture_native
    if feature_layer not in ('conv2','conv3'):
        raise ValueError('Coordinate pilot supports Conv2/Conv3 only')
    router=dict(router_mode='binary_cosine',context_layers=LAYERS,stable_feature_mask=np.zeros(548,bool),
        binarize_thresholds=None,activation_memory={},episode_classes={})
    initial=capture_native(model,router_state=router,routing_dependencies=('bn1.num_batches_tracked',),**identity)
    payload=snapshot(initial)
    payload['versions']['feature']='p2_cnn_coordinate_binary_v1'
    payload['algorithm_state'].update(routing_contract_version=POLICY,coordinate_router_layer=feature_layer,
        routing_coordinate_masks={'bn1.num_batches_tracked':np.array(True)},routing_weight_mask_coordinates={},
        routing_bias_mask_coordinates={},classification_objective='cumulative_registered_ce_v1',
        seen_classes=(),learned_classes_by_task={})
    payload['protected_parameters']['bn1.num_batches_tracked']=torch.tensor(True)
    return recreate(payload)


def recreate(payload):
    return ClusterState.create(**{k:v for k,v in payload.items() if k not in ('model_spec','routing_space_hash')})


def cnn_closure(state,feature_mask):
    """Raw-state helper: callable by validation without recursive validate()."""
    target=state.algorithm_state['coordinate_router_layer']
    if target not in ('conv2','conv3') or tuple(state.router_state['context_layers'])!=LAYERS:
        raise ValueError('Invalid coordinate feature layer')
    segments={};offset=0
    for layer in LAYERS:
        n=len(state.algorithm_state['unit_ranks'][layer])
        segments[layer]=np.asarray(feature_mask[offset:offset+n],bool);offset+=n
    if len(feature_mask)!=offset or not segments[target].any() or any(segments[k].any() for k in LAYERS if k!=target):
        raise ValueError('Coordinate mask must select only configured CNN layer')
    if np.any(segments[target] & (array(state.algorithm_state['unit_ranks'][target])==0)):
        raise ValueError('Cannot select unallocated CNN features')
    outputs=segments[target]
    params,weights,biases,channels={},{},{},{}
    for index in range(int(target[-1]),0,-1):
        if not outputs.any():
            break # Disconnected root is independent of upstream activations.
        layer,bn=f'conv{index}',f'bn{index}'
        shape=state.model_state[layer+'.weight'].shape
        if shape[0]!=len(outputs):
            raise ValueError('CNN channel schema mismatch')
        rowmask=np.broadcast_to(outputs[:,None,None],shape).copy()
        params[layer+'.weight']=rowmask
        params[layer+'.bias']=outputs.copy()
        weights[layer]=rowmask.copy();biases[layer]=outputs.copy()
        channels[layer]=outputs.copy()
        for suffix in ('weight','bias','running_mean','running_var'):
            params[bn+'.'+suffix]=outputs.copy()
        params[bn+'.num_batches_tracked']=np.array(True)
        # Conv groups=1 under plain legacy architecture. All kernel positions
        # and mask bits on selected output rows are bound, even if masked zero.
        incoming=array(state.algorithm_state['weight_masks'][layer]).astype(bool)
        outputs=incoming[outputs].any(axis=(0,2))
    return dict(parameters=params,weight_masks=weights,bias_masks=biases,channels=channels,
        selected={k:int(v.sum()) for k,v in segments.items()})


def coordinate_fingerprint(state):
    alg=state.algorithm_state
    params=alg['routing_coordinate_masks']
    return state_hash(dict(policy=POLICY,feature_layer=alg['coordinate_router_layer'],
        architecture=state.versions['architecture'],feature_version=state.versions['feature'],
        native_input_shape=alg['native_input_shape'],parameter_coordinates=params,
        values={k:array(state.model_state[k])[array(m).astype(bool)] for k,m in params.items()},
        weight_coordinates=alg['routing_weight_mask_coordinates'],
        weight_values={k:array(alg['weight_masks'][k])[array(m).astype(bool)] for k,m in alg['routing_weight_mask_coordinates'].items()},
        bias_coordinates=alg['routing_bias_mask_coordinates'],
        bias_values={k:array(alg['bias_masks'][k])[array(m).astype(bool)] for k,m in alg['routing_bias_mask_coordinates'].items()},
        feature_mask=state.router_state['stable_feature_mask'],thresholds=state.router_state['binarize_thresholds']))


def validate_coordinate_contract(state):
    alg=state.algorithm_state
    if (state.versions['architecture']!='denice_legacy_plain_v1' or state.versions['feature']!='p2_cnn_coordinate_binary_v1'
        or state.router_state['router_mode']!='binary_cosine' or alg.get('coordinate_router_layer') not in ('conv2','conv3')
        or alg.get('classification_objective')!='cumulative_registered_ce_v1'):
        raise ValueError('Unsupported coordinate contract')
    params=alg.get('routing_coordinate_masks',{})
    if set(params)!=set(state.routing_dependencies):
        raise ValueError('Coordinate dependency keys mismatch')
    for name,mask in params.items():
        if (tuple(array(mask).shape)!=tuple(state.model_state[name].shape) or not binary(mask) or not array(mask).any()
            or name not in state.protected_parameters
            or np.any(array(mask).astype(bool)&~array(state.protected_parameters[name]).astype(bool))):
            raise ValueError('Missing/invalid coordinate parameter protection')
    for key,source in (('routing_weight_mask_coordinates','weight_masks'),('routing_bias_mask_coordinates','bias_masks')):
        for layer,mask in alg.get(key,{}).items():
            if layer not in alg[source] or tuple(mask.shape)!=tuple(alg[source][layer].shape) or not binary(mask):
                raise ValueError('Invalid coordinate connection-mask binding')
    seen,history=alg.get('seen_classes'),alg.get('learned_classes_by_task')
    if (not isinstance(seen,tuple) or tuple(sorted(set(seen)))!=seen or not set(seen)<=set(state.versions['class_ids'])
        or not isinstance(history,dict) or any(type(t) is not int or not 0<=t<=state.task or not isinstance(cs,tuple)
            or tuple(sorted(set(cs)))!=cs or not set(cs)<=set(state.versions['class_ids']) for t,cs in history.items())
        or tuple(sorted({c for cs in history.values() for c in cs}))!=seen
        or any(array(alg['unit_ranks']['fc2'])[c]==0 for c in seen)):
        raise ValueError('Invalid coordinate learned classes')
    if state.router_state['activation_memory']:
        required=cnn_closure(state,state.router_state['stable_feature_mask'])
        for name,mask in required['parameters'].items():
            if name not in params or np.any(mask&~array(params[name]).astype(bool)):
                raise ValueError('Incomplete upstream coordinate closure')
        for key,required_key in (('routing_weight_mask_coordinates','weight_masks'),('routing_bias_mask_coordinates','bias_masks')):
            for layer,mask in required[required_key].items():
                if layer not in alg[key] or np.any(mask&~array(alg[key][layer]).astype(bool)):
                    raise ValueError('Incomplete connection-mask coordinate binding')
        registered=tuple(sorted({c for cs in state.router_state['episode_classes'].values() for c in cs}))
        if registered!=seen:
            raise ValueError('Coordinate class registry and router registration disagree')


def protect_coordinate_closure(state,feature_mask):
    if state.router_state['activation_memory'] and not np.array_equal(feature_mask,state.router_state['stable_feature_mask']):
        raise ValueError('Cannot change committed CNN feature mask')
    required=cnn_closure(state,feature_mask)
    payload=snapshot(state);alg=payload['algorithm_state']
    params=alg['routing_coordinate_masks']
    for name,mask in required['parameters'].items():
        params[name]=array(params.get(name,np.zeros_like(mask))).astype(bool)|mask
    # Preserve every already declared/protected dependency, never narrow it.
    for name in state.routing_dependencies:
        params.setdefault(name,np.ones(state.model_state[name].shape,bool))
    payload['routing_dependencies']=tuple(sorted(params))
    for name,mask in params.items():
        old=payload['protected_parameters'].get(name,torch.zeros_like(state.model_state[name],dtype=torch.bool))
        payload['protected_parameters'][name]=torch.as_tensor(array(old).astype(bool)|array(mask).astype(bool))
    for key,required_key in (('routing_weight_mask_coordinates','weight_masks'),('routing_bias_mask_coordinates','bias_masks')):
        for layer,mask in required[required_key].items():
            alg[key][layer]=array(alg[key].get(layer,np.zeros_like(mask))).astype(bool)|mask
    payload['router_state']['stable_feature_mask']=np.asarray(feature_mask,bool)
    candidate=recreate(payload)
    return candidate,required
