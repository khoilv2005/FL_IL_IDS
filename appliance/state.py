"""Canonical fingerprints and complete isolated snapshots, including Python state."""
import copy
import hashlib
import json
import random

import joblib
import numpy as np
import torch


def digest(value):
    # Torch pickle storage identifiers are not content fingerprints. Normalize
    # tensors nested in CGoFed, BN and router Python state before hashing.
    # Internal fingerprinting only: no untrusted pickle is deserialized.
    return joblib.hash(canonical_value(value),hash_name='sha1')


def canonical_value(value):
    if isinstance(value,torch.Tensor):
        return {'__appliance_tensor__':tensor_hash(value)}
    if isinstance(value,dict):return {key:canonical_value(item) for key,item in value.items()}
    if isinstance(value,list):return [canonical_value(item) for item in value]
    if isinstance(value,tuple):return tuple(canonical_value(item) for item in value)
    # sklearn's binary LR can expose coef_ with a broadcast singleton stride
    # (0, itemsize). Pickle/Torch restore makes it contiguous without changing
    # any coefficient. Hash the same logical estimator in either layout, while
    # retaining its type, parameters, learned fields, array shapes and dtypes.
    from sklearn.linear_model import LogisticRegression
    if type(value) is LogisticRegression:
        replacements={name:np.array(item,copy=True,order='C')
                      for name,item in vars(value).items()
                      if isinstance(item,np.ndarray) and
                      any(size==1 and stride==0 for size,stride in zip(item.shape,item.strides))}
        if replacements:
            value=copy.copy(value)
            value.__dict__=dict(vars(value),**replacements)
    return value


def tensor_hash(value):
    value=value.detach().cpu().contiguous()
    data=value.numpy()
    return hashlib.sha256(str((data.dtype.str,data.shape)).encode()+data.tobytes()).hexdigest()


def module_descriptions(model,excluded,normalize_batchnorm=True):
    """Normalize only the known BatchNorm display change across Torch versions.

    Torch 2.10 omits the bias-presence field; newer Torch prints it. Encode
    that field from the actual parameter, preserving the producing runtime's
    packet hash. Tensor/buffer hashes and all other module text stay strict.
    No module methods or model parameters are changed.
    """
    modules=list(model.named_modules())
    descriptions={name:repr(layer) for name,layer in modules if not name.startswith(excluded)}
    if not normalize_batchnorm:return descriptions
    replacements={}
    for _,layer in modules:
        if type(layer) not in (torch.nn.BatchNorm1d,torch.nn.BatchNorm2d,torch.nn.BatchNorm3d):
            continue
        prefix=(f'{type(layer).__name__}({layer.num_features}, eps={layer.eps}, '
                f'momentum={layer.momentum}, affine={layer.affine}, ')
        legacy=prefix+f'track_running_stats={layer.track_running_stats})'
        canonical=prefix+f'bias={layer.bias is not None}, track_running_stats={layer.track_running_stats})'
        raw=repr(layer)
        # Unknown representation changes still fail the boundary check.
        if raw==legacy:replacements[raw]=canonical
    for name,description in descriptions.items():
        for raw,canonical in replacements.items():description=description.replace(raw,canonical)
        descriptions[name]=description
    return descriptions


def model_function_state(model,include_fc1=True,normalize_batchnorm=True):
    excluded=('fc2.',) if include_fc1 else ('fc1.','fc2.')
    state={k:tensor_hash(v) for k,v in model.state_dict().items() if not k.startswith(excluded)}
    layers=[l for l in model.weight_masks if l!='fc2' and (include_fc1 or l!='fc1')]
    state['weight_masks']={l:tensor_hash(model.weight_masks[l]) for l in layers}
    state['bias_masks']={l:tensor_hash(model.bias_masks[l]) for l in layers}
    state['gru_connection_masks']={k:tensor_hash(v) for k,v in getattr(model,'gru_connection_masks',{}).items()}
    state['adapter_input_masks']={k:tensor_hash(v) for k,v in getattr(model,'adapter_input_masks',{}).items()}
    state['adapter_output_masks']={k:tensor_hash(v) for k,v in getattr(model,'adapter_output_masks',{}).items()}
    state['metadata']=dict(type=f'{type(model).__module__}.{type(model).__qualname__}',
        input_shape=(model.seq_length,model.num_features),architecture_version=getattr(model,'architecture_version',1),
        adapter_mode=getattr(model,'adapter_mode',None),adapter_registry=getattr(model,'adapter_registry',{}),
        structural_protection=getattr(model,'structural_protection',False),
        modules=module_descriptions(model,excluded,normalize_batchnorm))
    return state


def boundary_hash(model,include_fc1=True):return digest(model_function_state(model,include_fc1))


def complete_hash(model,detector):
    from fed_learning.training.checkpoint_state import snapshot_denice_state
    return digest(dict(weights={k:tensor_hash(v) for k,v in model.state_dict().items()},
                       algorithm=snapshot_denice_state(model,detector)))


def state_fingerprint(model,detector):
    from fed_learning.training.checkpoint_state import snapshot_denice_state
    return dict(weights={k:tensor_hash(v) for k,v in model.state_dict().items()},
                algorithm={k:digest(v) for k,v in snapshot_denice_state(model,detector).items()})


def changed_state(before,model,detector):
    after=state_fingerprint(model,detector)
    changes={group:sorted(key for key in set(before[group])|set(after[group])
                         if before[group].get(key)!=after[group].get(key)) for group in before}
    return dict(unchanged=not any(changes.values()),changed=changes)


def rng_snapshot():
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['cpu'])
    if state['cuda'] is not None:torch.cuda.set_rng_state_all(state['cuda'])


def clone(value):return copy.deepcopy(value)


def json_value(value):
    if isinstance(value,(np.ndarray,torch.Tensor)):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):json_value(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_value(v) for v in value]
    return value


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(json_value(value),indent=2,allow_nan=False)+'\n',encoding='utf-8')
