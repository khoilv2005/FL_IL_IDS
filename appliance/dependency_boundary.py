"""Exact effective dependency fingerprints for the adapter-free CNN-GRU head.

This audits a receiver function; it does not claim donor/receiver coordinates
are equivalent, freeze an entire backbone, or authorize transplantation. The
GRU dependency closure includes all gates and recurrent predecessors in both
layers. Unsupported branches are rejected instead of silently omitted.
"""
import hashlib
import inspect

import numpy as np
import torch

from .closure import effective_linear
from .config import Rejected
from .state import digest


def head_dependency_boundary(model,head_weights,*,tensor_sink=None):
    if model.adapter_registry or model.continual_head is not None or model.local_classifier is not None:
        raise Rejected('UNSUPPORTED_DEPENDENCY_BOUNDARY_BRANCH')
    weights=torch.as_tensor(head_weights).cpu()
    if weights.ndim==1:weights=weights[None,:]
    if weights.ndim!=2 or weights.shape[1]!=model.fc1.out_features or not torch.isfinite(weights).all():
        raise Rejected('INVALID_HEAD_DEPENDENCY_WEIGHTS')
    fc1_rows=torch.nonzero((weights!=0).any(0)).flatten()
    w,b=effective_linear(model,'fc1')
    input_cols=torch.nonzero((w[fc1_rows]!=0).any(0)).flatten().numpy()
    cnn_len=model.seq_length//8;cnn_width=model.conv3.out_channels*cnn_len
    convolution={};current=set(int(c)//cnn_len for c in input_cols if c<cnn_width)
    tensors={'fc1.weight':w[fc1_rows],'fc1.bias':b[fc1_rows]}
    for layer in ('conv3','conv2','conv1'):
        convolution[layer]=sorted(current);rows=torch.as_tensor(sorted(current),dtype=torch.long)
        conv=getattr(model,layer);index=layer[-1];bn=getattr(model,'bn'+index)
        ew=conv.weight.detach().cpu()*model.weight_masks[layer].cpu()
        eb=conv.bias.detach().cpu()*model.bias_masks[layer].cpu()
        tensors[layer+'.weight']=ew[rows];tensors[layer+'.bias']=eb[rows]
        for name in ('weight','bias','running_mean','running_var'):tensors['bn'+index+'.'+name]=getattr(bn,name).detach().cpu()[rows]
        current=set(torch.nonzero((ew[rows]!=0).any(0).any(1)).flatten().tolist())
    gru_units=set(int(c)-cnn_width for c in input_cols if c>=cnn_width and model.weight_masks['gru'][int(c)-cnn_width]!=0)
    tensors['gru.output_mask']=model.weight_masks['gru'].cpu()[sorted(gru_units)]
    recurrent={};hidden=model.gru.hidden_size
    if model.gru.num_layers!=2 or model.gru.bidirectional:raise Rejected('UNSUPPORTED_RECURRENT_BOUNDARY')
    for layer in (1,0):
        values={}
        for prefix in ('weight_ih','weight_hh','bias_ih','bias_hh'):
            name=f'{prefix}_l{layer}';value=getattr(model.gru,name).detach().cpu()
            if model.structural_protection and name in model.gru_connection_masks:value=value*model.gru_connection_masks[name].cpu()
            values[prefix]=value
        # The recurrent predecessor relation must be transitively closed.
        while True:
            gates=sorted(u+g*hidden for u in gru_units for g in range(3))
            parents=set(torch.nonzero((values['weight_hh'][gates]!=0).any(0)).flatten().tolist())
            expanded=gru_units|parents
            if expanded==gru_units:break
            gru_units=expanded
        recurrent[str(layer)]=sorted(gru_units);gates=sorted(u+g*hidden for u in gru_units for g in range(3))
        for name,value in values.items():tensors[f'gru.{name}_l{layer}']=value[gates]
        gru_units=set(torch.nonzero((values['weight_ih'][gates]!=0).any(0)).flatten().tolist()) if layer else set()
    scope=dict(fc1=fc1_rows.tolist(),**convolution,gru=recurrent)
    mature={name:bool(np.all(np.asarray(model.unit_ranks[name])[rows]>=2)) for name,rows in scope.items() if name!='gru'}
    mature['gru']=all(bool(np.all(np.asarray(model.unit_ranks['gru'])[rows]>=2)) for rows in recurrent.values())
    methods=('penultimate_features','_forward_backbone','_apply_masked_conv','_apply_partial_frozen_bn',
             '_run_gru','_apply_masked_linear')
    implementation={name:hashlib.sha256(inspect.getsource(getattr(model,name)).encode()).hexdigest() for name in methods}
    operators=dict(input_shape=list(model.input_shape),gru_hidden=hidden,gru_layers=model.gru.num_layers,
        model_type=f'{type(model).__module__}.{type(model).__qualname__}',
        activation=repr(model.relu),forward_implementation=implementation,
        head_query_fingerprint=digest(weights),
        conv={n:repr(getattr(model,n)) for n in ('conv1','conv2','conv3')},
        pool={n:repr(getattr(model,n)) for n in ('pool1','pool2','pool3')},
        bn_eps={n:float(getattr(model,n).eps) for n in ('bn1','bn2','bn3')})
    if any(not torch.isfinite(value).all() for value in tensors.values()):
        raise Rejected('NONFINITE_DEPENDENCY_FUNCTION')
    if tensor_sink is not None:
        tensor_sink.update({name:value.clone() for name,value in tensors.items()})
    # Preserve the strict byte fingerprint above. A separate value fingerprint
    # identifies signed-zero changes caused by masks; it is not a certificate.
    semantic_tensors={name:torch.where(value==0,torch.zeros_like(value),value)
                      for name,value in tensors.items()}
    # Keep the original byte/value schemas intact for existing certificates.
    # The original value schema canonicalized tensors but omitted the query
    # inside operators, so equivalent +0/-0 could incorrectly report drift.
    semantic_operators=dict(operators,head_query_fingerprint=digest(
        torch.where(weights==0,torch.zeros_like(weights),weights)))
    return dict(version='appliance_head_dependency_boundary_v2',scope=scope,
        mature_dependency_scope=mature,all_dependencies_mature=all(mature.values()),
        function_sha=digest(dict(scope=scope,operators=operators,tensors=tensors)),
        parameter_value_function_sha=digest(dict(scope=scope,operators=operators,tensors=semantic_tensors)),
        parameter_value_schema='effective-eval-tensors-signed-zero-equivalence-v1',
        canonical_value_function_sha=digest(dict(scope=scope,operators=semantic_operators,tensors=semantic_tensors)),
        canonical_value_schema='effective-eval-tensors-and-query-signed-zero-equivalence-v2',
        effective_tensor_fingerprints={name:digest(value) for name,value in tensors.items()},
        effective_tensor_shapes={name:list(value.shape) for name,value in tensors.items()},
        effective_tensor_elements=sum(v.numel() for v in tensors.values()),
        extra_parameters_frozen=0,authorizes_installation=False,
        limitation='head/margin function only; normal router/confidence can change independently')
