"""Receiver routing features from an existing, invariant mature FC1 branch.

No extra backbone and no additional freezing. Invalidating a branch never reads
historical samples or silently refreshes stored class moments.
"""
import copy
import numpy as np
import torch
from .config import Rejected
from .dependency_boundary import head_dependency_boundary

class MatureRoutingBranch:
    VERSION='appliance_receiver_mature_routing_branch_v1'
    SMALL_VERSION='appliance_receiver_mature_routing_branch_16_v2'

    def __init__(self,state):
        self._state=copy.deepcopy(state)
        rows=state.get('rows',[])
        expected_width={self.VERSION:64,self.SMALL_VERSION:16}.get(state.get('version'))
        if (expected_width is None or not isinstance(rows,list) or len(rows)!=expected_width
                or any(type(k) is not int or k<0 for k in rows) or len(rows)!=len(set(rows))
                or not 0<=int(state.get('birth_task',-1))<6):
            raise Rejected('INVALID_MATURE_ROUTING_BRANCH')

    @staticmethod
    def runtime(model):
        parameter=next(model.parameters())
        cuda=parameter.device.type=='cuda'
        return dict(torch_version=str(torch.__version__),device_type=parameter.device.type,
            parameter_dtype=str(parameter.dtype),matmul_tf32=bool(torch.backends.cuda.matmul.allow_tf32) if cuda else None,
            cudnn_tf32=bool(torch.backends.cudnn.allow_tf32) if cuda else None)

    @classmethod
    def capture(cls,model,preprocessing_sha256,task,width=64):
        if type(task) is not int or not 0<=task<6 or type(width) is not int or width not in (16,64):raise Rejected('MATURE_BRANCH_POLICY_CHANGED')
        if cls.runtime(model)['parameter_dtype']!='torch.float32':raise Rejected('MATURE_BRANCH_REQUIRES_FP32')
        rows=np.flatnonzero(np.asarray(model.unit_ranks['fc1'])>=2)[:width].tolist()
        if len(rows)!=width:raise Rejected('INSUFFICIENT_MATURE_ROUTING_FEATURES')
        query=torch.eye(model.fc1.out_features)[rows]
        boundary=head_dependency_boundary(model,query)
        if not boundary['all_dependencies_mature']:raise Rejected('UNPROTECTED_ROUTING_DEPENDENCY')
        return cls(dict(version=cls.VERSION if width==64 else cls.SMALL_VERSION,rows=rows,input_shape=list(model.input_shape),birth_task=int(task),
            preprocessing_sha256=preprocessing_sha256,runtime=cls.runtime(model),
            dependency_parameter_value_sha=boundary['parameter_value_function_sha'],
            dependency_schema=boundary['parameter_value_schema'],extra_backbones=0,extra_parameters_frozen=0))

    def state(self):return copy.deepcopy(self._state)

    def validate(self,model,preprocessing_sha256):
        s=self._state
        if (s['preprocessing_sha256']!=preprocessing_sha256 or list(model.input_shape)!=s['input_shape']
                or self.runtime(model)!=s['runtime'] or any(k>=model.fc1.out_features for k in s['rows'])):
            raise Rejected('MATURE_ROUTING_SPACE_CHANGED')
        boundary=head_dependency_boundary(model,torch.eye(model.fc1.out_features)[s['rows']])
        if (not boundary['all_dependencies_mature'] or boundary['parameter_value_schema']!=s['dependency_schema']
                or boundary['parameter_value_function_sha']!=s['dependency_parameter_value_sha']):
            raise Rejected('MATURE_ROUTING_FUNCTION_DRIFT')
        return True

    @torch.no_grad()
    def features(self,model,inputs,preprocessing_sha256,batch_size=512):
        self.validate(model,preprocessing_sha256)
        x=np.asarray(inputs,np.float32)
        if tuple(x.shape[1:])!=tuple(self._state['input_shape']) or not np.isfinite(x).all() or batch_size<1:
            raise Rejected('MATURE_ROUTING_INPUT_MISMATCH')
        modes=[(module,module.training) for module in model.modules()]
        active=copy.deepcopy(model.active_adapters);parts=[]
        try:
            model.eval();model.clear_active_adapters();device=next(model.parameters()).device
            for start in range(0,len(x),batch_size):
                tensor=torch.as_tensor(x[start:start+batch_size],device=device)
                # Training may have an outer AMP context. Routing statistics
                # always use the declared FP32 space instead of inheriting it.
                with torch.autocast(device_type=device.type,enabled=False):
                    values=model.penultimate_features(tensor)[:,self._state['rows']].cpu().numpy()
                if not np.isfinite(values).all():raise Rejected('NONFINITE_MATURE_ROUTING_FEATURES')
                parts.append(values)
        finally:
            model.active_adapters=active
            for module,flag in modes:module.training=flag
        self.validate(model,preprocessing_sha256)
        return np.concatenate(parts) if parts else np.zeros((0,len(self._state['rows'])),np.float32)
