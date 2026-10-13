"""Conservative exact compiler for the existing CNN-GRU/FC model.

Supports a shared full backbone boundary, plus FC1 dependency rows and a head
row. Divergent CNN/GRU or adapter-dependent FC1 remapping is audited/rejected;
there is no hidden approximation or full-backbone fallback.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
import torch

from .config import Rejected
from .state import boundary_hash,complete_hash
from .codec import encode


@dataclass
class Compiled:
    metadata: dict
    tensors: dict
    pin_masks: dict
    packet: bytes
    report: dict


def effective_linear(model,layer):
    module=getattr(model,layer)
    return (module.weight.detach().cpu()*model.weight_masks[layer].detach().cpu(),
            module.bias.detach().cpu()*model.bias_masks[layer].detach().cpu())


def empty_pins(model):return {n:torch.zeros_like(p,dtype=torch.bool,device='cpu') for n,p in model.named_parameters()}


def compile_patch(receiver,donor,receiver_router,donor_router,class_id,task,seen,
                  receiver_id,donor_id,preprocessing_hash,protocol,kind='dependency_complete'):
    protocol.validate();report=dict(kind=kind,class_id=class_id,task=task,receiver=receiver_id,donor=donor_id,
        implementation_scope='shared CNN/GRU boundary; FC1 row closure and head; no CNN/GRU remapping',
        receiver_backbone_hash=boundary_hash(receiver,False),donor_backbone_hash=boundary_hash(donor,False),
        receiver_penultimate_hash=boundary_hash(receiver,True),donor_penultimate_hash=boundary_hash(donor,True))
    params=dict(donor.named_parameters())
    report['conservative_full_dependency_bytes']=sum(p.numel()*p.element_size() for n,p in params.items() if not n.startswith('fc2.'))
    report['conservative_full_dependency_bytes']+=sum(v.numel()*v.element_size() for n,v in donor.named_buffers())
    try:
        if kind not in ('head_only','dependency_complete'):raise Rejected('INVALID_PATCH_KIND')
        if class_id not in seen:raise Rejected('FUTURE_SCOPE')
        if receiver.fc2.weight.shape!=donor.fc2.weight.shape or receiver.fc1.weight.shape!=donor.fc1.weight.shape:
            raise Rejected('ARCHITECTURE_MISMATCH')
        if receiver.unit_ranks['fc2'][class_id]!=0:raise Rejected('PROTECTED_OR_OCCUPIED_OUTPUT_SLOT')
        if donor.unit_ranks['fc2'][class_id]==0 or class_id not in donor_router.episode_classes.get(task,[]):
            raise Rejected('DONOR_CLASS_NOT_REPRESENTED')
        if getattr(receiver,'continual_head',None) is not None or getattr(donor,'continual_head',None) is not None:
            raise Rejected('UNSUPPORTED_RESIDUAL_HEAD')
        if getattr(receiver,'local_classifier',None) is not None or getattr(donor,'local_classifier',None) is not None:
            raise Rejected('UNSUPPORTED_LOCAL_CLASSIFIER')
        if receiver_router.router_mode!=donor_router.router_mode:raise Rejected('ROUTER_POLICY_MISMATCH')
        if task not in receiver_router.activation_memory or not receiver_router.episode_classes.get(task):
            raise Rejected('ROUTE_UNAVAILABLE','V1 reuses an existing receiver task profile; does not fabricate one')
        dw,db=effective_linear(donor,'fc1');rw,rb=effective_linear(receiver,'fc1')
        head,bias=effective_linear(donor,'fc2')
        used=torch.nonzero(head[class_id]!=0).flatten().tolist()
        report['donor_fc1_dependency_rows']=used
        report['dependency_graph']=dict(output=f'fc2[{class_id}]',linear_rows=used,
            parents=['fc1 ReLU/bias/masks','CNN conv1-3 and BN affine/running buffers','GRU both layers/all gates/recurrent edges',
                     'active context adapters and their masks','receiver task routing boundary'],
            boundary_policy='full shared prefix; no unproven coordinate correspondence')
        report['fc1_dependency_snapshot_bytes']=len(used)*(donor.fc1.in_features+1)*4
        report['head_snapshot_bytes']=(donor.fc2.in_features+1)*4
        report['size_estimate_policy']='dense FP32 rows, excludes reuse/remapping, manifest overhead and expanded CNN/GRU dependencies; not a minimal-closure lower bound'
        if report['receiver_backbone_hash']!=report['donor_backbone_hash']:
            report['closure_expansion_required']='CNN/BN/GRU prefix and route-space dependencies'
            raise Rejected('INCOMPATIBLE_BOUNDARY','Divergent backbone; expanded closure requires a mapper not implemented in V1')
        if kind=='head_only' and report['receiver_penultimate_hash']!=report['donor_penultimate_hash']:
            raise Rejected('INCOMPATIBLE_PENULTIMATE','Head-only does not satisfy full feature boundary')
        adapters=bool(getattr(receiver,'adapter_registry',{})) or bool(getattr(donor,'adapter_registry',{}))
        if adapters and report['receiver_penultimate_hash']!=report['donor_penultimate_hash']:
            raise Rejected('UNSUPPORTED_ADAPTER_CLOSURE','FC1 remapping with adapter dependencies is not yet supported')
        occupied=np.flatnonzero(np.asarray(receiver.unit_ranks['fc2'])>0)
        aw,_=effective_linear(receiver,'fc2')
        free=[]
        for row in np.flatnonzero(np.asarray(receiver.unit_ranks['fc1'])==0):
            if not len(occupied) or not torch.any(aw[occupied,int(row)]!=0):free.append(int(row))
        mapping={};copied=[];reserved=set()
        for src in used:
            equal=torch.equal(dw[src],rw[src]) and torch.equal(db[src],rb[src])
            if equal and (int(receiver.unit_ranks['fc1'][src])>0 or src in free):
                target=src
            else:
                if kind=='head_only':raise Rejected('INCOMPATIBLE_PENULTIMATE')
                available=[r for r in free if r not in reserved and r not in used]
                if not available:raise Rejected('INSUFFICIENT_CAPACITY','No young FC1 slot isolated from currently supported outputs')
                target=available[0];copied.append((src,target))
            if target in reserved:raise Rejected('NONINJECTIVE_MAPPING')
            mapping[src]=target;reserved.add(target)
        # Copy at the same coordinate still requires explicit assignment/rank promotion
        # when a matching dependency is an unallocated reserve slot.
        for src,target in mapping.items():
            if receiver.unit_ranks['fc1'][target]==0 and (src,target) not in copied:
                if kind=='head_only':raise Rejected('HEAD_REQUIRES_DEPENDENCY_PROMOTION')
                copied.append((src,target))
        pins=empty_pins(receiver)
        for name in pins:
            if not name.startswith(('fc1.','fc2.')):pins[name].fill_(True)
        if adapters:
            pins['fc1.weight'].fill_(True);pins['fc1.bias'].fill_(True)
        else:
            for row in reserved:pins['fc1.weight'][row]=True;pins['fc1.bias'][row]=True
        pins['fc2.weight'][class_id]=True;pins['fc2.bias'][class_id]=True
        from fed_learning.strategies.decentralized.denice_aggregation import build_compatible_mask
        allowed=build_compatible_mask(receiver.state_dict(),receiver.unit_ranks)
        new=sum(int((mask & allowed[name].cpu().bool()).sum()) for name,mask in pins.items())
        total=sum(p.numel() for p in receiver.parameters())
        report.update(mapping=mapping,copied_fc1_rows=copied,reused_fc1_rows=[s for s,t in mapping.items() if (s,t) not in copied],
            reserved_slots_used=len(copied),new_pinned_parameters=new,total_parameters=total,new_pinned_fraction=new/total,
            pinned_parameters=sum(int(m.sum()) for m in pins.values()),capacity_available=len(free))
        if new/total>protocol.max_new_pinned_fraction:raise Rejected('PIN_BUDGET_EXCEEDED',str(new/total))
        row=torch.zeros_like(head[class_id])
        for src,target in mapping.items():row[target]=head[class_id,src]
        tensors=dict(head_weight=row.numpy(),head_bias=np.asarray(bias[class_id].item(),dtype=np.float32))
        if copied:
            tensors['fc1_weight']=torch.stack([dw[s] for s,t in copied]).numpy()
            tensors['fc1_bias']=torch.stack([db[s] for s,t in copied]).numpy()
        metadata=dict(protocol=protocol.version,kind=kind,transfer_kind='snapshot',base_version=None,
            receiver=receiver_id,donor=donor_id,class_id=class_id,task=task,
            preprocessing_hash=preprocessing_hash,receiver_base_hash=complete_hash(receiver,receiver_router),
            boundary_hash=report['receiver_backbone_hash'],mapping=[[s,t] for s,t in sorted(mapping.items())],
            copied_fc1_rows=[[s,t] for s,t in copied],dependency_contract='pin prefix tensors/BN and used FC1 rows; invalidate on drift')
        packet=encode(metadata,tensors,protocol.max_patch_bytes)
        report.update(accepted_for_staging=True,payload_bytes=len(packet),patch_id=hashlib.sha256(packet).hexdigest())
        return report,Compiled(metadata,tensors,pins,packet,report)
    except Rejected as exc:
        report.update(accepted_for_staging=False,reason=exc.reason,detail=exc.detail)
        return report,None
