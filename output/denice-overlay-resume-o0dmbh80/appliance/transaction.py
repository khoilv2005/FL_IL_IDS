"""Isolated staging, acceptance and whole-state rollback for one exact patch."""
import copy
import hashlib
import numpy as np
import torch
from .codec import decode
from .config import Rejected
from .state import complete_hash,rng_snapshot,restore_rng


def stage(model,detector,compiled,protocol,seen,preprocessing_hash):
    meta,arrays=decode(compiled.packet,protocol.max_patch_bytes,hashlib.sha256(compiled.packet).hexdigest())
    if meta!=compiled.metadata:raise Rejected('MANIFEST_CHANGED')
    if meta['class_id'] not in seen:raise Rejected('FUTURE_SCOPE')
    if meta['transfer_kind']!='snapshot' or meta['base_version'] is not None:raise Rejected('STALE_BASE')
    if meta['preprocessing_hash']!=preprocessing_hash:raise Rejected('PREPROCESSING_MISMATCH')
    if complete_hash(model,detector)!=meta['receiver_base_hash']:raise Rejected('STALE_BASE')
    candidate=copy.deepcopy(model);route=copy.deepcopy(detector);cid=meta['class_id']
    copied=meta['copied_fc1_rows']
    expected={'head_weight','head_bias'} | ({'fc1_weight','fc1_bias'} if copied else set())
    if set(arrays)!=expected:raise Rejected('INVALID_PATCH_FIELDS')
    if arrays['head_weight'].shape!=tuple(candidate.fc2.weight[cid].shape) or arrays['head_bias'].shape!=():raise Rejected('INVALID_PATCH_SHAPE')
    if copied and (arrays['fc1_weight'].shape!=(len(copied),candidate.fc1.in_features) or arrays['fc1_bias'].shape!=(len(copied),)):
        raise Rejected('INVALID_PATCH_SHAPE')
    with torch.no_grad():
        for position,(src,dst) in enumerate(copied):
            if candidate.unit_ranks['fc1'][dst]!=0:raise Rejected('INSUFFICIENT_CAPACITY')
            candidate.fc1.weight[dst].copy_(torch.from_numpy(arrays['fc1_weight'][position]).to(candidate.fc1.weight.device))
            candidate.fc1.bias[dst].copy_(torch.as_tensor(arrays['fc1_bias'][position],device=candidate.fc1.bias.device))
            candidate.weight_masks['fc1'][dst]=1;candidate.bias_masks['fc1'][dst]=1
            candidate.unit_ranks['fc1'][dst]=2
        candidate.fc2.weight[cid].copy_(torch.from_numpy(arrays['head_weight']).to(candidate.fc2.weight.device))
        candidate.fc2.bias[cid].copy_(torch.as_tensor(arrays['head_bias'],device=candidate.fc2.bias.device))
        candidate.weight_masks['fc2'][cid]=1;candidate.bias_masks['fc2'][cid]=1;candidate.unit_ranks['fc2'][cid]=2
    route.episode_classes[meta['task']]=sorted(set(route.episode_classes[meta['task']])|{cid})
    # All modifications must be in the compiler-authorized parameter coordinates.
    for name,value in model.state_dict().items():
        changed=value.detach().cpu()!=candidate.state_dict()[name].detach().cpu()
        allowed=torch.zeros_like(changed)
        if name in ('fc2.weight','fc2.bias'):allowed[cid]=True
        elif name in ('fc1.weight','fc1.bias'):
            for _,dst in copied:allowed[dst]=True
        if (changed & ~allowed).any():raise Rejected('UNSAFE_UPDATE',name)
    return candidate,route


def install(model,detector,compiled,protocol,seen,preprocessing_hash,acceptance_inputs,acceptance_labels,
            predict,ledger,registry,optimizer=None):
    patch_id=hashlib.sha256(compiled.packet).hexdigest()
    if patch_id in registry.entries:
        valid=registry.validate(model,detector,ledger).get(patch_id,False)
        return model,detector,dict(status='already_committed' if valid else 'rejected',
            reason=None if valid else 'STALE_BASE',patch_id=patch_id,applied=False)
    before=complete_hash(model,detector);rng=rng_snapshot()
    # Live model/router/optimizer/ledger are never written before all checks pass.
    try:
        n=len(acceptance_labels)
        if not protocol.acceptance_min_rows<=n<=protocol.acceptance_max_rows:raise Rejected('INSUFFICIENT_ACCEPTANCE_DATA',str(n))
        if not protocol.allow_unverified_positive_install and not np.any(acceptance_labels==compiled.metadata['class_id']):
            raise Rejected('RECEIVER_POSITIVE_EVIDENCE_UNKNOWN')
        candidate,route=stage(model,detector,compiled,protocol,seen,preprocessing_hash)
        old=predict(copy.deepcopy(model),copy.deepcopy(detector),acceptance_inputs)
        candidate_hash=complete_hash(candidate,route)
        new=predict(candidate,route,acceptance_inputs)
        if complete_hash(candidate,route)!=candidate_hash:raise Rejected('PREDICTION_MUTATED_CANDIDATE')
        old_accuracy=float((old==acceptance_labels).mean());new_accuracy=float((new==acceptance_labels).mean())
        if new_accuracy+protocol.max_accuracy_drop<old_accuracy:raise Rejected('ACCEPTANCE_REGRESSION',f'{old_accuracy}->{new_accuracy}')
        if complete_hash(model,detector)!=before:raise Rejected('LIVE_STATE_MUTATED_DURING_STAGING')
        # Prepare all bookkeeping in isolation before committing any live state.
        pending_registry=copy.deepcopy(registry);pending_ledger=copy.deepcopy(ledger)
        pending_registry.register(patch_id,candidate,route,compiled.metadata['class_id'],compiled.metadata['task'],compiled.pin_masks)
        pending_ledger.install(compiled.metadata['class_id'],patch_id,compiled.metadata['donor'],compiled.metadata['boundary_hash'])
        if not all(pending_registry.validate(candidate,route,pending_ledger).values()):raise Rejected('REGISTRATION_INVARIANT')
        if optimizer is not None:
            old_names={id(p):name for name,p in model.named_parameters()};new_params=dict(candidate.named_parameters())
            rebinding=[]
            for group in optimizer.param_groups:
                rebinding.append([new_params[old_names[id(p)]] for p in group['params']])
            states={new_params[old_names[id(p)]]:copy.deepcopy(v) for p,v in optimizer.state.items()}
            for name,param in candidate.named_parameters():
                for slot in states.get(param,{}).values():
                    if torch.is_tensor(slot) and slot.shape==param.shape:slot[compiled.pin_masks[name].to(slot.device)]=0
            for group,params in zip(optimizer.param_groups,rebinding):group['params']=params
            optimizer.state.clear();optimizer.state.update(states)
        registry.entries=pending_registry.entries
        ledger.observed=pending_ledger.observed;ledger.installed=pending_ledger.installed;ledger.evidence=pending_ledger.evidence
        report=dict(status='committed',patch_id=patch_id,applied=True,acceptance_rows=n,
            old_accuracy=old_accuracy,new_accuracy=new_accuracy,
            rescue=int(((old!=acceptance_labels)&(new==acceptance_labels)).sum()),
            break_count=int(((old==acceptance_labels)&(new!=acceptance_labels)).sum()),receiver_positive_verified=False)
        return candidate,route,report
    except Rejected as exc:
        if complete_hash(model,detector)!=before:raise RuntimeError('Rollback invariant: live state changed') from exc
        return model,detector,dict(status='rejected',reason=exc.reason,detail=exc.detail,applied=False,rollback_verified=True,patch_id=patch_id)
    finally:restore_rng(rng)
