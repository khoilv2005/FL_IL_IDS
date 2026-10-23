"""Transactional receiver-native head patches; separate from exact closure.

Only the imported FC2 row is pinned. Upstream drift requires functional
recertification with the locked calibration acceptance set. No donor backbone
equality, dependency graft, or validation-driven threshold update is performed.
"""
import copy
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib

import numpy as np
import torch

from .config import Rejected
from .portable_route import PORTABLE_RULES, ProtectedRoute, receiver_signals, transitions
from .state import boundary_hash, complete_hash, digest, rng_snapshot, restore_rng


@dataclass(frozen=True)
class HeadContract:
    version: str = 'appliance_guarded_head_install_v1'
    min_receiver_rows: int = 32
    min_positive_rows: int = 32
    min_recall: float = .95
    max_receiver_far: float = .001
    max_break: int = 0


def router_hash(router):
    from fed_learning.training.checkpoint_state import snapshot_context_detector
    return digest(snapshot_context_detector(router))


def head_snapshot(model, cid):
    return dict(weight=model.fc2.weight[cid].detach().cpu().clone(),
                bias=model.fc2.bias[cid].detach().cpu().clone(),
                weight_mask=model.weight_masks['fc2'][cid].detach().cpu().clone(),
                bias_mask=model.bias_masks['fc2'][cid].detach().cpu().clone(),
                rank=int(model.unit_ranks['fc2'][cid]))


@torch.no_grad()
def put_head(model, cid, state):
    for name in ('weight','bias'):
        value=getattr(model.fc2,name)
        value[cid].copy_(state[name].to(value.device))
    for name, field in (('weight_mask','weight_masks'),('bias_mask','bias_masks')):
        target=getattr(model,field)['fc2']
        target[cid].copy_(state[name].to(target.device))
    model.unit_ranks['fc2'][cid]=state['rank']


@contextmanager
def original_local_head(model, cid, backup):
    """One local backbone, two head views. Single-thread pairwise runtime.

    Preserve the complete pre-import row, including masked zero logits that
    might have been in an inherited router mask. Do not silently widen masks.
    """
    live=head_snapshot(model,cid)
    try:
        put_head(model,cid,backup)
        yield
    finally:
        put_head(model,cid,live)


def compile_guarded_head(model, router, packet, receiver_id, seen, preprocessing_hash):
    route=ProtectedRoute.from_packet(packet)
    meta=route.metadata
    if meta.get('version')!=PORTABLE_RULES['version']:
        raise Rejected('UNSUPPORTED_ROUTE_VERSION')
    cid=int(meta['class_id'])
    if int(meta['receiver'])!=receiver_id:
        raise Rejected('WRONG_RECEIVER')
    if meta['signature']['kind']!='shared_sketch':
        raise Rejected('PORTABLE_SHARED_ROUTE_REQUIRED')
    if cid not in seen or not 0<=cid<model.fc2.out_features:
        raise Rejected('FUTURE_SCOPE')
    if int(meta['task'])>max(router.activation_memory):
        raise Rejected('FUTURE_ROUTE')
    if int(model.unit_ranks['fc2'][cid])!=0:
        raise Rejected('PROTECTED_OR_OCCUPIED_OUTPUT_SLOT')
    if getattr(model,'continual_head',None) is not None or getattr(model,'local_classifier',None) is not None:
        raise Rejected('UNSUPPORTED_AUXILIARY_CLASSIFIER')
    route.validate(model,preprocessing_hash,seen)
    if not np.isfinite(route.head_bias) or not np.isfinite(route.head_weight).all():
        raise Rejected('NONFINITE_GUARDED_HEAD')
    return dict(kind='guarded_head_only',version=HeadContract().version,receiver=receiver_id,
        donor=int(meta['donor']),class_id=cid,task=int(meta['task']),
        receiver_base_hash=complete_hash(model,router),packet=bytes(packet),
        patch_id=hashlib.sha256(packet).hexdigest(),preprocessing_sha256=preprocessing_hash,
        donor_boundary_equality_required=False,fc1_rows_copied=0)


class GuardedHeadRegistry:
    """One capability per probe; row pinning plus fail-closed imported route."""
    freeze_bn=False

    def __init__(self):
        self.entries={}

    def summary(self):
        fields=('class_id','task','donor','patch_id','patch_version','route_version',
                'installed','protected','valid','reason','feature_version','route_state_version')
        return {cid:{k:e.get(k) for k in fields} for cid,e in self.entries.items()}

    def sync(self,model):
        model.imported_registry=self.summary()
        # The complete registry, including packet and original head backup,
        # belongs to the receiver's algorithm checkpoint, even while inactive.
        model.appliance_guarded_head_entries=self.entries

    @torch.no_grad()
    def protect(self,model,optimizer=None):
        # Invalid routes remain protected until an explicit uninstall/refresh.
        for cid,e in self.entries.items():
            put_head(model,cid,e['installed_head'])
            if optimizer is not None:
                for parameter in (model.fc2.weight,model.fc2.bias):
                    for slot in optimizer.state.get(parameter,{}).values():
                        if torch.is_tensor(slot) and slot.shape==parameter.shape:
                            slot[cid]=0
        self.sync(model)

    def gradient_filter(self,model):
        model.reset_frozen_gradients()
        for cid in self.entries:
            for parameter in (model.fc2.weight,model.fc2.bias):
                if parameter.grad is not None:
                    parameter.grad[cid]=0

    def head_matches(self,model,cid):
        now=head_snapshot(model,cid)
        return all(torch.equal(now[k],v) if torch.is_tensor(v) else now[k]==v
                   for k,v in self.entries[cid]['installed_head'].items())

    def certificate_current(self,model,router,cid):
        e=self.entries[cid]
        return (e['valid'] and self.head_matches(model,cid) and
                boundary_hash(model,True)==e['feature_version'] and
                router_hash(router)==e['route_state_version'])

    def records(self,model,router,inputs,seen,device,batch_size,candidate=False):
        if len(self.entries)!=1:
            raise Rejected('SINGLE_CAPABILITY_PROBE_ONLY')
        cid,e=next(iter(self.entries.items()))
        with original_local_head(model,cid,e['backup']):
            base=receiver_signals(model,router,inputs,seen,e['task'],cid,batch_size,device)
        certified=self.certificate_current(model,router,cid)
        if not candidate and not certified:
            # Never score a corrupted/uncertified installed row before falling
            # back. In particular a nonfinite imported head must not poison the
            # otherwise finite original local prediction.
            e.update(valid=False,reason='uncertified_feature_router_or_head_drift')
            self.sync(model)
            return dict(pred=base['local_pred'],activated=np.zeros(len(inputs),bool),
                local_pred=base['local_pred'],local_confidence=base['local_confidence'],
                signature_score=None,margin=None,certified=False,imported_route_scored=False)
        route=ProtectedRoute.from_packet(e['packet'])
        # Inference uses the installed row, not an independent cached head.
        head=head_snapshot(model,cid)
        route.head_weight=(head['weight']*head['weight_mask']).numpy()
        route.head_bias=float(head['bias']*head['bias_mask'])
        sig=route.signals(base,inputs)
        decision=route.decisions(sig)
        return dict(**decision,local_pred=base['local_pred'],local_confidence=base['local_confidence'],
                    signature_score=sig['signature_score'],margin=sig['margin'],certified=certified,
                    imported_route_scored=True)

    def recertify(self,model,router,acceptance,seen,device,batch_size,contract=HeadContract()):
        cid,e=next(iter(self.entries.items()))
        if not self.head_matches(model,cid):
            e.update(valid=False,reason='imported_head_mutated');self.sync(model)
            return dict(passed=False,reason=e['reason'])
        report=acceptance_gate(self,model,router,acceptance,seen,device,batch_size,contract)
        e.update(valid=report['passed'],reason=None if report['passed'] else 'calibration_recertification_failed')
        if report['passed']:
            e.update(feature_version=boundary_hash(model,True),route_state_version=router_hash(router))
        self.sync(model)
        return report


def acceptance_gate(registry,model,router,pool,seen,device,batch_size,contract):
    cid,e=next(iter(registry.entries.items()))
    y=np.asarray(pool['y']);origin=np.asarray(pool['origin_client'])
    receiver=origin==e['receiver'];positive=y==cid
    if int(receiver.sum())<contract.min_receiver_rows or int(positive.sum())<contract.min_positive_rows:
        raise Rejected('INSUFFICIENT_GUARDED_ACCEPTANCE')
    if np.any(positive&receiver):
        raise Rejected('MISSING_CLASS_RECEIVER_HAS_POSITIVES')
    record=registry.records(model,router,pool['X'],seen,device,batch_size,candidate=True)
    cells=transitions(y,record['local_pred'],record['pred'])
    recall=float((record['pred'][positive]==cid).mean())
    far=float(record['activated'][receiver].mean())
    passed=(recall>=contract.min_recall and far<=contract.max_receiver_far and
            cells['break_count']<=contract.max_break and cells['rescue']>0)
    return dict(passed=bool(passed),recall=recall,receiver_far=far,
                receiver_rows=int(receiver.sum()),positive_rows=int(positive.sum()),
                transitions=cells,thresholds_retuned=False,role='locked calibration HOLDOUT; development only')


def install_guarded_head(model,router,compiled,registry,acceptance,seen,device,batch_size,contract=HeadContract()):
    before=complete_hash(model,router);bookkeeping=digest(registry.entries);rng=rng_snapshot()
    cid=compiled['class_id'];patch_id=compiled['patch_id']
    try:
        if cid in registry.entries:
            e=registry.entries[cid]
            if e['patch_id']==patch_id and registry.certificate_current(model,router,cid):
                return model,router,registry,dict(status='already_committed',applied=False,patch_id=patch_id)
            raise Rejected('INSTALLED_VERSION_CONFLICT_OR_STALE')
        if registry.entries:
            raise Rejected('SINGLE_CAPABILITY_PROBE_ONLY')
        if before!=compiled['receiver_base_hash']:
            raise Rejected('STALE_BASE')
        if hashlib.sha256(compiled['packet']).hexdigest()!=patch_id:
            raise Rejected('PACKET_HASH_MISMATCH')
        route=ProtectedRoute.from_packet(compiled['packet'])
        if compiled.get('version')!=HeadContract().version or route.metadata.get('version')!=PORTABLE_RULES['version']:
            raise Rejected('UNSUPPORTED_INSTALL_OR_ROUTE_VERSION')
        if any(int(route.metadata[k])!=int(compiled[k]) for k in ('class_id','task','receiver','donor')):
            raise Rejected('INSTALL_MANIFEST_MISMATCH')
        route.validate(model,compiled['preprocessing_sha256'],seen)
        candidate=copy.deepcopy(model);detector=copy.deepcopy(router);pending=copy.deepcopy(registry)
        backup=head_snapshot(candidate,cid)
        put_head(candidate,cid,dict(weight=torch.from_numpy(route.head_weight),bias=torch.tensor(route.head_bias),
            weight_mask=torch.ones_like(backup['weight_mask']),bias_mask=torch.ones_like(backup['bias_mask']),rank=2))
        for name,value in model.state_dict().items():
            changed=value.detach().cpu()!=candidate.state_dict()[name].detach().cpu()
            allowed=torch.zeros_like(changed)
            if name in ('fc2.weight','fc2.bias'):allowed[cid]=True
            if (changed&~allowed).any():raise Rejected('UNSAFE_STAGING_UPDATE',name)
        pending.entries[cid]=dict(class_id=cid,task=compiled['task'],receiver=compiled['receiver'],
            donor=compiled['donor'],patch_id=patch_id,packet=compiled['packet'],backup=backup,
            installed_head=head_snapshot(candidate,cid),patch_version=compiled['version'],
            route_version=route.metadata['version'],installed=True,protected=True,valid=False,reason='staging',
            feature_version=boundary_hash(candidate,True),route_state_version=router_hash(detector))
        report=pending.recertify(candidate,detector,acceptance,seen,device,batch_size,contract)
        if not report['passed']:
            raise Rejected('GUARDED_ACCEPTANCE_FAILED',str(report))
        if before!=complete_hash(model,router) or bookkeeping!=digest(registry.entries):
            raise RuntimeError('Live source changed during staging')
        return candidate,detector,pending,dict(status='committed',applied=True,patch_id=patch_id,
            acceptance=report,only_fc2_row_written=True,donor_boundary_equality_required=False)
    except Rejected as exc:
        if before!=complete_hash(model,router) or bookkeeping!=digest(registry.entries):
            raise RuntimeError('Rollback invariant violated') from exc
        return model,router,registry,dict(status='rejected',applied=False,reason=exc.reason,
            detail=exc.detail,rollback_verified=True,patch_id=patch_id)
    finally:
        restore_rng(rng)
