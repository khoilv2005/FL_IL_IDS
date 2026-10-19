"""Receiver-local stable head guards with an owned finite BASE support shield.

The activation policy is explicitly signature+fixed-context margin+BASE veto.
Normal router confidence is logged only, not a guard dependency. Acceptance
still requires client-local current CAL aggregates. No historical raw recertify.
"""
import copy
import hashlib
import inspect
from contextlib import ExitStack
import numpy as np
import torch
from .base_sketch_shield import CurrentBaseSketchShield
from .current_calibration_data import CurrentCalibrationData
from .imported_route import stratified_roles,ROUTE_RULES
from .closure import effective_linear
from .config import Rejected
from .dependency_boundary import head_dependency_boundary
from .distributed_calibration import LocalCalibrationEndpoint
from .guarded_head import GuardedHeadRegistry,compile_guarded_head,head_snapshot,put_head,original_local_head
from .portable_route import ProtectedRoute,receiver_signals
from .state import complete_hash,digest,rng_snapshot,restore_rng
from .patch_lifecycle import (bind_initial_certificate, certificate_binding_current,
    reconcile, route_authorized, sealed_sha256)
from .cumulative_certificate import append_current_evidence, validated_scope
from .empirical_deployment import bind_deployment

POLICY='appliance_signature_mature_margin_BASE_veto_v1'

def references(router,task,cid,seen):
    result=sorted((set(map(int,router.episode_classes.get(task,[]))) & set(seen))-{cid})
    if not result:raise Rejected('NO_STABLE_MARGIN_REFERENCE')
    return result

def guard_function(model,route,refs):
    if route.metadata.get('margin_reference_scope','import_context')!='import_context':
        raise Rejected('STABLE_FIXED_CONTEXT_MARGIN_REQUIRED')
    if not refs or any(type(c) is not int or not 0<=c<model.fc2.out_features for c in refs):
        raise Rejected('INVALID_STABLE_HEAD_REFERENCE')
    if any(int(model.unit_ranks['fc2'][c])<2 for c in refs):
        raise Rejected('NONMATURE_STABLE_MARGIN_REFERENCE')
    w,b=effective_linear(model,'fc2')
    query=torch.cat([torch.from_numpy(route.head_weight).reshape(1,-1),w[refs]],0)
    boundary=head_dependency_boundary(model,query)
    if not boundary['all_dependencies_mature']:raise Rejected('NONMATURE_STABLE_GUARD_DEPENDENCY')
    weight,bias=w[refs],b[refs]
    weight=torch.where(weight==0,torch.zeros_like(weight),weight)
    bias=torch.where(bias==0,torch.zeros_like(bias),bias)
    function=dict(policy=POLICY,prefix_value_fingerprint=boundary['parameter_value_function_sha'],
        reference_classes=refs,reference_weight=weight,reference_bias=bias,
        patch_weight=route.head_weight,patch_bias=route.head_bias,signature=route.metadata['signature'],
        tau=route.metadata['tau'],gamma=route.metadata['gamma'],precision='FP32',
        torch_version=torch.__version__,device_type=next(model.parameters()).device.type,
        implementation_fingerprint=digest({name:hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
            for name,fn in {'guard_function':guard_function,'stable_signals':stable_signals,
                           'shield_veto':CurrentBaseSketchShield.veto}.items()}))
    return dict(fingerprint=digest(function),reference_classes=refs,
        all_dependencies_mature=True,extra_parameters_frozen=0,prefix=boundary,
        policy=POLICY,normal_router_confidence_is_guard_dependency=False)

@torch.no_grad()
def stable_signals(model,router,x,seen,route,refs,batch_size,device):
    # Router supplies baseline prediction only. The imported decision has a
    # fixed reference mask, independent of task-router refresh.
    with torch.autocast(device_type=next(model.parameters()).device.type,enabled=False):
        base=receiver_signals(model,router,x,seen,route.metadata['task'],route.metadata['class_id'],batch_size,device)
    h=base['imported_features'];w,b=effective_linear(model,'fc2')
    local=torch.nn.functional.linear(torch.as_tensor(h),w,b).numpy()
    base['local_best_import_context']=local[:,refs].max(1)
    base['local_unmasked_logits']=local
    sig=route.signals(base,x)
    return sig

class StableCalibrationEndpoint(LocalCalibrationEndpoint):
    """Only current local aggregates leave; support veto is part of guard ID."""
    def __init__(self,client,session,views,guard_declaration):
        if (guard_declaration.get('policy')!=POLICY or
                session.capability_sha!=hashlib.sha256(repr(sorted(guard_declaration.items())).encode()).hexdigest()):
            raise Rejected('STABLE_CALIBRATION_GUARD_BINDING_CHANGED')
        local={}
        for role,view in views.items():
            s=view['signals'];veto=np.asarray(s.get('base_support_veto'))
            if veto.dtype!=np.bool_ or veto.shape!=(len(view['y']),):
                raise Rejected('STABLE_CALIBRATION_BASE_VETO_REQUIRED')
            # Grid beta=1 is an aggregate representation of no confidence gate.
            # Do not mutate original signals or quietly change packet beta.
            signals=dict(s,signature_valid=np.asarray(s['signature_valid'])&~veto,
                local_confidence=np.zeros(len(veto),np.float64))
            local[role]=dict(view,signals=signals)
        super().__init__(client,session,local)
        self.guard_declaration=copy.deepcopy(guard_declaration)

    def count_packet(self,grid,role='selection'):
        if not np.array_equal(grid.betas,np.array([1.])):
            raise Rejected('STABLE_POLICY_FORBIDS_CONFIDENCE_GRID')
        return super().count_packet(grid,role)

def guard_declaration(packet,function,shield,required):
    return dict(policy=POLICY,packet_sha256=hashlib.sha256(packet).hexdigest(),
        fixed_guard_function_fingerprint=function['fingerprint'],
        reference_classes=tuple(function['reference_classes']),
        finite_owned_shield_state_digest=shield.state()['state_digest'],
        required_old_classes=tuple(required),applied_packet_beta=False,
        retained_raw_examples=0,CAL_acceptance_substitution=False)

def declaration_sha(decl):
    return hashlib.sha256(repr(sorted(decl.items())).encode()).hexdigest()

def canonical_guard_fingerprint(model,route,refs):
    """Same fixed guard, canonicalizing signed zeros including its head query.

    Do not change guard_function: its source is bound by existing certificates.
    This separate schema permits a verified numeric-equivalence attestation.
    """
    old=guard_function(model,route,refs)
    w,b=effective_linear(model,'fc2')
    function=dict(policy=POLICY,prefix_value_fingerprint=old['prefix']['canonical_value_function_sha'],
        reference_classes=refs,
        reference_weight=torch.where(w[refs]==0,torch.zeros_like(w[refs]),w[refs]),
        reference_bias=torch.where(b[refs]==0,torch.zeros_like(b[refs]),b[refs]),
        patch_weight=route.head_weight,patch_bias=route.head_bias,signature=route.metadata['signature'],
        tau=route.metadata['tau'],gamma=route.metadata['gamma'],precision='FP32',
        torch_version=torch.__version__,device_type=next(model.parameters()).device.type,
        implementation_fingerprint=digest({name:hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
            for name,fn in {'guard_function':guard_function,'stable_signals':stable_signals,
                           'shield_veto':CurrentBaseSketchShield.veto}.items()}))
    return digest(function)

class StableHeadRegistry(GuardedHeadRegistry):
    def summary(self):
        result=super().summary()
        for cid,e in self.entries.items():
            result[cid].update(lifecycle_state=e.get('lifecycle_state'),
                lifecycle_reason=e.get('lifecycle_reason'),
                lifecycle_certificate_sha256=e.get('lifecycle_certificate_sha256'))
            try:result[cid]['effective_CAL_scope']=validated_scope(e) if 'lifecycle_certificate' in e else None
            except (Rejected,KeyError,ValueError,TypeError):result[cid]['effective_CAL_scope']=None
        return result

    def bind_lifecycle(self,model,router,cid,scope):
        if not self.certificate_current(model,router,cid):
            raise Rejected('INITIAL_FUNCTION_CERTIFICATE_NOT_CURRENT')
        sealed=bind_initial_certificate(self.entries[cid],scope)
        self.sync(model)
        return sealed

    def update_lifecycle(self,model,router,cid,scope,task,conflict=False,new_cal_rows=None):
        e=self.entries[cid]
        if conflict:e['new_conflict_latched']=True
        report=reconcile(e,self.certificate_current(model,router,cid),scope,task,
            conflict=conflict or e.get('new_conflict_latched',False),new_cal_rows=new_cal_rows)
        self.sync(model)
        return report

    def extend_current_scope(self,model,router,cid,receipts):
        e=self.entries[cid]
        if not certificate_binding_current(e):raise Rejected('INITIAL_SCOPE_CERTIFICATE_INVALID')
        result=append_current_evidence(e,receipts,self.certificate_current(model,router,cid))
        self.sync(model)
        return result

    def bind_empirical_deployment(self,model,router,cid):
        e=self.entries[cid]
        if not certificate_binding_current(e) or not self.certificate_current(model,router,cid):
            raise Rejected('EMPIRICAL_DEPLOYMENT_CURRENT_FUNCTION_REQUIRED')
        result=bind_deployment(e,next(model.parameters()).device.type,str(torch.__version__))
        self.sync(model)
        return result

    def attest_signed_zero_equivalence(self,model,router,cid):
        """Bind a new hash schema only while the original certificate matches.

        No CAL access, reacceptance, threshold change, or scope grant. The old
        sealed certificate and its acceptance remain immutable. Never call
        this on an endpoint which already reports original function drift.
        """
        e=self.entries[cid]
        if 'numeric_equivalence_attestation' in e:
            raise Rejected('NUMERIC_EQUIVALENCE_ALREADY_ATTESTED')
        if not certificate_binding_current(e) or not self.certificate_current(model,router,cid):
            raise Rejected('ORIGINAL_CERTIFICATE_REQUIRED_FOR_NUMERIC_EQUIVALENCE')
        route=ProtectedRoute.from_packet(e['packet'])
        a=dict(version='signed_zero_effective_query_v2',
            original_certificate_sha256=e['lifecycle_certificate_sha256'],
            original_guard_function_fingerprint=e['guard_function_fingerprint'],
            canonical_guard_function_fingerprint=canonical_guard_fingerprint(model,route,e['reference_classes']),
            old_certificate_modified=False,thresholds_changed=False,scope_expanded=False,
            calibration_reads=0,numeric_tolerance=0.)
        e['numeric_equivalence_attestation']=a
        e['numeric_equivalence_attestation_sha256']=sealed_sha256(a)
        self.sync(model)
        return copy.deepcopy(a)

    def _numeric_equivalent(self,model,route,e):
        a=e.get('numeric_equivalence_attestation')
        return bool(isinstance(a,dict) and a.get('version')=='signed_zero_effective_query_v2' and
            sealed_sha256(a)==e.get('numeric_equivalence_attestation_sha256') and
            certificate_binding_current(e) and
            a.get('original_certificate_sha256')==e['lifecycle_certificate_sha256'] and
            a.get('original_guard_function_fingerprint')==e['guard_function_fingerprint'] and
            a.get('old_certificate_modified') is False and a.get('thresholds_changed') is False and
            a.get('scope_expanded') is False and a.get('calibration_reads')==0 and a.get('numeric_tolerance')==0. and
            a.get('canonical_guard_function_fingerprint')==canonical_guard_fingerprint(model,route,e['reference_classes']))

    def certificate_current(self,model,router,cid):
        e=self.entries[cid]
        if not e['valid'] or e.get('activation_policy')!=POLICY or not self.head_matches(model,cid):return False
        if 'lifecycle_certificate' in e and not certificate_binding_current(e):return False
        for prior,patch_id in e.get('prior_patch_ids',[]):
            if (prior not in self.entries or self.entries[prior]['patch_id']!=patch_id or
                    self.entries[prior]['installed_sequence']>=e['installed_sequence'] or
                    not self.certificate_current(model,router,prior)):return False
        try:
            route=ProtectedRoute.from_packet(e['packet'])
            live=head_snapshot(model,cid)
            actual_weight=(live['weight']*live['weight_mask']).numpy()
            actual_bias=float(live['bias']*live['bias_mask'])
            if not np.array_equal(actual_weight,route.head_weight) or actual_bias!=route.head_bias:return False
            shield=CurrentBaseSketchShield.restore(e['shield_at_install'])
            function=guard_function(model,route,e['reference_classes'])
            matched=(self._numeric_equivalent(model,route,e) if 'numeric_equivalence_attestation' in e
                     else function['fingerprint']==e['guard_function_fingerprint'])
            # The declaration remains bound to its original immutable seal;
            # numeric-equivalence attestation cannot grant a different policy.
            declared=dict(function,fingerprint=e['guard_function_fingerprint'])
            return (matched and
                not shield.coverage(e['required_old_classes'])['missing_classes'] and
                shield.owner==e['receiver'] and
                e['guard_declaration']==guard_declaration(e['packet'],declared,shield,e['required_old_classes']))
        except (Rejected,KeyError,TypeError,ValueError):return False

    def records(self,model,router,x,seen,device,batch_size,candidate=False,runtime_scope=None):
        if not self.entries:raise Rejected('STABLE_REGISTRY_EMPTY')
        # Oldest accepted capability takes priority; later patches never
        # override the decision of an earlier active imported capability.
        with ExitStack() as stack:
            for cid,e in self.entries.items():stack.enter_context(original_local_head(model,cid,e['backup']))
            first=next(iter(self.entries.values()));refroute=ProtectedRoute.from_packet(first['packet'])
            base=receiver_signals(model,router,x,seen,first['task'],first['class_id'],batch_size,device)
            baseline=base['local_pred'].copy()
        pred=baseline.copy();activated=np.zeros(len(x),bool);details={}
        for cid,e in sorted(self.entries.items(),key=lambda kv:(kv[1]['task'],kv[1]['installed_sequence'],kv[0])):
            certified=self.certificate_current(model,router,cid)
            if not candidate and not certified:
                e.update(valid=False,reason='stable_guard_head_function_or_support_changed');continue
            if (not candidate and 'lifecycle_certificate' in e and
                    not route_authorized(e,runtime_scope)):
                details[cid]=dict(certified=certified,activated=np.zeros(len(x),bool),
                    lifecycle_state=e.get('lifecycle_state'),route_suspended=True)
                continue
            route=ProtectedRoute.from_packet(e['packet'])
            live=head_snapshot(model,cid)
            route.head_weight=(live['weight']*live['weight_mask']).numpy()
            route.head_bias=float(live['bias']*live['bias_mask'])
            with original_local_head(model,cid,e['backup']):
                sig=stable_signals(model,router,x,seen,route,e['reference_classes'],batch_size,device)
            sig['local_pred']=baseline
            shield=CurrentBaseSketchShield.restore(e['shield_at_install'])
            veto=shield.veto(x,e['required_old_classes'])
            hit=(sig['signature_valid'] & (sig['signature_score']>route.metadata['tau']) &
                (sig['margin']>route.metadata['gamma']) & ~veto & ~activated)
            pred[hit]=cid;activated|=hit
            details[cid]=dict(certified=certified,activated=hit,signature_score=sig['signature_score'],
                margin=sig['margin'],support_veto=veto,normal_router_confidence=sig['local_confidence'])
        self.sync(model)
        return dict(pred=pred,activated=activated,local_pred=baseline,details=details,
            certified=all(self.certificate_current(model,router,c) for c in self.entries),
            lifecycle_routes_authorized={c:('lifecycle_certificate' not in e or route_authorized(e,runtime_scope))
                for c,e in self.entries.items()},
            policy=POLICY,donor_inference=False,historical_raw_reads=0)

    def recertify(self,*args,**kwargs):
        raise Rejected('HISTORICAL_RAW_RECERTIFICATION_FORBIDDEN_USE_CURRENT_AGGREGATES')

    def current_negative_gate(self,model,router,cid,current_view,seen,device,batch_size,budget=.001,runtime_scope=None):
        e=self.entries[cid]
        if (not isinstance(current_view,CurrentCalibrationData) or current_view.client_id!=e['receiver'] or
                current_view.store['role_manifest_sha256']!=e['shield_at_install']['role_manifest_sha256'] or
                current_view.store['metadata_sha256']!=e['shield_at_install']['preprocessing_sha256'] or
                current_view.task<max(e['task'],e.get('last_current_negative_task',e['task']))):
            raise Rejected('CURRENT_NEGATIVE_SCOPE_CHANGED')
        pool=current_view.current_pool(e['receiver'],'calibration',current_view.store['task_classes'][str(current_view.task)])
        held=stratified_roles(pool,ROUTE_RULES['seed']+e['receiver'])['holdout']
        x,y=pool['X'][held],pool['y'][held]
        if np.any(y==cid):raise Rejected('CURRENT_NEGATIVE_TARGET_NOT_MISSING')
        function_current=self.certificate_current(model,router,cid)
        # Counterfactual monitor of the fixed guard, not activation of a
        # suspended route and never permission to expand certificate scope.
        record=self.records(model,router,x,seen,device,batch_size,candidate=True) if len(y) and function_current else None
        hit=(record['details'].get(cid,{}).get('activated',np.zeros(len(y),bool)) if record else np.zeros(0,bool))
        fars={str(int(c)):float(hit[y==c].mean()) for c in np.unique(y)} if record is not None else {}
        broken=int((hit&(record['local_pred']==y)).sum()) if record else 0
        conflict=bool(record is not None and (float(hit.mean())>budget or any(f>budget for f in fars.values()) or broken>0))
        passed=(len(y)>=32 and function_current and not conflict)
        report=dict(passed=bool(passed),task=current_view.task,client=e['receiver'],role='current CAL HOLDOUT negative monitor; no threshold tuning',
            rows=len(y),far=float(hit.mean()) if record is not None else None,far_by_class=fars,
            break_count=broken if record is not None else None,
            role_manifest_sha256=pool['role_manifest_sha256'],partition_sha256=pool['partition_sha256'],
            no_historical_raw_reads=True,threshold_retuning=False,target_positive_retested=False,
            positive_function_certificate_unchanged=function_current,
            monitor_evaluated=record is not None,new_CAL_certificate_issued=False,
            insufficient_CAL_is_certificate_corruption=False)
        if conflict:e['new_conflict_latched']=True
        if 'lifecycle_certificate' in e:
            report['lifecycle']=self.update_lifecycle(model,router,cid,runtime_scope,current_view.task,
                conflict=conflict,new_cal_rows=len(y))
        else:
            e.update(lifecycle_state='SUSPENDED',lifecycle_reason='certificate_scope_not_bound')
            report['lifecycle']=dict(state='SUSPENDED',reason='certificate_scope_not_bound')
        e['last_current_negative_task']=current_view.task
        e.setdefault('current_negative_history',[]).append(report);self.sync(model)
        return report

def install_stable_head(model,router,packet,receiver,seen,shield,required,acceptance,registry=None,certificate_scope=None):
    registry=registry or StableHeadRegistry();before=complete_hash(model,router);book=digest(registry.entries);rng=rng_snapshot()
    patch_id=hashlib.sha256(packet).hexdigest();cid=None
    try:
        route=ProtectedRoute.from_packet(packet);cid=int(route.metadata['class_id'])
        if cid in registry.entries:raise Rejected('STABLE_PATCH_ALREADY_INSTALLED_OR_CONFLICT')
        if (not isinstance(shield,CurrentBaseSketchShield) or shield.owner!=receiver or
                shield.pp_sha!=route.metadata['signature']['preprocessing_sha256'] or
                cid in required or shield.coverage(required)['missing_classes']):
            raise Rejected('STABLE_OWNED_OLD_SUPPORT_REQUIRED')
        compiled=compile_guarded_head(model,router,packet,receiver,seen,shield.pp_sha)
        refs=references(router,route.metadata['task'],cid,seen)
        function=guard_function(model,route,refs)
        declaration=guard_declaration(packet,function,shield,required)
        if (not acceptance.get('passed_current_scope') or acceptance.get('guard_declaration_sha256')!=declaration_sha(declaration) or
                acceptance.get('receiver_rows',0)<32 or acceptance.get('positive_rows',0)<32 or
                acceptance.get('recall',0)<.95 or acceptance.get('receiver_far',1)>.001 or
                any(v>.001 for v in acceptance.get('receiver_far_by_class',{}).values()) or
                acceptance.get('break_count',1)!=0 or acceptance.get('rescue',0)<1 or
                acceptance.get('source_role')!='current client-local CAL HOLDOUT aggregates'):
            raise Rejected('STABLE_CURRENT_AGGREGATE_ACCEPTANCE_REQUIRED')
        candidate=copy.deepcopy(model);detector=copy.deepcopy(router);pending=copy.deepcopy(registry)
        backup=head_snapshot(candidate,cid)
        put_head(candidate,cid,dict(weight=torch.from_numpy(route.head_weight),bias=torch.tensor(route.head_bias),
            weight_mask=torch.ones_like(backup['weight_mask']),bias_mask=torch.ones_like(backup['bias_mask']),rank=2))
        pending.entries[cid]=dict(class_id=cid,task=int(route.metadata['task']),receiver=receiver,donor=int(route.metadata['donor']),
            patch_id=patch_id,packet=bytes(packet),backup=backup,installed_head=head_snapshot(candidate,cid),
            installed_sequence=len(pending.entries),prior_patch_ids=[(p,v['patch_id']) for p,v in pending.entries.items()],
            patch_version='stable_guarded_head_install_v1',route_version=POLICY,
            installed=True,protected=True,valid=True,reason=None,activation_policy=POLICY,
            reference_classes=refs,required_old_classes=list(required),shield_at_install=copy.deepcopy(shield.state()),
            guard_function_fingerprint=function['fingerprint'],guard_declaration=declaration,
            current_acceptance=copy.deepcopy(acceptance),feature_version=function['fingerprint'],route_state_version=None)
        pending.sync(candidate)
        if not pending.certificate_current(candidate,detector,cid):raise Rejected('STABLE_STAGING_CERTIFICATE_FAILED')
        if certificate_scope is not None:
            pending.bind_lifecycle(candidate,detector,cid,certificate_scope)
            pending.attest_signed_zero_equivalence(candidate,detector,cid)
        if before!=complete_hash(model,router) or book!=digest(registry.entries):raise RuntimeError('Stable staging mutated live source')
        return candidate,detector,pending,dict(status='committed',applied=True,patch_id=patch_id,
            activation_policy=POLICY,current_acceptance=acceptance,old_protection_scope='finite enclosed own BASE support',
            lifecycle_state=pending.entries[cid].get('lifecycle_state'),
            numeric_equivalence_attestation_sha256=pending.entries[cid].get('numeric_equivalence_attestation_sha256'),
            certificate_scope=copy.deepcopy(certificate_scope),
            population_FAR_certified=False,historical_raw_recertification=False)
    except Rejected as exc:
        if before!=complete_hash(model,router) or book!=digest(registry.entries):raise RuntimeError('Stable rollback invariant failed') from exc
        return model,router,registry,dict(status='rejected',applied=False,reason=exc.reason,detail=exc.detail,rollback_verified=True,patch_id=patch_id)
    finally:restore_rng(rng)
