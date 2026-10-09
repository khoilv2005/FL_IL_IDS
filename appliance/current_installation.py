"""Build and stage a capability from live peers and own current CAL.

No prepared patch input, historical CAL, validation or test data. Pair discovery
must be locked before this transaction. Current acceptance certifies only the
explicit local domain/classes; it grants no cumulative-domain permission.
"""
import copy,hashlib,io,uuid
from pathlib import Path
import numpy as np
import torch
from .config import Protocol,Rejected
from .active_pair_calibration import split_counts
from .patch_lifecycle import checked_scope
from .current_calibration_data import CurrentCalibrationData
from .base_sketch_shield import CurrentBaseSketchShield
from .closure import effective_linear
from .distributed_calibration import CalibrationSession,GuardGrid,select_distributed_guard,current_acceptance
from .imported_route import ROUTE_RULES,stratified_roles
from .portable_route import (SharedSketch,ProtectedRoute,PORTABLE_RULES,prototype_summary,
                             fit_support_cosine_floor)
from .stable_head import (StableCalibrationEndpoint,StableHeadRegistry,guard_function,references,
                          guard_declaration,declaration_sha,stable_signals,install_stable_head)
from .state import complete_hash,boundary_hash,write_json,rng_snapshot,restore_rng
from .transport import Transport


def install_current_pair(pair,model,router,donor_model,donor_router,config,
                         receiver_view,donor_view,shield,groups,alphas,seen,scope,out,batch_size=512,
                         session_nonce=None):
    """A single locked request; return live originals on rejection.

    This is the staging service, not an offer scheduler. Caller commits the
    returned model/router together and registers protection hooks. No failed
    pair is replaced here. All endpoint labels stay with the local endpoint;
    only the capsule, capability and aggregate counts cross the transport API.
    """
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    cid,donor,c,task,round_id=(int(pair[k]) for k in ('receiver','donor','class_id','task','round'))
    original=complete_hash(model,router);rng=rng_snapshot()
    existing=StableHeadRegistry();existing.entries=copy.deepcopy(getattr(model,'appliance_guarded_head_entries',{}))
    write_json(out/'pair_lock.json',dict(pair=pair,scope=scope,selection_opened=False,holdout_opened=False,
        historical_CAL_reads=0,prepared_packet_used=False,failed_pair_substitution_allowed=False))
    try:
        if config.get('denice_cl_method')!='legacy':raise Rejected('CURRENT_INSTALL_LEGACY_REQUIRED')
        if existing.entries:raise Rejected('CURRENT_MULTI_CAPABILITY_ACCEPTANCE_NOT_YET_VERIFIED')
        if (pair.get('selection_scope')!='current_local_fit' or donor==cid or
            donor not in groups.get(cid,[]) or not np.isfinite(alphas.get(cid,{}).get(donor,0)) or
            alphas.get(cid,{}).get(donor,0)<=0):raise Rejected('CURRENT_INSTALL_PAIR_OR_GRAPH_CHANGED')
        if (not isinstance(receiver_view,CurrentCalibrationData) or not isinstance(donor_view,CurrentCalibrationData)
            or receiver_view.client_id!=cid or donor_view.client_id!=donor or
            receiver_view.task!=task or donor_view.task!=task):raise Rejected('CURRENT_INSTALL_CAL_AUTHORITY_CHANGED')
        classes=receiver_view.store['task_classes'][str(task)]
        role_sha=receiver_view.store['role_manifest_sha256'];pp_sha=receiver_view.store['metadata_sha256']
        if (donor_view.store['role_manifest_sha256']!=role_sha or donor_view.store['metadata_sha256']!=pp_sha
            or donor_view.store['task_classes'][str(task)]!=classes):raise Rejected('CURRENT_INSTALL_DATA_PROTOCOL_CHANGED')
        scope=checked_scope(scope)
        own_counts=receiver_view.store['clients'][str(cid)][str(task)]['class_counts']
        reserved={c}|{int(k) for k,n in own_counts.items() if split_counts(int(n))[2]>0}
        if c not in scope['classes'] or not set(scope['classes']).issubset(reserved):
            raise Rejected('CURRENT_INSTALL_SCOPE_LACKS_RESERVED_CAL')
        if pair.get('donor_model_fingerprint')!=complete_hash(donor_model,donor_router):
            raise Rejected('CURRENT_INSTALL_DONOR_SNAPSHOT_CHANGED')
        if (not isinstance(shield,CurrentBaseSketchShield) or shield.owner!=cid or
            shield.role_sha!=role_sha or shield.pp_sha!=pp_sha):raise Rejected('CURRENT_INSTALL_OWNED_SHIELD_REQUIRED')
        # Prior-task knowledge uses owned BASE summaries. Current-task damage
        # is checked on the reserved current CAL negatives; neither role is
        # relabelled or silently used as the other's acceptance evidence.
        required=sorted(int(k) for k in seen if k not in classes and int(model.unit_ranks['fc2'][k])>=2)
        if shield.coverage(required)['missing_classes']:raise Rejected('CURRENT_INSTALL_OLD_BASE_SUPPORT_MISSING')
        if (int(model.unit_ranks['fc2'][c])!=0 or int(donor_model.unit_ranks['fc2'][c])<2 or
            int(donor_view.manifest['clients'][str(donor)]['role_class_counts']['base'].get(str(c),0))<=0):
            raise Rejected('CURRENT_INSTALL_OUTPUT_OR_DONOR_PROVENANCE_CHANGED')
        # The existing receiver function is evaluated by the donor locally.
        # Account setup traffic; do not call the KiB packet total transfer cost.
        from fed_learning.training.checkpoint_state import snapshot_denice_state
        from eval_checkpoint import _make_denice_client_model
        algorithm=snapshot_denice_state(model,router)
        if algorithm['context_detector'].get('reference_input_memory'):raise Rejected('RAW_REFERENCE_MEMORY_IN_CAPSULE')
        capsule=dict(config=copy.deepcopy(config),task=task,round=round_id,
            client_model_states={cid:{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}},
            client_algorithm_states={cid:{'denice':algorithm}})
        transport=Transport(out/'application_wire.jsonl',[(cid,donor),(donor,cid)],
            Protocol(max_incoming_bytes=32*1024*1024,max_outgoing_bytes=32*1024*1024).validate())
        buf=io.BytesIO();torch.save(capsule,buf)
        payload=transport.send(cid,donor,'current_receiver_function_capsule',buf.getvalue())
        device=str(next(model.parameters()).device)
        replica,replica_router=_make_denice_client_model(torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False),cid,device)
        if complete_hash(replica,replica_router)!=original:raise Rejected('CURRENT_INSTALL_INEXACT_CAPSULE')
        pools={i:v.current_pool(i,'calibration',classes) for i,v in ((cid,receiver_view),(donor,donor_view))}
        split={i:stratified_roles(p,ROUTE_RULES['seed']+i) for i,p in pools.items()}
        fit=split[donor]['fit'];positive=fit[pools[donor]['y'][fit]==c]
        if len(positive)<32:raise Rejected('CURRENT_INSTALL_DONOR_FIT_SUPPORT_MISSING')
        sketch=SharedSketch(tuple(config['input_shape']),16,pp_sha)
        z,valid=sketch.features(pools[donor]['X'][positive]);proto,var,support=prototype_summary(z,valid)
        w,b=effective_linear(donor_model,'fc2');floor=fit_support_cosine_floor(support)
        route=ProtectedRoute(dict(kind='protected_imported_route',version=PORTABLE_RULES['version'],
            receiver=cid,donor=donor,class_id=c,task=task,tau=1.,gamma=0.,beta=1.,
            receiver_feature_hash=boundary_hash(model,True),signature=sketch.manifest(),support=support,
            role_manifest_sha256=role_sha,margin_reference_scope='import_context',
            guard_calibration_version='current-stable-client-local-v1',
            fit_support_min_cosine=floor,fit_support_policy='donor-FIT p95 radius; fixed before SELECTION',
            self_confidence=PORTABLE_RULES['self_confidence']),proto,var,w[c].numpy(),float(b[c]))
        refs=references(router,task,c,seen)
        private={}
        def make_views(owner,include_holdout):
            local_model,local_router=(model,router) if owner==cid else (replica,replica_router)
            local={}
            for role in ('fit','selection','holdout'):
                indices=split[owner][role] if include_holdout or role!='holdout' else np.zeros(0,np.int64)
                x=pools[owner]['X'][indices]
                if len(x):sig=stable_signals(local_model,local_router,x,seen,route,refs,batch_size,device)
                else:sig=dict(signature_score=np.zeros(0),signature_valid=np.zeros(0,bool),margin=np.zeros(0),
                              local_confidence=np.zeros(0),local_pred=np.zeros(0,np.int64))
                sig['base_support_veto']=shield.veto(x,required) if len(x) else np.zeros(0,bool)
                local[role]=dict(y=pools[owner]['y'][indices],row_id=pools[owner]['rows'][indices],signals=sig)
            return local
        session_index=0
        def session_for(packet):
            nonlocal session_index
            session_index+=1
            fn=guard_function(model,route,refs);decl=guard_declaration(packet,fn,shield,required)
            session=CalibrationSession(cid,donor,c,task,round_id,tuple(classes),original,
                                       declaration_sha(decl),role_sha,
                                       f'{session_nonce}:{session_index}' if session_nonce else uuid.uuid4().hex)
            return session,decl
        packet=route.packet();session,decl=session_for(packet)
        private={i:make_views(i,False) for i in (cid,donor)}
        endpoints={i:StableCalibrationEndpoint(i,session,private[i],decl) for i in private}
        summaries=[endpoints[cid].quantiles(),endpoints[donor].quantiles()]
        transport.send(donor,cid,'current_selection_quantiles',summaries[1])
        proposed=GuardGrid.from_quantiles(session,summaries,floor)
        grid=GuardGrid(session,proposed.taus,proposed.gammas,[1.])
        rp=endpoints[cid].count_packet(grid);dp=transport.send(donor,cid,'current_selection_counts',endpoints[donor].count_packet(grid))
        selected=select_distributed_guard(session,grid,rp,dp)
        transport.send(cid,donor,'frozen_current_selection_guard',selected)
        route.metadata.update(tau=selected['tau'],gamma=selected['gamma'],beta=1.)
        packet=transport.send(donor,cid,'frozen_current_capability',route.packet())
        # New declaration/session binds the final packet before HOLDOUT. The
        # initial selection session cannot certify a changed packet hash.
        session,decl=session_for(packet);final=GuardGrid(session,[selected['tau']],[selected['gamma']],[1.])
        write_json(out/'guard_lock_before_CURRENT_HOLDOUT.json',dict(session=session.manifest(),declaration=decl,
            selected=selected,scope=scope,holdout_scored=False,thresholds_frozen=True))
        endpoints={i:StableCalibrationEndpoint(i,session,make_views(i,True),decl) for i in (cid,donor)}
        for ep in endpoints.values():ep.lock_guard(final)
        rp=endpoints[cid].count_packet(final,'holdout')
        dp=transport.send(donor,cid,'current_holdout_counts',endpoints[donor].count_packet(final,'holdout'))
        (out/'receiver_current_counts.bin').write_bytes(rp);(out/'donor_current_counts.bin').write_bytes(dp)
        accepted=current_acceptance(session,final,rp,dp)
        accepted.update(guard_declaration_sha256=declaration_sha(decl),source_role='current client-local CAL HOLDOUT aggregates',
            role_manifest_sha256=role_sha,receiver_packet_sha256=hashlib.sha256(rp).hexdigest(),
            donor_packet_sha256=hashlib.sha256(dp).hexdigest())
        candidate,detector,registry,transaction=install_stable_head(model,router,packet,cid,seen,shield,required,
            accepted,registry=existing,certificate_scope=scope)
        if complete_hash(model,router)!=original:raise RuntimeError('Current transaction mutated live receiver')
        report=dict(completed_execution=True,pair=pair,transaction=transaction,acceptance=accepted,
            required_old_classes=required,capability_bytes=len(packet),setup_capsule_bytes=len(payload),
            historical_CAL_reads=0,prepared_packet_used=False,validation_opened=False,final_test_opened=False,
            source_receiver_unchanged=True,current_CAL_access={i:v.access_log for i,v in ((cid,receiver_view),(donor,donor_view))},
            trusted_transport_simulation=True,main_install_authorized=False)
        write_json(out/'transaction.json',report)
        return candidate,detector,registry,report
    except Rejected as exc:
        if complete_hash(model,router)!=original:raise RuntimeError('Rejected current install mutated live receiver') from exc
        report=dict(completed_execution=True,pair=pair,transaction=dict(applied=False,status='rejected',reason=exc.reason,
            detail=exc.detail,rollback_verified=True),prepared_packet_used=False,historical_CAL_reads=0,
            main_install_authorized=False)
        write_json(out/'transaction.json',report)
        return model,router,existing,report
    finally:restore_rng(rng)
