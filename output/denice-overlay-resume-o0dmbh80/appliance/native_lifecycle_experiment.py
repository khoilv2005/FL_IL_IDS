"""Prepare an exact task-boundary continuation with receiver-bound patches."""
import copy
import hashlib
import io
import json
import math
import shutil
import tempfile
from pathlib import Path
import zipfile

import joblib
import numpy as np
import torch

from .closure import effective_linear
from .config import Rejected
from .guarded_head import GuardedHeadRegistry, compile_guarded_head, install_guarded_head
from .guarded_head_experiment import combine, locked_pool
from .imported_route import stratified_roles,ROUTE_RULES
from .portable_route import ProtectedRoute,SharedSketch,prototype_summary,choose_protection,receiver_signals
from .selector import lookup,recorded_graph,current_pool
from .state import boundary_hash,digest,write_json


def load_continuation(checkpoint,expected_terminal_sha256,task=4):
    from fed_learning.data.denice_clean_roles import file_sha256
    path=Path(checkpoint)
    if path.is_dir():
        manifest=json.loads((path/'checkpoint_archive_manifest.json').read_text(encoding='utf-8'))
        def read(name):
            target=(path/name).resolve();target.relative_to(path.resolve())
            if file_sha256(target)!=manifest['checksums'][name]:raise Rejected('NATIVE_CONTINUATION_CHECKSUM_MISMATCH')
            return torch.load(target,map_location='cpu',weights_only=False)
    else:
        with zipfile.ZipFile(path) as archive:manifest=json.loads(archive.read('checkpoint_archive_manifest.json'))
        def read(name):
            with zipfile.ZipFile(path) as archive:data=archive.read(name)
            if hashlib.sha256(data).hexdigest()!=manifest['checksums'][name]:raise Rejected('NATIVE_CONTINUATION_CHECKSUM_MISMATCH')
            return torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)
    terminal=manifest.get('full_terminal_checkpoint');member=manifest.get('continuation_checkpoint')
    if int(manifest.get('task_id',-1))!=int(task) or manifest['checksums'].get(terminal)!=expected_terminal_sha256:
        raise Rejected('NATIVE_TASK_CHECKPOINT_MISMATCH')
    if not member:raise Rejected('NATIVE_FULL_CONTINUATION_REQUIRED')
    state=read(member)
    if (state.get('continuation_type')!='denice_decentralized_continuation'
            or int(state.get('denice_continuation_schema_version',-1))!=1
            or state['meta'].get('completed_task')!=task or state['meta'].get('resume_from_task')!=task+1):
        raise Rejected('NATIVE_TASK_BOUNDARY_SCOPE_MISMATCH')
    return state,manifest['checksums'][member]


def select_active_pair(ckpt,role_manifest):
    """Freeze one structural candidate; never search again using validation."""
    groups,alphas=recorded_graph(ckpt,4,19)
    c=24;excluded=[]
    for receiver in sorted(groups):
        counts=role_manifest['clients'][str(receiver)]['role_class_counts']
        future=sum(int(counts['base'].get(str(label),0)) for label in (30,31,32,33))
        calibration=[int(counts['calibration'].get(str(label),0)) for label in range(24,30)]
        holdout=0
        for n in calibration:
            if n>=3:
                fit=min(n-2,max(1,n//2));selection=min(n-fit-1,max(1,n//4))
                holdout+=n-fit-selection
        alg=lookup(ckpt['client_algorithm_states'],receiver,{})
        denice=alg.get('denice',alg)
        reasons=[]
        if int(counts['base'].get(str(c),0))!=0:reasons.append('class24 locally supported')
        if future<=0:reasons.append('no Task5 BASE')
        if holdout<32:reasons.append('fewer than 32 receiver calibration HOLDOUT rows')
        if int(denice.get('neuron_ages',{}).get('fc2',np.ones(34))[c])!=0:reasons.append('occupied output slot')
        if denice.get('context_detector',{}).get('router_last_refresh_task')!=4:reasons.append('no local Task4 router evidence')
        donors=[]
        for donor in groups[receiver]:
            donor_state=lookup(ckpt['client_algorithm_states'],donor,{})
            donor_state=donor_state.get('denice',donor_state)
            donor_counts=role_manifest['clients'][str(donor)]['role_class_counts']
            n=int(donor_counts['calibration'].get(str(c),0))
            fit_count=min(n-2,max(1,n//2)) if n>=3 else 0
            selection_count=min(n-fit_count-1,max(1,n//4)) if n>=3 else 0
            positive_holdout=n-fit_count-selection_count if n>=3 else 0
            if (alphas[receiver].get(donor,0)>0 and int(donor_counts['base'].get(str(c),0))>0
                    and fit_count>=32 and positive_holdout>=32
                    and donor_state.get('context_detector',{}).get('router_last_refresh_task')==4
                    and int(donor_state.get('neuron_ages',{}).get('fc2',np.zeros(34))[c])>=2):
                donors.append(donor)
        if not donors:reasons.append('no positive Task4 neighbor with locally learned class24 and sufficient calibration FIT support')
        if reasons:
            excluded.append(dict(receiver=receiver,reasons=reasons));continue
        return dict(receiver=receiver,donor=min(donors),class_id=c,task=4,round=19,
            next_task=5,next_task_base_rows=future,receiver_holdout_rows=holdout,
            rule='class24 fixed; first eligible receiver ID then first eligible donor ID: positive Task4 edge, local provenance, calibration support, free receiver head and Task5 BASE scheduling; no prediction/validation/test selection',
            excluded_before_selection=excluded)
    return dict(selected=False,reason='No structurally eligible Task5-active receiver/donor for fixed class24',excluded=excluded)


def _splice(state,model,router,cid):
    from fed_learning.training.checkpoint_state import snapshot_denice_state
    state['client_model_states'][cid]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    state['client_algorithm_states'][cid]={'denice':snapshot_denice_state(model,router)}


def prepare_active_patch(ckpt,roles,source,pair,out,device,batch_size):
    """Fixed shared16 architecture; donor FIT signature and SELECTION guards."""
    from eval_checkpoint import _make_denice_client_model
    from fed_learning.training.checkpoint_state import snapshot_context_detector
    from fed_learning.data.denice_clean_roles import file_sha256
    out=Path(out);out.mkdir(parents=True,exist_ok=True);source=Path(source)
    cid,donor,c=map(int,(pair[k] for k in ('receiver','donor','class_id')))
    model,router=_make_denice_client_model(ckpt,cid,device,router_mode='multiclass_balanced')
    donor_model,donor_router=_make_denice_client_model(ckpt,donor,device,router_mode='multiclass_balanced')
    donor_snap=snapshot_context_detector(donor_router)
    old_decision=json.loads((source/'decision_lock.json').read_text(encoding='utf-8'))
    packet=(source/'packets'/f'{old_decision["primary_shared"]}.bin').read_bytes()
    route=ProtectedRoute.from_packet(packet)
    if route.metadata['signature']['kind']!='shared_sketch' or route.metadata['signature']['dimension']!=16:
        raise Rejected('LOCKED_SHARED16_REFERENCE_REQUIRED')
    if int(route.metadata['class_id'])!=c:
        raise Rejected('FROZEN_DONOR_CAPABILITY_SCOPE_MISMATCH')
    groups,alphas=recorded_graph(ckpt,4,19)
    if donor not in groups[cid] or alphas[cid].get(donor,0)<=0:raise Rejected('ACTIVE_PAIR_GRAPH_CHANGED')
    # Assess the one preselected donor on FIT only. HOLDOUT remains reserved
    # for acceptance, and failed pairs are not replaced.
    from .evaluate import Predictor
    from .ledger import quality
    from .config import Protocol
    classes=list(range(24,30));seen=list(range(30))
    pools={i:current_pool(roles,i,'calibration',classes) for i in (cid,donor)}
    split={i:stratified_roles(pools[i],ROUTE_RULES['seed']+i) for i in pools}
    donor_fit=split[donor]['fit'];fit_y=pools[donor]['y'][donor_fit]
    donor_pred=Predictor(seen,device,batch_size)(donor_model,donor_router,pools[donor]['X'][donor_fit])
    evidence=dict(positive=int((fit_y==c).sum()),predicted=int((donor_pred==c).sum()),
        correct_positive=int(((donor_pred==c)&(fit_y==c)).sum()))
    lcb=quality(evidence,Protocol().validate())
    weight,bias=effective_linear(donor_model,'fc2')
    fit_x=pools[donor]['X'][donor_fit[fit_y==c]]
    signature=route.metadata['signature']
    sketch=SharedSketch(tuple(signature['input_shape']),16,signature['preprocessing_sha256'],signature['seed'])
    z,valid=sketch.features(fit_x)
    route.prototype,route.variance,support=prototype_summary(z,valid)
    route.head_weight,route.head_bias=weight[c].numpy(),float(bias[c])
    del donor_model,donor_router
    manifest={str(i):{role:dict(rows=pools[i]['rows'][indices].tolist()) for role,indices in entries.items()}
        for i,entries in split.items()}
    write_json(out/'calibration_split_manifest.json',manifest)
    route.metadata.update(receiver=cid,donor=donor,support=support,receiver_feature_hash=boundary_hash(model,True),
        split_manifest_sha256=file_sha256(out/'calibration_split_manifest.json'),
        original_reference_packet_sha256=hashlib.sha256(packet).hexdigest())
    signals={i:receiver_signals(model,router,pools[i]['X'][split[i]['selection']],seen,4,c,batch_size,device) for i in pools}
    scored={i:route.signals(signals[i],pools[i]['X'][split[i]['selection']]) for i in pools}
    selected,_=choose_protection(scored[cid],pools[cid]['y'][split[cid]['selection']],
        scored[donor],pools[donor]['y'][split[donor]['selection']],c)
    route.metadata.update(tau=selected['tau'],gamma=selected['gamma'],beta=selected['beta'])
    packet=route.packet();(out/'packets').mkdir(exist_ok=True);(out/'packets/shared16.bin').write_bytes(packet)
    protocol=json.loads((source/'protocol_lock.json').read_text(encoding='utf-8'))
    protocol.update(receiver=cid,donor=donor,prototype_fit='preselected donor Task4 calibration FIT positives; architecture/seed fixed shared16',
        threshold_selection='receiver-specific guards from current Task4 calibration SELECTION only',
        future_data_access='manifest BASE counts only to require an active Task5 schedule; no future rows read')
    write_json(out/'protocol_lock.json',protocol)
    write_json(out/'decision_lock.json',dict(primary_shared='shared16',selection=selected,
        packet_sha256={'shared16':hashlib.sha256(packet).hexdigest()},protocol_sha256=file_sha256(out/'protocol_lock.json'),
        primary_selected_using='fixed shared16 architecture; SELECTION guard thresholds only',validation_loaded=False))
    joblib.dump({cid:snapshot_context_detector(router),donor:donor_snap},out/'fitted_current_task_routers.joblib',compress=3)
    acceptance=combine([locked_pool(roles,i,classes,manifest[str(i)]['holdout']['rows']) for i in pools])
    compiled=compile_guarded_head(model,router,packet,cid,seen,protocol['preprocessing_sha256'])
    installed,detector,registry,transaction=install_guarded_head(model,router,compiled,GuardedHeadRegistry(),
        acceptance,seen,device,batch_size)
    write_json(out/'transaction.json',transaction)
    write_json(out/'pair_lock.json',dict(**pair,donor_quality_lcb=lcb,donor_calibration=evidence,
        no_validation_selection=True,thresholds_frozen_before_holdout=True))
    write_json(out/'completion.json',dict(completed_execution=True,installed=bool(transaction['applied']),
        final_test_opened=False,no_validation_loaded=True,passed_feasibility=False))
    if not transaction['applied']:return None,None
    registry.sync(installed)
    return installed,detector


def install_frozen_active_patch(ckpt,roles,source,pair,out,device,batch_size):
    """Reinstall a checksum-locked active capability without refitting guards."""
    from eval_checkpoint import _make_denice_client_model
    from fed_learning.training.checkpoint_state import restore_context_detector
    from fed_learning.data.denice_clean_roles import file_sha256
    source,out=Path(source),Path(out);out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((source/'protocol_lock.json').read_text(encoding='utf-8'))
    locked_pair=json.loads((source/'pair_lock.json').read_text(encoding='utf-8'))
    for field in ('receiver','donor','class_id','task','round','next_task'):
        if pair[field]!=locked_pair[field]:raise Rejected('NATIVE_FROZEN_ACTIVE_PAIR_MISMATCH',field)
    if file_sha256(roles.root/'role_manifest.json')!=protocol['role_manifest_sha256']:
        raise Rejected('NATIVE_FROZEN_ACTIVE_ROLES_CHANGED')
    for name in ('protocol_lock.json','decision_lock.json','pair_lock.json','calibration_split_manifest.json',
            'fitted_current_task_routers.joblib','packets/shared16.bin'):
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,target)
    decision=json.loads((source/'decision_lock.json').read_text(encoding='utf-8'))
    packet=(source/'packets/shared16.bin').read_bytes()
    if hashlib.sha256(packet).hexdigest()!=decision['packet_sha256']['shared16']:
        raise Rejected('NATIVE_FROZEN_ACTIVE_PACKET_CHANGED')
    cid=int(pair['receiver']);donor=int(pair['donor']);c=int(pair['class_id'])
    model,router=_make_denice_client_model(ckpt,cid,device)
    frozen=lookup(joblib.load(source/'fitted_current_task_routers.joblib'),cid)
    if not frozen:raise Rejected('NATIVE_FROZEN_ACTIVE_ROUTER_MISSING')
    restore_context_detector(router,frozen)
    classes=list(range(24,30));seen=list(range(30))
    manifest=json.loads((source/'calibration_split_manifest.json').read_text(encoding='utf-8'))
    acceptance=combine([locked_pool(roles,i,classes,manifest[str(i)]['holdout']['rows']) for i in (cid,donor)])
    compiled=compile_guarded_head(model,router,packet,cid,seen,protocol['preprocessing_sha256'])
    installed,detector,registry,transaction=install_guarded_head(model,router,compiled,GuardedHeadRegistry(),
        acceptance,seen,device,batch_size)
    write_json(out/'transaction.json',transaction)
    write_json(out/'completion.json',dict(completed_execution=True,installed=bool(transaction['applied']),
        frozen_capability_reused=True,thresholds_retuned=False,prototype_refitted=False,
        final_test_opened=False,passed_feasibility=False))
    if not transaction['applied']:return None,None
    registry.sync(installed)
    return installed,detector


def run_native_lifecycle(checkpoint,data_dir,runtime_dir,out,round_budget=3,device='cuda',audit_batch_size=512,
                         amp_initial_scale=None):
    from eval_checkpoint import _make_denice_client_model
    from fed_learning.data.denice_clean_roles import CleanRoleData,file_sha256
    from fed_learning.training.decentralized_denice_il import run_decentralized_denice_il
    from .runner import load_input
    runtime,out=Path(runtime_dir),Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new native probe output directory')
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'completion.json',dict(completed_execution=False,passed_feasibility=False,stage='prepare'))
    seed_path=None
    try:
        write_json(out/'runtime_manifest.json',json.loads((runtime/'runtime_manifest.json').read_text(encoding='utf-8')))
        source=runtime/'frozen_capability';roles_dir=runtime/'roles'
        protocol=json.loads((source/'protocol_lock.json').read_text(encoding='utf-8'))
        state,continuation_sha=load_continuation(checkpoint,protocol['terminal_sha256'])
        roles=CleanRoleData(roles_dir,source_data_dir=data_dir)
        for cid in roles.manifest['clients']:
            if not (roles_dir/'indices'/f'client_{cid}.npz').is_file():raise Rejected('NATIVE_FULL_ROLES_REQUIRED',cid)
        if file_sha256(roles_dir/'role_manifest.json')!=protocol['role_manifest_sha256']:raise Rejected('NATIVE_ROLE_LOCK_MISMATCH')
        config=copy.deepcopy(state['config'])
        if config.get('denice_cl_method')!='legacy' or config['denice_similarity_threshold']!=.8:
            raise Rejected('NATIVE_LEGACY_XI8_REQUIRED')
        if not 1<=int(round_budget)<=int(config['rounds_per_task']):raise ValueError('Native round budget must be in [1,20]')
        if amp_initial_scale is not None:
            if isinstance(amp_initial_scale,bool) or not math.isfinite(float(amp_initial_scale)) or float(amp_initial_scale)<=0:
                raise ValueError('Native AMP initial scale must be positive and finite')
            config['denice_amp_initial_scale']=float(amp_initial_scale)
        seed=torch.load(runtime/'seed/guarded_receiver.pt',map_location='cpu',weights_only=False)
        cid=19
        # Seed is the already audited guarded receiver. Only the missing head
        # may differ from the official continuation's tensor state.
        baseline=lookup(state['client_model_states'],cid);patched=lookup(seed['client_model_states'],cid)
        if set(baseline)!=set(patched):raise Rejected('NATIVE_SEED_MODEL_KEYS_CHANGED')
        for name,value in baseline.items():
            changed=value!=patched[name]
            allowed=torch.zeros_like(changed)
            if name in ('fc2.weight','fc2.bias'):allowed[24]=True
            if bool((changed&~allowed).any()):raise Rejected('NATIVE_SEED_NONHEAD_CHANGE',name)
        model,router=_make_denice_client_model(seed,cid,device)
        reg=GuardedHeadRegistry();reg.entries=seed['guarded_head_entries'];reg.sync(model)
        if not reg.certificate_current(model,router,24):raise Rejected('NATIVE_SEED_CERTIFICATE_STALE')
        _splice(state,model,router,cid);del model,router,reg,seed
        portable_dirs={19:str(source)}
        ckpt,_=load_input(checkpoint,4,19)
        pair=json.loads((runtime/'active_pair_lock.json').read_text(encoding='utf-8'))
        expected=select_active_pair(ckpt,roles.manifest)
        if pair!=expected:raise Rejected('NATIVE_ACTIVE_PAIR_ORDERING_CHANGED')
        write_json(out/'active_pair_lock.json',pair)
        if pair.get('selected',True):
            candidate_dir=out/'active_pair_preparation'
            try:
                frozen=runtime/'active_capability'
                if frozen.is_dir():
                    active_model,active_router=install_frozen_active_patch(ckpt,roles,frozen,pair,candidate_dir,device,audit_batch_size)
                else:
                    active_model,active_router=prepare_active_patch(ckpt,roles,source,pair,candidate_dir,device,audit_batch_size)
            except Rejected as exc:
                candidate_dir.mkdir(parents=True,exist_ok=True)
                write_json(candidate_dir/'completion.json',dict(completed_execution=True,installed=False,
                    stage='rejected',reason=exc.reason,detail=exc.detail,pair=pair,no_replacement=True,
                    final_test_opened=False,passed_feasibility=False))
                active_model,active_router=None,None
            if active_model is not None:
                active_id=int(pair['receiver']);_splice(state,active_model,active_router,active_id)
                portable_dirs[active_id]=str(candidate_dir)
                del active_model,active_router
        del ckpt
        # Reporting/storage controls and the explicitly recorded AMP-scale
        # variant leave CANC/aggregation/batch/refresh/data sampling intact.
        config.update(data_dir=str(data_dir),denice_clean_roles_dir=str(roles_dir),
            denice_evaluation_data_role='validation',denice_post_task_eval=False,
            denice_eval_final_round=False,denice_eval_last_round_only=False,
            denice_eval_terminal_state_only=True,denice_eval_local_validation=False,
            eval_every=999999,denice_cme_after_each_task=False,denice_save_round_artifacts=False,
            round_checkpoint_every=None,save_continuation_every_task=False,
            denice_archive_checkpoints=False,output_dir=str(out/'native_training'),resume_output_dir=str(out/'native_training'))
        # Full federation seed is temporary input, not duplicated in output ZIP.
        handle=tempfile.NamedTemporaryFile(prefix='appliance-native-continuation-',suffix='.pt',delete=False)
        seed_path=Path(handle.name);handle.close();torch.save(state,seed_path)
        spec=dict(version='appliance_native_lifecycle_probe_v1',output_dir=str(out),roles_dir=str(roles_dir),
            portable_dirs=portable_dirs,round_budget=int(round_budget),audit_batch_size=int(audit_batch_size))
        spec_path=out/'native_probe_manifest.json';write_json(spec_path,spec)
        config.update(resume_state_path=str(seed_path),appliance_native_probe_manifest=str(spec_path))
        write_json(out/'protocol_lock.json',dict(method='legacy',xi=.8,source_terminal_sha256=protocol['terminal_sha256'],
            source_continuation_sha256=continuation_sha,role_manifest_sha256=protocol['role_manifest_sha256'],
            round_budget=round_budget,original_schedule=config['rounds_per_task'],training_batch_size=config['batch_size'],
            training_amp=config['denice_amp_enabled'],active_pair=pair,installed_receivers=sorted(portable_dirs),
            amp_initial_scale=config.get('denice_amp_initial_scale',65536.),
            numeric_policy=('amp_initial_scale_override' if amp_initial_scale is not None else 'original_amp_policy'),
            numeric_policy_scope='same initial AMP scale for every federation client; dynamic backoff/growth retained',
            frozen_active_capability_reused=(runtime/'active_capability').is_dir(),
            pair_selection='frozen before validation; one candidate, no replacement after acceptance failure',
            final_test_opened=False,validation_for_fitting=False,
            seed_scope='original full lifecycle + exact audited guarded receiver19; active receiver added only after calibration acceptance',
            limitation='retains Task4 calibration for offline recertification; not a replay-free calibration claim'))
        del state
        if device.startswith('cuda'):torch.cuda.empty_cache()
        result=run_decentralized_denice_il(config)
        return result
    except Rejected as exc:
        result=dict(completed_execution=True,stage='rejected',reason=exc.reason,detail=exc.detail,
            passed_feasibility=False,final_test_opened=False)
        write_json(out/'completion.json',result);return result
    except Exception as exc:
        write_json(out/'completion.json',dict(completed_execution=False,stage='execution_error',
            error_type=type(exc).__name__,detail=str(exc),passed_feasibility=False,final_test_opened=False))
        raise
    finally:
        if seed_path is not None:seed_path.unlink(missing_ok=True)
