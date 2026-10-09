"""Install and continue a locked portable head on the legacy Task4 pair.

Recorded-neighborhood continuation is a stress probe, not a replay of the
complete federation runner. Calibration alone controls commit/recertification.
"""
from collections import OrderedDict
import copy
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import random

import joblib
import numpy as np
import torch

from .config import Rejected
from .guarded_head import (GuardedHeadRegistry, HeadContract, compile_guarded_head,
    install_guarded_head, head_snapshot, put_head, router_hash)
from .parallel_route_experiment import metrics
from .portable_route import ProtectedRoute, transitions
from .runner import load_input
from .selector import current_pool, recorded_graph
from .state import complete_hash, boundary_hash, digest, write_json, model_function_state
from .survival import local_update


def locked_pool(roles,cid,classes,rows):
    pool=current_pool(roles,cid,'calibration',classes)
    mapping={int(row):pos for pos,row in enumerate(pool['rows'])}
    if len(set(rows))!=len(rows) or any(int(row) not in mapping for row in rows):
        raise Rejected('ACCEPTANCE_ROLE_OR_ROW_MISMATCH')
    indices=np.asarray([mapping[int(row)] for row in rows],dtype=np.int64)
    return dict(X=pool['X'][indices],y=pool['y'][indices],rows=pool['rows'][indices],
                origin_client=np.full(len(indices),cid,dtype=np.int64))


def combine(pools):
    return {key:np.concatenate([p[key] for p in pools]) for key in pools[0]}


def run_guarded_head(checkpoint,role_dir,data_dir,portable_dir,out,device='cpu',batch_size=512,
                     rounds=3,receiver_row_cap=0,neighbor_row_cap=0):
    from eval_checkpoint import _make_denice_client_model
    from fed_learning.data.denice_clean_roles import CleanRoleData,file_sha256
    from fed_learning.training.checkpoint_state import snapshot_denice_state,restore_context_detector
    from fed_learning.strategies.decentralized.denice_aggregation import age_aware_aggregate,AggregationConfig
    out,source=Path(out),Path(portable_dir)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'completion.json',dict(completed_execution=False,passed_feasibility=False))
    read=lambda name:json.loads((source/name).read_text(encoding='utf-8'))
    try:
        if rounds<1 or batch_size<1 or min(receiver_row_cap,neighbor_row_cap)<0:
            raise ValueError('Invalid continuation budget')
        protocol=read('protocol_lock.json');decisions=read('decision_lock.json')
        completion=read('completion.json');router_lock=read('router_lock.json')
        if not completion.get('routing_candidate_gate') or protocol['base_method']!='legacy':
            raise Rejected('QUALIFIED_LEGACY_ROUTING_ARTIFACT_REQUIRED')
        if file_sha256(source/'protocol_lock.json')!=decisions['protocol_sha256']:
            raise Rejected('SOURCE_PROTOCOL_LOCK_CHANGED')
        for name in ('holdout_predictions.npz','calibration_split_manifest.json','fitted_current_task_routers.joblib'):
            if not (source/name).is_file():raise Rejected('MISSING_LOCKED_ARTIFACT',name)
        if file_sha256(source/'fitted_current_task_routers.joblib')!=router_lock['fitted_artifact_sha256']:
            raise Rejected('FROZEN_ROUTER_CHANGED')
        primary=decisions['primary_shared'];packet=(source/'packets'/f'{primary}.bin').read_bytes()
        if hashlib.sha256(packet).hexdigest()!=decisions['packet_sha256'][primary]:
            raise Rejected('FROZEN_PACKET_CHANGED')
        task,round_id,cid,donor,c=map(int,(protocol[k] for k in ('task','round','receiver','donor','class_id')))
        ckpt,hashes=load_input(checkpoint,task,round_id)
        if any(hashes[k]!=protocol[k] for k in hashes):raise Rejected('CHECKPOINT_SOURCE_CHANGED')
        config=ckpt['config']
        if config.get('denice_cl_method','legacy')!='legacy':raise Rejected('LEGACY_BASELINE_REQUIRED')
        roles=CleanRoleData(role_dir,source_data_dir=data_dir)
        if file_sha256(roles.root/'role_manifest.json')!=protocol['role_manifest_sha256']:
            raise Rejected('ROLE_SOURCE_CHANGED')
        if config.get('denice_data_roles_sha256')!=protocol['role_manifest_sha256']:
            raise Rejected('BACKBONE_ROLE_LOCK_MISMATCH')
        metadata=json.loads((roles.source/'metadata.json').read_text(encoding='utf-8'))
        classes=list(map(int,metadata['task_structure']['task_classes'][str(task)]))
        seen=sorted({int(v) for t,values in metadata['task_structure']['task_classes'].items() if int(t)<=task for v in values})
        if seen!=sorted(map(int,ckpt['seen_classes'])) or c not in classes:raise Rejected('SCOPE_MISMATCH')
        groups,alphas=recorded_graph(ckpt,task,round_id)
        if donor not in groups[cid] or alphas[cid].get(donor,0)<=0:raise Rejected('DONOR_NOT_IN_RECORDED_GRAPH')
        random.seed(20261008);np.random.seed(20261008);torch.manual_seed(20261008)
        contract=HeadContract()
        source_files=list(Path(__file__).resolve().parent.glob('*.py'))
        root=Path(__file__).resolve().parents[1]
        source_files += [root/n for n in ('eval_checkpoint.py','fed_learning/clients/nice_client.py',
            'fed_learning/clients/denice_client.py','fed_learning/models/nice_model.py',
            'fed_learning/models/denice_model.py','fed_learning/strategies/incremental/nice.py',
            'fed_learning/strategies/decentralized/denice_aggregation.py','fed_learning/training/checkpoint_state.py',
            'fed_learning/training/denice_eval.py','fed_learning/data/denice_clean_roles.py')]
        import sklearn
        write_json(out/'protocol_lock.json',dict(version=contract.version,contract=asdict(contract),**hashes,
            role_manifest_sha256=protocol['role_manifest_sha256'],source_decision_sha256=file_sha256(source/'decision_lock.json'),
            source_router_sha256=router_lock['fitted_artifact_sha256'],packet_sha256=hashlib.sha256(packet).hexdigest(),
            task=task,round=round_id,receiver=cid,donor=donor,class_id=c,primary=primary,
            xi=config.get('denice_similarity_threshold'),method='legacy',rounds=rounds,
            receiver_row_cap=receiver_row_cap,neighbor_row_cap=neighbor_row_cap,device=device,batch_size=batch_size,
            no_threshold_retuning=True,acceptance_role='locked calibration HOLDOUT of receiver and donor; offline simulation',
            validation_role='previously observed development; metrics only',final_test_opened=False,
            head_protection='only imported FC2 row/mask/rank; zero corresponding optimizer slots',
            drift_policy='uncertified drift disables route; same locked calibration gates can recertify; otherwise refresh required',
            single_model_inference=True,donor_model_at_inference=False,
            full_federation_round=False,task_transition=False,
            limitations=['fixed recorded graph','no CANC/age merge/router refresh','historical old-task damage unmeasured',
                         'calibration feedback transport and total communication not implemented','not untouched confirmation'],
            versions=dict(torch=torch.__version__,numpy=np.__version__,sklearn=sklearn.__version__),
            source_sha256={p.relative_to(root).as_posix():file_sha256(p) for p in source_files}))
        receiver,router=_make_denice_client_model(ckpt,cid,device)
        checkpoint_weights=ckpt['client_model_states'].get(cid,ckpt['client_model_states'].get(str(cid)))
        restored_weights=receiver.state_dict()
        missing=sorted(set(checkpoint_weights)-set(restored_weights))
        unexpected=sorted(set(restored_weights)-set(checkpoint_weights))
        changed=sorted(n for n in set(checkpoint_weights)&set(restored_weights)
            if not torch.equal(checkpoint_weights[n].detach().cpu(),restored_weights[n].detach().cpu()))
        expected_boundary=ProtectedRoute.from_packet(packet).metadata['receiver_feature_hash']
        actual_boundary=boundary_hash(receiver,True)
        raw_boundary=digest(model_function_state(receiver,True,normalize_batchnorm=False))
        write_json(out/'boundary_preflight.json',dict(expected_packet_boundary=expected_boundary,
            actual_boundary=actual_boundary,raw_runtime_boundary=raw_boundary,
            normalized_batchnorm_display=actual_boundary!=raw_boundary,
            boundary_matches=actual_boundary==expected_boundary,
            restore_exact=not (missing or unexpected or changed),missing_weights=missing,
            unexpected_weights=unexpected,changed_weights=changed,torch_version=torch.__version__,
            fingerprint_policy='normalize known BatchNorm bias display; retain weights/buffers/masks/other module metadata'))
        if missing or unexpected or changed:raise Rejected('INEXACT_RECEIVER_MODEL_RESTORE')
        if actual_boundary!=expected_boundary:
            raise Rejected('PROTECTED_HEAD_BOUNDARY_MISMATCH','See boundary_preflight.json; no patch installed')
        frozen=joblib.load(source/'fitted_current_task_routers.joblib')
        snap=frozen.get(cid,frozen.get(str(cid)))
        if snap is None:raise Rejected('RECEIVER_ROUTER_MISSING')
        if any(int(t)>task for t in snap['activation_memory']):raise Rejected('FUTURE_ROUTER_MEMORY')
        restore_context_detector(router,copy.deepcopy(snap))
        row_manifest=read('calibration_split_manifest.json')
        if file_sha256(source/'calibration_split_manifest.json')!=ProtectedRoute.from_packet(packet).metadata['split_manifest_sha256']:
            raise Rejected('CALIBRATION_SPLIT_CHANGED')
        acceptance=combine([locked_pool(roles,i,classes,row_manifest[str(i)]['holdout']['rows']) for i in (cid,donor)])
        write_json(out/'acceptance_manifest.json',dict(role='calibration HOLDOUT',
            origins=acceptance['origin_client'],rows=acceptance['rows'],class_counts={int(k):int(v) for k,v in zip(*np.unique(acceptance['y'],return_counts=True))}))
        compiled=compile_guarded_head(receiver,router,packet,cid,seen,protocol['preprocessing_sha256'])
        registry=GuardedHeadRegistry()
        before_source=complete_hash(receiver,router)
        baseline_row=head_snapshot(receiver,c)
        # Negative-control transaction: valid dimensions, disabled activation.
        bad=ProtectedRoute.from_packet(packet);bad.metadata['beta']=0.
        rejected_compiled=compile_guarded_head(receiver,router,bad.packet(),cid,seen,protocol['preprocessing_sha256'])
        _,_,_,rollback=install_guarded_head(receiver,router,rejected_compiled,registry,acceptance,seen,device,batch_size,contract)
        if rollback['applied'] or not rollback.get('rollback_verified'):raise RuntimeError('Rollback control failed')
        model,detector,registry,txn=install_guarded_head(receiver,router,compiled,registry,acceptance,seen,device,batch_size,contract)
        write_json(out/'transaction.json',dict(rollback_control=rollback,install=txn))
        if not txn['applied']:raise Rejected('INSTALL_REJECTED',str(txn))
        if complete_hash(receiver,router)!=before_source:raise RuntimeError('Original source checkpoint model changed')
        _,_,_,replay=install_guarded_head(model,detector,compiled,registry,acceptance,seen,device,batch_size,contract)
        if replay['status']!='already_committed':raise RuntimeError('Idempotence failed')
        # Explicit overwrite stress is reported separately from natural training.
        stress=copy.deepcopy(model);stress_registry=copy.deepcopy(registry)
        optimizer=torch.optim.Adam(stress.parameters(),lr=.001)
        for parameter in (stress.fc2.weight,stress.fc2.bias):
            optimizer.state[parameter]['exp_avg']=torch.ones_like(parameter)
        with torch.no_grad():stress.fc2.weight[c].add_(.1);stress.fc2.bias[c].add_(.1)
        unprotected_changed=not stress_registry.head_matches(stress,c)
        stress_registry.protect(stress,optimizer)
        protection=unprotected_changed and stress_registry.head_matches(stress,c)
        zero_slots=all(not bool(optimizer.state[p]['exp_avg'][c].any()) for p in (stress.fc2.weight,stress.fc2.bias))
        if not protection or not zero_slots:raise RuntimeError('Row protection stress failed')
        del stress,stress_registry,optimizer
        validation_pools=[]
        for i in (cid,donor):
            pool=current_pool(roles,i,'validation',classes)
            validation_pools.append(dict(X=pool['X'],y=pool['y'],rows=pool['rows'],
                origin_client=np.full(len(pool['y']),i,dtype=np.int64)))
        validation=combine(validation_pools)
        stages={};preinstall=None

        def measure(name):
            nonlocal preinstall
            record=registry.records(model,detector,validation['X'],seen,device,batch_size)
            if preinstall is None:preinstall=record['local_pred'].copy()
            y=validation['y'];masks=dict(pooled=np.ones(len(y),bool),receiver=validation['origin_client']==cid,donor=validation['origin_client']==donor)
            result={scope:dict(**metrics(y[m],record['pred'][m],record['local_pred'][m],record['activated'][m],c),
                vs_current_local=transitions(y[m],record['local_pred'][m],record['pred'][m]),
                vs_preinstall=transitions(y[m],preinstall[m],record['pred'][m])) for scope,m in masks.items()}
            result.update(registry=registry.summary(),certified=record['certified'],
                          imported_row_unchanged=registry.head_matches(model,c),feature_hash=boundary_hash(model,True))
            stages[name]=result;write_json(out/'stage_metrics.json',stages)
            np.savez_compressed(out/f'{name}_predictions.npz',y_true=y,origin_client=validation['origin_client'],
                row_id=validation['rows'],local_pred=record['local_pred'],pred=record['pred'],activated=record['activated'])
            print(f'{name}: recall={result["pooled"]["recall"]:.6f}, break={result["pooled"]["break_count"]}, valid={record["certified"]}',flush=True)
            return result

        # Reproduce locked calibration prediction before any continuation.
        initial=registry.records(model,detector,acceptance['X'],seen,device,batch_size)
        with np.load(source/'holdout_predictions.npz',allow_pickle=False) as z:
            reproduction={k:int(np.count_nonzero(z[k]!=v)) for k,v in dict(y_true=acceptance['y'],
                origin_client=acceptance['origin_client'],row_id=acceptance['rows'],local_pred=initial['local_pred'],
                **{primary+'_pred':initial['pred']}).items()}
        if any(reproduction.values()):raise Rejected('LOCKED_HOLDOUT_REPRODUCTION_FAILED',str(reproduction))
        measure('installed')
        history=[]
        for step in range(1,rounds+1):
            baseline=OrderedDict((n,v.detach().cpu().clone()) for n,v in model.state_dict().items())
            feature_before=boundary_hash(model,True)
            update=local_update(model,detector,cid,roles,classes,task,round_id+step-1,config,device,batch_size,receiver_row_cap,registry)
            feature_after=boundary_hash(model,True)
            local_gate=registry.recertify(model,detector,acceptance,seen,device,batch_size,contract)
            local_measure=measure(f'round_{step}_after_local')
            deltas=[];ages=[];labels=[];weights=[];neighbor_updates=[]
            for peer in groups[cid]:
                print(f'Continuation {step}/{rounds}: graph peer {peer}',flush=True)
                neighbor,nroute=_make_denice_client_model(ckpt,peer,device)
                start={n:v.detach().cpu().clone() for n,v in neighbor.state_dict().items()}
                nupdate=local_update(neighbor,nroute,peer,roles,classes,task,round_id+step-1,config,device,batch_size,neighbor_row_cap)
                deltas.append(OrderedDict((n,v.detach().cpu()-start[n]) for n,v in neighbor.state_dict().items() if v.is_floating_point()))
                ages.append(copy.deepcopy(neighbor.unit_ranks));labels.append(sorted({int(v) for values in nroute.episode_classes.values() for v in values}))
                weights.append(alphas[cid][peer]);neighbor_updates.append(dict(client_id=peer,**nupdate))
                del neighbor,nroute;gc.collect()
            deltas.append(OrderedDict((n,v.detach().cpu()-baseline[n]) for n,v in model.state_dict().items() if v.is_floating_point()))
            ages.append(copy.deepcopy(model.unit_ranks));labels.append(sorted({int(v) for values in detector.episode_classes.values() for v in values}))
            weights.append(alphas[cid][cid])
            aggregated=age_aware_aggregate(baseline,model.unit_ranks,deltas,np.asarray(weights),
                AggregationConfig(eta=float(config.get('denice_eta_agg',config.get('eta_agg',1.))),protect_mature=True),
                neighbor_ages=ages,neighbor_labels=labels,target_labels=labels[-1])
            model.load_state_dict(aggregated,strict=True);registry.protect(model)
            aggregate_gate=registry.recertify(model,detector,acceptance,seen,device,batch_size,contract)
            aggregate_measure=measure(f'round_{step}_after_aggregation')
            history.append(dict(continuation_index=step,receiver_update=update,neighbor_updates=neighbor_updates,
                feature_changed_after_local=feature_before!=feature_after,
                feature_changed_after_aggregation=feature_after!=boundary_hash(model,True),
                local_acceptance=local_gate,aggregation_acceptance=aggregate_gate,
                local_valid=local_measure['certified'],aggregate_valid=aggregate_measure['certified']))
            write_json(out/'continuation_history.json',history)
            # Deliberately reload neighbors from the recorded terminal snapshot
            # each probe: report this; do not pretend it is evolving federation.
        bundle=dict(config=config,client_model_states={cid:model.state_dict()},
            client_algorithm_states={cid:{'denice':snapshot_denice_state(model,detector)}},
            guarded_head_entries=registry.entries,receiver=cid,seen_classes=seen)
        torch.save(bundle,out/'guarded_receiver.pt')
        loaded=torch.load(out/'guarded_receiver.pt',map_location='cpu',weights_only=False)
        restored,rroute=_make_denice_client_model(loaded,cid,device)
        restored_registry=GuardedHeadRegistry();restored_registry.entries=loaded['guarded_head_entries'];restored_registry.sync(restored)
        original_prediction=registry.records(model,detector,acceptance['X'],seen,device,batch_size)['pred']
        restored_prediction=restored_registry.records(restored,rroute,acceptance['X'],seen,device,batch_size)['pred']
        restore_ok=(complete_hash(restored,rroute)==complete_hash(model,detector) and
            digest(restored_registry.entries)==digest(registry.entries) and np.array_equal(original_prediction,restored_prediction))
        if not restore_ok:raise RuntimeError('Guarded receiver save/restore failed')
        checks=dict(rollback_verified=rollback['rollback_verified'],idempotent=replay['status']=='already_committed',
            pinned_row_overwrite_repaired=protection,optimizer_row_slots_zero=zero_slots,
            source_unchanged=complete_hash(receiver,router)==before_source,save_restore=restore_ok,
            locked_holdout_prediction_mismatch=reproduction,other_output_rows_unchanged_at_install=True)
        write_json(out/'integrity_checks.json',checks)
        observed_survival=all(h['local_valid'] and h['aggregate_valid'] for h in history)
        result=dict(completed_execution=True,installed=True,primary=primary,base_method='legacy',
            continuation_probe_passed=bool(observed_survival),passed_feasibility=False,
            full_training_round_verified=False,task_transition_verified=False,
            historical_retention_verified=False,final_test_opened=False,rounds=rounds,
            source_packet_bytes=len(packet),checkpoint_bytes=(out/'guarded_receiver.pt').stat().st_size,
            total_communication_measured=False,reason='Head installed; controlled continuation only, full-round/next-task proof remains open')
        write_json(out/'completion.json',result)
        return result
    except Rejected as exc:
        result=dict(completed_execution=True,passed_feasibility=False,stage='rejected',reason=exc.reason,detail=exc.detail)
        write_json(out/'completion.json',result);return result
    except Exception as exc:
        write_json(out/'completion.json',dict(completed_execution=False,passed_feasibility=False,
            stage='execution_error',error_type=type(exc).__name__,detail=str(exc)))
        raise
