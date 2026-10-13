"""Controlled Task4 continuation of an accepted historical-calibration packet.

Evolving receiver and recorded neighbors, full current BASE, FP32 NICE phases.
This excludes native CANC/age merge/router refresh and is not a federation replay.
"""
from collections import OrderedDict
import copy
import gc
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .config import Rejected
from .guarded_head import GuardedHeadRegistry, HeadContract
from .guarded_head_experiment import combine, locked_pool
from .parallel_route_experiment import metrics
from .portable_route import ProtectedRoute, receiver_signals, transitions
from .runner import load_input
from .selector import current_pool, recorded_graph
from .state import boundary_hash, complete_hash, digest, rng_snapshot, restore_rng, write_json
from .survival import local_update


def run_historical_survival(checkpoint,data_dir,roles_dir,source,out,device='cpu',cycles=3,batch_size=512):
    from eval_checkpoint import _make_denice_client_model
    from fed_learning.data.denice_clean_roles import CleanRoleData,file_sha256
    from fed_learning.training.checkpoint_state import snapshot_denice_state
    from fed_learning.strategies.decentralized.denice_aggregation import age_aware_aggregate,AggregationConfig
    source,out=Path(source),Path(out)
    if cycles<1 or batch_size<1:raise ValueError('Positive cycle and measurement budgets required')
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'completion.json',dict(completed_execution=False,final_test_opened=False))
    started=time.perf_counter()
    try:
        cid,donor,c,task,round_id=19,71,24,4,19
        parent_protocol=json.loads((source.parent/'protocol_lock.json').read_text(encoding='utf-8'))
        original_completion=json.loads((source/'completion.json').read_text(encoding='utf-8'))
        if not original_completion['capability_enabled'] or not original_completion['installed']:
            raise Rejected('ACCEPTED_PACKET_REQUIRED')
        ckpt,hashes=load_input(checkpoint,task,round_id)
        if any(parent_protocol[k]!=hashes[k] for k in hashes):raise Rejected('SURVIVAL_SOURCE_CHECKPOINT_CHANGED')
        config=ckpt['config']
        if config.get('denice_cl_method')!='legacy' or config['denice_similarity_threshold']!=.8:
            raise Rejected('LEGACY_XI8_REQUIRED')
        roles=CleanRoleData(roles_dir,source_data_dir=data_dir)
        role_sha=file_sha256(roles.root/'role_manifest.json')
        if role_sha!=parent_protocol['role_manifest_sha256'] or role_sha!=config['denice_data_roles_sha256']:
            raise Rejected('SURVIVAL_ROLES_CHANGED')
        metadata=json.loads((roles.source/'metadata.json').read_text(encoding='utf-8'))
        classes=list(map(int,metadata['task_structure']['task_classes'][str(task)]))
        seen=sorted(int(c) for t,values in metadata['task_structure']['task_classes'].items() if int(t)<=task for c in values)
        groups,alphas=recorded_graph(ckpt,task,round_id)
        if donor not in groups[cid]:raise Rejected('ORIGINAL_DONOR_NOT_IN_GRAPH')
        lock=json.loads((source/'decision_lock.json').read_text(encoding='utf-8'))
        packet=(source/'candidate.bin').read_bytes()
        if hashlib.sha256(packet).hexdigest()!=lock['candidate_packet_sha256']:raise Rejected('ACCEPTED_PACKET_CHANGED')
        route=ProtectedRoute.from_packet(packet)
        if (route.metadata['receiver'],route.metadata['donor'],route.metadata['class_id'])!=(cid,donor,c):
            raise Rejected('ACCEPTED_PAIR_CHANGED')
        split=json.loads((source/'calibration_manifest.json').read_text(encoding='utf-8'))
        if file_sha256(source/'calibration_manifest.json')!=route.metadata['historical_calibration_manifest_sha256']:
            raise Rejected('HISTORICAL_ACCEPTANCE_COORDINATES_CHANGED')
        acceptance=combine([locked_pool(roles,i,seen if i==cid else classes,split[str(i)]['holdout']['rows']) for i in (cid,donor)])
        validations=[]
        for i,scope in ((cid,seen),(donor,classes)):
            pool=current_pool(roles,i,'validation',scope)
            validations.append(dict(X=pool['X'],y=pool['y'],rows=pool['rows'],origin_client=np.full(len(pool['y']),i,dtype=np.int64)))
        validation=combine(validations)
        seed=torch.load(source/'guarded_receiver.pt',map_location='cpu',weights_only=False)
        model,router=_make_denice_client_model(seed,cid,device)
        registry=GuardedHeadRegistry();registry.entries=seed['guarded_head_entries'];registry.sync(model)
        if registry.entries[c]['packet']!=packet or not registry.certificate_current(model,router,cid=c):
            raise Rejected('ACCEPTED_SEED_CERTIFICATE_STALE')
        peers={i:_make_denice_client_model(ckpt,i,device) for i in groups[cid]}
        base_counts={i:sum(int(roles.manifest['clients'][str(i)]['role_class_counts']['base'].get(str(k),0)) for k in classes)
                     for i in (cid,*groups[cid])}
        training_batch=int(config['batch_size'])
        protocol=dict(version='appliance_historical_survival_v1',receiver=cid,donor=donor,class_id=c,task=task,
            **hashes,role_manifest_sha256=role_sha,source_decision_sha256=file_sha256(source/'decision_lock.json'),
            packet_sha256=hashlib.sha256(packet).hexdigest(),seed_sha256=file_sha256(source/'guarded_receiver.pt'),
            cycles=cycles,recorded_graph_round=round_id,neighbors=groups[cid],graph_alphas=alphas[cid],base_rows=base_counts,
            training_batch_size=training_batch,measurement_batch_size=batch_size,device=device,
            numeric_policy='FP32; AMP disabled for this controlled probe',all_current_base_rows=True,
            peer_weights_evolve_between_cycles=True,thresholds_retuned=False,seed=20261008,
            recertification='same frozen historical CALIBRATION HOLDOUT + per-class receiver FAR',
            validation='historical receiver Task0..4 + donor Task4; measurement only',
            limitations=['static recorded graph','no native task preparation/CANC/age merge/router refresh',
                'continued completed Task4, not Task5/full federation round','historical calibration retained; not replay-free proof'],
            versions=dict(torch=torch.__version__,numpy=np.__version__))
        write_json(out/'protocol_lock.json',protocol)
        np.savez_compressed(out/'validation_manifest.npz',y_true=validation['y'],row_id=validation['rows'],origin_client=validation['origin_client'])
        random.seed(20261008);np.random.seed(20261008);torch.manual_seed(20261008)
        stages=[];history=[];initial_pred=None

        def measure(stage,cycle,update=None):
            nonlocal initial_pred
            rng=rng_snapshot()
            try:
                if any(not torch.isfinite(v).all() for v in model.state_dict().values() if v.is_floating_point()):
                    raise FloatingPointError('Nonfinite receiver parameters')
                exact=registry.head_matches(model,c)
                before=registry.certificate_current(model,router,c)
                gate=registry.recertify(model,router,acceptance,seen,device,batch_size,HeadContract())
                candidate=registry.records(model,router,acceptance['X'],seen,device,batch_size,candidate=True)
                receiver=acceptance['origin_client']==cid
                class_far={int(k):float(candidate['activated'][receiver&(acceptance['y']==k)].mean())
                           for k in np.unique(acceptance['y'][receiver])}
                class_gate=all(far<=.001 for far in class_far.values())
                if not class_gate:
                    registry.entries[c].update(valid=False,reason='historical_per_class_recertification_failed');registry.sync(model)
                gate.update(per_class_far=class_far,per_class_passed=class_gate,combined_passed=bool(gate['passed'] and class_gate))
                record=registry.records(model,router,validation['X'],seen,device,batch_size)
                if initial_pred is None:initial_pred=record['pred'].copy()
                summaries={}
                for scope,mask in [('pooled',np.ones(len(validation['y']),bool)),('receiver',validation['origin_client']==cid),
                                   ('donor',validation['origin_client']==donor)]:
                    summaries[scope]=dict(**metrics(validation['y'][mask],record['pred'][mask],record['local_pred'][mask],record['activated'][mask],c),
                        vs_initial_installed=transitions(validation['y'][mask],initial_pred[mask],record['pred'][mask]))
                result=dict(stage=stage,cycle=cycle,head_exact_before_protection=exact,certificate_before_recertification=before,
                    certified=record['certified'],acceptance=gate,metrics=summaries,update=update,
                    feature_version=boundary_hash(model,True),registry=registry.summary())
                stages.append(result);write_json(out/'stage_metrics.json',stages)
                np.savez_compressed(out/f'{stage}_cycle_{cycle}_predictions.npz',pred=record['pred'],
                    local_pred=record['local_pred'],activated=record['activated'])
                print(f'Controlled survival cycle={cycle} stage={stage}: valid={record["certified"]}, '
                      f'recall={summaries["pooled"]["recall"]:.6f}, break={summaries["pooled"]["break_count"]}, '
                      f'false_activation={summaries["pooled"]["false_activation_rows"]}',flush=True)
                return result
            finally:restore_rng(rng)

        initial=measure('installed',0)
        with np.load(source/'validation_predictions.npz',allow_pickle=False) as previous:
            old=previous['accepted_or_local_pred']
            if not np.array_equal(old,initial_pred):raise Rejected('ACCEPTED_INITIAL_PREDICTIONS_CHANGED')
        for cycle in range(1,cycles+1):
            cycle_started=time.perf_counter()
            baseline=OrderedDict((n,v.detach().cpu().clone()) for n,v in model.state_dict().items())
            print(f'Controlled survival cycle={cycle}/{cycles}: receiver BASE={base_counts[cid]}, batch={training_batch}',flush=True)
            update=local_update(model,router,cid,roles,classes,task,round_id+cycle-1,config,device,training_batch,0,registry,
                                audit_optimizer_delta=True)
            after_local=measure('after_local',cycle,update)
            deltas=[];ages=[];labels=[];weights=[];peer_updates=[]
            for i,(peer,peer_router) in peers.items():
                print(f'Controlled survival cycle={cycle}: peer={i}, BASE={base_counts[i]}',flush=True)
                start={n:v.detach().cpu().clone() for n,v in peer.state_dict().items()}
                peer_update=local_update(peer,peer_router,i,roles,classes,task,round_id+cycle-1,config,device,training_batch,0)
                peer_updates.append(dict(client_id=i,**peer_update))
                deltas.append(OrderedDict((n,v.detach().cpu()-start[n]) for n,v in peer.state_dict().items() if v.is_floating_point()))
                ages.append(copy.deepcopy(peer.unit_ranks));labels.append(sorted({int(k) for values in peer_router.episode_classes.values() for k in values}))
                weights.append(alphas[cid][i]);del start
            deltas.append(OrderedDict((n,v.detach().cpu()-baseline[n]) for n,v in model.state_dict().items() if v.is_floating_point()))
            ages.append(copy.deepcopy(model.unit_ranks));labels.append(sorted({int(k) for values in router.episode_classes.values() for k in values}))
            weights.append(alphas[cid][cid])
            aggregated=age_aware_aggregate(baseline,model.unit_ranks,deltas,np.asarray(weights),
                AggregationConfig(eta=float(config.get('denice_eta_agg',config.get('eta_agg',1.))),protect_mature=True),
                neighbor_ages=ages,neighbor_labels=labels,target_labels=labels[-1])
            model.load_state_dict(aggregated,strict=True)
            head_before=registry.head_matches(model,c);registry.protect(model)
            after_aggregation=measure('after_aggregation',cycle)
            history.append(dict(cycle=cycle,receiver_update=update,peer_updates=peer_updates,
                head_exact_after_aggregation_before_protection=head_before,local_valid=after_local['certified'],
                aggregation_valid=after_aggregation['certified'],seconds=time.perf_counter()-cycle_started))
            write_json(out/'continuation_history.json',history)
            del baseline,deltas,aggregated;gc.collect()
        bundle=dict(config=config,seen_classes=seen,receiver=cid,client_model_states={cid:model.state_dict()},
            client_algorithm_states={cid:{'denice':snapshot_denice_state(model,router)}},guarded_head_entries=registry.entries)
        torch.save(bundle,out/'guarded_receiver.pt')
        loaded=torch.load(out/'guarded_receiver.pt',map_location='cpu',weights_only=False)
        restored,rrouter=_make_denice_client_model(loaded,cid,device)
        rreg=GuardedHeadRegistry();rreg.entries=loaded['guarded_head_entries'];rreg.sync(restored)
        equality=complete_hash(restored,rrouter)==complete_hash(model,router) and digest(rreg.entries)==digest(registry.entries)
        restore_checks=dict(state_equal=bool(equality),head_exact=rreg.head_matches(restored,c),
                            certified=rreg.certificate_current(restored,rrouter,c),cross_runtime_verified=False)
        write_json(out/'restore_checks.json',restore_checks)
        if not equality:raise RuntimeError('Endpoint state save/restore mismatch')
        steps=sum(int(h['receiver_update']['optimizer_steps']) for h in history)
        changed_steps=sum(int(h['receiver_update']['successful_steps_with_parameter_change']) for h in history)
        result=dict(completed_execution=True,receiver=cid,cycles=cycles,receiver_successful_steps=steps,
            receiver_steps_with_parameter_change=changed_steps,
            survived_controlled_local_and_aggregation=bool(steps>0 and all(s['certified'] for s in stages)),
            nonzero_weight_update_survival_verified=bool(changed_steps>0 and all(s['certified'] for s in stages)),
            final=stages[-1],final_test_opened=False,full_native_round_verified=False,passed_full_feasibility=False,
            elapsed_seconds=time.perf_counter()-started)
        write_json(out/'completion.json',result);return result
    except Exception as exc:
        write_json(out/'completion.json',dict(completed_execution=False,final_test_opened=False,passed_full_feasibility=False,
            error_type=type(exc).__name__,error=str(exc)))
        raise
