"""Opt-in observation/protection of guarded heads in the real DeNICE runner.

This is a diagnostic continuation, not a new aggregation algorithm. Historical
calibration is explicitly retained by the offline observer for recertification;
this experiment does not establish a replay-free calibration protocol.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from .config import Rejected
from .guarded_head import GuardedHeadRegistry, HeadContract, original_local_head
from .guarded_head_experiment import combine, locked_pool
from .parallel_route_experiment import metrics
from .portable_route import receiver_signals, transitions
from .selector import current_pool, lookup
from .state import boundary_hash, complete_hash, digest, rng_snapshot, restore_rng, write_json


class NativeLifecycleObserver:
    def __init__(self,config):
        from fed_learning.data.denice_clean_roles import CleanRoleData,file_sha256
        self.config=config
        self.spec=json.loads(Path(config['appliance_native_probe_manifest']).read_text(encoding='utf-8'))
        self.out=Path(self.spec['output_dir']);self.out.mkdir(parents=True,exist_ok=True)
        self.round_budget=int(self.spec['round_budget'])
        if not 1<=self.round_budget<=int(config['rounds_per_task']):raise ValueError('Invalid native probe budget')
        self.roles=CleanRoleData(self.spec['roles_dir'],source_data_dir=config['data_dir'])
        self.metadata=json.loads((self.roles.source/'metadata.json').read_text(encoding='utf-8'))
        self.tracked={};self.events=[];self.rounds=[]
        self.instrumentation_seconds=0.0
        self.endpoint_seconds=0.0
        for key,folder in self.spec['portable_dirs'].items():
            cid=int(key);source=Path(folder)
            protocol=json.loads((source/'protocol_lock.json').read_text(encoding='utf-8'))
            if protocol['role_manifest_sha256']!=file_sha256(self.roles.root/'role_manifest.json'):
                raise Rejected('NATIVE_ROLE_LOCK_MISMATCH')
            if cid!=int(protocol['receiver']):raise Rejected('NATIVE_RECEIVER_SCOPE_MISMATCH')
            donor,task,c=map(int,(protocol[n] for n in ('donor','task','class_id')))
            classes=list(map(int,self.metadata['task_structure']['task_classes'][str(task)]))
            old_classes=sorted({int(c) for t,v in self.metadata['task_structure']['task_classes'].items() if int(t)<=task for c in v})
            split=json.loads((source/'calibration_split_manifest.json').read_text(encoding='utf-8'))
            scopes=protocol.get('acceptance_class_scope_by_client',{})
            acceptance=combine([locked_pool(self.roles,i,scopes.get(str(i),classes),split[str(i)]['holdout']['rows']) for i in (cid,donor)])
            # Broader old-class damage is measured, never used for fitting or
            # for selecting another pair after seeing these predictions.
            validation=[]
            if self.spec.get('measure_validation',True):
                for i,scope in ((cid,old_classes),(donor,classes)):
                    p=current_pool(self.roles,i,'validation',scope)
                    validation.append(dict(X=p['X'],y=p['y'],rows=p['rows'],origin_client=np.full(len(p['y']),i,dtype=np.int64)))
            else:
                validation=[acceptance]
            self.tracked[cid]=dict(task=task,class_id=c,donor=donor,acceptance=acceptance,
                validation=combine(validation),reference_features=None,initial_prediction=None,
                initial_local_prediction=None,registry=None,local_updates=[],phase_protection=[],
                optimizer_delta_audit=[],delta_report_cursor=0)
        write_json(self.out/'observer_protocol.json',dict(round_budget=self.round_budget,
            tracked_receivers=sorted(self.tracked),native_schedule_rounds=config['rounds_per_task'],
            xi=config['denice_similarity_threshold'],training_batch_size=config['batch_size'],
            training_amp=config['denice_amp_enabled'],final_test_opened=False,thresholds_retuned=False,
            amp_initial_scale=config.get('denice_amp_initial_scale',65536.),
            calibration_policy='offline diagnostic retains locked calibration HOLDOUT during Task5; not replay-free calibration proof',
            validation_policy=('receiver historical validation + donor Task4 validation; metrics only' if self.spec.get('measure_validation',True) else
                'not opened; stage measurements use locked calibration HOLDOUT only'),
            per_class_receiver_far_budget=self.spec.get('receiver_per_class_far_budget'),
            optimizer_parameter_delta_audit=bool(self.spec.get('audit_optimizer_delta',False)),
            checkpoint_policy='overwrite one endpoint for tracked receivers; partial native round state is NOT a task-boundary resume state'))

    def _registry(self,cid,model):
        item=self.tracked[cid]
        if item['registry'] is None:
            entries=getattr(model,'appliance_guarded_head_entries',None)
            if not entries:raise Rejected('NATIVE_IMPORTED_REGISTRY_MISSING',str(cid))
            if len(entries)!=1:raise Rejected('NATIVE_SINGLE_CAPABILITY_REQUIRED',str(cid))
            reg=GuardedHeadRegistry();reg.entries=entries
            entry=next(iter(entries.values()))
            if int(entry['receiver'])!=cid or int(entry['class_id'])!=item['class_id']:
                raise Rejected('NATIVE_IMPORTED_REGISTRY_SCOPE_MISMATCH',str(cid))
            item['registry']=reg;reg.sync(model)
        return item['registry']

    def training_hooks(self,cid,model):
        if cid not in self.tracked:return {}
        if self.config.get('denice_cl_method')!='legacy':raise Rejected('NATIVE_LEGACY_REQUIRED')
        registry=self._registry(cid,model);registry.protect(model)
        optimizers=[]
        snapshots={}
        def factory(parameters,lr):
            optimizer=torch.optim.Adam(parameters,lr=lr)
            # Native NICE phase selection/pruning precedes optimizer creation.
            # Restore the protected row before the first forward of the phase.
            self.tracked[cid]['phase_protection'].append(dict(
                head_exact_after_native_pruning=registry.head_matches(model,self.tracked[cid]['class_id'])))
            registry.protect(model,optimizer)
            optimizers.append(optimizer)
            return optimizer
        def before(current):
            snapshots.clear()
            snapshots.update({name:p.detach().clone() for name,p in current.named_parameters()})
        def after(current):
            registry.protect(current,optimizers[-1] if optimizers else None)
            if self.spec.get('audit_optimizer_delta',False):
                changed={}
                for name,p in current.named_parameters():
                    delta=p.detach()-snapshots[name];count=int(torch.count_nonzero(delta))
                    if count:changed[name]=dict(elements=count,max_absolute=float(delta.abs().max()))
                self.tracked[cid]['optimizer_delta_audit'].append(dict(changed_parameters=changed,
                    changed_elements=sum(v['elements'] for v in changed.values()),
                    nonzero_gradient_elements=sum(int(torch.count_nonzero(p.grad)) for p in current.parameters() if p.grad is not None)))
                snapshots.clear()
        hooks=dict(optimizer_factory=factory,gradient_filter=lambda:registry.gradient_filter(model),after_optimizer_step_update=after)
        if self.spec.get('audit_optimizer_delta',False):hooks['before_optimizer_step']=before
        return hooks

    def observe_client(self,stage,cid,models,routers,task,round_id,active_ids,extra=None):
        if cid not in self.tracked:return
        started=time.perf_counter()
        # All metric/recertification instrumentation must leave the native
        # training random streams as they were before observation.
        rng=rng_snapshot()
        try:
            item=self.tracked[cid];model=models[cid];router=routers[cid]
            device=str(next(model.parameters()).device);batch=int(self.spec.get('audit_batch_size',512))
            seen=sorted({int(c) for t,v in self.metadata['task_structure']['task_classes'].items() if int(t)<=task for c in v})
            registry=self._registry(cid,model);c=item['class_id'];entry=registry.entries[c]
            row_before=registry.head_matches(model,c)
            registry.protect(model)
            certified_before=registry.certificate_current(model,router,c)
            calibration=registry.recertify(model,router,item['acceptance'],seen,device,batch,HeadContract())
            far_budget=self.spec.get('receiver_per_class_far_budget')
            if far_budget is not None:
                measured=registry.records(model,router,item['acceptance']['X'],seen,device,batch,candidate=True)
                ay=item['acceptance']['y'];own=item['acceptance']['origin_client']==cid
                fars={int(cc):float(measured['activated'][own & (ay==cc)].mean()) for cc in np.unique(ay[own])}
                per_class_pass=all(v<=float(far_budget) for v in fars.values())
                calibration.update(receiver_far_by_class=fars,per_class_far_passed=per_class_pass)
                if not per_class_pass:
                    entry.update(valid=False,reason='historical_per_class_recertification_failed');registry.sync(model)
                    calibration['passed']=False
            record=registry.records(model,router,item['validation']['X'],seen,device,batch)
            with original_local_head(model,c,entry['backup']):
                features=receiver_signals(model,router,item['acceptance']['X'],seen,item['task'],c,batch,device)['imported_features']
            if item['reference_features'] is None:
                item['reference_features']=features.copy()
                item['initial_prediction']=record['pred'].copy()
                item['initial_local_prediction']=record['local_pred'].copy()
                np.savez_compressed(self.out/f'receiver_{cid}_validation_manifest.npz',
                    y_true=item['validation']['y'],row_id=item['validation']['rows'],origin_client=item['validation']['origin_client'])
            delta=np.asarray(features,np.float64)-np.asarray(item['reference_features'],np.float64)
            y=item['validation']['y'];origin=item['validation']['origin_client']
            summaries={}
            for name,mask in (('pooled',np.ones(len(y),bool)),('receiver',origin==cid),('donor',origin==item['donor'])):
                summaries[name]=dict(**metrics(y[mask],record['pred'][mask],record['local_pred'][mask],record['activated'][mask],c),
                    vs_initial_local=transitions(y[mask],item['initial_local_prediction'][mask],record['pred'][mask]),
                    vs_initial_installed=transitions(y[mask],item['initial_prediction'][mask],record['pred'][mask]))
            event=dict(stage=stage,task=int(task),round=int(round_id),receiver=cid,
                active=cid in active_ids,class_id=c,head_exact_before_protection=row_before,
                head_exact_after_protection=registry.head_matches(model,c),certified_before_recertification=certified_before,
                acceptance=calibration,valid=record['certified'],feature_hash=boundary_hash(model,True),
                feature_drift=dict(mean_absolute=float(np.abs(delta).mean()),max_absolute=float(np.abs(delta).max()),
                    changed_elements=int(np.count_nonzero(delta))),metrics=summaries,
                registry=registry.summary(),extra=extra or {})
            event['phase_protection']=list(item['phase_protection'])
            if extra and 'training' in extra:
                training=extra['training'];event['extra']={'training':{k:training.get(k) for k in
                    ('loss','optimizer_steps','skipped_optimizer_steps','amp')},
                    'capacity_reserve_released':extra.get('capacity_reserve_released')}
                if self.spec.get('audit_optimizer_delta',False):
                    audits=item['optimizer_delta_audit'][item['delta_report_cursor']:]
                    item['delta_report_cursor']=len(item['optimizer_delta_audit'])
                    event['extra']['training'].update(optimizer_delta_audit=audits,
                        successful_steps_with_parameter_change=sum(v['changed_elements']>0 for v in audits))
                item['local_updates'].append(dict(task=task,round=round_id,**event['extra']['training']))
            event['measurement_seconds_before_artifact_write']=time.perf_counter()-started
            self.events.append(event)
            write_json(self.out/'lifecycle_history.json',self.events)
            # Small prediction arrays only; no raw calibration inputs stored.
            np.savez_compressed(self.out/f'receiver_{cid}_{stage}_t{task}_r{round_id}_predictions.npz',
                pred=record['pred'],local_pred=record['local_pred'],activated=record['activated'])
            print(f'APPLIANCE native receiver={cid} stage={stage} active={event["active"]} '
                f'valid={event["valid"]} recall={summaries["pooled"]["recall"]:.4%} '
                f'break={summaries["pooled"]["break_count"]} drift_MAE={event["feature_drift"]["mean_absolute"]:.6g}',flush=True)
        finally:
            restore_rng(rng)
            self.instrumentation_seconds+=time.perf_counter()-started

    def observe(self,stage,models,routers,task,round_id,active_ids,extra=None):
        for cid in self.tracked:
            if cid not in models or cid not in routers:raise Rejected('NATIVE_RECEIVER_DISAPPEARED',str(cid))
            self.observe_client(stage,cid,models,routers,task,round_id,active_ids,extra)

    def _save_restore_endpoint(self,models,routers,task,round_id):
        started=time.perf_counter()
        from fed_learning.training.checkpoint_state import snapshot_denice_state
        from eval_checkpoint import _make_denice_client_model
        ids=sorted(self.tracked)
        state=dict(config={k:v for k,v in self.config.items() if k!='resume_state_path'},task=task,round=round_id,
            client_model_states={i:{k:v.detach().cpu().clone() for k,v in models[i].state_dict().items()} for i in ids},
            client_algorithm_states={i:{'denice':snapshot_denice_state(models[i],routers[i])} for i in ids},
            checkpoint_kind='appliance_native_probe_endpoint; not a complete federation/intra-task resume state')
        path=self.out/'native_probe_endpoint.pt';torch.save(state,path)
        saved=torch.load(path,map_location='cpu',weights_only=False);checks={}
        rng=rng_snapshot()
        try:
            for cid in ids:
                model,router=_make_denice_client_model(saved,cid,'cpu')
                entries=getattr(model,'appliance_guarded_head_entries',None)
                registry=GuardedHeadRegistry();registry.entries=entries or {};registry.sync(model)
                original=self.tracked[cid]['registry']
                checks[cid]=dict(registry_persisted=bool(entries),
                    complete_model_router_state_equal=complete_hash(model,router)==complete_hash(models[cid],routers[cid]),
                    complete_registry_equal=digest(entries)==digest(original.entries),
                    imported_head_exact=bool(entries) and registry.head_matches(model,self.tracked[cid]['class_id']))
                del model,router,registry
        finally:restore_rng(rng)
        write_json(self.out/'endpoint_restore_checks.json',checks)
        if not all(all(v.values()) for v in checks.values()):raise Rejected('NATIVE_ENDPOINT_RESTORE_MISMATCH')
        self.endpoint_seconds+=time.perf_counter()-started
        return checks

    def round_finished(self,task,round_id,schedule_rounds,models,routers,active_ids,cluster):
        self.observe('native_round_finished',models,routers,task,round_id,active_ids)
        restore=self._save_restore_endpoint(models,routers,task,round_id)
        self.rounds.append(dict(task=task,round=round_id,active_client_count=len(active_ids),
            tracked_active={cid:cid in active_ids for cid in self.tracked},
            graph_recomputed=True,group_size_stats=cluster.get('group_size_stats'),
            age_merge_policy=self.config.get('denice_age_merge_policy','none'),
            router_freshness=cluster.get('router_freshness'),restore_checks=restore))
        write_json(self.out/'native_round_history.json',self.rounds)
        # A limited diagnostic ends at a real round boundary. Do not shorten
        # rounds_per_task and accidentally invoke finalization after 3 phases.
        return len(self.rounds)>=self.round_budget and round_id<schedule_rounds-1

    def finish(self,models,routers,task,round_id,task_completed,native_history):
        self._save_restore_endpoint(models,routers,task,round_id)
        survival={cid:dict(local_training_attempted=bool(item['local_updates']),
            local_training_performed=sum(int(u.get('optimizer_steps') or 0) for u in item['local_updates'])>0,
            optimizer_steps=sum(int(u.get('optimizer_steps') or 0) for u in item['local_updates']),
            skipped_optimizer_steps=sum(int(u.get('skipped_optimizer_steps') or 0) for u in item['local_updates']),
            certified_survival_after_successful_local_updates=(
                sum(int(u.get('optimizer_steps') or 0) for u in item['local_updates'])>0
                and all(e['valid'] for e in self.events if e['receiver']==cid)),
            valid_at_endpoint=item['registry'].certificate_current(models[cid],routers[cid],item['class_id']),
            all_observed_stages_valid=all(e['valid'] for e in self.events if e['receiver']==cid),
            head_repairs_at_observed_boundaries=sum(not e['head_exact_before_protection'] for e in self.events if e['receiver']==cid),
            head_repairs_after_native_phase_pruning=sum(not p['head_exact_after_native_pruning'] for p in item['phase_protection']),
            observed_feature_drift_max=max(e['feature_drift']['max_absolute'] for e in self.events if e['receiver']==cid))
            for cid,item in self.tracked.items()}
        if self.spec.get('audit_optimizer_delta',False):
            for cid,item in self.tracked.items():
                steps=sum(v['changed_elements']>0 for v in item['optimizer_delta_audit'])
                survival[cid].update(successful_steps_with_parameter_change=steps,
                    nonzero_weight_update_survival_verified=steps>0 and survival[cid]['all_observed_stages_valid'])
            write_json(self.out/'optimizer_parameter_delta_audit.json',{cid:item['optimizer_delta_audit'] for cid,item in self.tracked.items()})
        result=dict(completed_execution=True,stage='native_probe_completed',native_rounds_completed=len(self.rounds),
            original_round_schedule=self.config['rounds_per_task'],task=task,last_round=round_id,
            task_preparation_executed=True,full_native_rounds_executed=True,task_completed=bool(task_completed),
            tracked_receivers=survival,passed_feasibility=False,final_test_opened=False,
            historical_validation_measured=bool(self.spec.get('measure_validation',True)),replay_free_calibration_verified=False,
            measurement_role=('validation' if self.spec.get('measure_validation',True) else 'locked calibration HOLDOUT'),
            observer_reporting_schema_version=3 if self.spec.get('audit_optimizer_delta',False) else 2,
            diagnostic_instrumentation_seconds=self.instrumentation_seconds,
            endpoint_save_restore_seconds=self.endpoint_seconds,
            diagnostic_timing_is_production_appliance_overhead=False,
            entire_task_transition_verified=bool(task_completed),
            partial_endpoint_is_native_resume_state=False,
            reason='Bounded next-task native diagnostic; retained historical calibration, not a full APPLIANCE campaign')
        write_json(self.out/'completion.json',result)
        write_json(self.out/'native_training_round_metrics.json',[r for r in native_history.get('round_metrics',[]) if int(r['task'])==task])
        write_json(self.out/'federation_optimizer_audit.json',[r for r in native_history.get('continual_losses',[]) if int(r['task'])==task])
        return result
