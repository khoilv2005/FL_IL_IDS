"""Native lifecycle observer using current own CAL only; no historical recertify.

This is the bounded native verification path for scope lifecycle V1. It retains
installed heads during SUSPENDED, but does not grant an old certificate to the
new cumulative test domain. Automatic discovery/install is a separate runner
controller; pre-installed diagnostic seeds are explicitly reported as such.
"""
import json
import shutil
import time
from pathlib import Path
from .config import Rejected
from .current_calibration_data import CurrentCalibrationData
from .native_lifecycle import NativeLifecycleObserver
from .stable_head import StableHeadRegistry, guard_function,canonical_guard_fingerprint
from .portable_route import ProtectedRoute
from .state import complete_hash, digest, rng_snapshot, restore_rng, write_json


class CurrentNativeLifecycleObserver:
    # These methods perform row protection and serialization only. Neither
    # constructor nor historical observe/recertify methods are reused.
    training_hooks=NativeLifecycleObserver.training_hooks

    def __init__(self,config):
        self.config=config
        self.spec=json.loads(Path(config['appliance_current_native_manifest']).read_text(encoding='utf-8'))
        if self.spec.get('version')!='appliance_current_native_scope_probe_v1':
            raise Rejected('CURRENT_NATIVE_MANIFEST_VERSION_CHANGED')
        self.out=Path(self.spec['output_dir']);self.out.mkdir(parents=True,exist_ok=True)
        self.round_budget=int(self.spec['round_budget'])
        if not 1<=self.round_budget<int(config['rounds_per_task']):raise Rejected('CURRENT_NATIVE_BOUNDED_BUDGET_REQUIRED')
        self.metadata=json.loads((Path(self.spec['calibration_store'])/'metadata.json').read_text(encoding='utf-8'))
        self.tracked={int(cid):dict(class_id=int(c),registry=None,local_updates=[],phase_protection=[],
            optimizer_delta_audit=[],delta_report_cursor=0) for cid,c in self.spec['tracked'].items()}
        self.events=[];self.rounds=[];self.endpoint_seconds=0.;self.current_views={}
        write_json(self.out/'observer_protocol.json',dict(version=self.spec['version'],current_CAL_only=True,
            historical_raw_recertification=False,initial_acceptance_replayed=False,
            initial_patch_source='pre-installed development fixture; not automatic in-training discovery',
            dependency_drift_budget=0.,CAL_minimum_unchanged=32,scope_expansion=False,
            suspended_head_protection=True,final_test_opened=False))

    def _registry(self,cid,model):
        item=self.tracked[cid]
        if item['registry'] is None:
            reg=StableHeadRegistry();reg.entries=getattr(model,'appliance_guarded_head_entries',{})
            if set(reg.entries)!={item['class_id']} or 'lifecycle_certificate' not in reg.entries[item['class_id']]:
                raise Rejected('CURRENT_NATIVE_SCOPED_REGISTRY_REQUIRED')
            item['registry']=reg
        item['registry'].sync(model)
        return item['registry']

    def _current_scope(self,task):
        # Application domain declaration, not router prediction or sample label.
        # This cumulative domain differs from initial local CAL development.
        seen=sorted({int(c) for t,v in self.metadata['task_structure']['task_classes'].items() if int(t)<=task for c in v})
        return dict(domain_id=f'cumulative_data_domain_through_task_{task}',classes=seen)

    def observe_client(self,stage,cid,models,routers,task,round_id,active_ids,extra=None):
        if cid not in self.tracked:return
        rng=rng_snapshot()
        try:
            item=self.tracked[cid];model=models[cid];router=routers[cid];c=item['class_id']
            reg=self._registry(cid,model);entry=reg.entries[c]
            before=reg.head_matches(model,c);reg.protect(model)
            physical=reg.certificate_current(model,router,c)
            scope=self._current_scope(task)
            report=reg.update_lifecycle(model,router,c,scope,int(task))
            monitor=None
            if stage=='native_round_finished':
                view=self.current_views.get(cid)
                if view is None:
                    view=CurrentCalibrationData(self.spec['calibration_store'],cid,int(task),self.spec['role_manifest_sha256'])
                    self.current_views[cid]=view
                elif view.task!=task:view.advance(int(task))
                monitor=reg.current_negative_gate(model,router,c,view,scope['classes'],
                    str(next(model.parameters()).device),int(self.spec['audit_batch_size']),runtime_scope=scope)
                report=monitor['lifecycle']
            try:
                route=ProtectedRoute.from_packet(entry['packet'])
                actual=guard_function(model,route,entry['reference_classes'])['fingerprint']
                canonical=canonical_guard_fingerprint(model,route,entry['reference_classes'])
            except Rejected as exc:
                actual=f'rejected:{exc.reason}'
                canonical=actual
            event=dict(stage=stage,task=int(task),round=int(round_id),receiver=cid,class_id=c,
                active=cid in active_ids,head_exact_before_protection=before,
                head_exact_after_protection=reg.head_matches(model,c),
                certificate_function_current=physical,actual_guard_function_fingerprint=actual,
                actual_canonical_guard_fingerprint=canonical,
                numeric_equivalence_attestation_sha256=entry.get('numeric_equivalence_attestation_sha256'),
                initial_guard_function_fingerprint=entry['guard_function_fingerprint'],
                immutable_certificate_sha256=entry['lifecycle_certificate_sha256'],
                lifecycle=report,current_negative_monitor=monitor,raw_historical_CAL_reads=0)
            if extra and 'training' in extra:
                training=extra['training']
                audits=item['optimizer_delta_audit'][item['delta_report_cursor']:]
                item['delta_report_cursor']=len(item['optimizer_delta_audit'])
                update=dict(task=task,round=round_id,**{k:training.get(k) for k in ('optimizer_steps','skipped_optimizer_steps','amp')},
                    successful_steps_with_parameter_change=sum(a['changed_elements']>0 for a in audits))
                item['local_updates'].append(update);event['training']=update
            self.events.append(event);write_json(self.out/'lifecycle_history.json',self.events)
            print(f'APPLIANCE current receiver={cid} stage={stage} state={report["state"]} '
                  f'function_current={physical} reason={report["reason"]}',flush=True)
        finally:restore_rng(rng)

    def observe(self,stage,models,routers,task,round_id,active_ids,extra=None):
        for cid in self.tracked:
            if cid not in models or cid not in routers:raise Rejected('CURRENT_NATIVE_RECEIVER_MISSING')
            self.observe_client(stage,cid,models,routers,task,round_id,active_ids,extra)

    def _save_restore_endpoint(self,models,routers,task,round_id):
        import torch
        from fed_learning.training.checkpoint_state import snapshot_denice_state
        from eval_checkpoint import _make_denice_client_model
        started=time.perf_counter();ids=sorted(self.tracked)
        saved=dict(config={k:v for k,v in self.config.items() if k!='resume_state_path'},task=task,round=round_id,
            client_model_states={cid:{k:v.detach().cpu().clone() for k,v in models[cid].state_dict().items()} for cid in ids},
            client_algorithm_states={cid:{'denice':snapshot_denice_state(models[cid],routers[cid])} for cid in ids},
            checkpoint_kind='current native scoped endpoint; not full federation resume')
        path=self.out/'native_probe_endpoint.pt';torch.save(saved,path)
        restored=torch.load(path,map_location='cpu',weights_only=False);checks={};rng=rng_snapshot()
        try:
            for cid in ids:
                model,router=_make_denice_client_model(restored,cid,'cpu')
                reg=StableHeadRegistry();reg.entries=model.appliance_guarded_head_entries;reg.sync(model)
                original=self.tracked[cid]['registry'];c=self.tracked[cid]['class_id']
                checks[cid]=dict(complete_state_equal=complete_hash(model,router)==complete_hash(models[cid],routers[cid]),
                    registry_equal=digest(reg.entries)==digest(original.entries),head_exact=reg.head_matches(model,c),
                    state_equal=reg.entries[c]['lifecycle_state']==original.entries[c]['lifecycle_state'])
        finally:restore_rng(rng)
        write_json(self.out/'endpoint_restore_checks.json',checks)
        self.endpoint_seconds+=time.perf_counter()-started
        if not all(all(v.values()) for v in checks.values()):raise Rejected('CURRENT_NATIVE_ENDPOINT_RESTORE_MISMATCH')
        return checks

    def round_finished(self,task,round_id,schedule_rounds,models,routers,active_ids,cluster):
        self.observe('native_round_finished',models,routers,task,round_id,active_ids)
        restored=self._save_restore_endpoint(models,routers,task,round_id)
        shutil.copyfile(self.out/'native_probe_endpoint.pt',self.out/f'endpoint_task_{task}_round_{round_id}.pt')
        self.rounds.append(dict(task=task,round=round_id,active_clients=len(active_ids),restore=restored,
            group_size_stats=cluster.get('group_size_stats'),graph_recomputed=True))
        write_json(self.out/'native_round_history.json',self.rounds)
        return len(self.rounds)>=self.round_budget and round_id<schedule_rounds-1

    def finish(self,models,routers,task,round_id,task_completed,native_history):
        from eval_checkpoint import _make_denice_client_model
        import torch
        restored=self._save_restore_endpoint(models,routers,task,round_id)
        saved=torch.load(self.out/'native_probe_endpoint.pt',map_location='cpu',weights_only=False)
        checks={};summaries={}
        for cid,item in self.tracked.items():
            c=item['class_id'];reg=item['registry'];entry=reg.entries[c]
            clone,detector=_make_denice_client_model(saved,cid,'cpu')
            rr=StableHeadRegistry();rr.entries=clone.appliance_guarded_head_entries;rr.sync(clone)
            checks[f'lifecycle_restore_{cid}']=digest(rr.entries[c])==digest(entry)
            checks[f'head_stays_protected_{cid}']=all(e['head_exact_after_protection'] for e in self.events if e['receiver']==cid)
            checks[f'new_domain_never_carries_{cid}']=all(e['lifecycle']['state']=='SUSPENDED' for e in self.events if e['receiver']==cid)
            checks[f'certificate_not_overwritten_{cid}']=len({e['immutable_certificate_sha256'] for e in self.events if e['receiver']==cid})==1
            checks[f'CAL_only_current_task_{cid}']=all(a['task']==task for a in self.current_views[cid].access_log)
            summaries[cid]=dict(class_id=c,state=entry['lifecycle_state'],reason=entry['lifecycle_reason'],
                successful_parameter_changing_steps=sum(u['successful_steps_with_parameter_change'] for u in item['local_updates']),
                skipped_optimizer_steps=sum(int(u['skipped_optimizer_steps'] or 0) for u in item['local_updates']),
                initial_function_survived=all(e['certificate_function_current'] for e in self.events if e['receiver']==cid))
        write_json(self.out/'optimizer_parameter_delta_audit.json',{cid:v['optimizer_delta_audit'] for cid,v in self.tracked.items()})
        result=dict(completed_execution=all(checks.values()),native_rounds_completed=len(self.rounds),
            original_round_schedule=self.config['rounds_per_task'],task=task,last_round=round_id,task_completed=task_completed,
            checks=checks,comparisons=len(checks),mismatches=sum(not v for v in checks.values()),tracked_receivers=summaries,
            current_CAL_access={cid:v.access_log for cid,v in self.current_views.items()},historical_raw_CAL_read=False,
            final_test_opened=False,partial_endpoint_is_full_resume_state=False,
            automatic_discovery_install_verified=False,active_patch_recall_survival_verified=False,
            main_install_authorized=False,limitation='Scoped preinstalled fixture; suspension expected in unverified cumulative domain')
        write_json(self.out/'completion.json',result)
        if not result['completed_execution']:raise AssertionError('Current native lifecycle checks failed')
        return result
