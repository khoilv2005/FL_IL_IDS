"""Calibration-only multi-class pilot for receivers scheduled in Task5.

Freeze one structural pair per Task4 class before any prediction. Failed pairs
are retained, never replaced. No validation or final test is opened.
"""
import copy
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import torch

from .closure import effective_linear
from .config import Protocol,Rejected
from .evaluate import Predictor
from .guarded_head import GuardedHeadRegistry,HeadContract,compile_guarded_head,install_guarded_head
from .guarded_head_experiment import combine
from .historical_calibration import calibration_views,select_guard,summarize,subset
from .imported_route import ROUTE_RULES,stratified_roles
from .ledger import quality
from .portable_route import PORTABLE_RULES,ProtectedRoute,SharedSketch,prototype_summary,receiver_signals
from .runner import load_input
from .selector import current_pool,lookup,recorded_graph
from .state import boundary_hash,write_json


def split_counts(n):
    if n<3:return (n if n==1 else 1 if n==2 else 0,n==2,0)
    fit=min(n-2,max(1,n//2));selection=min(n-fit-1,max(1,n//4))
    return fit,selection,n-fit-selection


def structural_pairs(ckpt,manifest,classes,seen,future_classes,all_options=False,task=4,round_id=19):
    groups,alphas=recorded_graph(ckpt,task,round_id)
    chosen=[];coverage={};used=set()
    def state(cid):
        value=lookup(ckpt['client_algorithm_states'],cid,{})
        return value.get('denice',value)
    for c in sorted(classes):
        options=[]
        for receiver in sorted(groups):
            counts=manifest['clients'][str(receiver)]['role_class_counts'];alg=state(receiver)
            future=sum(int(counts['base'].get(str(k),0)) for k in future_classes)
            neg_holdout=sum(split_counts(int(counts['calibration'].get(str(k),0)))[2] for k in seen if k!=c)
            if ((not all_options and receiver in used) or int(counts['base'].get(str(c),0)) or int(counts['calibration'].get(str(c),0))
                or future<=0 or neg_holdout<32 or int(alg.get('neuron_ages',{}).get('fc2',np.ones(34))[c])!=0
                or alg.get('context_detector',{}).get('router_last_refresh_task')!=task):continue
            for donor in groups[receiver]:
                dc=manifest['clients'][str(donor)]['role_class_counts'];ds=state(donor)
                f,s,h=split_counts(int(dc['calibration'].get(str(c),0)))
                if (int(dc['base'].get(str(c),0))>0 and f>=32 and s>=8 and h>=32
                    and ds.get('context_detector',{}).get('router_last_refresh_task')==task
                    and int(ds.get('neuron_ages',{}).get('fc2',np.zeros(34))[c])>=2
                    and alphas[receiver].get(donor,0)>0):
                    options.append(dict(receiver=receiver,donor=donor,class_id=c,task=task,round=round_id,
                        next_task=task+1,next_task_base_rows=future,receiver_historical_holdout_rows=neg_holdout,
                        donor_calibration_split_counts=dict(fit=f,selection=s,holdout=h)))
        coverage[c]=dict(structural_options=len(options),reason=None if options else 'no structural candidate under common support/provenance rules')
        if all_options:
            chosen.extend(options)
        elif options:
            pair=options[0];chosen.append(pair);used.add(pair['receiver'])
    return chosen,coverage


def historical_fit_pool(roles,cid,tasks,current_task=4):
    """Score only deterministic FIT rows; no SELECTION/HOLDOUT predictions."""
    parts=[]
    for task in range(current_task+1):
        pool=current_pool(roles,cid,'calibration',tasks[task])
        seed=ROUTE_RULES['seed']+cid+(100003*(task+1) if task<current_task else 0)
        positions=stratified_roles(pool,seed)['fit']
        parts.append(subset(pool,positions,cid))
    return combine(parts)


def discover_fit_pairs(ckpt, roles, classes, seen, future_classes, out, device, batch_size, functional=False, guarded=False,
                       task=4,round_id=19,max_pairs_per_class=1):
    """Discover offers on FIT only; freeze the entire schedule before guard selection.

    This is a new development experiment, not replacement inside the ID-first
    pilot. Reused donor calibration splits remain development evidence.
    """
    from eval_checkpoint import _make_denice_client_model
    options, coverage = structural_pairs(ckpt, roles.manifest, classes, seen, future_classes, all_options=True,task=task,round_id=round_id)
    write_json(out/'discovery_protocol.json', dict(
        version=('guarded_fit_offer_discovery_v1' if guarded else 'functional_fit_offer_discovery_v1' if functional else 'fit_offer_discovery_v1'), structural_options=options,
        rule='class ascending; eligible receiver with most next-task BASE rows, tie lowest ID; best qualified donor FIT Wilson LCB, tie lowest ID; unique receiver',
        quality=Protocol().locked(), read_roles=('donor current-task and receiver historical CALIBRATION FIT only' if guarded else 'current-task CALIBRATION FIT only'),
        functional_filter=('necessary maximum activation >= existing min_recall .95 on donor-positive FIT using receiver head features; margin>0, confidence<1, valid sketch; no threshold tuning' if functional else None),
        no_guard_selection_or_holdout_access=True, validation_opened=False, final_test_opened=False,
        guard_fit_filter=('same select_guard objective on FIT only; require target activation >= .95, per-class receiver FAR <= .001, pooled break=0; discard FIT thresholds and select final thresholds independently on SELECTION' if guarded else None),
        reused_calibration_is_development=True, no_replacement_after_guard_selection=True))
    evidence={};audited=[];heads={}
    for donor in sorted({p['donor'] for p in options}):
        model, detector = _make_denice_client_model(ckpt, donor, device, router_mode='multiclass_balanced')
        pool=current_pool(roles,donor,'calibration',classes)
        positions=stratified_roles(pool,ROUTE_RULES['seed']+donor)['fit']
        y=pool['y'][positions]
        pred=Predictor(seen,device,batch_size)(model,detector,pool['X'][positions])
        for c in sorted({p['class_id'] for p in options if p['donor']==donor}):
            counts=dict(positive=int((y==c).sum()),predicted=int((pred==c).sum()),correct_positive=int(((pred==c)&(y==c)).sum()))
            entry=dict(donor=donor,class_id=c,counts=counts,fit_rows=pool['rows'][positions].tolist())
            try:entry.update(qualified=True,quality_lcb=quality(counts,Protocol().validate()))
            except Rejected as exc:entry.update(qualified=False,reason=exc.reason,detail=exc.detail)
            evidence[(donor,c)]=entry;audited.append(entry)
            if functional and entry['qualified']:
                w,b=effective_linear(model,'fc2')
                heads[(donor,c)]=dict(X=pool['X'][positions[y==c]].copy(),rows=pool['rows'][positions[y==c]].copy(),
                                     weight=w[c].numpy().copy(),bias=float(b[c]))
        write_json(out/'donor_fit_offers.json',audited)
        print(f'FIT offer discovery: donor={donor}, classes={[v[1] for v in evidence if v[0]==donor]}',flush=True)
        del model,detector,pool,pred,y;gc.collect()
    functional_evidence={};guarded_evidence={};guarded_rows=[]
    if guarded:
        meta=json.loads((roles.source/'metadata.json').read_text(encoding='utf-8'))
        tasks={int(t):list(map(int,v)) for t,v in meta['task_structure']['task_classes'].items()}
        guarded_dir=out/'guarded_fit_signals';guarded_dir.mkdir()
        receiver_fit_manifests={}
    if functional:
        functional_rows=[];signals_dir=out/'functional_fit_signals';signals_dir.mkdir()
        for receiver in sorted({p['receiver'] for p in options if evidence[(p['donor'],p['class_id'])]['qualified']}):
            model,detector=_make_denice_client_model(ckpt,receiver,device,router_mode='multiclass_balanced')
            receiver_fit=historical_fit_pool(roles,receiver,tasks,current_task=task) if guarded else None
            if guarded:
                receiver_fit_manifests[str(receiver)]=dict(rows=receiver_fit['rows'].tolist(),
                    class_counts={int(c):int(n) for c,n in zip(*np.unique(receiver_fit['y'],return_counts=True))})
                write_json(out/'receiver_fit_manifest.json',receiver_fit_manifests)
            receiver_bases={}
            for pair in [p for p in options if p['receiver']==receiver and evidence[(p['donor'],p['class_id'])]['qualified']]:
                donor,c=pair['donor'],pair['class_id'];head=heads[(donor,c)]
                try:
                    base=receiver_signals(model,detector,head['X'],seen,task,c,batch_size,device)
                    logits=base['imported_features']@head['weight']+head['bias']
                    margin=logits-base['local_best_import_context']
                    sketch=SharedSketch(tuple(ckpt['config']['input_shape']),16,'FIT preflight; no transported signature')
                    _,valid=sketch.features(head['X'])
                    possible=valid & (margin>0) & (base['local_confidence']<1.)
                    rate=float(possible.mean())
                    entry=dict(receiver=receiver,donor=donor,class_id=c,fit_positive_rows=len(possible),
                        possible_activations=int(possible.sum()),maximum_activation_rate=rate,
                        margin_quantiles=np.quantile(margin,[0,.25,.5,.75,1]).tolist(),
                        qualified=rate>=HeadContract().min_recall,necessary_condition_only=True)
                    np.savez_compressed(signals_dir/f'receiver_{receiver}_donor_{donor}_class_{c}.npz',
                        row_id=head['rows'],margin=margin,local_confidence=base['local_confidence'],signature_valid=valid)
                    if guarded and entry['qualified']:
                        if c not in receiver_bases:
                            receiver_bases[c]=receiver_signals(model,detector,receiver_fit['X'],seen,task,c,batch_size,device)
                        z,valid_proto=sketch.features(head['X'])
                        proto,var,support=prototype_summary(z,valid_proto)
                        route=ProtectedRoute(dict(signature=sketch.manifest(),tau=1.,gamma=0.,beta=0.,class_id=c),
                                             proto,var,head['weight'],head['bias'])
                        rsignals=route.signals(receiver_bases[c],receiver_fit['X'])
                        dsignals=route.signals(base,head['X'])
                        joined={k:np.concatenate([rsignals[k],dsignals[k]]) for k in rsignals}
                        labels=np.r_[receiver_fit['y'],np.full(len(head['X']),c,dtype=np.int64)]
                        origins=np.r_[receiver_fit['origin_client'],np.full(len(head['X']),donor,dtype=np.int64)]
                        choice=select_guard(joined,labels,origins,receiver,c,route.metadata)
                        guard_rate=choice['target_activations']/choice['positive_rows']
                        guard_entry=dict(receiver=receiver,donor=donor,class_id=c,qualified=guard_rate>=HeadContract().min_recall,
                            fit_recall=guard_rate,guard_fit=choice,thresholds_used_for_final_packet=False)
                        np.savez_compressed(guarded_dir/f'receiver_{receiver}_donor_{donor}_class_{c}.npz',
                            y_true=labels,origin_client=origins,row_id=np.r_[receiver_fit['rows'],head['rows']],
                            **{k:joined[k] for k in ('local_pred','signature_score','signature_valid','margin','local_confidence')})
                        guarded_evidence[(receiver,donor,c)]=guard_entry;guarded_rows.append(guard_entry)
                        del z,valid_proto,rsignals,dsignals,joined,labels,origins
                    del base,logits,margin,valid,possible
                except Rejected as exc:entry=dict(receiver=receiver,donor=donor,class_id=c,qualified=False,reason=exc.reason)
                functional_evidence[(receiver,donor,c)]=entry;functional_rows.append(entry)
            write_json(out/'functional_fit_offers.json',functional_rows)
            if guarded:write_json(out/'guarded_fit_offers.json',guarded_rows)
            print(f'Functional FIT preflight: receiver={receiver}',flush=True)
            del model,detector,receiver_fit,receiver_bases;gc.collect()
    chosen=[];used=set()
    for c in sorted(classes):
        qualified=[p for p in options if p['class_id']==c and p['receiver'] not in used and evidence[(p['donor'],c)]['qualified']]
        coverage[c]['donor_fit_qualified_options']=len(qualified)
        if functional:
            qualified=[p for p in qualified if functional_evidence[(p['receiver'],p['donor'],c)]['qualified']]
        coverage[c]['functional_fit_qualified_options']=len(qualified)
        if guarded:
            qualified=[p for p in qualified if guarded_evidence.get((p['receiver'],p['donor'],c),{}).get('qualified',False)]
        coverage[c]['qualified_options']=len(qualified)
        if not qualified:
            coverage[c]['reason']=('no guarded FIT-qualified offer' if guarded and coverage[c]['functional_fit_qualified_options'] else
                'no functional FIT-qualified offer' if functional and coverage[c]['donor_fit_qualified_options'] else
                'no FIT-qualified donor among structural options') if coverage[c]['structural_options'] else coverage[c]['reason']
            continue
        ordered=sorted(qualified,key=lambda p:(-p['next_task_base_rows'],p['receiver'],-evidence[(p['donor'],c)]['quality_lcb'],p['donor']))
        selected=0
        for pair in ordered:
            if pair['receiver'] in used:continue
            pair=dict(pair,discovery_donor_quality_lcb=evidence[(pair['donor'],c)]['quality_lcb'])
            chosen.append(pair);used.add(pair['receiver']);selected+=1
            if selected>=max_pairs_per_class:break
    return chosen,coverage


def run_active_pair_calibration(checkpoint,data_dir,roles_dir,out,device='cpu',batch_size=512,fit_discovery=False,functional_fit_discovery=False,guard_fit_discovery=False,
                                task=4,round_id=19,max_pairs_per_class=1):
    from eval_checkpoint import _make_denice_client_model
    from fed_learning.data.denice_clean_roles import CleanRoleData,file_sha256
    from fed_learning.training.checkpoint_state import snapshot_denice_state,snapshot_context_detector
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('New output directory required')
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'completion.json',dict(completed_execution=False,validation_opened=False,final_test_opened=False))
    started=time.perf_counter()
    try:
        if task not in range(5) or max_pairs_per_class<1:raise ValueError('Active calibration requires task0..4 and positive pair budget')
        ckpt,hashes=load_input(checkpoint,task,round_id);config=ckpt['config']
        if config.get('denice_cl_method')!='legacy' or config['denice_similarity_threshold']!=.8:raise Rejected('LEGACY_XI8_REQUIRED')
        roles=CleanRoleData(roles_dir,source_data_dir=data_dir);role_hash=file_sha256(roles.root/'role_manifest.json')
        if config['denice_data_roles_sha256']!=role_hash:raise Rejected('ROLE_LOCK_CHANGED')
        meta=json.loads((roles.source/'metadata.json').read_text(encoding='utf-8'))
        tasks={int(t):list(map(int,v)) for t,v in meta['task_structure']['task_classes'].items()}
        seen=sorted(c for t in range(task+1) for c in tasks[t]);classes=tasks[task]
        write_json(out/'input_lock.json',dict(**hashes,role_manifest_sha256=role_hash,
            base_method='legacy',xi=.8,task=task,round=round_id,validation_opened=False,final_test_opened=False))
        functional_fit_discovery=functional_fit_discovery or guard_fit_discovery
        fit_discovery=fit_discovery or functional_fit_discovery
        if fit_discovery:
            pairs,coverage=discover_fit_pairs(ckpt,roles,classes,seen,tasks[task+1],out,device,batch_size,
                functional=functional_fit_discovery,guarded=guard_fit_discovery,task=task,round_id=round_id,max_pairs_per_class=max_pairs_per_class)
        else:
            pairs,coverage=structural_pairs(ckpt,roles.manifest,classes,seen,tasks[task+1],task=task,round_id=round_id)
        write_json(out/'pair_lock.json',dict(pairs=pairs,coverage=coverage,
            rule=('guarded FIT offer discovery v1; discard FIT thresholds, freeze pairs before SELECTION/HOLDOUT' if guard_fit_discovery else
                  'functional FIT offer discovery v1; full schedule frozen before guard SELECTION/HOLDOUT' if functional_fit_discovery else
                  'FIT offer discovery v1; full schedule frozen before guard SELECTION/HOLDOUT' if fit_discovery else
                  'class ascending, first eligible receiver/donor IDs; unique receiver per class; frozen before all predictions'),
            donor_quality_not_yet_scored=not fit_discovery,max_pairs_per_class=max_pairs_per_class,
            validation_opened=False,future_access='locked next-task BASE counts only, no next-task rows'))
        protocol=dict(version=('appliance_active_pair_guarded_fit_discovery_v1' if guard_fit_discovery else
            'appliance_active_pair_functional_fit_discovery_v1' if functional_fit_discovery else
            'appliance_active_pair_fit_discovery_v1' if fit_discovery else 'appliance_active_pair_historical_calibration_v1'),**hashes,task=task,round=round_id,
            base_method='legacy',xi=.8,role_manifest_sha256=role_hash,sketch_dimension=16,
            donor_offer='current-task CALIBRATION FIT only, quality Wilson LCB>=.5',
            prototype='donor current-task FIT positives only; shared16',
            selection='historical receiver + current donor CALIBRATION SELECTION; per-class FAR .001 and break0',
            acceptance=asdict(HeadContract()),receiver_per_class_holdout_far_budget=.001,
            no_replacement_on_failure=True,validation_opened=False,final_test_opened=False,device=device,
            batch_size=batch_size,versions=dict(torch=torch.__version__,numpy=np.__version__),
            calibration_replay_free_proof=False)
        write_json(out/'protocol_lock.json',protocol)
        outcomes=[]
        for pair in pairs:
            cid,donor,c=pair['receiver'],pair['donor'],pair['class_id']
            target=out/f'receiver_{cid}_class_{c}';target.mkdir()
            write_json(target/'pair_lock.json',pair)
            print(f'Active-pair calibration: class={c}, donor={donor}, receiver={cid}, Task{task+1} BASE={pair["next_task_base_rows"]}',flush=True)
            try:
                model,router=_make_denice_client_model(ckpt,cid,device,router_mode='multiclass_balanced')
                donor_model,donor_router=_make_denice_client_model(ckpt,donor,device,router_mode='multiclass_balanced')
                current={i:current_pool(roles,i,'calibration',classes) for i in (cid,donor)}
                splits={i:stratified_roles(pool,ROUTE_RULES['seed']+i) for i,pool in current.items()}
                locked={str(i):{r:dict(rows=current[i]['rows'][positions].tolist()) for r,positions in split.items()} for i,split in splits.items()}
                write_json(target/'current_calibration_split.json',locked)
                views={cid:calibration_views(roles,cid,tasks,locked,True,current_task=task),donor:calibration_views(roles,donor,tasks,locked,False,current_task=task)}
                calibration={str(i):{r:dict(rows=p['rows']) for r,p in client.items()} for i,client in views.items()}
                write_json(target/'calibration_manifest.json',calibration)
                fit=views[donor]['fit'];fit_pred=Predictor(seen,device,batch_size)(donor_model,donor_router,fit['X'])
                counts=dict(positive=int((fit['y']==c).sum()),predicted=int((fit_pred==c).sum()),correct_positive=int(((fit_pred==c)&(fit['y']==c)).sum()))
                try:lcb=quality(counts,Protocol().validate())
                except Rejected as exc:
                    write_json(target/'donor_fit_quality.json',dict(counts=counts,qualified=False,reason=exc.reason));raise
                write_json(target/'donor_fit_quality.json',dict(counts=counts,quality_lcb=lcb,qualified=True))
                sketch=SharedSketch(tuple(config['input_shape']),16,file_sha256(roles.source/'metadata.json'))
                z,valid=sketch.features(fit['X'][fit['y']==c]);prototype,variance,support=prototype_summary(z,valid)
                weight,bias=effective_linear(donor_model,'fc2')
                route=ProtectedRoute(dict(kind='protected_imported_route',version=PORTABLE_RULES['version'],
                    receiver=cid,donor=donor,class_id=c,task=task,tau=1.,gamma=0.,beta=0.,
                    receiver_feature_hash=boundary_hash(model,True),signature=sketch.manifest(),support=support,
                    checkpoint_hashes=hashes,role_manifest_sha256=role_hash,
                    split_manifest_sha256=file_sha256(target/'current_calibration_split.json'),
                    historical_calibration_manifest_sha256=file_sha256(target/'calibration_manifest.json'),
                    guard_calibration_version=protocol['version'],self_confidence=PORTABLE_RULES['self_confidence']),
                    prototype,variance,weight[c].numpy(),float(bias[c]))
                joblib.dump({cid:snapshot_context_detector(router),donor:snapshot_context_detector(donor_router)},target/'fitted_current_task_routers.joblib',compress=3)
                del donor_model,donor_router,current,fit,fit_pred,z,valid,weight,bias
                selection=combine([views[i]['selection'] for i in (cid,donor)])
                base=receiver_signals(model,router,selection['X'],seen,task,c,batch_size,device);sig=route.signals(base,selection['X'])
                choice=select_guard(sig,selection['y'],selection['origin_client'],cid,c,route.metadata)
                np.savez_compressed(target/'selection_signals.npz',y_true=selection['y'],origin_client=selection['origin_client'],row_id=selection['rows'],
                    **{k:sig[k] for k in ('local_pred','signature_score','signature_valid','margin','local_confidence')})
                route.metadata.update(tau=choice['tau'],gamma=choice['gamma'],beta=choice['beta'])
                packet=route.packet();(target/'candidate.bin').write_bytes(packet)
                write_json(target/'decision_lock.json',dict(selected=choice,candidate_packet_sha256=hashlib.sha256(packet).hexdigest(),
                    holdout_signals_scored=False,validation_opened=False,source_router_sha256=file_sha256(target/'fitted_current_task_routers.joblib')))
                del selection,base,sig
                acceptance=combine([views[i]['holdout'] for i in (cid,donor)])
                base=receiver_signals(model,router,acceptance['X'],seen,task,c,batch_size,device);sig=route.signals(base,acceptance['X']);decision=route.decisions(sig)
                reports=summarize(acceptance,decision,base,cid,c)
                class_gate=all(v['false_activation_rate']<=.001 for v in reports['receiver_by_class'].values())
                compiled=compile_guarded_head(model,router,packet,cid,seen,sketch.preprocessing_sha256)
                if class_gate:
                    installed,detector,registry,transaction=install_guarded_head(model,router,compiled,GuardedHeadRegistry(),acceptance,seen,device,batch_size)
                else:
                    installed,detector,registry=model,router,GuardedHeadRegistry()
                    transaction=dict(status='rejected',applied=False,reason='HISTORICAL_PER_CLASS_ACCEPTANCE_FAILED')
                write_json(target/'transaction.json',transaction)
                write_json(target/'holdout.json',dict(metrics=reports,per_class_far_passed=class_gate,capability_enabled=transaction['applied'],
                    class_counts={int(k):int(v) for k,v in zip(*np.unique(acceptance['y'],return_counts=True))}))
                np.savez_compressed(target/'holdout_predictions.npz',y_true=acceptance['y'],origin_client=acceptance['origin_client'],
                    row_id=acceptance['rows'],pred=decision['pred'],local_pred=base['local_pred'],activated=decision['activated'])
                if transaction['applied']:
                    torch.save(dict(config=config,receiver=cid,seen_classes=seen,client_model_states={cid:installed.state_dict()},
                        client_algorithm_states={cid:{'denice':snapshot_denice_state(installed,detector)}},guarded_head_entries=registry.entries),target/'guarded_receiver.pt')
                result=dict(**pair,installed=bool(transaction['applied']),reason=transaction.get('reason'),
                    holdout_recall=reports['pooled']['recall'],holdout_receiver_far=reports['receiver']['false_activation_rate'],
                    holdout_break=reports['pooled']['break_count'],packet_bytes=len(packet),donor_quality_lcb=lcb)
                print(f'Active-pair outcome: receiver={cid}, class={c}, installed={result["installed"]}, recall={result["holdout_recall"]}',flush=True)
            except Rejected as exc:
                result=dict(**pair,installed=False,reason=exc.reason,detail=exc.detail)
                print(f'Active-pair rejected: receiver={cid}, class={c}, reason={exc.reason}',flush=True)
            outcomes.append(result);write_json(target/'completion.json',dict(completed_execution=True,validation_opened=False,final_test_opened=False,**result))
            write_json(out/'partial_outcomes.json',outcomes)
            # Release model/data graphs after each structurally frozen pair.
            model=router=donor_model=donor_router=installed=detector=registry=views=acceptance=base=sig=decision=None
            gc.collect()
        result=dict(completed_execution=True,outcomes=outcomes,coverage=coverage,accepted_count=sum(r['installed'] for r in outcomes),
            validation_opened=False,final_test_opened=False,training_performed=False,passed_full_feasibility=False,elapsed_seconds=time.perf_counter()-started)
        write_json(out/'completion.json',result);return result
    except Exception as exc:
        write_json(out/'completion.json',dict(completed_execution=False,error_type=type(exc).__name__,error=str(exc),
            validation_opened=False,final_test_opened=False,passed_full_feasibility=False));raise
