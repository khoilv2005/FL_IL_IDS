"""Locked development probes of representations and legitimate donor heads.

No production guard change, test reads, historical raw CAL, or backbone update.
Frozen receiver-feature probes of prior BASE are RETROSPECTIVE diagnostics,
not a claim that those feature statistics existed during the original run.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from appliance.base_sketch_shield import CurrentBaseSketchShield
from appliance.closure import effective_linear
from appliance.config import Rejected
from appliance.current_base_data import CurrentBaseData
from appliance.current_calibration_data import CurrentCalibrationData
from appliance.imported_route import ROUTE_RULES, stratified_roles, unit_rows
from appliance.portable_route import SharedSketch, receiver_signals
from appliance.receiver_aware_discovery import head_offer, maturity_precheck
from appliance.selector import lookup
from appliance.state import complete_hash, digest, write_json
from fed_learning.data.denice_clean_roles import CleanRoleData, file_sha256
from tools.audit_appliance_provenance_acceptance import original_model
from tools.audit_appliance_discriminative_guard import content_hashes

SEED = 20261010
RIDGE = .001
REPS = ('sketch16','sketch32','sketch64','sketch128','preprocessed_input',
        'unit_preprocessed_input','receiver_fc1','unit_receiver_fc1')
OBJECTIVES = ('all_protected', '20_vs_13')
BASE_CAP, VAL_CAP = 2048, 2048


@torch.no_grad()
def fc1(model, x, batch_size):
    modes=[(m,m.training) for m in model.modules()]
    active=copy.deepcopy(model.active_adapters)
    try:
        model.eval();model.clear_active_adapters()
        chunks=[]
        for start in range(0,len(x),batch_size):
            with torch.autocast(device_type='cpu',enabled=False):
                chunks.append(model.penultimate_features(torch.from_numpy(x[start:start+batch_size])).cpu().numpy())
        return np.concatenate(chunks)
    finally:
        model.active_adapters=active
        for m,flag in modes:m.training=flag


def representations(x, model, sketches, batch_size):
    raw=x.reshape(len(x),-1)
    h=fc1(model,x,batch_size)
    result={f'sketch{d}':s.features(x) for d,s in sketches.items()}
    result['preprocessed_input']=(raw,np.ones(len(x),bool))
    result['unit_preprocessed_input']=unit_rows(raw)
    result['receiver_fc1']=(h,np.ones(len(x),bool))
    result['unit_receiver_fc1']=unit_rows(h)
    return result


def moment(z, valid):
    z=z[valid].astype(np.float64)
    if not len(z):raise Rejected('EMPTY_PROBE_FIT')
    mean=z.mean(0);center=z-mean
    return dict(n=len(z),mean=mean,scatter=center.T@center)


def fit_probe(positive, negatives, objective):
    groups={}
    for item in negatives:
        if objective=='20_vs_13' and item['class_id']!=13:continue
        groups.setdefault(item['class_id'],[]).append(item['moment'])
    if not groups:raise Rejected('NO_NEGATIVE_PROBE_SUPPORT')
    parts=[(.5,positive,1.)]
    for values in groups.values():
        n=sum(v['n'] for v in values)
        parts.extend((.5/len(groups)*v['n']/n,v,-1.) for v in values)
    d=len(positive['mean'])
    mu=sum(mass*v['mean'] for mass,v,_ in parts)
    second=sum(mass*(v['scatter']/v['n']+np.outer(v['mean'],v['mean'])) for mass,v,_ in parts)
    scale=np.sqrt(np.maximum(np.diag(second)-mu**2,1e-12))
    gram=np.zeros((d+1,d+1));rhs=np.zeros(d+1)
    for mass,v,label in parts:
        centered=(v['mean']-mu)/scale
        block=np.zeros_like(gram)
        block[:d,:d]=v['scatter']/v['n']/np.outer(scale,scale)+np.outer(centered,centered)
        block[:d,d]=block[d,:d]=centered;block[d,d]=1
        gram+=mass*block;rhs+=mass*label*np.r_[centered,1.]
    theta=np.linalg.solve(gram+np.diag(np.r_[np.full(d,RIDGE),0.]),rhs)
    # Fixed FP64 probe coordinates avoid ill-conditioned conversion to raw
    # FP32 coefficients, especially large FC1 values. Not a production codec.
    return dict(mean=mu,scale=scale,weight=theta[:d],bias=float(theta[d]),
                negative_classes=sorted(groups),dimension=d)


def score(probe, z, valid):
    result=(z.astype(np.float64)-probe['mean'])/probe['scale']@probe['weight']+probe['bias']
    if not np.isfinite(result).all():raise Rejected('NONFINITE_PROBE_SCORE')
    return np.where(valid,result,-1e30)


def threshold_at_95(positive):
    if len(positive)<32:raise Rejected('INSUFFICIENT_SELECTION_POSITIVES')
    ordered=np.sort(positive)
    failures=len(positive)-math.ceil(.95*len(positive))
    return float(np.nextafter(ordered[failures],-np.inf))  # strict score>tau


def metric(pos, pools, tau):
    all_neg=np.concatenate([s for _,_,s in pools])
    hard=np.concatenate([s for _,c,s in pools if c==13])
    def auc(neg):return float(roc_auc_score(np.r_[np.ones(len(pos)),np.zeros(len(neg))],np.r_[pos,neg]))
    classes=sorted({c for _,c,_ in pools})
    per_class={str(c):dict(rows=sum(len(s) for _,k,s in pools if k==c),
        activated=sum(int((s>tau).sum()) for _,k,s in pools if k==c)) for c in classes}
    for value in per_class.values():value['far']=value['activated']/value['rows']
    by_source=[dict(owner=owner,class_id=c,rows=len(s),activated=int((s>tau).sum()),far=float((s>tau).mean())) for owner,c,s in pools]
    recall=float((pos>tau).mean())
    return dict(positive_rows=len(pos),recall=recall,threshold=tau,
        negative_rows=len(all_neg),negative_activated=int((all_neg>tau).sum()),far=float((all_neg>tau).mean()),
        class13_rows=len(hard),class13_activated=int((hard>tau).sum()),class13_far=float((hard>tau).mean()),
        auroc_all=auc(all_neg),auroc_20_vs_13=auc(hard),far_by_class=per_class,far_by_owner_class=by_source,
        max_owner_class_far=max(v['far'] for v in by_source),
        passed_observed_recall_and_FAR=bool(recall>=.95 and all(v['far']<=.001 for v in by_source)),
        class13_gate=bool(recall>=.95 and float((hard>tau).mean())<=.001),
        positive_quantiles=np.quantile(pos,[.05,.5,.95]).tolist(),
        class13_quantiles=np.quantile(hard,[.05,.5,.95]).tolist(),
        recall_95_achieved_on_this_split=recall>=.95,
        population_FAR_certified=False)


def run(a):
    a.out.mkdir(parents=True,exist_ok=False)
    write_json(a.out/'completion.json',dict(completed=False))
    protocol=dict(kind='Feature Separability & Donor Compatibility development audit',receiver=65,import_class=20,
        hard_negative=13,representations=list(REPS),objectives=list(OBJECTIVES),ridge=RIDGE,
        fixed_linear_diagnostic_probes=True,no_MLP=True,guard_modified=False,
        BASE_per_owner_class_cap=BASE_CAP,selection_evaluation_per_owner_class_cap=VAL_CAP,
        BASE_sampling='row ID digest sorted, fixed seed; only checkpoint-owned task support',
        positive_fitting='donor current CAL-FIT only',negative_fitting='peer current BASE moments, chronological simulation',
        validation_exclusion='first 768 rows/class per owner plus matching content across owners',
        split='content hash + seed, 50/50 SELECTION/EVALUATION; same content never crosses splits',
        threshold='SELECTION positive order statistic for >=95% recall; frozen, never retune on EVALUATION',
        choice='SELECTION only: joint recall/FAR pass, then lowest max owner/class FAR, then dimension/name',
        FAR_budget=.001,recall_min=.95,primary_objective='20_vs_13',
        raw_historical_CAL_opened=False,production_enabled=False,final_test_opened=False,
        current_task3_frozen_features_on_old_BASE_are_retrospective=True,
        inference_cost_not_optimized=True,break_gate_not_a_CAL_acceptance_in_this_audit=True)
    write_json(a.out/'protocol_before_data.json',protocol)
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if ckpt['task']!=3 or ckpt['config']['denice_cl_method']!='legacy':raise Rejected('LEGACY_TASK3_REQUIRED')
    graph=next(g for g in json.loads(a.graphs.read_text()) if g['task']==3 and g['round']==ckpt['final_round_id'])
    edge=lookup(graph['alpha_debug'],65)
    peers={int(p):float(w) for p,w in zip(edge['group_ids'],edge['alphas']) if p!=65 and w>0}
    roles=CleanRoleData(a.roles,source_data_dir=a.data)
    role_sha=ckpt['config']['denice_data_roles_sha256']
    if file_sha256(a.roles/'role_manifest.json')!=role_sha:raise Rejected('ROLE_AUTHORITY_CHANGED')
    receiver,router=original_model(ckpt,65); receiver_before=complete_hash(receiver,router)
    owned={};donor_candidates=[];donor_models={}
    for p in sorted(peers):
        state=lookup(ckpt['client_algorithm_states'],p);state=state.get('denice',state)
        owned[p]=CurrentBaseSketchShield.restore(state['appliance_base_sketch_shield_state'])
        count=roles.manifest['clients'][str(p)]['role_class_counts']['base'].get('20',0)
        rank=int(state['neuron_ages']['fc2'][20])
        record=dict(donor=p,alpha=peers[p],owned_BASE_class20_rows=count,fc2_class20_rank=rank,eligible=False)
        if count>0 and rank>=2:
            dm,dr=original_model(ckpt,p)
            pre=maturity_precheck(receiver,router,head_offer(dm,20,digest(dm.state_dict())),3,ckpt['seen_classes'])
            record.update(precheck=pre,eligible=bool(pre['eligible']))
            if pre['eligible']:donor_models[p]=(dm,dr,complete_hash(dm,dr))
        else:record['reason']='no owned class20 support or immature head'
        donor_candidates.append(record)
    write_json(a.out/'donor_authority_before_predictions.json',dict(candidates=donor_candidates,graph=graph['task'],
        other_donors_outside_graph_not_queried=True,model_fit_roles='existing locked roles'))
    if sorted(donor_models)!=[88]:raise Rejected('EXPECTED_SINGLE_DONOR_FIXTURE_CHANGED')
    dm,dr,donor_before=donor_models[88]
    view=CurrentCalibrationData(a.calibration_store,88,3,role_sha)
    pool=view.current_pool(88,'calibration',view.store['task_classes']['3'])
    split=stratified_roles(pool,ROUTE_RULES['seed']+88)
    fit=split['fit'];fit=fit[pool['y'][fit]==20]
    if len(fit)<32:raise Rejected('DONOR_CURRENT_FIT_TOO_SMALL')
    pp=view.store['metadata_sha256']
    sketches={d:SharedSketch(tuple(view.store['input_shape']),d,pp) for d in (16,32,64,128)}
    positive=representations(pool['X'][fit],receiver,sketches,a.batch_size)
    positive_moments={r:moment(*positive[r]) for r in REPS}
    fit_hashes=set(content_hashes(pool['X'][fit]))
    negatives={r:[] for r in REPS};base_access={};negative_fit_rows=[]
    for p,authority in owned.items():
        bv=CurrentBaseData(a.base_store,p,0,role_sha)
        for t in range(4):
            if t:bv.advance(t)
            current=bv.current_pool(p,'base',bv.store['task_classes'][str(t)])
            for c in sorted(set(map(int,current['y']))):
                if c==20 or str(c) not in authority.memory.entries or authority.memory.entries[str(c)]['task']!=t:continue
                ix=np.flatnonzero(current['y']==c)
                keys=[hashlib.sha256(f'{SEED}:{p}:{int(current["rows"][i])}'.encode()).hexdigest() for i in ix]
                ix=ix[np.argsort(keys,kind='stable')[:BASE_CAP]]
                x=current['X'][ix]
                reps=representations(x,receiver,sketches,a.batch_size)
                for r in REPS:negatives[r].append(dict(owner=p,class_id=c,task=t,moment=moment(*reps[r])))
                fit_hashes.update(content_hashes(x))
                negative_fit_rows.append(dict(owner=p,task=t,class_id=c,available=len(keys),sampled=len(ix),
                    row_ids_sha256=hashlib.sha256(current['rows'][ix].astype('<i8').tobytes()).hexdigest()))
            del current
        base_access[str(p)]=bv.access_log
        print(f'Separability BASE summarized peer={p}',flush=True)
    write_json(a.out/'fit_support_lock.json',dict(positive_rows=len(fit),negative_rows=negative_fit_rows,
        positive_row_ids_sha256=hashlib.sha256(pool['rows'][fit].astype('<i8').tobytes()).hexdigest(),
        historical_CAL_reads=0,source_role_sha256=role_sha,BASE_runtime_access=base_access))
    probes={}
    for r in REPS:
        for objective in OBJECTIVES:probes[f'{r}/{objective}']=fit_probe(positive_moments[r],negatives[r],objective)
    write_json(a.out/'probe_models_before_selection.json',probes)
    probe_digest_before=digest(probes)
    # Preselect rows without evaluating scores; all old-panel contents are
    # excluded across owners, not just by their old local row coordinates.
    client_data={i:roles.client_role(i,'validation') for i in sorted({65}|set(peers))}
    scopes={p:set(int(c) for c in shield.memory.entries) for p,shield in owned.items()}
    scopes[65]=set(map(int,ckpt['seen_classes']))
    inspected=set()
    for p,(x,y,_) in client_data.items():
        for c in sorted(scopes[p]|({20} if p==88 else set())):
            inspected.update(content_hashes(x[np.flatnonzero(y==c)[:768]]))
    panels={'selection':[],'evaluation':[]};panel_manifest=[];excluded_counts=[]
    for p,(x,y,rows) in client_data.items():
        for c in sorted(scopes[p]|({20} if p==88 else set())):
            if c==20 and p!=88:continue
            candidates=np.flatnonzero(y==c)[768:]
            hashes=content_hashes(x[candidates])
            buckets={'selection':[],'evaluation':[]};old_hits=fit_hits=0
            for i,h in zip(candidates,hashes):
                if h in inspected:old_hits+=1;continue
                if h in fit_hashes:fit_hits+=1;continue
                # Shared hash grouping across owners prevents duplicate-content leakage.
                key=int(hashlib.sha256(f'{SEED}:{h}'.encode()).hexdigest(),16)
                buckets['selection' if key%2==0 else 'evaluation'].append((i,h))
            excluded_counts.append(dict(owner=p,class_id=c,remaining_after_row_exclusion=len(candidates),
                prior_content_excluded=old_hits,training_content_excluded=fit_hits))
            for role,items in buckets.items():
                items=sorted(items,key=lambda item:(item[1],int(rows[item[0]])))[:VAL_CAP]
                ix=np.array([v[0] for v in items],np.int64)
                panel_manifest.append(dict(owner=p,class_id=c,role=role,rows=len(ix),row_ids=rows[ix],
                    content_hashes=[v[1] for v in items],positive=c==20))
                if len(ix):panels[role].append(dict(owner=p,class_id=c,X=x[ix],rows=rows[ix],hashes=[v[1] for v in items]))
    hashes_by_role={r:{h for v in panels[r] for h in v['hashes']} for r in panels}
    if hashes_by_role['selection']&hashes_by_role['evaluation']:raise Rejected('CONTENT_SPLIT_LEAKAGE')
    if any(hashes_by_role[r]&(fit_hashes|inspected) for r in panels):raise Rejected('USED_CONTENT_IN_NEW_PANEL')
    write_json(a.out/'development_panels_before_scores.json',dict(panels=panel_manifest,exclusions=excluded_counts,
        split_seed=SEED,content_disjoint=True,unused_rows_not_independent_population_claim=True,
        missing_classes=[dict(owner=p,class_id=c,role=role) for p in scopes for c in sorted(scopes[p]) for role in panels
            if not any(v['owner']==p and v['class_id']==c for v in panels[role])]))
    del client_data,pool,positive,negatives
    def collect(role):
        records={key:dict(positive=[],negative=[]) for key in probes}
        head_records=[]
        rw,rb=effective_linear(receiver,'fc2');dw,db=effective_linear(dm,'fc2')
        refs=next(v['precheck']['reference_classes'] for v in donor_candidates if v['donor']==88)
        donor_refs=sorted((set(map(int,dr.episode_classes.get(3,[])))&set(ckpt['seen_classes']))-{20})
        for v in panels[role]:
            reps=representations(v['X'],receiver,sketches,a.batch_size)
            for key,probe in probes.items():
                r=key.split('/')[0];s=score(probe,*reps[r])
                if v['class_id']==20:records[key]['positive'].append(s)
                else:records[key]['negative'].append((v['owner'],v['class_id'],s))
            h=reps['receiver_fc1'][0];hd=fc1(dm,v['X'],a.batch_size)
            transported=h@dw[20].numpy()+float(db[20])
            donor_logit=hd@dw[20].numpy()+float(db[20])
            receiver_margin=transported-(h@rw[refs].numpy().T+rb[refs].numpy()).max(1)
            donor_margin=donor_logit-(hd@dw[donor_refs].numpy().T+db[donor_refs].numpy()).max(1)
            # Diagnostic native predictions; labels are compared AFTER each
            # model/router has predicted, never passed into that predictor.
            rnative=receiver_signals(receiver,router,v['X'],ckpt['seen_classes'],3,20,a.batch_size,'cpu')
            dnative=receiver_signals(dm,dr,v['X'],ckpt['seen_classes'],3,20,a.batch_size,'cpu')
            head_records.append(dict(owner=v['owner'],class_id=v['class_id'],rows=len(h),
                transported_margin=receiver_margin,donor_margin=donor_margin,
                receiver_native_correct=int((rnative['local_pred']==v['class_id']).sum()),
                donor_native_correct=int((dnative['local_pred']==v['class_id']).sum()),
                donor_task_histogram={str(int(t)):int((dnative['legacy_task']==t).sum()) for t in np.unique(dnative['legacy_task'])},
                donor_predicted_class_histogram={str(int(k)):int((dnative['local_pred']==k).sum()) for k in np.unique(dnative['local_pred'])},
                logit_MAE=float(np.abs(transported-donor_logit).mean()),
                feature_cosine_mean=float(np.sum(unit_rows(h)[0]*unit_rows(hd)[0],axis=1).mean())))
        for v in records.values():v['positive']=np.concatenate(v['positive'])
        return records,head_records
    selection,head_selection=collect('selection')
    thresholds={k:threshold_at_95(v['positive']) for k,v in selection.items()}
    selection_metrics={k:metric(v['positive'],v['negative'],thresholds[k]) for k,v in selection.items()}
    # All representations were declared in advance; pick only from SELECTION.
    keys=[k for k in probes if k.endswith('/20_vs_13')]
    chosen=min(keys,key=lambda k:(not selection_metrics[k]['passed_observed_recall_and_FAR'],
        selection_metrics[k]['max_owner_class_far'],probes[k]['dimension'],k))
    head_positive=np.concatenate([v['transported_margin'] for v in head_selection if v['class_id']==20])
    head_tau=threshold_at_95(head_positive)
    donor_head_positive=np.concatenate([v['donor_margin'] for v in head_selection if v['class_id']==20])
    donor_head_tau=threshold_at_95(donor_head_positive)
    write_json(a.out/'configuration_lock_before_evaluation.json',dict(chosen=chosen,thresholds=thresholds,
        selection_metrics=selection_metrics,receiver_head_margin_tau=head_tau,donor_candidates=donor_candidates,
        donor_native_head_margin_tau=donor_head_tau,
        representation_parameters_digest=digest(probes),panel_digest=digest(panel_manifest),
        thresholds_retuned_on_evaluation=False,evaluation_predictions_opened=False,
        no_install_authority=True,chosen_has_selection_gate=selection_metrics[chosen]['passed_observed_recall_and_FAR']))
    evaluation,head_evaluation=collect('evaluation')
    results={k:metric(v['positive'],v['negative'],thresholds[k]) for k,v in evaluation.items()}
    def head_summary(items):
        pos=np.concatenate([v['transported_margin'] for v in items if v['class_id']==20])
        neg=[(v['owner'],v['class_id'],v['transported_margin']) for v in items if v['class_id']!=20]
        positive=[v for v in items if v['class_id']==20]
        native_pos=np.concatenate([v['donor_margin'] for v in positive])
        native_neg=[(v['owner'],v['class_id'],v['donor_margin']) for v in items if v['class_id']!=20]
        return dict(transported_head_at_selection95=metric(pos,neg,head_tau),
            donor_native_head_at_selection95=metric(native_pos,native_neg,donor_head_tau),
            donor_own_forced_context_recall=float((np.concatenate([v['donor_margin'] for v in positive])>0).mean()),
            receiver_transferred_forced_context_recall=float((pos>0).mean()),
            target_head_logit_MAE=sum(v['logit_MAE']*v['rows'] for v in positive)/sum(v['rows'] for v in positive),
            target_feature_cosine_mean=sum(v['feature_cosine_mean']*v['rows'] for v in positive)/sum(v['rows'] for v in positive),
            all_pool_details=[{k:v for k,v in item.items() if k not in ('transported_margin','donor_margin')} for item in items])
    lock=json.loads((a.out/'configuration_lock_before_evaluation.json').read_text(encoding='utf-8'))
    controls=dict(content_selection_evaluation_disjoint=not bool(hashes_by_role['selection']&hashes_by_role['evaluation']),
        evaluation_excludes_fit_and_previously_inspected_content=not bool(hashes_by_role['evaluation']&(fit_hashes|inspected)),
        receiver_unchanged=complete_hash(receiver,router)==receiver_before,
        donor_unchanged=complete_hash(dm,dr)==donor_before,
        no_outside_graph_donor=set(donor_models).issubset(peers),
        historical_CAL_runtime_reads=sum(e['task']!=3 for e in view.access_log),
        only_current_CAL_FIT_used=all(e['task']==3 and e['client_id']==88 for e in view.access_log),
        eval_thresholds_identical_to_lock=thresholds==lock['thresholds'] and chosen==lock['chosen']
            and head_tau==lock['receiver_head_margin_tau'] and donor_head_tau==lock['donor_native_head_margin_tau'],
        probe_models_unchanged=probe_digest_before==digest(probes))
    if not all(v for k,v in controls.items() if k!='historical_CAL_runtime_reads'):raise AssertionError(controls)
    completion=dict(completed=True,selection=selection_metrics,evaluation=results,primary_choice=chosen,
        primary_evaluation=results[chosen],donor_compatibility=dict(selection=head_summary(head_selection),evaluation=head_summary(head_evaluation)),
        donor_candidates=donor_candidates,alternative_eligible_donors=0,controls=controls,
        production_guard_modified=False,backbone_trained=False,final_test_opened=False,
        raw_historical_CAL_opened=False,installed=False,native_smoke_authorized=False,
        fixed_receiver_representation_reconstruction_is_retrospective=True,
        statistical_population_independence_not_claimed=True,
        linear_probe_failure_not_proof_of_nonlinear_inseparability=True,
        primary_pass_not_sufficient_for_CAL_or_native_authorization=True,
        previous_transfer_cost_bytes=6926708,previous_capability_bytes=12835,
        transfer_cost_not_remeasured_in_feature_audit=True,
        calibration_runtime_access=view.access_log,
        projection_matrix_ranks={str(d):int(np.linalg.matrix_rank(s.matrix().astype(np.float64))) for d,s in sketches.items()},
        source_sha256={f:file_sha256(Path(f)) for f in ['tools/audit_appliance_feature_separability.py']})
    write_json(a.out/'completion.json',completion)
    for k,v in results.items():print(f'{k}: recall={v["recall"]:.4%} class13 FAR={v["class13_far"]:.4%} all FAR={v["far"]:.4%}',flush=True)
    print(f'Selection-locked primary={chosen}, observed gate={results[chosen]["passed_observed_recall_and_FAR"]}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('checkpoint','graphs','calibration-store','base-store','roles','data','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=512)
    a=p.parse_args();torch.set_num_threads(4)
    with threadpool_limits(limits=1):run(a)
