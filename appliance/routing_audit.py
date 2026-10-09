"""APPLIANCE routing diagnosis on one frozen pair; no router/gate fitting.

All predictions and raw signals accept x only. Labels are joined afterward for
failure analysis. Imported-context scores are counterfactual diagnostics.
"""
import copy
import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch

from .closure import effective_linear
from .config import Protocol,Rejected
from .evaluate import Predictor,compare
from .runner import load_input
from .selector import current_pool,recorded_graph,select_pair
from .state import changed_state,digest,state_fingerprint,write_json
from .transfer_audit import features,probe,fixed_context_prediction


RULES=dict(version='appliance_routing_audit_v1',no_fit=True,no_install=True,no_final_test=True,
    no_bridge=True,no_threshold_sweep=True,diagnostic_only=True,
    pair_rule='same pair as previous transfer audit; donor quality rechecked on calibration',
    score_semantics='binary cosine softmax is not calibrated confidence',
    historical_retention='unmeasured: current task validation only')


def quantiles(value):
    value=np.asarray(value,dtype=float);value=value[np.isfinite(value)]
    if not len(value):return dict(rows=0)
    return dict(rows=len(value),mean=float(value.mean()),min=float(value.min()),p10=float(np.quantile(value,.10)),
                median=float(np.median(value)),p90=float(np.quantile(value,.90)),max=float(value.max()))


def histogram(value):
    keys,counts=np.unique(value,return_counts=True)
    return {int(k):int(n) for k,n in zip(keys,counts)}


def cosine_scores(router,binary):
    width=max([int(ep) for ep in router.episode_classes]+[int(ep) for ep in router.activation_memory])+1
    scores=np.full((len(binary),width),-np.inf,dtype=np.float64)
    for ep,memory in router.activation_memory.items():
        if len(memory):
            prototype=np.asarray(memory,dtype=np.float32).mean(axis=0)
            denom=np.linalg.norm(binary,axis=1)*np.linalg.norm(prototype)
            scores[:,int(ep)]=(binary@prototype)/np.maximum(denom,1e-8)
    if not np.isfinite(scores).any():scores[:,max(router.episode_classes)]=0.
    return scores


@torch.no_grad()
def routing_signals(model,router,inputs,seen,task,c,batch_size,device):
    """Label-blind, reproduces production pred_hard and captures router inputs."""
    from fed_learning.training.denice_eval import _denice_routed_logits_with_episodes
    modes=[(m,m.training) for m in model.modules()];active=copy.deepcopy(model.active_adapters)
    chunks=[]
    try:
        model.eval()
        for start in range(0,len(inputs),batch_size):
            x=torch.as_tensor(inputs[start:start+batch_size],dtype=torch.float32,device=device)
            acts={k:v.cpu().numpy() for k,v in model.get_context_activations_per_sample(x).items()}
            binary=router.binarize_layer_activations(acts)
            episodes,scores=router.predict_episodes_with_scores(binary)
            episodes=np.asarray(episodes,dtype=np.int64);scores=np.asarray(scores)
            if scores.ndim!=2 or not np.isfinite(scores).all():raise Rejected('INVALID_ROUTER_SCORES')
            logits,routed_ep=_denice_routed_logits_with_episodes(model,x,router,seen,device,inference_policy='pred_hard')
            if not np.array_equal(episodes,routed_ep):raise RuntimeError('Router instrumentation changed routing')
            if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite routed logits')
            idx=np.arange(len(episodes));best=scores[idx,episodes]
            alternatives=scores.copy();alternatives[idx,episodes]=-np.inf
            second=alternatives.max(axis=1)
            target=scores[:,task] if task<scores.shape[1] else np.zeros(len(episodes))
            other=scores.copy()
            if task<other.shape[1]:other[:,task]=-np.inf
            target_rank=1+(scores>target[:,None]).sum(axis=1)
            ties=(scores==target[:,None]).sum(axis=1)
            part=dict(task=episodes,router_score_chosen=best,router_score_import=target,
                router_score_best_alternative_to_import=other.max(axis=1),
                router_import_minus_best_alternative=target-other.max(axis=1),
                router_top1_minus_top2=best-second,router_import_rank=target_rank,
                router_import_score_tie_count=ties,pred=logits.argmax(1).cpu().numpy(),
                local_head_score_chosen_context=logits.max(1).values.cpu().numpy(),
                original_class_logit_routed=logits[:,c].cpu().numpy(),
                binary_active_bits=binary.sum(axis=1),binary=binary,
                continuous_context=np.concatenate([acts[k] for k in ('conv1','conv2','conv3','gru')],axis=1),
                all_router_scores=scores)
            if router.router_mode=='binary_cosine':
                raw=cosine_scores(router,binary)
                if not np.array_equal(raw.argmax(1),episodes):raise RuntimeError('Raw cosine reproduction mismatch')
                alternatives=raw.copy();alternatives[:,task]=-np.inf
                part.update(raw_cosine_import=raw[:,task],raw_cosine_chosen=raw[idx,episodes],
                    raw_cosine_import_minus_best_alternative=raw[:,task]-alternatives.max(axis=1),
                    all_raw_cosines=raw)
            chunks.append(part)
    finally:
        model.active_adapters=active
        for module,training in modes:module.training=training
    if not chunks:raise Rejected('EMPTY_AUDIT_INPUT')
    return {key:np.concatenate([v[key] for v in chunks]) for key in chunks[0]}


def inventory(router,c,task):
    entries=[];prototypes=[];eps=[]
    for ep in sorted(set(router.episode_classes)|set(router.activation_memory)):
        memory=np.asarray(router.activation_memory.get(ep,[]))
        classes=list(map(int,router.episode_classes.get(ep,[])))
        item=dict(task=int(ep),classes=classes,class_entry_present=c in classes,
            memory_rows=len(memory),feature_width=int(memory.shape[1]) if memory.ndim==2 else None,
            reference_input_rows=len(router.reference_input_memory.get(ep,[])))
        if memory.ndim==2 and len(memory):
            pred,scores=router.predict_episodes_with_scores(memory)
            item.update(memory_self_route_accuracy=float(np.mean(pred==ep)),memory_route_histogram=histogram(pred))
            prototypes.append(memory.astype(np.float32).mean(axis=0));eps.append(int(ep))
        entries.append(item)
    pair_cosines=[]
    for a in range(len(prototypes)):
        for b in range(a+1,len(prototypes)):
            norm=np.linalg.norm(prototypes[a])*np.linalg.norm(prototypes[b])
            pair_cosines.append(dict(task_a=eps[a],task_b=eps[b],cosine=float(prototypes[a]@prototypes[b]/max(norm,1e-8))))
    feature_width=next((int(np.asarray(v).shape[1]) for v in router.activation_memory.values() if np.asarray(v).ndim==2),0)
    active_mask=getattr(router,'routing_feature_mask',None)
    if active_mask is None:active_mask=getattr(router,'stable_feature_mask',None)
    mask=np.asarray(active_mask,dtype=bool) if active_mask is not None else np.ones(feature_width,dtype=bool)
    return dict(router_mode=router.router_mode,feature_width=feature_width,active_routing_features=int(mask.sum()),
        excluded_routing_features=int((~mask).sum()),historical_feature_freshness_independently_verified=False,task_entry_present=bool(router.episode_classes.get(task)),
        task_memory_present=bool(len(router.activation_memory.get(task,[]))),
        class_present_in_import_task=c in router.episode_classes.get(task,[]),
        class_present_any_task=any(c in values for values in router.episode_classes.values()),
        memory_has_per_row_class_labels=False,imported_route_registry_present=False,
        imported_route_registry_note='prior probe extends class mask only; no signature registration implemented',
        freshness=dict(fresh=router.router_state_fresh,last_task=router.router_last_refresh_task,
                       last_round=router.router_last_refresh_round,stale_reason=router.router_stale_reason),
        calibration_signature=router.calibration_signature(),thresholds=router.binarize_thresholds,
        context_masks_used_in_binary_cosine=False if router.router_mode=='binary_cosine' else None,
        stable_feature_mask_present=getattr(router,'stable_feature_mask',None) is not None,
        routing_feature_mask_present=getattr(router,'routing_feature_mask',None) is not None,
        entries=entries,prototype_pair_cosines=pair_cosines)


def prior_read(path,name):
    if path is None:return None
    path=Path(path)
    if path.is_dir():return (path/name).read_bytes()
    with zipfile.ZipFile(path) as archive:
        matches=[n for n in archive.namelist() if n==name or n.endswith('/'+name)]
        # Combined results ZIP has multiple locks: pick the Task 4 transfer prefix.
        preferred=[n for n in matches if n.startswith('task4_transfer/')]
        if preferred:matches=preferred
        if len(matches)!=1:raise Rejected('AMBIGUOUS_PRIOR_ARTIFACT',name)
        return archive.read(matches[0])


def write_rows(path,signals,labels,origins,row_ids,patch_pred,fixed_pred,patch_logits,local_fixed,original_fixed,donor_task):
    """Labels join after inference. They never influence scores or predictions."""
    base={k:v for k,v in signals.items() if v.ndim==1}
    base.update(origin_client=origins,row_id=row_ids,true_class=labels,head_only_pred=patch_pred,
        fixed_import_context_pred=fixed_pred,patch_head_score_import_context=patch_logits,
        local_head_score_import_context=local_fixed,original_class_logit_import_context=original_fixed,
        patch_margin_within_import_context=patch_logits-local_fixed,
        patch_margin_vs_local_chosen_context=patch_logits-signals['local_head_score_chosen_context'],
        patch_activated=signals['task']==signals['import_task_id'],donor_router_task=donor_task)
    # import_task_id is injected explicitly before this writer, independent of labels.
    base.pop('import_task_id',None)
    for column in range(signals['all_router_scores'].shape[1]):base[f'router_score_task_{column}']=signals['all_router_scores'][:,column]
    for column in range(signals.get('all_raw_cosines',np.empty((len(labels),0))).shape[1]):
        base[f'raw_cosine_task_{column}']=signals['all_raw_cosines'][:,column]
    names=list(base)
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(names)
        for pos in range(len(labels)):
            writer.writerow([base[name][pos].item() if np.isfinite(base[name][pos]) else '' for name in names])


def run_routing_audit(checkpoint,role_dir,data_dir,out,prior=None,task=4,round_id=19,receiver_id=19,
                      donor_id=71,class_id=24,device='cpu',batch_size=512,allow_cgofed_fixture=False):
    from eval_checkpoint import _make_denice_client_model
    from fed_learning.data.denice_clean_roles import CleanRoleData,file_sha256
    from fed_learning.training.denice_eval import _mask_logits_to_classes
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'completion.json',dict(completed_execution=False,passed_feasibility=False,no_install=True))
    try:
        if task not in range(6) or round_id<0 or batch_size<1:raise ValueError('Invalid audit scope')
        protocol=Protocol().validate();ckpt,hashes=load_input(checkpoint,task,round_id);config=ckpt['config']
        method=config.get('denice_cl_method','legacy')
        if method!='legacy' and not (method=='cgofed' and allow_cgofed_fixture):raise Rejected('LEGACY_BASELINE_REQUIRED',method)
        roles=CleanRoleData(role_dir,source_data_dir=data_dir);role_hash=file_sha256(roles.root/'role_manifest.json')
        if config.get('denice_data_roles_sha256')!=role_hash:raise Rejected('BACKBONE_ROLE_LOCK_MISMATCH')
        if tuple(config['input_shape'])!=tuple(roles.manifest['input_shape']):raise Rejected('PREPROCESSING_SHAPE_MISMATCH')
        if config.get('denice_max_train_samples_per_client') or config.get('denice_max_clients') not in (None,100):raise Rejected('TRUNCATED_BASE_TRAINING_NOT_SUPPORTED')
        metadata=json.loads((roles.source/'metadata.json').read_text(encoding='utf-8'))
        task_classes={int(k):list(map(int,v)) for k,v in metadata['task_structure']['task_classes'].items()}
        seen=sorted({c for t,values in task_classes.items() if t<=task for c in values});classes=task_classes[task]
        if sorted(map(int,ckpt['seen_classes']))!=seen:raise Rejected('SEEN_SCOPE_MISMATCH')
        if class_id not in classes:raise Rejected('CLASS_OUTSIDE_CURRENT_TASK')
        groups,alphas=recorded_graph(ckpt,task,round_id);predict=Predictor(seen,device,batch_size)
        if prior is not None:
            previous=json.loads(prior_read(prior,'protocol_lock.json'))
            for key,value in {**hashes,'task':task,'round':round_id,'role_manifest_sha256':role_hash}.items():
                if previous.get(key)!=value:raise Rejected('PRIOR_PROTOCOL_MISMATCH',key)
            chosen=json.loads(prior_read(prior,'pair_selection.json'))['selected']
            if [chosen['receiver'],chosen['donor'],chosen['class_id']]!=[receiver_id,donor_id,class_id]:raise Rejected('PRIOR_PAIR_MISMATCH')
        root=Path(__file__).resolve().parents[1]
        source_files=list((root/'appliance').glob('*.py'))+[root/name for name in (
            'eval_checkpoint.py','fed_learning/models/nice_model.py','fed_learning/models/denice_model.py',
            'fed_learning/servers/nice_server.py','fed_learning/training/denice_eval.py',
            'fed_learning/training/checkpoint_state.py','fed_learning/data/denice_clean_roles.py')]
        import sklearn
        lock=dict(**RULES,**hashes,task=task,round=round_id,receiver=receiver_id,donor=donor_id,class_id=class_id,
            xi=config.get('denice_similarity_threshold'),base_method=method,diagnostic_fixture=method!='legacy',
            role_manifest_sha256=role_hash,prior_artifact=str(prior) if prior is not None else None,
            device=device,batch_size=batch_size,versions=dict(torch=torch.__version__,numpy=np.__version__,sklearn=sklearn.__version__),
            source_sha256={p.relative_to(root).as_posix():file_sha256(p) for p in source_files},
            roles='donor current calibration for quality recheck; both clients current validation for offline diagnosis',
            prediction_contract='x only; labels added after all predictions; no rank/margin/threshold selection')
        write_json(out/'protocol_lock.json',lock)
        def make_model(cid):
            model,router=_make_denice_client_model(ckpt,cid,device)
            state=ckpt['client_model_states'].get(cid,ckpt['client_model_states'].get(str(cid)))
            restored=model.state_dict()
            if set(restored)!=set(state) or any(not torch.equal(v.detach().cpu(),state[n]) for n,v in restored.items()):raise Rejected('INEXACT_MODEL_RESTORE',str(cid))
            if any(int(ep)>task for ep in router.activation_memory):raise Rejected('FUTURE_ROUTER_MEMORY')
            return model,router
        selected,ledger,offers=select_pair(ckpt,roles,task,classes,groups,make_model,predict,protocol,
            (receiver_id,donor_id,class_id),eligibility_path=out/'pair_eligibility.json')
        write_json(out/'pair_selection.json',dict(selected=selected,offers=offers,ledger=ledger.to_dict(),same_pair_recheck=True))
        receiver,route=make_model(receiver_id);donor,donor_route=make_model(donor_id)
        original={name:state_fingerprint(model,router) for name,model,router in (
            ('receiver',receiver,route),('donor',donor,donor_route))}
        if any(getattr(model,'continual_head',None) is not None or getattr(model,'local_classifier',None) is not None for model in (receiver,donor)):
            raise Rejected('UNSUPPORTED_AUXILIARY_CLASSIFIER')
        inventories=dict(receiver=inventory(route,class_id,task),donor=inventory(donor_route,class_id,task),
            feature_space=dict(same_fc1_shape=tuple(receiver.fc1.weight.shape)==tuple(donor.fc1.weight.shape),
                same_fc2_shape=tuple(receiver.fc2.weight.shape)==tuple(donor.fc2.weight.shape),same_calibration_signature=route.calibration_signature()==donor_route.calibration_signature(),
                warning='same dimensions/calibration alone do not establish shared encoder coordinates; receiver/donor weights differ'))
        write_json(out/'router_inventory.json',inventories)
        receiver_pool=current_pool(roles,receiver_id,'validation',classes);donor_pool=current_pool(roles,donor_id,'validation',classes)
        inputs=np.concatenate([receiver_pool['X'],donor_pool['X']]);labels=np.concatenate([receiver_pool['y'],donor_pool['y']])
        origins=np.concatenate([np.full(len(receiver_pool['y']),receiver_id),np.full(len(donor_pool['y']),donor_id)])
        rows=np.concatenate([receiver_pool['rows'],donor_pool['rows']])
        write_json(out/'diagnostic_manifest.json',dict(task=task,role='validation; offline development only',
            origins=[dict(client_id=receiver_id,rows=receiver_pool['rows']),dict(client_id=donor_id,rows=donor_pool['rows'])],
            final_test_rows=0,not_fitting_data=True))
        print(f'Routing audit: receiver={receiver_id}, donor={donor_id}, class={class_id}, rows={len(inputs)}',flush=True)
        signals=routing_signals(receiver,route,inputs,seen,task,class_id,batch_size,device)
        donor_signals=routing_signals(donor,donor_route,inputs,seen,task,class_id,batch_size,device)
        head,bias=effective_linear(donor,'fc2');weight=head[class_id].numpy();b=float(bias[class_id])
        candidate,detector=probe(receiver,route,weight,b,class_id,task)
        patched=predict.records(candidate,detector,inputs)
        if not np.array_equal(patched['task'],signals['task']):raise RuntimeError('Mask extension changed routing')
        fixed_features=features(receiver,inputs,task,batch_size,device)
        fixed_pred=fixed_context_prediction(receiver,route,fixed_features,seen,task,class_id,weight,b)
        local_weights,local_bias=effective_linear(receiver,'fc2')
        raw_logits=torch.nn.functional.linear(torch.as_tensor(fixed_features),local_weights,local_bias)
        allowed=[c for c in route.episode_classes.get(task,[]) if c in seen]
        if not allowed:allowed=seen
        local_fixed=_mask_logits_to_classes(raw_logits,allowed).max(1).values.numpy()
        original_fixed=raw_logits[:,class_id].numpy();patch_logits=fixed_features@weight+b
        reproduced=dict(prior_artifact_present=prior is not None)
        if prior is not None:
            with np.load(io.BytesIO(prior_read(prior,'head_only/diagnostic_predictions.npz')),allow_pickle=False) as saved:
                required=dict(origin_client=origins,row_id=rows,y_true=labels,baseline=signals['pred'],
                    probe=patched['pred'],donor=donor_signals['pred'],routed_task=signals['task'])
                checks={name:np.array_equal(saved[name],value) for name,value in required.items()}
                mismatch={name:int(np.count_nonzero(saved[name]!=value)) if saved[name].shape==value.shape else -1 for name,value in required.items()}
            reproduced.update(checks=checks,mismatch_counts=mismatch)
            if not all(checks.values()):
                write_json(out/'reproduction.json',reproduced);raise Rejected('PRIOR_PREDICTION_REPRODUCTION_FAILED')
        write_json(out/'reproduction.json',reproduced)
        signals['import_task_id']=np.full(len(labels),task)
        write_rows(out/'all_rows.csv',signals,labels,origins,rows,patched['pred'],fixed_pred,patch_logits,local_fixed,original_fixed,donor_signals['task'])
        positive=labels==class_id;old=~positive;route_ok=signals['task']==task;fixed_ok=fixed_pred==class_id
        subset={k:v[positive] for k,v in signals.items()}
        write_rows(out/'imported_class_rows.csv',subset,labels[positive],origins[positive],rows[positive],
            patched['pred'][positive],fixed_pred[positive],patch_logits[positive],local_fixed[positive],original_fixed[positive],donor_signals['task'][positive])
        masks=dict(wrong_route_head_would_win=positive&~route_ok&fixed_ok,
            wrong_route_head_would_fail=positive&~route_ok&~fixed_ok,
            correct_route_patch_correct=positive&route_ok&(patched['pred']==class_id),
            correct_route_head_loses=positive&route_ok&(patched['pred']!=class_id))
        counts={key:int(mask.sum()) for key,mask in masks.items()}
        if sum(counts.values())!=int(positive.sum()):raise RuntimeError('Positive buckets are not exhaustive')
        broken=(signals['pred']==labels)&(patched['pred']!=labels)
        rescued=(signals['pred']!=labels)&(patched['pred']==labels)
        fixed_margin=patch_logits-local_fixed
        margins=dict(positive_fixed_context=quantiles(fixed_margin[positive]),
            receiver_current_supported_fixed_context=quantiles(fixed_margin[(origins==receiver_id)&old]),
            broken_old_fixed_context=quantiles(fixed_margin[broken&old]),
            raw_router_positive=quantiles(signals.get('raw_cosine_import_minus_best_alternative',np.full(len(labels),np.nan))[positive]),
            softmax_router_positive=quantiles(signals['router_import_minus_best_alternative'][positive]))
        def score_subset(mask):
            return dict(rows=int(mask.sum()),receiver_route_histogram=histogram(signals['task'][mask]),
                donor_route_histogram=histogram(donor_signals['task'][mask]),
                target_rank_histogram=histogram(signals['router_import_rank'][mask]),
                selected_score=quantiles(signals['router_score_chosen'][mask]),
                import_score=quantiles(signals['router_score_import'][mask]),
                top1_minus_top2=quantiles(signals['router_top1_minus_top2'][mask]))
        metrics=compare(signals['pred'],patched['pred'],labels,class_id,ledger.observed,signals['task'],task)
        report=dict(pair=selected,rows=len(labels),positive_rows=int(positive.sum()),
            failure_buckets=counts,positive_scores=score_subset(positive),receiver_old_scores=score_subset((origins==receiver_id)&old),
            normal_head_metrics=metrics,fixed_context_recall=float(np.mean(fixed_ok[positive])) if positive.any() else None,
            route_reachability=float(np.mean(route_ok[positive])) if positive.any() else None,
            current_receiver_accuracy=dict(before=float(np.mean(signals['pred'][origins==receiver_id]==labels[origins==receiver_id])),
                after=float(np.mean(patched['pred'][origins==receiver_id]==labels[origins==receiver_id]))),
            rescue=int(rescued.sum()),break_count=int(broken.sum()),break_by_class=histogram(labels[broken]),
            break_by_route=histogram(signals['task'][broken]),margins=margins,
            score_warning=RULES['score_semantics'],margin_warning='cross-context logits may not be calibrated; fixed-context margin is separately logged',
            imported_class_mask_extended_only=True,router_metadata_fitted=False,
            validation_is_development=True,threshold_selected=False,donor_inference_required_by_audit=True,
            future_portable_patch_donor_at_inference='not implemented or evaluated',
            no_final_test=True,no_install=True,not_full_method_accuracy=True)
        write_json(out/'routing_summary.json',report)
        # Binary memory already exists: self-reencoding is not attempted and no historical raw data is reopened.
        if signals['binary'].shape==donor_signals['binary'].shape:
            same=signals['binary']==donor_signals['binary']
            masks=[]
            for detector in (route,donor_route):
                feature_mask=getattr(detector,'routing_feature_mask',None)
                if feature_mask is None:feature_mask=getattr(detector,'stable_feature_mask',None)
                masks.append(np.asarray(feature_mask,dtype=bool) if feature_mask is not None else np.ones(same.shape[1],dtype=bool))
            common=masks[0]&masks[1];union=masks[0]|masks[1]
            unions=np.logical_or(signals['binary'],donor_signals['binary']).sum(1)
            intersections=np.logical_and(signals['binary'],donor_signals['binary']).sum(1)
            write_json(out/'feature_space_diagnostics.json',dict(binary_agreement=quantiles(same.mean(1)),
                binary_jaccard=quantiles(intersections/np.maximum(unions,1)),
                common_active_features=int(common.sum()),union_active_features=int(union.sum()),
                agreement_on_common_active=quantiles(same[:,common].mean(1)) if common.any() else None,
                agreement_on_union_active=quantiles(same[:,union].mean(1)) if union.any() else None,
                agreement_warning='full-width agreement includes shared zeroed reserve features',
                warning='computed on shared evaluator validation inputs; not fit or a portable signature',
                feature_width=signals['binary'].shape[1]))
        checks={name:changed_state(original[name],model,router) for name,model,router in (
            ('receiver',receiver,route),('donor',donor,donor_route))}
        write_json(out/'source_state_checks.json',checks)
        if not all(v['unchanged'] for v in checks.values()):raise RuntimeError('Routing audit mutated source model/router')
        write_json(out/'completion.json',dict(completed_execution=True,passed_feasibility=False,stage='complete',
            no_install=True,no_fit=True,final_test_opened=False,diagnostic_fixture=method!='legacy',
            base_method=method,pair=selected,prior_reproduced=prior is not None,
            reason='Frozen routing diagnosis only; no routing signature or threshold chosen'))
        print(f'Routing audit complete: {int((positive&route_ok).sum())}/{int(positive.sum())} imported positives reach context',flush=True)
    except Rejected as exc:
        write_json(out/'completion.json',dict(completed_execution=True,passed_feasibility=False,no_install=True,
            stage='protocol_rejected',reason=exc.reason,detail=exc.detail))
        print(f'Routing audit rejected: {exc.reason}: {exc.detail}',flush=True)
    except Exception as exc:
        write_json(out/'completion.json',dict(completed_execution=False,passed_feasibility=False,no_install=True,
            stage='execution_error',error_type=type(exc).__name__,detail=str(exc)))
        raise
    return json.loads((out/'completion.json').read_text(encoding='utf-8'))
