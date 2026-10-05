"""Retrospective shared competence gate; frozen experts, no test-label fitting."""
import hashlib
import json
from pathlib import Path
import time
import zipfile
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from eval_checkpoint import _make_denice_client_model
from tools.eval_denice_tip_router import write_json,balanced_cap
from tools.eval_denice_peer_coverage import recorded_peers,lookup
from tools.denice_peer_voting import peer_orders,vote
from tools.denice_competence_gate import content_hash,content_role,competence_priors,feature_matrix,gate_decision
from fed_learning.training.denice_eval import _denice_routed_logits_with_episodes,_route_episodes_with_scores
from fed_learning.strategies.incremental.denice_tip_router import encoder_fingerprint


@torch.no_grad()
def expert_features(model,detector,inputs,seen,device,batch_size):
    model.eval();result={name:[] for name in ['pred','task','router_confidence','class_confidence',
        'class_margin','class_entropy','mask_count','task_supported','class_supported']}
    for start in range(0,len(inputs),batch_size):
        xb=inputs[start:start+batch_size].to(device)
        logits,task=_denice_routed_logits_with_episodes(model,xb,detector,seen,device,
                                                     inference_policy='pred_hard')
        check,prob=_route_episodes_with_scores(model,xb,detector)
        if not np.array_equal(task,check):raise RuntimeError('Router-confidence route mismatch')
        prediction=logits.argmax(1).cpu().numpy()
        n=len(task)
        confidence=np.empty(n);margin=np.empty(n);entropy=np.empty(n);counts=np.empty(n)
        supported=np.empty(n,dtype=bool);class_supported=np.empty(n,dtype=bool)
        for t in np.unique(task):
            rows=np.flatnonzero(task==t)
            allowed=sorted(set(detector.episode_classes.get(int(t),[]))&set(seen))
            if not allowed:raise ValueError('Competence benchmark refuses empty-mask fallback')
            values=torch.softmax(logits[torch.as_tensor(rows,device=logits.device)][:,allowed].float(),dim=1).cpu().numpy()
            ordered=np.sort(values,axis=1)
            confidence[rows]=ordered[:,-1]
            margin[rows]=ordered[:,-1]-(ordered[:,-2] if len(allowed)>1 else 0)
            entropy[rows]=-(values*np.log(np.maximum(values,1e-12))).sum(1)/(np.log(len(allowed)) if len(allowed)>1 else 1.)
            counts[rows]=len(allowed);supported[rows]=int(t) in detector.activation_memory
            class_supported[rows]=np.isin(prediction[rows],allowed)
        values=dict(pred=prediction,task=task,router_confidence=prob[np.arange(len(task)),task],
            class_confidence=confidence,class_margin=margin,class_entropy=entropy,mask_count=counts,
            task_supported=supported,class_supported=class_supported)
        for name,value in values.items():result[name].append(np.asarray(value))
    return {name:np.concatenate(value) for name,value in result.items()}


def gate_data(ckpt,data_dir,classes,ids,test_shards,out,limits,seed):
    """Historical local rows restricted by recorded local task evidence.

    Hashes assign calibration/fit/validation roles globally, so equal inputs in
    different client files cannot cross roles. Exact panel inputs are excluded.
    These are gate holdouts, not proof of holdout from original backbone training.
    """
    forbidden={content_hash(row) for shard in test_shards.values() for row in shard['X_test'].numpy()}
    pools={role:{} for role in limits};rows=[];hashes={role:set() for role in limits}
    for cid in ids:
        state=lookup(ckpt['client_algorithm_states'],cid);state=state.get('denice',state)
        detector=state['context_detector'];evidence={int(t['task_id']) for t in (state.get('cgofed_projection_state') or {}).get('tasks',[]) if 'task_id' in t}
        refresh=detector.get('router_last_refresh_task')
        if refresh is not None:evidence.add(int(refresh))
        masks=detector['episode_classes']
        allowed=sorted({int(c) for t in evidence if t in classes
            for c in set(classes[t])&set(lookup(masks,t) or [])})
        if not allowed:raise ValueError(f'{cid}: no proven local historical training support')
        with np.load(Path(data_dir)/f'client_{cid}_train.npz') as data:
            x=np.asarray(data['X_train'],dtype=np.float32);y=np.asarray(data['y_train'],dtype=np.int64)
        candidates=np.flatnonzero(np.isin(y,allowed));rng=np.random.default_rng(seed+1009*cid)
        # Balanced scan caps IO/hashing work. Do not replace unavailable roles
        # using test data or task IDs inherited without participation evidence.
        candidates=balanced_cap(candidates,y,max(sum(limits.values())*20,20000),rng)
        selected={role:[] for role in limits};local_seen=set();excluded=0
        provenance=[]
        for index in candidates:
            digest=content_hash(x[index])
            if digest in forbidden:excluded+=1;continue
            if digest in local_seen:continue
            local_seen.add(digest);role=content_role(digest)
            if len(selected[role])>=limits[role]:continue
            selected[role].append(int(index));hashes[role].add(digest)
            provenance.append(dict(client_id=cid,row_id=int(index),role=role,input_sha256=digest,
                role_row_index=len(selected[role])-1,task_evidence=','.join(map(str,sorted(evidence)))))
            if all(len(selected[r])==limits[r] for r in limits):break
        for role,indices in selected.items():
            if len(indices)<32:raise ValueError(f'{cid}: only {len(indices)} {role} rows; refusing unreliable fit')
            pools[role][cid]=dict(X=torch.from_numpy(x[indices]),y=y[indices])
        rows.extend(provenance)
        print(f'Gate data cid={cid}: '+str({r:len(v) for r,v in selected.items()})+f', excluded panel duplicates={excluded}',flush=True)
    for a in limits:
        for b in limits:
            if a!=b and hashes[a]&hashes[b]:raise RuntimeError('Content overlap across gate roles')
    pd.DataFrame(rows).to_csv(out/'gate_data_provenance.csv',index=False)
    write_json(out/'gate_split_audit.json',dict(unique_content_by_role={r:len(v) for r,v in hashes.items()},
        exact_panel_content_excluded=True,roles_globally_content_disjoint=True,
        original_backbone_holdout=False,historical_revisit=True))
    return pools


def collect(ckpt,pools,needed,ids,seen,device,batch_size,manifest,out,stage):
    caches={role:{cid:{} for cid in ids} for role in pools};runtime=[]
    folder=out/f'{stage}_expert_features';folder.mkdir(exist_ok=True)
    for position,donor in enumerate(ids,1):
        pieces=[(role,cid,pools[role][cid]['X']) for role in pools for cid in ids if donor in needed[cid]]
        if not pieces:continue
        model,detector=_make_denice_client_model(ckpt,donor,device)
        fingerprint=encoder_fingerprint(model)
        if fingerprint!=manifest[str(donor)]['encoder_hash']:raise ValueError('Expert checkpoint hash mismatch')
        detector.router_mode='multiclass_balanced';detector.train_models(max(detector.activation_memory))
        inputs=torch.cat([x for _,_,x in pieces])
        if str(device).startswith('cuda'):torch.cuda.synchronize()
        begin=time.perf_counter();record=expert_features(model,detector,inputs,seen,device,batch_size)
        if str(device).startswith('cuda'):torch.cuda.synchronize()
        elapsed=time.perf_counter()-begin;offset=0
        for role,cid,x in pieces:
            n=len(x);part={name:value[offset:offset+n] for name,value in record.items()};offset+=n
            caches[role][cid][donor]=part
            np.savez_compressed(folder/f'{role}_receiver_{cid}_donor_{donor}.npz',**part)
        if encoder_fingerprint(model)!=fingerprint:raise RuntimeError('Expert weights changed')
        runtime.append(dict(donor=donor,samples=len(inputs),seconds=elapsed))
        pd.DataFrame(runtime).to_csv(out/f'{stage}_runtime.csv',index=False)
        print(f'{stage} experts {position}/{len(ids)} donor={donor}, samples={len(inputs)}, seconds={elapsed:.1f}',flush=True)
        del model,detector,inputs
        if str(device).startswith('cuda'):torch.cuda.empty_cache()
    return caches


def predict_gate(gate,features,pred,chosen,seen,top):
    classes=gate.classes_
    column=int(np.flatnonzero(classes==1)[0])
    scores=gate.predict_proba(features)[:,column].reshape(pred.shape)
    return gate_decision(scores,pred,chosen,seen,top),scores


def prior_baselines(features,pred,names,chosen,seen):
    policies={'GlobalDonorPrior':'calibration_donor_rate','GlobalTaskPrior':'calibration_donor_task_rate',
              'ReceiverTaskPrior':'calibration_receiver_donor_task_rate'}
    return {policy:gate_decision(features[:,names.index(name)].reshape(pred.shape),pred,chosen,seen,1)[0]
            for policy,name in policies.items()}


def run_competence_gate(ckpt,shards,classes,ids,out,device,diagnostic_input,data_dir,batch_size=512,
                        budgets=(4,8,16),candidate_seed=42,limits=None,data_builder=None,
                        extra_baselines=False,variant='v1_local_support'):
    limits=limits or dict(calibration=128,fit=512,validation=256)
    if (set(limits)!={'calibration','fit','validation'}
        or any(not isinstance(value,int) or value<32 for value in limits.values())):
        raise ValueError('Three gate partitions with integer caps >=32 required')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    write_json(out/'gate_completion.json',dict(completed=False))
    if Path(diagnostic_input).is_dir():
        reference=pd.read_csv(Path(diagnostic_input)/'predictions.csv')
        manifest=json.loads((Path(diagnostic_input)/'profile_manifest.json').read_text())
    else:
        with zipfile.ZipFile(diagnostic_input) as z:
            reference=pd.read_csv(z.open('predictions.csv'));manifest=json.loads(z.read('profile_manifest.json'))
    if not str(ckpt['config'].get('git_commit','')).startswith('03b9b53'):
        raise ValueError('Original frozen checkpoint required')
    if not budgets or list(budgets)!=sorted(set(budgets)) or min(budgets)<4 or batch_size<=0:
        raise ValueError('Use increasing declared budgets >=4')
    peers,edges=recorded_peers(ckpt,ids);pd.DataFrame(edges).to_csv(out/'peer_edges.csv',index=False)
    alphas={cid:{int(d):float(w) for d,w in zip(lookup(ckpt['cluster']['alpha_debug'],cid)['group_ids'],
        lookup(ckpt['cluster']['alpha_debug'],cid)['alphas'])} for cid in ids}
    order={cid:peer_orders(cid,peers[cid],alphas[cid],(candidate_seed,))[f'random_{candidate_seed}'] for cid in ids}
    if any(len(order[cid])<max(budgets) for cid in ids):raise ValueError('Insufficient permitted peers')
    needed={cid:[cid]+order[cid][:max(budgets)] for cid in ids}
    seen=sorted({int(c) for values in classes.values() for c in values})
    write_json(out/'gate_protocol.json',dict(budgets=budgets,candidate_seed=candidate_seed,orders=order,
        limits=limits,variant=variant,extra_prior_baselines=extra_baselines,
        stage='retrospective shared gate diagnostic',centralized_gate_fit=True,
        test_panel_role='diagnostic/development; not final untouched test',
        gate_training_uses_historical_local_rows=True,raw_samples_transmitted=False,
        experts_evaluated_in_same_runtime=True,privacy_deployment_claim=False,
        experts_not_retrained=True,training_commit=ckpt['config']['git_commit']))
    pools=(gate_data(ckpt,data_dir,classes,ids,shards,out,limits,20261005) if data_builder is None else
        data_builder(ckpt,data_dir,classes,ids,shards,out,limits,20261005,needed=needed))
    target_dir=out/'gate_role_targets';target_dir.mkdir(exist_ok=True)
    for role,clients in pools.items():
        for cid,pool in clients.items():
            values=dict(y_true=pool['y'],role_row_index=np.arange(len(pool['y'])))
            if 'receiver_local_covered' in pool:values['receiver_local_covered']=pool['receiver_local_covered']
            np.savez_compressed(target_dir/f'{role}_receiver_{cid}.npz',**values)
    cache=collect(ckpt,pools,needed,ids,seen,device,batch_size,manifest,out,'gate_train')
    priors=competence_priors(cache['calibration'],{cid:pools['calibration'][cid]['y'] for cid in ids})
    gates={};validation=[];validation_frames={};names=None
    val_prediction_dir=out/'gate_validation_predictions';val_prediction_dir.mkdir(exist_ok=True)
    val_saved={cid:dict(client_id=np.full(len(pools['validation'][cid]['y']),cid),
        role_row_index=np.arange(len(pools['validation'][cid]['y'])),y_true=pools['validation'][cid]['y']) for cid in ids}
    for cid in ids:
        if 'receiver_local_covered' in pools['validation'][cid]:
            val_saved[cid]['receiver_local_covered']=pools['validation'][cid]['receiver_local_covered']
    for k in budgets:
        fitting=[];targets=[]
        for cid in ids:
            chosen=[cid]+order[cid][:k]
            features,pred,names=feature_matrix(cid,chosen,cache['fit'][cid],alphas[cid],priors,len(classes),max(seen)+1)
            fitting.append(features);targets.append((pred==pools['fit'][cid]['y'][:,None]).reshape(-1))
        x=np.concatenate(fitting);y=np.concatenate(targets).astype(int)
        if len(np.unique(y))!=2:raise ValueError('Gate training target requires both correctness outcomes')
        candidates=dict(LR=LogisticRegression(C=1,max_iter=1000,random_state=20261005),
            MLP=MLPClassifier(hidden_layer_sizes=(32,16),alpha=.001,batch_size=1024,
                max_iter=50,early_stopping=False,random_state=20261005))
        validation_frames[k]={}
        if extra_baselines:
            for cid in ids:
                chosen=[cid]+order[cid][:k]
                features,pred,feature_names=feature_matrix(cid,chosen,cache['validation'][cid],alphas[cid],priors,len(classes),max(seen)+1)
                outputs=dict(majority=vote(pred.T,np.ones(len(chosen)),seen,pred[:,0]),self=pred[:,0])
                outputs.update(prior_baselines(features,pred,feature_names,chosen,seen))
                for policy,result in outputs.items():
                    validation_frames[k].setdefault(policy,[]).append((pools['validation'][cid]['y'],result))
                    val_saved[cid][f'k{k}_{policy}']=result
        for family,estimator in candidates.items():
            gate=make_pipeline(StandardScaler(),estimator);gate.fit(x,y);gates[k,family]=gate
            for cid in ids:
                chosen=[cid]+order[cid][:k];features,pred,_=feature_matrix(cid,chosen,
                    cache['validation'][cid],alphas[cid],priors,len(classes),max(seen)+1)
                for top in (1,2,4):
                    (result,_),_=predict_gate(gate,features,pred,chosen,seen,top)
                    policy=f'{family}_top{top}'
                    validation_frames[k].setdefault(policy,[]).append((pools['validation'][cid]['y'],result))
                    val_saved[cid][f'k{k}_{policy}']=result
            print(f'Fitted gate k={k} family={family}, expert-row targets={len(y)}',flush=True)
        for policy,values in validation_frames[k].items():
            truth=np.concatenate([v[0] for v in values]);prediction=np.concatenate([v[1] for v in values])
            validation.append(dict(k=k,policy=policy,accuracy=float((truth==prediction).mean()),
                macro_f1=float(f1_score(truth,prediction,labels=seen,average='macro',zero_division=0))))
        del x,y,fitting,targets
    validation=pd.DataFrame(validation);validation.to_csv(out/'gate_validation_metrics.csv',index=False)
    for cid,values in val_saved.items():pd.DataFrame(values).to_csv(val_prediction_dir/f'client_{cid}.csv',index=False)
    if extra_baselines:
        validation_full=pd.concat([pd.DataFrame(values) for values in val_saved.values()],ignore_index=True)
        covered_rows=[]
        if 'receiver_local_covered' in validation_full:
            for row in validation.itertuples():
                for covered,group in validation_full.groupby('receiver_local_covered'):
                    correct=group[f'k{row.k}_{row.policy}'].to_numpy()==group.y_true.to_numpy()
                    covered_rows.append(dict(k=int(row.k),policy=row.policy,receiver_local_covered=bool(covered),
                        rows=len(group),correct=int(correct.sum()),accuracy=float(correct.mean())))
            pd.DataFrame(covered_rows).to_csv(out/'gate_validation_coverage_metrics.csv',index=False)
        del validation_full
    # Primary gate selection remains among the same six learned policies as V1.
    # Prior-only/majority baselines are reported, not promoted using test outcomes.
    learned=validation[validation.policy.str.startswith(('LR_','MLP_'))]
    selected={k:learned[learned.k==k].sort_values(['accuracy','macro_f1','policy'],
        ascending=[False,False,True]).iloc[0].policy for k in budgets}
    import sklearn
    bundle=dict(gates=gates,priors=priors,feature_names=names,selected=selected,orders=order,budgets=budgets,
        variant=variant,sklearn_version=sklearn.__version__)
    joblib.dump(bundle,out/'frozen_gate.joblib')
    digest=hashlib.sha256((out/'frozen_gate.joblib').read_bytes()).hexdigest()
    write_json(out/'gate_lock.json',dict(locked_before_test_expert_inference=True,gate_sha256=digest,
        selected_by_validation=selected,selection_rule='validation accuracy, macro-F1, alphabetical policy',
        feature_names=names,sklearn_version=sklearn.__version__,variant=variant,
        test_labels_used_for_fit=False,test_labels_used_for_selection=False))
    # Test exactly the restored artifact, rather than an unsaved in-memory fit.
    restored=joblib.load(out/'frozen_gate.joblib')
    gates=restored['gates'];priors=restored['priors'];selected=restored['selected']
    # Clear all fitting data before test experts execute. Gate is frozen on disk.
    del cache,pools,validation_frames,val_saved
    test_pools={'test':{cid:dict(X=shards[cid]['X_test']) for cid in ids}}
    test=collect(ckpt,test_pools,needed,ids,seen,device,batch_size,manifest,out,'test')['test']
    metrics=[];frames=[];pred_dir=out/'predictions';pred_dir.mkdir(exist_ok=True)
    for cid in ids:
        truth=shards[cid]['y_test'].numpy();own=test[cid][cid]['pred']
        prior=reference[reference.client_id==cid].sort_values('sample_in_shard')
        if (not np.array_equal(prior.global_test_row,shards[cid]['sample_ids'])
            or not np.array_equal(prior.y_true,truth) or not np.array_equal(prior.Multiclass,own)
            or not np.array_equal(prior.Multiclass_task,test[cid][cid]['task'])):
            raise RuntimeError(f'{cid}: original self baseline/panel mismatch')
        frame=dict(client_id=np.full(len(truth),cid),global_test_row=shards[cid]['sample_ids'],y_true=truth,self=own)
        for k in budgets:
            chosen=[cid]+order[cid][:k];features,pred,feature_names=feature_matrix(cid,chosen,test[cid],alphas[cid],priors,len(classes),max(seen)+1)
            output=dict(majority=vote(pred.T,np.ones(len(chosen)),seen,own),self=own)
            if extra_baselines:output.update(prior_baselines(features,pred,feature_names,chosen,seen))
            for family in ('LR','MLP'):
                for top in (1,2,4):
                    (result,donor),scores=predict_gate(gates[k,family],features,pred,chosen,seen,top)
                    policy=f'{family}_top{top}';output[policy]=result
                    frame[f'k{k}_{policy}_top_donor']=donor
            output['ValidationSelected']=output[selected[k]]
            oracle=(pred==truth[:,None]).any(1)
            frame[f'k{k}_OracleActualRouted']=oracle
            for policy,result in output.items():
                if np.any((result==truth)&~oracle):raise RuntimeError('Gate exceeds routed expert bound')
                frame[f'k{k}_{policy}']=result
                metrics.append(dict(client_id=cid,k=k,policy=policy,n=len(truth),correct=int((result==truth).sum()),
                    accuracy=float((result==truth).mean()),oracle_correct=int(oracle.sum())))
        df=pd.DataFrame(frame);df.to_csv(pred_dir/f'client_{cid}.csv',index=False);frames.append(df)
    if hashlib.sha256((out/'frozen_gate.joblib').read_bytes()).hexdigest()!=digest:
        raise RuntimeError('Frozen gate artifact mutated during evaluation')
    metrics=pd.DataFrame(metrics);metrics.to_csv(out/'per_client_metrics.csv',index=False)
    full=pd.concat(frames,ignore_index=True);summary=[];rng=np.random.default_rng(20261005)
    for (k,policy),group in metrics.groupby(['k','policy']):
        group=group.set_index('client_id').loc[ids]
        majority=metrics[(metrics.k==k)&(metrics.policy=='majority')].set_index('client_id').loc[ids]
        delta=group.accuracy.to_numpy()-majority.accuracy.to_numpy()
        draw=rng.integers(0,len(ids),(2000,len(ids)));ci=np.quantile(delta[draw].mean(1),[.025,.975])*100
        prediction=full[f'k{k}_{policy}'].to_numpy();truth=full.y_true.to_numpy()
        summary.append(dict(k=int(k),policy=policy,pooled_accuracy=float(group.correct.sum()/group.n.sum()),
            client_mean_accuracy=float(group.accuracy.mean()),pooled_macro_f1=float(f1_score(truth,prediction,labels=seen,average='macro',zero_division=0)),
            delta_vs_majority_pp=float(delta.mean()*100),paired_ci_low_pp=float(ci[0]),paired_ci_high_pp=float(ci[1]),
            actual_routed_oracle=float(group.oracle_correct.sum()/group.n.sum()),queried_models=int(k+1)))
    summary=pd.DataFrame(summary);summary.to_csv(out/'summary.csv',index=False)
    plot_gate_budget(summary,out)
    write_json(out/'gate_completion.json',dict(completed=True,retrospective_shared_gate=True,
        variant=variant,test_panel_role='diagnostic/development',final_untouched_test=False,
        test_labels_used_only_for_metrics=True,gate_locked_before_test_inference=True,
        independent_backbone_validation=False,streaming_replay_free_claim=False,
        selected_policy=selected,no_raw_sample_network_queries=True))
    return summary


def plot_gate_budget(summary,out):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    for policy in ('self','majority','LR_top1','MLP_top1','ValidationSelected'):
        values=summary[summary.policy==policy].sort_values('k')
        ax.plot(values.k,100*values.pooled_accuracy,marker='o',label=policy)
    oracle=summary[summary.policy=='majority'].sort_values('k')
    ax.plot(oracle.k,100*oracle.actual_routed_oracle,linestyle='--',label='Actual-routed oracle (labels)')
    ax.axhline(50,color='grey',linestyle=':',label='50% reference')
    ax.set(xlabel='Candidate peers k (self additional)',ylabel='Pooled accuracy (%)',
        title='Frozen experts: validation-selected competence gate',ylim=(0,100))
    ax.legend(fontsize=9);ax.grid(alpha=.25);fig.tight_layout()
    fig.savefig(out/'competence_gate_vs_budget.png',dpi=180)
    fig.savefig(out/'competence_gate_vs_budget.pdf');plt.close(fig)
