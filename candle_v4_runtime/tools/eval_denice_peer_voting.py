"""Frozen label-free peer consensus with separate label-assisted budget bounds."""
import json
from pathlib import Path
import time
import zipfile
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from eval_checkpoint import _make_denice_client_model
from tools.eval_denice_tip_router import classify,write_json
from tools.eval_denice_peer_coverage import recorded_peers,lookup
from tools.denice_peer_voting import peer_orders,vote
from fed_learning.training.denice_eval import _route_episodes_with_scores
from fed_learning.strategies.incremental.denice_tip_router import encoder_fingerprint


@torch.no_grad()
def routed_confidence(model,detector,inputs,device,batch_size):
    confidence=[];routes=[]
    for start in range(0,len(inputs),batch_size):
        task,prob=_route_episodes_with_scores(model,inputs[start:start+batch_size].to(device),detector)
        q=prob[np.arange(len(task)),task]
        if not np.isfinite(q).all() or (q<0).any() or (q>1+1e-8).any():
            raise ValueError('Invalid multiclass task confidence')
        routes.append(task);confidence.append(q)
    return np.concatenate(routes),np.concatenate(confidence)


def run_peer_voting(ckpt,shards,classes,ids,out,device,diagnostic_input,batch_size=512,
                    budgets=(0,1,2,4,8,16),random_seeds=(42,43,44,45,46)):
    if (not budgets or list(budgets)!=sorted(set(budgets)) or budgets[0]!=0
            or any(not isinstance(k,int) or k<0 for k in budgets)
            or not random_seeds or len(set(random_seeds))!=len(random_seeds)
            or batch_size<=0):
        raise ValueError('Use increasing unique budgets starting at 0, distinct random seeds and positive batch size')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    write_json(out/'voting_completion.json',dict(completed=False))
    source=Path(diagnostic_input)
    if source.is_dir():
        reference=pd.read_csv(source/'predictions.csv')
        manifest=json.loads((source/'profile_manifest.json').read_text())
        protocol=json.loads((source/'protocol.json').read_text())
    else:
        with zipfile.ZipFile(source) as z:
            reference=pd.read_csv(z.open('predictions.csv'))
            manifest=json.loads(z.read('profile_manifest.json'))
            protocol=json.loads(z.read('protocol.json'))
    if protocol['client_ids']!=ids or not protocol['training_commit'].startswith('03b9b53'):
        raise ValueError('Original checkpoint and panel required')
    peers,edges=recorded_peers(ckpt,ids)
    pd.DataFrame(edges).to_csv(out/'peer_edges.csv',index=False)
    seen=sorted({c for values in classes.values() for c in values})
    alphas={cid:{int(d):float(w) for d,w in zip(lookup(ckpt['cluster']['alpha_debug'],cid)['group_ids'],
                                              lookup(ckpt['cluster']['alpha_debug'],cid)['alphas'])} for cid in ids}
    orders={cid:peer_orders(cid,peers[cid],alphas[cid],random_seeds) for cid in ids}
    max_k=max(budgets)
    if any(len(peers[cid])-1<max_k for cid in ids):
        raise ValueError('Insufficient legitimate peers for declared budget')
    needed={cid:sorted({cid}|{d for order in orders[cid].values() for d in order[:max_k]}) for cid in ids}
    write_json(out/'selection_manifest.json',dict(budgets=budgets,random_seeds=random_seeds,
        orders=orders,needed=needed,selection='Recorded positive alpha descending, client ID tie; nested random prefixes',
        locked_before_inference=True,truth_used_for_selection=False))
    cache={cid:{} for cid in ids};runtime=[]
    cache_dir=out/'expert_cache';cache_dir.mkdir(exist_ok=True)
    # Only declared graph/random selections are queried; no oracle expert choice
    # changes which donors execute. Each donor fits from its own saved sketches.
    for position,donor in enumerate(ids,1):
        receivers=[cid for cid in ids if donor in needed[cid]]
        if not receivers:continue
        model,detector=_make_denice_client_model(ckpt,donor,device)
        fingerprint=encoder_fingerprint(model)
        if fingerprint!=manifest[str(donor)]['encoder_hash']:
            raise ValueError(f'Donor {donor}: model fingerprint mismatch')
        detector.router_mode='multiclass_balanced';detector.train_models(max(detector.activation_memory))
        x=torch.cat([shards[cid]['X_test'] for cid in receivers])
        y=np.concatenate([shards[cid]['y_test'].numpy() for cid in receivers])
        if device=='cuda':torch.cuda.synchronize()
        begin=time.perf_counter()
        normal,task=classify(model,detector,x,seen,device,batch_size)
        task_check,q=routed_confidence(model,detector,x,device,batch_size)
        if not np.array_equal(task,task_check):raise RuntimeError('Task confidence route differs')
        if device=='cuda':torch.cuda.synchronize()
        normal_seconds=time.perf_counter()-begin
        # Bounds use labels AFTER prediction. The normal policies use only normal,
        # task confidence and training-time graph alpha, never these correctness flags.
        begin=time.perf_counter();best=np.zeros(len(y),dtype=bool)
        legal=sorted(int(t) for t,values in detector.activation_memory.items() if len(values))
        for episode in legal:
            if not detector.episode_classes.get(episode):
                raise ValueError('Expert oracle requires nonempty existing masks')
            pred,_=classify(model,detector,x,seen,device,batch_size,
                            np.full(len(y),episode,dtype=np.int64),'oracle_hard')
            best|=pred==y
        if device=='cuda':torch.cuda.synchronize()
        oracle_seconds=time.perf_counter()-begin
        offset=0
        for cid in receivers:
            n=len(shards[cid]['y_test']);sl=slice(offset,offset+n);offset+=n
            cache[cid][donor]=dict(pred=normal[sl],task=task[sl],q=q[sl],best=best[sl],
                                   cost_per_sample=normal_seconds/len(y))
            np.savez_compressed(cache_dir/f'receiver_{cid}_donor_{donor}.npz',
                                prediction=normal[sl],predicted_task=task[sl],router_confidence=q[sl],
                                best_allowed_correct=best[sl],sample_ids=shards[cid]['sample_ids'])
        if encoder_fingerprint(model)!=fingerprint:raise RuntimeError('Donor weights changed')
        runtime.append(dict(donor=donor,receivers=len(receivers),samples=len(y),routes=len(legal),
                            normal_plus_confidence_seconds=normal_seconds,oracle_seconds=oracle_seconds))
        pd.DataFrame(runtime).to_csv(out/'runtime.csv',index=False)
        print(f'Experts {position}/{len(ids)} donor={donor}, receivers={len(receivers)}, '
              f'normal={normal_seconds:.2f}s oracle={oracle_seconds:.2f}s',flush=True)
        del model,detector,x
        if device=='cuda':torch.cuda.empty_cache()
    metrics=[];saved_predictions={};prediction_dir=out/'predictions';prediction_dir.mkdir(exist_ok=True)
    for cid in ids:
        y=shards[cid]['y_test'].numpy()
        prior=reference[reference.client_id==cid].sort_values('sample_in_shard').reset_index(drop=True)
        own=cache[cid][cid]
        if (not np.array_equal(prior.global_test_row,shards[cid]['sample_ids'])
                or not np.array_equal(prior.y_true,y)
                or not np.array_equal(prior.Multiclass,own['pred'])
                or not np.array_equal(prior.Multiclass_task,own['task'])):
            raise ValueError(f'{cid}: baseline panel/predictions changed')
        frame=dict(global_test_row=prior.global_test_row.to_numpy(),y_true=y,true_task=prior.true_task.to_numpy())
        for selection,order in orders[cid].items():
            previous_actual=np.zeros(len(y),dtype=bool);previous_best=np.zeros(len(y),dtype=bool)
            for k in budgets:
                chosen=[cid]+order[:k]
                pred=np.stack([cache[cid][d]['pred'] for d in chosen])
                confidence=np.stack([cache[cid][d]['q'] for d in chosen])
                weight=np.asarray([alphas[cid].get(d,0.0) for d in chosen])
                actual=(pred==y[None,:]).any(0)
                best=np.stack([cache[cid][d]['best'] for d in chosen]).any(0)
                if (np.any(previous_actual & ~actual) or np.any(previous_best & ~best)
                        or np.any(actual & ~best)):
                    raise RuntimeError('Nested budget oracle bounds violated')
                previous_actual,previous_best=actual,best
                outputs=dict(majority=vote(pred,np.ones(len(chosen)),seen,own['pred']),
                    alpha_vote=vote(pred,weight,seen,own['pred']),
                    alpha_router_confidence=vote(pred,weight[:,None]*confidence,seen,own['pred']))
                if k==0:
                    outputs={name:own['pred'] for name in outputs}
                elif k==1:
                    outputs['single_peer']=pred[1]
                for policy,result in outputs.items():
                    if np.any((result==y)&~actual):
                        raise RuntimeError('Vote exceeds the fixed routed-expert action set')
                    column=f'{selection}_k{k}_{policy}';frame[column]=result
                    metrics.append(dict(client_id=cid,selection=selection,k=k,policy=policy,samples=len(y),
                        correct=int((result==y).sum()),accuracy=float((result==y).mean()),
                        f1_macro=float(f1_score(y,result,labels=seen,average='macro',zero_division=0)),
                        queried_models=1 if k==0 or policy=='single_peer' else k+1,
                        amortized_sequential_seconds_per_sample=float(sum(cache[cid][d]['cost_per_sample']
                            for d in (chosen[1:] if policy=='single_peer' else chosen)))))
                for policy,flags in [('OracleActualRouted',actual),('PeerExpertBestAllowed',best)]:
                    frame[f'{selection}_k{k}_{policy}']=flags
                    metrics.append(dict(client_id=cid,selection=selection,k=k,policy=policy,samples=len(y),
                        correct=int(flags.sum()),accuracy=float(flags.mean()),f1_macro=None,
                        queried_models=k+1,amortized_sequential_seconds_per_sample=None))
        pd.DataFrame(frame).to_csv(prediction_dir/f'client_{cid}.csv',index=False)
        saved_predictions[cid]=frame
    table=pd.DataFrame(metrics);table.to_csv(out/'per_client_metrics.csv',index=False)
    summary=[];rng=np.random.default_rng(20261005)
    baseline={cid:float((cache[cid][cid]['pred']==shards[cid]['y_test'].numpy()).mean()) for cid in ids}
    for (selection,k,policy),group in table.groupby(['selection','k','policy'],sort=False):
        group=group.set_index('client_id').loc[ids]
        delta=group.accuracy.to_numpy()-np.asarray([baseline[c] for c in ids])
        draws=rng.integers(0,len(ids),size=(2000,len(ids)))
        ci=np.quantile(delta[draws].mean(1),[0.025,0.975])*100
        # Pooled F1 is recomputed from per-client output; oracle hit sets have no
        # deployable prediction vector and therefore no F1.
        f1=None
        if policy not in ('OracleActualRouted','PeerExpertBestAllowed'):
            col=f'{selection}_k{k}_{policy}'
            labels_all=[];pred_all=[]
            for cid in ids:
                values=saved_predictions[cid]
                labels_all.append(values['y_true']);pred_all.append(values[col])
            f1=float(f1_score(np.concatenate(labels_all),np.concatenate(pred_all),labels=seen,
                              average='macro',zero_division=0))
        summary.append(dict(selection=selection,k=int(k),policy=policy,
            pooled_accuracy=float(group.correct.sum()/group.samples.sum()),
            client_mean_accuracy=float(group.accuracy.mean()),pooled_macro_f1=f1,
            delta_vs_self_pp=float(delta.mean()*100),paired_ci_low_pp=float(ci[0]),paired_ci_high_pp=float(ci[1]),
            models_per_sample=int(group.queried_models.iloc[0]),
            amortized_sequential_seconds_per_sample=(float(group.amortized_sequential_seconds_per_sample.mean())
                if group.amortized_sequential_seconds_per_sample.notna().any() else None)))
    summary=pd.DataFrame(summary);summary.to_csv(out/'summary.csv',index=False)
    random=summary[summary.selection.str.startswith('random_')]
    random.groupby(['k','policy'])[['pooled_accuracy','client_mean_accuracy']].agg(['mean','std','min','max']).to_csv(out/'random_summary.csv')
    # Paired top-alpha versus random mean, averaged within client before bootstrap.
    comparisons=[]
    for (k,policy),group in table.groupby(['k','policy']):
        top=group[group.selection=='alpha'].set_index('client_id').loc[ids].accuracy.to_numpy()
        random_mean=group[group.selection.str.startswith('random_')].groupby('client_id').accuracy.mean().loc[ids].to_numpy()
        delta=top-random_mean;draws=rng.integers(0,len(ids),size=(2000,len(ids)))
        ci=np.quantile(delta[draws].mean(1),[0.025,0.975])*100
        comparisons.append(dict(k=int(k),policy=policy,alpha_minus_random_mean_pp=float(delta.mean()*100),
                                paired_ci_low_pp=float(ci[0]),paired_ci_high_pp=float(ci[1])))
    pd.DataFrame(comparisons).to_csv(out/'alpha_vs_random.csv',index=False)
    plot_budget(summary,out)
    write_json(out/'voting_completion.json',dict(completed=True,normal_policy_uses_labels=False,
        oracle_uses_labels=True,self_client_mean_accuracy=float(np.mean(list(baseline.values()))),
        confidence='Uncalibrated task-router probability; exploratory multiplier; single-task router confidence is 1',
        cost='Logical expert queries, not literal neural module forwards. Timing includes an extra confidence router pass.',
        selection='All budgets reported; no best budget promoted from this test panel'))
    return summary[summary.selection=='alpha']


def plot_budget(summary,out):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    selected=summary[summary.selection=='alpha']
    for policy in ('majority','alpha_vote','alpha_router_confidence'):
        values=selected[selected.policy==policy].sort_values('k')
        axes[0].plot(values.k,100*values.pooled_accuracy,marker='o',label=policy)
    baseline=selected[(selected.k==0)&(selected.policy=='majority')].pooled_accuracy.iloc[0]
    axes[0].axhline(100*baseline,color='grey',linestyle='--',label='Self multiclass')
    for policy in ('majority','alpha_vote','alpha_router_confidence'):
        random=summary[(summary.policy==policy)&summary.selection.str.startswith('random_')].groupby('k').pooled_accuracy.mean()
        axes[0].plot(random.index,100*random.values,linestyle=':',label='Random mean '+policy)
    for policy in ('OracleActualRouted','PeerExpertBestAllowed'):
        values=selected[selected.policy==policy].sort_values('k')
        axes[1].plot(values.k,100*values.pooled_accuracy,marker='o',label='Top-alpha '+policy)
        random=summary[(summary.policy==policy)&summary.selection.str.startswith('random_')].groupby('k').pooled_accuracy.mean()
        axes[1].plot(random.index,100*random.values,linestyle='--',label='Random mean '+policy)
    axes[1].axhline(50,color='grey',linestyle=':',label='50% reference')
    for ax,title in zip(axes,('Label-free accuracy vs peer budget','Label-assisted oracle coverage vs budget')):
        ax.set(xlabel='Queried peers k (self additional)',ylabel='Accuracy / coverage (%)',title=title,ylim=(0,100))
        ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'accuracy_oracle_vs_budget.png',dpi=180)
    fig.savefig(out/'accuracy_oracle_vs_budget.pdf');plt.close(fig)
