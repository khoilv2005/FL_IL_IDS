"""Frozen coverage diagnostics on the recorded directed aggregation graph.

PeerExpertBestAllowed is a label-assisted offline expert-selection bound, not
deployable routing. No peer features, bases or classifier weights are averaged.
"""
import copy
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import torch

from eval_checkpoint import _make_denice_client_model
from tools.eval_denice_tip_router import classify, write_json
from fed_learning.strategies.incremental.denice_tip_router import encoder_fingerprint


def lookup(mapping,key):
    return mapping.get(key,mapping.get(str(key)))


def recorded_peers(checkpoint,ids):
    cluster=checkpoint.get('cluster') or {}
    if int(cluster.get('task',-1))!=5 or int(cluster.get('round',-1))!=19:
        raise ValueError('Expected recorded graph for task 5 round 19')
    peers={}; edges=[]
    for cid in ids:
        group=lookup(cluster.get('groups',{}),cid)
        audit=lookup(cluster.get('alpha_debug',{}),cid)
        if not group or not audit or len(audit['group_ids'])!=len(audit['alphas']):
            raise ValueError(f'{cid}: incomplete graph/weight evidence')
        if set(group)!=set(audit['group_ids']):
            raise ValueError(f'{cid}: recorded group and alpha IDs disagree')
        allowed={cid}
        for donor,weight in zip(audit['group_ids'],audit['alphas']):
            if not np.isfinite(weight) or weight<0:
                raise ValueError('Invalid recorded graph weight')
            if weight>0:
                if int(donor) not in ids:
                    raise ValueError('Graph donor missing from frozen panel')
                allowed.add(int(donor))
                edges.append(dict(receiver=cid,donor=int(donor),alpha=float(weight),self_edge=cid==int(donor)))
        peers[cid]=sorted(allowed)
    return peers,edges


def run_peer_coverage(ckpt,shards,classes,ids,out,device,diagnostic_input,batch_size=512):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    write_json(out/'peer_definition.json',dict(completed=False,status='running_or_failed'))
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
    if ids!=protocol['client_ids'] or not protocol['training_commit'].startswith('03b9b53'):
        raise ValueError('Original checkpoint/panel required')
    peers,edges=recorded_peers(ckpt,ids)
    pd.DataFrame(edges).to_csv(out/'peer_edges.csv',index=False)
    seen=sorted({c for values in classes.values() for c in values})
    mapping={c:t for t,values in classes.items() for c in values}
    support={};tasks={}
    for cid in ids:
        state=lookup(ckpt['client_algorithm_states'],cid);state=state.get('denice',state)
        detector=state['context_detector']
        support[cid]={int(t):list(map(int,values)) for t,values in detector['episode_classes'].items()}
        tasks[cid]=sorted(int(t) for t,values in detector['activation_memory'].items() if len(values))
        if not tasks[cid] or any(not support[cid].get(t) for t in tasks[cid]):
            raise ValueError('This diagnostic requires nonempty masks for every valid route')
    frames={};rows=[]
    for position,cid in enumerate(ids,1):
        model,detector=_make_denice_client_model(ckpt,cid,device)
        fingerprint=encoder_fingerprint(model)
        if fingerprint!=manifest[str(cid)]['encoder_hash']:
            raise ValueError(f'{cid}: model signature mismatch')
        x=shards[cid]['X_test'];y=shards[cid]['y_test'].numpy()
        prior=reference[reference.client_id==cid].sort_values('sample_in_shard').reset_index(drop=True)
        if not (np.array_equal(prior.y_true,y) and np.array_equal(prior.global_test_row,shards[cid]['sample_ids'])):
            raise ValueError('Sample identities differ')
        detector.router_mode='multiclass_balanced';detector.train_models(max(detector.activation_memory))
        pred,route=classify(model,detector,x,seen,device,batch_size)
        if not (np.array_equal(pred,prior.Multiclass) and np.array_equal(route,prior.Multiclass_task)):
            raise ValueError(f'{cid}: integrated multiclass baseline differs')
        union={t:sorted({c for donor in peers[cid] for c in support[donor].get(t,[])}) for t in classes}
        widened=copy.copy(detector);widened.episode_classes=union
        normal_union,_=classify(model,widened,x,seen,device,batch_size,route,'oracle_hard')
        truth=np.asarray([mapping[int(v)] for v in y]); matched_route=np.where(np.isin(truth,tasks[cid]),truth,route)
        matched_union,_=classify(model,widened,x,seen,device,batch_size,matched_route,'oracle_hard')
        own_best=np.zeros(len(y),dtype=bool);union_best=np.zeros(len(y),dtype=bool)
        for task in tasks[cid]:
            fixed=np.full(len(y),task,dtype=np.int64)
            own,_=classify(model,detector,x,seen,device,batch_size,fixed,'oracle_hard')
            extended,_=classify(model,widened,x,seen,device,batch_size,fixed,'oracle_hard')
            own_best|=own==y;union_best|=extended==y
        frame=pd.DataFrame(dict(client_id=cid,global_test_row=prior.global_test_row,y_true=y,true_task=truth,
            baseline_correct=pred==y,local_best_correct=own_best,
            union_mask_normal_correct=normal_union==y,union_mask_matched_correct=matched_union==y,
            union_mask_best_correct=union_best,
            local_class_covered=[int(v) in support[cid].get(int(t),[]) for v,t in zip(y,truth)],
            peer_class_covered=[int(v) in union[int(t)] for v,t in zip(y,truth)],
            local_task_available=np.isin(truth,tasks[cid]),
            peer_task_available=np.isin(truth,sorted({t for donor in peers[cid] for t in tasks[donor]})),
            peer_expert_best_correct=own_best.copy(),first_success_donor=np.where(own_best,cid,-1)))
        frames[cid]=frame
        rows.append(dict(client_id=cid,group_size=len(peers[cid]),peer_count=len(peers[cid])-1,
                         local_routes=tasks[cid],union_class_counts={t:len(v) for t,v in union.items()}))
        if encoder_fingerprint(model)!=fingerprint: raise RuntimeError('Local model mutated')
        print(f'Local/union mask {position}/{len(ids)} cid={cid}',flush=True)
        del model,detector,widened
        if device=='cuda':torch.cuda.empty_cache()
    pd.concat(frames.values(),ignore_index=True).to_csv(out/'peer_predictions.csv',index=False)
    write_json(out/'graph_support.json',rows)
    own_mean=float(np.mean([f.local_best_correct.mean() for f in frames.values()]))
    if abs(own_mean-0.4536601333292612)>0.001:
        raise RuntimeError('Local BestAllowedRoute no longer reproduces 45.3660%')
    # Donor-major traversal: each donor model is loaded once, and evaluates only
    # receivers that recorded a positive incoming edge from this donor.
    for position,donor in enumerate(ids,1):
        receivers=[cid for cid in ids if donor in peers[cid] and cid!=donor]
        if not receivers:continue
        model,detector=_make_denice_client_model(ckpt,donor,device)
        fingerprint=encoder_fingerprint(model)
        all_x=torch.cat([shards[cid]['X_test'] for cid in receivers])
        all_y=np.concatenate([shards[cid]['y_test'].numpy() for cid in receivers])
        best=np.zeros(len(all_y),dtype=bool)
        for task in tasks[donor]:
            pred,_=classify(model,detector,all_x,seen,device,batch_size,
                            np.full(len(all_y),task,dtype=np.int64),'oracle_hard')
            best|=pred==all_y
        offset=0
        for cid in receivers:
            frame=frames[cid];n=len(frame);success=best[offset:offset+n];offset+=n
            new=success & ~frame.peer_expert_best_correct.to_numpy()
            frame.loc[new,'first_success_donor']=donor
            frame['peer_expert_best_correct']=frame.peer_expert_best_correct.to_numpy()|success
        if encoder_fingerprint(model)!=fingerprint:raise RuntimeError('Donor model mutated')
        print(f'Peer experts {position}/{len(ids)} donor={donor}, receivers={len(receivers)}',flush=True)
        del model,detector,all_x
        if device=='cuda':torch.cuda.empty_cache()
    predictions=pd.concat(frames.values(),ignore_index=True)
    predictions.to_csv(out/'peer_predictions.csv',index=False)
    columns=[c for c in predictions if c.endswith('_correct') or c.endswith('_covered') or c.endswith('_available')]
    per_client=predictions.groupby('client_id')[columns].mean()
    per_client.to_csv(out/'peer_per_client.csv')
    summary=pd.DataFrame([dict(metric=c,pooled_fraction=float(predictions[c].mean()),
                               client_mean_fraction=float(per_client[c].mean())) for c in columns])
    summary.to_csv(out/'peer_summary.csv',index=False)
    predictions.groupby('true_task')[columns].mean().to_csv(out/'peer_per_task.csv')
    write_json(out/'peer_definition.json',dict(completed=True,graph='checkpoint task5 round19, directed positive alpha edges plus self',
        mask_branch='local model, same local routes/adapters; class mask union from legitimate peers',
        expert_branch='offline oracle choice of donor and donor route; donor-local class mask and adapter',
        caveat='Expert oracle uses labels and offline access to receiver inputs. Communication/inference protocol is not implemented.',
        alignment='No continuous features or bases are transferred between encoders',
        scope='Fixed 50k panel and frozen checkpoint only; expanded mask need not improve accuracy'))
    return summary
