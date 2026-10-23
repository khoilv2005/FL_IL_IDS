"""Historical gate data from legitimate origins with local learning evidence."""
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tools.denice_competence_gate import content_hash,content_role
from tools.eval_denice_tip_router import balanced_cap,write_json
from tools.eval_denice_peer_coverage import lookup,recorded_peers


def local_training_support(ckpt,cid,classes):
    state=lookup(ckpt['client_algorithm_states'],cid)
    state=state.get('denice',state);detector=state['context_detector']
    evidence={int(entry['task_id']):'cgofed_completed_task'
        for entry in (state.get('cgofed_projection_state') or {}).get('tasks',[]) if 'task_id' in entry}
    refreshed=detector.get('router_last_refresh_task')
    if refreshed is not None:evidence.setdefault(int(refreshed),'local_router_refresh')
    support={}
    for task,source in evidence.items():
        if task in classes:
            for label in set(classes[task])&set(lookup(detector['episode_classes'],task) or []):
                support[int(label)]=(task,source)
    return support


def peer_supported_data(ckpt,data_dir,classes,ids,test_shards,out,limits,seed,needed):
    """Sample class-balanced roles from self + fixed max-budget candidate origins.

    Origin shards are read once into bounded banks. Origin labels require that
    origin's local task evidence, not the receiver's memory. Content role is
    global; repeated inputs never cross calibration/fit/validation partitions.
    """
    out=Path(out);peers,_=recorded_peers(ckpt,ids)
    for cid in ids:
        if cid not in needed[cid] or not set(needed[cid]).issubset(peers[cid]):
            raise ValueError('Gate data origin outside fixed legitimate candidate pool')
    forbidden={content_hash(row) for shard in test_shards.values() for row in shard['X_test'].numpy()}
    support={cid:local_training_support(ckpt,cid,classes) for cid in ids}
    source_bank={};digest_labels={};conflicts=set();origin_audit=[]
    for cid in sorted({d for receivers in needed.values() for d in receivers}):
        allowed=sorted(support[cid])
        if not allowed:raise ValueError(f'Origin {cid}: no proven local training support')
        with np.load(Path(data_dir)/f'client_{cid}_train.npz') as data:
            x=np.asarray(data['X_train'],dtype=np.float32);y=np.asarray(data['y_train'],dtype=np.int64)
        rng=np.random.default_rng(seed+1009*cid)
        candidates=balanced_cap(np.flatnonzero(np.isin(y,allowed)),y,
            max(20000,sum(limits.values())*20),rng)
        caps={role:min(limit,max(32,math.ceil(limit/len(allowed))*4)) for role,limit in limits.items()}
        bank={role:{label:[] for label in allowed} for role in limits};local_seen=set();excluded=0
        for index in candidates:
            label=int(y[index]);digest=content_hash(x[index])
            previous=digest_labels.setdefault(digest,label)
            if previous!=label:conflicts.add(digest)
            if digest in forbidden:excluded+=1;continue
            if digest in local_seen:continue
            local_seen.add(digest);role=content_role(digest)
            if len(bank[role][label])<caps[role]:
                bank[role][label].append((cid,int(index),digest,x[index].copy(),label))
        source_bank[cid]=bank
        origin_audit.append(dict(origin_client_id=cid,allowed_classes=allowed,
            support_evidence={label:dict(task=t,evidence=e) for label,(t,e) in support[cid].items()},
            bank_counts={role:{label:len(entries) for label,entries in bank[role].items()} for role in limits},
            excluded_exact_panel_rows=excluded,scanned_rows=len(candidates),per_class_caps=caps))
        print(f'V2 source bank origin={cid}, supported classes={len(allowed)}',flush=True)
    write_json(out/'gate_origin_audit.json',origin_audit)
    pools={role:{} for role in limits};rows=[];hashes={role:set() for role in limits};coverage=[]
    for cid in ids:
        detector=lookup(ckpt['client_algorithm_states'],cid)
        detector=detector.get('denice',detector)['context_detector']
        receiver_classes={int(c) for values in detector['episode_classes'].values() for c in values}
        supported_union=sorted({label for donor in needed[cid] for label in support[donor]})
        local_seen=set()
        for role,limit in limits.items():
            # Uniform class cycling; random choice within class. No alpha,
            # validation metric or test class frequency affects data sampling.
            buckets={}
            rng=np.random.default_rng(seed+1009*cid+{'calibration':1,'fit':2,'validation':3}[role]*104729)
            for label in supported_union:
                entries=[entry for donor in needed[cid] for entry in source_bank[donor][role].get(label,[])
                         if entry[2] not in conflicts]
                if entries:buckets[label]=[entries[i] for i in rng.permutation(len(entries))]
            selected=[]
            while buckets and len(selected)<limit:
                for label in list(buckets):
                    bucket=buckets[label]
                    while bucket and bucket[-1][2] in local_seen:bucket.pop()
                    if bucket and len(selected)<limit:
                        entry=bucket.pop();selected.append(entry);local_seen.add(entry[2]);hashes[role].add(entry[2])
                    if not bucket:del buckets[label]
            if len(selected)<32:
                raise ValueError(f'Receiver {cid}: only {len(selected)} peer-supported {role} rows')
            labels=np.asarray([entry[4] for entry in selected],dtype=np.int64)
            covered=np.isin(labels,list(receiver_classes))
            pools[role][cid]=dict(X=torch.from_numpy(np.stack([entry[3] for entry in selected])),
                y=labels,receiver_local_covered=covered)
            for position,(origin,index,digest,_,label) in enumerate(selected):
                task,evidence=support[origin][label]
                rows.append(dict(client_id=cid,role=role,role_row_index=position,origin_client_id=origin,
                    row_id=index,label=label,true_task=task,input_sha256=digest,origin_task_evidence=evidence,
                    receiver_local_covered=bool(covered[position]),origin_is_self=origin==cid))
            coverage.append(dict(client_id=cid,role=role,rows=len(selected),
                supported_union_classes=len(supported_union),represented_classes=len(set(labels)),
                missing_supported_classes=','.join(map(str,sorted(set(supported_union)-set(labels)))),
                local_covered_rows=int(covered.sum()),local_uncovered_rows=int((~covered).sum())))
        print(f'V2 receiver={cid}, class union={len(supported_union)}, '
            f'validation uncovered={int((~pools["validation"][cid]["receiver_local_covered"]).sum())}',flush=True)
    for a in limits:
        for b in limits:
            if a!=b and hashes[a]&hashes[b]:raise RuntimeError('Gate V2 content crosses roles')
    pd.DataFrame(rows).to_csv(out/'gate_data_provenance.csv',index=False)
    pd.DataFrame(coverage).to_csv(out/'gate_support_coverage.csv',index=False)
    write_json(out/'gate_split_audit.json',dict(unique_content_by_role={r:len(v) for r,v in hashes.items()},
        exact_panel_content_excluded=True,roles_globally_content_disjoint=True,
        training_content_with_conflicting_labels_excluded=len(conflicts),
        origin_policy='self + fixed maximum-budget positive-alpha candidate peers; origin-local participation required',
        sampling='uniform class cycling from training-supported union; no test-label frequencies',
        original_backbone_holdout=False,historical_revisit=True,development_panel=True))
    return pools
