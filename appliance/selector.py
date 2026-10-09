"""Select one pair using current local calibration and recorded edges, never test."""
import gc
import numpy as np
from .config import Rejected
from .ledger import Ledger, quality
from .state import write_json


def lookup(mapping, key, default=None):
    return mapping.get(key, mapping.get(str(key), default))


def recorded_graph(ckpt, task, round_id):
    matches=[v for v in ckpt.get('cluster_history', [])
             if int(v.get('task', -1))==task and int(v.get('round', -1))==round_id]
    if len(matches)!=1:raise Rejected('GRAPH_SNAPSHOT_UNAVAILABLE',str(len(matches)))
    record=matches[0];ids=set(map(int,ckpt['client_model_states']));groups={};alphas={}
    for cid in sorted(ids):
        group=lookup(record.get('groups',{}),cid)
        audit=lookup(record.get('alpha_debug',{}),cid)
        if group is None or audit is None:raise Rejected('GRAPH_PROVENANCE_MISSING',str(cid))
        donors=list(map(int,audit['group_ids']));weights=np.asarray(audit['alphas'],dtype=float)
        if (len(donors)!=len(weights) or len(set(donors))!=len(donors) or cid not in donors
                or set(donors)!=set(map(int,group)) or not set(donors).issubset(ids)
                or not np.isfinite(weights).all() or (weights<0).any()):raise Rejected('INVALID_GRAPH',str(cid))
        # A request is legal only for an actual positive-weight collaboration edge.
        groups[cid]=sorted(d for d,w in zip(donors,weights) if d!=cid and w>0)
        alphas[cid]=dict(zip(donors,map(float,weights)))
    return groups,alphas


def participated(ckpt,cid,task,roles,current_classes):
    state=lookup(ckpt.get('client_algorithm_states',{}),cid,{}) or {}
    state=state.get('denice',state)
    router=state.get('context_detector',{}) or {}
    # CGoFed persists task records, not a vector of integer task IDs. See
    # collect_task_features(): {task_id, layers, sample_count, class_support, ...}.
    records=(state.get('cgofed_projection_state') or {}).get('tasks',[]) or []
    completed=set()
    for entry in records:
        if not isinstance(entry,dict) or 'task_id' not in entry:
            raise Rejected('INVALID_LOCAL_TASK_PROVENANCE',f'client={cid}: expected CGoFed task record with task_id')
        try:
            completed.add(int(entry['task_id']))
        except (TypeError,ValueError,OverflowError) as exc:
            raise Rejected('INVALID_LOCAL_TASK_PROVENANCE',f'client={cid}: invalid task_id') from exc
    eligible=roles.manifest.get('eligible_base_experts',{})
    scheduled=any(cid in list(map(int,eligible.get(str(c),[]))) for c in current_classes)
    return scheduled and (task in completed or router.get('router_last_refresh_task')==task)


def current_pool(roles,cid,role,classes):
    # Scoped runtime providers reject old/future/client access before any I/O.
    if callable(getattr(roles, 'current_pool', None)):
        return roles.current_pool(cid,role,classes)
    x,y,rows=roles.client_role(cid,role)
    chosen=np.flatnonzero(np.isin(y,classes))
    return dict(X=x[chosen],y=y[chosen],rows=rows[chosen],origin_client=cid,role=role)


def current_ledger(roles,ckpt,cid,task,classes):
    result=Ledger()
    if participated(ckpt,cid,task,roles,classes):
        counts=roles.manifest['clients'][str(cid)]['role_class_counts']['base']
        for c in classes:result.observe(c,int(counts.get(str(c),0)),task,'locked BASE counts + schedule + local checkpoint task evidence')
    return result


def select_pair(ckpt,roles,task,classes,groups,make_model,predict,protocol,manual=None,transport=None,eligibility_path=None):
    """First class/receiver with a qualified donor; donor ordered by LCB then ID.

    Compatibility is deliberately not consulted when selecting the scientific
    pair. A failed closure therefore remains a negative result for that pair.
    Only scalar donor counts cross the simulated calibration boundary.
    """
    ids=sorted(groups);ledgers={i:current_ledger(roles,ckpt,i,task,classes) for i in ids}
    counts={i:roles.manifest['clients'][str(i)]['role_class_counts']['base'] for i in ids}
    audited=[];cache={}
    missing_pairs=[(c,i) for c in sorted(classes) for i in ids
                   if c not in ledgers[i].observed and int(counts[i].get(str(c),0))==0]
    acceptance_counts={i:sum(int(roles.manifest['clients'][str(i)]['role_class_counts']['calibration'].get(str(c),0))
                             for c in classes) for i in ids}
    candidates=[(c,i) for c,i in missing_pairs if acceptance_counts[i]>=protocol.acceptance_min_rows]
    if eligibility_path is not None:
        write_json(eligibility_path,dict(rule='locked current receiver calibration counts; min acceptance rows; no compatibility/test selection',
            acceptance_min_rows=protocol.acceptance_min_rows,receiver_current_calibration_counts=acceptance_counts,
            missing_pairs=len(missing_pairs),eligible_pairs=len(candidates),
            excluded=[dict(class_id=c,receiver=i,rows=acceptance_counts[i],reason='INSUFFICIENT_RECEIVER_ACCEPTANCE_DATA')
                      for c,i in missing_pairs if acceptance_counts[i]<protocol.acceptance_min_rows]))
    if manual and manual[0] in acceptance_counts and acceptance_counts[manual[0]]<protocol.acceptance_min_rows:
        raise Rejected('INSUFFICIENT_RECEIVER_ACCEPTANCE_DATA',f'client={manual[0]}, rows={acceptance_counts[manual[0]]}')
    if missing_pairs and not candidates:
        raise Rejected('NO_RECEIVER_WITH_ACCEPTANCE_DATA','Missing-class receivers lack the locked minimum current calibration rows')
    if manual:
        i,j,c=manual
        if (c,i) not in candidates or j not in groups.get(i,[]):raise Rejected('INVALID_MANUAL_PAIR')
        candidates=[(c,i)]
    for c,i in candidates:
        offers=[]
        for j in groups[i]:
            if manual and j!=manual[1]:continue
            if c not in ledgers[j].observed:continue
            if transport:transport.send(i,j,'DISCOVERY_REQUEST',dict(class_id=c,task=task))
            if j not in cache:
                model,router=make_model(j)
                pool=current_pool(roles,j,'calibration',classes)
                print(f'APPLIANCE donor calibration: client={j}, current rows={len(pool["y"])}',flush=True)
                pred=predict(model,router,pool['X']) if len(pool['y']) else np.empty(0,dtype=np.int64)
                cache[j]={k:dict(positive=int((pool['y']==k).sum()),predicted=int((pred==k).sum()),
                            correct_positive=int(((pool['y']==k)&(pred==k)).sum())) for k in classes}
                del model,router,pool;gc.collect()
            evidence=cache[j][c]
            if transport:transport.send(j,i,'CALIBRATION_OFFER',dict(class_id=c,task=task,counts=evidence))
            item=dict(receiver=i,donor=j,class_id=c,counts=evidence)
            try:
                score=quality(evidence,protocol);item.update(qualified=True,quality_lcb=score)
                offers.append((score,j,evidence))
            except Rejected as exc:item.update(qualified=False,reason=exc.reason)
            audited.append(item)
        if offers:
            score,j,evidence=sorted(offers,key=lambda value:(-value[0],value[1]))[0]
            return dict(receiver=i,donor=j,class_id=c,task=task,quality_lcb=score,donor_counts=evidence,
                        rule='first current class/receiver; highest qualified donor LCB, tie lowest ID; no closure/test selection'),ledgers[i],audited
    exc=Rejected('NO_QUALIFIED_PAIR',f'{len(candidates)} missing-class pairs; {len(audited)} donor offers audited')
    exc.audit=audited
    raise exc
