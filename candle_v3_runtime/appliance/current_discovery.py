"""Donor offers and missing-class requests from current, local evidence only.

Input records belong to clients; the scheduler sees offers and requests, never
raw calibration samples or future-task schedules. Selection precedes functional
guard SELECTION/HOLDOUT. A failed pair is not replaced using its HOLDOUT score.
"""
import numpy as np

from .active_pair_calibration import split_counts
from .config import Protocol,Rejected
from .ledger import quality


def local_offers(client,task,classes,labels,predictions,base_counts,ranks,model_fingerprint,role_sha256):
    labels=np.asarray(labels,np.int64);predictions=np.asarray(predictions,np.int64)
    classes=sorted(set(map(int,classes)))
    if len(labels)!=len(predictions) or not set(map(int,np.unique(labels))).issubset(classes):
        raise Rejected('OFFER_CURRENT_FIT_SCOPE_CHANGED')
    offers=[];rejected=[]
    for c in classes:
        # Inherited context descriptors and imported rows do not create local
        # BASE provenance; source counters must already belong to this client.
        if int(base_counts.get(str(c),0))<=0 or int(ranks[c])<2:continue
        counts=dict(positive=int((labels==c).sum()),predicted=int((predictions==c).sum()),
            correct_positive=int(((labels==c)&(predictions==c)).sum()))
        try:lcb=quality(counts,Protocol().validate())
        except Rejected as exc:
            rejected.append(dict(client=int(client),class_id=c,task=int(task),reason=exc.reason,counts=counts));continue
        offers.append(dict(donor=int(client),class_id=c,task=int(task),counts=counts,quality_lcb=lcb,
            model_fingerprint=model_fingerprint,role_sha256=role_sha256,
            provenance='locally observed current-task BASE + mature output + own current calibration FIT'))
    return offers,rejected


def local_requests(client,task,classes,base_counts,calibration_counts,ranks,router_refresh_task):
    if router_refresh_task!=task:return []
    result=[]
    # Only current task counts are consulted. No next-task BASE counts, future
    # membership, old calibration rows, or imported-memory evidence are used.
    for c in sorted(set(map(int,classes))):
        if (int(base_counts.get(str(c),0)) or int(calibration_counts.get(str(c),0)) or int(ranks[c])!=0):continue
        totals=[sum(split_counts(int(calibration_counts.get(str(k),0)))[i] for k in classes if k!=c) for i in range(3)]
        if min(totals)<32:continue
        result.append(dict(receiver=int(client),class_id=c,task=int(task),current_calibration_counts=totals,
            output_slot_free=True,locally_observed=False,historical_damage_coverage='not yet certified'))
    return result


def select_current_pairs(requests,offers,groups,alphas,task,round_id=19):
    if type(round_id) is not int or round_id<0:raise Rejected('INVALID_CURRENT_DISCOVERY_ROUND')
    selected=[];used=set();coverage={}
    for c in sorted({v['class_id'] for v in requests}):
        options=[]
        for request in sorted((v for v in requests if v['class_id']==c),key=lambda v:v['receiver']):
            receiver=request['receiver']
            if receiver in used:continue
            if request['task']!=task:raise Rejected('REQUEST_TASK_CHANGED')
            candidates=[offer for offer in offers if offer['class_id']==c and offer['task']==task
                and offer['donor'] in groups.get(receiver,[]) and alphas.get(receiver,{}).get(offer['donor'],0)>0]
            options += [dict(request=request,offer=offer) for offer in sorted(candidates,key=lambda v:(-v['quality_lcb'],v['donor']))]
        coverage[c]=dict(eligible_current_pairs=len(options))
        if options:
            item=options[0];used.add(item['request']['receiver'])
            selected.append(dict(receiver=item['request']['receiver'],donor=item['offer']['donor'],class_id=c,
                task=task,round=round_id,selection_scope='current_local_fit',quality_lcb=item['offer']['quality_lcb'],donor_model_fingerprint=item['offer']['model_fingerprint'],
                rule='class ascending; first current-eligible receiver ID; best local FIT Wilson LCB donor, tie donor ID; unique receiver'))
    return selected,coverage
