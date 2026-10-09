"""Bounded deployment/certificate checks using a previously accepted real entry.

Synthetic receipt and graph cases check rejection/serialization logic only.
They are not empirical acceptance evidence or readiness for full training.
"""
import argparse
import copy
import json
from pathlib import Path

import torch

from appliance.config import Rejected
from appliance.cumulative_certificate import (RECEIPT_VERSION, append_current_evidence,
    checked_receipt, decode_receipt, encode_receipt, sha, validated_scope)
from appliance.discovery_preflight import schedule_requests
from appliance.empirical_deployment import bind_deployment, deployment_scope
from appliance.patch_lifecycle import certificate_binding_current, reconcile, route_authorized
from appliance.state import write_json, digest


def run(a):
    source=torch.load(a.source,map_location='cpu',weights_only=False)
    original=copy.deepcopy(source['client_algorithm_states'][0]['denice']['appliance_guarded_head_entries'][21])
    checks={}
    def check(name,condition):
        checks[name]=bool(condition)
    def rejected(name,fn):
        try:fn()
        except Rejected:
            check(name,True)
        else:check(name,False)
    seal=original['lifecycle_certificate_sha256']
    runtime=dict(domain_id='cumulative_dataset',classes=list(range(24)))
    strict=copy.deepcopy(original)
    check('real_initial_certificate_valid',certificate_binding_current(strict))
    check('strict_missing_scope_suspends',reconcile(strict,True,runtime,3)['state']=='SUSPENDED')
    check('strict_not_authorized',not route_authorized(strict,runtime))
    e=copy.deepcopy(original)
    bind_deployment(e,'cpu',str(torch.__version__))
    state=reconcile(e,True,runtime,3)
    check('empirical_scope_active',state['state']=='COMMITTED' and route_authorized(e,runtime))
    check('outside_CAL_evidence_reported',state['empirical_outside_CAL_scope'] and not state['inside_effective_scope'])
    check('original_acceptance_and_seal_unchanged',e['lifecycle_certificate']==original['lifecycle_certificate'] and
          e['current_acceptance']==original['current_acceptance'] and e['lifecycle_certificate_sha256']==seal)
    check('small_new_CAL_carries',reconcile(e,True,dict(runtime,classes=list(range(30))),4,new_cal_rows=9)['state']=='CARRY_FORWARD')
    check('domain_change_suspends',reconcile(copy.deepcopy(e),True,dict(runtime,domain_id='other'),4)['state']=='SUSPENDED')
    check('function_drift_suspends',reconcile(copy.deepcopy(e),False,runtime,4)['state']=='SUSPENDED')
    conflict=copy.deepcopy(e)
    conflict['new_conflict_latched']=True
    check('conflict_latch_rejects_even_active_state',not route_authorized(conflict,runtime))
    check('conflict_suspends',reconcile(conflict,True,runtime,4,conflict=True)['state']=='SUSPENDED')
    forged=copy.deepcopy(e)
    forged['empirical_deployment']['population_FAR_claim']=True
    forged['empirical_deployment_sha256']=sha(forged['empirical_deployment'])
    rejected('rehashed_false_safety_claim_rejected',lambda:deployment_scope(forged))
    check('invalid_deployment_fails_closed',not route_authorized(forged,runtime))
    check('valid_declaration_JSON_roundtrip',deployment_scope(
        dict(e,empirical_deployment=json.loads(json.dumps(e['empirical_deployment']))))==deployment_scope(e))
    value=dict(version=RECEIPT_VERSION,receiver=0,owner=1,task=4,class_id=21,patch_id=original['patch_id'],
        initial_certificate_sha256=seal,guard_function_fingerprint=original['guard_function_fingerprint'],
        guard_declaration_sha256=sha(original['guard_declaration']),
        role_manifest_sha256=original['lifecycle_certificate']['initial_acceptance']['role_manifest_sha256'],
        partition_sha256='a'*64,coordinate_sha256='b'*64,current_classes=list(range(24,30)),
        counts={'24':dict(rows=64,activated=0,break_count=0)},
        source_role='current owner-local CAL HOLDOUT aggregates',historical_raw_reads=0,thresholds_retuned=False)
    value['receipt_id']=sha(value)
    check('receipt_packet_roundtrip',decode_receipt(encode_receipt(value),original)==value)
    grant=copy.deepcopy(original)
    append_current_evidence(grant,[value],True)
    check('new_evidence_adds_only_observed_class',set(validated_scope(grant)['classes'])==set(original['lifecycle_certificate']['scope']['classes'])|{24})
    check('grant_keeps_initial_seal',certificate_binding_current(grant) and grant['lifecycle_certificate_sha256']==seal)
    book=copy.deepcopy(grant)
    rejected('receipt_replay_rejected',lambda:append_current_evidence(grant,[value],True))
    check('failed_append_atomic',digest(grant)==digest(book))
    for name,change in (
        ('insufficient_CAL',{'counts':{'24':dict(rows=10,activated=0,break_count=0)}}),
        ('new_FAR_conflict',{'counts':{'24':dict(rows=64,activated=1,break_count=0)}}),
        ('wrong_guard',{'guard_function_fingerprint':'wrong'}),
        ('old_CAL_read',{'historical_raw_reads':1}),
        ('threshold_retuned',{'thresholds_retuned':True}),
        ('bad_counts_schema',{'counts':{'24':None}}),
        ('bad_label_schema',{'counts':{24:dict(rows=64,activated=0,break_count=0)}})):
        bad=dict(value,**change)
        bad['receipt_id']=sha({k:v for k,v in bad.items() if k!='receipt_id'})
        rejected(name,lambda bad=bad:append_current_evidence(copy.deepcopy(original),[bad],True))
    requests=[dict(receiver=i,class_id=c,task=3) for c in (21,23) for i in range(4)]
    offers=[dict(donor=9,class_id=c,task=3,quality_lcb=.9,model_fingerprint='FIT') for c in (21,23)]
    groups={i:[9] for i in range(4)}
    alphas={i:{i:.5,9:.5} for i in range(4)}
    gates={i:dict(eligible=i!=0,reason='missing_owned_old_BASE_support' if i==0 else None) for i in range(4)}
    pairs,coverage,skips=schedule_requests(requests,offers,groups,alphas,3,19,gates,4,3)
    check('metadata_prefilter_applied',all(p['receiver']!=0 for p in pairs) and len(skips)==2)
    check('multiple_receiver_same_class',coverage[21]['selected_receivers']==2)
    check('unique_receiver_and_budget',len(pairs)==3 and len({p['receiver'] for p in pairs})==3)
    check('class_round_robin', [p['class_id'] for p in pairs]==[21,23,21])
    check('only_positive_live_edges',all(alphas[p['receiver']][p['donor']]>0 for p in pairs))
    check('no_HOLDOUT_fallback',all('no HOLDOUT fallback' in p['rule'] for p in pairs))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    result=dict(completed=all(checks.values()),checks=checks,mismatches=sum(not v for v in checks.values()),
        comparisons=len(checks),synthetic_receipts=True,native_training=False,final_test_opened=False)
    write_json(a.out,result)
    print(json.dumps(result,indent=2),flush=True)
    if not result['completed']:raise AssertionError(checks)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('audit_denice/appliance_production_smoke_local/task3_to4_v2/controlled/controlled_automatic_endpoint.pt'))
    p.add_argument('--out',type=Path,default=Path('audit_denice/appliance_empirical_integration/protocol_checks.json'))
    run(p.parse_args())
