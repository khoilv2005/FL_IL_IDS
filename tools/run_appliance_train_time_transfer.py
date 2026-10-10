"""Task1 prototype: functional-gap discovery by default; old ablation opt-in."""
import argparse
import copy
import gc
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from appliance.train_time_transfer import (RULES, stratified_cap, donor_proposal,
    receiver_integration, shadow, update_packet, read_update)
from appliance.runner import load_input
from appliance.selector import recorded_graph
from appliance.current_base_data import CurrentBaseData
from appliance.current_calibration_data import CurrentCalibrationData
from appliance.imported_route import stratified_roles, ROUTE_RULES
from appliance.transport import Transport
from appliance.config import Protocol
from appliance.state import complete_hash, write_json, digest
from appliance.evaluate import Predictor
from fed_learning.data.denice_clean_roles import CleanRoleData,file_sha256
from fed_learning.training.checkpoint_state import snapshot_denice_state
from fed_learning.training.denice_eval import _denice_routed_logits_with_episodes
from eval_checkpoint import _make_denice_client_model


@torch.no_grad()
def all_seen(model,router,x,seen,device):
    result=[]
    for start in range(0,len(x),RULES['batch_size']):
        batch=torch.as_tensor(x[start:start+RULES['batch_size']],device=device)
        logits,_=_denice_routed_logits_with_episodes(model,batch,router,seen,device,
            inference_policy='backbone_nomask')
        result.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(result) if result else np.empty(0,np.int64)


def counters(before,after,y,target):
    pos=y==target;neg=~pos
    correct_before=before==y;correct_after=after==y
    rescue=int((~correct_before&correct_after).sum());broken=int((correct_before&~correct_after).sum())
    delta=int(correct_after.sum()-correct_before.sum())
    if delta!=rescue-broken:raise AssertionError('Accuracy/rescue/break accounting failed')
    old=np.isin(y,list(range(6)))
    return dict(rows=len(y),positive_rows=int(pos.sum()),negative_rows=int(neg.sum()),
        accuracy_before=float(correct_before.mean()) if len(y) else None,
        accuracy_after=float(correct_after.mean()) if len(y) else None,
        target_recall_before=float((before[pos]==target).mean()) if pos.any() else None,
        target_recall=float((after[pos]==target).mean()) if pos.any() else None,
        negative_FAR=float((after[neg]==target).mean()) if neg.any() else None,
        negative_FAR_by_class={str(int(c)):float((after[y==c]==target).mean()) for c in np.unique(y[neg])},
        old_rows=int(old.sum()),old_accuracy_before=float(correct_before[old].mean()) if old.any() else None,
        old_accuracy_after=float(correct_after[old].mean()) if old.any() else None,
        old_false_positive_rate=float((after[old]==target).mean()) if old.any() else None,
        rescue=rescue,break_count=broken,correct_delta=delta,
        negative_break=int((correct_before&~correct_after&neg).sum()))


def views_metrics(models,routers,pool,target,seen,device):
    original,candidate=models;original_router,new_router=routers
    old=all_seen(original,original_router,pool['X'],seen,device)
    new=all_seen(candidate,new_router,pool['X'],seen,device)
    primary=counters(old,new,pool['y'],target)
    predictor=Predictor(seen,device,RULES['batch_size'])
    routed=counters(predictor(original,original_router,pool['X']),
        predictor(candidate,new_router,pool['X']),pool['y'],target)
    return dict(primary_all_seen=primary,secondary_existing_router=routed)


def held(view,classes):
    p=view.current_pool(view.client_id,'calibration',classes)
    indices=stratified_roles(p,ROUTE_RULES['seed']+view.client_id)['holdout']
    return dict(p,X=p['X'][indices],y=p['y'][indices],rows=p['rows'][indices])


def run_legacy(a):
    a.out.mkdir(parents=True,exist_ok=False)
    write_json(a.out/'completion.json',dict(completed=False))
    ckpt,hashes=load_input(a.checkpoint,1,19)
    config=ckpt['config'];role_sha=config['denice_data_roles_sha256']
    if config.get('denice_cl_method')!='legacy':raise ValueError('Legacy Task1 checkpoint required')
    roles=CleanRoleData(a.roles,source_data_dir=a.data)
    if file_sha256(a.roles/'role_manifest.json')!=role_sha:raise ValueError('Clean role authority changed')
    meta=json.loads((a.data/'metadata.json').read_text())
    taskmap={int(t):list(map(int,cs)) for t,cs in meta['task_structure']['task_classes'].items()}
    seen=sorted(taskmap[0]+taskmap[1]);current=taskmap[1]
    if seen!=list(range(12)) or sorted(ckpt['seen_classes'])!=seen:raise ValueError('Task1 class scope changed')
    groups,alphas=recorded_graph(ckpt,1,19)
    pairs=[dict(receiver=2,donor=62,class_id=6),dict(receiver=2,donor=64,class_id=8)]
    lock=dict(version=RULES['version'],rules=RULES,**hashes,task=1,round=19,pairs=pairs,
        role_manifest_sha256=role_sha,current_classes=current,seen_classes=seen,
        primary_inference='one receiver backbone; all-seen 0..11; same mask before/after',
        secondary_inference='existing binary_cosine route; class availability registered only; no router refit',
        fit_role='owned current BASE only, per endpoint',CAL_role='Task1 replay current CAL H, never old Task0 CAL',
        development_role='original FIT of seen tasks only; used exclusively after updates/thresholds locked',
        development_reused=True,pairs_selected_retrospectively_from_Task5_development=True,
        automatic_discovery_verified=False,real_time_prospective_run=False,
        imported_route=False,CME=False,encoder_update=False,strict_per_class_safety_claim=False,
        empirical_risk_control=True,production_runner_modified=False,final_test_opened=False,
        source_sha256={n:file_sha256(Path(n)) for n in ['appliance/train_time_transfer.py',
            'tools/run_appliance_train_time_transfer.py']})
    write_json(a.out/'protocol_before_BASE_or_CAL.json',lock)
    reports=[]
    for pair in pairs:
        started=time.time();i,j,c=(pair[k] for k in ('receiver','donor','class_id'))
        target=a.out/f'receiver_{i}_donor_{j}_class_{c}';target.mkdir()
        if j not in groups[i] or alphas[i].get(j,0)<=0:raise ValueError('No legitimate Task1 live graph edge')
        model,router=_make_denice_client_model(ckpt,i,a.device)
        donor,donor_router=_make_denice_client_model(ckpt,j,a.device)
        if int(model.unit_ranks['fc2'][c])!=0 or int(donor.unit_ranks['fc2'][c])<2:
            raise ValueError('Missing receiver slot/mature donor condition failed')
        if getattr(model,'appliance_guarded_head_entries',{}):raise ValueError('Receiver already has imported routes')
        before=complete_hash(model,router); donor_before=complete_hash(donor,donor_router)
        transport=Transport(target/'communication.jsonl',[(i,j),(j,i)],
            Protocol(max_incoming_bytes=32*1024*1024,max_outgoing_bytes=32*1024*1024).validate())
        alg=snapshot_denice_state(model,router)
        if alg['context_detector'].get('reference_input_memory'):
            raise ValueError('Raw reference memory cannot cross endpoint')
        capsule=dict(config=config,task=1,client_model_states={i:{k:v.detach().cpu() for k,v in model.state_dict().items()}},
            client_algorithm_states={i:{'denice':alg}})
        buffer=io.BytesIO();torch.save(capsule,buffer)
        delivered=transport.send(i,j,'train_time_receiver_function_capsule',buffer.getvalue())
        replica,replica_router=_make_denice_client_model(torch.load(io.BytesIO(delivered),map_location='cpu',weights_only=False),i,a.device)
        if complete_hash(replica,replica_router)!=before:raise ValueError('Inexact receiver capsule')
        rv=CurrentBaseData(a.base_store,i,1,role_sha);dv=CurrentBaseData(a.base_store,j,1,role_sha)
        dp=stratified_cap(dv.current_pool(j,'base',current),current,RULES['cap_BASE_per_class'],42+j)
        rp=stratified_cap(rv.current_pool(i,'base',current),current,RULES['cap_BASE_per_class'],42+i)
        write_json(target/'BASE_coordinates.json',dict(donor=dp['rows'],receiver=rp['rows'],
            donor_sha256=dp['partition_sha256'],receiver_sha256=rp['partition_sha256']))
        print(f'Train-time Task1: receiver={i}, donor={j}, class={c}, donor BASE={len(dp["y"])} receiver BASE={len(rp["y"])}',flush=True)
        proposal=donor_proposal(replica,dp,c,seen,a.device)
        packet=update_packet(proposal['weight'],proposal['bias'],dict(pair=pair,
            receiver_sha256=before,role_manifest_sha256=role_sha,task=1,fit_role='current donor BASE'))
        delivered=transport.send(j,i,'class_specific_trained_readout',packet)
        update=read_update(delivered)
        if update['metadata']['receiver_sha256']!=before:raise ValueError('Update receiver binding changed')
        integrated=receiver_integration(model,rp,c,seen,update['weight'],update['bias'],a.device)
        candidate,detector=shadow(model,router,c,1,integrated['weight'],integrated['bias'])
        write_json(target/'update_lock_before_CAL_development.json',dict(pair=pair,
            candidate_sha256=complete_hash(candidate,detector),proposal_trace=proposal['donor_trace'],
            receiver_trace=integrated['receiver_trace'],thresholds_tuned=False,
            update_frozen=True,current_CAL_opened=False,development_opened=False))
        torch.save(dict(weight=integrated['weight'],bias=integrated['bias'],metadata=update['metadata']),target/'trained_row.pt')
        # This replay opens only Task1's current CAL at each owner endpoint.
        cv={o:CurrentCalibrationData(a.cal_store,o,1,role_sha) for o in (i,j)}
        cal={o:views_metrics((model,candidate),(router,detector),held(v,current),c,seen,a.device) for o,v in cv.items()}
        dc=cal[j]['primary_all_seen'];rc=cal[i]['primary_all_seen']
        cal_pass=(dc['positive_rows']>=32 and rc['negative_rows']>=32 and
            dc['target_recall']>=.95 and rc['negative_FAR']<=.001 and
            all(v<=.001 for v in rc['negative_FAR_by_class'].values()) and
            rc['negative_break']+dc['negative_break']==0 and dc['rescue']>0)
        development={};old_before=old_after=old_rows=0;old_fa=0
        for owner in (i,j):
            x,y,rows=roles.client_role(owner,'fit')
            keep=np.flatnonzero(np.isin(y,seen))
            p=stratified_cap(dict(X=x[keep],y=y[keep],rows=rows[keep]),seen,
                RULES['cap_development_per_owner_class'],42+owner+9001)
            write_json(target/f'development_FIT_coordinates_{owner}.json',dict(owner=owner,role='FIT',rows=p['rows']))
            metrics=views_metrics((model,candidate),(router,detector),p,c,seen,a.device)
            development[owner]=metrics
            m=metrics['primary_all_seen'];n=m['old_rows']
            old_rows+=n
            if n:
                old_before+=round(m['old_accuracy_before']*n);old_after+=round(m['old_accuracy_after']*n)
                old_fa+=round(m['old_false_positive_rate']*n)
            del x,y,rows,p;gc.collect()
        old_drop=(old_before-old_after)/old_rows if old_rows else None
        old_far=old_fa/old_rows if old_rows else None
        risk_pass=bool(old_rows>=32 and old_drop<=RULES['max_development_old_accuracy_drop'] and
            old_far<=RULES['max_development_old_false_positive_rate'])
        changed=[k for k,v in model.state_dict().items() if not torch.equal(v,candidate.state_dict()[k])]
        if any(k not in ('fc2.weight','fc2.bias') for k in changed):raise ValueError('Unexpected prefix update')
        old=[k for k in seen if k!=c]
        if not torch.equal(model.fc2.weight[old],candidate.fc2.weight[old]):raise ValueError('Old head rows changed')
        if complete_hash(model,router)!=before or complete_hash(donor,donor_router)!=donor_before:
            raise ValueError('Live receiver or donor mutated')
        report=dict(pair=pair,current_CAL_pass=bool(cal_pass),development_risk_pass=risk_pass,
            useful_prototype=bool(cal_pass and risk_pass),production_install_authorized=False,
            shadow_decision='eligible_for_next_native_smoke' if cal_pass and risk_pass else 'rollback_do_not_integrate',
            current_CAL=cal,development=development,
            pooled_old_development=dict(rows=old_rows,accuracy_drop=old_drop,false_positive_rate=old_far,
                repeated_contents_not_independent=True),
            communication=transport.summary(),capability_bytes=len(packet),elapsed_seconds=time.time()-started,
            source_receiver_and_donor_unchanged=True,prefix_and_old_parameter_rows_unchanged=True,
            current_BASE_access={o:v.access_log for o,v in ((i,rv),(j,dv))},
            CAL_access={o:v.access_log for o,v in cv.items()},raw_examples_transmitted=0,
            old_raw_CAL_reads=0,imported_routes_created=0,final_test_opened=False,
            main_training_protocol_reproduction=False,population_FAR_claim=False)
        write_json(target/'result.json',report);reports.append(report)
        print(f'Train-time class={c}: CAL recall={dc["target_recall"]:.4f}, receiver FAR={rc["negative_FAR"]:.4f}, old drop={old_drop}, decision={report["shadow_decision"]}',flush=True)
        del model,router,donor,donor_router,replica,replica_router,candidate,detector,dp,rp;gc.collect()
    result=dict(completed=True,protocol=lock,reports=reports,
        useful_prototypes=sum(v['useful_prototype'] for v in reports),production_ready=False,
        full_training_started=False,final_test_opened=False)
    write_json(a.out/'completion.json',result)
    write_json(a.publish,result)


def run(a):
    if getattr(a, 'protocol', 'functional_gap') == 'legacy_all_seen':
        return run_legacy(a)
    from tools.run_appliance_functional_gap_transfer import run as run_functional
    return run_functional(a)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('checkpoint','roles','data','base-store','cal-store','out','publish'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--protocol',choices=('functional_gap','legacy_all_seen'),default='functional_gap',
        help='Functional native discovery is the default; legacy_all_seen reproduces the historical diagnostic only')
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):run(a)
