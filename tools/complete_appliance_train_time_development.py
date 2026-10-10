"""Complete old-class development on graph witnesses, without refit/CAL reread.

Consumes sealed run01 updates. Witnesses are chosen from role counts and the
Task1 positive-alpha graph before reading any FIT outcomes. Historical FIT is
offline qualification only, not acceptance, runtime replay or CAL evidence.
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from appliance.train_time_transfer import RULES,stratified_cap,shadow
from appliance.runner import load_input
from appliance.selector import recorded_graph
from appliance.state import complete_hash,write_json
from fed_learning.data.denice_clean_roles import CleanRoleData,file_sha256
from eval_checkpoint import _make_denice_client_model
from tools.run_appliance_train_time_transfer import views_metrics


def run(a):
    a.out.mkdir(parents=True,exist_ok=False)
    write_json(a.out/'completion.json',dict(completed=False))
    prior=json.loads((a.prepared/'completion.json').read_text())
    if not prior['completed']:raise ValueError('Prepared transfer unfinished')
    ckpt,hashes=load_input(a.checkpoint,1,19)
    if any(hashes[k]!=prior['protocol'][k] for k in hashes):raise ValueError('Task1 authority changed')
    roles=CleanRoleData(a.roles,source_data_dir=a.data)
    if file_sha256(a.roles/'role_manifest.json')!=prior['protocol']['role_manifest_sha256']:
        raise ValueError('Role authority changed')
    groups,alphas=recorded_graph(ckpt,1,19)
    counts=roles.manifest['clients'];remaining=set(range(6));owners=[]
    peers=list(groups[2])
    while remaining and peers:
        def supported(owner):
            v=counts[str(owner)]['role_class_counts']['fit']
            return {c for c in remaining if v.get(str(c),0)>0}
        def rank(owner):
            v=counts[str(owner)]['role_class_counts']['fit']
            return (-len(supported(owner)),-sum(min(RULES['cap_development_per_owner_class'],v.get(str(c),0)) for c in supported(owner)),owner)
        owner=min(peers,key=rank)
        if not supported(owner):break
        owners.append(owner);remaining-=supported(owner);peers.remove(owner)
    protocol=dict(kind='metadata-selected graph-witness old FIT qualification',**hashes,
        prepared_completion_sha256=file_sha256(a.prepared/'completion.json'),
        rules=RULES,witness_owners=owners,missing_old_classes_by_role_metadata=sorted(remaining),
        selection='greedy old FIT class coverage; capped row capacity; owner ID tie',
        peer_alphas={str(i):alphas[2][i] for i in owners},
        historical_CAL_reads=0,new_CAL_reads=0,refitting=False,hyperparameters_changed=False,
        historical_FIT_is_offline_diagnostic=True,current_CAL_results_reused_not_replication=True,
        raw_examples_sent=False,independence_claim=False,production_install_authorized=False,
        population_FAR_claim=False,final_test_opened=False)
    write_json(a.out/'protocol_before_witness_FIT.json',protocol)
    reports=[]
    for r in prior['reports']:
        pair=r['pair'];i,j,c=(pair[k] for k in ('receiver','donor','class_id'))
        target=a.prepared/f'receiver_{i}_donor_{j}_class_{c}'
        update=torch.load(target/'trained_row.pt',map_location='cpu',weights_only=False)
        model,router=_make_denice_client_model(ckpt,i,'cpu')
        candidate,detector=shadow(model,router,c,1,update['weight'],update['bias'])
        locked=json.loads((target/'update_lock_before_CAL_development.json').read_text())
        if complete_hash(candidate,detector)!=locked['candidate_sha256']:
            raise ValueError('Previously evaluated candidate changed')
        baseline=complete_hash(model,router); frozen=complete_hash(candidate,detector)
        results={};n=correct0=correct1=fp=0;present=set();per_pool_pass=True
        for owner in owners:
            x,y,rows=roles.client_role(owner,'fit')
            chosen=np.flatnonzero(np.isin(y,list(range(6))))
            pool=stratified_cap(dict(X=x[chosen],y=y[chosen],rows=rows[chosen]),list(range(6)),
                RULES['cap_development_per_owner_class'],42+owner+9107)
            write_json(a.out/f'witness_{owner}_class_{c}_coordinates.json',dict(owner=owner,role='old FIT',rows=pool['rows']))
            metrics=views_metrics((model,candidate),(router,detector),pool,c,list(range(12)),'cpu')
            per_class={str(int(k)):int((pool['y']==k).sum()) for k in np.unique(pool['y'])}
            present.update(map(int,per_class))
            m=metrics['primary_all_seen'];rows_count=m['rows']
            n+=rows_count;correct0+=round(m['accuracy_before']*rows_count)
            correct1+=round(m['accuracy_after']*rows_count);fp+=round(m['negative_FAR']*rows_count)
            per_pool_pass &= all(v<=RULES['max_development_old_false_positive_rate']
                for v in m['negative_FAR_by_class'].values())
            results[owner]=dict(metrics=metrics,class_rows=per_class,
                classes_below32=[int(k) for k,v in per_class.items() if v<32])
            print(f'Train-time frozen class={c}: old FIT witness={owner}, rows={rows_count}, break={m["break_count"]}, FAR={m["negative_FAR"]:.6f}',flush=True)
            del x,y,rows,pool;gc.collect()
        if complete_hash(model,router)!=baseline or complete_hash(candidate,detector)!=frozen:
            raise ValueError('Qualification mutated original or frozen shadow')
        drop=(correct0-correct1)/n if n else None;far=fp/n if n else None
        passed=bool(n>=32 and not remaining and drop<=RULES['max_development_old_accuracy_drop'] and
            far<=RULES['max_development_old_false_positive_rate'] and per_pool_pass)
        summary=dict(rows=n,accuracy_before=correct0/n if n else None,accuracy_after=correct1/n if n else None,
            accuracy_drop=drop,false_positives=fp,false_positive_rate=far,
            present_old_classes=sorted(present),missing_old_classes=sorted(set(range(6))-present),
            all_observed_owner_class_FAR_pass=bool(per_pool_pass),empirical_development_risk_pass=passed,
            strict_safety_certified=False,raw_CAL_read=False)
        reports.append(dict(pair=pair,current_CAL=r['current_CAL'],
            current_CAL_pass=r['current_CAL_pass'],initial_development=r['development'],
            old_witness_development=results,old_risk=summary,
            useful_prototype=bool(passed and r['current_CAL_pass']),
            candidate_function_binding_verified=True,
            transfer_communication=r['communication'],capability_bytes=r['capability_bytes'],
            qualification_cost_separate='local offline emulator; endpoint capsules/aggregate wire not implemented, no production communication claim',
            decision='eligible_for_next_native_smoke' if passed and r['current_CAL_pass'] else 'rollback_do_not_integrate',
            production_install_authorized=False,old_raw_CAL_reads=0))
        del model,router,candidate,detector;gc.collect()
    result=dict(completed=True,protocol=protocol,reports=reports,
        useful_prototypes=sum(r['useful_prototype'] for r in reports),
        native_survival_verified=False,production_ready=False,full_training_started=False,
        final_test_opened=False)
    write_json(a.out/'completion.json',result);write_json(a.publish,result)
    print(json.dumps(dict(completed=True,useful_prototypes=result['useful_prototypes']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('prepared','checkpoint','roles','data','out','publish'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):run(a)
