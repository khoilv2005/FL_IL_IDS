"""Amended development experiment: positive-FIT quantile gates, no optimizer."""
import argparse
import copy
import gc
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from appliance.runner import load_input
from appliance.state import complete_hash
from fed_learning.data.denice_clean_roles import CleanRoleData,file_sha256
from tools.audit_candle_v1_two_tier import read,write,features,save_rows,digest
from tools.audit_candle_v1_head_unlock import (RULES as BASE_RULES,development,cache,head_recipes,
    candidate_prediction,stats,gate_values,combine,export_value,distributions)

RULES=dict(version='candle_positive_quantile_gate_v1',seed=20261022,
    amendment='Chosen after exploratory validation exposed overpermissive negative-only tie breaks',
    cutoff_quantiles=[0.,.01,.05,.1],primary_quantile=.05,
    primary_heads={'7':'direct_eta_1.0','65':'diagonal_moment_transport'},
    primary_gates={'7':'donor_proto','65':'receiver_proto'},
    evaluation_role='original META-FIT, scoring only, never used for threshold/model fitting',
    evaluation_cap_per_owner_class=256,current_CAL_FIT_only_for_thresholds=True,
    no_threshold_selection_on_META_FIT=True,no_optimizer=True,no_binary_memory_import=True,
    empirical_gate=dict(recall=.95,FAR=.001,breaks=0),
    prior_exposure='Method amended using earlier validation; different content-disjoint role evaluated once; not pristine scientific confirmation')


def evaluation_panel(roles,receiver,donor,learned,exclude):
    pools=[]
    for cid,classes in [(receiver,set(learned)-{24}),(donor,{24})]:
        x,y,_=roles.client_role(cid,'fit')
        ids=np.concatenate([np.flatnonzero(y==c)[:RULES['evaluation_cap_per_owner_class']] for c in sorted(classes)])
        pools.append(dict(x=x[ids],y=y[ids],hashes=[digest(v) for v in x[ids]]))
    panel=combine(pools)
    if set(panel['hashes']) & exclude:raise ValueError('META-FIT overlaps calibration content')
    return panel


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--inputs',type=Path,default=ROOT/'audit_denice/appliance_legacy_results11/inputs')
    p.add_argument('--roles',type=Path,default=ROOT/'audit_denice/appliance_legacy_results11/inputs/roles')
    p.add_argument('--data',type=Path,default=ROOT/'audit_denice/appliance_transfer_results9/inputs/dataset')
    p.add_argument('--store',type=Path,default=ROOT/'audit_denice/appliance_current_runtime_data_local/full100_v1/calibration_store')
    p.add_argument('--device',default='cuda')
    a=p.parse_args();a.task=4
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(20261022)
    started=time.monotonic()
    write(a.out/'protocol_before_data.json',dict(rules=RULES,script_sha256=file_sha256(__file__)))
    write(a.out/'completion.json',dict(completed=False))
    from eval_checkpoint import _make_denice_client_model
    ckpt,source=load_input(a.inputs/'checkpoint_task_4_all_rounds.zip',4,19)
    roles=CleanRoleData(a.roles,source_data_dir=a.data);role_sha=file_sha256(a.roles/'role_manifest.json')
    if ckpt['config']['denice_data_roles_sha256']!=role_sha:raise ValueError('Role identity changed')
    store=read(a.store/'calibration_store_manifest.json');classes=store['task_classes']['4']
    seen=sorted(map(int,ckpt['seen_classes']));debug=read(a.inputs/'training_artifacts/denice_debug_history.json')
    results=[]
    for cid in [7,65]:
        out=a.out/f'receiver_{cid}';out.mkdir()
        print('Prepare fixed positive-FIT rules, receiver',cid,flush=True)
        fit,selection=development(a,cid,11,role_sha,classes)
        model,router=_make_denice_client_model(ckpt,cid,a.device)
        donor,dr=_make_denice_client_model(ckpt,11,a.device)
        model.eval();donor.eval();before=(complete_hash(model,router),complete_hash(donor,dr))
        fit_cache=cache(model,router,fit['x'],seen,a.device);fit_cache['truth']=fit['y']
        donor_h,_=features(donor,dr,fit['x'][fit['y']==24],4,a.device)
        prototypes=dict(donor=donor_h.mean(0),receiver=fit_cache['h'][4][fit['y']==24].mean(0),
            threshold=fit_cache['h'][4][fit['y']!=24].mean(0)+fit_cache['h'][4][fit['y']!=24].std(0))
        heads=head_recipes(model,donor,fit_cache,donor_h)
        specs=[]
        for name in ['direct_eta_1.0','diagonal_moment_transport']:
            for gate,values in gate_values(fit_cache,heads[name],prototypes).items():
                for q in RULES['cutoff_quantiles']:
                    spec=dict(family='targeted_'+gate,gate=gate,tau=float(np.quantile(values[fit['y']==24],q)),gamma=None,score='raw')
                    specs.append(dict(head=name,quantile=q,spec=spec,
                        primary=(name==RULES['primary_heads'][str(cid)] and gate==RULES['primary_gates'][str(cid)] and q==RULES['primary_quantile'])))
        lock=dict(rules=RULES,source_hashes=source,specs=specs,META_FIT_opened=False,
            calibration_counts=dict(zip(*[list(map(str,v)) if i==0 else list(map(int,v)) for i,v in enumerate(np.unique(fit['y'],return_counts=True))])))
        write(out/'fixed_gate_lock.json',lock);lock_sha=file_sha256(out/'fixed_gate_lock.json')
        print('Lock primary:',next(v for v in specs if v['primary']),flush=True)
        learned=set()
        for event in debug:
            if event.get('type')=='task_start' and int(event['task'])<=4:
                rec=event['client_data'].get(str(cid))
                if rec:learned.update(map(int,rec['labels']))
        panel=evaluation_panel(roles,cid,11,learned,set(fit['hashes'])|set(selection['hashes']))
        values=cache(model,router,panel['x'],seen,a.device);values['truth']=panel['y']
        rows=[]
        for row in specs:
            pred=candidate_prediction(values,heads[row['head']],prototypes,row['spec'])
            m=stats(pred,panel['y'],values['pred'])
            rows.append(dict(**row,metrics=m,empirical_gate=m['recall']>=.95 and m['FAR']<=.001 and m['negative_break']==0))
        primary=next(r for r in rows if r['primary'])
        mechanism=stats(candidate_prediction(values,heads['availability'],prototypes,primary['spec']),panel['y'],values['pred'])
        if before!=(complete_hash(model,router),complete_hash(donor,dr)) or file_sha256(out/'fixed_gate_lock.json')!=lock_sha:
            raise ValueError('Reference model/router or fixed gate lock changed')
        result=dict(receiver=cid,donor=11,class_id=24,primary=primary,ablations=rows,
            original_weights_same_gate=mechanism,baseline=stats(values['pred'],panel['y'],values['pred']),
            diagnostics=distributions(values,panel['y'],heads['direct_eta_1.0'],prototypes),
            META_FIT_counts={str(int(c)):int(n) for c,n in zip(*np.unique(panel['y'],return_counts=True))},
            old_class_negative_rows=int((panel['y']<24).sum()),source_references_unchanged=True,
            no_final_TEST_read=True,no_optimizer=True,no_installation=True,lock_sha256=lock_sha,
            limitation='Development-only: donor target domain; no online historical-risk authorization or population safety certificate')
        write(out/'result.json',result)
        write(out/'primary_packet.json',export_value(dict(class_id=24,context_id=4,
            head=heads[primary['head']],prototypes=prototypes,spec=primary['spec'],source=source,
            authorization='offline diagnostic only')))
        rows_flat=[dict(head=r['head'],gate=r['spec']['gate'],q=r['quantile'],tau=r['spec']['tau'],
            primary=r['primary'],gate_pass=r['empirical_gate'],**r['metrics']) for r in rows]
        save_rows(out/'fixed_ablation_results.csv',rows_flat)
        results.append(result)
        print('META-FIT PRIMARY',cid,primary['metrics'],'gate',primary['empirical_gate'],flush=True)
        print('PREDECLARED ABLATIONS PASS',cid,sum(r['empirical_gate'] for r in rows),'/',len(rows),flush=True)
        del model,router,donor,dr,fit_cache,values
        gc.collect();torch.cuda.empty_cache()
    write(a.out/'completion.json',dict(completed=True,seconds=time.monotonic()-started,rules=RULES,
        source_hashes=source,results=results,final_TEST_read=False,backbone_training=False,head_training=False))
    print('Completed:',a.out/'completion.json',flush=True)


if __name__=='__main__':main()
