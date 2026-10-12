"""Fresh CNN coordinate router; same cumulative CE/budget as routerless pilot."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from fed_learning.data.denice_clean_roles import file_sha256
from fed_learning.models.denice_model import DeNICEModel
from fed_learning.strategies.decentralized.denice_v2.contracts import encode,decode,state_hash
from fed_learning.strategies.decentralized.denice_v2.coordinate_router import fresh_coordinate_state
from fed_learning.strategies.decentralized.denice_v2.native import consolidate_native_task,run_native_round
from fed_learning.strategies.decentralized.denice_v2.knowledge import ClassKnowledgeLedger
from fed_learning.strategies.decentralized.denice_v2.router import ClusterRouter,stage_cluster_router,probe_router_capacity
from fed_learning.strategies.decentralized.denice_v2.state import handoff,snapshot,restore
from tools.run_denice_v2_routerless_smoke import read_current,metrics,protection_exact,group_recall
from tools.audit_denice_v2_routerless_old_logits import masked_argmax


POLICY='cnn_coordinate_router_v1'


def bases(prepared,task):
    from fed_learning.strategies.decentralized.denice_v2.native import CurrentBase
    return {cid:CurrentBase(cid,task,*read_current(prepared,cid,'base',task)) for cid in (9,30)}


def validation(prepared,task):
    rows=[read_current(prepared,cid,'validation',task) for cid in (9,30)]
    return torch.cat([p[0] for p in rows]),np.concatenate([p[1].numpy() for p in rows])


def infer(state,x,target):
    predictor=ClusterRouter(state)
    predictions,tasks,scores,bits=predictor.predict(x)
    with torch.no_grad():
        logits,acts=predictor.model.get_output_and_context_activations(x)
    return predictions,tasks,logits.numpy(),acts[target].numpy(),bits


def save_state(state,path):
    path.write_text(json.dumps(encode(snapshot(state))),encoding='utf-8')
    return dict(file=path.name,sha256=file_sha256(path),bytes=path.stat().st_size)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',required=True)
    parser.add_argument('--feature-layer',choices=('conv2','conv3'),default='conv2')
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    cache,out,target=Path(args.cache),Path(args.out),args.feature_layer
    if out.exists():raise ValueError('Use new receipt path')
    started=time.monotonic();torch.set_num_threads(2);torch.manual_seed(42)
    pilot_path=Path('artifacts/denice_v2_routerless_smoke_20261012.json')
    previous=json.loads(pilot_path.read_text(encoding='utf-8'))
    prepared=(cache/'routerless_v1_current_roles',previous['role_preparation_sha256'])
    # Reuse pre-split locked files, never rerun role sampling or read multi-task BASE.
    data=bases(prepared,0)
    shape=tuple(data[9].x.shape[1:])
    protocol=Path(__file__).resolve().parents[1]/'docs/DENICE_V2_CNN_COORDINATE_PROTOCOL_20261012_VI.md'
    report=dict(policy=POLICY,feature_layer=target,protocol_sha256=file_sha256(protocol),seed=42,
        classification_objective='cumulative_registered_ce_v1',base_cap=32,validation_cap=64,
        rounds_per_task=1,test_read=False,historical_base_cal_read=False,distillation=False,hyperparameter_tuning=False,
        role_files=previous['role_preparation_sha256'],full_scheduler=False,tasks={},checks={})
    state=fresh_coordinate_state(DeNICEModel(shape,34),feature_layer=target,members=(9,30),coordinator_id=9,
        cluster_uid=f'coordinate-{target}-t0')
    ledger=ClassKnowledgeLedger({t:tuple(range(t*6,min(t*6+6,34))) for t in range(6)})
    print(f'Coordinate {target}: native T0',flush=True)
    result,plan,packets,training=run_native_round(state,data,tuple(range(6)),seed=42)
    proposal=stage_cluster_router(result.state,plan,packets,data,feature_policy=POLICY)
    report['tasks'][0]=dict(training=training,router=proposal.receipt,probe=probe_router_capacity(result.state,POLICY))
    if not proposal.receipt['capacity']['passed']:
        report['decision']='NO_GO_CAPACITY'
    else:
        state0=proposal.commit()
        report['tasks'][0]['checkpoint']=save_state(state0,cache/f'coordinate_{target}_task0.json')
        x0,y0=validation(prepared,0)
        pred0,_,_,activations0,bits0=infer(state0,x0,target)
        eval0=metrics(y0,pred0,6)
        report['tasks'][0]['development']=eval0
        memory_hash=state_hash(state0.router_state['activation_memory'][0])
        thresholds_hash=state_hash(state0.router_state['binarize_thresholds'])
        for p in packets:ledger.record(p.event,at_task=0,at_round=0)
        next_state=handoff([consolidate_native_task(state0)],members=(9,30),cluster_uid=f'coordinate-{target}-t1',task=1,ledger=ledger)
        del x0,y0,data,result,plan,packets,proposal;gc.collect()
        data=bases(prepared,1)
        witness=torch.cat([b.x[:4] for b in data.values()])
        current_features=infer(next_state,witness,target)[3]
        print(f'Coordinate {target}: native T1',flush=True)
        result,plan,packets,training=run_native_round(next_state,data,tuple(range(6,12)),seed=42)
        proposal=stage_cluster_router(result.state,plan,packets,data,feature_policy=POLICY)
        final=proposal.commit()
        report['tasks'][1]=dict(training=training,router=proposal.receipt,probe=probe_router_capacity(plan.reference,POLICY),
            checkpoint=save_state(final,cache/f'coordinate_{target}_task1.json'))
        x0,y0=validation(prepared,0);x1,y1=validation(prepared,1)
        a,t0,z0,features0,new_bits0=infer(final,x0,target)
        b,t1,z1,_,_=infer(final,x1,target)
        mask=final.router_state['stable_feature_mask'];offset=64 if target=='conv2' else 192
        chosen=mask[offset:offset+features0.shape[1]]
        history_drift=float(np.abs(features0[:,chosen]-activations0[:,chosen]).max())
        witness_after=infer(final,witness,target)[3]
        current_drift=float(np.abs(witness_after[:,chosen]-current_features[:,chosen]).max())
        truth=np.concatenate([y0,y1]);true_tasks=np.concatenate([np.zeros(len(y0),int),np.ones(len(y1),int)])
        predicted_tasks=np.concatenate([t0,t1])
        predicted=np.concatenate([a,b]);logits=np.concatenate([z0,z1])
        old_classes=final.router_state['episode_classes'][0];new_classes=final.router_state['episode_classes'][1]
        cumulative=masked_argmax(logits,final.algorithm_state['seen_classes'])
        oracle=np.concatenate([masked_argmax(z0,old_classes),masked_argmax(z1,new_classes)])
        actual_metrics=metrics(truth,predicted,12);cum_metrics=metrics(truth,cumulative,12);oracle_metrics=metrics(truth,oracle,12)
        actual_old=metrics(y0,a,12);actual_new=metrics(y1,b,12)
        recalls=[float((predicted_tasks[true_tasks==t]==t).mean()) for t in (0,1)]
        confusion=[[int(((true_tasks==t)&(predicted_tasks==p)).sum()) for p in (0,1)] for t in (0,1)]
        report['routing']=dict(confusion=confusion,recall_by_task=recalls,balanced_accuracy=float(np.mean(recalls)),
            pooled_accuracy=float((predicted_tasks==true_tasks).mean()),ground_truth_only_for_metrics=True)
        report['classification']=dict(actual=actual_metrics,cumulative=cum_metrics,oracle_diagnostic=oracle_metrics,
            actual_old=actual_old,actual_new=actual_new,old_only_diagnostic=metrics(y0,masked_argmax(z0,old_classes),6),
            old_recall=group_recall(actual_old,range(6)),new_recall=group_recall(actual_new,range(6,12)),
            rescue=int(((cumulative!=truth)&(predicted==truth)).sum()),break_count=int(((cumulative==truth)&(predicted!=truth)).sum()))
        report['forgetting_pp']=100*(eval0['accuracy']-actual_old['accuracy'])
        report['retention_by_class']={c:dict(t0_recall=eval0['per_class'][c]['recall'],
            actual_t1_recall=actual_old['per_class'][c]['recall'],oracle_old_t1_recall=report['classification']['old_only_diagnostic']['per_class'][c]['recall'],
            validation_support=eval0['per_class'][c]['support']) for c in range(6)}
        restored=restore(decode(json.loads((cache/f'coordinate_{target}_task1.json').read_text(encoding='utf-8'))))
        aa=ClusterRouter(final).predict(x1[:16]);bb=ClusterRouter(restored).predict(x1[:16])
        report['checks']=dict(capacity=all(v['router']['capacity']['passed'] for v in report['tasks'].values()),
            native_optimizer=training['workers'][1]['steps']>0,empty_client_zero_steps=training['workers'][0]['steps']==0,
            protected_mature_exact=protection_exact(next_state,final),
            history_activation_drift_le_1e6=history_drift<=1e-6,current_activation_drift_le_1e6=current_drift<=1e-6,
            history_bits_exact=np.array_equal(bits0,new_bits0),routing_fingerprint_exact=final.routing_space_hash==state0.routing_space_hash,
            memory_exact=state_hash(final.router_state['activation_memory'][0])==memory_hash,
            thresholds_exact=state_hash(final.router_state['binarize_thresholds'])==thresholds_hash,
            restore_exact=all(np.array_equal(x,y) for x,y in zip(aa,bb)))
        report['activation_drift']=dict(historical_validation=history_drift,current_base_witness=current_drift)
        report['minimum_routing_feasibility']=float(np.mean(recalls))>=.60 and min(recalls)>=.25
        report['quality_checks']=dict(routing_balanced_ge_75pct=float(np.mean(recalls))>=.75,
            each_task_recall_ge_60pct=min(recalls)>=.60,actual_accuracy_beats_own_cumulative=actual_metrics['accuracy']>cum_metrics['accuracy'],
            actual_macro_f1_beats_own_cumulative=actual_metrics['macro_f1']>cum_metrics['macro_f1'],old_accuracy_loss_le_10pp=report['forgetting_pp']<=10)
        report['technical_pass']=all(report['checks'].values())
        report['decision']='GO_BOUNDED_P3_PREPARATION' if report['technical_pass'] and all(report['quality_checks'].values()) else 'NO_GO_BOUNDED_PILOT'
    report['completed']=True;report['full_training_started']=False;report['elapsed_seconds']=time.monotonic()-started
    root=Path(__file__).resolve().parents[1]
    paths=[Path(__file__).resolve(),protocol,*sorted((root/'fed_learning/strategies/decentralized/denice_v2').glob('*.py'))]
    report['source_sha256']={p.relative_to(root).as_posix():file_sha256(p) for p in paths}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(decision=report['decision'],routing=report.get('routing'),
        actual_accuracy=report.get('classification',{}).get('actual',{}).get('accuracy'),
        oracle_accuracy=report.get('classification',{}).get('oracle_diagnostic',{}).get('accuracy'),
        forgetting_pp=report.get('forgetting_pp'),checks=report['checks']),indent=2),flush=True)


if __name__=='__main__':main()
