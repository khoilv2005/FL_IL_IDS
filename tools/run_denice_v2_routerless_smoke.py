"""Bounded routerless CIL pilot, one native round per task, never TEST."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from fed_learning.data.denice_clean_roles import CleanRoleData,file_sha256
from fed_learning.models.denice_model import DeNICEModel
from fed_learning.strategies.decentralized.denice_v2.contracts import encode,decode,state_hash
from fed_learning.strategies.decentralized.denice_v2.knowledge import ClassKnowledgeLedger
from fed_learning.strategies.decentralized.denice_v2.native import CurrentBase,consolidate_native_task,run_native_round
from fed_learning.strategies.decentralized.denice_v2.normalization import protected_bn_channels
from fed_learning.strategies.decentralized.denice_v2.router import capacity_receipt
from fed_learning.strategies.decentralized.denice_v2.routerless import capture_routerless,register_learned_classes,RouterlessPredictor
from fed_learning.strategies.decentralized.denice_v2.state import snapshot,restore,handoff


def prepare_roles(roles,cache):
    directory=cache/'routerless_v1_current_roles'
    directory.mkdir(parents=True,exist_ok=True)
    hashes={}
    for cid in (9,30):
        for role,cap in (('base',32),('validation',64)):
            x,y,rows=roles.client_role(cid,role)
            for task in (0,1):
                indices=np.concatenate([np.flatnonzero(y==c)[:cap] for c in range(task*6,task*6+6)])
                path=directory/f'task_{task}_client_{cid}_{role}.npz'
                np.savez(path,x=x[indices],y=y[indices],row_ids=rows[indices])
                hashes[path.name]=file_sha256(path)
            del x,y,rows
    return directory,hashes


def read_current(prepared,cid,role,task):
    directory,hashes=prepared
    path=directory/f'task_{task}_client_{cid}_{role}.npz'
    if file_sha256(path)!=hashes[path.name]:
        raise ValueError('Prepared role changed')
    with np.load(path,allow_pickle=False) as data:
        x,y,rows=data['x'],data['y'],data['row_ids']
    if not set(y.tolist())<=set(range(task*6,task*6+6)):
        raise ValueError('Noncurrent role rows')
    return torch.from_numpy(x),torch.from_numpy(y),rows,hashes[path.name]


def current_bases(prepared,task):
    return {cid:CurrentBase(cid,task,*read_current(prepared,cid,'base',task)) for cid in (9,30)}


def current_validation(prepared,task):
    parts=[read_current(prepared,cid,'validation',task)[:2] for cid in (9,30)]
    return torch.cat([p[0] for p in parts]),np.concatenate([p[1].numpy() for p in parts])


def from_confusion(confusion):
    support=confusion.sum(1)
    predicted=confusion.sum(0)
    tp=np.diag(confusion)
    recall=np.divide(tp,support,out=np.zeros_like(tp,dtype=float),where=support>0)
    f1=np.divide(2*tp,support+predicted,out=np.zeros_like(tp,dtype=float),where=(support+predicted)>0)
    return dict(rows=int(support.sum()),accuracy=float(tp.sum()/support.sum()),macro_f1=float(f1.mean()),
        label_scope=list(range(len(confusion))),missing_validation_classes=np.flatnonzero(support==0).tolist(),
        per_class={c:dict(support=int(support[c]),recall=float(recall[c]),f1=float(f1[c]),
            recall_observed=bool(support[c])) for c in range(len(confusion))},confusion=confusion.tolist())


def metrics(truth,pred,scope):
    if not len(truth) or not set(truth.tolist())<=set(range(scope)) or not set(pred.tolist())<=set(range(scope)):
        raise ValueError('Invalid development prediction/label scope')
    confusion=np.zeros((scope,scope),int)
    np.add.at(confusion,(truth,pred),1)
    return from_confusion(confusion)


def evaluate(state,x,y,scope):
    predictor=RouterlessPredictor(state)
    predictions,masked=predictor.predict(x)
    result=metrics(y,predictions,scope)
    seen=state.algorithm_state['seen_classes']
    excluded=[c for c in range(masked.shape[1]) if c not in seen]
    with torch.no_grad():
        raw=predictor.model(x).cpu().numpy()
    result['logit_checks']=dict(raw_all_finite=bool(np.isfinite(raw).all()),
        registered_finite=bool(np.isfinite(masked[:,seen]).all()),
        excluded_negative_infinity=bool(np.isneginf(masked[:,excluded]).all()),
        prediction_only_registered=bool(set(predictions.tolist())<=set(seen)))
    return result,predictions,masked


def group_recall(result,classes):
    available=[c for c in classes if result['per_class'][c]['support']]
    return dict(fixed_scope_mean=float(np.mean([result['per_class'][c]['recall'] for c in classes])),
        present_class_mean=float(np.mean([result['per_class'][c]['recall'] for c in available])) if available else None,
        present_classes=available)


def protection_exact(before,after):
    for name,mask in before.protected_parameters.items():
        if not torch.equal(before.model_state[name][mask],after.model_state[name][mask]):
            return False
    for layer in ('bn1','bn2','bn3'):
        mask=torch.tensor(protected_bn_channels(before,layer))
        for suffix in ('weight','bias','running_mean','running_var'):
            name=layer+'.'+suffix
            if not torch.equal(before.model_state[name][mask],after.model_state[name][mask]):
                return False
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    out,cache=Path(args.out),Path(args.cache)
    if out.exists():
        raise ValueError('Use new receipt path')
    started=time.monotonic()
    torch.set_num_threads(2);torch.manual_seed(42)
    roles=CleanRoleData(cache/'roles',cache/'dataset')
    if not json.loads((cache/'roles/role_lock.json').read_text())['locked_before_backbone_training']:
        raise ValueError('Roles not locked before training')
    shape=tuple(roles.manifest['input_shape'])
    prepared=prepare_roles(roles,cache)
    del roles
    root=Path(__file__).resolve().parents[1]
    protocol=root/'docs/DENICE_V2_ROUTERLESS_PROTOCOL_20261012_VI.md'
    report=dict(variant='routerless_v1',seed=42,protocol_sha256=file_sha256(protocol),test_read=False,
        raw_training_replay=False,distillation=False,hyperparameter_tuning=False,rounds_per_task=1,
        base_cap_per_class=32,validation_cap_per_class=64,membership_fixture=[9,30],
        full_scheduler=False,role_preparation_sha256=prepared[1],tasks={},checks={})
    state=capture_routerless(DeNICEModel(shape,34),members=(9,30),coordinator_id=9,cluster_uid='routerless-t0')
    ledger=ClassKnowledgeLedger({t:tuple(range(t*6,min(t*6+6,34))) for t in range(6)})
    print('Routerless: native Task0 round',flush=True)
    bases=current_bases(prepared,0)
    result,plan,packets,training=run_native_round(state,bases,tuple(range(6)),seed=42)
    state0,registration0=register_learned_classes(result.state,plan,packets)
    checkpoint0=cache/'routerless_v1_task0_model_state.json'
    checkpoint0.write_text(json.dumps(encode(snapshot(state0))),encoding='utf-8')
    x0,y0=current_validation(prepared,0)
    evaluation0,_,_=evaluate(state0,x0,y0,6)
    del x0,y0
    for p in packets:
        ledger.record(p.event,at_task=0,at_round=0)
    report['tasks'][0]=dict(training=training,registration=registration0,capacity=capacity_receipt(state0,()),
        development=evaluation0,checkpoint=dict(path=checkpoint0.name,sha256=file_sha256(checkpoint0),
            bytes=checkpoint0.stat().st_size,scope='model_class_registry_only_not_optimizer_resume'))
    mature=consolidate_native_task(state0)
    next_state=handoff([mature],members=(9,30),cluster_uid='routerless-t1',task=1,ledger=ledger)
    del bases,result,plan,packets
    gc.collect()
    print('Routerless: native Task1 round, current BASE only',flush=True)
    bases=current_bases(prepared,1)
    result,plan,packets,training=run_native_round(next_state,bases,tuple(range(6,12)),seed=42)
    state1,registration1=register_learned_classes(result.state,plan,packets)
    report['tasks'][1]=dict(training=training,registration=registration1,capacity=capacity_receipt(plan.reference,()))
    # Separate read-only cumulative evaluator. Historical VALIDATION is never
    # passed to native training, BN collection, class registration or any fit.
    x0,y0=current_validation(prepared,0)
    x1,y1=current_validation(prepared,1)
    old,pred0,logits0=evaluate(state1,x0,y0,12)
    new,pred1,logits1=evaluate(state1,x1,y1,12)
    cumulative=metrics(np.concatenate([y0,y1]),np.concatenate([pred0,pred1]),12)
    report['tasks'][1]['development']=dict(old=old,new=new,cumulative=cumulative,
        old_recall=group_recall(old,range(6)),new_recall=group_recall(new,range(6,12)))
    forgetting=evaluation0['accuracy']-old['accuracy']
    report['forgetting']=dict(old_accuracy_drop_pp=100*forgetting,same_old_validation_rows=len(y0),
        old_class_recall_drop_pp={c:100*(evaluation0['per_class'][c]['recall']-old['per_class'][c]['recall']) for c in range(6)})
    report['cross_task_competition']=dict(old_rows_predicted_new=int((pred0>=6).sum()),
        new_rows_predicted_old=int((pred1<6).sum()),old_rows=len(y0),new_rows=len(y1),
        median_new_minus_old_logit_on_old=float(np.median(logits0[:,[c for c in registration1['seen_classes'] if c>=6]].max(1)-logits0[:,registration0['seen_classes']].max(1))),
        median_new_minus_old_logit_on_new=float(np.median(logits1[:,[c for c in registration1['seen_classes'] if c>=6]].max(1)-logits1[:,registration0['seen_classes']].max(1))))
    report['checks']=dict(native_optimizer_both_tasks=all(t['training']['workers'][1]['steps']>0 for t in report['tasks'].values()),
        empty_client_zero_step=training['workers'][0]['steps']==0,
        no_router_or_dependency=not state1.routing_dependencies and not state1.router_state['activation_memory'] and not state1.router_state['context_layers'],
        protected_mature_exact=protection_exact(next_state,state1),
        loss_finite=all(np.isfinite(w['loss_mean']) for t in report['tasks'].values() for w in t['training']['workers'] if w['steps']),
        broadcast_consistent=all(t['training']['broadcast_state_identical'] and t['training']['broadcast_logits_identical'] for t in report['tasks'].values()),
        logit_masks_valid=all(all(m['logit_checks'].values()) for m in (evaluation0,old,new)),
        task0_seen_classes_retained=set(registration0['seen_classes'])<=set(registration1['seen_classes']))
    checkpoint=cache/'routerless_v1_model_state.json'
    checkpoint.write_text(json.dumps(encode(snapshot(state1))),encoding='utf-8')
    restored=restore(decode(json.loads(checkpoint.read_text(encoding='utf-8'))))
    a=RouterlessPredictor(state1).predict(x1[:16]);b=RouterlessPredictor(restored).predict(x1[:16])
    report['checks']['restore_prediction_exact']=all(np.array_equal(x,y) for x,y in zip(a,b))
    report['checks']['restore_state_exact']=state1.fingerprint()==restored.fingerprint()
    report['checkpoint']=dict(bytes=checkpoint.stat().st_size,sha256=file_sha256(checkpoint),scope='model_class_registry_only_not_optimizer_resume')
    truth=np.concatenate([y0,y1]);seen=state1.algorithm_state['seen_classes']
    expected=np.zeros((12,12),float)
    for c in range(12):
        expected[c,list(seen)]=(truth==c).sum()/len(seen)
    baseline=from_confusion(expected)
    report['uniform_registered_baseline']=dict(macro_f1=baseline['macro_f1'],accuracy=baseline['accuracy'],scope=12,
        definition='metrics of expected confusion from uniform registered-class predictions')
    report['quality_checks']=dict(old_accuracy_drop_at_most_10pp=forgetting<=.10,
        present_new_macro_recall_at_least_25pct=group_recall(new,range(6,12))['present_class_mean']>=.25,
        cumulative_macro_f1_above_uniform=cumulative['macro_f1']>baseline['macro_f1'])
    report['technical_pass']=all(report['checks'].values())
    report['quality_feasibility_pass']=all(report['quality_checks'].values())
    report['decision']='GO_BOUNDED_NEXT_PILOT' if report['technical_pass'] and report['quality_feasibility_pass'] else 'NO_GO_BOUNDED_PILOT'
    report['full_training_started']=False
    report['completed']=True
    report['elapsed_seconds']=time.monotonic()-started
    paths=[Path(__file__),protocol,*sorted((root/'fed_learning/strategies/decentralized/denice_v2').glob('*.py'))]
    report['source_sha256']={p.relative_to(root).as_posix():file_sha256(p) for p in paths}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(decision=report['decision'],technical_pass=report['technical_pass'],quality_checks=report['quality_checks'],
        task0_accuracy=evaluation0['accuracy'],task1_cumulative_accuracy=cumulative['accuracy'],task1_cumulative_macro_f1=cumulative['macro_f1'],
        old_accuracy=old['accuracy'],new_accuracy=new['accuracy'],forgetting_pp=100*forgetting),indent=2),flush=True)


if __name__=='__main__':
    main()
