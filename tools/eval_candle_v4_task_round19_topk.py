"""Frozen T0-T5 round19 cumulative TEST, actual Top1 vs Top2 class/expert union."""
import argparse
import csv
import gc
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'candle_v4_runtime'))
from candle_v4_historical_repair import copy_member,member,sha256
from eval_checkpoint import _make_denice_client_model
from fed_learning.training.candle_v4_original_train import OriginalTrainService
from fed_learning.training.candle_v4_continuous_runtime import ContinuousRoutingView,validate_task_support
from fed_learning.training.candle_v4_bank import score_matrix,fuse,base_predictions
from fed_learning.training.candle_v4_shadow import digest
from fed_learning.training.candle_v4_terminal_test import statistics
from fed_learning.training.denice_eval import _denice_routed_logits_with_episodes,_allowed_classes_for_episode,_mask_logits_to_allowed
from fed_learning.training.decentralized_denice_il import _partition_test_data_by_client

SOURCE='3731fe1cee41c693fbfb7327b7fc3b2e86a0f3ab'
PROTOCOL='candle_v4_actual_top2_union_round19_v1'


def prepare(results19,results21,cache):
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True);paths={}
    with zipfile.ZipFile(results21) as z:
        lock=z.read(member(z,'candle_v4_run_lock.json'))
        original=json.loads(z.read(member(z,'candle_v4_final_test.json')))
        if not original['completed'] or (original['task'],original['round'])!=(5,19):raise ValueError('Need completed results21')
    if json.loads(lock)['source_commit']!=SOURCE:raise ValueError('Unexpected training source')
    for task in range(6):
        with zipfile.ZipFile(results19 if task<5 else results21) as outer:
            if outer.read(member(outer,'candle_v4_run_lock.json'))!=lock:raise ValueError('Input ZIPs are different runs')
            target=cache/f'continuation_state_task_{task}.pt';receipt=cache/f'extraction_{task}.json'
            if target.exists() and receipt.exists() and sha256(target)==json.loads(receipt.read_text())['sha256']:
                paths[task]=target;continue
            nested=cache/f'input_task_{task}.zip'
            copy_member(outer,member(outer,f'checkpoint_task_{task}_all_rounds.zip'),nested)
            try:
                with zipfile.ZipFile(nested) as inner:
                    manifest=json.loads(inner.read('checkpoint_archive_manifest.json'))
                    name=f'continuation_state_task_{task}.pt'
                    if (manifest.get('task_id')!=task or manifest.get('completed_rounds')!=list(range(20))
                            or manifest.get('continuation_checkpoint')!=name):raise ValueError('Not a complete round19 consolidated boundary')
                    expected=manifest['checksums'][name]
                    paths[task]=copy_member(inner,name,target,expected)
                    receipt.write_text(json.dumps(dict(sha256=expected,task=task)))
            finally:nested.unlink(missing_ok=True)
        print(f'Checkpoint T{task}/R19 verified',flush=True)
    return paths,original


def candidates(top1,scores,available):
    """Keep production Top1; choose second only among task memories that exist."""
    top1=np.asarray(top1,dtype=np.int64);scores=np.asarray(scores)
    result=np.full((len(top1),2),-1,dtype=np.int64);result[:,0]=top1
    for row,first in enumerate(top1):
        valid=[t for t in sorted(available) if t!=first]
        if valid:result[row,1]=max(valid,key=lambda t:(scores[row,t],-t))
    return result


def fuse_candidates(native,scores,classes,births,selected):
    active=(selected[:,:,None]==np.asarray(births)[None,None,:]).any(axis=1)
    return fuse(native,np.where(active,scores,-1.),[.5]*len(classes),classes)


def predict_pair(service,cid,model,det,inputs,seen):
    """No target labels/task IDs accepted here. Adapter remains production Top1."""
    view=ContinuousRoutingView(det,service.continuous_routers[cid]);device=next(model.parameters()).device
    logits,top1=_denice_routed_logits_with_episodes(model,inputs.to(device),view,seen,str(device),inference_policy='pred_adapter_nomask')
    with torch.no_grad():
        acts=model.get_context_activations_per_sample(inputs.to(device))
    encoded=view.binarize_layer_activations({k:v.cpu().numpy() for k,v in acts.items()})
    check,scores=view.memory.predict(encoded)
    if not np.array_equal(top1,check):raise ValueError('Router changed between paths')
    selected=candidates(top1,scores,set(view.memory.class_to_task.values()))
    keys=sorted(k for k,status in service.status['coverage'].items() if k[0]==cid and status=='ACTIVE')
    classes=[k[1] for k in keys];births=np.asarray([service.entries[k]['task'] for k in keys],dtype=np.int64)
    expert_scores=score_matrix({k[1]:service.engines['coverage'][k] for k in keys},classes,inputs.cpu())
    predictions={};counts={}
    for k in (1,2):
        sel=selected[:,:k];allow=np.zeros((len(inputs),34),dtype=bool)
        for row,tasks in enumerate(sel):
            for task in tasks:
                if task>=0:allow[row,_allowed_classes_for_episode(det,task,seen,34)]=True
        masked=_mask_logits_to_allowed(logits,torch.as_tensor(allow,device=device))
        fallback=masked.argmax(1).cpu().numpy()
        pred,fire=fuse_candidates(fallback,expert_scores,classes,births,sel)
        predictions[f'top{k}']=pred
        counts[f'top{k}']=dict(triggered=int((fire>0).sum()),multiple_triggers=int((fire>1).sum()),overrides=int((pred!=fallback).sum()))
    return predictions,selected,counts


def evaluate_task(state,task,x,y,original,out):
    saved=state['candle_v4_shadow_state']
    if (state['meta'].get('completed_task')!=task or state['meta'].get('boundary')!='task'
            or list(saved['boundary'])!=[task,19] or saved['version']!='candle_v4_continuous_bootstrap_v8'):
        raise ValueError('Incorrect task-boundary continuation')
    service=OriginalTrainService(saved['spec'],saved);bank_before=digest(service.state_dict())
    ids=sorted(state['client_model_states']);seen=list(range(min(34,6*(task+1))))
    include=y<len(seen);xx=x[include];yy=y[include]
    shards,partition=_partition_test_data_by_client(xx,yy,ids,seed=42+104729*task,lazy_features=True)
    del xx,yy
    cms={k:np.zeros((34,34),dtype=np.int64) for k in ('top1','top2')};clients={}
    routecm=np.zeros((6,7),dtype=np.int64);top2hits=0;rescue=0;broken=0;started=time.perf_counter()
    for i,cid in enumerate(ids,1):
        model,det=_make_denice_client_model(state,cid,'cuda');model.eval();before=digest(model.state_dict())
        validate_task_support(service.continuous_routers[cid],det)
        if any(t>task for t in service.continuous_routers[cid].class_to_task.values()):raise ValueError('Future task in checkpoint memory')
        shard=shards[cid];local={k:np.zeros_like(cm) for k,cm in cms.items()}
        lroute=np.zeros_like(routecm);lhits=0;lr=0;lb=0
        for start in range(0,len(shard['y_test']),512):
            inputs=shard['X_test'][start:start+512].reshape(-1,39).cpu();truth=shard['y_test'][start:start+512].numpy()
            with torch.no_grad():pred,selected,counts=predict_pair(service,cid,model,det,inputs,seen)
            if start==0:
                ref,ep=base_predictions(model,ContinuousRoutingView(det,service.continuous_routers[cid]),inputs,seen,True)
                ref,_=service.infer(cid,'coverage',ref,inputs,predicted_tasks=ep)
                if not np.array_equal(ref,pred['top1']):raise ValueError('Top1 differs from production')
            for k in cms:np.add.at(local[k],(truth,pred[k]),1)
            np.add.at(lroute,(truth//6,selected[:,0]),1)
            lhits+=int((selected==truth[:,None]//6).any(1).sum())
            lr+=int(((pred['top2']==truth)&(pred['top1']!=truth)).sum())
            lb+=int(((pred['top2']!=truth)&(pred['top1']==truth)).sum())
        if digest(model.state_dict())!=before:raise ValueError('Native weights changed')
        if task==5 and local['top1'].tolist()!=original['per_client'][str(cid)]['V4']['confusion_matrix']:
            raise ValueError('T5 Top1 matrix differs from results21')
        for k in cms:cms[k]+=local[k]
        routecm+=lroute;top2hits+=lhits;rescue+=lr;broken+=lb
        clients[cid]=dict(metrics={k:statistics(v) for k,v in local.items()},rescue=lr,break_count=lb,
                          top1_route_correct=int(np.trace(lroute[:,:6])),top2_task_hits=lhits)
        del model,det,shards[cid]
        if i%10==0 or i==len(ids):print(f'T{task}/R19 eval {i}/{len(ids)} clients; rows={int(cms["top1"].sum())}',flush=True)
    if bank_before!=digest(service.state_dict()):raise ValueError('Expert/router/optimizer state changed')
    n=int(cms['top1'].sum())
    if n!=int(include.sum()) or cms['top2'].sum()!=n:raise ValueError('Missing test rows')
    result=dict(completed=True,protocol=PROTOCOL,task=task,round=19,after_consolidation=True,rows=n,
        classes=seen,metrics={k:statistics(v) for k,v in cms.items()},per_client=clients,partition=partition,
        top1_route_accuracy=float(np.trace(routecm[:,:6])/n),top2_task_recall=top2hits/n,
        route_confusion=routecm.tolist(),rescue=rescue,break_count=broken,net_rescue=rescue-broken,
        macro_f1_policy='Fixed34 labels as native reports; additional seen-class macro-F1 supplied',
        rule='Actual Top2 union; same Top1 adapter; expert threshold0.5; no true labels used in predictions',
        no_training=True,no_refit=True,no_threshold_tuning=True,frozen_state_verified=True,
        t5_top1_exact_results21_match=task==5,wall_seconds=time.perf_counter()-started)
    for m in result['metrics'].values():m['macro_f1_seen_classes']=float(np.mean(m['per_class_f1'][:len(seen)]))
    (out/f'task_{task}_round19_topk.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--results19',required=True);p.add_argument('--results21',required=True)
    p.add_argument('--data-dir',required=True);p.add_argument('--cache',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    if not Path('/kaggle/working').is_dir() or not torch.cuda.is_available():raise RuntimeError('Full evaluation is Kaggle GPU only')
    import sklearn
    if sklearn.__version__!='1.6.1':raise RuntimeError('Use pinned sklearn1.6.1')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Use a fresh output directory')
    paths,original=prepare(a.results19,a.results21,a.cache)
    from fed_learning.data.incremental_loader import IncrementalDataLoader
    data=Path(a.data_dir);roots=[data] if (data/'metadata.json').exists() else [m.parent for m in data.rglob('metadata.json')]
    roots=[d for d in roots if (d/'global_test_data.npz').exists()]
    if len(roots)!=1:raise ValueError('Expected one original dataset')
    x,y=IncrementalDataLoader(str(roots[0])).get_full_test_data();summary=[]
    for task,path in paths.items():
        state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        import hashlib
        if hashlib.sha256((roots[0]/'metadata.json').read_bytes()).hexdigest()!=state['candle_v4_shadow_state']['spec']['input_sha256']:
            raise ValueError('Dataset metadata differs from training')
        result=evaluate_task(state,task,x,y,original,out)
        for k,m in result['metrics'].items():
            summary.append(dict(task=task,round=19,mode=k,rows=result['rows'],accuracy=m['accuracy'],
                macro_f1=m['macro_f1'],macro_f1_seen_classes=m['macro_f1_seen_classes'],
                top1_route_accuracy=result['top1_route_accuracy'],top2_task_recall=result['top2_task_recall'],
                rescue=result['rescue'] if k=='top2' else 0,break_count=result['break_count'] if k=='top2' else 0))
        with (out/'summary.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
        del state,result;gc.collect();torch.cuda.empty_cache()
    (out/'completion.json').write_text(json.dumps(dict(completed=True,protocol=PROTOCOL,source_training_commit=SOURCE,
        checkpoints={t:sha256(path) for t,path in paths.items()},summary=summary,full_TEST_development_readout=True,
        actual_top2_not_oracle=True,training=False),indent=2))
    print('Completed all six checkpoints. See summary.csv and per-task reports.',flush=True)


if __name__=='__main__':main()
