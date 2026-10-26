"""Frozen terminal full-TEST evaluation; never called during training rounds."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from fed_learning.data.incremental_loader import IncrementalDataLoader
from .candle_v4_bank import base_predictions
from .candle_v4_service import ARMS
from .candle_v4_shadow import digest,native_fingerprint


POLICY='full_34_disjoint_receivers_v1'


def statistics(cm):
    hits=np.diag(cm);positive=cm.sum(1);predicted=cm.sum(0);total=int(cm.sum())
    recall=np.divide(hits,positive,out=np.zeros(34),where=positive>0)
    f1=np.divide(2*hits,positive+predicted,out=np.zeros(34),where=positive+predicted>0)
    return dict(samples=total,accuracy=float(hits.sum()/total) if total else None,
                macro_f1=float(f1.mean()),per_class_recall=recall.tolist(),per_class_f1=f1.tolist(),
                per_class_counts=positive.tolist(),confusion_matrix=cm.tolist())


def evaluate_terminal(service,models,detectors,task,round_id):
    if service.spec.get('evaluation_mode')=='v4_only':
        from .candle_v4_only_eval import evaluate_v4_only
        return evaluate_v4_only(service,models,detectors,task,round_id)
    if (task,round_id)!=(5,19) or service.boundary!=[5,19] or 5 not in service.gated:
        raise ValueError('Full TEST requires completed Task5/Round19 and terminal gate')
    if [r['task'] for r in service.evaluations]!=list(range(6)):
        raise ValueError('Incomplete cumulative development evaluations')
    directory=Path(service.spec['output_dir']);target=directory/'candle_v4_final_test.json'
    if target.exists():raise ValueError('Refuse to overwrite terminal TEST results')
    before=native_fingerprint(models,detectors);expert_before=digest(service.state_dict())
    from .decentralized_denice_il import _partition_test_data_by_client
    loader=IncrementalDataLoader(service.spec['data_dir']);x,y=loader.get_full_test_data()
    if not np.array_equal(np.unique(y.numpy()),np.arange(34)):
        raise ValueError('Terminal TEST must contain all 34 classes')
    expected=len(y);ids=sorted(models)
    if not ids:raise ValueError('No terminal receiver predictors')
    shards,partition=_partition_test_data_by_client(x,y,ids,seed=42+104729*5,lazy_features=True)
    scoped=service.spec['version'] in ('candle_v4_current_graph_handoff_v6','candle_v4_continuous_class_router_v7','candle_v4_continuous_bootstrap_v8')
    continuous=service.spec['version'] in ('candle_v4_continuous_class_router_v7','candle_v4_continuous_bootstrap_v8')
    route_counts=np.zeros((6,7),dtype=np.int64)
    names=('V1',*(['continuous_native'] if continuous else []),*service.arms);cms={a:np.zeros((34,34),dtype=np.int64) for a in names}
    fusion_counts={a:dict(rows=0,triggered=0,multiple_triggers=0,overrides=0) for a in service.arms}
    per_client={};rescue={a:0 for a in service.arms};broken={a:0 for a in service.arms}
    timing={a:0. for a in names};evaluated=0;started=time.perf_counter()
    for position,receiver in enumerate(ids,1):
        local={a:np.zeros((34,34),dtype=np.int64) for a in names};shard=shards[receiver]
        for start in range(0,len(shard['y_test']),512):
            inputs=shard['X_test'][start:start+512].reshape(-1,39).cpu()
            truth=shard['y_test'][start:start+512].cpu().numpy()
            clock=time.perf_counter()
            if scoped:
                native,predicted_tasks=base_predictions(models[receiver],detectors[receiver],inputs,list(range(34)),return_episodes=True)
            else:native=base_predictions(models[receiver],detectors[receiver],inputs,list(range(34)))
            if torch.cuda.is_available():torch.cuda.synchronize()
            timing['V1']+=time.perf_counter()-clock;predictions={'V1':native}
            routed_native=native
            if continuous:
                from .candle_v4_continuous_runtime import predictions as continuous_predictions
                clock=time.perf_counter()
                routed_native,predicted_tasks=continuous_predictions(service,receiver,models[receiver],detectors[receiver],inputs,list(range(34)))
                if torch.cuda.is_available():torch.cuda.synchronize()
                timing['continuous_native']+=time.perf_counter()-clock
                predictions['continuous_native']=routed_native
            if scoped:np.add.at(route_counts,(truth//6,np.where(predicted_tasks<0,6,predicted_tasks)),1)
            for arm in service.arms:
                clock=time.perf_counter()
                extra=dict(predicted_tasks=predicted_tasks) if scoped else {}
                predictions[arm],nfire=service.infer(receiver,arm,routed_native,inputs,**extra)
                counts=fusion_counts[arm];counts['rows']+=len(truth)
                counts['triggered']+=int((nfire>0).sum());counts['multiple_triggers']+=int((nfire>1).sum())
                counts['overrides']+=int((predictions[arm]!=routed_native).sum())
                timing[arm]+=time.perf_counter()-clock
                rescue[arm]+=int(((native!=truth)&(predictions[arm]==truth)).sum())
                broken[arm]+=int(((native==truth)&(predictions[arm]!=truth)).sum())
            for name in names:
                np.add.at(local[name],(truth,predictions[name]),1)
            evaluated+=len(truth)
        for name in names:cms[name]+=local[name]
        per_client[receiver]={name:statistics(local[name]) for name in names}
        if position%10==0 or position==len(ids):
            print(f'CANDLE V4 final FULL TEST: receiver {position}/{len(ids)}, rows={evaluated}/{expected}',flush=True)
        del shards[receiver]
    if evaluated!=expected or any(int(cm.sum())!=expected for cm in cms.values()):
        raise RuntimeError('Incomplete full TEST coverage')
    if before!=native_fingerprint(models,detectors) or expert_before!=digest(service.state_dict()):
        raise RuntimeError('TEST evaluation mutated frozen native/expert state')
    result=dict(completed=True,task=5,round=19,after_consolidation=True,policy=POLICY,TEST_opened=True,
        activation_policy=service.spec.get('activation_policy'),prediction_fusion=service.spec.get('prediction_fusion'),fusion_counts=fusion_counts,
        method_router_mode=service.spec.get('router_mode'),
        baseline_router_mode='multiclass_balanced' if continuous else service.spec.get('router_mode'),
        router_metrics=dict(samples=int(route_counts.sum()),accuracy=float(np.trace(route_counts[:,:6])/expected) if scoped else None,
                            missing_route_rows=int(route_counts[:,6].sum()),confusion_matrix=route_counts.tolist()),
        test_rows=expected,evaluated_rows=evaluated,classes=list(range(34)),receivers=ids,
        partition=partition,metrics={a:statistics(cm) for a,cm in cms.items()},per_client=per_client,
        rescue=rescue,break_count=broken,native_sha256=before,expert_state_sha256=expert_before,
        native_inference_seconds=timing['V1'],extra_inference_seconds={a:timing[a] for a in service.arms},
        continuous_native_inference_seconds=timing.get('continuous_native'),
        rescue_reference='unchanged native multiclass' if continuous else 'native',
        wall_seconds=time.perf_counter()-started,
        evaluation_protocol='Entire original TEST partitioned once across receivers, V1-identical seed; pooled metrics; identical rows for native counterfactual and enabled expert arms',
        scientific_status='Frozen terminal readout of a development run; no TEST-based selection or acceptance')
    temporary=target.with_suffix('.tmp');temporary.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');temporary.replace(target)
    return result
