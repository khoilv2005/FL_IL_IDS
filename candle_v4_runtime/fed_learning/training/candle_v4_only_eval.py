"""Terminal V4 predictor only; no baseline or ablation inference passes."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from fed_learning.data.incremental_loader import IncrementalDataLoader
from .candle_v4_continuous_runtime import predictions
from .candle_v4_shadow import digest,native_fingerprint
from .candle_v4_terminal_test import statistics


def evaluate_v4_only(service,models,detectors,task,round_id):
    if ((task,round_id)!=(5,19) or service.boundary!=[5,19] or 5 not in service.gated or
            [r['task'] for r in service.evaluations]!=list(range(6))):
        raise ValueError('V4 TEST requires completed T5/R19 after task consolidation')
    if (service.spec.get('evaluation_mode')!='v4_only' or service.arms!=('coverage',) or
            service.spec['version']!='candle_v4_continuous_bootstrap_v8'):
        raise ValueError('V4-only readout requires the locked bootstrap-fixed V4 coverage protocol')
    target=Path(service.spec['output_dir'])/'candle_v4_final_test.json'
    if target.exists():raise ValueError('Refuse to overwrite V4 terminal TEST')
    model_before=native_fingerprint(models,detectors);bank_before=digest(service.state_dict())
    from .decentralized_denice_il import _partition_test_data_by_client
    x,y=IncrementalDataLoader(service.spec['data_dir']).get_full_test_data()
    if not np.array_equal(np.unique(y.numpy()),np.arange(34)):raise ValueError('Full34-class TEST required')
    ids=sorted(models)
    if not ids:raise ValueError('No V4 predictors')
    shards,partition=_partition_test_data_by_client(x,y,ids,seed=523687,lazy_features=True)
    cm=np.zeros((34,34),dtype=np.int64);routes=np.zeros((6,7),dtype=np.int64)
    counts=dict(rows=0,triggered=0,multiple_triggers=0,overrides=0)
    clients={};elapsed=0.;started=time.perf_counter()
    for i,cid in enumerate(ids,1):
        local=np.zeros_like(cm);local_routes=np.zeros_like(routes);shard=shards[cid]
        for start in range(0,len(shard['y_test']),512):
            inputs=shard['X_test'][start:start+512].reshape(-1,39).cpu()
            truth=shard['y_test'][start:start+512].numpy();clock=time.perf_counter()
            fallback,task_ids=predictions(service,cid,models[cid],detectors[cid],inputs,list(range(34)))
            pred,nfire=service.infer(cid,'coverage',fallback,inputs,predicted_tasks=task_ids)
            if torch.cuda.is_available():torch.cuda.synchronize()
            elapsed+=time.perf_counter()-clock
            np.add.at(local,(truth,pred),1)
            np.add.at(local_routes,(truth//6,np.where(task_ids<0,6,task_ids)),1)
            counts['rows']+=len(truth);counts['triggered']+=int((nfire>0).sum())
            counts['multiple_triggers']+=int((nfire>1).sum());counts['overrides']+=int((pred!=fallback).sum())
        cm+=local;routes+=local_routes
        clients[cid]=dict(V4=statistics(local),router_metrics=dict(samples=int(local_routes.sum()),
            accuracy=float(np.trace(local_routes[:,:6])/local_routes.sum()) if local_routes.sum() else None,
            missing_route_rows=int(local_routes[:,6].sum()),confusion_matrix=local_routes.tolist()))
        del shards[cid]
        if i%10==0 or i==len(ids):print(f'CANDLE V4 TEST: {i}/{len(ids)} clients; rows={int(cm.sum())}/{len(y)}',flush=True)
    if int(cm.sum())!=len(y) or int(routes.sum())!=len(y):raise RuntimeError('Incomplete V4 TEST rows')
    if model_before!=native_fingerprint(models,detectors) or bank_before!=digest(service.state_dict()):
        raise RuntimeError('V4 evaluation changed frozen predictor state')
    result=dict(completed=True,task=5,round=19,after_consolidation=True,TEST_opened=True,
        evaluation_mode='v4_only',test_rows=len(y),evaluated_rows=int(cm.sum()),classes=list(range(34)),receivers=ids,
        metrics=dict(V4=statistics(cm)),per_client=clients,partition=partition,
        method_router_mode=service.spec['router_mode'],activation_policy=service.spec['activation_policy'],
        prediction_fusion=service.spec['prediction_fusion'],fusion_counts=counts,
        router_metrics=dict(samples=int(routes.sum()),accuracy=float(np.trace(routes[:,:6])/len(y)),
                            missing_route_rows=int(routes[:,6].sum()),confusion_matrix=routes.tolist()),
        inference_seconds=elapsed,wall_seconds=time.perf_counter()-started,
        predictor_sha256=model_before,expert_state_sha256=bank_before,
        evaluation_protocol='Full original TEST assigned once across clients; disjoint shards seed523687; pooled metrics',
        comparisons_computed=False,scientific_status='Development seed42; frozen terminal V4 readout')
    tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2),encoding='utf-8');tmp.replace(target)
    return result
