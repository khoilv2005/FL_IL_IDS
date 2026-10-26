"""Historical-feature counterfactual. No raw replay or backbone/expert updates.

Rebuild missing inherited task memory using exact recorded clone lineage, fit
only the existing class-LR recipe on saved TRAIN features, then read TEST once
for fixed clients35/43. Writes audit artifacts only, never official checkpoints.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
from copy import deepcopy
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import torch
import sklearn
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'candle_v4_runtime'))
from eval_checkpoint import _make_denice_client_model
from fed_learning.servers.continuous_class_router import ContinuousClassMemory
from fed_learning.training.candle_v4_original_train import OriginalTrainService
from fed_learning.training.candle_v4_continuous_runtime import predictions
from fed_learning.training.candle_v4_bank import base_predictions
from fed_learning.training.candle_v4_shadow import digest
from fed_learning.training.candle_v4_terminal_test import statistics
from fed_learning.training.decentralized_denice_il import _partition_test_data_by_client


def main():
    out=ROOT/'audit_denice/candle_v4_bootstrap_reconstruction_20261026'
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    warnings.filterwarnings('ignore',message='Trying to unpickle estimator')
    final=torch.load(ROOT/'audit_denice/candle_v4_results17_20261026/continuation_state_latest.pt',map_location='cpu',weights_only=False,mmap=True)
    official=json.loads((ROOT/'audit_denice/candle_v4_results17_20261026/candle_v4_final_test.json').read_text())
    lineage=json.loads((ROOT/'artifacts/candle_v4_results17_bootstrap_coverage_20261026.json').read_text())['clients_detail']
    snapshots={5:final}
    for t in range(5):
        path=(out/f'continuation_state_task_{t}.pt') if t<4 else ROOT/'audit_denice/candle_v4_resume_task5_20261025/continuation_state_task_4.pt'
        snapshots[t]=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        assert snapshots[t]['meta']['completed_task']==t
    def context(ck,cid):
        a=ck['client_algorithm_states'][cid];return a.get('denice',a)['context_detector']
    def origin(cid,task,visited=()):
        if cid in visited:raise ValueError('Cycle in bootstrap history')
        row=lineage[str(cid)]
        if task in row['local_training_tasks']:return cid,task,list(visited)+( [cid] )
        event=next((e for e in row['bootstrap_events'] if e['bootstrap_policy']=='representative_clone'),None)
        if event is None or task>=event['task'] or event['param_distance_to_source']!=0:
            raise ValueError('No exact historical feature provenance')
        return origin(event['bootstrap_source'],task,(*visited,cid))
    result=dict(completed=False,clients_selected_before_readout=[35,43],protocol='historical-feature counterfactual, not official V8 run',
        local_sklearn=sklearn.__version__,original_sklearn='1.6.1',
        estimator_caveat='Local LR refit uses existing local sklearn; not exact environment reproduction',
        backbone_training=False,expert_training=False,raw_TRAIN_opened=False,TEST_used_for_fitting=False,
        threshold=.5,selection_or_threshold_tuning=False,clients={})
    service=OriginalTrainService(final['candle_v4_shadow_state']['spec'],final['candle_v4_shadow_state'])
    repairs={}
    for cid in [35,43]:
        original=service.continuous_routers[cid];native=context(final,cid)
        memory=deepcopy(original.state_dict());provenance=[]
        missing=sorted(set(native['activation_memory'])-set(original.class_to_task.values()))
        for task in missing:
            source,t,chain=origin(cid,task)
            ck=snapshots[t];donor=ck['candle_v4_shadow_state']['continuous_routers'][source];nd=context(ck,source)
            assert np.array_equal(original.indices,donor['indices'])
            assert digest(native['stable_feature_mask'])==digest(nd['stable_feature_mask'])
            assert digest(native.get('routing_feature_mask'))==digest(nd.get('routing_feature_mask'))
            assert digest(native['binarize_thresholds'])==digest(nd['binarize_thresholds'])
            assert digest(native['activation_memory'][task])==digest(nd['activation_memory'][task])
            assert native['episode_classes'][task]==nd['episode_classes'][task]
            added=[]
            for c,values in donor['memory'].items():
                if c//6==task:
                    assert c not in memory['memory'];memory['memory'][c]=values.copy();memory['class_to_task'][c]=task;added.append(int(c))
            assert added
            provenance.append(dict(task=task,source_client=source,snapshot_task=t,clone_chain=chain,classes=added,
                native_sketches_thresholds_coordinates_equal=True))
        repaired=ContinuousClassMemory(original.per_class)
        for key in ('indices','full_width','latest_task','memory','class_to_task'):setattr(repaired,key,deepcopy(memory[key]))
        with threadpool_limits(limits=2):repaired.fit(repaired.latest_task)
        assert set(repaired.class_to_task.values())==set(native['activation_memory'])
        repairs[cid]=repaired
        result['clients'][str(cid)]=dict(provenance=provenance,original_tasks=sorted(set(original.class_to_task.values())),
            repaired_tasks=sorted(set(repaired.class_to_task.values())),original_feature_bytes=original.stats()['feature_bytes'],
            repaired_feature_bytes=repaired.stats()['feature_bytes'])
        print('Historical memory verified',cid,'tasks',result['clients'][str(cid)]['repaired_tasks'],flush=True)
    # Same original disjoint TEST rows as the final evaluator; no model selection.
    path=Path(r'C:\Users\khoak\Downloads\candle_v1_multiclass_audit\global_test_data.npz')
    with np.load(path) as z:x=torch.from_numpy(z['X_test']);y=torch.from_numpy(z['y_test']).long()
    shards,_=_partition_test_data_by_client(x,y,sorted(final['client_ids']),seed=523687,lazy_features=True)
    device='cuda' if torch.cuda.is_available() else 'cpu';result['device']=device
    for cid in [35,43]:
        model,det=_make_denice_client_model(final,cid,device);before=digest(model.state_dict())
        inputs=shards[cid]['X_test'];truth=shards[cid]['y_test'].numpy();assert len(truth)==official['per_client'][str(cid)]['V1']['samples']
        # Fixed64/class geometry probe. It is an empirical compatibility check,
        # never a claim of formal function identity outside these inputs.
        ix=np.concatenate([np.flatnonzero(truth==c)[:64] for c in range(34)])
        sample=inputs[ix].to(device)
        with torch.no_grad():target_acts=model.get_context_activations_per_sample(sample)
        target=np.concatenate([target_acts[k].cpu().numpy() for k in ('conv1','conv2','conv3','gru')],axis=1)[:,repairs[cid].indices]
        checks=[]
        for p in result['clients'][str(cid)]['provenance']:
            historical,_=_make_denice_client_model(snapshots[p['snapshot_task']],p['source_client'],device)
            with torch.no_grad():acts=historical.get_context_activations_per_sample(sample)
            a=np.concatenate([acts[k].cpu().numpy() for k in ('conv1','conv2','conv3','gru')],axis=1)[:,repairs[cid].indices]
            checks.append(dict(task=p['task'],rows=len(ix),max_abs_drift=float(np.abs(a-target).max()),
                relative_l1_drift=float(np.abs(a-target).sum()/max(float(np.abs(a).sum()),1e-12))))
            del historical
        result['clients'][str(cid)]['geometry_probe']=checks
        matrices={k:np.zeros((34,34),dtype=np.int64) for k in ('native_local','old_v4_local','repaired_native','repaired_v4')}
        route_correct=dict(old=0,repaired=0);old_mem=service.continuous_routers[cid];rescue=0;broken=0
        for start in range(0,len(truth),1024):
            part=inputs[start:start+1024].reshape(-1,39).cpu();labels=truth[start:start+1024]
            service.continuous_routers[cid]=old_mem
            native=base_predictions(model,det,part,list(range(34)))
            old,ep=predictions(service,cid,model,det,part,list(range(34)))
            old_v4,_=service.infer(cid,'coverage',old,part,predicted_tasks=ep)
            service.continuous_routers[cid]=repairs[cid]
            new,ep2=predictions(service,cid,model,det,part,list(range(34)))
            new_v4,_=service.infer(cid,'coverage',new,part,predicted_tasks=ep2)
            for name,pred in [('native_local',native),('old_v4_local',old_v4),('repaired_native',new),('repaired_v4',new_v4)]:np.add.at(matrices[name],(labels,pred),1)
            route_correct['old']+=int((ep==labels//6).sum());route_correct['repaired']+=int((ep2==labels//6).sum())
            rescue+=int(((old_v4!=labels)&(new_v4==labels)).sum());broken+=int(((old_v4==labels)&(new_v4!=labels)).sum())
        service.continuous_routers[cid]=old_mem
        assert before==digest(model.state_dict())
        for local,old_name in [('native_local','V1'),('old_v4_local','coverage')]:
            assert matrices[local].tolist()==official['per_client'][str(cid)][old_name]['confusion_matrix']
        row=result['clients'][str(cid)];row.update(rows=len(truth),metrics={k:statistics(v) for k,v in matrices.items()},
            routing_accuracy={k:v/len(truth) for k,v in route_correct.items()},rescue_vs_old_v4=rescue,break_vs_old_v4=broken,
            native_weights_unchanged=True,official_metrics= {k:{n:m[n] for n in ('accuracy','macro_f1')} for k,m in official['per_client'][str(cid)].items()})
        (out/'readout.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print('Local readout',cid,{k:round(v['accuracy']*100,4) for k,v in row['metrics'].items()},'route',row['routing_accuracy'],flush=True)
        del model,det
    result['completed']=True
    (out/'readout.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()
