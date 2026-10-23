"""Frozen legacy self-only inference: original and refitted multiclass routers."""
import gc
import copy
import gzip
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from eval_checkpoint import _make_denice_client_model
from fed_learning.data.denice_clean_roles import CleanRoleData,file_sha256
from fed_learning.data.incremental_loader import IncrementalDataLoader
from fed_learning.training.checkpoint_state import snapshot_context_detector,restore_context_detector
from fed_learning.training.denice_delta_checkpoint import load_denice_checkpoint
from fed_learning.training.decentralized_denice_il import _partition_test_data_by_client
from fed_learning.training.denice_eval import _denice_routed_logits_with_episodes
from fed_learning.strategies.incremental.denice_tip_router import encoder_fingerprint


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def statistics(cm):
    correct=np.diag(cm);counts=cm.sum(1);denom=counts+cm.sum(0)
    f1=np.divide(2*correct,denom,out=np.zeros(34,dtype=float),where=denom>0)
    return dict(samples=int(cm.sum()),accuracy=float(correct.sum()/cm.sum()) if cm.sum() else None,
        macro_f1_34=float(f1.mean()),class_counts=counts.tolist(),per_class_f1=f1.tolist())


def resolve_evaluation_device(ckpt, requested='auto', include_appliance=False):
    """Honor sealed runtime bindings; auto selects a backend, never migrates it."""
    requirements = set()
    if include_appliance:
        from appliance.empirical_deployment import deployment_scope
        for algorithm in ckpt.get('client_algorithm_states', {}).values():
            algorithm = algorithm.get('denice', algorithm)
            for entry in algorithm.get('appliance_guarded_head_entries', {}).values():
                if deployment_scope(entry) is not None:
                    declaration = entry['empirical_deployment']
                    requirements.add((declaration['guard_backend'], declaration['torch_version']))
    if len(requirements) > 1:
        raise ValueError(f'Conflicting APPLIANCE runtime bindings: {sorted(requirements)}')
    required = next(iter(requirements), None)
    if requested == 'auto':
        requested = required[0] if required else ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(requested)
    if required and (device.type, str(torch.__version__)) != required:
        raise ValueError(f'APPLIANCE guard requires {required[0]} / Torch {required[1]}; '
                         f'requested {device} / {torch.__version__}. '
                         'Use device=auto and the matching Torch environment; do not rebind certificates.')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('This evaluation requires CUDA. Enable the Kaggle GPU accelerator; '
                           'a CPU fallback would change the certified guard function.')
    if device.type not in ('cpu', 'cuda'):
        raise ValueError(f'Unsupported evaluation backend: {device}')
    return str(device)


@torch.no_grad()
def run_legacy_self(checkpoint_archive,role_dir,output_dir,source_data_dir,device='auto',batch_size=512,expected_xi=.8,
                    include_appliance=False,include_global_task_mask=False):
    if include_appliance and include_global_task_mask:
        raise ValueError('Evaluate the class-mask variant separately from APPLIANCE certificates')
    ckpt=load_denice_checkpoint(str(checkpoint_archive));config=ckpt['config']
    device=resolve_evaluation_device(ckpt,device,include_appliance)
    print(f'Frozen evaluation backend: {device}; Torch {torch.__version__}',flush=True)
    roles=CleanRoleData(role_dir,source_data_dir=source_data_dir)
    if (config.get('denice_cl_method','legacy')!='legacy'
            or not np.isclose(config['denice_similarity_threshold'],expected_xi,rtol=0,atol=1e-12)
            or config['denice_data_roles_sha256']!=file_sha256(roles.root/'role_manifest.json')):
        raise ValueError('Expected matching clean legacy checkpoint, xi and data-role lock')
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'completion.json',dict(completed=False,stage='router_fit',device=device))
    ids=sorted(map(int,ckpt['client_model_states']));snapshots={};fingerprints={}
    with threadpool_limits(limits=1):
        for cid in ids:
            model,detector=_make_denice_client_model(ckpt,cid,device)
            if not detector.activation_memory:raise ValueError(f'{cid}: empty context memory')
            if detector.router_mode!='binary_cosine':raise ValueError('Expected original binary_cosine router')
            fingerprint=encoder_fingerprint(model)
            original=snapshot_context_detector(detector)
            detector.router_mode='multiclass_balanced'
            detector.train_models(max(detector.activation_memory))
            snapshots[cid]=dict(BinarySelf=original,MulticlassSelf=snapshot_context_detector(detector))
            if encoder_fingerprint(model)!=fingerprint:raise RuntimeError('Router fit mutated backbone')
            fingerprints[cid]=fingerprint
            del model,detector
    joblib.dump(snapshots,out/'frozen_routers.joblib',compress=3)
    del snapshots;gc.collect()
    checks={Path(checkpoint_archive):file_sha256(checkpoint_archive),
        roles.root/'role_manifest.json':file_sha256(roles.root/'role_manifest.json'),
        out/'frozen_routers.joblib':file_sha256(out/'frozen_routers.joblib')}
    lock=dict(method='DeNICE legacy + balanced multiclass router; self-only',task=5,round=19,
        training_method='legacy',training_router='binary_cosine',evaluation_router='multiclass_balanced',
        xi=expected_xi,experts_per_sample=1,cgofed=False,cme=False,peer_inference=False,
        router_fit_source='checkpoint activation memory only; no historical raw data refit',
        fit_before_test=True,checkpoint_sha256=checks[Path(checkpoint_archive)],
        role_manifest_sha256=checks[roles.root/'role_manifest.json'],
        router_sha256=checks[out/'frozen_routers.joblib'],receivers=ids)
    if include_appliance:
        lock.update(method='DeNICE legacy + APPLIANCE; one local backbone',
            appliance=True, application_scope=dict(domain_id='cumulative_dataset', classes=list(range(34))),
            scoped_certificate_expansion=False, suspended_route_fallback='original local prediction',
            scope_mode=config.get('appliance_scope_mode','initial_scope_v1'),
            population_FAR_claim=False, guard_backend=device,
            empirical_risk='Deployment outside finite CAL evidence is allowed only by an explicit empirical declaration')
    write_json(out/'pipeline_lock.json',lock)
    loader=IncrementalDataLoader(str(roles.source));x,y=loader.get_full_test_data()
    global_task_classes=None
    if include_global_task_mask:
        metadata=json.loads((roles.source/'metadata.json').read_text(encoding='utf-8'))
        global_task_classes={int(t):list(map(int,classes))
            for t,classes in metadata['task_structure']['task_classes'].items()}
        lock.update(experimental_class_mask_policy='global_task',
            experimental_policy='MulticlassGlobalTaskMask',
            experimental_policy_status='development coverage variant; no ownership or safety certification',
            unchanged_components=['router','weights','ranks','graph','optimizer','legal task profiles'])
        write_json(out/'pipeline_lock.json',lock)
    if not np.array_equal(np.unique(y.numpy()),np.arange(34)):raise ValueError('Source test must contain all 34 classes')
    expected=len(y)
    shards,partition=_partition_test_data_by_client(x,y,ids,seed=int(config.get('seed',42))+104729*5,lazy_features=True)
    del loader,x,y;gc.collect()
    frozen=joblib.load(out/'frozen_routers.joblib')
    cms={p:np.zeros((34,34),dtype=np.int64) for p in ('BinarySelf','MulticlassSelf')}
    if include_global_task_mask:
        cms['MulticlassGlobalTaskMask']=np.zeros((34,34),dtype=np.int64)
    if include_appliance:
        cms['APPLIANCE'] = np.zeros((34,34),dtype=np.int64)
    seen_rows=np.zeros(expected,dtype=bool);evaluated=0;clients=[];started=time.monotonic();header=True
    with gzip.open(out/'test_predictions.csv.gz','wt',encoding='utf-8',newline='') as destination:
        for position,cid in enumerate(ids,1):
            model,detector=_make_denice_client_model(ckpt,cid,device)
            model.eval()
            detectors={policy:copy.deepcopy(detector) for policy in ('BinarySelf','MulticlassSelf')}
            for policy in detectors:restore_context_detector(detectors[policy],frozen[cid][policy])
            global_mask_detector=None
            if include_global_task_mask:
                from fed_learning.strategies.incremental.denice_class_availability import detector_with_class_mask_policy
                global_mask_detector=detector_with_class_mask_policy(detectors['MulticlassSelf'],
                    'global_task',global_task_classes,list(range(34)),int(model.num_classes))
            from appliance.stable_head import StableHeadRegistry
            registry = StableHeadRegistry()
            registry.entries = getattr(model, 'appliance_guarded_head_entries', {})
            from appliance.patch_lifecycle import route_authorized
            route_status={c:dict(state=e.get('lifecycle_state'),reason=e.get('lifecycle_reason'),
                authorized=route_authorized(e,lock.get('application_scope')) if include_appliance else False,
                certificate_current=registry.certificate_current(model,detectors['MulticlassSelf'],c))
                for c,e in registry.entries.items()}
            if include_appliance and any(s['authorized'] and not s['certificate_current'] for s in route_status.values()):
                raise RuntimeError(f'{cid}: authorized imported route has a changed function; evaluation blocked')
            activation_count=0
            local={p:np.zeros((34,34),dtype=np.int64) for p in cms};shard=shards[cid]
            for start in range(0,len(shard['y_test']),batch_size):
                inputs=shard['X_test'][start:start+batch_size].to(device)
                predictions={}
                from contextlib import ExitStack
                from appliance.guarded_head import original_local_head
                with ExitStack() as stack:
                    for c, entry in registry.entries.items():
                        stack.enter_context(original_local_head(model, c, entry['backup']))
                    for policy in detectors:
                        logits,_=_denice_routed_logits_with_episodes(model,inputs,detectors[policy],list(range(34)),device,inference_policy='pred_hard')
                        predictions[policy]=logits.argmax(1).cpu().numpy()
                    if include_global_task_mask:
                        logits,_=_denice_routed_logits_with_episodes(model,inputs,global_mask_detector,
                            list(range(34)),device,inference_policy='pred_hard')
                        predictions['MulticlassGlobalTaskMask']=logits.argmax(1).cpu().numpy()
                if include_appliance:
                    appliance_record = (registry.records(model, detectors['MulticlassSelf'],
                        inputs.cpu().numpy(), list(range(34)), device, batch_size,
                        runtime_scope=lock['application_scope']) if registry.entries else None)
                    predictions['APPLIANCE'] = (appliance_record['pred'] if appliance_record else
                        predictions['MulticlassSelf'].copy())
                    if appliance_record:activation_count+=int(appliance_record['activated'].sum())
                truth=shard['y_test'][start:start+batch_size].numpy()
                rows=shard['X_test'].indices[start:start+batch_size].numpy()
                if len(np.unique(rows))!=len(rows) or seen_rows[rows].any():raise RuntimeError('Repeated test row')
                seen_rows[rows]=True
                for policy,pred in predictions.items():
                    cm=np.bincount(truth*34+pred,minlength=34*34).reshape(34,34)
                    cms[policy]+=cm;local[policy]+=cm
                pd.DataFrame(dict(client_id=cid,global_test_row=rows,y_true=truth,**predictions)).to_csv(destination,index=False,header=header)
                header=False;evaluated+=len(rows)
                if evaluated%(batch_size*100)<batch_size:print(f'Single-model full test: {evaluated}/{expected}',flush=True)
            if encoder_fingerprint(model)!=fingerprints[cid]:raise RuntimeError('Inference mutated expert weights/masks')
            clients.append(dict(receiver=cid,metrics={p:statistics(cm) for p,cm in local.items()},
                appliance_routes=route_status,appliance_activated_rows=activation_count))
            write_json(out/'client_metrics.json',clients)
            print(f'Single-model test receiver {position}/{len(ids)}: {evaluated}/{expected}; imported activation={activation_count}',flush=True)
            del shards[cid],model,detector,detectors;gc.collect()
            if device.startswith('cuda'):torch.cuda.empty_cache()
    if evaluated!=expected or not seen_rows.all():raise RuntimeError('Full-test coverage incomplete')
    for path,digest in checks.items():
        if file_sha256(path)!=digest:raise RuntimeError('Frozen inputs changed during evaluation')
    result=dict(completed=True,task=5,test_rows=evaluated,main_policy='APPLIANCE' if include_appliance else 'MulticlassSelf',
        metrics={p:statistics(cm) for p,cm in cms.items()},seconds=time.monotonic()-started,partition=partition,lock=lock,
        class_coverage_policy='report_only',limitation='Existing test source has previously been inspected; not untouched confirmation')
    if include_appliance:
        result['appliance_activity']=dict(activated_rows=sum(c['appliance_activated_rows'] for c in clients),
            authorized_routes=sum(s['authorized'] for c in clients for s in c['appliance_routes'].values()),
            suspended_routes=sum(not s['authorized'] for c in clients for s in c['appliance_routes'].values()))
    np.savez_compressed(out/'confusion_matrices.npz',**cms)
    write_json(out/'completion.json',result)
    return result
