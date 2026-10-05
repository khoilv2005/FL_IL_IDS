"""One predeclared fresh remaining-test panel, using exclusively frozen artifacts."""
from contextlib import contextmanager
import gc
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from eval_checkpoint import _make_denice_client_model
from fed_learning.servers.nice_server import ContextDetector
from fed_learning.training.checkpoint_state import restore_context_detector
from fed_learning.data.incremental_loader import IncrementalDataLoader
from fed_learning.training.decentralized_denice_il import _partition_test_data_by_client
from fed_learning.strategies.incremental.denice_tip_router import encoder_fingerprint
from tools.denice_competence_gate import content_hash,feature_matrix,gate_decision
from tools.denice_class_meta import class_features,masked_class_decision
from tools.denice_peer_voting import vote
from tools.eval_denice_competence_gate import expert_features
from tools.eval_denice_class_meta import GateArchive,write_json


def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def verify_graph_weights(recorded_edges,source,out):
    """Verify saved floats separately from historical feature CSV parsing.

    Default pandas parsing loses ~1e-16 absolute precision on small weights.
    Round-trip parsing recovers the original checkpoint float exactly. Preserve
    source.alphas (historical default parser) for unchanged feature construction.
    """
    precise=pd.read_csv(source.zip.open('peer_edges.csv'),float_precision='round_trip')
    if precise.duplicated(['receiver','donor']).any():raise ValueError('Duplicate graph CSV edges')
    expected={(int(e['receiver']),int(e['donor'])):float(e['alpha']) for e in recorded_edges}
    observed={(int(row.receiver),int(row.donor)):float(row.alpha) for row in precise.itertuples()}
    if expected.keys()!=observed.keys():raise ValueError('Graph edge membership changed')
    different=[pair for pair in expected if expected[pair]!=observed[pair]]
    if different:
        pair=different[0]
        raise ValueError(f'Graph alpha changed at {pair}: checkpoint={expected[pair]!r}, CSV={observed[pair]!r}')
    deltas=[abs(weight-float(source.alphas[receiver].get(donor,0.)))
        for (receiver,donor),weight in expected.items()]
    write_json(Path(out)/'graph_alpha_validation.json',dict(positive_edges=len(expected),
        edge_membership_exact=True,round_trip_checkpoint_mismatches=0,
        default_parser_differing_edges=int(sum(delta>0 for delta in deltas)),
        max_default_parser_absolute_error=float(max(deltas,default=0.)),
        validation_parser='round_trip',feature_parser='historical default, unchanged'))


@contextmanager
def prohibit_fit():
    """Abort any accidental fitting, including hidden router reconstruction."""
    calls=[];original=[]
    def refuse(*args,**kwargs):
        calls.append('fit');raise RuntimeError('Frozen evaluation prohibits all fitting/refitting')
    for cls,name in [(LogisticRegression,'fit'),(MLPClassifier,'fit'),(StandardScaler,'fit'),
        (Pipeline,'fit'),(ContextDetector,'train_models'),(ContextDetector,'refresh_activation_memory'),
        (ContextDetector,'push_activations')]:
        original.append((cls,name,getattr(cls,name)));setattr(cls,name,refuse)
    try:yield calls
    finally:
        for cls,name,method in original:setattr(cls,name,method)


def draw_panel(x,y,labels,old_ids,forbidden,size=50000,seed=20261006):
    """Predeclared class quotas with hash rejection and round-robin redistribution."""
    values=y.numpy();rng=np.random.default_rng(seed)
    old_mask=np.zeros(len(y),dtype=bool);old_mask[old_ids]=True
    queues={};offset={};chosen={label:[] for label in labels};hashes={};seen_hashes=set();audit={}
    for label in labels:
        raw=np.flatnonzero(values==label);remaining=raw[~old_mask[raw]]
        queues[label]=rng.permutation(remaining);offset[label]=0
        audit[label]=dict(raw_rows=len(raw),old_panel_rows=int(old_mask[raw].sum()),
            remaining_row_ids=len(remaining),scanned=0,forbidden_hash_rejections=0,
            duplicate_panel_hash_rejections=0)
    def next_row(label):
        queue=queues[label]
        while offset[label]<len(queue):
            index=int(queue[offset[label]]);offset[label]+=1;audit[label]['scanned']+=1
            digest=content_hash(x[index].numpy())
            if digest in forbidden:audit[label]['forbidden_hash_rejections']+=1;continue
            if digest in seen_hashes:
                audit[label]['duplicate_panel_hash_rejections']+=1;continue
            hashes[index]=digest;seen_hashes.add(digest);chosen[label].append(index);return True
        return False
    quota=size//len(labels)
    for label in labels:
        for _ in range(quota):
            if not next_row(label):break
        print(f'Fresh panel class={label}, accepted={len(chosen[label])}, remaining IDs={audit[label]["remaining_row_ids"]}',flush=True)
    count=sum(map(len,chosen.values()))
    while count<size:
        progress=False
        for label in labels:
            if next_row(label):count+=1;progress=True
            if count==size:break
        if not progress:break
    if count!=size:raise ValueError(f'Only {count} unique eligible samples remain; requested fixed {size}')
    selected=np.asarray([index for label in labels for index in chosen[label]],dtype=np.int64)
    selected=selected[rng.permutation(len(selected))]
    for label in labels:
        audit[label]['selected']=len(chosen[label])
        audit[label]['missing_reason']=('old_panel_exhausted_row_ids' if not audit[label]['remaining_row_ids']
            else 'all_scanned_contents_excluded' if not chosen[label] else '')
    return selected,hashes,audit


def collect_frozen(ckpt,pools,source,routers,manifest,out,device,batch_size):
    ids=source.ids;needed={cid:[cid]+source.bundle['orders'][cid][:16] for cid in ids}
    cache={role:{cid:{} for cid in ids} for role in pools};runtime=[]
    folder=out/'fresh_expert_features';folder.mkdir()
    comparisons=0
    for position,donor in enumerate(ids,1):
        spec=manifest['routers'][str(donor)];payload=routers.read(spec['path'])
        if hashlib.sha256(payload).hexdigest()!=spec['sha256']:raise ValueError('Router checksum changed')
        model,detector=_make_denice_client_model(ckpt,donor,device)
        fingerprint=encoder_fingerprint(model)
        if fingerprint!=spec['encoder_hash']:raise ValueError('Frozen expert model signature mismatch')
        state=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
        restore_context_detector(detector,state)
        if detector.router_mode!='multiclass_balanced':raise ValueError('Wrong frozen donor router')
        # Preserve V2's complete old receiver stream and batch boundaries.
        # A truncated reference concatenated with fresh rows changes GPU kernels
        # and route-group shapes, producing avoidable float32 margin drift.
        for role in ('reference','fresh'):
            pieces=[(cid,pools[role][cid]) for cid in ids if donor in needed[cid]]
            inputs=torch.cat([data for _,data in pieces])
            if device.startswith('cuda'):torch.cuda.synchronize()
            start=time.perf_counter();features=expert_features(model,detector,inputs,source.labels,device,batch_size)
            if device.startswith('cuda'):torch.cuda.synchronize()
            runtime.append(dict(donor=donor,stage=role,seconds=time.perf_counter()-start,samples=len(inputs)))
            offset=0
            for cid,data in pieces:
                n=len(data);record={name:values[offset:offset+n] for name,values in features.items()};offset+=n
                if role=='reference':
                    with np.load(io.BytesIO(source.zip.read(f'test_expert_features/test_receiver_{cid}_donor_{donor}.npz'))) as previous:
                        for name,value in record.items():
                            target=previous[name]
                            if value.shape!=target.shape:raise ValueError('Original reference stream size changed')
                            exact=name in ('pred','task','mask_count','task_supported','class_supported')
                            matched=(np.array_equal(value,target) if exact
                                else np.allclose(value,target,rtol=1e-4,atol=1e-6))
                            if not matched:
                                delta=np.abs(value.astype(np.float64)-target.astype(np.float64))
                                mask=(value!=target if exact else ~np.isclose(value,target,rtol=1e-4,atol=1e-6))
                                bad=np.flatnonzero(mask);row=int(bad[0])
                                failure=dict(passed=False,client_id=cid,donor=donor,field=name,
                                    mismatched_rows=len(bad),first_row_in_receiver=row,
                                    actual=float(value[row]),expected=float(target[row]),
                                    max_absolute_difference=float(delta.max()),batch_size=batch_size,
                                    reference_stream='full original V2 test stream; separately batched',
                                    integer_fields_exact=True,continuous_rtol=1e-4,continuous_atol=1e-6)
                                write_json(out/'reference_reproduction.json',failure)
                                np.savez_compressed(out/f'reference_failure_receiver_{cid}_donor_{donor}.npz',
                                    **{f'actual_{key}':v for key,v in record.items()},
                                    **{f'expected_{key}':previous[key] for key in previous.files})
                                raise ValueError(f'Frozen router/feature reproduction failed: {cid}/{donor}/{name}; '
                                    f'{len(bad)} rows, max abs={delta.max():.9g}, '
                                    f'first actual={value[row]!r}, expected={target[row]!r}; see reference_reproduction.json')
                    comparisons+=n
                else:
                    cache[role][cid][donor]=record
                    np.savez_compressed(folder/f'test_receiver_{cid}_donor_{donor}.npz',**record)
            del inputs,features
        if encoder_fingerprint(model)!=fingerprint:raise RuntimeError('Expert weights/masks mutated')
        pd.DataFrame(runtime).to_csv(out/'expert_runtime.csv',index=False)
        print(f'Frozen experts {position}/{len(ids)}, donor={donor}, rows={len(inputs)}',flush=True)
        del model,detector
        if device.startswith('cuda'):torch.cuda.empty_cache()
    write_json(out/'reference_reproduction.json',dict(passed=True,expert_sample_comparisons=comparisons,
        reference_rows_per_receiver='all original receiver rows',reference_batch_size=batch_size,
        reference_stream='full original V2 test stream; separately batched',
        integer_fields_exact=True,continuous_rtol=1e-4,continuous_atol=1e-6,
        reference_used_for_verification_only=True))
    return cache['fresh']


def run_frozen_panel(ckpt,data_dir,gate_zip,meta_zip,router_zip,out,checkpoint_sha256,
                     evaluation_commit='',device='cuda',batch_size=512):
    out=Path(out)
    if batch_size!=512:raise ValueError('Keep batch_size=512 to reproduce the original V2 notebook batching')
    if out.exists() and any(out.iterdir()):raise ValueError('Use a fresh output directory/session')
    out.mkdir(parents=True,exist_ok=True);write_json(out/'frozen_panel_completion.json',dict(completed=False))
    with prohibit_fit() as fit_calls,zipfile.ZipFile(meta_zip) as mz,zipfile.ZipFile(router_zip) as rz,torch.no_grad():
        source=GateArchive(gate_zip)
        try:
            manifest=json.loads(rz.read('manifest.json'));lock=json.loads(mz.read('class_meta_lock.json'))
            completion=json.loads(mz.read('class_meta_completion.json'));payload=mz.read('frozen_class_meta.joblib')
            if not completion['completed'] or hashlib.sha256(payload).hexdigest()!=lock['sha256']:
                raise ValueError('Completed, checksum-matching meta artifact required')
            if lock['sha256']!='786477d5c66d46cf5cf7db263b567bcb0e406ad9e11a6f5ad8613b7414c7b771':
                raise ValueError('Preserve the exact audited meta model')
            if source.lock['gate_sha256']!='3aebd1d68c4342d1e9d3addb38c53ec71364a69df0abb1d1d6f49611e47f9633':
                raise ValueError('Preserve the exact audited V2 gate')
            meta=joblib.load(io.BytesIO(payload))
            if (meta['source_gate_sha256']!=source.lock['gate_sha256'] or meta['selected'][16]!='ClassLR_C0.1'
                or source.bundle['selected'][16]!='MLP_top1' or manifest['client_ids']!=source.ids
                or meta['labels']!=source.labels or manifest['task_classes']!=source.protocol['task_classes']):
                raise ValueError('Frozen artifact protocol mismatch')
            if checkpoint_sha256!=source.protocol['checkpoint_file_sha256'] or checkpoint_sha256!=manifest['checkpoint_file_sha256']:
                raise ValueError('Wrong checkpoint file')
            if ckpt['config']['git_commit']!=source.protocol['training_commit']:raise ValueError('Wrong training source')
            # Candidate IDs/weights must still be supported by the recorded checkpoint graph.
            from tools.eval_denice_peer_coverage import recorded_peers,lookup
            from tools.denice_peer_voting import peer_orders
            peers,recorded_edges=recorded_peers(ckpt,source.ids)
            verify_graph_weights(recorded_edges,source,out)
            for cid in source.ids:
                audit=lookup(ckpt['cluster']['alpha_debug'],cid)
                alpha={int(d):float(w) for d,w in zip(audit['group_ids'],audit['alphas'])}
                expected=peer_orders(cid,peers[cid],alpha,(42,))['random_42']
                if expected!=source.bundle['orders'][cid]:raise ValueError('Candidate ordering changed')
            declaration=dict(kind='frozen remaining-test confirmation panel',evaluation_commit=evaluation_commit,
                training_commit=source.protocol['training_commit'],checkpoint_file_sha256=checkpoint_sha256,
                meta_sha256=lock['sha256'],gate_sha256=source.lock['gate_sha256'],router_archive_sha256=file_hash(router_zip),
                gate_archive_sha256=file_hash(gate_zip),meta_archive_sha256=file_hash(meta_zip),
                k=16,candidate_seed=42,client_ids=source.ids,task_classes=source.classes,
                panel_size=50000,panel_seed=20261006,partition_seed=523687,batch_size=batch_size,
                scope='remaining original test rows; explicitly report missing classes',
                unique_content_required=True,exclusions=['all old panel row IDs','all old panel content hashes',
                    'all V2 calibration/fit/validation content hashes','within-panel duplicate content'],
                sampling='class-stratified fixed quotas then round-robin redistribution among eligible classes',
                primary='ClassLR_C0.1',baselines=['self','majority','GateV2'],selection_allowed=False,
                metrics=['pooled accuracy','client mean accuracy','macro-F1 over all 34 classes',
                    'macro-F1 over present classes','macro recall over present classes','per-task/per-class recall',
                    'paired receiver bootstrap CI vs GateV2 and majority','actual-routed oracle'],
                metrics_declared_before_dataset_labels=True,full_34_class_final_claim=False,
                reference_policy='full original V2 receiver streams, batch 512, separate from fresh inputs',
                additional_original_training_seeds=False,shared_retrospective_meta=True)
            write_json(out/'frozen_panel_protocol.json',declaration)
            protocol_digest=file_hash(out/'frozen_panel_protocol.json')
            write_json(out/'frozen_panel_lock.json',dict(protocol_sha256=protocol_digest,
                locked_before_dataset_read=True,fit_allowed=False,policy_selection_allowed=False))
            loader=IncrementalDataLoader(data_dir);x,y=loader.get_test_data(5,cumulative=True)
            if len(y)!=source.protocol['sample_info']['total']:raise ValueError('Global test row coordinate system changed')
            old={cid:pd.read_csv(source.zip.open(f'predictions/client_{cid}.csv')) for cid in source.ids}
            old_ids=np.concatenate([frame.global_test_row.to_numpy() for frame in old.values()])
            if len(old_ids)!=50000 or len(np.unique(old_ids))!=50000:raise ValueError('Expected original unique 50k panel')
            if (old_ids<0).any() or (old_ids>=len(y)).any():raise ValueError('Old panel ID outside dataset')
            if not np.array_equal(y[torch.from_numpy(old_ids)].numpy(),np.concatenate([f.y_true for f in old.values()])):
                raise ValueError('Old panel labels/row IDs no longer match dataset')
            provenance=pd.read_csv(source.zip.open('gate_data_provenance.csv'))
            if set(provenance.role)!=set(('fit','validation','calibration')):raise ValueError('Incomplete role provenance')
            split_audit=json.loads(source.zip.read('gate_split_audit.json'))
            counts=provenance.groupby('role').input_sha256.nunique().to_dict()
            if counts!=split_audit['unique_content_by_role']:raise ValueError('Role hash provenance is incomplete')
            role_hashes=set(provenance.input_sha256)
            old_hashes={content_hash(x[int(index)].numpy()) for index in old_ids}
            forbidden=role_hashes|old_hashes
            selected,hashes,audit=draw_panel(x,y,source.labels,old_ids,forbidden)
            if np.isin(selected,old_ids).any() or set(hashes.values())&forbidden or len(set(hashes.values()))!=len(selected):
                raise RuntimeError('Fresh panel exclusion invariant failed')
            index_shards,partition=_partition_test_data_by_client(torch.from_numpy(selected)[:,None],
                y[torch.from_numpy(selected)],source.ids,523687)
            shards={};pools={'reference':{},'fresh':{}};identities=[]
            for cid in source.ids:
                indices=index_shards[cid]['X_test'].reshape(-1).long()
                shards[cid]=dict(sample_ids=indices.numpy(),y=index_shards[cid]['y_test'].numpy())
                pools['fresh'][cid]=x[indices]
                pools['reference'][cid]=x[torch.from_numpy(old[cid].global_test_row.to_numpy())]
                identities.extend(dict(client_id=cid,global_test_row=int(index),input_sha256=hashes[int(index)],
                    y_true=int(label)) for index,label in zip(indices.tolist(),shards[cid]['y']))
            frame=pd.DataFrame(identities);frame.to_csv(out/'panel_manifest.csv',index=False)
            present=sorted(frame.y_true.unique().tolist());missing=sorted(set(source.labels)-set(present))
            pd.DataFrame.from_dict(audit,orient='index').rename_axis('label').to_csv(out/'panel_sampling_audit.csv')
            write_json(out/'panel_scope.json',dict(present_classes=present,missing_classes=missing,
                full_34_class_evaluation=not missing,total_rows=len(frame),unique_content=frame.input_sha256.nunique(),
                old_panel_row_overlap=int(np.isin(frame.global_test_row,old_ids).sum()),
                old_panel_content_overlap=len(set(frame.input_sha256)&old_hashes),
                gate_role_content_overlap=len(set(frame.input_sha256)&role_hashes),partition=partition,
                original_test_file_sha256=file_hash(Path(data_dir)/'global_test_data.npz')))
            print('Frozen panel scope: present classes',present,'missing classes',missing,flush=True)
            loader._test_data=None;del loader,x,y,index_shards;gc.collect()
            records=collect_frozen(ckpt,pools,source,rz,manifest,out,device,batch_size)
            del pools;gc.collect()
            old_meta={cid:pd.read_csv(mz.open(f'predictions/k16_client_{cid}.csv')) for cid in source.ids}
            for cid in source.ids:
                if (not np.array_equal(old_meta[cid].global_test_row,old[cid].global_test_row)
                    or not np.array_equal(old_meta[cid].y_true,old[cid].y_true)):
                    raise ValueError('Old meta panel coordinate mismatch')
            evaluate(records,shards,source,meta,present,out,old,old_meta)
            if file_hash(out/'frozen_panel_protocol.json')!=protocol_digest:raise RuntimeError('Protocol mutated')
            if file_hash(router_zip)!=declaration['router_archive_sha256']:raise RuntimeError('Router artifact mutated')
            if file_hash(gate_zip)!=declaration['gate_archive_sha256'] or file_hash(meta_zip)!=declaration['meta_archive_sha256']:
                raise RuntimeError('Gate/meta archives changed')
            write_json(out/'frozen_panel_completion.json',dict(completed=True,fit_calls=len(fit_calls),
                new_expert_predictions=True,primary='ClassLR_C0.1',k=16,logical_expert_queries_per_sample=17,
                prediction_cache_from_old_panel_used_for_new_rows=False,nonoverlapping_panel=True,
                missing_classes=missing,full_34_class_final_claim=False,independent_training_seeds=False,
                frozen_artifacts_unchanged=True,raw_sample_network_queries=False))
            return pd.read_csv(out/'summary.csv')
        finally:source.zip.close()


def evaluate(records,shards,source,meta,present,out,old,old_meta):
    directory=out/'predictions';directory.mkdir();frames=[];metrics=[]
    gate=source.bundle['gates'][16,'MLP'];model=meta['models'][16,'ClassLR_C0.1']
    for cid in source.ids:
        chosen=[cid]+source.bundle['orders'][cid][:16]
        expert,pred,names=feature_matrix(cid,chosen,records[cid],source.alphas[cid],source.bundle['priors'],6,34)
        if names!=source.bundle['feature_names']:raise ValueError('Gate feature schema changed')
        scores=gate.predict_proba(expert)[:,int(np.flatnonzero(gate.classes_==1)[0])].reshape(pred.shape)
        matrix,available,schema=class_features(expert,pred,scores,names,source.labels)
        if schema!=meta['feature_schema']:raise ValueError('Class feature schema changed')
        majority=vote(pred.T,np.ones(17),source.labels,pred[:,0])
        output=dict(self=pred[:,0],majority=majority,
            GateV2=gate_decision(scores,pred,chosen,source.labels,1)[0],
            FrozenClassMeta=masked_class_decision(model.predict_proba(matrix),model.classes_,source.labels,
                available,pred[:,0],majority))
        truth=shards[cid]['y'];oracle=(pred==truth[:,None]).any(1)
        for policy,result in output.items():
            if not (pred==result[:,None]).any(1).all():raise ValueError('Prediction outside frozen action set')
            metrics.append(dict(client_id=cid,policy=policy,samples=len(truth),correct=int((result==truth).sum()),
                accuracy=float((result==truth).mean())))
        frame=pd.DataFrame(dict(client_id=np.full(len(truth),cid),global_test_row=shards[cid]['sample_ids'],
            y_true=truth,OracleActualRouted=oracle,**output))
        frame.to_csv(directory/f'client_{cid}.csv',index=False);frames.append(frame)
    full=pd.concat(frames,ignore_index=True);metrics=pd.DataFrame(metrics)
    metrics.to_csv(out/'per_client_metrics.csv',index=False);rows=[];rng=np.random.default_rng(20261006)
    draw=rng.integers(0,len(source.ids),(10000,len(source.ids)))
    label_task={label:t for t,labels in source.classes.items() for label in labels}
    full['true_task']=full.y_true.map(label_task)
    for policy in ('self','majority','GateV2','FrozenClassMeta'):
        group=metrics[metrics.policy==policy].set_index('client_id').loc[source.ids]
        row=dict(policy=policy,pooled_accuracy=float((full[policy]==full.y_true).mean()),
            client_mean_accuracy=float(group.accuracy.mean()),
            macro_f1_all_34=float(f1_score(full.y_true,full[policy],labels=source.labels,average='macro',zero_division=0)),
            macro_f1_present=float(f1_score(full.y_true,full[policy],labels=present,average='macro',zero_division=0)),
            macro_recall_present=float(np.mean([(full.loc[full.y_true==c,policy]==c).mean() for c in present])),
            actual_routed_oracle=float(full.OracleActualRouted.mean()),queried_models=17)
        for baseline in ('majority','GateV2'):
            base=metrics[metrics.policy==baseline].set_index('client_id').loc[source.ids]
            delta=group.accuracy.to_numpy()-base.accuracy.to_numpy();ci=np.quantile(delta[draw].mean(1),[.025,.975])*100
            row.update({f'gain_vs_{baseline}_pp':float(delta.mean()*100),f'ci_vs_{baseline}_low_pp':float(ci[0]),
                f'ci_vs_{baseline}_high_pp':float(ci[1])})
        rows.append(row)
    pd.DataFrame(rows).to_csv(out/'summary.csv',index=False)
    for key,name in [('y_true','class'),('true_task','task')]:
        report=[]
        for value,part in full.groupby(key):
            row={key:int(value),'samples':len(part),'oracle':float(part.OracleActualRouted.mean())}
            row.update({p:float((part[p]==part.y_true).mean()) for p in ('self','majority','GateV2','FrozenClassMeta')})
            report.append(row)
        pd.DataFrame(report).to_csv(out/f'per_{name}_metrics.csv',index=False)
    # Describe scope differences without treating old/new samples as paired observations.
    previous=pd.concat(list(old.values()),ignore_index=True)
    previous=previous[previous.y_true.isin(present)]
    previous_meta=pd.concat(list(old_meta.values()),ignore_index=True)
    previous_meta=previous_meta[previous_meta.y_true.isin(present)]
    write_json(out/'old_panel_scope_comparison.json',dict(remaining_classes=present,
        old_subset_rows=len(previous),old_scope_self=float((previous.self==previous.y_true).mean()),
        old_scope_majority=float((previous.k16_majority==previous.y_true).mean()),
        old_scope_gate_v2=float((previous.k16_ValidationSelected==previous.y_true).mean()),
        old_scope_class_meta=float((previous_meta.ValidationSelected==previous_meta.y_true).mean()),
        comparison_is_unpaired=True,old_full_34_accuracy_is_not_directly_comparable=True))
