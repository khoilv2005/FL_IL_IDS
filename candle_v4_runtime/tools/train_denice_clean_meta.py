"""Clean-role version of the frozen self + 16 peer Class Meta recipe.

Fit never reads test data. Prediction APIs accept expert evidence, not truth.
The historical 58.358% was a 28-class confirmation panel, not a guarantee here.
"""
import gc
import gzip
import json
import time
import zipfile
from collections import OrderedDict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from eval_checkpoint import _make_denice_client_model
from fed_learning.data.denice_clean_roles import CleanRoleData, file_sha256
from fed_learning.data.incremental_loader import IncrementalDataLoader
from fed_learning.training.checkpoint_state import snapshot_context_detector, restore_context_detector
from fed_learning.training.denice_delta_checkpoint import load_denice_checkpoint
from fed_learning.training.decentralized_denice_il import _partition_test_data_by_client
from fed_learning.strategies.incremental.denice_tip_router import encoder_fingerprint
from tools.denice_class_meta import class_features, masked_class_decision
from tools.denice_competence_gate import content_hash, competence_priors, feature_matrix, gate_decision
from tools.denice_peer_supported_gate_data import local_training_support
from tools.denice_peer_voting import peer_orders, vote
from tools.eval_denice_competence_gate import expert_features
from tools.eval_denice_peer_coverage import recorded_peers


LABELS = np.arange(34, dtype=np.int64)
LIMITS = dict(calibration=128, fit=512, validation=256)
RECIPE = dict(peer_budget=16, candidate_seed=42, gate='StandardScaler + MLP(32,16)_top1',
              gate_alpha=.001, gate_max_iter=50, meta='StandardScaler + ClassLR_C0.1',
              meta_C=.1, selector_seed=20261005, selection='fixed historical recipe; no test selection')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic)
                                    else x.tolist() if isinstance(x, np.ndarray) else str(x))+'\n', encoding='utf-8')


def metrics(confusion):
    matrix=np.asarray(confusion,dtype=np.float64); hits=np.diag(matrix)
    precision=np.divide(hits,matrix.sum(0),out=np.zeros(34),where=matrix.sum(0)>0)
    recall=np.divide(hits,matrix.sum(1),out=np.zeros(34),where=matrix.sum(1)>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(34),where=precision+recall>0)
    return dict(samples=int(matrix.sum()),accuracy=float(hits.sum()/max(1,matrix.sum())),
                macro_f1_34=float(f1.mean()),per_class_f1=f1.tolist(),
                per_class_recall=recall.tolist(),class_counts=matrix.sum(1).astype(int).tolist())


def add_confusion(matrix, truth, prediction):
    if not np.isin(truth,LABELS).all() or not np.isin(prediction,LABELS).all():
        raise ValueError('Class outside declared 34-class space')
    matrix += np.bincount(34*np.asarray(truth)+np.asarray(prediction),minlength=34*34).reshape(34,34)


def clean_pools(ckpt, roles, ids, chosen, classes, out):
    """Use only locked held-out roles from origins with local BASE participation."""
    banks={}; provenance=[]; origin_audit=[]
    for donor in ids:
        support=local_training_support(ckpt,donor,classes)
        base_x,base_y,_=roles.client_role(donor,'base')
        support={c:v for c,v in support.items() if np.any(base_y==c)}
        del base_x,base_y
        banks[donor]={}
        for role in LIMITS:
            x,y,rows=roles.client_role(donor,role); bank={}
            rng=np.random.default_rng(20261005+1009*donor+list(LIMITS).index(role)*104729)
            for label in sorted(support):
                entries=[]; seen=set()
                for index in rng.permutation(np.flatnonzero(y==label)):
                    digest=content_hash(x[index])
                    if digest in seen:continue
                    seen.add(digest); entries.append((donor,int(rows[index]),digest,x[index].copy(),label))
                    if len(entries)>=128:break
                bank[label]=entries
            banks[donor][role]=bank
            origin_audit.append(dict(donor=donor,role=role,support=list(support),
                                     counts={c:len(v) for c,v in bank.items()}))
            del x,y,rows
        print(f'Clean meta bank: donor={donor}',flush=True)
    write_json(out/'origin_support.json',origin_audit)
    pools={role:{} for role in LIMITS}; hashes={role:set() for role in LIMITS}
    for cid in ids:
        for role,limit in LIMITS.items():
            rng=np.random.default_rng(20261005+1009*cid+list(LIMITS).index(role)*104729)
            buckets={}
            for label in LABELS:
                entries=[e for d in chosen[cid] for e in banks[d][role].get(int(label),[])]
                if entries:buckets[int(label)]=[entries[i] for i in rng.permutation(len(entries))]
            selected=[]; seen=set()
            while buckets and len(selected)<limit:
                for label in list(buckets):
                    bucket=buckets[label]
                    while bucket and bucket[-1][2] in seen:bucket.pop()
                    if bucket and len(selected)<limit:
                        entry=bucket.pop();selected.append(entry);seen.add(entry[2]);hashes[role].add(entry[2])
                    if not bucket:del buckets[label]
            if len(selected)<32:raise ValueError(f'{cid}: insufficient clean {role} data ({len(selected)})')
            pools[role][cid]=dict(X=torch.from_numpy(np.stack([e[3] for e in selected])),
                                  y=np.asarray([e[4] for e in selected],dtype=np.int64))
            provenance.extend(dict(receiver=cid,role=role,origin=e[0],original_row=e[1],
                                   content_hash=e[2],label=e[4]) for e in selected)
    for role in LIMITS:
        for other in LIMITS:
            if role!=other and hashes[role]&hashes[other]:raise ValueError('Holdout content roles overlap')
        coverage=np.bincount(np.concatenate([v['y'] for v in pools[role].values()]),minlength=34)
        write_json(out/f'{role}_coverage.json',dict(counts=coverage,missing=np.flatnonzero(coverage==0)))
        if (coverage==0).any():raise ValueError(f'{role}: missing 34-class coverage; inspect coverage artifact')
    pd.DataFrame(provenance).to_csv(out/'meta_provenance.csv.gz',index=False,compression='gzip')
    del banks;gc.collect()
    return pools


class FrozenExperts:
    """Bound GPU residency to 17 models; reuse already fitted router state."""
    def __init__(self,ckpt,routers,device,fingerprints):
        self.ckpt=ckpt;self.routers=routers;self.device=device;self.fingerprints=fingerprints;self.cache=OrderedDict()

    def check(self,donor,model):
        if encoder_fingerprint(model)!=self.fingerprints[donor]:raise RuntimeError(f'Expert {donor} weights/masks mutated')

    def get(self,donor):
        if donor not in self.cache:
            if len(self.cache)>=17:
                old_id,(old,_) = self.cache.popitem(last=False)
                self.check(old_id,old)
                del old
            model,detector=_make_denice_client_model(self.ckpt,donor,self.device)
            self.check(donor,model)
            restore_context_detector(detector,self.routers[donor])
            model.eval()
            for parameter in model.parameters():parameter.requires_grad_(False)
            self.cache[donor]=(model,detector)
        self.cache.move_to_end(donor)
        return self.cache[donor]

    def records(self,chosen,inputs,batch_size):
        return {d:expert_features(*self.get(d),inputs,LABELS.tolist(),self.device,batch_size) for d in chosen}

    def clear(self):
        for donor,(model,_) in self.cache.items():self.check(donor,model)
        self.cache.clear();gc.collect()
        if self.device.startswith('cuda'):torch.cuda.empty_cache()


def predict_records(cid, records, bundle):
    """Label-blind end-to-end selector. No y/test task argument exists."""
    chosen=bundle['chosen'][cid]
    features,pred,names=feature_matrix(cid,chosen,records,bundle['alphas'][cid],bundle['priors'])
    if names!=bundle['gate_schema']:raise ValueError('Gate feature schema changed')
    scores=bundle['gate'].predict_proba(features)[:,list(bundle['gate'].classes_).index(1)].reshape(pred.shape)
    majority=vote(pred.T,np.ones(len(chosen)),LABELS,pred[:,0])
    gate,_=gate_decision(scores,pred,chosen,LABELS,top=1)
    matrix,available,schema=class_features(features,pred,scores,names,LABELS)
    if schema!=bundle['meta_schema']:raise ValueError('Class Meta feature schema changed')
    meta=masked_class_decision(bundle['meta'].predict_proba(matrix),bundle['meta'].classes_,
                               LABELS,available,pred[:,0],majority)
    return dict(Self=pred[:,0],Majority=majority,GateV2=gate,ClassMeta=meta),pred


def run_clean_meta(checkpoint_archive, role_dir, output_dir, device='cuda', batch_size=512):
    if batch_size<1:raise ValueError('Batch size must be positive')
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'completion.json',dict(completed=False,stage='fit',recipe=RECIPE))
    if sklearn.__version__!='1.6.1':raise RuntimeError('Use scikit-learn==1.6.1 for the locked historical recipe')
    roles=CleanRoleData(role_dir)
    archive=Path(checkpoint_archive)
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('checkpoint_archive_manifest.json'))
        if not manifest.get('full_terminal_checkpoint'):raise ValueError('Full terminal checkpoint required')
        with z.open(manifest['checkpoint']) as stream:
            last=torch.load(stream,map_location='cpu',weights_only=False)
        graph=last['cluster']; del last
    ckpt=load_denice_checkpoint(archive);ckpt['cluster']=graph
    if ckpt['config'].get('denice_data_roles_sha256')!=file_sha256(roles.root/'role_manifest.json'):
        raise ValueError('Backbone and selector data role locks differ')
    ids=sorted(int(cid) for cid in ckpt['client_model_states'])
    loader=IncrementalDataLoader(str(roles.source));classes=loader.task_classes
    if sorted(classes)!=list(range(6)) or sorted({c for values in classes.values() for c in values})!=LABELS.tolist():
        raise ValueError('Six-task 34-class metadata required')
    peers,edges=recorded_peers(ckpt,ids)
    alphas={cid:{} for cid in ids}
    for edge in edges:alphas[edge['receiver']][edge['donor']]=edge['alpha']
    chosen={cid:[cid]+peer_orders(cid,peers[cid],alphas[cid],(42,))['random_42'][:16] for cid in ids}
    if any(len(values)!=17 for values in chosen.values()):raise ValueError('Recipe requires 16 legitimate peers per receiver')
    pd.DataFrame(edges).to_csv(out/'peer_edges.csv',index=False)
    routers={};fingerprints={}
    for cid in ids:
        model,detector=_make_denice_client_model(ckpt,cid,device)
        fingerprints[cid]=encoder_fingerprint(model)
        detector.router_mode='multiclass_balanced'
        if not detector.activation_memory:raise ValueError(f'{cid}: no BASE router memory')
        detector.train_models(max(detector.activation_memory))
        routers[cid]=snapshot_context_detector(detector)
        if encoder_fingerprint(model)!=fingerprints[cid]:raise RuntimeError('Router fit changed backbone')
        del model,detector
        print(f'Frozen multiclass router: {cid}',flush=True)
    pools=clean_pools(ckpt,roles,ids,chosen,classes,out)
    experts=FrozenExperts(ckpt,routers,device,fingerprints);cache={role:{} for role in LIMITS}
    for role in LIMITS:
        for cid in ids:
            cache[role][cid]=experts.records(chosen[cid],pools[role][cid]['X'],batch_size)
            print(f'Clean expert evidence: role={role}, receiver={cid}',flush=True)
    priors=competence_priors(cache['calibration'],{cid:pools['calibration'][cid]['y'] for cid in ids})
    gate_x=[];gate_y=[];gate_schema=None
    for cid in ids:
        features,pred,names=feature_matrix(cid,chosen[cid],cache['fit'][cid],alphas[cid],priors)
        if gate_schema is not None and gate_schema!=names:raise ValueError('Gate schema mismatch')
        gate_schema=names;gate_x.append(features);gate_y.append((pred==pools['fit'][cid]['y'][:,None]).ravel().astype(int))
    gate=make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(32,16),alpha=.001,
        batch_size=1024,max_iter=50,early_stopping=False,random_state=20261005))
    with threadpool_limits(limits=1):gate.fit(np.concatenate(gate_x),np.concatenate(gate_y))
    del gate_x,gate_y
    meta_x=[];meta_y=[];coverage=np.zeros(34,dtype=int);meta_schema=None
    for cid in ids:
        features,pred,names=feature_matrix(cid,chosen[cid],cache['fit'][cid],alphas[cid],priors)
        scores=gate.predict_proba(features)[:,list(gate.classes_).index(1)].reshape(pred.shape)
        matrix,_,schema=class_features(features,pred,scores,names,LABELS)
        if meta_schema is not None and meta_schema!=schema:raise ValueError('Meta schema mismatch')
        meta_schema=schema;y=pools['fit'][cid]['y'];reachable=(pred==y[:,None]).any(1)
        meta_x.append(matrix[reachable]);meta_y.append(y[reachable]);coverage+=np.bincount(y[reachable],minlength=34)
    write_json(out/'reachable_fit_coverage.json',dict(counts=coverage,missing=np.flatnonzero(coverage==0)))
    if (coverage==0).any():raise ValueError('Missing reachable Class Meta fit classes; stop before test, inspect coverage')
    meta=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1000,solver='lbfgs',random_state=20261005))
    with threadpool_limits(limits=1):meta.fit(np.concatenate(meta_x),np.concatenate(meta_y))
    del meta_x,meta_y
    bundle=dict(recipe=RECIPE,routers=routers,fingerprints=fingerprints,chosen=chosen,alphas=alphas,
                priors=priors,gate=gate,meta=meta,gate_schema=gate_schema,meta_schema=meta_schema)
    validation={name:np.zeros((34,34),dtype=np.int64) for name in ('Self','Majority','GateV2','ClassMeta')}
    with threadpool_limits(limits=1):
        for cid in ids:
            predictions,_=predict_records(cid,cache['validation'][cid],bundle)
            for name,pred in predictions.items():add_confusion(validation[name],pools['validation'][cid]['y'],pred)
    write_json(out/'validation_metrics.json',{name:metrics(cm) for name,cm in validation.items()})
    joblib.dump(bundle,out/'frozen_pipeline.joblib',compress=3)
    pipeline_sha=file_sha256(out/'frozen_pipeline.joblib')
    lock=dict(recipe=RECIPE,checkpoint_archive_sha256=file_sha256(archive),
              role_manifest_sha256=file_sha256(roles.root/'role_manifest.json'),pipeline_sha256=pipeline_sha,
              sklearn_version=sklearn.__version__,torch_version=torch.__version__,client_ids=ids,
              test_policy='ALL original test rows; disjoint receiver shards; labels only after prediction',
              validation_policy='audit fixed historical recipe; no hyperparameter selection on test')
    write_json(out/'pipeline_lock.json',lock)
    # Reload persisted artifacts before accessing any test inputs or labels.
    bundle=joblib.load(out/'frozen_pipeline.joblib')
    experts.clear();experts=FrozenExperts(ckpt,bundle['routers'],device,bundle['fingerprints'])
    del cache,pools;gc.collect()
    write_json(out/'completion.json',dict(completed=False,stage='frozen_full_test',lock=lock))
    x,y=loader.get_full_test_data()
    if not np.array_equal(np.unique(y.numpy()),LABELS):raise ValueError('Final test must contain all 34 classes')
    seed=int(ckpt['config'].get('seed',42))+104729*5
    shards,partition=_partition_test_data_by_client(x,y,ids,seed,lazy_features=True)
    total=len(y);del loader,x,y;gc.collect()
    confusion={name:np.zeros((34,34),dtype=np.int64) for name in validation};clients=[]
    evaluated=0;oracle_hits=0;started=time.monotonic()
    with gzip.open(out/'final_predictions.csv.gz','wt',encoding='utf-8',newline='') as stream, threadpool_limits(limits=1):
        header=True
        for cid in ids:
            shard=shards[cid];local={name:np.zeros((34,34),dtype=np.int64) for name in confusion}
            for start in range(0,len(shard['y_test']),batch_size):
                stop=start+batch_size;inputs=shard['X_test'][start:stop]
                records=experts.records(bundle['chosen'][cid],inputs,batch_size)
                predictions,pred=predict_records(cid,records,bundle)
                # Truth enters only after the label-blind final decisions above.
                truth=shard['y_test'][start:stop].numpy()
                for name,value in predictions.items():
                    add_confusion(confusion[name],truth,value);add_confusion(local[name],truth,value)
                oracle=(pred==truth[:,None]).any(1);oracle_hits+=int(oracle.sum());evaluated+=len(truth)
                frame=pd.DataFrame(dict(client_id=cid,global_test_row=shard['X_test'].indices[start:stop].numpy(),
                                        y_true=truth,**predictions,actual_routed_oracle=oracle))
                frame.to_csv(stream,index=False,header=header);header=False
                if evaluated% (batch_size*100)<batch_size:
                    print(f'Frozen full test: {evaluated}/{total} rows',flush=True)
            clients.append(dict(client_id=cid,metrics={name:metrics(cm) for name,cm in local.items()}))
            write_json(out/'client_metrics.json',clients)
            del shards[cid]
            print(f'Frozen full test receiver={cid}: {evaluated}/{total}',flush=True)
    if evaluated!=total:raise RuntimeError('Whole-test coverage failed')
    if file_sha256(out/'frozen_pipeline.joblib')!=pipeline_sha:raise RuntimeError('Frozen artifact changed')
    for donor,(model,_) in experts.cache.items():
        if encoder_fingerprint(model)!=bundle['fingerprints'][donor]:raise RuntimeError('Expert weights mutated')
    result=dict(completed=True,metrics={name:metrics(cm) for name,cm in confusion.items()},
                actual_routed_oracle=oracle_hits/total,total_test_rows=total,classes=LABELS.tolist(),
                model_queries=17*total,seconds=time.monotonic()-started,partition=partition,lock=lock,
                scope='clean retrained backbone + self/16 cached peer experts + frozen Gate V2/Class Meta')
    np.savez_compressed(out/'confusion_matrices.npz',**confusion)
    write_json(out/'completion.json',result);experts.clear()
    print(json.dumps(result['metrics'],indent=2),flush=True)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-archive',required=True)
    parser.add_argument('--role-dir',required=True)
    parser.add_argument('--output-dir',required=True,help='New directory; existing artifacts are never overwritten')
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size',type=int,default=512)
    args=parser.parse_args()
    run_clean_meta(args.checkpoint_archive,args.role_dir,args.output_dir,args.device,args.batch_size)
