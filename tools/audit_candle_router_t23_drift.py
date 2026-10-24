"""Fixed metadata-selected panel: encoder drift vs router-bank competition."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import pathlib,sys,json,copy,time
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits


def main():
    root=pathlib.Path(r'C:\Users\khoak\Downloads\candle_v1_multiclass_audit')
    data=pathlib.Path(r'C:\Users\khoak\Downloads\candle_v4_debug')
    out=pathlib.Path('audit_denice/candle_v4_mask_router_audit_20261024');out.mkdir(exist_ok=True)
    sys.path.insert(0,str(root/'source'))
    from eval_checkpoint import _make_denice_client_model
    from fed_learning.training.decentralized_denice_il import _partition_test_data_by_client
    from fed_learning.training.candle_v4_shadow import digest
    torch.set_num_threads(2);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    print('Loading Task3/Task5 snapshots',flush=True)
    old=torch.load(data/'checkpoint_task_3.pt',map_location='cpu',weights_only=False)
    final=torch.load(data/'continuation_state_latest.pt',map_location='cpu',weights_only=False)
    assert old['task']==3 and old['final_round_id']==19 and final['meta']['completed_task']==5
    def context(ck,cid):
        a=ck['client_algorithm_states'][cid];return a.get('denice',a)['context_detector']
    eligible=[cid for cid in sorted(old['client_model_states']) if all(t in context(old,cid)['activation_memory'] and len(context(old,cid)['activation_memory'][t]) and t in context(final,cid)['activation_memory'] and len(context(final,cid)['activation_memory'][t]) for t in [2,3])]
    ids=eligible[:8];assert len(ids)==8
    with np.load(root/'global_test_data.npz') as z:x=torch.from_numpy(z['X_test']);y=torch.from_numpy(z['y_test']).long()
    shards,_=_partition_test_data_by_client(x,y,sorted(final['client_ids']),seed=523687,lazy_features=True)
    result=dict(completed=False,selection='First 8 sorted clients with T2 and T3 memories in both T3 and T5 checkpoints',
        client_ids=ids,eligible_client_count=len(eligible),sample_policy='Up to128 per class, T2/T3 only, deterministic RNG seed42+client; no score-based selection',
        no_training=True,no_TEST_fitting=True,clients={})
    for cid in ids:
        labels=shards[cid]['y_test'].numpy();rng=np.random.default_rng(42+cid)
        idx=np.concatenate([rng.permutation(np.flatnonzero(labels==c))[:128] for c in range(12,24)])
        inputs=shards[cid]['X_test'][idx];truth=labels[idx];tasks=truth//6
        models={};detectors={};features={};signatures={}
        with threadpool_limits(limits=2):
            for name,ck in [('T3',old),('T5',final)]:
                model,det=_make_denice_client_model(ck,cid,'cuda');model.eval();det.router_mode='multiclass_balanced'
                det.train_models(max(det.activation_memory));models[name]=model;detectors[name]=det;signatures[name]=digest(model.state_dict())
                acts={}
                with torch.no_grad():
                    for start in range(0,len(inputs),512):
                        part=model.get_context_activations_per_sample(inputs[start:start+512].to('cuda'))
                        for layer,value in part.items():acts.setdefault(layer,[]).append(value.cpu().numpy())
                features[name]={layer:np.concatenate(values) for layer,values in acts.items()}
        local_classes=context(old,cid)['episode_classes'];owned=np.array([int(c) in local_classes[int(c)//6] for c in truth])
        row=dict(rows=len(truth),owned_rows=int(owned.sum()),unowned_rows=int((~owned).sum()),conditions={},layer_drift={})
        for enc in ['T3','T5']:
            for bank in ['T3','T5']:
                det=detectors[bank];binary=det.binarize_layer_activations(features[enc]);pred=det.predict_episodes_batch(binary)
                row['conditions'][enc+'_encoder_'+bank+'_memory']=dict(correct=int((pred==tasks).sum()),
                    owned_correct=int(((pred==tasks)&owned).sum()),unowned_correct=int(((pred==tasks)&~owned).sum()),
                    per_task={str(t):dict(rows=int((tasks==t).sum()),correct=int(((tasks==t)&(pred==t)).sum())) for t in [2,3]})
        det=detectors['T5'];bits=det.binarize_layer_activations(features['T5'])
        _,probabilities=det.predict_episodes_with_scores(bits)
        restricted=np.array([2,3])[probabilities[:,[2,3]].argmax(1)]
        row['two_task_restricted_diagnostic']=dict(correct=int((restricted==tasks).sum()),note='Candidate-set oracle diagnostic only; not a global inference metric')
        row['shared_memory_equal']={str(t):digest(context(old,cid)['activation_memory'][t])==digest(context(final,cid)['activation_memory'][t]) for t in [2,3]}
        row['thresholds_equal']=digest(detectors['T3'].binarize_thresholds)==digest(detectors['T5'].binarize_thresholds)
        row['feature_policy_equal']=all(digest(getattr(detectors['T3'],k,None))==digest(getattr(detectors['T5'],k,None)) for k in ['stable_feature_mask','routing_feature_mask'])
        b3=detectors['T3'].binarize_layer_activations(features['T3']);b5=detectors['T3'].binarize_layer_activations(features['T5'])
        row['effective_binary_flip_fraction']=float((b3!=b5).mean())
        policy=getattr(detectors['T3'],'routing_feature_mask',None)
        if policy is None:policy=getattr(detectors['T3'],'stable_feature_mask',None)
        policy=np.ones(b3.shape[1],dtype=bool) if policy is None else np.asarray(policy).astype(bool)
        a=np.concatenate([features['T3'][k] for k in ['conv1','conv2','conv3','gru']],axis=1)[:,policy]
        b=np.concatenate([features['T5'][k] for k in ['conv1','conv2','conv3','gru']],axis=1)[:,policy]
        row['effective_feature_count']=int(policy.sum())
        row['selected_continuous_relative_l1_drift']=float(np.abs(a-b).sum()/max(float(np.abs(a).sum()),1e-12))
        row['selected_continuous_max_abs_drift']=float(np.abs(a-b).max())
        for layer in ['conv1','conv2','conv3','gru']:
            threshold=detectors['T3'].binarize_thresholds[layer]
            a=features['T3'][layer]>threshold;b=features['T5'][layer]>threshold
            row['layer_drift'][layer]=dict(bit_flip_fraction=float((a!=b).mean()),density_T3=float(a.mean()),density_T5=float(b.mean()),
                owned_bit_flip_fraction=float((a[owned]!=b[owned]).mean()))
        mem=detectors['T5'].activation_memory;xx=np.concatenate([np.asarray(v) for t,v in sorted(mem.items()) if len(v)]);yy=np.concatenate([np.full(len(v),int(t)) for t,v in sorted(mem.items()) if len(v)])
        memory_pred=detectors['T5'].predict_episodes_batch(xx);row['saved_memory_train']=dict(rows=len(yy),correct=int((memory_pred==yy).sum()))
        train=[];hold=[];rng=np.random.default_rng(42)
        for t in np.unique(yy):
            ix=rng.permutation(np.flatnonzero(yy==t));n=max(1,len(ix)//5);hold.extend(ix[:n]);train.extend(ix[n:])
        with threadpool_limits(limits=2):
            fit=LogisticRegression(C=1.,max_iter=1000,class_weight='balanced',random_state=0).fit(xx[train],yy[train]);hp=fit.predict(xx[hold])
        row['saved_memory_holdout']=dict(rows=len(hold),correct=int((hp==yy[hold]).sum()),note='Held-out saved sketches; not independent raw-data generalization')
        allbits=np.packbits(xx.astype(bool),axis=1);patterns={}
        for z,t in zip(allbits,yy):patterns.setdefault(z.tobytes(),set()).add(int(t))
        row['contradictory_memory_pattern_rows']=sum(len(patterns[z.tobytes()])>1 for z in allbits)
        pattern_counts={}
        for z,t in zip(allbits,yy):
            counts=pattern_counts.setdefault(z.tobytes(),{});counts[int(t)]=counts.get(int(t),0)+1
        row['saved_memory_identical_pattern_accuracy_ceiling']=sum(max(c.values()) for c in pattern_counts.values())/len(yy)
        assert all(signatures[name]==digest(model.state_dict()) for name,model in models.items())
        np.savez_compressed(out/f'router_panel_client_{cid}.npz',truth=truth,owned=owned,
            bits_T3=detectors['T3'].binarize_layer_activations(features['T3']),bits_T5=detectors['T3'].binarize_layer_activations(features['T5']))
        result['clients'][str(cid)]=row;result['completed']=len(result['clients'])==8
        (out/'router_t23_drift.json').write_text(json.dumps(result,indent=2))
        print('panel client',cid,'rows',len(truth),'conditions',row['conditions'],flush=True)
        del models,detectors,features


if __name__=='__main__':main()
