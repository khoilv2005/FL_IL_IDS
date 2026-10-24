"""Frozen full-TEST mask-only control; no training, calibration or gate search."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import pathlib,sys,json,time,copy,argparse,hashlib
import numpy as np
import torch
from threadpoolctl import threadpool_limits


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    root=pathlib.Path(r'C:\Users\khoak\Downloads\candle_v1_multiclass_audit')
    prior=pathlib.Path('audit_denice/candle_v4_multiclass_scoped_20261024')
    out=pathlib.Path('audit_denice/candle_v4_mask_router_audit_20261024');out.mkdir(exist_ok=True)
    sys.path.insert(0,str(root/'source'))
    from eval_checkpoint import _make_denice_client_model
    from fed_learning.training.decentralized_denice_il import _partition_test_data_by_client
    from fed_learning.training.denice_eval import _denice_routed_logits_with_episodes,_mask_logits_to_classes
    from fed_learning.training.candle_v4_terminal_test import statistics
    from fed_learning.training.candle_v4_shadow import digest
    torch.set_num_threads(2);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    ref_path=root/'router_comparison.json';ref=json.loads(ref_path.read_text());old=json.loads((prior/'comparison.json').read_text())
    assert ref['completed'] and old['completed']
    checkpoint=pathlib.Path(r'C:\Users\khoak\Downloads\candle_v4_debug\continuation_state_latest.pt')
    lock=dict(protocol='mask_only_full_v1',checkpoint_sha256=sha(checkpoint),test_sha256=sha(root/'global_test_data.npz'),
              reference_sha256=sha(ref_path),scoped_reference_sha256=sha(prior/'comparison.json'),seed=523687,
              batch=2048,fallback_batch=512,numerical_confusion_tolerance_fraction=.001,
              router='multiclass_balanced',threshold=.5,no_training=True)
    target=out/'mask_only.json';result=dict(lock=lock,completed=False,per_client={})
    if args.resume:
        result=json.loads(target.read_text());assert result['lock']==lock
    elif target.exists():raise ValueError('Use --resume')
    print('Loading checkpoint and original TEST',flush=True)
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
    with np.load(root/'global_test_data.npz') as z:x=torch.from_numpy(z['X_test']);y=torch.from_numpy(z['y_test']).long()
    ids=sorted(ckpt['client_ids']);shards,_=_partition_test_data_by_client(x,y,ids,seed=523687,lazy_features=True)
    assert len(y)==13505771
    previous=result.get('wall_seconds',0.);started=time.time();device='cuda'
    for cid in ids:
        if str(cid) in result['per_client']:del shards[cid];continue
        tick=time.time();model,det=_make_denice_client_model(ckpt,cid,device);model.eval();before=digest(model.state_dict())
        det.router_mode='multiclass_balanced'
        with threadpool_limits(limits=2):det.train_models(max(det.activation_memory))
        local_classes=copy.deepcopy(det.episode_classes)
        assert all(local_classes.get(t) for t in det.multiclass_episodes)
        # Same router estimator, thresholds, weights and adapters. Only class availability changes.
        det.episode_classes={int(t):list(range(int(t)*6,min(int(t)*6+6,34))) for t in local_classes}
        cache=np.load(prior/f'client_{cid}_overrides.npz');override_idx=cache['indices'].astype(np.int64)
        for batch in [2048,512]:
            truth=shards[cid]['y_test'].numpy();native=np.empty(len(truth),dtype=np.int64);mask=np.empty_like(native);routes=np.empty_like(native)
            with torch.no_grad(),threadpool_limits(limits=2):
                for start in range(0,len(truth),batch):
                    inputs=shards[cid]['X_test'][start:start+batch].to(device)
                    logits,ep=_denice_routed_logits_with_episodes(model,inputs,det,list(range(34)),device,inference_policy='pred_hard')
                    mask[start:start+len(inputs)]=logits.argmax(1).cpu().numpy();routes[start:start+len(inputs)]=ep
                    restored=torch.empty(len(inputs),dtype=torch.long,device=device)
                    for task in np.unique(ep):
                        idx=np.flatnonzero(ep==task);selected=torch.as_tensor(idx,device=device)
                        restored[selected]=_mask_logits_to_classes(logits.index_select(0,selected),local_classes[int(task)]).argmax(1)
                    native[start:start+len(inputs)]=restored.cpu().numpy()
            def matrix(pred,n=34):
                cm=np.zeros((n,n),dtype=np.int64);np.add.at(cm,(truth if n==34 else truth//6,pred),1);return cm
            nativecm=matrix(native);routecm=matrix(routes,6)
            reference=ref['per_client'][str(cid)]['multiclass']
            native_l1=int(np.abs(nativecm-np.asarray(reference['confusion_matrix'])).sum())
            route_l1=int(np.abs(routecm-np.asarray(reference['route_confusion_matrix'])).sum())
            cached_route_exact=np.array_equal(routes[override_idx],cache['route'].astype(np.int64))
            if max(native_l1,route_l1)/(2*len(truth))<=.001 and cached_route_exact:break
            if batch==512:raise RuntimeError(f'{cid}: failed native/router reproduction')
        scoped=native.copy();scoped[override_idx]=cache['expert'].astype(np.int64)
        scopedcm=matrix(scoped);maskcm=matrix(mask)
        expected=np.asarray(old['per_client'][str(cid)]['scoped']['confusion_matrix'])
        scoped_l1=int(np.abs(scopedcm-expected).sum())
        if scoped_l1/(2*len(truth))>.001:raise RuntimeError(f'{cid}: cached scoped prediction mismatch')
        assert before==digest(model.state_dict())
        row=dict(rows=len(truth),batch_used=batch,native_confusion_l1=native_l1,route_confusion_l1=route_l1,
                 scoped_confusion_l1=scoped_l1,cached_override_routes_exact=cached_route_exact,
                 native=statistics(nativecm),mask_only=statistics(maskcm),v4_scoped=statistics(scopedcm),
                 mask_rescue_vs_native=int(((native!=truth)&(mask==truth)).sum()),
                 mask_break_vs_native=int(((native==truth)&(mask!=truth)).sum()),
                 v4_rescue_vs_mask=int(((mask!=truth)&(scoped==truth)).sum()),
                 v4_break_vs_mask=int(((mask==truth)&(scoped!=truth)).sum()),seconds=time.time()-tick)
        np.savez_compressed(out/f'client_{cid}_predictions.npz',native=native.astype(np.int8),mask_only=mask.astype(np.int8),
                            v4_scoped=scoped.astype(np.int8),route=routes.astype(np.int8))
        result['per_client'][str(cid)]=row
        result['metrics']={a:statistics(sum((np.asarray(v[a]['confusion_matrix']) for v in result['per_client'].values()),np.zeros((34,34),dtype=np.int64))) for a in ['native','mask_only','v4_scoped']}
        result['totals']={k:sum(v[k] for v in result['per_client'].values()) for k in ['rows','mask_rescue_vs_native','mask_break_vs_native','v4_rescue_vs_mask','v4_break_vs_mask']}
        result['completed']=len(result['per_client'])==100;result['wall_seconds']=previous+time.time()-started
        tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(target)
        if cid==0 or (cid+1)%10==0:print('done',cid+1,'/100',result['totals'],flush=True)
        del model,det,shards[cid]
    print(json.dumps({a:{k:v[k] for k in ['accuracy','macro_f1']} for a,v in result['metrics'].items()}),flush=True)


if __name__=='__main__':main()
