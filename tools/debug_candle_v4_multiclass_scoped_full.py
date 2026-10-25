"""Read-only full-TEST scoped-expert audit; never trains or tunes thresholds.

Reuse the verified native-multiclass confusion matrices. Recompute native class
decisions only where experts override, then subtract/add those confusion cells.
Routing is recomputed on ALL rows and checked against the native reference.
The forced-route executor receives input-predicted routes, NEVER true tasks.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import sys,json,pathlib,time,copy,hashlib,argparse,csv
import numpy as np
import torch
from threadpoolctl import threadpool_limits


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(8*1024**2),b''):h.update(part)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    root=pathlib.Path(r'C:\Users\khoak\Downloads\candle_v1_multiclass_audit')
    checkpoint=pathlib.Path(r'C:\Users\khoak\Downloads\candle_v4_debug\continuation_state_latest.pt')
    out=pathlib.Path('audit_denice/candle_v4_multiclass_scoped_20261024');out.mkdir(exist_ok=True)
    sys.path.insert(0,str(root/'source'))
    from eval_checkpoint import _make_denice_client_model
    from fed_learning.training.decentralized_denice_il import _partition_test_data_by_client
    from fed_learning.training.denice_eval import _route_episodes,_denice_routed_logits_with_episodes
    from fed_learning.training.candle_v4_shadow import make_expert,digest
    from fed_learning.training.candle_v4_terminal_test import statistics
    torch.set_num_threads(2);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    reference_path=root/'router_comparison.json';reference=json.loads(reference_path.read_text())
    assert reference['completed'] and reference['completed_clients']==100
    lock=dict(version='candle_v4_mc_scoped_readout_v1',reference_sha256=sha(reference_path),
              checkpoint_sha256=sha(checkpoint),test_sha256=sha(root/'global_test_data.npz'),
              threshold=.5,router='multiclass_balanced',seed=523687,batch=512,
              source_commit=reference['source_commit'],no_training=True)
    target=out/'comparison.json';result=dict(lock=lock,completed=False,per_client={})
    if args.resume:
        result=json.loads(target.read_text());assert result['lock']==lock,'Resume inputs/config changed'
    elif target.exists():raise ValueError('Existing result requires --resume')
    print('Loading frozen checkpoint and TEST',flush=True)
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False);bank=ckpt['candle_v4_shadow_state']
    assert bank['spec']['fixed_threshold']==.5 and ckpt['meta']['completed_task']==5
    with np.load(root/'global_test_data.npz') as data:x=torch.from_numpy(data['X_test']);y=torch.from_numpy(data['y_test']).long()
    ids=sorted(ckpt['client_ids']);assert ids==list(range(100))
    shards,partition=_partition_test_data_by_client(x,y,ids,seed=523687,lazy_features=True)
    assert len(y)==reference['test_rows']==13505771
    device='cuda' if torch.cuda.is_available() else 'cpu';result['device']=device
    started=time.time();previous=result.get('wall_seconds',0.)
    print('Clients already saved:',len(result['per_client']),flush=True)
    for cid in ids:
        if str(cid) in result['per_client']:
            del shards[cid];continue
        tick=time.time();ref=reference['per_client'][str(cid)]['multiclass']
        keys=sorted(k for k in bank['entries'] if k[0]==cid and bank['status']['coverage'][k]=='ACTIVE')
        classes=np.array([k[1] for k in keys],dtype=np.int64)
        tasks=np.array([bank['entries'][k]['task'] for k in keys],dtype=np.int64)
        assert np.array_equal(tasks,classes//6)
        experts=[]
        for key in keys:
            model=make_expert(42);state=bank['engines']['coverage'][key]['branches']['joint_class_balanced']['model']
            cert=bank['certificates']['coverage'][key]
            assert cert['sha256']==digest(state) and cert['input_sha256']==bank['spec']['input_sha256']
            assert cert['birth_task']==bank['entries'][key]['task'] and cert['threshold']==.5
            model.load_state_dict(state);model.eval();experts.append(model)
        model,detector=_make_denice_client_model(ckpt,cid,device);model.eval()
        before=digest(model.state_dict());detector.router_mode='multiclass_balanced'
        with threadpool_limits(limits=2):detector.train_models(max(detector.activation_memory))
        cm=np.asarray(ref['confusion_matrix'],dtype=np.int64).copy();routecm=np.zeros((6,6),dtype=np.int64)
        removed=np.zeros((34,34),dtype=np.int64);added=np.zeros_like(removed)
        summary=dict(rows=len(shards[cid]['y_test']),expert_count=len(keys),triggered=0,multiple_triggers=0,
                     rescue=0,break_count=0,unchanged_correct=0,changed_wrong=0)
        attrs={int(c):dict(selected=0,rescue=0,break_count=0,false_activation=0) for c in classes}
        saved={k:[] for k in ['indices','truth','route','native','expert','score']}
        with torch.no_grad(),threadpool_limits(limits=2):
            for start in range(0,summary['rows'],512):
                inputs=shards[cid]['X_test'][start:start+512];truth=shards[cid]['y_test'][start:start+512].numpy()
                routes=np.asarray(_route_episodes(model,inputs.to(device),detector),dtype=np.int64)
                np.add.at(routecm,(truth//6,routes),1)
                if not keys:continue
                scores=np.column_stack([torch.sigmoid(e(inputs.reshape(-1,39))).flatten().numpy() for e in experts])
                eligible=(tasks[None,:]==routes[:,None])&(scores>=.5);fires=eligible.any(1)
                idx=np.flatnonzero(fires)
                if not len(idx):continue
                winner=np.where(eligible,scores,-1.).argmax(1)[idx];prediction=classes[winner]
                # Force the already predicted route for subset execution; no label is passed.
                logits,_=_denice_routed_logits_with_episodes(model,inputs[idx].to(device),detector,
                    list(range(34)),device,inference_policy='oracle_hard',oracle_episodes=routes[idx])
                native=logits.argmax(1).cpu().numpy();labels=truth[idx]
                np.add.at(removed,(labels,native),1);np.add.at(added,(labels,prediction),1)
                rescue=(native!=labels)&(prediction==labels);broken=(native==labels)&(prediction!=labels)
                summary['triggered']+=len(idx);summary['multiple_triggers']+=int((eligible[idx].sum(1)>1).sum())
                summary['rescue']+=int(rescue.sum());summary['break_count']+=int(broken.sum())
                summary['unchanged_correct']+=int(((native==labels)&(prediction==labels)).sum())
                summary['changed_wrong']+=int(((native!=labels)&(prediction!=labels)&(native!=prediction)).sum())
                for c in np.unique(prediction):
                    chosen=prediction==c;a=attrs[int(c)];a['selected']+=int(chosen.sum())
                    a['rescue']+=int((chosen&rescue).sum());a['break_count']+=int((chosen&broken).sum());a['false_activation']+=int((chosen&(labels!=prediction)).sum())
                for name,value in dict(indices=start+idx,truth=labels,route=routes[idx],native=native,
                                       expert=prediction,score=scores[idx,winner]).items():saved[name].append(value)
        assert before==digest(model.state_dict()),'Model mutated by inference'
        discrepancy=int(np.abs(routecm-np.asarray(ref['route_confusion_matrix'])).sum())
        if discrepancy>0:raise RuntimeError(f'{cid}: multiclass routing does not reproduce reference ({discrepancy})')
        if np.any(removed>cm):raise RuntimeError(f'{cid}: selected native cells exceed reference; check subset numerical reproduction')
        fused=cm-removed+added
        assert fused.sum()==summary['rows'] and np.trace(fused)-np.trace(cm)==summary['rescue']-summary['break_count']
        summary.update(native=statistics(cm),scoped=statistics(fused),route_accuracy=float(np.trace(routecm)/routecm.sum()),
                       experts=attrs,route_reference_exact=True,seconds=time.time()-tick)
        np.savez_compressed(out/f'client_{cid}_overrides.npz',**{k:np.concatenate(v) if v else np.array([]) for k,v in saved.items()})
        result['per_client'][str(cid)]=summary
        total=sum((np.asarray(v['scoped']['confusion_matrix']) for v in result['per_client'].values()),np.zeros((34,34),dtype=np.int64))
        base=sum((np.asarray(v['native']['confusion_matrix']) for v in result['per_client'].values()),np.zeros((34,34),dtype=np.int64))
        result['metrics']=dict(V1_multiclass=statistics(base),V4_multiclass_scoped=statistics(total))
        result['totals']={k:sum(v[k] for v in result['per_client'].values()) for k in ['rows','expert_count','triggered','multiple_triggers','rescue','break_count']}
        result['completed']=len(result['per_client'])==100;result['wall_seconds']=previous+time.time()-started
        tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(target)
        if (cid+1)%10==0 or cid==0:print(f'done={cid+1}/100 rows={result["totals"]["rows"]} rescue={result["totals"]["rescue"]} break={result["totals"]["break_count"]}',flush=True)
        del model,detector,shards[cid]
    print(json.dumps({k:{a:v[a] for a in ['accuracy','macro_f1']} for k,v in result['metrics'].items()}),flush=True)


if __name__=='__main__':main()
