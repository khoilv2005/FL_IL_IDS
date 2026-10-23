"""Audit a trusted user training archive, with optional small checkpoint probe.

Source archives are immutable. Outputs are local diagnostic evidence only.
The balanced panel uses a local router and is not the logged cluster benchmark.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.audit_denice_artifacts import cluster_summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dataset', type=Path)
    p.add_argument('--probe-only', action='store_true', help='Use already extracted local evidence/checkpoints')
    args = p.parse_args()
    if args.probe_only:
        evidence=json.loads((args.output/'archive_evidence.json').read_text(encoding='utf-8'))
        if evidence['archive'] != str(args.archive):
            raise ValueError('Archive does not match extracted evidence')
        probe(args, evidence['config'], args.output/'checkpoints')
        return
    args.output.mkdir(parents=True, exist_ok=False)
    with args.archive.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    with ZipFile(args.archive) as z:
        config_name = next(n for n in z.namelist() if n.endswith('/config.json'))
        prefix = config_name.removesuffix('config.json')
        def read(name):
            return json.loads(z.read(prefix + name))
        config, tasks, rounds = read('config.json'), read('task_metrics.json'), read('round_metrics.json')
        clusters = read('cluster_history.json')
        names = set(z.namelist())
        index = read('checkpoint_index.json')['checkpoints']
        missing = sorted({prefix+r[k] for r in index for k in ['path','base','previous']
                          if r.get(k)} - names)
        expected = {(t,r) for t in range(config['task_start'], config['task_end']+1)
                    for r in range(config['rounds_per_task'])}
        logged = Counter((r['task'],r['round']) for r in rounds)
        stored = Counter((r['task_id'],r['round_id']) for r in index if r['kind']=='round')
        final = tasks[-1]
        confusion = final.get('route_confusion', {})
        evidence = {
            'archive': str(args.archive), 'sha256': digest, 'config': config,
            'rounds_complete': set(logged)==expected and all(v==1 for v in logged.values()),
            'round_checkpoints_complete': set(stored)==expected and all(v==1 for v in stored.values()),
            'missing_index_dependencies': missing,
            'continuation_files': [n for n in names if 'continuation_state' in n],
            'task_metrics': [{k:v for k,v in r.items() if k!='per_client'} for r in tasks],
            'cluster_summary': cluster_summary(clusters),
            'routing_by_true_task': {t:{'accuracy':row.get(t,0)/sum(row.values()),
                                       'counts':row} for t,row in confusion.items()},
            'train_loss_trajectories': {t:[r['train_loss'] for r in rounds if r['task']==t]
                                        for t in range(config['task_end']+1)},
            'selective_fc2_protection': [r.get('selective_fc2_row_protection') for r in clusters],
            'final_router_freshness': {t:next(r['router_freshness'] for r in reversed(clusters) if r['task']==t)
                                      for t in range(config['task_end']+1)},
        }
        (args.output/'archive_evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
        print('Archive audit:', {k:evidence[k] for k in ['rounds_complete','round_checkpoints_complete','missing_index_dependencies','continuation_files','routing_by_true_task']}, flush=True)
        if not args.dataset:
            return
        # Only basenames inside the known run are written; no archive paths are extracted.
        checkpoint_dir = args.output/'checkpoints'
        checkpoint_dir.mkdir()
        import shutil
        for info in z.infolist():
            if info.filename.startswith(prefix) and info.filename.endswith('.pt'):
                with z.open(info) as source, (checkpoint_dir/Path(info.filename).name).open('xb') as dest:
                    shutil.copyfileobj(source, dest, length=1024*1024)
        print('Checkpoint files extracted with CRC verification.', flush=True)
    probe(args, config, checkpoint_dir)


def probe(args, config, checkpoint_dir):
    import numpy as np
    import torch
    from sklearn.metrics import f1_score
    from tools.diagnose_denice_real_client import sample_npz
    from eval_checkpoint import _make_denice_client_model
    from fed_learning.training.denice_delta_checkpoint import load_denice_checkpoint
    from fed_learning.training.denice_eval import _denice_routed_logits_with_episodes
    torch.set_num_threads(1)
    panel_path=args.output/'probe_panel.npz'
    if panel_path.exists():
        with np.load(panel_path,allow_pickle=False) as panel:
            x,y,indices=torch.from_numpy(panel['x']),torch.from_numpy(panel['y']),panel['indices'].tolist()
    else:
        with ZipFile(args.dataset) as z:
            x,y,indices = sample_npz(z.read('100-clients/global_test_data.npz'),'test',32,2027,num_classes=24)
        np.savez_compressed(panel_path,x=x.numpy(),y=y.numpy(),indices=indices)
    import sklearn
    result = {'panel_seed':2027,'per_class_cap':32,'indices':indices,'sample_count':len(y),
              'sklearn_evaluation_version':sklearn.__version__,
              'scope':'All checkpoint clients, local router; balanced diagnostic, not logged cluster evaluation',
              'caveats':['fp16 delta reconstruction is lossy','round checkpoint precedes end-task aging',
                         'serialized sklearn estimator version can differ; inspect warnings'],
              'tasks':[]}
    for task in range(config['task_end']+1):
        print('Reconstruct task',task,flush=True)
        ckpt=load_denice_checkpoint(str(checkpoint_dir/f"checkpoint_task_{task}_round_{config['rounds_per_task']-1}.pt"))
        rows=[]
        selected=y<6*(task+1)
        hx,hy=x[selected],y[selected]
        for cid in sorted(ckpt['client_model_states']):
            model,router=_make_denice_client_model(ckpt,cid,'cpu')
            states=ckpt['client_algorithm_states'][cid]
            state=states.get('denice',states)
            item={'client':cid,'has_connection_masks':bool(state.get('connection_masks')),'policies':{}}
            known={int(c) for cs in router.episode_classes.values() for c in cs}
            supported=np.isin(hy.numpy(),list(known))
            item['known_classes']=sorted(known)
            for policy in ['backbone_nomask','oracle_hard','pred_hard']:
                kwargs={'oracle_episodes':hy.numpy()//6} if policy=='oracle_hard' else {}
                logits,routes=_denice_routed_logits_with_episodes(model,hx,router,list(range(6*(task+1))),
                                                                'cpu',inference_policy=policy,**kwargs)
                pred=logits.argmax(1).numpy()
                metrics={}
                for old in range(task+1):
                    mask=(hy.numpy()//6==old)&supported
                    metrics[str(old)]={'n':int(mask.sum()),'correct':int((pred[mask]==hy.numpy()[mask]).sum()),
                        'f1_macro':float(f1_score(hy.numpy()[mask],pred[mask],labels=list(range(6*old,6*(old+1))),average='macro',zero_division=0)) if mask.any() else None}
                item['policies'][policy]=metrics
            rows.append(item)
        result['tasks'].append({'task':task,'clients':rows})
        (args.output/'checkpoint_probe.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        for policy in ['backbone_nomask','oracle_hard','pred_hard']:
            acc=[]
            for old in range(task+1):
                ms=[r['policies'][policy][str(old)] for r in rows]
                acc.append(sum(m['correct'] for m in ms)/max(1,sum(m['n'] for m in ms)))
            print('Task',task,policy,acc,flush=True)
        del ckpt


def summarize_probe(output):
    """Keep a fixed participant cohort when reporting cross-task differences."""
    output=Path(output)
    p=json.loads((output/'checkpoint_probe.json').read_text(encoding='utf-8'))
    cfg=json.loads((output/'archive_evidence.json').read_text(encoding='utf-8'))['config']
    if len(p['tasks']) != cfg['task_end']+1:
        raise ValueError('Wait for all task probes before summarizing')
    cohort=set.intersection(*[{c['client'] for c in t['clients']} for t in p['tasks']])
    result={'fixed_client_ids':sorted(cohort),'matrices':{},
            'checkpoint_client_count':sum(len(t['clients']) for t in p['tasks']),
            'clients_with_connection_masks':sum(c['has_connection_masks'] for t in p['tasks'] for c in t['clients']),
            'scope':p['scope'],'caveats':p['caveats']}
    for subset in ['all_checkpoint_clients','fixed_cohort']:
        result['matrices'][subset]={}
        for policy in ['backbone_nomask','oracle_hard','pred_hard']:
            matrix=[]
            for task in p['tasks']:
                rows=[c for c in task['clients'] if subset=='all_checkpoint_clients' or c['client'] in cohort]
                values=[]
                for old in range(task['task']+1):
                    ms=[c['policies'][policy][str(old)] for c in rows]
                    n=sum(m['n'] for m in ms)
                    values.append({'n':n,'accuracy':sum(m['correct'] for m in ms)/n if n else None})
                matrix.append(values)
            result['matrices'][subset][policy]=matrix
    (output/'probe_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    main()
