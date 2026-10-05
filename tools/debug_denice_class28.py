"""Class-28 acquisition/routing/forgetting audit. Never reads final test data."""
import gc
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from eval_checkpoint import _make_denice_client_model
from fed_learning.data.denice_clean_roles import CleanRoleData,file_sha256
from fed_learning.training.denice_delta_checkpoint import load_denice_checkpoint
from fed_learning.training.denice_eval import _denice_routed_logits_with_episodes
from fed_learning.strategies.incremental.denice_tip_router import encoder_fingerprint
from tools.denice_competence_gate import content_hash
from tools.denice_peer_voting import peer_orders
from tools.eval_denice_peer_coverage import lookup
from tools.eval_denice_tip_router import write_json


def unpack_results(source, destination):
    """Extract only roles, provenance/config and the two required task archives."""
    root=Path(destination);root.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(source) as archive:
        wanted=[name for name in archive.namelist() if not name.endswith('/') and (
            name.startswith('denice_clean_roles/') or name.endswith('/clean_class_meta/meta_provenance.csv.gz')
            or name.endswith('/clean_class_meta/origin_support.json')
            or name.endswith('/config.json') or name.endswith('/checkpoint_task_4_all_rounds.zip')
            or name.endswith('/checkpoint_task_5_all_rounds.zip'))]
        if not wanted:raise ValueError('Expected results (8) archive layout')
        for name in wanted:
            target=(root/name).resolve()
            if not target.is_relative_to(root.resolve()):raise ValueError('Unsafe ZIP member')
            target.parent.mkdir(parents=True,exist_ok=True)
            import shutil
            with archive.open(name) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst,8*1024*1024)
    manifests=list(root.rglob('role_manifest.json'));final=list(root.rglob('checkpoint_task_5_all_rounds.zip'))
    if len(manifests)!=1 or len(final)!=1:raise ValueError('Ambiguous role/run folders')
    return manifests[0].parent,final[0].parent


def graph(archive,ids,task):
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('checkpoint_archive_manifest.json'))
        if not manifest.get('full_terminal_checkpoint'):raise ValueError('Exact FP32 terminal checkpoint required')
        with z.open(manifest['checkpoint']) as stream:delta=torch.load(stream,map_location='cpu',weights_only=False)
        cluster=delta['cluster']
    if int(cluster['task'])!=task or int(cluster['round'])!=19:raise ValueError('Wrong graph task/round')
    peers={};alphas={}
    for cid in ids:
        audit=lookup(cluster['alpha_debug'],cid);group=lookup(cluster['groups'],cid)
        if not audit or set(group)!=set(audit['group_ids']) or len(audit['group_ids'])!=len(audit['alphas']):
            raise ValueError('Incomplete recorded graph')
        weights={int(d):float(w) for d,w in zip(audit['group_ids'],audit['alphas'])}
        if any(not np.isfinite(w) or w<0 for w in weights.values()):raise ValueError('Invalid alpha')
        allowed=sorted({cid}|{d for d,w in weights.items() if w>0})
        if not set(allowed).issubset(ids):raise ValueError('Missing recorded expert')
        peers[cid]=allowed;alphas[cid]=weights
    return peers,alphas


def pool_data(roles,run,out):
    provenance=pd.read_csv(run/'clean_class_meta'/'meta_provenance.csv.gz',compression='gzip')
    rows=provenance[(provenance.label==28)&provenance.role.isin(['fit','validation'])].copy()
    if rows.empty:raise ValueError('Missing saved class-28 fitting/validation provenance')
    unique=rows.drop_duplicates(['role','content_hash']).sort_values(['role','content_hash'])
    evidence=json.loads((run/'clean_class_meta'/'origin_support.json').read_text())
    eligible={(int(r['donor']),r['role']) for r in evidence if 28 in r['support']}
    entries={};origin_rows=[]
    for (cid,role),group in unique.groupby(['origin','role']):
        if (int(cid),role) not in eligible:raise ValueError('Class28 origin lacks recorded local support')
        x,y,indices=roles.client_role(int(cid),role);coordinates={int(v):i for i,v in enumerate(indices)}
        for record in group.itertuples():
            if int(record.original_row) not in coordinates:raise ValueError('Provenance outside locked data role')
            pos=coordinates[int(record.original_row)]
            if int(y[pos])!=28 or content_hash(x[pos])!=record.content_hash:raise ValueError('Class-28 input provenance mismatch')
            entries[role,record.content_hash]=x[pos].copy()
            origin_rows.append(dict(role=role,content_hash=record.content_hash,origin=int(cid),original_row=int(record.original_row)))
        del x,y,indices
    data={};maps={}
    for role in ('fit','validation'):
        hashes=unique[unique.role==role].content_hash.tolist()
        if not hashes:raise ValueError(f'Missing class-28 {role} pool')
        data[role]=torch.from_numpy(np.stack([entries[role,h] for h in hashes]))
        maps[role]={h:i for i,h in enumerate(hashes)}
    if set(maps['fit'])&set(maps['validation']):raise ValueError('Fit/validation content overlap')
    for row in origin_rows:row['pool_index']=maps[row['role']][row['content_hash']]
    pd.DataFrame(origin_rows).to_csv(out/'sample_provenance.csv',index=False)
    return data,maps,rows


@torch.no_grad()
def donor_predictions(model,detector,inputs,seen,device,batch_size):
    normal=[];routes=[];oracle=[];ranks=[];margins=[]
    allowed=sorted(set(detector.episode_classes.get(4,[]))&set(seen))
    mask28=28 in allowed
    for start in range(0,len(inputs),batch_size):
        xb=inputs[start:start+batch_size].to(device)
        logits,episodes=_denice_routed_logits_with_episodes(model,xb,detector,seen,device,inference_policy='pred_hard')
        normal.extend(logits.argmax(1).cpu().tolist());routes.extend(np.asarray(episodes).tolist())
        if mask28:
            forced,_=_denice_routed_logits_with_episodes(model,xb,detector,seen,device,
                inference_policy='oracle_hard',oracle_episodes=np.full(len(xb),4,dtype=np.int64))
            values=forced[:,allowed].float();target=forced[:,28].float()
            other=[c for c in allowed if c!=28]
            best_other=forced[:,other].max(1).values if other else target
            oracle.extend(forced.argmax(1).cpu().tolist())
            ranks.extend((1+(values>target[:,None]).sum(1)).cpu().tolist())
            margins.extend((target-best_other).cpu().tolist())
        else:
            oracle.extend([-1]*len(xb));ranks.extend([-1]*len(xb));margins.extend([0.]*len(xb))
    return dict(normal=np.asarray(normal),route=np.asarray(routes),oracle=np.asarray(oracle),
                rank=np.asarray(ranks),margin=np.asarray(margins)),mask28


def run_debug(run_dir,role_dir,data_dir,out_dir,device='cuda',batch_size=512):
    if batch_size<1:raise ValueError('Batch size must be positive')
    run=Path(run_dir);out=Path(out_dir);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'completion.json',dict(completed=False,final_test_read=False))
    roles=CleanRoleData(role_dir,source_data_dir=data_dir)
    data,maps,occurrences=pool_data(roles,run,out)
    results={};donor_rows=[];receiver_rows=[];arrays={}
    with threadpool_limits(limits=1):
        for task in (4,5):
            archive=run/f'checkpoint_task_{task}_all_rounds.zip'
            ckpt=load_denice_checkpoint(archive)
            if ckpt['config'].get('denice_data_roles_sha256')!=file_sha256(roles.root/'role_manifest.json'):
                raise ValueError('Checkpoint and clean role lock differ')
            ids=sorted(int(cid) for cid in ckpt['client_model_states']);peers,alphas=graph(archive,ids,task)
            seen=sorted(map(int,ckpt['seen_classes']));predictions={role:{} for role in data}
            for donor in ids:
                model,detector=_make_denice_client_model(ckpt,donor,device)
                fingerprint=encoder_fingerprint(model);detector.router_mode='multiclass_balanced'
                if not detector.activation_memory:raise ValueError(f'{task}/{donor}: missing router memory')
                detector.train_models(max(detector.activation_memory))
                for role,inputs in data.items():
                    values,mask28=donor_predictions(model,detector,inputs,seen,device,batch_size)
                    predictions[role][donor]=values
                    donor_rows.append(dict(checkpoint_task=task,donor=donor,role=role,samples=len(inputs),
                        task4_mask_has_class28=mask28,task4_profile_present=4 in detector.activation_memory,
                        normal_correct=int((values['normal']==28).sum()),route_task4=int((values['route']==4).sum()),
                        oracle_correct=int((values['oracle']==28).sum()),
                        mean_oracle_class28_rank=float(values['rank'].mean()) if mask28 else None,
                        mean_oracle_margin=float(values['margin'].mean()) if mask28 else None))
                if encoder_fingerprint(model)!=fingerprint:raise RuntimeError('Expert weights/masks mutated')
                del model,detector
                print(f'Class28 debug: task={task}, donor={donor}',flush=True)
            for role in data:
                for name in ('normal','route','oracle','rank','margin'):
                    arrays[f'task{task}_{role}_{name}']=np.stack([predictions[role][d][name] for d in ids])
                arrays[f'task{task}_{role}_donor_ids']=np.asarray(ids)
                for cid,group in occurrences[occurrences.role==role].groupby('receiver'):
                    cid=int(cid)
                    if cid not in peers:continue  # Late joiners are absent at task 4; do not invent a model.
                    indices=np.asarray([maps[role][h] for h in group.content_hash])
                    fixed=[cid]+peer_orders(cid,peers[cid],alphas[cid],(42,))['random_42'][:16]
                    for budget,donors in (('self_plus_16',fixed),('all_positive_alpha_peers',peers[cid])):
                        normal=np.stack([predictions[role][d]['normal'][indices] for d in donors])
                        oracle=np.stack([predictions[role][d]['oracle'][indices] for d in donors])
                        receiver_rows.append(dict(checkpoint_task=task,role=role,receiver=cid,budget=budget,
                            model_count=len(donors),samples=len(indices),
                            normal_any_correct=int((normal==28).any(0).sum()),
                            oracle_task4_any_correct=int((oracle==28).any(0).sum())))
            results[task]=dict(training_xi=ckpt['config'].get('denice_similarity_threshold'),client_ids=ids,
                              exact_terminal=True,router='multiclass_balanced',checkpoint_archive_sha256=file_sha256(archive))
            del ckpt,predictions;gc.collect()
            if device.startswith('cuda'):torch.cuda.empty_cache()
    pd.DataFrame(donor_rows).to_csv(out/'donor_class28.csv',index=False)
    receivers=pd.DataFrame(receiver_rows);receivers.to_csv(out/'receiver_coverage.csv',index=False)
    np.savez_compressed(out/'expert_predictions.npz',**arrays)
    summary=[]
    for keys,group in receivers.groupby(['checkpoint_task','role','budget']):
        n=int(group.samples.sum())
        summary.append(dict(checkpoint_task=int(keys[0]),role=keys[1],budget=keys[2],receiver_count=len(group),
            samples=n,normal_any_correct=int(group.normal_any_correct.sum()),
            normal_any_correct_rate=float(group.normal_any_correct.sum()/n),
            oracle_task4_any_correct=int(group.oracle_task4_any_correct.sum()),
            oracle_task4_any_correct_rate=float(group.oracle_task4_any_correct.sum()/n)))
    # Paired common donors and identical unique role contents separate acquisition from later loss.
    common=sorted(set(results[4]['client_ids'])&set(results[5]['client_ids']));paired=[]
    for role in data:
        for donor in common:
            i=list(arrays[f'task4_{role}_donor_ids']).index(donor);j=list(arrays[f'task5_{role}_donor_ids']).index(donor)
            a=arrays[f'task4_{role}_oracle'][i]==28;b=arrays[f'task5_{role}_oracle'][j]==28
            paired.append(dict(donor=donor,role=role,samples=len(a),task4_correct=int(a.sum()),task5_correct=int(b.sum()),
                               lost=int((a&~b).sum()),gained=int((~a&b).sum())))
    pd.DataFrame(paired).to_csv(out/'paired_task4_task5.csv',index=False)
    completion=dict(completed=True,final_test_read=False,backbone_retrained=False,target_class=28,
        diagnostic_only=True,pools={role:len(x) for role,x in data.items()},checkpoints=results,coverage=summary,
        comparison='Oracle task4 is label-assisted; masks/adapters remain donor-local. Paired donor comparison uses common donors.',
        limitation='Only saved class28 META-fit/validation contents; no final accuracy or universal knowledge ceiling.')
    write_json(out/'completion.json',completion);print(json.dumps(summary,indent=2),flush=True)
    return completion
