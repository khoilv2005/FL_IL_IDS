"""Locked donor-only raw-input expert feasibility and native-graph reachability audit."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random
import sys
import time
import zipfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'candle_v4_runtime'))
import numpy as np
import torch
from sklearn.metrics import average_precision_score,roc_auc_score
from fed_learning.data.denice_clean_roles import CleanRoleData,file_sha256


def write(path,value):
    Path(path).write_bytes((json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8'))


def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True


def graph_snapshot(checkpoint,task):
    name=f'checkpoint_task_{task}_round_19.pt'
    with zipfile.ZipFile(checkpoint) as archive:
        manifest=json.loads(archive.read('checkpoint_archive_manifest.json'))
        if int(manifest['task_id'])!=task: raise ValueError('Wrong task archive')
        raw=archive.read(name)
        checksum=hashlib.sha256(raw).hexdigest()
        if checksum!=manifest['checksums'][name]: raise ValueError('Graph checkpoint checksum mismatch')
        packet=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    record=packet['cluster']; groups={}; weights={}
    if record.get('similarity_threshold')!=.8: raise ValueError('Expected native xi=0.8')
    for key,row in record['alpha_debug'].items():
        owner=int(key); ids=list(map(int,row['group_ids'])); values=list(map(float,row['alphas']))
        membership=record['groups'].get(owner,record['groups'].get(str(owner)))
        if (len(ids)!=len(values) or len(set(ids))!=len(ids) or owner not in ids
                or set(ids)!=set(map(int,membership)) or any(not math.isfinite(v) or v<0 for v in values)):
            raise ValueError('Malformed native graph')
        groups[owner]=[j for j,v in zip(ids,values) if j!=owner and v>0]
        weights[owner]={j:v for j,v in zip(ids,values) if j!=owner and v>0}
    return groups,weights,checksum


def reachability(manifest,classes,groups):
    rows=[]
    global_classes={c for c in classes if any(manifest['clients'][str(j)]['role_class_counts']['base'].get(str(c),0)
                    for j in groups)}
    for owner,peers in sorted(groups.items()):
        own=manifest['clients'][str(owner)]['role_class_counts']
        for c in sorted(global_classes):
            if own['base'].get(str(c),0): continue
            sources=[j for j in peers if manifest['clients'][str(j)]['role_class_counts']['base'].get(str(c),0)]
            receiver_negative=sum(n for label,n in own['calibration'].items() if int(label) in classes and int(label)!=c)
            evidence=[j for j in sources if manifest['clients'][str(j)]['role_class_counts']['fit'].get(str(c),0)>=32
                      and manifest['clients'][str(j)]['role_class_counts']['calibration'].get(str(c),0)>=32
                      and receiver_negative>=32]
            rows.append(dict(receiver=owner,class_id=c,donors=sources,evidence_count_eligible_donors=evidence,
                             receiver_negative_cal_count=receiver_negative,functional_gap='UNMEASURED'))
    count=len(rows); reachable=sum(bool(r['donors']) for r in rows); feasible=sum(bool(r['evidence_count_eligible_donors']) for r in rows)
    return dict(active_receivers=len(groups),current_classes=classes,metadata_gaps=count,
        reachable_gaps=reachable,reachable_coverage=reachable/count if count else None,
        evidence_count_eligible_gaps=feasible,evidence_count_eligible_coverage=feasible/count if count else None,
        zero_peer_receivers=sum(not peers for peers in groups.values()),rows=rows,
        scope='One native Task2/R19 graph snapshot; data-ownership gaps, not confirmed functional gaps or deployment coverage')


def stage_current(roles,owner,role,classes,out,cap,task,log):
    """Preparation only: verify original shard, stage current role; no CAL/TEST access."""
    if role not in ('base','fit'): raise ValueError('Only BASE/FIT authorized')
    source=roles.source/f'client_{owner}_train.npz'; index=roles.root/'indices'/f'client_{owner}.npz'
    entry=roles.manifest['clients'][str(owner)]
    if file_sha256(source)!=entry['source_sha256'] or file_sha256(index)!=entry['index_sha256']:
        raise ValueError('Source shard or role indices changed')
    with np.load(index,allow_pickle=False) as split: selected=split[role]
    with np.load(source,allow_pickle=False) as archive:
        labels=archive['y_train']
        if len(labels)!=entry['source_rows']: raise ValueError('Source coordinates changed')
        selected=selected[np.isin(labels[selected],classes)]
        # Original NPZ decoding is confined to preparation; trainer later opens staged current files only.
        x=np.asarray(archive['X_train'][selected],np.float32).reshape(len(selected),39)
        y=np.asarray(labels[selected],np.int64)
    if not np.isfinite(x).all(): raise ValueError('Nonfinite input')
    order=np.random.default_rng(42+owner*1009+task*100003).permutation(len(y))
    seen=set(); counts=Counter(); kept=[]; hashes=[]
    for row in order:
        key=hashlib.sha256(x[row].tobytes()).hexdigest()
        if key in seen or (cap is not None and counts[int(y[row])]>=cap): continue
        seen.add(key); counts[int(y[row])]+=1; kept.append(int(row)); hashes.append(key)
    path=out/f'client_{owner}_task_{task}_{role}.npz'
    np.savez_compressed(path,X=x[kept],y=y[kept],content_sha256=np.asarray(hashes),rows=selected[kept])
    log.append(dict(owner=owner,role=role,task=task,current_classes=classes,raw_current_rows=len(y),
        staged_unique_rows=len(kept),class_counts=dict(counts),file=path.name,sha256=file_sha256(path),
        source_sha256=entry['source_sha256'],index_sha256=entry['index_sha256']))
    return path


def read_stage(path,classes):
    with np.load(path,allow_pickle=False) as data:
        x=torch.from_numpy(data['X'].copy()); y=torch.from_numpy(data['y'].copy()); hashes=set(data['content_sha256'].tolist())
    if x.shape[1:]!=(39,) or not set(y.tolist())<=set(classes): raise ValueError('Staged input contract violated')
    return x,y,hashes


@torch.no_grad()
def scores(model,x,device):
    return torch.cat([torch.sigmoid(model(x[start:start+2048].to(device))).cpu().flatten()
                      for start in range(0,len(x),2048)]).numpy()


def metrics(labels,values,target,threshold):
    positive=labels==target; negative=~positive
    activated=values>=threshold
    by_class={str(c):dict(n=int((labels==c).sum()),activated=int(((labels==c)&activated).sum())) for c in sorted(set(labels.tolist()))}
    return dict(n=len(labels),positive=int(positive.sum()),negative=int(negative.sum()),threshold=float(threshold),
        true_positive=int((positive&activated).sum()),false_positive=int((negative&activated).sum()),
        recall=float(activated[positive].mean()) if positive.any() else None,
        false_activation_rate=float(activated[negative].mean()) if negative.any() else None,
        positive_prevalence=float(positive.mean()),
        average_precision_PR_AUC=float(average_precision_score(positive,values)) if positive.any() and negative.any() else None,
        roc_auc=float(roc_auc_score(positive,values)) if positive.any() and negative.any() else None,
        by_class=by_class)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--roles',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2); seed(42); started=time.monotonic()
    task,receiver,target=2,30,13
    roles=CleanRoleData(args.roles,source_data_dir=args.data_dir)
    metadata=json.loads((roles.source/'metadata.json').read_text()); classes=metadata['task_structure']['task_classes'][str(task)]
    groups,weights,graph_sha=graph_snapshot(args.checkpoint,task)
    coverage=reachability(roles.manifest,classes,groups); write(args.out/'reachable_coverage.json',coverage)
    candidates=next(row for row in coverage['rows'] if row['receiver']==receiver and row['class_id']==target)['evidence_count_eligible_donors']
    if not candidates: raise ValueError('Fixed receiver/target has no graph/evidence eligible donor; no automatic fallback')
    donor=min(candidates)
    lock=dict(schema='candle_v4_standalone_v1',task=task,round_graph=19,receiver=receiver,donor=donor,target=target,
        case_reason='Previously observed development head/router difficulty for receiver30/class13; smallest-ID eligible actual neighbor, no TEST selection',
        graph_sha256=graph_sha,graph_alpha=weights[receiver][donor],role_manifest_sha256=file_sha256(roles.root/'role_manifest.json'),
        metadata_sha256=roles.manifest['metadata_sha256'],architecture=[39,64,1],parameter_count=2625,
        optimizer='Adam',learning_rate=.001,epochs=20,batch_size=256,seed=42,gradient_clip_norm=5.,
        base_unique_cap_per_class=4096,loss='BCEWithLogits; equal total weight for donor positives and donor negatives',
        weight_decay=0.,extra_normalization=False,early_stopping=False,final_epoch_selection=20,
        fit_observation_epochs=[0,1,5,10,20],fixed_diagnostic_threshold=.5,
        far_at_recall95_threshold='derived from donor FIT only, diagnostic, never a deployment gate',
        calibration_counts_only=True,CAL_opened=False,TEST_opened=False,expert_activation=False,
        preprocessing_authority='Verified original dataset shards and matching metadata/role lock; no separate scaler-file fingerprint recovered',
        runtime_scope='staged current T2 BASE/FIT only; original compressed shards opened only in explicit preparation')
    write(args.out/'protocol_lock.json',lock)
    protocol_hash=file_sha256(args.out/'protocol_lock.json')
    print('LOCKED',json.dumps(lock),flush=True)
    staged=args.out/'current_data'; staged.mkdir(); access=[]
    train_path=stage_current(roles,donor,'base',classes,staged,4096,task,access)
    fit_path=stage_current(roles,donor,'fit',classes,staged,None,task,access)
    receiver_path=stage_current(roles,receiver,'fit',classes,staged,None,task,access)
    write(args.out/'preparation_access.json',access)
    x,y,train_hashes=read_stage(train_path,classes)
    fx,fy,fit_hashes=read_stage(fit_path,classes)
    rx,ry,receiver_hashes=read_stage(receiver_path,classes)
    if train_hashes & (fit_hashes|receiver_hashes): raise ValueError('Training/evaluation content overlap')
    if (ry==target).any(): raise ValueError('Receiver unexpectedly owns target positives')
    positive=y==target; negative=~positive
    if int(positive.sum())<32 or int(negative.sum())<32: raise ValueError('Insufficient unique donor training groups')
    sample_weight=torch.where(positive,len(y)/(2*int(positive.sum())),len(y)/(2*int(negative.sum())))
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=torch.nn.Sequential(torch.nn.Linear(39,64),torch.nn.ReLU(),torch.nn.Linear(64,1)).to(device)
    if sum(p.numel() for p in model.parameters())!=2625: raise AssertionError('Parameter budget mismatch')
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.)
    steps=0; history=[]
    def evaluate(epoch,train_loss):
        model.eval(); donor_scores=scores(model,fx,device); receiver_scores=scores(model,rx,device)
        row=dict(epoch=epoch,optimizer_steps=steps,training_loss=train_loss,
            donor_fit=metrics(fy.numpy(),donor_scores,target,.5),receiver_negative_fit=metrics(ry.numpy(),receiver_scores,target,.5))
        history.append(row); print('EPOCH',epoch,'donor FIT',row['donor_fit'],'receiver FAR',row['receiver_negative_fit']['false_activation_rate'],flush=True)
        return donor_scores,receiver_scores
    evaluate(0,None)
    for epoch in range(1,21):
        model.train(); order=torch.randperm(len(y)); loss_sum=0.
        for start in range(0,len(order),256):
            rows=order[start:start+256]; bx=x[rows].to(device); by=positive[rows].float().to(device)
            optimizer.zero_grad(set_to_none=True)
            logits=model(bx).flatten()
            loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,by,reduction='none')*sample_weight[rows].to(device)).mean()
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite expert loss')
            loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            if not torch.isfinite(norm): raise FloatingPointError('Nonfinite expert gradients')
            optimizer.step(); steps+=1; loss_sum+=float(loss.detach())*len(rows)
        if epoch in (1,5,10,20): final_donor,final_receiver=evaluate(epoch,loss_sum/len(y))
    positives=final_donor[fy.numpy()==target]
    kth=math.ceil(.95*len(positives))-1
    threshold=float(np.sort(positives)[::-1][kth])
    donor95=metrics(fy.numpy(),final_donor,target,threshold)
    receiver95=metrics(ry.numpy(),final_receiver,target,threshold)
    torch.save(dict(model=model.cpu().state_dict(),optimizer=optimizer.state_dict(),epoch=20,optimizer_steps=steps,
                    protocol_sha256=protocol_hash),args.out/'standalone_expert.pt')
    if file_sha256(args.out/'protocol_lock.json')!=protocol_hash: raise AssertionError('Protocol lock changed')
    report=dict(completed=True,protocol=lock,protocol_sha256=protocol_hash,
        train_unique_counts=dict(Counter(y.tolist())),donor_fit_unique_counts=dict(Counter(fy.tolist())),
        receiver_fit_unique_counts=dict(Counter(ry.tolist())),train_evaluation_disjoint=True,
        history=history,final_fixed_threshold=history[-1],
        donor_fit_at_recall95_DIAGNOSTIC_ONLY=donor95,receiver_fit_at_same_threshold_DIAGNOSTIC_ONLY=receiver95,
        runtime_original_shards_opened=False,CAL_opened=False,TEST_opened=False,expert_activated=False,
        federated_training=False,roundwise_training=False,old_task_breaks_measured=False,
        runtime_device=device,torch=str(torch.__version__),optimizer_steps=steps,
        model_parameter_bytes=2625*4,peer_communication_bytes=0,communication_scope='Standalone local donor training only',
        reachable_coverage_summary={k:v for k,v in coverage.items() if k!='rows'},elapsed_seconds=time.monotonic()-started,
        limitation='Development capacity test, not deployment acceptance; donor target recall does not establish target recall in receiver domain. Receiver negatives never used for training.')
    write(args.out/'report.json',report)
    with (args.out/'learning_curve.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['epoch','steps','train_loss','donor_recall','donor_FAR','donor_AP','receiver_negative_FAR'])
        writer.writeheader()
        for r in history: writer.writerow(dict(epoch=r['epoch'],steps=r['optimizer_steps'],train_loss=r['training_loss'],
            donor_recall=r['donor_fit']['recall'],donor_FAR=r['donor_fit']['false_activation_rate'],
            donor_AP=r['donor_fit']['average_precision_PR_AUC'],receiver_negative_FAR=r['receiver_negative_fit']['false_activation_rate']))
    print('COMPLETED',args.out/'report.json',flush=True)


if __name__=='__main__': main()
