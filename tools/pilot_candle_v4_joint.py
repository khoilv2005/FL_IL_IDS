"""Matched-work positive/negative expert gossip replay; no CAL, ACTIVE gate or native training."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import pickle
import sys
import time
import zipfile

from pilot_candle_v4_standalone import seed,stage_current,metrics,write,file_sha256,CleanRoleData
import numpy as np
import torch
from fed_learning.training.decentralized_denice_il import _make_model
from fed_learning.training.checkpoint_state import restore_denice_state,snapshot_denice_state
from fed_learning.servers.nice_server import ContextDetector
from fed_learning.training.denice_eval import _denice_routed_logits_with_episodes


TARGET=13; RECEIVER=30; DONOR=16; TASK=2


def vector(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu().float().numpy().tobytes()


def load_vector(model,raw):
    values=torch.from_numpy(np.frombuffer(raw,dtype=np.float32).copy()).to(next(model.parameters()).device)
    if len(values)!=2625 or not torch.isfinite(values).all(): raise ValueError('Invalid expert tensor')
    position=0
    with torch.no_grad():
        for p in model.parameters():
            count=p.numel(); p.copy_(values[position:position+count].view_as(p)); position+=count


class Wire:
    def __init__(self,path):
        self.path=path; self.total=0; self.kinds=Counter(); self.events=0
    def send(self,sender,receiver,kind,message):
        if sender==receiver: raise ValueError('Local computation is not communication')
        raw=pickle.dumps(message,protocol=5); size=len(raw)
        if self.total+size>128*1024**2: raise RuntimeError('Locked 128 MiB trial budget exhausted')
        self.total+=size; self.kinds[kind]+=size; self.events+=1
        with self.path.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(sender=sender,receiver=receiver,kind=kind,bytes=size))+'\n')
        return pickle.loads(raw)
    def summary(self):
        return dict(application_wire_bytes=self.total,by_kind=dict(self.kinds),messages=self.events,
                    native_bytes=None,total_system_bytes=None,mode='serialized in-process peer simulation, not measured network traffic')


def staged(path):
    with np.load(path,allow_pickle=False) as data:
        x=torch.from_numpy(data['X'].copy()); y=torch.from_numpy(data['y'].copy()); hashes=data['content_sha256'].tolist()
    if x.shape[1:]!=(39,) or not set(y.tolist())<=set(range(12,18)): raise ValueError('Wrong current input scope')
    return dict(x=x,y=y,hashes=hashes)


def flat_gradient(loss,params,retain=False):
    if not loss.requires_grad: return torch.zeros(2625,device=params[0].device)
    return torch.cat([g.reshape(-1) for g in torch.autograd.grad(loss,params,retain_graph=retain)])


class Peer:
    def __init__(self,owner,data,model):
        self.owner=owner; self.x=data['x']; self.y=data['y']; self.model=model
        self.indices={int(c):torch.nonzero(self.y==c).flatten() for c in torch.unique(self.y)}
    def batch(self,request,weights,step):
        rows=[]; weight=[]
        for c,n in sorted(request.items()):
            generator=torch.Generator().manual_seed(42+step*100003+(0 if c==TARGET else self.owner*1009)+c*17)
            local=self.indices[c]
            chosen=local[torch.randint(len(local),(n,),generator=generator)]
            rows.append(chosen); weight.extend([weights[c]]*n)
        selected=torch.cat(rows)
        self.last_batch_sha256=hashlib.sha256(selected.numpy().tobytes()).hexdigest()
        self.last_positive_sha256=hashlib.sha256(selected[self.y[selected]==TARGET].numpy().tobytes()).hexdigest()
        return self.x[selected],(self.y[selected]==TARGET).float(),torch.as_tensor(weight),self.y[selected]
    def gradient(self,request,weights,step):
        x,y,w,labels=self.batch(request,weights,step); device=next(self.model.parameters()).device
        x,y,w=x.to(device),y.to(device),w.to(device)
        losses=torch.nn.functional.binary_cross_entropy_with_logits(self.model(x).flatten(),y,reduction='none')*w
        positive=(y==1); pos=losses[positive].sum(); neg=losses[~positive].sum(); params=list(self.model.parameters())
        gp=flat_gradient(pos,params,True); gn=flat_gradient(neg,params,False)
        gradient=gp+gn
        if not torch.isfinite(gradient).all(): raise FloatingPointError('Nonfinite participant gradient')
        return gradient.detach().cpu(),dict(owner=self.owner,positive=int(positive.sum()),negative=int((~positive).sum()),
            positive_gradient_norm=float(gp.norm()),negative_gradient_norm=float(gn.norm()),
            loss=float(losses.detach().sum()),classes=dict(Counter(labels.tolist())),
            BASE_sample_indices_sha256=self.last_batch_sha256,positive_sample_indices_sha256=self.last_positive_sha256)


def requests(counts,step,balanced):
    negative=[(owner,c,n) for owner,classes in sorted(counts.items()) for c,n in sorted(classes.items()) if c!=TARGET]
    total=sum(n for _,_,n in negative); class_counts=Counter()
    for _,c,n in negative: class_counts[c]+=n
    generator=torch.Generator().manual_seed(42+step*100003+202)
    tags=torch.randint(total,(128,),generator=generator)
    allocation={o:Counter() for o in counts}; allocation[DONOR][TARGET]=128
    offset=0
    for owner,c,n in negative:
        allocation[owner][c]=int(((tags>=offset)&(tags<offset+n)).sum()); offset+=n
    allocation={o:{c:n for c,n in row.items() if n} for o,row in allocation.items()}
    weights={TARGET:.5/128}
    weights.update({c:(.5/128*total/(len(class_counts)*n) if balanced else .5/128) for c,n in class_counts.items()})
    return allocation,weights


@torch.no_grad()
def expert_scores(model,x):
    device=next(model.parameters()).device
    return torch.cat([torch.sigmoid(model(x[s:s+2048].to(device))).cpu().flatten() for s in range(0,len(x),2048)]).numpy()


@torch.no_grad()
def base_predict(model,detector,x):
    device=str(next(model.parameters()).device); values=[]
    for start in range(0,len(x),512):
        out,_=_denice_routed_logits_with_episodes(model,x[start:start+512].to(device),detector,list(range(18)),device,
                                                  inference_policy='pred_hard')
        values.extend(out.argmax(1).cpu().tolist())
    return np.asarray(values)


def fusion(labels,scores,baseline,threshold):
    predicted=np.where(scores>=threshold,TARGET,baseline)
    before=baseline==labels; after=predicted==labels; negative=labels!=TARGET
    result=metrics(labels,scores,TARGET,threshold)
    margin=scores[labels==TARGET]-threshold
    result['positive_score_margin']=(dict(mean=float(margin.mean()),median=float(np.median(margin)),
        minimum=float(margin.min()),maximum=float(margin.max())) if len(margin) else None)
    result.update(baseline_correct=int(before.sum()),candidate_correct=int(after.sum()),
        rescue=int((~before&after).sum()),break_count=int((before&~after).sum()),
        negative_break=int((negative&before&~after).sum()),net_gain=int(after.sum()-before.sum()),
        changed_wrong=int((~before&~after&(predicted!=baseline)).sum()))
    for c,row in result['by_class'].items():
        mask=labels==int(c)
        row.update(baseline_correct=int((before&mask).sum()),candidate_correct=int((after&mask).sum()),
                   break_count=int((before&~after&mask).sum()),
                   changed_wrong=int((~before&~after&mask&(predicted!=baseline)).sum()))
    assert result['net_gain']==result['rescue']-result['break_count']
    return result


def thresholds(values,labels):
    positive=values[labels==TARGET]
    return float(np.sort(positive)[::-1][int(np.ceil(.95*len(positive)))-1])


def evaluate(model,fit,base,pool_keep,wire,branch,round_id,model_hash):
    model.eval(); ds=expert_scores(model,fit[DONOR]['x']); rs=expert_scores(model,fit[RECEIVER]['x'])
    threshold=thresholds(ds,fit[DONOR]['y'].numpy())
    params=vector(model)
    if branch=='donor_only':
        received=wire.send(DONOR,RECEIVER,'expert_export_for_FIT',dict(parameters=params,expert_sha256=model_hash,round=round_id))
    else:
        received=wire.send(RECEIVER,DONOR,'expert_reference_for_FIT',dict(parameters=params,expert_sha256=model_hash,round=round_id))
    assert hashlib.sha256(received['parameters']).hexdigest()==model_hash
    result={}
    for name,tau in (('fixed_0p5',.5),('donor_FIT_recall95',threshold)):
        donor=fusion(fit[DONOR]['y'].numpy(),ds,base[DONOR],tau)
        receiver=fusion(fit[RECEIVER]['y'].numpy(),rs,base[RECEIVER],tau)
        y=np.concatenate([fit[DONOR]['y'].numpy(),fit[RECEIVER]['y'].numpy()[pool_keep]])
        s=np.concatenate([ds,rs[pool_keep]]); b=np.concatenate([base[DONOR],base[RECEIVER][pool_keep]])
        pooled=fusion(y,s,b,tau)
        result[name]=dict(donor=donor,receiver=receiver,pooled_unique=pooled)
    wire.send(DONOR,RECEIVER,'FIT_metric_receipt',dict(round=round_id,expert_sha256=model_hash,
        donor={k:v['donor'] for k,v in result.items()},threshold95=threshold))
    return result


def train_branch(name,train,fit,base,pool_keep,base_payload,availability,out,device):
    directory=out/name; directory.mkdir(); wire=Wire(directory/'wire.jsonl')
    seed(42)
    model=torch.nn.Sequential(torch.nn.Linear(39,64),torch.nn.ReLU(),torch.nn.Linear(64,1)).to(device)
    initial=hashlib.sha256(vector(model)).hexdigest()
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.)
    owners=[DONOR] if name=='donor_only' else [DONOR,RECEIVER]
    counts={o:dict(Counter(train[o]['y'].tolist())) for o in owners}
    peers={o:Peer(o,train[o],model if o==(DONOR if name=='donor_only' else RECEIVER) else deepcopy(model)) for o in owners}
    balanced=name=='joint_class_balanced'; pending=0; step=0; history=[]; last=None; setup=False
    checks=dict(same_reference=True,finite_gradients=True,receiver_positive_samples=0,
                pooled_gradient_max_abs_difference=None,positive_steps=0,receiver_negative_steps=0)
    sample_trace=hashlib.sha256(); positive_trace=hashlib.sha256()
    contributions={o:Counter() for o in owners}
    for round_id in range(20):
        before_bytes=wire.total; pending+=44; updates=0
        before_vector=np.frombuffer(vector(model),np.float32).copy()
        round_loss=0.; round_positive_norm=0.; round_receiver_negative_norm=0.
        if availability[round_id]:
            if not setup:
                # Counts are the only training-data metadata transmitted.
                if name!='donor_only': wire.send(DONOR,RECEIVER,'BASE_class_counts',dict(task=TASK,counts=counts[DONOR]))
                wire.send(RECEIVER,DONOR,'verification_base_reference',base_payload)
                wire.send(DONOR,RECEIVER,'FIT_content_ids',fit[DONOR]['hashes'])
                setup=True
            updates=min(pending,88)
            for inner in range(updates):
                allocation,weights=requests(counts,step,balanced)
                raw=vector(model); reference=hashlib.sha256(raw).hexdigest(); global_gradient=torch.zeros(2625)
                gradients=[]; step_details=[]
                for owner in owners:
                    request=allocation[owner]
                    if not request: continue
                    if name!='donor_only' and owner==DONOR:
                        packet=wire.send(RECEIVER,DONOR,'expert_reference',dict(task=TASK,target=TARGET,round=round_id,
                            step=step,parameters=raw,expert_sha256=reference,request=request,loss_weights=weights,
                            input_fingerprint=base_payload['input_fingerprint'],mask='all young expert parameters'))
                        if (packet['step']!=step or packet['round']!=round_id or packet['task']!=TASK or packet['target']!=TARGET
                                or packet['input_fingerprint']!=base_payload['input_fingerprint']
                                or hashlib.sha256(packet['parameters']).hexdigest()!=reference):
                            raise ValueError('Stale expert reference')
                        load_vector(peers[owner].model,packet['parameters'])
                    if hashlib.sha256(vector(peers[owner].model)).hexdigest()!=reference:
                        raise ValueError('Participant not on common expert reference')
                    gradient,detail=peers[owner].gradient(request,weights,step)
                    if name!='donor_only' and owner==DONOR:
                        packet=wire.send(DONOR,RECEIVER,'expert_gradient',dict(task=TASK,target=TARGET,round=round_id,
                            step=step,expert_sha256=reference,
                            gradient=gradient.numpy().tobytes(),statistics=detail))
                        if (packet['step']!=step or packet['round']!=round_id or packet['task']!=TASK
                                or packet['target']!=TARGET or packet['expert_sha256']!=reference): raise ValueError('Stale gradient')
                        gradient=torch.from_numpy(np.frombuffer(packet['gradient'],np.float32).copy())
                    if owner==RECEIVER:
                        checks['receiver_positive_samples']+=detail['positive']
                        checks['receiver_negative_steps']+=int(detail['negative_gradient_norm']>0)
                    checks['positive_steps']+=int(detail['positive_gradient_norm']>0)
                    round_loss+=detail['loss']; round_positive_norm+=detail['positive_gradient_norm']
                    if owner==RECEIVER: round_receiver_negative_norm+=detail['negative_gradient_norm']
                    sample_trace.update(pickle.dumps((owner,detail['BASE_sample_indices_sha256']),protocol=5))
                    if owner==DONOR: positive_trace.update(detail['positive_sample_indices_sha256'].encode())
                    contributions[owner].update(detail['classes']); global_gradient+=gradient
                    gradients.append(gradient); step_details.append(detail)
                if step==0:
                    # Offline algebra audit only; pooled features are never a transmitted runtime packet.
                    bx=[]; by=[]; bw=[]
                    for owner in owners:
                        if not allocation[owner]: continue
                        px,py,pw,_=peers[owner].batch(allocation[owner],weights,step)
                        bx.append(px); by.append(py); bw.append(pw)
                    total_loss=(torch.nn.functional.binary_cross_entropy_with_logits(
                        model(torch.cat(bx).to(device)).flatten(),torch.cat(by).to(device),reduction='none')*
                        torch.cat(bw).to(device)).sum()
                    pooled=flat_gradient(total_loss,list(model.parameters())).detach().cpu()
                    difference=float((pooled-global_gradient).abs().max()); checks['pooled_gradient_max_abs_difference']=difference
                    torch.testing.assert_close(pooled,global_gradient,rtol=1e-4,atol=2e-6)
                if not torch.isfinite(global_gradient).all(): raise FloatingPointError('Nonfinite aggregate')
                optimizer.zero_grad(set_to_none=True); position=0
                for parameter in model.parameters():
                    count=parameter.numel(); parameter.grad=global_gradient[position:position+count].to(device).view_as(parameter).clone()
                    position+=count
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.); optimizer.step(); step+=1
            pending-=updates
            current=hashlib.sha256(vector(model)).hexdigest()
            last=evaluate(model,fit,base,pool_keep,wire,name,round_id,current)
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,round=round_id,
                            pending=pending,initial_sha256=initial),directory/f'round_{round_id}.pt')
        elif last is not None:
            # No parameter/base/FIT changes: reuse an exact prior-version receipt, no forbidden edge query.
            current=hashlib.sha256(vector(model)).hexdigest()
            if current!=history[-1]['expert_sha256']: raise AssertionError('Uncommunicated expert changed')
        else: current=hashlib.sha256(vector(model)).hexdigest()
        row=dict(round=round_id,edge_available=availability[round_id],updates=updates,optimizer_steps=step,pending_quota=pending,
                 wire_bytes_this_round=wire.total-before_bytes,expert_sha256=current,FIT_receipts_reused=not availability[round_id] and last is not None,
                 mean_sampled_weighted_BCE=round_loss/updates if updates else None,
                 mean_positive_gradient_norm=round_positive_norm/updates if updates else None,
                 mean_receiver_negative_gradient_norm=round_receiver_negative_norm/updates if updates else None,
                 expert_delta_l2=float(np.linalg.norm(np.frombuffer(vector(model),np.float32)-before_vector)),
                 metrics=deepcopy(last))
        history.append(row)
        if last:
            fit95=last['donor_FIT_recall95']
            print(name,'ROUND',round_id,'steps',step,'recall',fit95['donor']['recall'],
                'receiver class12 FP',fit95['receiver']['by_class']['12']['activated'],flush=True)
    if step!=880 or pending: raise AssertionError('Matched optimizer budget not achieved')
    if checks['receiver_positive_samples']: raise AssertionError('Receiver positive leakage')
    result=dict(name=name,initial_sha256=initial,optimizer_steps=step,examples_processed=sum(sum(v.values()) for v in contributions.values()),
                BASE_sample_trace_sha256=sample_trace.hexdigest(),positive_sample_trace_sha256=positive_trace.hexdigest(),
                training_class_counts=counts,contribution_samples={o:dict(v) for o,v in contributions.items()},checks=checks,
                history=history,final=history[-1],wire=wire.summary(),active=False)
    write(directory/'report.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--roles',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--standalone',type=Path,required=True)
    parser.add_argument('--coverage',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2); started=time.monotonic(); coverage=json.loads(args.coverage.read_text())
    availability=[r['pair_30_from_16_available'] for r in coverage['rounds']]
    protocol=dict(schema='candle_v4_joint_pilot_v1',task=TASK,receiver=RECEIVER,donor=DONOR,target=TARGET,
        role_manifest_sha256=file_sha256(args.roles/'role_manifest.json'),
        pilot_script_sha256_lf=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        standalone_model_sha256=file_sha256(args.standalone/'standalone_expert.pt'),
        branches=['donor_only','joint_class_balanced','joint_unbalanced'],architecture=[39,64,1],seed=42,
        optimizer='Adam at reference owner only',lr=.001,gradient_clip_norm=5.,regularization_mu=0.,
        total_optimizer_steps=880,total_expert_rounds=20,quota_per_round=44,max_updates_per_available_round=88,
        unavailable_round_policy='freeze expert; carry quota causally; reuse unchanged FIT receipt only',
        availability=availability,availability_source_sha256=file_sha256(args.coverage),
        per_update_positive_samples=128,per_update_negative_samples=128,participant_weights='equal; native alpha used for eligibility only',
        negative_class_balancing='importance weight N_negative/(number_negative_classes*N_class), BASE counts only',
        paired_sampling='joint branches have identical positive and pooled-negative sampling; only loss weights differ',
        evaluation='FIT each available expert round, final round19 locked; thresholds .5 and donor-FIT order statistic recall>=.95',
        baseline='Frozen native receiver30 T2 terminal checkpoint, input-predicted task; not a changing native round model',
        scope='Controlled expert gossip with recorded edge availability; native local training/aggregation NOT replayed',
        CAL_opened=False,TEST_opened=False,ACTIVE=False,native_integration=False,
        communication='pickle application messages including parameter references, gradients, counts, verification function and receipts')
    write(args.out/'protocol_lock.json',protocol); protocol_hash=file_sha256(args.out/'protocol_lock.json')
    roles=CleanRoleData(args.roles,source_data_dir=args.data_dir)
    data=args.out/'current_data'; data.mkdir(); access=[]
    prior=json.loads((args.standalone/'preparation_access.json').read_text())
    for row in prior:
        source=args.standalone/'current_data'/row['file']
        if file_sha256(source)!=row['sha256']: raise ValueError('Standalone staged data changed')
        (data/row['file']).write_bytes(source.read_bytes())
    receiver_base=stage_current(roles,RECEIVER,'base',list(range(12,18)),data,4096,TASK,access)
    write(args.out/'new_preparation_access.json',access)
    train={o:staged(data/f'client_{o}_task_2_base.npz') for o in (DONOR,RECEIVER)}
    fit={o:staged(data/f'client_{o}_task_2_fit.npz') for o in (DONOR,RECEIVER)}
    if (train[RECEIVER]['y']==TARGET).any() or int((train[RECEIVER]['y']==12).sum())<32:
        raise ValueError('Receiver BASE must have class12 negatives and no target positives')
    train_hashes=set(train[DONOR]['hashes'])|set(train[RECEIVER]['hashes'])
    fit_hashes=set(fit[DONOR]['hashes'])|set(fit[RECEIVER]['hashes'])
    if train_hashes&fit_hashes: raise ValueError('BASE/FIT leakage')
    keep=np.array([h not in set(fit[DONOR]['hashes']) for h in fit[RECEIVER]['hashes']])
    with zipfile.ZipFile(args.checkpoint) as archive:
        index=json.loads(archive.read('checkpoint_archive_manifest.json')); name=index['full_terminal_checkpoint']
        raw=archive.read(name)
        if hashlib.sha256(raw).hexdigest()!=index['checksums'][name]: raise ValueError('Corrupt base checkpoint')
        saved=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False); del raw
    config=saved['config']
    if config.get('denice_data_roles_sha256')!=file_sha256(args.roles/'role_manifest.json'): raise ValueError('Wrong BASE authority')
    algorithm=saved['client_algorithm_states'][RECEIVER]; algorithm=algorithm.get('denice',algorithm)
    model=_make_model(config,torch.device('cuda')); detector=ContextDetector(router_mode='binary_cosine')
    restore_denice_state(model,detector,algorithm); model.load_state_dict(saved['client_model_states'][RECEIVER]); model.eval()
    state=snapshot_denice_state(model,detector)
    fields=['neuron_ages','freeze_masks','connection_masks','continual_width','structural_protection','fixed_task_allocation',
            'allocation_policy','capacity_per_class','adapter_mode','active_adapters','bn_frozen_state','adapter_registry','architecture_version']
    minimal={k:state[k] for k in fields if k in state}
    detector_fields=['memo_per_class','router_reference_per_class','router_mode','stable_feature_mask','routing_feature_mask',
                     'activation_memory','context_masks','binarize_thresholds','episode_classes','router_state_fresh',
                     'router_last_refresh_task','router_last_refresh_round','router_stale_reason']
    minimal['context_detector']={k:state['context_detector'][k] for k in detector_fields if k in state['context_detector']}
    payload=dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},algorithm=minimal,
                 input_shape=config['input_shape'],num_classes=config['num_classes'],
                 input_fingerprint=hashlib.sha256(json.dumps(dict(role_manifest=protocol['role_manifest_sha256'],
                     metadata=roles.manifest['metadata_sha256'],shape=[39,1],dtype='float32'),sort_keys=True).encode()).hexdigest())
    base={o:base_predict(model,detector,fit[o]['x'].reshape(-1,39,1)) for o in (DONOR,RECEIVER)}
    replica=_make_model(config,torch.device('cuda')); rd=ContextDetector(router_mode='binary_cosine')
    copied=pickle.loads(pickle.dumps(payload,protocol=5))
    restore_denice_state(replica,rd,copied['algorithm']); replica.load_state_dict(copied['model']); replica.eval()
    for o in (DONOR,RECEIVER):
        assert np.array_equal(base[o],base_predict(replica,rd,fit[o]['x'].reshape(-1,39,1)))
    del saved,model,replica,state,algorithm; torch.cuda.empty_cache()
    branches={}
    for name in protocol['branches']:
        branches[name]=train_branch(name,train,fit,base,keep,payload,availability,args.out,'cuda')
    if len({r['initial_sha256'] for r in branches.values()})!=1: raise AssertionError('Initializations differ')
    if len({r['positive_sample_trace_sha256'] for r in branches.values()})!=1: raise AssertionError('Positive draws differ')
    if branches['joint_class_balanced']['BASE_sample_trace_sha256']!=branches['joint_unbalanced']['BASE_sample_trace_sha256']:
        raise AssertionError('Negative balancing ablation used different samples')
    original_model=torch.nn.Sequential(torch.nn.Linear(39,64),torch.nn.ReLU(),torch.nn.Linear(64,1)).cuda()
    original_model.load_state_dict(torch.load(args.standalone/'standalone_expert.pt',map_location='cuda',weights_only=False)['model'])
    original_scores={o:expert_scores(original_model,fit[o]['x']) for o in (DONOR,RECEIVER)}
    old_tau=thresholds(original_scores[DONOR],fit[DONOR]['y'].numpy())
    historical={name:{o:fusion(fit[o]['y'].numpy(),original_scores[o],base[o],tau) for o in (DONOR,RECEIVER)}
                for name,tau in (('fixed_0p5',.5),('donor_FIT_recall95',old_tau))}
    assert file_sha256(args.out/'protocol_lock.json')==protocol_hash
    result=dict(completed=True,protocol=protocol,protocol_sha256=protocol_hash,
        receiver_BASE_class12_unique=int((train[RECEIVER]['y']==12).sum()),
        train_evaluation_content_disjoint=True,pooled_FIT_duplicates_removed=int((~keep).sum()),
        baseline_minimal_payload_prediction_equivalence=True,baseline_payload_contains_raw_examples=False,
        branches=branches,historical_standalone_controls=historical,CAL_opened=False,TEST_opened=False,ACTIVE=False,
        native_integration=False,elapsed_seconds=time.monotonic()-started,
        limitation='FIT development comparison against a frozen legacy base predictor. No independent CAL acceptance, historical retention, or native round-training gain established.')
    write(args.out/'report.json',result); print('COMPLETED',args.out/'report.json',flush=True)


if __name__=='__main__': main()
