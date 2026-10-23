"""Registered BASE/FIT-only coverage-aware negative experiment; no CAL or ACTIVE API."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

from pilot_candle_v4_standalone import REPO,write,file_sha256
import numpy as np
import torch
from sklearn.metrics import f1_score
from fed_learning.training.candle_v4_shadow import ShadowService,digest
from fed_learning.training.candle_v4_coverage import CoverageService,negative_contributors,VERSION
from fed_learning.training.candle_v4_preflight import positive_neighbors
from fed_learning.training.candle_v4_bank import fuse,base_predictions
from fed_learning.training.decentralized_denice_il import _split_local_validation
from fed_learning.training.checkpoint_state import snapshot_denice_state
from eval_checkpoint import _make_denice_client_model

SOURCE=REPO/'audit_denice/candle_v4_graph_transition_20261023'
ARMS=['current','coverage_aware','no_extra_ablation']


class Wire:
    def __init__(self,path,receiver):
        self.path=path;self.receiver=receiver;self.graph=None;self.total=0;self.kinds=Counter();self.messages=0
    def send(self,sender,recipient,kind,message):
        if sender==recipient:return deepcopy(message)
        remote=recipient if sender==self.receiver else sender
        if remote not in positive_neighbors(self.graph,self.receiver):raise ValueError('Wire peer outside current graph')
        raw=pickle.dumps(message,protocol=5)
        if self.total+len(raw)>128*1024**2:raise RuntimeError('Locked development wire budget exhausted')
        self.total+=len(raw);self.kinds[kind]+=len(raw);self.messages+=1
        with self.path.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(sender=sender,receiver=recipient,kind=kind,bytes=len(raw)))+'\n')
        return pickle.loads(raw)
    def summary(self):return dict(bytes=self.total,messages=self.messages,by_kind=dict(self.kinds))


def attach_wire(engine,wire):
    def send(kind,message):
        owner=message.get('contributor',engine.spec['donor'])
        if kind.endswith('reference'):
            return wire.send(engine.spec['receiver'],owner,kind,message)
        return wire.send(owner,engine.spec['receiver'],kind,message)
    engine.send=send


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    fixture=json.loads((SOURCE/'protocol_lock.json').read_text())
    preflight=json.loads((SOURCE/'candidate_preflight.json').read_text())
    source_paths=['tools/pilot_candle_v4_coverage.py',
        'candle_v4_runtime/fed_learning/training/candle_v4_coverage.py',
        'candle_v4_runtime/fed_learning/training/candle_v4_shadow.py']
    data_paths=[SOURCE/'evidence/task_2'/f'client_{o}_{role}.npz' for o in fixture['owners'] for role in ('base','fit')]
    graphs=json.loads((SOURCE/'baseline/cluster_history.json').read_text())
    graphs=[g for g in graphs if g['task']==2]
    if [g['round'] for g in graphs]!=list(range(8)):raise ValueError('Original eight current graphs required')
    write(out/'protocol_lock.json',dict(version=VERSION,source_fixture=fixture['fixture'],receiver=fixture['receiver'],
        owners=fixture['owners'],experts=fixture['experts'],task=2,current_classes=list(range(12,18)),seed=42,
        arms=ARMS,model='39->64->1, ReLU, FP32 CPU',optimizer='Adam lr=.001 weight_decay=0 clip5',
        rounds=8,quota_per_round=44,maximum_catchup=88,steps_per_expert=352,
        minibatch=dict(positives=128,negatives=128,total=256),
        loss='0.5 positive mean + 0.5 negative class-balanced expectation; importance weights from BASE counts only',
        negative_selection='Classes with zero BASE coverage across primary donor+receiver; choose smallest current eligible peer ID with >=32 BASE samples per missing class; request only that negative class',
        ablation='Same generic implementation with extra negative contributors disabled',
        schedule='Replay only the current recorded native graph per round; no native retraining or claim of new online integration',
        endpoints='Round7 only, no early stopping or winning-round selection; FIT opened after all training arms complete',
        primary='FIT-select donor recall95 order statistic then evaluate FIT-assess',secondary='fixed threshold0.5',
        FIT_partition='sha256(FP32 input bytes) first8 bytes as big-endian integer %2: 0=threshold select,1=assessment; owner-independent',
        pooled_dedup='assessment: unique (feature content hash,label), first owner ID retained; owner pools also reported',
        objective='Lower expert12->class13 false activation without donor/pooled target recall or candidate accuracy loss; report each criterion separately, no deployment PASS',
        CAL_samples_opened=False,TEST_opened=False,ACTIVE=False,full_training=False,
        evidence_independence='Development only. Hypothesis motivated by earlier aggregate CAL rejection; no rejected CAL rows read or used to fit/tune/confirm',
        communication='Independent serialized references/gradients per arm; common immutable native FIT reference and identity receipts charged separately; no real network benchmark',
        wire_budget_per_arm=128*1024**2,
        roles_authority_sha256=fixture['roles_sha256'],
        source_sha256={str(p.relative_to(REPO)):file_sha256(p) for p in data_paths},
        code_sha256={p:file_sha256(REPO/p) for p in source_paths},
        graph_sha256=file_sha256(SOURCE/'baseline/cluster_history.json'),
        baseline_checkpoint_sha256=file_sha256(SOURCE/'baseline/checkpoint_task_2.pt'),
        prior_expert_hashes={c:r['experts'][str(c)]['expert_sha256'] for r in
            json.loads((REPO/'artifacts/candle_v4_graph_transition_20261023.json').read_text())['expert_history']
            if r['task']==2 and r['round']==7 for c in (e['target'] for e in fixture['experts'])},
        metadata_current_BASE_counts={o:preflight['metadata'][str(o)]['current_train_BASE'] for o in fixture['owners']}))


def load_role(owner,role,classes):
    if role not in ('base','fit'):raise ValueError('Only BASE/FIT roles authorized')
    path=SOURCE/'evidence/task_2'/f'client_{owner}_{role}.npz'
    with np.load(path,allow_pickle=False) as z:x=torch.from_numpy(z['X'].copy());y=torch.from_numpy(z['y'].copy())
    if x.shape[1:]!=(39,) or not torch.isfinite(x).all() or not set(y.tolist())<=set(classes):
        raise ValueError('Input/current-task contract mismatch')
    return x,y


def train(out,lock,graphs):
    receiver=lock['receiver'];data={};engines={};wires={};history=[]
    helper=ShadowService(dict(mode='DRAFT',task=2,receiver=receiver,donor=lock['experts'][0]['donor'],
        target=lock['experts'][0]['target'],current_classes=lock['current_classes'],input_sha256=lock['roles_authority_sha256'],fit_paths={}))
    for owner in lock['owners']:
        x,y=load_role(owner,'base',lock['current_classes'])
        x,y,_,_=_split_local_validation(x,y,.1,42+20000+owner)
        data[owner]=helper.data(SimpleNamespace(X_train=x,y_train=y))
        if dict(Counter(data[owner][1].tolist()))!={int(c):n for c,n in lock['metadata_current_BASE_counts'][str(owner)].items()}:
            raise ValueError('BASE differs from native smoke')
    for arm in ARMS:
        directory=out/arm;directory.mkdir()
        wires[arm]=Wire(directory/'wire.jsonl',receiver);engines[arm]={}
        for entry in lock['experts']:
            spec=dict(mode='DRAFT',task=2,receiver=receiver,donor=entry['donor'],target=entry['target'],
                current_classes=lock['current_classes'],input_sha256=lock['roles_authority_sha256'],fit_paths={})
            engine=(ShadowService(spec) if arm=='current' else CoverageService(spec,enabled=arm=='coverage_aware'))
            attach_wire(engine,wires[arm]);engines[arm][entry['target']]=engine
        wires[arm].graph=graphs[0]
        advertised=set(e['donor'] for e in lock['experts'])
        if arm=='coverage_aware':advertised.update(positive_neighbors(graphs[0],receiver))
        for owner in sorted(advertised):
            wires[arm].send(owner,receiver,'BASE_count_metadata',dict(owner=owner,task=2,
                counts=dict(Counter(data[owner][1].tolist())),input_authority=lock['roles_authority_sha256']))
    initial={arm:{c:digest(e.branches['joint_class_balanced'][0].state_dict()) for c,e in bank.items()} for arm,bank in engines.items()}
    if len({h for row in initial.values() for h in row.values()})!=1:raise AssertionError('Initialization mismatch')
    for round_id,graph in enumerate(graphs):
        for arm in ARMS:
            wires[arm].graph=graph
            for entry in lock['experts']:
                c=entry['target'];engine=engines[arm][c];engine.pending+=44
                if entry['donor'] not in positive_neighbors(graph,receiver):continue
                if isinstance(engine,CoverageService):engine.graph=graph
                extra=0;loss=0.
                for _ in range(min(engine.pending,88)):
                    result=engine.train_step('joint_class_balanced',data,engine.steps,2,round_id)
                    engine.steps+=1;engine.pending-=1;loss+=result['loss'];extra+=result.get('extra_negative_samples',0)
                history.append(dict(round=round_id,arm=arm,target=c,steps=engine.steps,loss_sum=loss,
                    extra_negative_samples=extra,bytes=wires[arm].total))
        print('DEV TRAIN round',round_id,'steps',{arm:{c:e.steps for c,e in row.items()} for arm,row in engines.items()},flush=True)
    hashes={arm:{c:digest(e.branches['joint_class_balanced'][0].state_dict()) for c,e in row.items()} for arm,row in engines.items()}
    checks=dict(matched_initialization=True,all_steps_352=all(e.steps==352 for row in engines.values() for e in row.values()),
        no_extra_equals_current=hashes['current']==hashes['no_extra_ablation'],
        current_reproduces_previous_native_experts=hashes['current']=={int(c):h for c,h in lock['prior_expert_hashes'].items()})
    if not all(checks.values()):raise AssertionError(checks)
    torch.save({arm:{c:e.state_dict() for c,e in row.items()} for arm,row in engines.items()},out/'development_experts.pt')
    receipt=dict(checks=checks,initial_hashes=initial,final_hashes=hashes,history=history,
        examples_per_expert=352*256,positive_examples_per_expert=352*128,negative_examples_per_expert=352*128,
        coverage_traces={str(c):e.trace for c,e in engines['coverage_aware'].items()},FIT_opened_for_training=False,
        negative_plans={str(e['target']):negative_contributors(receiver,e['donor'],e['target'],
            {o:Counter(y.tolist()) for o,(_,y) in data.items()},graphs[0]) for e in lock['experts']})
    write(out/'training_complete.json',receipt)
    return engines,wires,receipt


def partition(x):
    hashes=[hashlib.sha256(row.numpy().tobytes()).hexdigest() for row in x]
    select=np.asarray([int(h[:16],16)%2==0 for h in hashes])
    return hashes,select


def panel(y,baseline,pred,assigned,classes,collisions):
    before=baseline==y;after=pred==y
    return dict(rows=len(y),correct=int(after.sum()),accuracy=float(after.mean()),
        Macro_F1=float(f1_score(y,pred,labels=list(range(12,18)),average='macro',zero_division=0)),
        baseline_correct=int(before.sum()),rescue=int((~before&after).sum()),break_count=int((before&~after).sum()),
        changed_wrong=int((~before&~after&(pred!=baseline)).sum()),multi_gate_rows=int((collisions>1).sum()),
        targets={str(c):dict(positive=int((y==c).sum()),hits=int(((pred==c)&(y==c)).sum()),
            recall=float((pred[y==c]==c).mean()) if (y==c).any() else None,
            precision=float((y[pred==c]==c).mean()) if (pred==c).any() else None,
            negatives=int((y!=c).sum()),false_activations=int(((assigned==c)&(y!=c)).sum()),
            output_false_positives=int(((pred==c)&(y!=c)).sum()),
            per_negative_class={str(label):dict(n=int((y==label).sum()),FP=int(((assigned==c)&(y==label)).sum()))
                for label in sorted(set(y.tolist())) if label!=c}) for c in classes})


def evaluate(out,lock,graphs,engines,wires,training):
    receiver=lock['receiver'];classes=sorted(engines['current']);fit={};base={};common=Wire(out/'common_wire.jsonl',receiver)
    common.graph=graphs[-1]
    ckpt=torch.load(SOURCE/'baseline/checkpoint_task_2.pt',map_location='cpu',weights_only=False)
    native,detector=_make_denice_client_model(ckpt,receiver,'cuda')
    state=dict(config=ckpt['config'],weights={k:v.detach().cpu() for k,v in native.state_dict().items()},
               algorithm=snapshot_denice_state(native,detector))
    state['algorithm'].get('context_detector',{}).pop('reference_input_memory',None)
    native_sha=digest(state);native_equal=True;global_train_hashes=set()
    for owner in lock['owners']:
        x,_=load_role(owner,'base',lock['current_classes'])
        global_train_hashes.update(hashlib.sha256(row.numpy().tobytes()).hexdigest() for row in x)
    for owner in lock['owners']:
        x,y=load_role(owner,'fit',lock['current_classes']);hashes,select=partition(x)
        if global_train_hashes.intersection(hashes):raise ValueError('BASE/FIT content overlap')
        predictions=base_predictions(native,detector,x,list(range(18)))
        if owner!=receiver:
            packet=common.send(receiver,owner,'native_FIT_reference',state)
            if digest(packet)!=native_sha:raise ValueError('Native function snapshot changed on wire')
            remote,remote_detector=_make_denice_client_model(dict(config=packet['config'],
                client_model_states={receiver:packet['weights']},client_algorithm_states={receiver:{'denice':packet['algorithm']}}),receiver,'cuda')
            native_equal &= np.array_equal(predictions,base_predictions(remote,remote_detector,x,list(range(18))))
            del remote,remote_detector
        base[owner]=predictions
        frame=dict(content_hashes=[h for h,s in zip(hashes,select) if not s],labels=y.numpy()[~select],native=predictions[~select])
        common.send(owner,receiver,'FIT_assessment_identity',frame)
        fit[owner]=dict(x=x,y=y.numpy(),hashes=hashes,select=select)
    if not native_equal:raise AssertionError('Restored remote native predictions differ')
    metrics={};predictions_by_arm={};thresholds={}
    for arm in ARMS:
        wire=wires[arm];wire.graph=graphs[-1];scores={};thresholds[arm]={}
        states={c:e.branches['joint_class_balanced'][0].state_dict() for c,e in engines[arm].items()}
        for owner,data in fit.items():
            packet=wire.send(receiver,owner,'FIT_expert_reference',dict(states=states,classes=classes,input_sha256=lock['roles_authority_sha256']))
            if digest(packet['states'])!=digest(states):raise ValueError('Expert function changed on FIT wire')
            scores[owner]=np.column_stack([torch.sigmoid(engines[arm][c].branches['joint_class_balanced'][0](data['x'])).detach().numpy().flatten()
                                          for c in classes])
        for entry in lock['experts']:
            c,donor=entry['target'],entry['donor'];data=fit[donor]
            values=scores[donor][data['select']&(data['y']==c),classes.index(c)]
            if len(values)<32:raise ValueError('Insufficient pre-registered FIT-select positives')
            tau=float(np.sort(values)[::-1][int(np.ceil(.95*len(values)))-1])
            receipt=wire.send(donor,receiver,'FIT_threshold_receipt',dict(target=c,positive_rows=len(values),tau=tau,
                expert_sha256=digest(states[c]),split='content-hash FIT-select'))
            thresholds[arm][c]=receipt['tau']
        profiles=dict(primary_FIT_select95=[thresholds[arm][c] for c in classes],secondary_fixed_0p5=[.5]*len(classes))
        for owner in lock['owners']:wire.send(receiver,owner,'locked_FIT_thresholds',profiles)
        metrics[arm]={};predictions_by_arm[arm]={}
        for profile,taus in profiles.items():
            owner_metrics={};pools=[];seen=set();raw={};removed=0
            for owner,data in fit.items():
                assessment=~data['select'];y=data['y'][assessment];old=base[owner][assessment];s=scores[owner][assessment]
                pred,n=fuse(old,s,taus,classes);assigned=np.where(n>0,pred,-1)
                owner_metrics[str(owner)]=panel(y,old,pred,assigned,classes,n)
                raw[owner]=dict(predictions=pred,assigned=assigned,activation_count=n)
                keep=[]
                for i,h in enumerate(np.asarray(data['hashes'])[assessment]):
                    key=(str(h),int(y[i]))
                    if key not in seen:seen.add(key);keep.append(i)
                    else:removed+=1
                pools.append(tuple(v[keep] for v in (y,old,pred,assigned,n)))
            pooled=panel(*(np.concatenate([p[i] for p in pools]) for i in range(4)),classes,
                         np.concatenate([p[4] for p in pools]))
            metrics[arm][profile]=dict(thresholds={str(c):t for c,t in zip(classes,taus)},owners=owner_metrics,
                pooled_unique=pooled,removed_cross_owner_duplicates=removed)
            predictions_by_arm[arm][profile]=raw
        for owner in lock['owners']:
            wire.send(owner,receiver,'FIT_prediction_receipt',{p:predictions_by_arm[arm][p][owner] for p in profiles})
    del native,detector,ckpt
    pairs={}
    for profile in metrics['current']:
        rows={}
        for arm in ('coverage_aware','no_extra_ablation'):
            gains=losses=0
            for owner,data in fit.items():
                y=data['y'][~data['select']]
                control=predictions_by_arm['current'][profile][owner]['predictions']==y
                candidate=predictions_by_arm[arm][profile][owner]['predictions']==y
                gains+=int((~control&candidate).sum());losses+=int((control&~candidate).sum())
            rows[arm]=dict(gained_correct_events=gains,lost_correct_events=losses,net_events=gains-losses)
        pairs[profile]=rows
    result=dict(version=VERSION,completed=True,protocol=lock,training=training,FIT_metrics=metrics,
        thresholds=thresholds,paired_owner_event_comparison=pairs,native_remote_reference_exact=native_equal,
        FIT_partitions={o:dict(select_rows=int(d['select'].sum()),assessment_rows=int((~d['select']).sum()),
            select_class_counts=dict(Counter(d['y'][d['select']].tolist())),
            assessment_class_counts=dict(Counter(d['y'][~d['select']].tolist()))) for o,d in fit.items()},
        communication=dict(common=common.summary(),arms={a:w.summary() for a,w in wires.items()},
            isolated_arm_total={a:w.total+common.total for a,w in wires.items()},
            combined_experiment=common.total+sum(w.total for w in wires.values()),
            extra_coverage_vs_no_extra=wires['coverage_aware'].total-wires['no_extra_ablation'].total,
            native_training_traffic=None,mode='Serialized in-process simulation; native reference immutable/shared once, BASE/FIT raw samples never sent'),
        CAL_samples_opened=False,TEST_opened=False,ACTIVE=False,full_training=False,
        conclusion_scope='One seed, development only; no confirmation, deployment acceptance or native integration claim')
    write(out/'report.json',result)
    return result


def run(out):
    lock=json.loads((out/'protocol_lock.json').read_text())
    for path,sha in lock['code_sha256'].items():
        if file_sha256(REPO/path)!=sha:raise ValueError('Registered implementation changed')
    for path,sha in lock['source_sha256'].items():
        if file_sha256(REPO/path)!=sha:raise ValueError('Registered data changed')
    if file_sha256(SOURCE/'baseline/cluster_history.json')!=lock['graph_sha256']:raise ValueError('Graph changed')
    if file_sha256(SOURCE/'baseline/checkpoint_task_2.pt')!=lock['baseline_checkpoint_sha256']:raise ValueError('Native reference changed')
    graphs=[g for g in json.loads((SOURCE/'baseline/cluster_history.json').read_text()) if g['task']==2]
    engines,wires,receipt=train(out,lock,graphs)
    result=evaluate(out,lock,graphs,engines,wires,receipt)
    print(json.dumps(dict(completed=result['completed'],checks=receipt['checks'],communication=result['communication']),indent=2),flush=True)


if __name__=='__main__':
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8',errors='replace')
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--mode',choices=['prepare','run'],required=True)
    args=parser.parse_args();out=args.out.resolve()
    if args.mode=='prepare':prepare(out)
    else:run(out)
