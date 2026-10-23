"""Bounded V4 bank: one-shot current-CAL acceptance and immutable active experts."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import torch

from .candle_v4_shadow import ShadowService, digest, native_fingerprint

VERSION = 'candle_v4_active_bank_v1'


def fuse(native, scores, thresholds, classes):
    """Fixed normalized excess; smallest class ID breaks ties. No true-task input."""
    native = np.asarray(native).copy()
    scores = np.asarray(scores, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if not len(classes): return native, np.zeros(len(native), dtype=np.int64)
    if scores.shape != (len(native), len(classes)) or not np.isfinite(scores).all():
        raise ValueError('Malformed expert scores')
    if not np.isfinite(thresholds).all() or np.any((thresholds < 0) | (thresholds > 1)):
        raise ValueError('Invalid locked bank thresholds')
    eligible = scores >= thresholds
    excess = np.where(eligible, (scores-thresholds)/np.maximum(1-thresholds,np.finfo(np.float32).eps), -np.inf)
    winner = excess.argmax(1); triggered = eligible.any(1)
    native[triggered] = np.asarray(classes)[winner[triggered]]
    return native, eligible.sum(1)


def score_matrix(engines, classes, x):
    with torch.no_grad():
        return np.column_stack([torch.sigmoid(engines[c].branches['joint_class_balanced'][0](x)).flatten().numpy()
                                for c in classes]) if classes else np.empty((len(x), 0))


def base_predictions(model, detector, x, seen_classes):
    from .denice_eval import _denice_routed_logits_with_episodes
    modes = [(module, module.training) for module in model.modules()]
    device = next(model.parameters()).device; model.eval(); predictions=[]
    try:
        with torch.no_grad():
            for start in range(0, len(x), 512):
                logits,_ = _denice_routed_logits_with_episodes(model,x[start:start+512].to(device),
                    detector,seen_classes,str(device),inference_policy='pred_hard')
                predictions.extend(logits.argmax(1).cpu().tolist())
    finally:
        for module, mode in modes: module.training=mode
    return np.asarray(predictions, dtype=np.int64)


class ExpertBank:
    def __init__(self, spec, saved=None):
        self.spec=deepcopy(spec)
        if spec['version']!=VERSION or spec['policy']!='empirical_current_CAL':
            raise ValueError('Unregistered bank protocol')
        self.engines={}; self.status={}; self.thresholds={}; self.certificates={}
        for entry in spec['experts']:
            c=entry['target']
            child=dict(task=entry['birth_task'],receiver=spec['receiver'],donor=entry['donor'],target=c,
                current_classes=entry['current_classes'],input_sha256=spec['input_sha256'],fit_paths={},mode='DRAFT',
                wire_budget=spec['per_expert_budget'])
            self.engines[c]=ShadowService(child,(saved or {}).get('engines',{}).get(c))
            self.status[c]='DRAFT'
        self.last_boundary=None; self.history=[]; self.acceptances=[]; self.bytes=0; self.by_kind=Counter()
        self.per_class_bytes=Counter(); self.gated_tasks=[]; self.closed_tasks=[]
        if saved:
            if saved['version']!=VERSION or saved['spec']!=self.spec: raise ValueError('Bank resume contract changed')
            for key in ('status','thresholds','certificates','last_boundary','history','acceptances','bytes',
                        'gated_tasks','closed_tasks'):
                setattr(self,key,deepcopy(saved[key]))
            self.by_kind=Counter(saved['by_kind']); self.per_class_bytes=Counter(saved['per_class_bytes'])

    def state_dict(self):
        return deepcopy(dict(version=VERSION,spec=self.spec,engines={c:e.state_dict() for c,e in self.engines.items()},
            status=self.status,thresholds=self.thresholds,certificates=self.certificates,last_boundary=self.last_boundary,
            history=self.history,acceptances=self.acceptances,bytes=self.bytes,by_kind=dict(self.by_kind),
            per_class_bytes=dict(self.per_class_bytes),gated_tasks=self.gated_tasks,closed_tasks=self.closed_tasks))

    def send(self, kind, message, classes):
        if message.get('owner')==self.spec['receiver']:
            return deepcopy(message)  # owner-local computation is not a peer transmission
        raw=pickle.dumps(message,protocol=5)
        if self.bytes+len(raw)>self.spec['bank_budget']: raise RuntimeError('Bank budget exhausted')
        self.bytes+=len(raw); self.by_kind[kind]+=len(raw)
        # Deterministic allocation, including remainder, sums to exact packet bytes.
        for index,c in enumerate(sorted(classes)):
            self.per_class_bytes[c]+=len(raw)//len(classes)+(index<len(raw)%len(classes))
        return pickle.loads(raw)

    def role(self, owner, task, role):
        """Read only the current role. Stage provenance before returning any labels."""
        if role not in ('base','fit','calibration') or task in self.closed_tasks:
            raise ValueError('Historical or unauthorized evidence access')
        if self.last_boundary and task!=self.last_boundary[0]: raise ValueError('Evidence outside current task')
        stage=Path(self.spec['stage_dir'])/f'task_{task}'; stage.mkdir(parents=True,exist_ok=True)
        path=stage/f'client_{owner}_{role}.npz'
        if role=='calibration':
            # CAL is materialized lazily at the locked task-end gate, never for gradient/FIT selection.
            if task in self.gated_tasks: raise ValueError('CAL acceptance cannot be retried')
            if not path.exists():
                from fed_learning.data.denice_clean_roles import CleanRoleData,file_sha256
                authority=CleanRoleData(self.spec['roles_dir'],source_data_dir=self.spec['data_dir'])
                item=authority.manifest['clients'][str(owner)]
                source=authority.source/f'client_{owner}_train.npz'; indices=authority.root/'indices'/f'client_{owner}.npz'
                if file_sha256(source)!=item['source_sha256'] or file_sha256(indices)!=item['index_sha256']:
                    raise ValueError('Evidence source changed')
                with np.load(indices,allow_pickle=False) as z: selected=z['calibration']
                with np.load(source,allow_pickle=False) as z:
                    y=z['y_train']; selected=selected[np.isin(y[selected],range(task*6,min(task*6+6,34)))]
                    x=np.asarray(z['X_train'][selected],np.float32).reshape(len(selected),39); y=y[selected]
                seen=set(); keep=[]
                for i,row in enumerate(x):
                    h=hashlib.sha256(row.tobytes()).hexdigest()
                    if h not in seen:keep.append(i);seen.add(h)
                np.savez_compressed(path,X=x[keep],y=y[keep],source_rows=selected[keep])
                self.send('CAL_provenance',dict(owner=owner,task=task,source_sha256=item['source_sha256'],
                    index_sha256=item['index_sha256'],rows=len(keep),stage_sha256=file_sha256(path)),list(self.engines))
        with np.load(path,allow_pickle=False) as z:
            x=torch.from_numpy(z['X'].copy()); y=z['y'].copy()
        if x.shape[1:]!=(39,) or not set(y.tolist())<=set(range(task*6,min(task*6+6,34))):
            raise ValueError('Off-task role stage')
        return x,y

    def infer(self, native, x):
        classes=sorted(c for c in self.engines if self.status[c]=='ACTIVE')
        for c in classes:
            cert=self.certificates[c]
            if (cert['status']!='PASS' or cert['version']!=VERSION or
                    cert['input_sha256']!=self.spec['input_sha256'] or cert['threshold']!=self.thresholds[c] or
                    digest(self.engines[c].branches['joint_class_balanced'][0].state_dict())!=cert['expert_sha256']):
                raise ValueError('Active expert mutated after certification')
        return fuse(native,score_matrix(self.engines,classes,x),[self.thresholds[c] for c in classes],classes)

    def begin_task(self,task):
        if self.last_boundary and task!=self.last_boundary[0]:
            if self.last_boundary[0] not in self.closed_tasks:self.closed_tasks.append(self.last_boundary[0])
            for c,engine in self.engines.items():
                engine.pending=0
                if self.status[c]=='ACTIVE':self.status[c]='PENDING_NEW_TASK'

    def after_round(self,task,round_id,models,detectors,clients,active_ids,graph):
        if self.last_boundary and (task,round_id)<=tuple(self.last_boundary): raise ValueError('Repeated bank round')
        before=native_fingerprint(models,detectors)
        if self.last_boundary and task!=self.last_boundary[0]:
            if self.last_boundary[0] not in self.closed_tasks:self.closed_tasks.append(self.last_boundary[0])
            for engine in self.engines.values(): engine.pending=0
        self.last_boundary=[task,round_id]
        audit={}
        for c,engine in sorted(self.engines.items()):
            entry=next(e for e in self.spec['experts'] if e['target']==c)
            if task!=entry['birth_task']:
                audit[c]=dict(status=self.status[c],frozen=True,expert_sha256=digest(engine.branches['joint_class_balanced'][0].state_dict()))
                continue
            if self.status[c]!='DRAFT': raise ValueError('Committed expert cannot keep training')
            engine.pending+=44
            receiver=self.spec['receiver']; donor=entry['donor']
            row=graph.get('alpha_debug',{}).get(receiver,graph.get('alpha_debug',{}).get(str(receiver),{}))
            eligible=(receiver in active_ids and donor in active_ids and any(int(j)==donor and a>0
                for j,a in zip(row.get('group_ids',[]),row.get('alphas',[]))))
            updates=0
            if eligible:
                data={o:engine.data(clients[o]) for o in (receiver,donor)}
                if (data[receiver][1]==c).any() or not (data[donor][1]==c).any(): raise ValueError('Invalid ownership fixture')
                budget_before=engine.bytes
                for _ in range(min(88,engine.pending)):
                    engine.train_step('joint_class_balanced',data,engine.steps,task,round_id)
                    engine.steps+=1; engine.pending-=1; updates+=1
                used=engine.bytes-budget_before
                self.bytes+=used; self.per_class_bytes[c]+=used; self.by_kind['training_reference_gradient']+=used
                if self.bytes>self.spec['bank_budget']:raise RuntimeError('Bank training communication exhausted')
            audit[c]=dict(eligible=eligible,updates=updates,steps=engine.steps,pending=engine.pending,
                expert_sha256=digest(engine.branches['joint_class_balanced'][0].state_dict()))
        if before!=native_fingerprint(models,detectors): raise RuntimeError('Bank mutated native state')
        receipt=dict(task=task,round=round_id,experts=audit,bytes=self.bytes,native_sha256=before)
        self.history.append(receipt)
        print(f'CANDLE V4 BANK T{task} R{round_id} {audit}',flush=True)
        return receipt

    def finish_task(self,task,models,detectors,clients,active_ids,graph):
        if task in self.gated_tasks: raise ValueError('Repeated CAL gate')
        classes=sorted(self.engines)
        birth=all(e['birth_task']==task for e in self.spec['experts'])
        receiver=self.spec['receiver']
        row=graph.get('alpha_debug',{}).get(receiver,graph.get('alpha_debug',{}).get(str(receiver),{}))
        if len(row.get('group_ids',[]))!=len(row.get('alphas',[])):
            raise ValueError('Malformed current gate graph')
        peers={int(j) for j,a in zip(row.get('group_ids',[]),row.get('alphas',[]))
               if int(j)!=receiver and np.isfinite(a) and a>0}
        untrained=[c for c in classes if self.engines[c].steps<=0]
        required={o for o in self.spec['owners'] if o!=receiver}
        missing=sorted(required-peers)
        terminal=None; reasons=[]
        if birth and untrained:
            terminal='DEFERRED_NO_AUTHORIZED_TRAINING'; reasons=[f'UNTRAINED_CLASS_{c}' for c in untrained]
        elif not birth and not any(s in ('ACTIVE','PENDING_NEW_TASK') for s in self.status.values()):
            terminal='SKIPPED_NO_ACTIVE_BANK'; reasons=['NO_PREVIOUSLY_ACCEPTED_BANK']
        elif missing or not required.issubset(set(active_ids)):
            terminal='DEFERRED_GATE_GRAPH_SCOPE'; reasons=[f'UNREACHABLE_CAL_OWNER_{o}' for o in missing]
            if not reasons:reasons=['INACTIVE_REQUIRED_CAL_OWNER']
        if terminal:
            for c in classes:
                if birth:self.status[c]='DEFERRED'
                elif self.status[c] in ('ACTIVE','PENDING_NEW_TASK'):self.status[c]='SUSPENDED'
            receipt=dict(task=task,outcome=terminal,birth_acceptance=birth,failures=reasons,
                status=deepcopy(self.status),steps={c:self.engines[c].steps for c in classes},
                current_peers=sorted(peers),FIT_opened=False,CAL_opened=False,bytes=self.bytes,
                policy='Preflight only; no candidate/threshold search and no evidence retry')
            self.gated_tasks.append(task); self.acceptances.append(receipt)
            print(f'CANDLE V4 ACTIVE GATE T{task}: {terminal} {reasons}',flush=True)
            return receipt
        before=native_fingerprint(models,detectors)
        hashes={c:digest(self.engines[c].branches['joint_class_balanced'][0].state_dict()) for c in classes}
        if birth:
            for entry in self.spec['experts']:
                c=entry['target']; x,y=self.role(entry['donor'],task,'fit')
                values=score_matrix(self.engines,[c],x)[:,0][y==c]
                if len(values)<32: raise ValueError('Insufficient donor FIT for fixed threshold rule')
                tau=float(np.sort(values)[::-1][int(np.ceil(.95*len(values)))-1])
                self.thresholds[c]=tau
            # Thresholds/versions are persisted before CAL is opened.
            lock=dict(task=task,thresholds=self.thresholds,expert_hashes=hashes,
                rule='donor FIT recall95 order statistic; normalized-excess multi-expert arbitration',version=VERSION)
            path=Path(self.spec['stage_dir'])/f'threshold_lock_task_{task}.json'
            if path.exists(): raise ValueError('Never overwrite an existing CAL threshold lock')
            path.write_text(json.dumps(lock,indent=2),encoding='utf-8')
        else:
            for c in classes:
                old=self.certificates.get(c)
                if old and hashes[c]!=old['expert_sha256']:raise RuntimeError('Historical expert weights drifted')
        failures=[]; rows={}; total_rescue=0; total_break=0; collisions=0; inference_pools=[]
        receiver=self.spec['receiver']; seen=list(range(min(6*(task+1),34)))
        # Verification scope is the current required owner set, not the global population.
        owners=[o for o in self.spec['owners'] if o in active_ids]
        if not owners:failures.append('NO_CURRENT_EVIDENCE_OWNERS')
        from .checkpoint_state import snapshot_denice_state
        payload=dict(weights={k:v.detach().cpu() for k,v in models[receiver].state_dict().items()},
                     algorithm=snapshot_denice_state(models[receiver],detectors[receiver]))
        payload['algorithm'].get('context_detector',{}).pop('reference_input_memory',None)
        for owner in owners:
            if owner!=receiver:self.send('gate_base_reference',dict(receiver=receiver,owner=owner,state=payload),classes)
            self.send('gate_expert_reference',dict(owner=owner,thresholds=self.thresholds,
                states={c:self.engines[c].branches['joint_class_balanced'][0].state_dict() for c in classes}),classes)
            x,y=self.role(owner,task,'calibration')
            # Exclude every fit/base content staged for this task, across owners.
            excluded=set()
            for peer in self.spec['owners']:
                for role in ('base','fit'):
                    path=Path(self.spec['stage_dir'])/f'task_{task}'/f'client_{peer}_{role}.npz'
                    if path.exists():
                        with np.load(path,allow_pickle=False) as z:
                            excluded.update(hashlib.sha256(row.tobytes()).digest() for row in z['X'])
            kept=[i for i,row in enumerate(x) if hashlib.sha256(row.numpy().tobytes()).digest() not in excluded]
            excluded_count=len(x)-len(kept); x=x[kept]; y=y[kept]
            if not len(y):
                failures.append(f'OWNER_{owner}_NO_DISJOINT_CAL'); continue
            baseline=base_predictions(models[receiver],detectors[receiver],x,seen)
            scores=score_matrix(self.engines,classes,x)
            predicted,activation_count=fuse(baseline,scores,[self.thresholds[c] for c in classes],classes)
            inference_pools.append((x,baseline,predicted))
            old=baseline==y; new=predicted==y; negative=~np.isin(y,classes)
            rescue=int((~old&new).sum()); broken=int((old&~new).sum())
            total_rescue+=rescue; total_break+=broken; collisions+=int((activation_count>1).sum())
            if int(negative.sum())<32:failures.append(f'OWNER_{owner}_NEGATIVE_COUNT')
            if broken:failures.append(f'OWNER_{owner}_NEGATIVE_BREAK')
            targets={}
            for c in classes:
                positive=y==c; negatives=y!=c
                fp=int(((predicted==c)&negatives).sum())
                if negatives.sum() and fp/negatives.sum()>.001:failures.append(f'OWNER_{owner}_CLASS_{c}_FAR')
                per_negative={str(label):dict(n=int((y==label).sum()),fp=int(((predicted==c)&(y==label)).sum()))
                              for label in sorted(set(y.tolist())) if label!=c}
                if any(r['fp']/r['n']>.001 for r in per_negative.values()):failures.append(f'OWNER_{owner}_CLASS_{c}_PER_CLASS_FAR')
                n=int(positive.sum()); hits=int(((predicted==c)&positive).sum())
                # Required primary positive owner is the declared donor, never chosen from CAL outcomes.
                donor=next(e['donor'] for e in self.spec['experts'] if e['target']==c)
                if birth and owner==donor and (n<32 or hits/n<.95):failures.append(f'CLASS_{c}_POSITIVE_RECALL_OR_COUNT')
                targets[c]=dict(positive=n,hits=hits,negative=int(negatives.sum()),false_activations=fp,
                    required_positive_owner=owner==donor,per_negative_class=per_negative)
            rows[owner]=dict(rows=len(y),rescue=rescue,break_count=broken,correct=int(new.sum()),
                native_correct=int(old.sum()),collisions=int((activation_count>1).sum()),targets=targets,
                excluded_BASE_FIT_overlaps=excluded_count)
            self.send('gate_CAL_receipt',dict(owner=owner,task=task,result=rows[owner]),classes)
        if birth and total_rescue<=total_break:failures.append('NO_POSITIVE_NET_GAIN')
        if any(not np.isfinite(t) or t>=1 or t<0 for t in self.thresholds.values()):failures.append('INVALID_THRESHOLD')
        outcome='PASS' if not failures else 'REJECT'
        for c in classes:
            if birth:
                self.status[c]='ACTIVE' if not failures else 'REJECTED'
                self.certificates[c]=dict(expert_sha256=hashes[c],threshold=self.thresholds[c],birth_task=task,
                    status=outcome,version=VERSION,input_sha256=self.spec['input_sha256'],
                    scope='empirical observed current CAL owner pools; historical risk UNKNOWN')
            elif self.status[c] in ('ACTIVE','PENDING_NEW_TASK'):
                self.status[c]='SUSPENDED' if failures else 'ACTIVE'
        active_verified=False
        if all(self.status[c]=='ACTIVE' for c in classes):
            # Technical equality only: same current-CAL vectors, no new evidence read,
            # calibration or decision changes after the one-shot acceptance outcome.
            for x,baseline,candidate in inference_pools:
                deployed,_=self.infer(baseline,x)
                if not np.array_equal(deployed,candidate):raise RuntimeError('ACTIVE inference differs from accepted bank')
            active_verified=True
        receipt=dict(task=task,outcome=outcome,birth_acceptance=birth,failures=sorted(set(failures)),
            thresholds=deepcopy(self.thresholds),expert_hashes=hashes,owners=rows,collisions=collisions,
            total_rescue=total_rescue,total_break=total_break,status=deepcopy(self.status),bytes=self.bytes,
            ACTIVE_inference_verified=active_verified,
            evidence_policy='Current CAL, previously available development source; not independent confirmation')
        self.gated_tasks.append(task); self.acceptances.append(receipt)
        if before!=native_fingerprint(models,detectors):raise RuntimeError('Gate mutated native state')
        print(f'CANDLE V4 ACTIVE GATE T{task}: {outcome} {receipt["failures"]}',flush=True)
        return receipt
