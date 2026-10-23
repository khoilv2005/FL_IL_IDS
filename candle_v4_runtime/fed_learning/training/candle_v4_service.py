"""Native multi-receiver V4 development service. No native parameter mutation.

Production protocol v1: current-role evidence, three matched expert arms,
atomic combined-bank acceptance, explicit application-wire resource limits.
"""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import pickle
import time

import numpy as np
import torch

from .candle_v4_bank import base_predictions, fuse, score_matrix
from .candle_v4_coverage import CoverageService
from .candle_v4_preflight import positive_neighbors
from .candle_v4_shadow import digest, native_fingerprint

VERSION = 'candle_v4_native_coverage_full_dev_v1'
ARMS = ('coverage', 'no_aux', 'donor_only')


class FullCoverageService:
    def __init__(self, spec, saved=None, authority=None):
        if spec['version'] != VERSION or spec['seed'] != 42:
            raise ValueError('Unregistered full development protocol')
        self.spec = deepcopy(spec)
        if authority is None:
            from fed_learning.data.denice_clean_roles import CleanRoleData
            authority = CleanRoleData(spec['roles_dir'], source_data_dir=spec['data_dir'])
        self.authority = authority
        if not authority.manifest.get('content_disjoint'):
            raise ValueError('Globally content-disjoint locked roles required')
        self.entries = {}; self.engines = {a: {} for a in ARMS}
        self.status = {a: {} for a in ARMS}; self.thresholds = {a: {} for a in ARMS}
        self.certificates = {a: {} for a in ARMS}
        self.task = None; self.boundary = None; self.gated = []
        self.history = []; self.acceptances = []; self.evaluations = []
        self.bytes = Counter(); self.runtime = 0.; self.cache = {}; self.base_cache = {}
        if saved:
            if saved['version'] != VERSION or saved['spec'] != spec:
                raise ValueError('V4 source/config/roles contract changed on resume')
            for key in ('entries', 'status', 'thresholds', 'certificates', 'task', 'boundary',
                        'gated', 'history', 'acceptances', 'evaluations', 'runtime'):
                setattr(self, key, deepcopy(saved[key]))
            self.bytes = Counter(saved['bytes'])
            for arm in ARMS:
                for key, value in saved['engines'][arm].items():
                    e = self._engine(self.entries[key], arm, value)
                    self.engines[arm][key] = e

    def _engine(self, entry, arm, saved=None):
        child = dict(task=entry['task'], receiver=entry['receiver'], donor=entry['donor'],
                     target=entry['target'], current_classes=list(range(entry['task']*6,min(entry['task']*6+6,34))),
                     input_sha256=self.spec['input_sha256'], fit_paths={}, mode='DRAFT',
                     wire_budget=self.spec['per_expert_bytes'], donor_only_control=arm=='donor_only',
                     branch_names=['joint_class_balanced'])
        engine = CoverageService(child, arm=='coverage', saved)
        # Offline pilots retain their legacy state schema. Full service independently
        # binds algorithm/arm identity in VERSION/spec and the checkpoint envelope.
        return engine

    def state_dict(self):
        result = dict(version=VERSION, spec=self.spec, engines={a:{k:e.state_dict() for k,e in es.items()}
                      for a,es in self.engines.items()}, bytes=dict(self.bytes))
        for key in ('entries','status','thresholds','certificates','task','boundary','gated',
                    'history','acceptances','evaluations','runtime'):
            result[key] = getattr(self,key)
        return deepcopy(result)

    def begin_task(self, task):
        if self.task is not None and task < self.task:raise ValueError('Historical task access')
        if task != self.task:
            self.cache.clear(); self.base_cache.clear()
            for arm in ARMS:
                for key, state in self.status[arm].items():
                    self.engines[arm][key].pending = 0
                    if state == 'ACTIVE':self.status[arm][key] = 'PENDING_NEW_TASK'
            self.task = task

    def counts(self, owner, role):
        row = self.authority.manifest['clients'][str(owner)]['role_class_counts'][role]
        return {int(c):int(n) for c,n in row.items() if self.task*6<=int(c)<min(self.task*6+6,34)}

    def role(self, owner, task, role):
        if task != self.task or role not in ('fit','calibration','base'):
            raise ValueError('Historical raw evidence forbidden')
        key = (owner,role)
        if key not in self.cache:
            x,y,_ = self.authority.client_role(owner,role)
            selected=np.flatnonzero((y>=task*6)&(y<min(task*6+6,34)))
            x=np.asarray(x[selected],np.float32).reshape(-1,39);y=np.asarray(y[selected],np.int64)
            _, indices=np.unique(x,axis=0,return_index=True);indices=np.sort(indices)
            self.cache[key]=(torch.from_numpy(x[indices].copy()),y[indices].copy())
        return self.cache[key]

    def _data(self, owner, clients, engine):
        if owner not in self.base_cache:self.base_cache[owner]=engine.data(clients[owner])
        return self.base_cache[owner]

    def _discover(self, task, clients, active_ids, graph):
        for receiver in sorted(active_ids):
            for c in range(task*6,min(task*6+6,34)):
                if self.counts(receiver,'base').get(c,0):continue
                key=(receiver,c)
                if key in self.entries:continue
                peers=[o for o in positive_neighbors(graph,receiver) if o in active_ids]
                donors=[o for o in peers if all(self.counts(o,r).get(c,0)>=32
                        for r in ('base','fit','calibration'))]
                if not donors:continue
                self.entries[key]=dict(receiver=receiver,target=c,donor=min(donors),task=task,
                                       discovered_round=self.boundary[1])
                for arm in ARMS:
                    self.engines[arm][key]=self._engine(self.entries[key],arm)
                    self.status[arm][key]='DRAFT'

    def after_round(self,task,round_id,models,detectors,clients,active_ids,graph):
        # Tiny CPU MLP updates do not benefit from a many-core BLAS pool.
        # Restore the native learner's setting even on an exceptional path.
        threads=torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            return self._after_round(task,round_id,models,detectors,clients,active_ids,graph)
        finally:torch.set_num_threads(threads)

    def _after_round(self,task,round_id,models,detectors,clients,active_ids,graph):
        if self.task != task:self.begin_task(task)
        if self.boundary and (task,round_id)<=tuple(self.boundary):raise ValueError('Repeated V4 round')
        start=time.perf_counter();before=native_fingerprint(models,detectors)
        self.boundary=[task,round_id]; self._discover(task,clients,active_ids,graph)
        for receiver in sorted(active_ids):
            for peer in positive_neighbors(graph,receiver):
                if peer not in active_ids:continue
                metadata=dict(owner=peer,task=task,round=round_id,counts={r:self.counts(peer,r)
                              for r in ('base','fit','calibration')},input_sha256=self.spec['input_sha256'])
                for arm in ARMS:
                    if self.bytes[arm]+len(pickle.dumps(metadata,protocol=5))<=self.spec['per_arm_bytes']:
                        self._wire(arm,'discovery',metadata)
        receipt=[]
        for key,entry in sorted(self.entries.items()):
            if entry['task']!=task:continue
            receiver,c=key;donor=entry['donor']
            peers=[o for o in positive_neighbors(graph,receiver) if o in active_ids]
            if receiver not in active_ids:continue
            engine=self.engines['coverage'][key]
            data={o:self._data(o,clients,engine) for o in sorted({receiver,*peers})}
            eligible=donor in peers and bool((data[donor][1]==c).any())
            if eligible:
                eligible=all(sum(int((data[o][1]!=c).sum()) for o in owners)>0
                             for owners in ({donor},{receiver,donor}))
            for arm in ARMS:
                e=self.engines[arm][key];e.graph=graph;e.pending+=44
            n=min(88,*(self.engines[a][key].pending for a in ARMS)) if eligible else 0
            updated=0
            for _ in range(n):
                # All arms stop together: matched optimizer/sample budgets.
                limit=48_000*(len(peers)+1)
                if any(self.engines[a][key].bytes+limit>self.spec['per_expert_bytes'] or
                       self.bytes[a]+limit>.9*self.spec['per_arm_bytes'] for a in ARMS):break
                for arm in ARMS:
                    e=self.engines[arm][key];used=e.bytes
                    e.train_step('joint_class_balanced',data,e.steps,task,round_id)
                    self.bytes[arm]+=e.bytes-used;e.steps+=1;e.pending-=1
                updated+=1
            for arm in ARMS:
                e=self.engines[arm][key]
                receipt.append(dict(receiver=receiver,target=c,arm=arm,eligible=eligible,
                                    updates=updated,steps=e.steps,pending=e.pending))
        if before!=native_fingerprint(models,detectors):raise RuntimeError('V4 mutated native state')
        self.runtime+=time.perf_counter()-start
        row=dict(task=task,round=round_id,experts=receipt,bytes=dict(self.bytes),native_sha256=before,
                 native_dense_payload_upper_bound=graph.get('dense_model_transfer_upper_bound_bytes'))
        self.history.append(row)
        print(f'CANDLE V4 T{task} R{round_id}: experts={len(self.entries)} updates={sum(r["updates"] for r in receipt)} bytes={dict(self.bytes)}',flush=True)
        return row

    def _wire(self,arm,kind,message,local=False):
        if local:return message
        raw=pickle.dumps(message,protocol=5)
        if self.bytes[arm]+len(raw)>self.spec['per_arm_bytes']:raise ValueError('Verification wire budget exhausted')
        self.bytes[arm]+=len(raw);self.bytes[arm+':'+kind]+=len(raw)
        return pickle.loads(raw)

    def infer(self,receiver,arm,native,x):
        keys=sorted(k for k,s in self.status[arm].items() if k[0]==receiver and s=='ACTIVE')
        engines={k[1]:self.engines[arm][k] for k in keys};classes=sorted(engines)
        for key in keys:
            cert=self.certificates[arm][key]
            if (cert['threshold']!=self.thresholds[arm][key] or cert['input_sha256']!=self.spec['input_sha256'] or
                cert['sha256']!=digest(engines[key[1]].branches['joint_class_balanced'][0].state_dict())):
                raise ValueError('ACTIVE expert changed')
        return fuse(native,score_matrix(engines,classes,x),[self.thresholds[arm][(receiver,c)] for c in classes],classes)

    def _gate(self,receiver,arm,task,models,detectors,active_ids,graph):
        keys=sorted(k for k,s in self.status[arm].items() if k[0]==receiver and s in ('DRAFT','PENDING_NEW_TASK'))
        if not keys:return dict(receiver=receiver,arm=arm,outcome='NO_BANK')
        for key in keys:
            if self.entries[key]['task']!=task:
                cert=self.certificates[arm][key]
                if (cert['threshold']!=self.thresholds[arm][key] or
                    cert['sha256']!=digest(self.engines[arm][key].branches['joint_class_balanced'][0].state_dict())):
                    raise RuntimeError('Historical expert drift before recertification')
        peers=set(positive_neighbors(graph,receiver))&set(active_ids)
        current=[k for k in keys if self.entries[k]['task']==task]
        owners=sorted({receiver,*peers});failures=[]
        # Metadata-only preflight precedes FIT/CAL access. No replacing donors from outcomes.
        for key in current:
            entry=self.entries[key]
            if entry['donor'] not in peers or self.engines[arm][key].steps==0:failures.append('NO_AUTHORIZED_TRAINING_OR_DONOR')
        classes=[k[1] for k in keys]
        if receiver not in active_ids:failures.append('INACTIVE_RECEIVER')
        for o in owners:
            if sum(n for c,n in self.counts(o,'calibration').items() if c not in classes)<32:
                failures.append(f'OWNER_{o}_NEGATIVE_COUNT')
        if failures:
            for key in keys:self.status[arm][key]='DEFERRED' if key in current else 'SUSPENDED'
            return dict(receiver=receiver,arm=arm,outcome='DEFERRED',failures=sorted(set(failures)),CAL_opened=False)
        engines={k[1]:self.engines[arm][k] for k in keys}
        for key in current:
            x,y=self.role(self.entries[key]['donor'],task,'fit')
            values=score_matrix(engines,[key[1]],x)[:,0][y==key[1]]
            if len(values)<32:failures.append('FIT_POSITIVE_COUNT');continue
            self.thresholds[arm][key]=float(np.sort(values)[::-1][int(np.ceil(.95*len(values)))-1])
        if failures:
            for key in keys:self.status[arm][key]='DEFERRED' if key in current else 'SUSPENDED'
            return dict(receiver=receiver,arm=arm,outcome='DEFERRED',failures=failures,CAL_opened=False)
        thresholds=[self.thresholds[arm][k] for k in keys]
        if any(not 0<=t<1 for t in thresholds):failures.append('INVALID_THRESHOLD')
        lock=dict(receiver=receiver,arm=arm,task=task,thresholds=thresholds,classes=classes,
                  hashes={str(k[1]):digest(engines[k[1]].branches['joint_class_balanced'][0].state_dict()) for k in keys})
        directory=Path(self.spec['output_dir'])/'v4_threshold_locks';directory.mkdir(parents=True,exist_ok=True)
        path=directory/f't{task}_r{receiver}_{arm}.json'
        if path.exists():
            for key in keys:self.status[arm][key]='DEFERRED' if key in current else 'SUSPENDED'
            return dict(receiver=receiver,arm=arm,outcome='DEFERRED_INTERRUPTED_GATE',
                        failures=['LOCK_WITHOUT_COMMITTED_RECEIPT_NO_CAL_RETRY'],CAL_opened=False)
        path.write_text(json.dumps(lock,indent=2),encoding='utf-8')
        # Locked roles are globally disjoint by float32 content (checked by
        # CleanRoleData); no need to rematerialize millions of BASE rows here.
        rescue=broken=collisions=0;rows={}
        from .checkpoint_state import snapshot_denice_state
        reference=dict(weights={k:v.detach().cpu() for k,v in models[receiver].state_dict().items()},
                       algorithm=snapshot_denice_state(models[receiver],detectors[receiver]))
        (reference['algorithm'].get('context_detector') or {}).pop('reference_input_memory',None)
        packet=dict(reference=reference,lock=lock,
                    experts={c:e.branches['joint_class_balanced'][0].state_dict() for c,e in engines.items()})
        estimate=(len(owners)-1)*(len(pickle.dumps(packet,protocol=5))+4096*len(classes))
        if self.bytes[arm]+estimate>self.spec['per_arm_bytes']:
            for key in keys:self.status[arm][key]='DEFERRED' if key in current else 'SUSPENDED'
            return dict(receiver=receiver,arm=arm,outcome='DEFERRED_BUDGET',CAL_opened=False)
        for owner in owners:
            self._wire(arm,'verification',dict(owner=owner,reference=reference,lock=lock,
                experts={c:e.branches['joint_class_balanced'][0].state_dict() for c,e in engines.items()}),owner==receiver)
            x,y=self.role(owner,task,'calibration')
            native=base_predictions(models[receiver],detectors[receiver],x,list(range(min(6*(task+1),34)))) if len(y) else np.empty(0,dtype=int)
            pred,activation=fuse(native,score_matrix(engines,classes,x),thresholds,classes)
            old=native==y;new=pred==y;nb=int((old&~new).sum());nr=int((~old&new).sum())
            broken+=nb;rescue+=nr;collisions+=int((activation>1).sum())
            if nb:failures.append(f'OWNER_{owner}_BREAK')
            if int((~np.isin(y,classes)).sum())<32:failures.append(f'OWNER_{owner}_NEGATIVE_COUNT')
            detail={}
            for key in keys:
                c=key[1];positive=y==c;negative=y!=c
                fp=int(((pred==c)&negative).sum());n=int(positive.sum());hits=int(((pred==c)&positive).sum())
                per_class={int(label):dict(n=int((y==label).sum()),fp=int(((pred==c)&(y==label)).sum()))
                           for label in sorted(set(y.tolist())) if label!=c}
                if any(v['fp']/v['n']>.001 for v in per_class.values()):failures.append(f'OWNER_{owner}_CLASS_{c}_FAR')
                if key in current and owner==self.entries[key]['donor'] and (n<32 or hits/max(n,1)<.95):
                    failures.append(f'CLASS_{c}_RECALL_OR_COUNT')
                detail[c]=dict(positive=n,hits=hits,fp=fp,negative=int(negative.sum()),per_class=per_class)
            rows[owner]=dict(n=len(y),rescue=nr,break_count=nb,targets=detail)
            self._wire(arm,'receipt',dict(owner=owner,result=rows[owner]),owner==receiver)
        if current and rescue<=broken:failures.append('NO_NET_GAIN')
        outcome='REJECT' if failures else 'PASS'
        for key in keys:
            self.status[arm][key]=('REJECTED' if key in current else 'SUSPENDED') if failures else 'ACTIVE'
            if not failures:self.certificates[arm][key]=dict(sha256=lock['hashes'][str(key[1])],task=task,
                threshold=self.thresholds[arm][key],input_sha256=self.spec['input_sha256'],
                scope='Observed current-owner CAL; unchanged historical expert receipts; unobserved historical risk UNKNOWN')
        return dict(receiver=receiver,arm=arm,task=task,outcome=outcome,failures=sorted(set(failures)),
                    CAL_opened=True,rescue=rescue,break_count=broken,collisions=collisions,owners=rows,lock=lock)

    def finish_task(self,task,models,detectors,clients,active_ids,graph):
        if task in self.gated:raise ValueError('Repeated task acceptance')
        before=native_fingerprint(models,detectors)
        for receiver in sorted({k[0] for k in self.entries}):
            for arm in ARMS:
                directory=Path(self.spec['output_dir'])/'v4_gate_journal';directory.mkdir(parents=True,exist_ok=True)
                path=directory/f't{task}_r{receiver}_{arm}.pt'
                keys=[k for k in self.entries if k[0]==receiver]
                hashes={k:digest(self.engines[arm][k].branches['joint_class_balanced'][0].state_dict()) for k in keys}
                if path.exists():
                    journal=torch.load(path,map_location='cpu',weights_only=False)
                    if journal['native_sha256']!=before or journal['engine_hashes']!=hashes or journal['spec_sha256']!=digest(self.spec):
                        raise RuntimeError('Gate journal does not match resumed checkpoint')
                    for attr in ('status','thresholds','certificates'):
                        getattr(self,attr)[arm].update(deepcopy(journal[attr]))
                    self.bytes.update(journal['bytes_delta']);receipt=journal['receipt']
                else:
                    previous=self.bytes.copy();receipt=self._gate(receiver,arm,task,models,detectors,active_ids,graph)
                    journal=dict(native_sha256=before,engine_hashes=hashes,spec_sha256=digest(self.spec),
                        bytes_delta=dict(self.bytes-previous),receipt=receipt)
                    for attr in ('status','thresholds','certificates'):
                        journal[attr]={k:deepcopy(v) for k,v in getattr(self,attr)[arm].items() if k[0]==receiver}
                    temporary=path.with_suffix('.tmp');torch.save(journal,temporary);temporary.replace(path)
                self.acceptances.append(receipt)
        self.gated.append(task)
        if before!=native_fingerprint(models,detectors):raise RuntimeError('Gate mutated native state')
        self._evaluate(task,models,detectors)
        self.cache.clear();self.base_cache.clear()
        self.write_report()
        return dict(task=task,status_counts={a:dict(Counter(self.status[a].values())) for a in ARMS},bytes=dict(self.bytes))

    def _evaluate(self,task,models,detectors):
        """Report-only cumulative validation; each local predictor on its own shard."""
        from sklearn.metrics import accuracy_score,f1_score,recall_score
        names=('V1',*ARMS); truth=[];predictions={a:[] for a in names};per_client={}
        times=Counter();labels=list(range(min(6*(task+1),34)))
        for receiver in sorted(models):
            x,y,_=self.authority.client_role(receiver,'validation')
            selected=np.flatnonzero(y<min(6*(task+1),34));y=y[selected]
            x=torch.from_numpy(np.asarray(x[selected],np.float32).reshape(-1,39))
            if not len(y):continue
            start=time.perf_counter();native=base_predictions(models[receiver],detectors[receiver],x,labels)
            if torch.cuda.is_available():torch.cuda.synchronize()
            times['V1']+=time.perf_counter()-start
            local={'V1':native}
            for arm in ARMS:
                start=time.perf_counter();local[arm]=self.infer(receiver,arm,native,x)[0]
                times[arm]+=time.perf_counter()-start
            truth.extend(y.tolist());per_client[receiver]={}
            for name in names:
                predictions[name].extend(local[name].tolist())
                per_client[receiver][name]=dict(n=len(y),accuracy=float(accuracy_score(y,local[name])))
        y=np.asarray(truth);result={}
        for name in names:
            p=np.asarray(predictions[name]);correct=p==y;base=np.asarray(predictions['V1'])==y
            result[name]=dict(accuracy=float(accuracy_score(y,p)),macro_f1=float(f1_score(y,p,labels=labels,average='macro',zero_division=0)),
                per_class_recall=dict(zip(labels,recall_score(y,p,labels=labels,average=None,zero_division=0).tolist())),
                rescue=int((~base&correct).sum()),break_count=int((base&~correct).sum()),
                extra_inference_seconds=times[name] if name!='V1' else 0.,
                per_task_accuracy={t:float(accuracy_score(y[y//6==t],p[y//6==t])) for t in range(task+1) if (y//6==t).any()})
            result[name]['per_class_false_activation']={c:dict(negative=int((y!=c).sum()),
                fp=int(((p==c)&(y!=c)).sum())) for c in labels}
            old=[]
            for t in range(task):
                previous=[r['metrics'][name]['per_task_accuracy'].get(t,r['metrics'][name]['per_task_accuracy'].get(str(t))) for r in self.evaluations]
                previous=[v for v in previous if v is not None]
                if previous and t in result[name]['per_task_accuracy']:old.append(max(previous)-result[name]['per_task_accuracy'][t])
            result[name]['average_forgetting']=float(np.mean(old)) if old else None
        self.evaluations.append(dict(task=task,role='validation',rows=len(y),metrics=result,per_client=per_client,
                                     native_inference_seconds=times['V1']))

    def write_report(self):
        path=Path(self.spec['output_dir']);path.mkdir(parents=True,exist_ok=True)
        report=dict(version=VERSION,development_only=True,TEST_opened=False,bytes=dict(self.bytes),
                    expert_runtime_seconds=self.runtime,entries=[v for v in self.entries.values()],
                    status_counts={a:dict(Counter(self.status[a].values())) for a in ARMS},
                    acceptances=self.acceptances,evaluations=self.evaluations,
                    communication_note='Serialized application-wire simulation; excludes native transport, reported separately by native runner')
        report['training_history']=self.history
        report['expert_summary']=[dict(**self.entries[key],arm=a,status=self.status[a][key],steps=e.steps,
            training_bytes=e.bytes,model_bytes=sum(p.numel()*p.element_size() for p in e.branches['joint_class_balanced'][0].parameters()),
            sha256=digest(e.branches['joint_class_balanced'][0].state_dict())) for a in ARMS for key,e in self.engines[a].items()]
        report['native_dense_payload_upper_bound_bytes']=sum(r.get('native_dense_payload_upper_bound') or 0 for r in self.history)
        report['native_communication_missing_rounds']=sum(r.get('native_dense_payload_upper_bound') is None for r in self.history)
        report['total_payload_estimate_by_arm']={a:(report['native_dense_payload_upper_bound_bytes']+self.bytes[a]
            if not report['native_communication_missing_rounds'] else None) for a in ARMS}
        (path/'candle_v4_development_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
