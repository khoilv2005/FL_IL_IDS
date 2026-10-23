"""Original TRAIN + existing native local validation; no external data roles."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import torch

from .candle_v4_service import FullCoverageService
from .candle_v4_bank import base_predictions,score_matrix,fuse
from .candle_v4_preflight import positive_neighbors
from .candle_v4_shadow import digest

VERSION='candle_v4_original_train_empirical_gate_v2'


def validate_original_config(config):
    required=dict(denice_clean_roles_dir=None,candle_v4_original_train=True,denice_router_mode='binary_cosine',
        denice_clustering_mode='paper',denice_similarity_threshold=.8,denice_validation_fraction=.1,
        denice_post_task_eval=False,denice_eval_final_round=False,denice_eval_terminal_state_only=True,
        rounds_per_task=20,task_end=5,num_clients=100,seed=42)
    for name,value in required.items():
        if config.get(name)!=value:raise ValueError(f'Original TRAIN V4 requires {name}={value!r}')
    if config.get('denice_max_train_samples_per_client') or config.get('denice_max_batches_per_epoch'):
        raise ValueError('No pilot native training sample/step cap in original TRAIN full run')
    if config['candle_v4_full_service']['version']!=VERSION:raise ValueError('Wrong data-service version')


class OriginalTrainService(FullCoverageService):
    def __init__(self,spec,saved=None):
        if (spec.get('version')!=VERSION or spec.get('arms') not in (['coverage'],[]) or
                spec.get('fixed_threshold')!=.5):
            raise ValueError('Original TRAIN protocol requires one coverage bank or native-only baseline')
        self.sampling_cache={};self.count_cache={}
        super().__init__(spec,saved,authority=SimpleNamespace(manifest={}))
        self.metadata_roles=('train','native_validation')
        self.current_clients={};self.eligible_validation={}

    def begin_task(self,task):
        if task!=self.task:
            self.current_clients={};self.eligible_validation={};self.sampling_cache.clear();self.count_cache.clear()
        super().begin_task(task)

    def _engine(self,entry,arm,saved=None):
        engine=super()._engine(entry,arm,saved);engine.sampling_cache=self.sampling_cache
        return engine

    def bind_clients(self,task,clients,active_ids):
        if task!=self.task:self.begin_task(task)
        self.current_clients={o:clients[o] for o in active_ids}
        for owner,client in self.current_clients.items():
            for attribute in ('y_train','y_validation'):
                labels=getattr(client,attribute).detach().cpu()
                if len(labels) and not bool(((labels>=task*6)&(labels<min(task*6+6,34))).all()):
                    raise ValueError(f'Historical labels exposed to V4 {attribute}, owner {owner}')

    def counts(self,owner,role):
        if role not in self.metadata_roles:raise ValueError('No BASE/FIT/CAL role API in original TRAIN mode')
        if owner not in self.current_clients:return {}
        attr='y_train' if role=='train' else 'y_validation'
        key=(owner,role)
        if key not in self.count_cache:
            labels,counts=torch.unique(getattr(self.current_clients[owner],attr).detach().cpu(),return_counts=True)
            self.count_cache[key]=dict(zip(labels.tolist(),counts.tolist()))
        return self.count_cache[key]

    def role(self,*args,**kwargs):
        raise ValueError('External/historical data roles are disabled')

    def _data(self,owner,clients,engine):
        # Entire current native training view, no pilot 4096/class cap or role repartition.
        if owner not in self.base_cache:
            c=self.current_clients[owner]
            self.base_cache[owner]=(c.X_train.detach().cpu().reshape(-1,39),c.y_train.detach().cpu())
        return self.base_cache[owner]

    def _discover(self,task,clients,active_ids,graph):
        if not self.arms:return
        for receiver in sorted(active_ids):
            for c in range(task*6,min(task*6+6,34)):
                if self.counts(receiver,'train').get(c,0) or self.counts(receiver,'native_validation').get(c,0):continue
                key=(receiver,c)
                if key in self.entries:continue
                donors=[o for o in positive_neighbors(graph,receiver) if o in active_ids and self.counts(o,'train').get(c,0)>=32]
                if not donors:continue
                self.entries[key]=dict(receiver=receiver,target=c,donor=min(donors),task=task,discovered_round=self.boundary[1])
                for arm in self.arms:
                    self.engines[arm][key]=self._engine(self.entries[key],arm);self.status[arm][key]='DRAFT'

    def after_round(self,task,round_id,models,detectors,clients,active_ids,graph):
        self.bind_clients(task,clients,active_ids)
        return super().after_round(task,round_id,models,detectors,clients,active_ids,graph)

    def prepare_validation(self):
        """Exclude duplicate validation contents and training overlaps, without making a new split."""
        candidates=set();pools={}
        for owner,client in self.current_clients.items():
            x=client.X_validation.detach().cpu().reshape(-1,39);y=client.y_validation.detach().cpu().numpy()
            hashes=[hashlib.sha256(row.tobytes()).digest() for row in x.numpy()]
            pools[owner]=(x,y,hashes);candidates.update(hashes)
        overlaps=set()
        for client in self.current_clients.values():
            for row in client.X_train.detach().cpu().reshape(-1,39).numpy():
                h=hashlib.sha256(row.tobytes()).digest()
                if h in candidates:overlaps.add(h)
        self.eligible_validation={}
        for owner,(x,y,hashes) in pools.items():
            seen=set();keep=[]
            for i,h in enumerate(hashes):
                if h not in seen and h not in overlaps:keep.append(i);seen.add(h)
            self.eligible_validation[owner]=(x[keep],y[keep])

    def _gate(self,receiver,arm,task,models,detectors,active_ids,graph):
        keys=sorted(k for k,s in self.status[arm].items() if k[0]==receiver and s in ('DRAFT','PENDING_NEW_TASK'))
        if not keys:return dict(receiver=receiver,arm=arm,outcome='NO_BANK')
        current=[k for k in keys if self.entries[k]['task']==task]
        peers=set(positive_neighbors(graph,receiver))&set(active_ids);owners=sorted({receiver,*peers})
        failures=[];classes=[k[1] for k in keys]
        if receiver not in active_ids:failures.append('INACTIVE_RECEIVER')
        for k in keys:
            if k not in current:
                if digest(self.engines[arm][k].branches['joint_class_balanced'][0].state_dict())!=self.certificates[arm][k]['sha256']:
                    raise RuntimeError('Frozen historical expert mutated')
            elif self.entries[k]['donor'] not in peers or not self.engines[arm][k].steps:
                failures.append('NO_CURRENT_AUTHORIZED_DONOR_OR_UPDATES')
        evidence_counts={}
        seen_classes=set(range(min(6*(task+1),34)))
        for k in keys:
            c=k[1];negative_counts={o:int((self.eligible_validation.get(o,(None,np.empty(0,dtype=int)))[1]!=c).sum()) for o in owners}
            observed={int(label) for o in owners for label in self.eligible_validation.get(o,(None,np.empty(0,dtype=int)))[1] if label!=c}
            evidence_counts[c]=dict(negative_by_owner=negative_counts,total_negative=sum(negative_counts.values()),
                positive_by_owner={o:int((self.eligible_validation.get(o,(None,np.empty(0,dtype=int)))[1]==c).sum()) for o in owners},
                observed_negative_classes=sorted(observed),unobserved_negative_classes=sorted(seen_classes-{c}-observed),
                future_unseen_classes=list(range(min(6*(task+1),34),34)))
            if not sum(negative_counts.values()):failures.append(f'CLASS_{c}_NO_NEGATIVE_VALIDATION')
        for k in current:
            _,y=self.eligible_validation.get(self.entries[k]['donor'],(None,np.empty(0,dtype=int)))
            n=int((y==k[1]).sum());evidence_counts[k[1]]['required_donor_positive']=n
            if not n:failures.append(f'CLASS_{k[1]}_NO_POSITIVE_VALIDATION')
        if failures:
            for k in keys:self.status[arm][k]='DEFERRED' if k in current else 'SUSPENDED'
            return dict(receiver=receiver,arm=arm,outcome='DEFERRED',failures=sorted(set(failures)),
                        evidence='existing native local validation',evidence_counts=evidence_counts,
                        independent_confirmation=False,CAL_opened=False)
        engines={k[1]:self.engines[arm][k] for k in keys}
        for k in current:self.thresholds[arm][k]=.5
        thresholds=[self.thresholds[arm][k] for k in keys]
        lock=dict(task=task,receiver=receiver,thresholds=thresholds,classes=classes,rule='fixed .5; no threshold fitting',
                  hashes={str(k[1]):digest(engines[k[1]].branches['joint_class_balanced'][0].state_dict()) for k in keys})
        folder=Path(self.spec['output_dir'])/'v4_threshold_locks';folder.mkdir(parents=True,exist_ok=True)
        path=folder/f't{task}_r{receiver}_{arm}.json'
        if path.exists():
            for k in keys:self.status[arm][k]='DEFERRED' if k in current else 'SUSPENDED'
            return dict(receiver=receiver,arm=arm,outcome='DEFERRED_INTERRUPTED_GATE',CAL_opened=False)
        path.write_text(json.dumps(lock,indent=2),encoding='utf-8')
        from .checkpoint_state import snapshot_denice_state
        reference=dict(weights={k:v.detach().cpu() for k,v in models[receiver].state_dict().items()},
                       algorithm=snapshot_denice_state(models[receiver],detectors[receiver]))
        (reference['algorithm'].get('context_detector') or {}).pop('reference_input_memory',None)
        states={c:e.branches['joint_class_balanced'][0].state_dict() for c,e in engines.items()}
        estimated=(len(owners)-1)*(len(pickle.dumps(dict(reference=reference,lock=lock,experts=states),protocol=5))+8192*len(classes))
        if self.bytes[arm]+estimated>self.spec['per_arm_bytes']:
            for k in keys:self.status[arm][k]='DEFERRED' if k in current else 'SUSPENDED'
            return dict(receiver=receiver,arm=arm,outcome='DEFERRED_BUDGET',CAL_opened=False)
        rows={};rescue=broken=collisions=0
        for owner in owners:
            self._wire(arm,'native_validation_reference',dict(reference=reference,lock=lock,
                experts=states),owner==receiver)
            x,y=self.eligible_validation[owner]
            native=(base_predictions(models[receiver],detectors[receiver],x,list(range(min(6*(task+1),34))))
                    if len(y) else np.empty(0,dtype=np.int64))
            pred,activation=fuse(native,score_matrix(engines,classes,x),thresholds,classes)
            old=native==y;new=pred==y;nb=int((old&~new).sum());nr=int((~old&new).sum())
            rescue+=nr;broken+=nb;collisions+=int((activation>1).sum());targets={}
            if nb:failures.append(f'OWNER_{owner}_BREAK')
            for k in keys:
                c=k[1];positive=y==c;n=int(positive.sum());hits=int(((pred==c)&positive).sum())
                per_class={int(label):dict(n=int((y==label).sum()),fp=int(((pred==c)&(y==label)).sum()))
                           for label in sorted(set(y.tolist())) if label!=c}
                if any(v['fp']/v['n']>.001 for v in per_class.values()):failures.append(f'OWNER_{owner}_CLASS_{c}_FAR')
                if k in current and owner==self.entries[k]['donor'] and hits/max(1,n)<.95:failures.append(f'CLASS_{c}_RECALL')
                negative=int((y!=c).sum());fp=int(((pred==c)&(y!=c)).sum())
                targets[c]=dict(positive=n,hits=hits,recall=hits/n if n else None,
                    negative=negative,false_activations=fp,observed_far=fp/negative if negative else None,
                    negative_evidence_status='OBSERVED' if negative else 'UNKNOWN_NO_SAMPLES',
                    per_negative_class=per_class,
                    unobserved_negative_classes=sorted(seen_classes-{c}-set(per_class)),
                    future_unseen_classes=list(range(min(6*(task+1),34),34)))
            rows[owner]=dict(rows=len(y),rescue=nr,break_count=nb,targets=targets)
            self._wire(arm,'native_validation_receipt',dict(owner=owner,result=rows[owner]),owner==receiver)
        if current and rescue<=broken:failures.append('NO_NET_GAIN')
        outcome='REJECT' if failures else 'PASS'
        for k in keys:
            self.status[arm][k]=('REJECTED' if k in current else 'SUSPENDED') if failures else 'ACTIVE'
            if not failures:self.certificates[arm][k]=dict(sha256=lock['hashes'][str(k[1])],
                threshold=self.thresholds[arm][k],input_sha256=self.spec['input_sha256'],task=task,
                acceptance_type='empirical_validation_acceptance',statistical_safety_certificate=False,
                evidence_counts=evidence_counts[k[1]],
                unobserved_negative_classes=sorted(seen_classes-{k[1]}-{int(label) for owner in rows.values() for label in owner['targets'][k[1]]['per_negative_class']}),
                future_unseen_classes=list(range(min(6*(task+1),34),34)),
                scope='Observed native validation only; zero-sample owner/classes UNKNOWN; not statistical safety certification')
        return dict(task=task,receiver=receiver,arm=arm,outcome=outcome,failures=sorted(set(failures)),
                    rescue=rescue,break_count=broken,collisions=collisions,owners=rows,lock=lock,
                    evidence='existing native local validation',evidence_counts=evidence_counts,
                    acceptance_type='empirical_validation_acceptance',statistical_safety_certificate=False,
                    independent_confirmation=False,CAL_opened=False)

    def evaluation_data(self,receiver,task):
        if task!=self.task:raise ValueError('Historical validation access forbidden')
        x,y=self.eligible_validation.get(receiver,(torch.empty(0,39),np.empty(0,dtype=int)))
        return x.numpy(),y,np.arange(len(y))

    def evaluation_metric_labels(self,task):
        return list(range(task*6,min(task*6+6,34)))

    def finish_task(self,task,models,detectors,clients,active_ids,graph):
        self.bind_clients(task,clients,active_ids);self.prepare_validation()
        result=super().finish_task(task,models,detectors,clients,active_ids,graph)
        self.eligible_validation.clear();self.current_clients={};self.sampling_cache.clear();self.count_cache.clear()
        return result

    def write_report(self):
        super().write_report()
        path=Path(self.spec['output_dir'])/'candle_v4_development_report.json'
        report=json.loads(path.read_text(encoding='utf-8'))
        report.update(data_protocol=VERSION,external_roles_created=False,shadow_native_models=0,
                      acceptance_type='empirical_validation_acceptance',statistical_safety_certificate=False,
                      validation_count_policy='No 32-sample cutoff; nonempty donor positives and pooled negatives per expert',
                      expert_arms=list(self.arms),validation_scope='current task native local validation only',
                      baseline='native V1 counterfactual from same original-TRAIN backbone; no second model',
                      forgetting_status='Not measured: no historical validation replay or intermediate TEST')
        for row in report['evaluations']:row['role']='native_local_validation_current_task'
        path.write_text(json.dumps(report,indent=2),encoding='utf-8')
