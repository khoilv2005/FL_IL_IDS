"""Full current-task TRAIN with provisional training-only expert activation."""
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from .candle_v4_service import FullCoverageService
from .candle_v4_preflight import positive_neighbors
from .candle_v4_shadow import digest

VERSION='candle_v4_full_train_provisional_v3'


def validate_original_config(config):
    required=dict(denice_clean_roles_dir=None,candle_v4_original_train=True,denice_router_mode='binary_cosine',
        denice_clustering_mode='paper',denice_similarity_threshold=.8,denice_validation_fraction=0.0,
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
                spec.get('fixed_threshold')!=.5 or spec.get('activation_policy')!='training_only_unverified'):
            raise ValueError('Original TRAIN protocol requires one coverage bank or native-only baseline')
        self.sampling_cache={};self.count_cache={}
        super().__init__(spec,saved,authority=SimpleNamespace(manifest={}))
        self.metadata_roles=('train',)
        self.current_clients={}

    def begin_task(self,task):
        if self.task is not None and task<self.task:raise ValueError('Historical task access')
        if task!=self.task:
            self.current_clients={};self.sampling_cache.clear();self.count_cache.clear()
            self.cache.clear();self.base_cache.clear()
            for arm in self.arms:
                for engine in self.engines[arm].values():engine.pending=0
            # Old ACTIVE experts remain available, with their frozen UNVERIFIED receipts.
            self.task=task

    def _engine(self,entry,arm,saved=None):
        engine=super()._engine(entry,arm,saved);engine.sampling_cache=self.sampling_cache
        return engine

    def bind_clients(self,task,clients,active_ids):
        if task!=self.task:self.begin_task(task)
        self.current_clients={o:clients[o] for o in active_ids}
        for owner,client in self.current_clients.items():
            for attribute in ('y_train',):
                labels=getattr(client,attribute).detach().cpu()
                if len(labels) and not bool(((labels>=task*6)&(labels<min(task*6+6,34))).all()):
                    raise ValueError(f'Historical labels exposed to V4 {attribute}, owner {owner}')

    def counts(self,owner,role):
        if role not in self.metadata_roles:raise ValueError('No BASE/FIT/CAL role API in original TRAIN mode')
        if owner not in self.current_clients:return {}
        attr='y_train'
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
                if self.counts(receiver,'train').get(c,0):continue
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

    def _training_proof(self,key,arm):
        entry=self.entries[key];engine=self.engines[arm][key];updates=0;boundaries=[]
        for row in self.history:
            for item in row['experts']:
                if (row['task']==entry['task'] and item['receiver']==key[0] and
                        item['target']==key[1] and item['arm']==arm and item['updates']>0):
                    if not item['eligible'] or item.get('donor')!=entry['donor'] or not row.get('graph_sha256'):
                        return dict(valid=False,reason='UNAUTHORIZED_TRAINING')
                    updates+=item['updates'];boundaries.append([row['task'],row['round']])
        valid=(updates>0 and updates==engine.steps and engine.spec['task']==entry['task'] and
               engine.spec['receiver']==key[0] and engine.spec['target']==key[1] and
               engine.spec['donor']==entry['donor'] and engine.spec['input_sha256']==self.spec['input_sha256'])
        return dict(valid=valid,updates=updates,steps=engine.steps,authorized_rounds=boundaries,
                    provenance_sha256=digest(dict(entry=entry,boundaries=boundaries,updates=updates)),
                    reason=None if valid else 'NO_UPDATES_OR_INCONSISTENT_PROVENANCE')

    def _gate(self,receiver,arm,task,models,detectors,active_ids,graph):
        entries={};activated=[];deferred=[]
        for key,entry in sorted(self.entries.items()):
            if key[0]!=receiver:continue
            model=self.engines[arm][key].branches['joint_class_balanced'][0]
            sha=digest(model.state_dict())
            if entry['task']!=task:
                if self.status[arm][key]=='ACTIVE':
                    cert=self.certificates[arm][key]
                    if cert['sha256']!=sha or cert['quality_status']!='UNVERIFIED':
                        raise RuntimeError('Frozen historical expert or receipt changed')
                entries[key[1]]=dict(status=self.status[arm][key],frozen=True,quality_status='UNVERIFIED')
                continue
            if self.status[arm][key]!='DRAFT':raise ValueError('Repeated task activation')
            proof=self._training_proof(key,arm)
            finite=all(bool(torch.isfinite(p).all()) for p in model.parameters())
            if not proof['valid'] or not finite:
                self.status[arm][key]='DEFERRED';deferred.append(key[1])
                entries[key[1]]=dict(status='DEFERRED',training_proof=proof,finite_weights=finite)
                continue
            self.thresholds[arm][key]=.5;self.status[arm][key]='ACTIVE';activated.append(key[1])
            self.certificates[arm][key]=dict(version=VERSION,sha256=sha,threshold=.5,
                input_sha256=self.spec['input_sha256'],birth_task=entry['task'],activation_task=task,
                donor=entry['donor'],training_proof=proof,quality_status='UNVERIFIED',
                acceptance_type='training_only_unverified',statistical_safety_certificate=False,
                quality_metrics_status='NOT_MEASURED',recall=None,far=None,break_count=None,net_rescue=None,
                scope='Training-qualified provisional activation; no validation or quality acceptance')
            entries[key[1]]=dict(status='ACTIVE',quality_status='UNVERIFIED',training_proof=proof,finite_weights=True)
        outcome='ACTIVE_UNVERIFIED' if activated else ('DEFERRED' if deferred else 'RETAINED_UNVERIFIED')
        return dict(task=task,receiver=receiver,arm=arm,outcome=outcome,
                    acceptance_type='training_only_unverified',quality_status='UNVERIFIED',
                    quality_metrics_status='NOT_MEASURED',activated=activated,deferred=deferred,experts=entries,
                    validation_opened=False,CAL_opened=False,TEST_opened=False)

    def infer(self,receiver,arm,native,x):
        for key,status in self.status[arm].items():
            if key[0]==receiver and status=='ACTIVE':
                cert=self.certificates[arm][key]
                if (cert['version']!=VERSION or cert['quality_status']!='UNVERIFIED' or
                        cert['acceptance_type']!='training_only_unverified'):
                    raise ValueError('Wrong provisional activation receipt')
        return super().infer(receiver,arm,native,x)

    def evaluation_data(self,*args):
        raise ValueError('No validation access under full-TRAIN protocol')

    def _evaluate(self,task,models,detectors):
        # Placeholders record completion, never fabricated validation measurements.
        self.evaluations.append(dict(task=task,role='NONE',status='NOT_MEASURED',rows=0,
            metrics={name:dict(accuracy=None,macro_f1=None,per_task_accuracy={},average_forgetting=None,
                              evidence_status='NOT_MEASURED') for name in ('V1',*self.arms)},per_client={}))

    def finish_task(self,task,models,detectors,clients,active_ids,graph):
        self.bind_clients(task,clients,active_ids)
        result=super().finish_task(task,models,detectors,clients,active_ids,graph)
        self.current_clients={};self.sampling_cache.clear();self.count_cache.clear()
        return result

    def write_report(self):
        super().write_report()
        path=Path(self.spec['output_dir'])/'candle_v4_development_report.json'
        report=json.loads(path.read_text(encoding='utf-8'))
        report.update(data_protocol=VERSION,train_fraction=1.0,local_validation_fraction=0.0,
                      validation_metrics_status='NOT_MEASURED',activation_policy='training_only_unverified',
                      quality_status='UNVERIFIED',statistical_safety_certificate=False,
                      external_roles_created=False,shadow_native_models=0,expert_arms=list(self.arms),
                      activation_receipts=[dict(receiver=k[0],target=k[1],arm=arm,certificate=cert)
                          for arm in self.arms for k,cert in self.certificates[arm].items()],
                      baseline='Native V1 counterfactual on the same full current-task TRAIN',
                      forgetting_status='NOT_MEASURED; no intermediate TEST or historical validation')
        path.write_text(json.dumps(report,indent=2),encoding='utf-8')
