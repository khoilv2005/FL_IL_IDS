"""Content-disjoint data-role views for clean DeNICE backbone training."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from .incremental_loader import IncrementalDataLoader


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


class CleanRoleData:
    def __init__(self, root, source_data_dir=None):
        self.root=Path(root)
        self.manifest=json.loads((self.root/'role_manifest.json').read_text(encoding='utf-8'))
        lock=json.loads((self.root/'role_lock.json').read_text(encoding='utf-8'))
        if not self.manifest.get('completed') or file_sha256(self.root/'role_manifest.json')!=lock['manifest_sha256']:
            raise ValueError('Clean role preparation is incomplete or manifest changed')
        if not self.manifest.get('content_disjoint'):
            raise ValueError('Content-disjoint roles required')
        # Relocated Kaggle mounts retain the locked manifest and checksums.
        self.source=Path(source_data_dir or self.manifest['source_data_dir']); self.verified=set()
        if file_sha256(self.source/'metadata.json')!=self.manifest['metadata_sha256']:
            raise ValueError('Dataset metadata/allocation changed after role locking')

    def client_role(self,cid,role):
        if role not in ('base','calibration','fit','validation'):raise ValueError('Unknown data role')
        entry=self.manifest['clients'][str(cid)]
        source=self.source/f'client_{cid}_train.npz'; index=self.root/'indices'/f'client_{cid}.npz'
        if cid not in self.verified:
            if file_sha256(source)!=entry['source_sha256'] or file_sha256(index)!=entry['index_sha256']:
                raise ValueError(f'Client {cid}: original train data or clean indices changed')
            self.verified.add(cid)
        with np.load(index,allow_pickle=False) as split:rows=split[role]
        with np.load(source,allow_pickle=False) as data:
            x=data['X_train']; y=data['y_train']
            if len(y)!=entry['source_rows']:raise ValueError('Original client row coordinates changed')
            return np.asarray(x[rows],dtype=np.float32),np.asarray(y[rows],dtype=np.int64),rows


class CleanRoleIncrementalDataLoader(IncrementalDataLoader):
    """Only BASE enters backbone clients; evaluation role is explicit."""
    def __init__(self,root,validation_max_samples=50000,evaluation_role='validation',source_data_dir=None):
        self.roles=CleanRoleData(root,source_data_dir=source_data_dir)
        super().__init__(str(self.roles.source))
        self.validation_max_samples=int(validation_max_samples)
        if self.validation_max_samples<34:raise ValueError('Validation cap must allow every class')
        self._validation=None
        if evaluation_role not in ('validation','test'):raise ValueError('Evaluation role must be validation or test')
        self.evaluation_role=evaluation_role
        if evaluation_role=='test' and not self.test_file.is_file():raise FileNotFoundError(self.test_file)

    @property
    def input_shape(self):
        return tuple(self.roles.manifest['input_shape'])

    def get_client_full_data(self,cid):
        x,y,_=self.roles.client_role(cid,'base')
        return torch.from_numpy(x),torch.from_numpy(y)

    def get_client_data(self,cid,task_id):
        x,y=self.get_client_full_data(cid)
        mask=torch.isin(y,torch.as_tensor(self.get_task_classes(task_id),dtype=torch.long))
        return x[mask],y[mask]

    def _validation_data(self):
        if self._validation is None:
            # Bound memory before concatenating shards; each class receives a
            # fixed cap per client. Final global subsampling uses labels only
            # for declared benchmark stratification, not inference.
            labels=sorted({c for values in self.task_classes.values() for c in values})
            clients=self.get_all_client_ids(); chunks_x=[]; chunks_y=[]
            for cid in clients:
                x,y,_=self.roles.client_role(cid,'validation'); selected=[]
                cap=max(1,(self.validation_max_samples+len(labels)-1)//len(labels))
                for label in labels:selected.extend(np.flatnonzero(y==label)[:cap].tolist())
                if selected:chunks_x.append(x[selected]);chunks_y.append(y[selected])
            if not chunks_y:raise ValueError('No clean validation data')
            x=np.concatenate(chunks_x); y=np.concatenate(chunks_y)
            selected=[]; rng=np.random.default_rng(self.roles.manifest['split_seed']+523687)
            for label in labels:
                rows=np.flatnonzero(y==label)
                if not len(rows):
                    print(f'Clean validation missing class {label}; report-only, continuing.',flush=True)
                    continue
                quota=self.validation_max_samples//len(labels)+(labels.index(label)<self.validation_max_samples%len(labels))
                selected.extend(rng.permutation(rows)[:quota].tolist())
            selected=np.asarray(selected,dtype=np.int64)
            self._validation=(torch.from_numpy(x[selected]),torch.from_numpy(y[selected]))
        return self._validation

    def get_test_data(self,task_id,cumulative=True):
        if self.evaluation_role=='test':
            labels=[c for t,values in self.task_classes.items() if (t<=task_id if cumulative else t==task_id) for c in values]
            expected=sorted({c for values in self.task_classes.values() for c in values})
            if sorted(set(labels))==expected:
                # Return the original full tensor without creating another
                # multi-million-row feature copy through a boolean mask.
                x,y=super().get_full_test_data()
                present=sorted(torch.unique(y).tolist())
                if present!=expected:raise ValueError(f'Full test class coverage changed: present={present}, expected={expected}')
                return x,y
            return super().get_test_data(task_id,cumulative)
        x,y=self._validation_data()
        labels=[c for t,values in self.task_classes.items() if (t<=task_id if cumulative else t==task_id) for c in values]
        mask=torch.isin(y,torch.as_tensor(labels,dtype=torch.long))
        return x[mask],y[mask]

    def get_full_test_data(self):
        if self.evaluation_role=='test':return super().get_full_test_data()
        return self._validation_data()


def validate_similarity_threshold(value):
    """Xi is configurable; reject invalid values before preparing data roles."""
    try:
        threshold=float(value)
    except (TypeError,ValueError) as exc:
        raise ValueError('denice_similarity_threshold must be a finite number in [0, 1]') from exc
    if isinstance(value,bool) or not math.isfinite(threshold) or not 0<=threshold<=1:
        raise ValueError('denice_similarity_threshold must be a finite number in [0, 1]')
    return threshold


def validate_clean_training_config(config):
    """Fail closed if a launcher tries to bypass the fixed clean protocol."""
    config['denice_similarity_threshold']=validate_similarity_threshold(config.get('denice_similarity_threshold'))
    protocol=config.get('denice_clean_protocol','cgofed')
    if protocol not in ('cgofed','legacy_multiclass'):raise ValueError('Unknown clean training protocol')
    method='legacy' if protocol=='legacy_multiclass' else 'cgofed'
    required={'denice_clustering_mode':'paper',
              'denice_cgofed_peer_projection':False,'denice_amp_enabled':True,
              'denice_cl_method':method,'denice_require_label_overlap':True,
              'denice_collab_use_context_edges':True}
    integration_smoke = bool(config.get('appliance_integration_smoke', False))
    if integration_smoke:
        if (not config.get('appliance_enabled') or protocol != 'legacy_multiclass' or
                config.get('denice_evaluation_data_role') != 'validation' or
                config.get('denice_post_task_eval') or config.get('denice_eval_final_round')):
            raise ValueError('Bounded APPLIANCE smoke requires legacy/current roles and unopened test')
        required['denice_amp_enabled'] = False  # explicit CPU FP32 integration smoke
    for key,value in required.items():
        if config.get(key)!=value:raise ValueError(f'Clean training requires {key}={value!r}')
    if protocol=='legacy_multiclass':
        if config.get('denice_cme_after_each_task') or config.get('denice_replay_capacity',0):
            raise ValueError('Legacy self-only protocol disables CME and replay')
        if config.get('denice_plasticity_enabled') or config.get('denice_continual_width',0):
            raise ValueError('Legacy protocol disables experimental plasticity and residual heads')
    if not config.get('denice_clean_roles_dir'):raise ValueError('Clean data-role manifest required')
    start=int(config.get('task_start',0))
    if (not integration_smoke and int(config.get('task_end',5))!=5) or not 0<=start<=5 or (start!=0 and not config.get('resume_state_path')):
        raise ValueError('Clean run must start fresh at task 0 or resume a matching clean task-boundary state')
    if not integration_smoke and (config.get('denice_max_train_samples_per_client') or config.get('denice_max_clients') not in (None,100)):
        raise ValueError('Bounded smoke configuration is not the clean main run')
    roles=CleanRoleData(config['denice_clean_roles_dir'],source_data_dir=config.get('data_dir'))
    config['denice_data_roles_sha256']=file_sha256(roles.root/'role_manifest.json')
    role=config.get('denice_evaluation_data_role','validation')
    if role not in ('validation','test'):raise ValueError('Invalid clean evaluation role')
    if role=='test' and config.get('denice_eval_max_samples') is not None:
        raise ValueError('Clean full-test evaluation requires denice_eval_max_samples=None')
    config['denice_evaluation_data_role']=role
    if config.get('denice_cme_after_each_task'):
        config.pop('meta_peer_budget',None)
        config.pop('meta_peer_budget_selection',None)
        config['denice_cme_expert_pool']='entire recorded task-final collaboration neighborhood including self'
    elif protocol=='cgofed':
        config['meta_peer_budget']=16
        config.setdefault('meta_peer_budget_selection','pending clean validation after backbone training')
    else:
        config.pop('meta_peer_budget',None)
        config.pop('meta_peer_budget_selection',None)
        config['denice_inference_expert_pool']='self_only'
    return roles
