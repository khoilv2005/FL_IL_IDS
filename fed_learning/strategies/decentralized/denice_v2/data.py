"""All original TRAIN rows per client, current-task cache and full TEST."""
import json
import os
from pathlib import Path
import shutil
import zipfile

import numpy as np
import torch

from fed_learning.data.denice_clean_roles import file_sha256
from .contracts import state_hash
from .evaluation import atomic_json
from .native import CurrentBase


class TaskData:
    evaluation_role='test'
    protocol='original_train_all_rows_current_task_v1'
    def __init__(self, dataset, cache):
        self.dataset, self.cache = Path(dataset), Path(cache).resolve()
        sources=sorted(self.dataset.glob('client_*_train.npz'),key=lambda p:int(p.stem.split('_')[1]))
        if not sources:raise FileNotFoundError('No original client TRAIN shards found')
        self.client_ids=tuple(int(p.stem.split('_')[1]) for p in sources)
        if len(set(self.client_ids))!=len(self.client_ids):raise ValueError('Duplicate client TRAIN identifiers')
        metadata = json.loads((self.dataset/'metadata.json').read_text(encoding='utf-8'))
        self.task_classes = {int(t): tuple(cs) for t, cs in metadata['task_structure']['task_classes'].items()}
        if self.task_classes != {t: tuple(range(t*6, min(t*6+6, 34))) for t in range(6)}:
            raise ValueError('Expected the locked six-task, 34-class mapping')
        # Read shape from TRAIN header, never from TEST or a calibration role.
        with zipfile.ZipFile(sources[0]) as z, z.open('X_train.npy') as stream:
            version=np.lib.format.read_magic(stream)
            if version==(1,0):shape,_,dtype=np.lib.format.read_array_header_1_0(stream)
            elif version==(2,0):shape,_,dtype=np.lib.format.read_array_header_2_0(stream)
            else:raise ValueError('Unsupported TRAIN NPY header version')
        if len(shape) not in (2,3) or dtype.hasobject:raise ValueError('Invalid TRAIN input shape/dtype')
        self.input_shape=tuple(shape[1:])
        self.source_hashes={}
        for index,path in enumerate(sources):
            self.source_hashes[int(path.stem.split('_')[1])]=file_sha256(path)
            if index%10==0 or index==len(sources)-1:
                print(f'TRAIN integrity: {index+1}/{len(sources)} original shards verified',flush=True)
        authority=dict(protocol=self.protocol,metadata=file_sha256(self.dataset/'metadata.json'),
            source_sha256=self.source_hashes,input_shape=self.input_shape)
        self.binding=state_hash(authority)
        atomic_json(self.cache.parent/'training_data_lock.json',dict(binding=self.binding,**authority,
            original_train_all_rows=self.protocol=='original_train_all_rows_current_task_v1',
            historical_raw_replay=False,test_used_for_training=False))
        self.counts, self.task = {}, None

    def training_rows(self,cid):
        source=self.dataset/f'client_{cid}_train.npz'
        if file_sha256(source)!=self.source_hashes[cid]:raise ValueError('Original TRAIN shard changed')
        with np.load(source,allow_pickle=False) as data:
            x=data['X_train'];labels=data['y_train']
            if (labels.ndim!=1 or len(x)!=len(labels) or tuple(x.shape[1:])!=self.input_shape
                or not np.isin(labels,np.arange(34)).all()):
                raise ValueError('Invalid original TRAIN labels/shape')
            return x,np.asarray(labels,np.int64),np.arange(len(labels),dtype=np.int64)

    def prepare(self, task):
        self.task, self.counts = task, {}
        directory = self.cache/f'task_{task}'
        directory.mkdir(parents=True, exist_ok=True)
        # Read original TRAIN, select every row of the current task. No
        # held-out roles, resampling, content deduplication or row cap.
        # Historical/future rows never enter training/BN/context statistics.
        for cid in self.client_ids:
            record = directory/f'{cid}.json'
            if not record.exists():
                x, y, rows = self.training_rows(cid)
                keep = np.isin(y, self.task_classes[task])
                hashes = {}
                for suffix, value in [('x',np.asarray(x[keep],np.float32)),('y',y[keep]),('rows',rows[keep])]:
                    path = directory/f'{cid}_{suffix}.npy'
                    np.save(path, value, allow_pickle=False)
                    hashes[path.name] = file_sha256(path)
                counts = {int(c): int(n) for c, n in zip(*np.unique(y[keep], return_counts=True))}
                value = dict(task=task, client=cid, role='base',binding=self.binding,
                    data_protocol=self.protocol,source_rows=len(y),current_train_rows=int(keep.sum()),
                    counts=counts, files=hashes)
                temporary = record.with_suffix('.partial')
                temporary.write_text(json.dumps(value), encoding='utf-8')
                os.replace(temporary, record)
            value = json.loads(record.read_text(encoding='utf-8'))
            if (value['task'], value['client'], value['role'], value['binding']) != (task, cid, 'base', self.binding):
                raise ValueError('Current TRAIN cache authority mismatch')
            for name, sha in value['files'].items():
                if Path(name).name != name or file_sha256(directory/name) != sha:
                    raise ValueError('Current TRAIN cache changed')
            self.counts[cid] = {int(c): int(n) for c, n in value['counts'].items()}
            if cid==self.client_ids[0] or cid==self.client_ids[-1] or self.client_ids.index(cid)%10==0:
                print(f'TRAIN cache T{task} client={cid}: {sum(self.counts[cid].values()):,} current rows; checksum verified',flush=True)
        print(f'Current TRAIN T{task}: {sum(sum(v.values()) for v in self.counts.values()):,} rows; all available rows',flush=True)

    def base(self, cid):
        directory = self.cache/f'task_{self.task}'
        values = [np.load(directory/f'{cid}_{s}.npy', mmap_mode='c', allow_pickle=False) for s in ('x', 'y', 'rows')]
        source_hash = state_hash(json.loads((directory/f'{cid}.json').read_text(encoding='utf-8')))
        return CurrentBase(cid, self.task, torch.from_numpy(values[0]), torch.from_numpy(values[1]),
                           values[2], source_hash,allow_singleton=True)

    def release(self, task):
        directory = (self.cache/f'task_{task}').resolve()
        if directory.parent != self.cache:
            raise ValueError('Invalid cache cleanup path')
        if directory.exists():
            shutil.rmtree(directory)

    def test(self):
        source = self.dataset/'global_test_data.npz'
        directory = self.cache/'test'
        directory.mkdir(parents=True, exist_ok=True)
        sha = file_sha256(source)
        lock = directory/'test_lock.json'
        if not lock.exists():
            hashes = {}
            with zipfile.ZipFile(source) as z:
                for name in ('X_test.npy', 'y_test.npy'):
                    target = directory/name
                    with z.open(name) as src, target.open('wb') as dst:
                        shutil.copyfileobj(src, dst, 8*1024**2)
                    hashes[name] = file_sha256(target)
            lock.write_text(json.dumps(dict(source_sha256=sha, files=hashes)), encoding='utf-8')
        value = json.loads(lock.read_text(encoding='utf-8'))
        if value['source_sha256'] != sha:
            raise ValueError('TEST source changed')
        for name, expected in value['files'].items():
            if file_sha256(directory/name) != expected:
                raise ValueError('TEST cache checksum mismatch')
        x, y = [np.load(directory/n, mmap_mode='c', allow_pickle=False) for n in ('X_test.npy', 'y_test.npy')]
        if len(x) != len(y) or y.ndim != 1 or tuple(x.shape[1:]) != self.input_shape:
            raise ValueError('TEST shape mismatch')
        if not np.isin(y, np.arange(34)).all():
            raise ValueError('Invalid TEST values')
        return x, y, sha

    def release_test(self):
        directory=(self.cache/'test').resolve()
        if directory.parent!=self.cache:raise ValueError('Invalid TEST cache cleanup path')
        if directory.exists():shutil.rmtree(directory)
