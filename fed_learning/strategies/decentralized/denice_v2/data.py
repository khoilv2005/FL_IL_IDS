"""Locked BASE views; current-task disk cache and shared, memory-mapped TEST."""
import json
import os
from pathlib import Path
import shutil
import zipfile

import numpy as np
import torch

from fed_learning.data.denice_clean_roles import CleanRoleData, file_sha256
from .contracts import state_hash
from .native import CurrentBase


class TaskData:
    evaluation_role='test'
    def __init__(self, dataset, roles, cache):
        self.roles = CleanRoleData(roles, source_data_dir=dataset)
        self.dataset, self.cache = Path(dataset), Path(cache).resolve()
        self.client_ids = tuple(sorted(map(int, self.roles.manifest['clients'])))
        metadata = json.loads((self.dataset/'metadata.json').read_text(encoding='utf-8'))
        self.task_classes = {int(t): tuple(cs) for t, cs in metadata['task_structure']['task_classes'].items()}
        if self.task_classes != {t: tuple(range(t*6, min(t*6+6, 34))) for t in range(6)}:
            raise ValueError('Expected the locked six-task, 34-class mapping')
        self.input_shape = tuple(self.roles.manifest['input_shape'])
        self.binding = state_hash(dict(manifest=file_sha256(Path(roles)/'role_manifest.json'),
            metadata=file_sha256(self.dataset/'metadata.json')))
        self.counts, self.task = {}, None

    def prepare(self, task):
        self.task, self.counts = task, {}
        directory = self.cache/f'task_{task}'
        directory.mkdir(parents=True, exist_ok=True)
        # Source loading is preprocessing of locked role indices. Only current
        # rows enter this cache, training, clustering, BN or routing statistics.
        for cid in self.client_ids:
            record = directory/f'{cid}.json'
            if not record.exists():
                x, y, rows = self.roles.client_role(cid, 'base')
                keep = np.isin(y, self.task_classes[task])
                hashes = {}
                for suffix, value in [('x', x[keep]), ('y', y[keep]), ('rows', rows[keep])]:
                    path = directory/f'{cid}_{suffix}.npy'
                    np.save(path, value, allow_pickle=False)
                    hashes[path.name] = file_sha256(path)
                counts = {int(c): int(n) for c, n in zip(*np.unique(y[keep], return_counts=True))}
                value = dict(task=task, client=cid, role='base', binding=self.binding,
                    counts=counts, files=hashes)
                temporary = record.with_suffix('.partial')
                temporary.write_text(json.dumps(value), encoding='utf-8')
                os.replace(temporary, record)
            value = json.loads(record.read_text(encoding='utf-8'))
            if (value['task'], value['client'], value['role'], value['binding']) != (task, cid, 'base', self.binding):
                raise ValueError('Current BASE cache authority mismatch')
            for name, sha in value['files'].items():
                if Path(name).name != name or file_sha256(directory/name) != sha:
                    raise ValueError('Current BASE cache changed')
            self.counts[cid] = {int(c): int(n) for c, n in value['counts'].items()}
            if cid==self.client_ids[0] or cid==self.client_ids[-1] or self.client_ids.index(cid)%10==0:
                print(f'BASE cache T{task} client={cid}: {sum(self.counts[cid].values()):,} current rows; checksum verified',flush=True)
        print(f'Current BASE T{task}: {sum(sum(v.values()) for v in self.counts.values()):,} rows; uncapped', flush=True)

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
