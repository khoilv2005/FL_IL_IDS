"""Task-scoped calibration materialization and a current-task-only runtime view.

Preparation may read the original compressed local shard once. Runtime opens
only a locked task partition; it never falls back to CleanRoleData.client_role.
This is data staging, not a certificate of old-class safety.
"""
import json
from pathlib import Path
import numpy as np
from .config import Rejected
from .state import write_json
from fed_learning.data.denice_clean_roles import CleanRoleData, file_sha256

VERSION = 'appliance_current_calibration_store_v1'


def prepare_store(roles_dir, data_dir, out, client_ids):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('New calibration store required')
    roles = CleanRoleData(roles_dir, source_data_dir=data_dir)
    metadata = json.loads((roles.source / 'metadata.json').read_text(encoding='utf-8'))
    tasks = {str(int(t)): list(map(int, v)) for t, v in
             metadata['task_structure']['task_classes'].items()}
    labels = [c for v in tasks.values() for c in v]
    if sorted(map(int, tasks)) != list(range(6)) or sorted(labels) != list(range(34)):
        raise Rejected('INVALID_TASK_PARTITION')
    ids = list(client_ids)
    if not ids or any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
        raise Rejected('INVALID_CALIBRATION_CLIENTS')
    if any(str(i) not in roles.manifest['clients'] for i in ids):
        raise Rejected('UNKNOWN_CALIBRATION_CLIENT')
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'preparation_completion.json', dict(completed=False))
    for name in ('role_manifest.json', 'role_lock.json'):
        (out / name).write_bytes((roles.root / name).read_bytes())
    (out / 'metadata.json').write_bytes((roles.source / 'metadata.json').read_bytes())
    manifest = dict(version=VERSION, completed=False, task_classes=tasks,
        role_manifest_sha256=file_sha256(roles.root / 'role_manifest.json'),
        metadata_sha256=file_sha256(roles.source / 'metadata.json'),
        input_shape=list(roles.manifest['input_shape']), clients={},
        preparation_scope='one source CALIBRATION materialization per owned client; no BASE/validation/test roles',
        runtime_scope='current task partition only; old raw partitions inaccessible through runtime API',
        source_shards_read_during_preparation=ids)
    for cid in sorted(ids):
        x, y, rows = roles.client_role(cid, 'calibration')
        if not np.isfinite(x).all() or not np.isin(y, labels).all():
            raise Rejected('INVALID_SOURCE_CALIBRATION')
        entries = {}
        for t, classes in tasks.items():
            chosen = np.flatnonzero(np.isin(y, classes))
            rel = f'client_{cid}/task_{t}_calibration.npz'
            path = out / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, X=x[chosen], y=y[chosen], rows=rows[chosen])
            entries[t] = dict(file=rel, sha256=file_sha256(path), rows=len(chosen),
                class_counts={str(c): int((y[chosen] == c).sum()) for c in classes})
        manifest['clients'][str(cid)] = entries
        del x, y, rows
    manifest['completed'] = True
    write_json(out / 'calibration_store_manifest.json', manifest)
    write_json(out / 'calibration_store_lock.json',
               dict(manifest_sha256=file_sha256(out / 'calibration_store_manifest.json')))
    write_json(out / 'preparation_completion.json', dict(completed=True,
        role_manifest_sha256=manifest['role_manifest_sha256'],
        store_manifest_sha256=file_sha256(out / 'calibration_store_manifest.json'),
        runtime_historical_raw_reads_authorized=False))
    return manifest


class CurrentCalibrationData:
    """Receiver-local authority; use one instance per client.

    The owner and task are bound on construction. Only advance to the next task;
    no raw arrays are cached across that transition. Access records remain metadata.
    """
    def __init__(self, root, client_id, task, expected_role_sha256):
        self.root = Path(root).resolve()
        path = self.root / 'calibration_store_manifest.json'
        lock = json.loads((self.root / 'calibration_store_lock.json').read_text(encoding='utf-8'))
        self.store_sha256 = file_sha256(path)
        self.store = json.loads(path.read_text(encoding='utf-8'))
        if (lock.get('manifest_sha256') != self.store_sha256 or
                self.store.get('version') != VERSION or not self.store.get('completed')):
            raise Rejected('CALIBRATION_STORE_NOT_LOCKED')
        if (self.store['role_manifest_sha256'] != expected_role_sha256 or
                file_sha256(self.root / 'role_manifest.json') != expected_role_sha256 or
                file_sha256(self.root / 'metadata.json') != self.store['metadata_sha256']):
            raise Rejected('CALIBRATION_STORE_PROVENANCE_CHANGED')
        self.manifest = json.loads((self.root / 'role_manifest.json').read_text(encoding='utf-8'))
        self.source = self.root  # metadata only; no original source shard fallback
        if type(client_id) is not int or str(client_id) not in self.store['clients']:
            raise Rejected('CALIBRATION_OWNER_UNAVAILABLE')
        if type(task) is not int or str(task) not in self.store['task_classes']:
            raise Rejected('INVALID_CALIBRATION_TASK')
        self.client_id, self.task = client_id, task
        self.access_log = []

    def advance(self, task):
        if type(task) is not int or task != self.task + 1 or str(task) not in self.store['task_classes']:
            raise Rejected('NONCHRONOLOGICAL_CALIBRATION_ADVANCE')
        self.task = task

    def state(self):
        return dict(version=VERSION, store_manifest_sha256=self.store_sha256,
            role_manifest_sha256=self.store['role_manifest_sha256'], client_id=self.client_id, task=self.task)

    @classmethod
    def restore(cls, root, state):
        result = cls(root, state['client_id'], state['task'], state['role_manifest_sha256'])
        if state != result.state():
            raise Rejected('CALIBRATION_STORE_RESTORE_CHANGED')
        return result

    def client_role(self, *args, **kwargs):
        raise Rejected('UNSCOPED_CALIBRATION_READ_FORBIDDEN')

    def current_pool(self, cid, role, classes):
        # Validate authority before opening any data file.
        expected = self.store['task_classes'][str(self.task)]
        if cid != self.client_id or type(cid) is not int:
            raise Rejected('CROSS_CLIENT_CALIBRATION_READ_FORBIDDEN')
        if role != 'calibration':
            raise Rejected('NONCALIBRATION_ROLE_FORBIDDEN')
        if list(classes) != expected:
            raise Rejected('PAST_FUTURE_OR_PARTIAL_TASK_READ_FORBIDDEN')
        entry = self.store['clients'][str(cid)][str(self.task)]
        path = (self.root / entry['file']).resolve()
        if self.root not in path.parents or path.suffix != '.npz':
            raise Rejected('CALIBRATION_PATH_ESCAPED_STORE')
        if file_sha256(path) != entry['sha256']:
            raise Rejected('CURRENT_CALIBRATION_PARTITION_CHANGED')
        with np.load(path, allow_pickle=False) as data:
            x, y, rows = data['X'], data['y'], data['rows']
        if (x.dtype != np.float32 or y.dtype != np.int64 or rows.dtype != np.int64 or
                len(x) != len(y) or len(y) != len(rows) or len(y) != entry['rows'] or
                tuple(x.shape[1:]) != tuple(self.store['input_shape']) or
                not np.isfinite(x).all() or not np.isin(y, expected).all() or
                len(set(rows.tolist())) != len(rows) or (rows < 0).any() or
                {str(c): int((y == c).sum()) for c in expected} != entry['class_counts']):
            raise Rejected('CURRENT_CALIBRATION_PARTITION_INVALID')
        self.access_log.append(dict(client_id=cid, task=self.task, role=role,
            file=entry['file'], sha256=entry['sha256'], rows=len(y)))
        return dict(X=x, y=y, rows=rows, origin_client=cid, role=role,
            task=self.task, role_manifest_sha256=self.store['role_manifest_sha256'],
            partition_sha256=entry['sha256'])
