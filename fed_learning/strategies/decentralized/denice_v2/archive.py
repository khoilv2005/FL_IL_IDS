"""Content-addressed round checkpoints, sealed into self-contained task ZIPs.

Only new 4-KiB tensor blocks are stored in each round part. Manifests reference blobs
in this task's earlier parts. Sealing copies every blob once and verifies it
before removing the parts. No pickle or historical raw data in checkpoints.
"""
import base64
import hashlib
import json
import os
from pathlib import Path
import zipfile

from .contracts import encode, decode, state_hash


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class TaskArchive:
    def __init__(self, root, task, budget_gib=18):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.task, self.budget = int(task), int(budget_gib*1024**3)
        self.prefix = f'checkpoint_task_{task}_'
        self.sealed = self.root/f'{self.prefix}all_rounds.zip'

    def sources(self):
        if self.sealed.exists():
            return [self.sealed]
        return sorted(self.root.glob(f'{self.prefix}round_*.zip'))

    def usage(self):
        return sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())

    def guard(self):
        if self.usage() > self.budget:
            raise RuntimeError('Checkpoint output budget exceeded; earlier checkpoints remain intact')

    def index(self):
        blobs, manifests = {}, {}
        for path in self.sources():
            with zipfile.ZipFile(path) as z:
                for name in z.namelist():
                    if name.startswith('blobs/'):
                        blobs.setdefault(name, path)
                    elif name.startswith('rounds/'):
                        if name in manifests:
                            raise ValueError('Duplicate round checkpoint')
                        manifests[name] = path
        return blobs, manifests

    def save(self, round_id, payload):
        if self.sealed.exists():
            raise ValueError('Cannot append to a sealed task')
        blobs, manifests = self.index()
        name = f'rounds/{int(round_id):03d}.json'
        if name in manifests:
            raise ValueError('Refuse to overwrite a round checkpoint')
        added = {}

        def external(node):
            if node[0] in ('tensor', 'array'):
                raw = base64.b64decode(node[3], validate=True)
                addresses=[]
                for start in range(0,len(raw),4096):
                    chunk=raw[start:start+4096]
                    address=f'blobs/{digest(chunk)}'
                    addresses.append(address)
                    if address not in blobs:added[address]=chunk
                return ['chunks',node[0],node[1],node[2],addresses]
            if node[0] == 'dict':
                return ['dict', [[external(k), external(v)] for k, v in node[1]]]
            if node[0] in ('tuple', 'list'):
                return [node[0], [external(v) for v in node[1]]]
            return node

        manifest = dict(schema=1, task=self.task, round=int(round_id),
            snapshot_kind='task_boundary' if payload.get('next_task',self.task)>self.task else 'round_boundary',
            binding=state_hash(payload), payload=external(encode(payload)))
        target = self.root/f'{self.prefix}round_{round_id:03d}.zip'
        temporary = target.with_suffix('.zip.partial')
        if temporary.exists():
            if temporary.resolve().parent!=self.root:raise ValueError('Invalid partial checkpoint path')
            temporary.unlink() # incomplete writes are never a resume source
        self.guard()
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            z.writestr(name, json.dumps(manifest, separators=(',', ':'), allow_nan=False))
            for address, raw in sorted(added.items()):
                z.writestr(address, raw)
        self.guard()  # includes peak temporary bytes
        os.replace(temporary, target)
        if state_hash(self.load(round_id)) != manifest['binding']:
            raise ValueError('Checkpoint readback mismatch')
        return dict(path=target.name, bytes=target.stat().st_size, new_blobs=len(added),
                    total_output_bytes=self.usage(), checkpoint_binding=manifest['binding'])

    def load(self, round_id=None):
        blobs, manifests = self.index()
        if not manifests:
            raise FileNotFoundError('No completed round checkpoint')
        name = max(manifests) if round_id is None else f'rounds/{int(round_id):03d}.json'
        with zipfile.ZipFile(manifests[name]) as z:
            manifest = json.loads(z.read(name))
        if manifest['schema'] != 1 or manifest['task'] != self.task:
            raise ValueError('Checkpoint schema/task mismatch')
        handles, cache = {}, {}

        def internal(node):
            if node[0] in ('blob','chunks'):
                addresses=[node[4]] if node[0]=='blob' else node[4]
                chunks=[]
                for address in addresses:
                    if address not in cache:
                        path = blobs[address]
                        if path not in handles:handles[path]=zipfile.ZipFile(path)
                        raw=handles[path].read(address)
                        if digest(raw)!=address.removeprefix('blobs/'):
                            raise ValueError('Checkpoint blob checksum mismatch')
                        cache[address]=raw
                    chunks.append(cache[address])
                return [node[1],node[2],node[3],base64.b64encode(b''.join(chunks)).decode()]
            if node[0] == 'dict':
                return ['dict', [[internal(k), internal(v)] for k, v in node[1]]]
            if node[0] in ('tuple', 'list'):
                return [node[0], [internal(v) for v in node[1]]]
            return node

        try:
            result = decode(internal(manifest['payload']))
        finally:
            for z in handles.values():
                z.close()
        if state_hash(result) != manifest['binding']:
            raise ValueError('Checkpoint manifest binding mismatch')
        return result

    def import_prefix(self, source, round_id):
        """Relocate an exact round prefix from a read-only dataset for resume."""
        if self.sources():
            raise ValueError('Resume import requires an empty task archive destination')
        blobs, manifests = source.index()
        selected = {n:p for n,p in manifests.items() if int(Path(n).stem)<=round_id}
        if f'rounds/{round_id:03d}.json' not in selected:
            raise ValueError('Requested resume round missing')
        needed=set();raw_manifests={}
        def walk(node):
            if node[0]=='blob':needed.add(node[4])
            elif node[0]=='chunks':needed.update(node[4])
            elif node[0]=='dict':
                for k,v in node[1]:walk(k);walk(v)
            elif node[0] in ('tuple','list'):
                for v in node[1]:walk(v)
        for name,path in selected.items():
            with zipfile.ZipFile(path) as z:raw=z.read(name)
            walk(json.loads(raw)['payload']);raw_manifests[name]=raw
        target=self.root/f'{self.prefix}round_{round_id:03d}.zip'
        temporary=target.with_suffix('.zip.partial')
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as out:
            for name,raw in sorted(raw_manifests.items()):out.writestr(name,raw)
            for name in sorted(needed):
                with zipfile.ZipFile(blobs[name]) as z:raw=z.read(name)
                if digest(raw)!=name[6:]:raise ValueError('Resume import blob checksum failure')
                out.writestr(name,raw);self.guard()
        self.guard();os.replace(temporary,target)
        self.load(round_id)

    def seal(self):
        if self.sealed.exists():
            self.load()
            return self.sealed
        parts = self.sources()
        blobs, manifests = self.index()
        if not manifests:
            raise ValueError('Cannot seal an empty task')
        temporary = self.sealed.with_suffix('.zip.partial')
        handles = {}
        checksums = {}
        try:
            with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as out:
                for name, path in sorted({**blobs, **manifests}.items()):
                    if path not in handles:
                        handles[path] = zipfile.ZipFile(path)
                    raw = handles[path].read(name)
                    if name.startswith('blobs/') and digest(raw) != name[6:]:
                        raise ValueError('Mutated round archive')
                    out.writestr(name, raw)
                    checksums[name] = digest(raw)
                    self.guard()
                out.writestr('archive_manifest.json', json.dumps(dict(task=self.task,
                    rounds=sorted(manifests), checksums=checksums, self_contained=True)))
            self.guard()
            with zipfile.ZipFile(temporary) as z:
                if z.testzip() is not None or any(digest(z.read(n)) != sha for n, sha in checksums.items()):
                    raise ValueError('Task seal checksum failure')
            os.replace(temporary, self.sealed)
            self.load()
        finally:
            for z in handles.values():
                z.close()
        for part in parts:
            if part.resolve().parent != self.root or part == self.sealed:
                raise ValueError('Archive cleanup path escaped output')
            part.unlink()
        return self.sealed
