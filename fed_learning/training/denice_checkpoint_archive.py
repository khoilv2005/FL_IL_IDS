"""Atomic per-round ZIPs, sealed into a self-contained archive at task end."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile


def _digest(stream):
    value=hashlib.sha256()
    for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def _verify(path,expected):
    with zipfile.ZipFile(path) as archive:
        for name,digest in expected.items():
            with archive.open(name) as stream:
                if _digest(stream)!=digest:raise RuntimeError(f'Checkpoint archive checksum failed: {name}')


def storage_guard(root,budget_gib):
    root=Path(root);used=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    if budget_gib and used>float(budget_gib)*2**30:
        raise RuntimeError(f'Checkpoint/output budget exceeded: {used/2**30:.2f} GiB > {budget_gib} GiB. All saved checkpoints retained; export archives before continuing.')
    return used


def compress_checkpoint(path,level=6,budget_gib=None):
    """Delete the PT only after its ZIP is closed, verified and atomically sealed."""
    path=Path(path);target=path.with_suffix('.zip');temporary=target.with_suffix('.zip.partial')
    if target.exists():raise FileExistsError(f'Refuse to overwrite checkpoint archive {target}')
    if shutil.disk_usage(path.parent).free<path.stat().st_size+64*2**20:
        raise RuntimeError('Insufficient temporary disk space for safe checkpoint compression')
    with path.open('rb') as stream:digest=_digest(stream)
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=level,allowZip64=True) as archive:
        archive.write(path,path.name)
        archive.writestr('checkpoint_archive_manifest.json',json.dumps(dict(checksums={path.name:digest},checkpoint=path.name)))
    _verify(temporary,{path.name:digest});os.replace(temporary,target)
    path.unlink()
    index_file=path.parent/'checkpoint_index.json'
    if index_file.exists():
        index=json.loads(index_file.read_text(encoding='utf-8'))
        for record in index['checkpoints']:
            if record.get('path')==path.name:record.update(archive=target.name,member=path.name)
        index_file.write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8')
    used=storage_guard(path.parent,budget_gib)
    print(f'Checkpoint compressed: {target.name}, {target.stat().st_size/2**20:.1f} MiB; output={used/2**30:.2f} GiB',flush=True)
    return target


def seal_task_archive(root,task_id,level=6,budget_gib=None):
    """Merge all task checkpoints; retain individual round ZIPs until verified."""
    root=Path(root);prefix=f'checkpoint_task_{task_id}_'
    parts=sorted(p for p in root.glob(f'{prefix}*.zip') if p.name!=f'checkpoint_task_{task_id}_all_rounds.zip')
    extra=[root/f'checkpoint_task_{task_id}.pt',root/f'continuation_state_task_{task_id}.pt']
    extra=[p for p in extra if p.exists()]
    if not parts:raise ValueError('No archived round checkpoints to seal')
    target=root/f'checkpoint_task_{task_id}_all_rounds.zip';temporary=target.with_suffix('.zip.partial')
    if target.exists():raise FileExistsError(f'Refuse to overwrite sealed task {target}')
    checksums={};rounds=[]
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=level,allowZip64=True) as combined:
        for part in parts:
            with zipfile.ZipFile(part) as archive:
                manifest=json.loads(archive.read('checkpoint_archive_manifest.json'))
                for name,expected in manifest['checksums'].items():
                    if name in checksums:raise ValueError('Duplicate checkpoint in task archive')
                    digest=hashlib.sha256()
                    with archive.open(name) as source,combined.open(name,'w',force_zip64=True) as destination:
                        for block in iter(lambda:source.read(8*1024*1024),b''):
                            digest.update(block);destination.write(block)
                    if digest.hexdigest()!=expected:raise RuntimeError('Round archive mutated before task seal')
                    checksums[name]=expected
                    if '_round_' in name and not name.endswith('_base.pt'):
                        rounds.append(int(name.rsplit('_round_',1)[1].split('.')[0]))
        for path in extra:
            with path.open('rb') as stream:checksums[path.name]=_digest(stream)
            combined.write(path,path.name)
        if not rounds:raise ValueError('Task archive has no rounds')
        combined.writestr('checkpoint_archive_manifest.json',json.dumps(dict(task_id=task_id,checksums=checksums,
            completed_rounds=sorted(rounds),checkpoint=f'checkpoint_task_{task_id}_round_{max(rounds)}.pt',
            full_terminal_checkpoint=f'checkpoint_task_{task_id}.pt' if (root/f'checkpoint_task_{task_id}.pt') in extra else None,
            continuation_checkpoint=f'continuation_state_task_{task_id}.pt' if (root/f'continuation_state_task_{task_id}.pt') in extra else None),indent=2))
    _verify(temporary,checksums);os.replace(temporary,target)
    index_file=root/'checkpoint_index.json'
    if index_file.exists():
        index=json.loads(index_file.read_text(encoding='utf-8'))
        for record in index['checkpoints']:
            if int(record.get('task_id',-1))==int(task_id):
                record.update(archive=target.name,member=record['path'])
        index_file.write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8')
    for path in parts+extra:path.unlink()
    storage_guard(root,budget_gib)
    print(f'Task {task_id} sealed: {target.name}; rounds={len(rounds)}, size={target.stat().st_size/2**30:.2f} GiB',flush=True)
    return target
