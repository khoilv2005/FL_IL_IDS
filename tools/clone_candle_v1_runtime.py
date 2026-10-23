"""Copy the complete code surface used by the pinned V1 notebook into V3's private runtime."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile


SOURCE_PATHS = ('fed_learning', 'appliance', 'tools', 'configs',
                'train_incremental_kaggle.py', 'eval_checkpoint.py',
                'requirements-candle-v1-kaggle.txt')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ref', help='Defaults to CODE_REF in train_candle_v1_kaggle.py')
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    ref = a.ref
    if ref is None:
        source = (root/'train_candle_v1_kaggle.py').read_text(encoding='utf-8')
        match = re.search(r'^CODE_REF\s*=\s*[\"\']([0-9a-f]{40})[\"\']', source, re.MULTILINE)
        if not match:
            raise ValueError('V1 notebook has no pinned source commit')
        ref = match.group(1)
    if not re.fullmatch('[0-9a-f]{40}', ref):
        raise ValueError('Use a full baseline commit SHA')
    destination = root/'candle_v3_runtime'
    if destination.exists():
        raise FileExistsError('V3 runtime already exists; never overwrite an edited fork')
    archive = subprocess.check_output(['git', '-c', 'core.autocrlf=false', '-c', 'core.eol=lf',
                                       'archive', '--format=tar', ref, *SOURCE_PATHS], cwd=root)
    tree = subprocess.check_output(['git', 'ls-tree', '-r', '-z', ref, *SOURCE_PATHS], cwd=root)
    blobs = {}
    for entry in tree.split(b'\0'):
        if not entry:
            continue
        metadata, path = entry.split(b'\t', 1)
        mode, kind, oid = metadata.split()
        if kind != b'blob' or mode not in (b'100644', b'100755'):
            raise ValueError('Only ordinary source blobs can be copied')
        blobs[path.decode('utf-8')] = oid.decode('ascii')
    entries = []
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:') as tf:
        for member in tf.getmembers():
            if member.isdir():
                continue
            name = PurePosixPath(member.name)
            if not member.isfile() or name.is_absolute() or '..' in name.parts:
                raise ValueError('Only ordinary relative source files can be copied')
            content = tf.extractfile(member).read()
            blob_oid = hashlib.sha1(b'blob ' + str(len(content)).encode('ascii') + b'\0' + content).hexdigest()
            if blobs.get(name.as_posix()) != blob_oid:
                raise ValueError(f'Archive content differs from Git blob: {name}')
            target = destination.joinpath(*name.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            entries.append(dict(path=name.as_posix(), bytes=len(content), sha256=hashlib.sha256(content).hexdigest(), git_blob=blob_oid))
    if len(entries) != len(blobs):
        raise ValueError('Archive omitted source files')
    manifest = dict(schema=1, experiment='CANDLE V3', stage='v1_clone', baseline='CANDLE V1',
        baseline_source_commit=ref, baseline_ref_source='train_candle_v1_kaggle.py CODE_REF',
        selection=list(SOURCE_PATHS), source_files=len(entries), source_bytes=sum(r['bytes'] for r in entries),
        source_changes=0, files=sorted(entries, key=lambda r:r['path']),
        no_training=True, no_evaluation=True,
        scope='Complete code trees from V1 sparse checkout; excludes datasets, checkpoints, outputs and Git metadata')
    (destination/'v1_clone_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='files'}, indent=2))


if __name__ == '__main__':
    main()
