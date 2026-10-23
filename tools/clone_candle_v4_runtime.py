"""Create V4's untouched V1 core from the commit pinned by the V1 notebook."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile

from clone_candle_v1_runtime import SOURCE_PATHS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ref',help='Full source SHA; defaults to pinned V1 CODE_REF')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    ref = args.ref
    if ref is None:
        source = (root/'train_candle_v1_kaggle.py').read_text(encoding='utf-8')
        match = re.search(r'^CODE_REF\s*=\s*["\']([0-9a-f]{40})["\']',source,re.MULTILINE)
        if not match: raise ValueError('V1 notebook has no pinned source SHA')
        ref = match.group(1)
    if not re.fullmatch('[0-9a-f]{40}',ref): raise ValueError('Use a full source commit SHA')
    destination = root/'candle_v4_runtime'
    if destination.exists(): raise FileExistsError('Never overwrite an existing V4 runtime')
    archive = subprocess.check_output(['git','-c','core.autocrlf=false','-c','core.eol=lf',
        'archive','--format=tar',ref,*SOURCE_PATHS],cwd=root)
    tree = subprocess.check_output(['git','ls-tree','-r','-z',ref,*SOURCE_PATHS],cwd=root)
    expected = {}
    for entry in tree.split(b'\0'):
        if not entry: continue
        metadata,path = entry.split(b'\t',1)
        mode,kind,oid = metadata.split()
        if kind!=b'blob' or mode not in (b'100644',b'100755'):
            raise ValueError('Only ordinary source files are allowed')
        expected[path.decode()] = oid.decode()
    # Validate the entire archive before writing the new directory.
    contents,entries = {},[]
    with tarfile.open(fileobj=io.BytesIO(archive),mode='r:') as stream:
        for member in stream.getmembers():
            if member.isdir(): continue
            name = PurePosixPath(member.name)
            if not member.isfile() or name.is_absolute() or '..' in name.parts:
                raise ValueError('Only ordinary relative files are allowed')
            name = name.as_posix()
            if name in contents: raise ValueError('Duplicate source path')
            value = stream.extractfile(member).read()
            oid = hashlib.sha1(b'blob '+str(len(value)).encode()+b'\0'+value).hexdigest()
            if expected.get(name)!=oid: raise ValueError(f'Git blob mismatch: {name}')
            contents[name] = value
            entries.append(dict(path=name,bytes=len(value),sha256=hashlib.sha256(value).hexdigest(),git_blob=oid))
    if set(contents)!=set(expected): raise ValueError('Incomplete archive')
    for name,value in contents.items():
        path = destination/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(value)
    manifest = dict(schema=1,experiment='CANDLE V4',stage='v1_clone',baseline='CANDLE V1',
        baseline_source_commit=ref,baseline_ref_source='train_candle_v1_kaggle.py CODE_REF',
        selection=list(SOURCE_PATHS),source_files=len(entries),source_bytes=sum(e['bytes'] for e in entries),
        source_changes=0,files=sorted(entries,key=lambda e:e['path']),
        algorithm_changes=False,training_executed=False,evaluation_executed=False,tests_executed=False,
        launcher_status='V4 launcher/notebook not created at core-clone stage',
        scope='Exact V1 sparse-checkout code; no datasets, checkpoints, outputs, Git metadata or V3 additions')
    (destination/'v1_clone_manifest.json').write_bytes((json.dumps(manifest,indent=2)+'\n').encode())
    for entry in manifest['files']:
        if hashlib.sha256((destination/entry['path']).read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('Copied file hash mismatch')
    print(json.dumps({k:v for k,v in manifest.items() if k!='files'},indent=2))


if __name__=='__main__': main()
