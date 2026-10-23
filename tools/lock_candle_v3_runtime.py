"""Record the exact V3 source tree and its delta from the immutable V1 clone."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1] / 'candle_v3_runtime'
    baseline = json.loads((root/'v1_clone_manifest.json').read_text(encoding='utf-8'))
    original = {entry['path']:entry['sha256'] for entry in baseline['files']}
    files, changes = [], []
    for path in sorted(root.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        name = path.relative_to(root).as_posix()
        if name == 'runtime_manifest.json':
            continue
        content = path.read_bytes()
        checksum = hashlib.sha256(content).hexdigest()
        files.append(dict(path=name, bytes=len(content), sha256=checksum))
        if name != 'v1_clone_manifest.json' and original.get(name) != checksum:
            changes.append(dict(path=name, change='modified' if name in original else 'added'))
    deleted = set(original) - {entry['path'] for entry in files}
    if deleted:
        raise ValueError(f'Baseline source deleted: {sorted(deleted)}')
    manifest = dict(schema=1, stage='peer_teaching_implementation',
        baseline_source_commit=baseline['baseline_source_commit'],
        baseline_manifest_sha256=hashlib.sha256((root/'v1_clone_manifest.json').read_bytes()).hexdigest(),
        files=files, changes=changes, training_executed=False, tests_executed=False,
        native_quality_status='UNVALIDATED', notebook_status='V1 baseline notebook remains unchanged')
    (root/'runtime_manifest.json').write_bytes((json.dumps(manifest,indent=2)+'\n').encode('utf-8'))
    print(json.dumps(dict(files=len(files), changes=len(changes), stage=manifest['stage'])))


if __name__ == '__main__':
    main()
