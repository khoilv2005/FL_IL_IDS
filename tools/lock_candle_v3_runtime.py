"""Record the exact V3 source tree and its delta from the immutable V1 clone."""
import hashlib
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--structural-receipt', type=Path)
    parser.add_argument('--native-receipt', type=Path)
    parser.add_argument('--receiver-structural-receipt', type=Path)
    parser.add_argument('--receiver-native-receipt', type=Path)
    args = parser.parse_args()
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
    if bool(args.structural_receipt) != bool(args.native_receipt):
        raise ValueError('Bind both smoke receipts together')
    if bool(args.receiver_structural_receipt) != bool(args.receiver_native_receipt):
        raise ValueError('Bind both receiver-head smoke receipts together')
    if args.receiver_structural_receipt and not args.structural_receipt:
        raise ValueError('Revalidate Direct FC2 before publishing the shared runtime')
    if args.structural_receipt:
        # Match the source_identity hash without importing the training runtime.
        source = hashlib.sha256()
        sources = sorted((root/'fed_learning').rglob('*.py')) + [root/'eval_checkpoint.py']
        for path in sources:
            source.update(path.relative_to(root).as_posix().encode())
            source.update(path.read_bytes().replace(b'\r\n',b'\n'))
        receipts = {}
        for name,path in (('structural',args.structural_receipt),('native',args.native_receipt)):
            row = json.loads(path.read_text(encoding='utf-8'))
            if not row.get('completed') or row.get('source_sha256') != source.hexdigest():
                raise ValueError(f'{name} receipt failed or covers another runtime source')
            receipts[name] = dict(path=path.as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                  source_sha256=source.hexdigest())
        manifest.update(tests_executed=True,training_executed=True,
                        full_training_executed=False,training_scope='bounded native smoke',
                        native_quality_status='ENGINEERING_SMOKE_PASS_QUALITY_UNPROVEN',
                        validated_variant='direct_fc2',unvalidated_variants=['receiver_head_learning'],
                        smoke_receipts=receipts)
        if args.receiver_structural_receipt:
            receiver_receipts = {}
            for name,path in (('structural',args.receiver_structural_receipt),
                              ('native',args.receiver_native_receipt)):
                row = json.loads(path.read_text(encoding='utf-8'))
                if (not row.get('completed') or row.get('variant') != 'receiver_head_learning'
                        or row.get('source_sha256') != source.hexdigest()):
                    raise ValueError(f'Receiver {name} smoke failed or covers another runtime')
                receiver_receipts[name] = dict(path=path.as_posix(),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_sha256=source.hexdigest())
            manifest.update(validated_variants=['direct_fc2','receiver_head_learning'],
                            unvalidated_variants=[],receiver_head_smoke_receipts=receiver_receipts)
    (root/'runtime_manifest.json').write_bytes((json.dumps(manifest,indent=2)+'\n').encode('utf-8'))
    print(json.dumps(dict(files=len(files), changes=len(changes), stage=manifest['stage'])))


if __name__ == '__main__':
    main()
