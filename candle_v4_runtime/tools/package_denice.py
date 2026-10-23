"""Build the upgraded DENICE Kaggle source bundle with SHA-256 checksums."""
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / 'fed_learning').rglob('*.py'))
    files += [root / name for name in (
        'train_incremental_kaggle.py', 'eval_checkpoint.py', 'requirements.txt',
        'docs/DENICE_UPGRADE.md', 'tools/package_denice.py',
        'docs/DENICE_INCREMENTAL_RESEARCH.md', 'tools/benchmark_denice_incremental.py',
        'tests/test_denice_replay.py', 'tests/test_denice_classifier.py',
        'tests/test_denice_transfer.py',
        'tools/benchmark_denice_transfer.py',
        'tests/test_denice_router_replay.py', 'tests/test_denice_retention.py')]
    files += [root / name for name in (
        'tests/test_denice_continual.py', 'tests/test_denice_plasticity.py')]
    files = sorted(set(files))
    files += [root / name for name in (
        'train_denice_der_kaggle.py', 'train_denice_ewc_kaggle.py',
        'fed_learning/strategies/incremental/denice_variants.py',
        'configs/denice_der.json', 'configs/denice_derpp.json', 'configs/denice_ewc.json',
        'configs/denice_validation_only.json',
        'tools/build_denice_launchers.py', 'tools/benchmark_denice_der_ewc.py',
        'tests/test_denice_der_ewc.py', 'docs/DENICE_DER_EWC_GUIDE.md',
        'docs/DENICE_DER_EWC_SMOKE.json')]
    files = sorted(set(files))
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'Cannot package missing source files: {missing}')
    output = root / 'output' / 'denice_source_20261003_der_ewc.zip'
    output.parent.mkdir(exist_ok=True)
    manifest = {}
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            data = path.read_bytes()
            name = path.relative_to(root).as_posix()
            manifest[name] = hashlib.sha256(data).hexdigest()
            archive.writestr(name, data)
        archive.writestr('SOURCE_MANIFEST.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        for name, digest in manifest.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    print(f'{output}: {len(manifest)} source files; CRC and SHA-256 verified')


if __name__ == '__main__':
    main()
