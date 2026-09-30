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
        'tests/test_denice_continual.py', 'tests/test_denice_plasticity.py',
        'docs/DENICE_PLASTICITY.md', 'docs/DENICE_PLASTICITY_RESULTS.json',
        'configs/denice_plasticity_experiment.json')]
    files = sorted(set(files))
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'Cannot package missing source files: {missing}')
    output = root / 'output' / 'denice_source_20260930_plasticity.zip'
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
