"""Build a self-contained source-only Kaggle archive with member hashes."""
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    paths = [root / 'train_incremental_kaggle.py', root / 'eval_checkpoint.py', root / 'requirements.txt',
             root / 'configs/denice_appliance.json',
             root / 'tests/test_denice_appliance.py', root / 'tests/test_denice_transfer.py']
    paths += list((root / 'fed_learning').rglob('*.py'))
    paths += list((root / 'tools').glob('*.py'))
    paths += list((root / 'docs').glob('DENICE_APPLIANCE*.md'))
    paths = sorted(set(p for p in paths if p.is_file()))
    manifest = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    output = root / 'output/denice_appliance_kaggle.zip'
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, p.relative_to(root).as_posix())
        z.writestr('SOURCE_MANIFEST.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
        for name, digest in manifest.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
    print(json.dumps({'archive': str(output), 'files': len(paths), 'bytes': output.stat().st_size,
                      'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
