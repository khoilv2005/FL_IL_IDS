"""Embed reproducible source in the maintained launcher for single-file uploads."""
from pathlib import Path
import base64
import hashlib
import io
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'train_incremental_kaggle.py'
    marker = '\n# BEGIN DENICE SUPPORT PAYLOAD\n'
    text = source.read_text(encoding='utf-8').split(marker, 1)[0].rstrip()+'\n'
    files = [*sorted((root/'fed_learning').rglob('*.py')),
             root/'tools/train_cluster_server.py', root/'tools/train_denice_v2.py',
             root/'requirements-denice-v2-kaggle.txt']
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            entry = zipfile.ZipInfo(path.relative_to(root).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, path.read_bytes())
    raw = buffer.getvalue()
    encoded = base64.b85encode(raw).decode('ascii')
    text += marker+'# sha256='+hashlib.sha256(raw).hexdigest()+'\n'
    text += ''.join('# '+encoded[i:i+120]+'\n' for i in range(0,len(encoded),120))
    data = text.encode('utf-8')
    compile(data, str(source), 'exec')
    source.write_bytes(data)
    output = root / 'output/single_file/train_incremental_kaggle.py'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    print(f'{output}: {len(data)} bytes (identical to root launcher)')


if __name__ == '__main__':
    main()
