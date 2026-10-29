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
    begin = '# BEGIN DENICE INLINE SUPPORT\n'
    end = '# END DENICE INLINE SUPPORT\n'
    if begin in text:
        before, rest = text.split(begin, 1)
        _, after = rest.split(end, 1)
        text = before + after
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
    block = begin+'def _denice_support_payload():\n'
    block += '    return ('+repr(hashlib.sha256(raw).hexdigest())+', (\n'
    block += ''.join('        '+repr(encoded[i:i+120])+'\n' for i in range(0,len(encoded),120))
    block += '    ))\n'+end
    # Define the literal before bootstrap executes: notebook cells have no __file__.
    anchor = 'def embedded_cluster_source():'
    if text.count(anchor) != 1:
        raise ValueError('Expected one embedded source helper')
    text = text.replace(anchor, block+anchor, 1)
    data = text.encode('utf-8')
    compile(data, str(source), 'exec')
    temporary = source.with_suffix('.py.building')
    temporary.write_bytes(data)
    temporary.replace(source)
    output = root / 'output/single_file/train_incremental_kaggle.py'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    print(f'{output}: {len(data)} bytes (identical to root launcher)')


if __name__ == '__main__':
    main()
