"""Refresh embedded support in the root launcher and its identical upload copy."""
import base64
import hashlib
import io
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    # Exclude the launcher itself: no nested bundles or recursive size growth.
    files = [root / 'configs/denice_appliance.json']
    for directory in ('fed_learning', 'appliance'):
        files.extend((root / directory).rglob('*.py'))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            info = zipfile.ZipInfo(path.relative_to(root).as_posix(), (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    encoded = base64.b85encode(payload).decode('ascii')
    begin, end = '# BEGIN GENERATED DENICE SUPPORT', '# END GENERATED DENICE SUPPORT'
    launcher = root / 'train_incremental_kaggle.py'
    source = launcher.read_text(encoding='utf-8')
    start, stop = source.index(begin), source.index(end) + len(end)
    block = (begin + '\n# Regenerate with python tools/package_denice_single_file.py after editing supporting code.\n'
             + f'EMBEDDED_SUPPORT_SHA256 = {digest!r}\nEMBEDDED_SUPPORT_B85 = (\n'
             + ''.join(f'    {encoded[i:i+120]!r}\n' for i in range(0, len(encoded), 120))
             + ')\n' + end)
    source = source[:start] + block + source[stop:]
    compile(source, str(launcher), 'exec')
    data = source.replace('\n', '\r\n').encode('utf-8')
    launcher.write_bytes(data)
    output = root / 'output/single_file/train_incremental_kaggle.py'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    print(f'Root launcher and upload copy updated: {len(data)} bytes; support SHA256 {digest}')


if __name__ == '__main__':
    main()
