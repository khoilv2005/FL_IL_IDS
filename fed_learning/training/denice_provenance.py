"""Source identity shared by Kaggle training and offline evaluation."""
from pathlib import Path
import hashlib
import subprocess


def source_identity(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    files = sorted((root / 'fed_learning').rglob('*.py'))
    files += [root / 'eval_checkpoint.py']
    for path in files:
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes().replace(b'\r\n', b'\n'))
    result = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True)
    return {'source_sha256': digest.hexdigest(),
            'git_commit': result.stdout.strip() if result.returncode == 0 else None}
