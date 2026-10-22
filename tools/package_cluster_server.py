"""Build the source bundle used by standalone Kaggle train/eval launchers."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def main():
    root=Path(__file__).resolve().parents[1]
    destination=root/'output/cluster_server_kaggle'
    destination.mkdir(parents=True,exist_ok=True)
    files=[*sorted((root/'fed_learning').rglob('*.py')),*sorted((root/'appliance').rglob('*.py'))]
    files += [root/p for p in ('tools/train_denice_v2.py','tools/train_cluster_server.py',
        'tools/eval_cluster_servers.py','eval_selective.py','requirements-denice-v2-kaggle.txt')]
    manifest={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    target=destination/'denice_cluster_server_source.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in files:archive.write(p,p.relative_to(root).as_posix())
        archive.writestr('source_manifest.json',json.dumps(manifest,indent=2))
    for name in ('train_incremental_kaggle.py','eval_selective.py','eval_selective.ipynb'):
        shutil.copyfile(root/name,destination/name)
    shutil.copyfile(root/'docs/CLUSTER_SERVER_TRAIN_EVAL_VI.md',destination/'README.md')
    print(target)
    print('Bundle SHA256:',hashlib.sha256(target.read_bytes()).hexdigest())


if __name__=='__main__':main()
