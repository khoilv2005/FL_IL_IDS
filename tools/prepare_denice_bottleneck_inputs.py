"""Stage only original test data and frozen self predictions, never train shards."""
import argparse
import json
import shutil
import zipfile
from pathlib import Path


def copy_member(archive, member, target):
    if target.exists():
        return
    partial = target.with_suffix(target.suffix + '.partial')
    with archive.open(member) as source, partial.open('wb') as destination:
        shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
    partial.replace(target)
    print(f'Staged {target.name}: {target.stat().st_size:,} bytes', flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--dataset-zip', required=True)
    parser.add_argument('--results-zip', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.dataset_zip) as archive:
        copy_member(archive, '100-clients/metadata.json', out / 'metadata.json')
        copy_member(archive, '100-clients/global_test_data.npz', out / 'global_test_data.npz')
    with zipfile.ZipFile(out / 'global_test_data.npz') as archive:
        for name in ('X_test.npy', 'y_test.npy'):
            copy_member(archive, name, out / name)
    with zipfile.ZipFile(args.results_zip) as archive:
        prefix = 'results_denice_legacy_multiclass_xi_0p8_seed_42/legacy_self_task_5/'
        copy_member(archive, prefix + 'test_predictions.csv.gz', out / 'test_predictions.csv.gz')
    (out / 'staging.json').write_text(json.dumps({
        'dataset_archive': str(Path(args.dataset_zip).resolve()),
        'results_archive': str(Path(args.results_zip).resolve()),
        'historical_train_read': False,
        'historical_cal_read': False,
    }, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
