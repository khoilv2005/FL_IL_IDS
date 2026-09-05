"""Inspect nested dataset NPZ headers/labels without extracting full feature arrays."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
from statistics import mean
from zipfile import ZipFile

import numpy as np


def inspect_npz(payload, split):
    with ZipFile(io.BytesIO(payload)) as z:
        with z.open(f'X_{split}.npy') as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError(f'Unsupported NPY header version: {version}')
        with z.open(f'y_{split}.npy') as stream:
            labels = np.load(stream, allow_pickle=False)
        if labels.ndim != 1 or len(labels) != shape[0]:
            raise ValueError('Feature/label shape mismatch')
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError('Labels must be integer global class IDs')
        classes, counts = np.unique(labels, return_counts=True)
        return {'shape': list(shape), 'feature_dtype': str(dtype),
                'label_dtype': str(labels.dtype),
                'class_histogram': {int(c): int(n) for c,n in zip(classes, counts)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('audit_denice/dataset_evidence.json'))
    args = parser.parse_args()
    digest = hashlib.sha256()
    with args.archive.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    report = {'archive': str(args.archive), 'sha256': digest.hexdigest(), 'clients': {}}
    with ZipFile(args.archive) as z:
        metadata_names = [n for n in z.namelist() if n.endswith('/metadata.json') or n == 'metadata.json']
        if len(metadata_names) != 1:
            raise ValueError('Expected one metadata.json')
        prefix = metadata_names[0][:-len('metadata.json')]
        metadata = json.loads(z.read(metadata_names[0]))
        report['metadata'] = metadata
        report['test'] = inspect_npz(z.read(prefix+'global_test_data.npz'), 'test')
        paths = [(int(m.group(1)), n) for n in z.namelist()
                 if (m := re.fullmatch(re.escape(prefix)+r'client_(\d+)_train\.npz', n))]
        for cid, name in sorted(paths):
            report['clients'][cid] = inspect_npz(z.read(name), 'train')
            if len(report['clients']) % 20 == 0:
                print(f'Checked {len(report["clients"])}/{len(paths)} client label arrays', flush=True)
    if set(report['clients']) != set(range(metadata['config']['num_clients'])):
        raise ValueError('Missing or unexpected client IDs')
    allowed = set(range(metadata['task_structure']['total_classes']))
    for entry in [report['test'], *report['clients'].values()]:
        if not set(entry['class_histogram']).issubset(allowed):
            raise ValueError('Out-of-range labels')
        if entry['shape'][1:] != report['test']['shape'][1:]:
            raise ValueError('Feature shape mismatch across files')
    report['tasks'] = {}
    for task, classes in metadata['task_structure']['task_classes'].items():
        rows = {}
        for cid, client in report['clients'].items():
            hist = {c: client['class_histogram'].get(c, 0) for c in classes}
            rows[cid] = {'samples': sum(hist.values()), 'present_classes': sum(v>0 for v in hist.values())}
        active = [cid for cid,r in rows.items() if r['samples']]
        assigned = metadata['client_allocation']['task_active_clients'][task]
        report['tasks'][task] = {
            'classes': classes, 'active_ids': active, 'active_count': len(active),
            'assigned_count': len(assigned), 'assigned_but_empty': sorted(set(assigned)-set(active)),
            'unexpected_active_ids': sorted(set(active)-set(assigned)),
            'train_samples': sum(r['samples'] for r in rows.values()),
            'active_sample_min': min(rows[c]['samples'] for c in active),
            'active_sample_max': max(rows[c]['samples'] for c in active),
            'mean_missing_class_fraction': mean(1-rows[c]['present_classes']/len(classes) for c in active),
            'class_histogram': {c: sum(a['class_histogram'].get(c,0) for a in report['clients'].values()) for c in classes},
            'clients': rows,
        }
    evidence_path = Path('audit_denice/artifact_evidence.json')
    if evidence_path.exists():
        prior = json.loads(evidence_path.read_text(encoding='utf-8'))
        expected = prior['full_d2']['42']['policies']['e4_pred_hard']['evaluation_sampling']['source_support_by_class']
        actual = {str(k):v for k,v in report['test']['class_histogram'].items()}
        report['test_support_matches_full_d2_seed42'] = actual == expected
        report['task0_hist_matches_historical_log'] = {
            str(k):v for k,v in report['tasks']['0']['class_histogram'].items()
        } == prior['historical']['data_by_task']['0']['class_hist']
    candidates = [cid for cid in report['clients'] if all(
        report['tasks'][str(t)]['clients'][cid]['present_classes'] == 6 for t in [0,1])]
    report['two_task_full_class_support_candidates'] = candidates
    report['total_train_samples'] = sum(c['shape'][0] for c in report['clients'].values())
    report['train_count_matches_metadata'] = report['total_train_samples'] == metadata['statistics']['total_train_samples']
    report['scope_limit'] = 'Headers and complete label arrays checked; feature contents and train/test duplicate overlap not scanned.'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['metadata','clients','tasks','test']}, indent=2))
    print('Active clients:', [t['active_count'] for t in report['tasks'].values()])
    print('Test shape:', report['test']['shape'])


if __name__ == '__main__':
    main()
