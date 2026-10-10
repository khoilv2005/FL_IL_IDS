"""Recompute audit totals from receiver receipts and original frozen metrics."""
import argparse
import json
from pathlib import Path

import numpy as np


def verify(audit_dir, legacy_dir):
    root = Path(audit_dir)
    result = json.loads((root / 'completion.json').read_text())
    if not result.get('completed'):
        raise ValueError('Full inference is not completed')
    source = json.loads((Path(legacy_dir) / 'completion.json').read_text())
    ids = result['protocol']['source_lock']['receivers']
    clients = [json.loads((root / f'client_{cid}.json').read_text()) for cid in ids]
    if len(clients) != 98 or len(set(c['receiver'] for c in clients)) != 98:
        raise ValueError('Receiver duplication/coverage failed')
    n = sum(c['rows'] for c in clients)
    checks = dict(full_rows=n == result['rows'] == 13_505_771,
                  frozen_model_state=all(c['state_unchanged'] for c in clients))
    for policy in ('BinarySelf', 'MulticlassSelf'):
        aggregate = sum(np.array(c['confusion_matrices'][policy]) for c in clients)
        saved = sum(np.array(c['confusion_matrices'][policy + '_Saved']) for c in clients)
        matched = sum(np.array(c['confusion_matrices'][policy + '_OracleMatched']) for c in clients)
        table = sum(np.array(c['buckets_by_class'][policy]) for c in clients)
        best = sum(c['best_allowed_correct'][policy] for c in clients)
        checks[policy + '_source_metrics'] = (
            int(saved.sum()) == source['metrics'][policy]['samples']
            and np.isclose(saved.diagonal().sum() / n, source['metrics'][policy]['accuracy'], rtol=0, atol=1e-14))
        checks[policy + '_exact_full_reproduction'] = np.array_equal(saved, aggregate) and all(
            c['reproduction'][policy]['saved'] == 0 for c in clients)
        checks[policy + '_exact_native_fast_equivalence'] = all(
            c['reproduction'][policy]['native'] == 0 for c in clients)
        checks[policy + '_decomposition'] = (
            int(table.sum()) == n
            and int(table[:, 0].sum()) == int(aggregate.diagonal().sum())
            and int(table[:, 0].sum() + table[:, 3].sum()) == best
            and int(matched.diagonal().sum()) <= best)
        checks[policy + '_original_mask_ownership'] = all(
            item['locally_trained_but_unreachable_rows'] == 0
            for c in clients for item in c['inventory'] if item['policy'] == policy)
        checks[policy + '_coverage_dominates_errors'] = int(table[:, 1].sum()) > int(table[:, 2:].sum())
    report = dict(passed=all(bool(v) for v in checks.values()), checks={k: bool(v) for k, v in checks.items()},
                  full_rows=n, receivers=len(clients), labels_used_for_oracles_only=True)
    (root / 'verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise ValueError('One or more audit integrity checks failed; inspect verification.json')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--audit-dir', required=True)
    parser.add_argument('--legacy-dir', required=True)
    args = parser.parse_args()
    verify(args.audit_dir, args.legacy_dir)
