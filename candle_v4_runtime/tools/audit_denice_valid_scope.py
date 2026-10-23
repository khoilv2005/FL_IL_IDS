"""Post-prediction DeNICE diagnostics within explicitly defined class scopes.

Reuse verified full-test confusion matrices. No predictions are recomputed or
changed; filtering is solely for metric analysis, never an inference input.
Conditional accuracy does not replace the full-test primary metric.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(cm, route_equal, oracle_cm, buckets):
    total = int(cm.sum())
    correct = int(np.trace(cm))
    counts = cm.sum(1)
    denominator = counts + cm.sum(0)
    f1 = np.divide(2 * np.diag(cm), denominator, out=np.zeros(34), where=denominator > 0)
    oracle_denominator = oracle_cm.sum(1) + oracle_cm.sum(0)
    oracle_f1 = np.divide(2 * np.diag(oracle_cm), oracle_denominator,
                          out=np.zeros(34), where=oracle_denominator > 0)
    return dict(samples=total, correct=correct, accuracy=correct / total if total else None,
        macro_f1_34=float(f1.mean()),
        macro_f1_true_classes_present=float(f1[counts > 0].mean()) if total else None,
        class_counts=counts.tolist(),
        route_correct=int((cm * route_equal).sum()),
        route_accuracy=float((cm * route_equal).sum() / total) if total else None,
        oracle_matched_correct=int(np.trace(oracle_cm)),
        oracle_matched_accuracy=float(np.trace(oracle_cm) / total) if total else None,
        oracle_matched_macro_f1_34=float(oracle_f1.mean()),
        recall_34=np.divide(np.diag(cm), counts, out=np.zeros(34), where=counts > 0).tolist(),
        oracle_matched_recall_34=np.divide(np.diag(oracle_cm), counts,
                                          out=np.zeros(34), where=counts > 0).tolist(),
        oracle_diagnostic_only=True, errors=int(total - correct),
        error_counts=dict(zip(['correct', 'coverage_unreachable', 'classifier_within_coverage',
                               'routing_recoverable'], map(int, buckets))))


def run(args):
    root = Path(args.audit_dir)
    completion, verification = read_json(root / 'completion.json'), read_json(root / 'verification.json')
    if not completion['completed'] or not verification['passed']:
        raise ValueError('Requires completed verified full audit')
    lock = completion['protocol']['source_lock']
    roles, metadata = read_json(args.roles), read_json(args.metadata)
    if sha256(args.roles) != lock['role_manifest_sha256'] or sha256(args.metadata) != roles['metadata_sha256']:
        raise ValueError('Role/dataset metadata lock mismatch')
    tasks = {int(t): list(map(int, classes)) for t, classes in metadata['task_structure']['task_classes'].items()}
    if sorted(c for classes in tasks.values() for c in classes) != list(range(34)):
        raise ValueError('Requires disjoint task mapping of 34 classes')
    task_of = np.array([next(t for t, classes in tasks.items() if c in classes) for c in range(34)])
    task_indicator = np.eye(6, dtype=np.int64)[task_of].T
    route_equal = task_of[:, None] == task_of[None, :]
    scope_names = ['full_test', 'local_base_classes', 'registered_classes',
                   'registered_without_local_base', 'unregistered_class_with_profile',
                   'missing_task_profile', 'task_profile_present']
    aggregate = {policy: {name: [np.zeros((34, 34), dtype=np.int64),
                               np.zeros((34, 34), dtype=np.int64), np.zeros(4, dtype=np.int64)]
                         for name in scope_names}
                 for policy in ['BinarySelf', 'MulticlassSelf']}
    per_client, checks = [], []
    for cid in lock['receivers']:
        receipt = read_json(root / f'client_{cid}.json')
        base = roles['clients'][str(cid)]['role_class_counts']['base']
        owned = np.array([int(base.get(str(c), 0)) > 0 for c in range(34)])
        client_metrics = {}
        for policy in aggregate:
            inv = next(item for item in receipt['inventory'] if item['policy'] == policy)
            if inv['fallback_all_seen_tasks']:
                raise ValueError('Cannot derive conditional route counts with all-class fallback')
            registered = np.array([c not in inv['unreachable_classes'] for c in range(34)])
            profile = np.isin(task_of, inv['bank'])
            scopes = dict(full_test=np.ones(34, dtype=bool), local_base_classes=owned,
                registered_classes=registered, registered_without_local_base=registered & ~owned,
                unregistered_class_with_profile=profile & ~registered,
                missing_task_profile=~profile, task_profile_present=profile)
            cm = np.array(receipt['confusion_matrices'][policy], dtype=np.int64)
            oracle = np.array(receipt['confusion_matrices'][policy + '_OracleMatched'], dtype=np.int64)
            table = np.array(receipt['buckets_by_class'][policy], dtype=np.int64)
            actual_routes = np.array(receipt['route_confusion'][policy], dtype=np.int64)
            # With disjoint nonempty class masks, the predicted class implies
            # its routed task. Verify against the actual full route receipt.
            checks.append(bool(np.array_equal(task_indicator @ cm @ task_indicator.T, actual_routes)))
            checks.append(bool(np.all(~owned | registered)))
            client_metrics[policy] = {}
            for name, selected in scopes.items():
                scoped, matched = cm * selected[:, None], oracle * selected[:, None]
                buckets = table[selected].sum(0)
                target = aggregate[policy][name]
                target[0] += scoped
                target[1] += matched
                target[2] += buckets
                client_metrics[policy][name] = metrics(scoped, route_equal, matched, buckets)
        per_client.append(dict(receiver=cid, metrics=client_metrics))
    result = {policy: {name: metrics(cm, route_equal, oracle, buckets)
                       for name, (cm, oracle, buckets) in scopes.items()}
              for policy, scopes in aggregate.items()}
    for policy in result:
        group = result[policy]
        partitions = ['local_base_classes', 'registered_without_local_base',
                      'unregistered_class_with_profile', 'missing_task_profile']
        checks.append(sum(group[name]['samples'] for name in partitions) == completion['rows'])
        checks.append(sum(group[name]['correct'] for name in partitions) == group['full_test']['correct'])
        checks.append(group['full_test']['samples'] == completion['rows'])
        checks.append(group['registered_classes']['error_counts']['coverage_unreachable'] == 0)
        for name in scope_names:
            checks.append(sum(group[name]['error_counts'].values()) == group[name]['samples'])
    # The original run disabled intermediate eval. End-point performance on
    # old classes is not a matched before/after forgetting measurement.
    report = dict(passed=all(checks), checks_passed=sum(checks), checks_total=len(checks),
        source_checkpoint_sha256=lock['checkpoint_sha256'], source_role_sha256=lock['role_manifest_sha256'],
        source_full_audit_completion_sha256=sha256(root / 'completion.json'),
        audit_source_sha256=sha256(__file__),
        scope='post-prediction conditional diagnostics; original full-test accuracy remains primary unless explicitly redefined',
        rows=completion['rows'], receivers=len(per_client), metrics=result,
        inference_changed=False, label_used_in_inference=False, raw_train_or_cal_read=False,
        new_test_predictions=False, no_training=True, no_test_reassignment=True,
        forgetting=dict(status='NOT_MEASURED',
            reason='Needs matched task-end checkpoints and client/class test rows. Terminal old-class accuracy alone is insufficient.'),
        collaboration_effect=dict(status='NOT_IDENTIFIED',
            reason='Requires a matched no-collaboration training control; inherited-subset accuracy is not a causal collaboration gain.'),
        registered_without_local_base_limitation='Registration metadata is not an independent certificate of safe peer transfer.',
        per_client=per_client)
    output = Path(args.out)
    if output.exists():
        raise FileExistsError('Use a new output path')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    compact = {policy: {name: {key: values[key] for key in ['samples', 'accuracy', 'route_accuracy',
                                                          'oracle_matched_accuracy']}
                        for name, values in groups.items()}
               for policy, groups in result.items()}
    print(json.dumps(dict(passed=report['passed'], checks_passed=report['checks_passed'],
                         checks_total=report['checks_total'], metrics=compact,
                         forgetting=report['forgetting'], collaboration_effect=report['collaboration_effect']),
                     indent=2), flush=True)
    if not report['passed']:
        raise ValueError('Conditional scope accounting checks failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ['audit-dir', 'roles', 'metadata', 'out']:
        parser.add_argument('--' + name, required=True)
    run(parser.parse_args())
