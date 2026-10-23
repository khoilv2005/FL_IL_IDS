"""Count correct-task predictions blocked by receiver class registration."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def run(args):
    root, out = Path(args.audit_dir), Path(args.out)
    completion, verification = read(root / 'completion.json'), read(root / 'verification.json')
    scope = read(args.scope)
    if not completion['completed'] or not verification['passed'] or not scope['passed']:
        raise ValueError('Requires completed verified audit and scope receipts')
    if hashlib.sha256((root / 'completion.json').read_bytes()).hexdigest() != scope['source_full_audit_completion_sha256']:
        raise ValueError('Scope receipt belongs to another audit')
    metadata = read(args.metadata)
    lock = completion['protocol']['source_lock']
    roles = read(args.roles)
    if hashlib.sha256(Path(args.roles).read_bytes()).hexdigest() != lock['role_manifest_sha256']:
        raise ValueError('Role manifest mismatch')
    if hashlib.sha256(Path(args.metadata).read_bytes()).hexdigest() != roles['metadata_sha256']:
        raise ValueError('Dataset metadata mismatch')
    task_map = metadata['task_structure']['task_classes']
    task_of = np.array([next(int(t) for t, cs in task_map.items() if c in cs) for c in range(34)])
    tasks = np.eye(6, dtype=np.int64)[task_of].T
    totals, rows, checks = {}, [], []
    for policy in ['BinarySelf', 'MulticlassSelf']:
        aggregate = dict(full_samples=0, route_correct=0, unregistered_with_profile_samples=0,
                         correct_task_unregistered_class=0, mask_only_rescue_in_this_group=0,
                         correct_task_registered_but_classifier_wrong=0)
        for cid in lock['receivers']:
            receipt = read(root / f'client_{cid}.json')
            inv = next(v for v in receipt['inventory'] if v['policy'] == policy)
            if inv['fallback_all_seen_tasks']:
                raise ValueError('Task cannot be inferred from predicted class with all-class fallback')
            cm = np.array(receipt['confusion_matrices'][policy], dtype=np.int64)
            mask_cm = np.array(receipt['confusion_matrices'][policy + '_MaskOnlyDiagnostic'], dtype=np.int64)
            actual_routes = np.array(receipt['route_confusion'][policy], dtype=np.int64)
            checks.extend([bool(np.array_equal(tasks @ cm @ tasks.T, actual_routes)),
                           bool(np.array_equal(tasks @ mask_cm @ tasks.T, actual_routes))])
            counts = cm.sum(1)
            aggregate['full_samples'] += int(cm.sum())
            aggregate['route_correct'] += int(np.trace(actual_routes))
            for c in range(34):
                ep = int(task_of[c])
                same_task = task_of == ep
                routed_correct = int(cm[c, same_task].sum())
                profile_present = ep in inv['bank']
                registered = c in inv['classes_by_task'].get(str(ep), [])
                if profile_present and not registered:
                    blocked, rescue = routed_correct, int(mask_cm[c, c])
                    checks.append(int(cm[c, c]) == 0 and 0 <= rescue <= blocked)
                    aggregate['unregistered_with_profile_samples'] += int(counts[c])
                    aggregate['correct_task_unregistered_class'] += blocked
                    aggregate['mask_only_rescue_in_this_group'] += rescue
                    rows.append(dict(policy=policy, receiver=cid, task=ep, class_id=c,
                        samples=int(counts[c]), correct_task_unregistered_class=blocked,
                        mask_only_rescue_in_this_group=rescue,
                        blocked_still_wrong_with_global_task_mask=blocked-rescue))
                elif registered:
                    aggregate['correct_task_registered_but_classifier_wrong'] += routed_correct - int(cm[c, c])
        expected = scope['metrics'][policy]
        checks.extend([aggregate['full_samples'] == expected['full_test']['samples'],
            aggregate['route_correct'] == expected['full_test']['route_correct'],
            aggregate['unregistered_with_profile_samples'] == expected['unregistered_class_with_profile']['samples'],
            aggregate['correct_task_unregistered_class'] == expected['unregistered_class_with_profile']['route_correct']])
        blocked = aggregate['correct_task_unregistered_class']
        for label, denominator in [('fraction_of_full_test', aggregate['full_samples']),
                                   ('fraction_of_correct_task_samples', aggregate['route_correct']),
                                   ('fraction_of_unregistered_with_profile', aggregate['unregistered_with_profile_samples'])]:
            aggregate[label] = blocked / denominator
        aggregate['mask_only_rescue_fraction_of_this_group'] = aggregate['mask_only_rescue_in_this_group'] / blocked
        totals[policy] = aggregate
    if not all(checks):
        raise ValueError('Task/mask/row accounting check failed')
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    with (out / 'receiver_class_counts.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(passed=True, checks_passed=sum(checks), checks_total=len(checks),
        metrics=totals, source_checkpoint_sha256=lock['checkpoint_sha256'],
        source_completion_sha256=scope['source_full_audit_completion_sha256'],
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        no_new_inference=True, no_training=True, production_changed=False,
        explanation='Post-prediction analysis. Mask-only diagnostic reuses the same predicted task with expanded class choices, not a certified registration repair.')
    (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ['audit-dir', 'metadata', 'roles', 'scope', 'out']:
        parser.add_argument('--' + name, required=True)
    run(parser.parse_args())
