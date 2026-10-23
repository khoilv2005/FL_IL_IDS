"""Recheck saved DeNICE evidence without changing data or inference policy."""
import argparse
import ast
import csv
import hashlib
import json
import subprocess
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def run(args):
    provenance = read(args.provenance)
    scope = read(args.scope)
    verification = read(Path(args.audit_dir) / 'verification.json')
    completion = read(Path(args.audit_dir) / 'completion.json')
    commit = provenance['training_commit']
    sources = {}
    checks = {
        'full_audit_complete': completion['completed'],
        'full_audit_verification': verification['passed'] and all(verification['checks'].values()),
        'provenance_verified': provenance['passed'] and all(provenance['checks'].values()),
        'classifier_ranks_verified': provenance['classifier_rank_audit']['passed'],
        'valid_scope_verified': scope['passed'] and scope['checks_passed'] == scope['checks_total'],
        'same_checkpoint': provenance['checkpoint_archive_sha256'] == scope['source_checkpoint_sha256'],
        'same_role_manifest': provenance['role_manifest_sha256'] == scope['source_role_sha256'],
        'same_full_audit': digest((Path(args.audit_dir) / 'completion.json').read_bytes()) == scope['source_full_audit_completion_sha256'],
    }
    for name, lock in provenance['audited_source_code'].items():
        data = subprocess.check_output(['git', 'show', f'{commit}:{name}'])
        sources[name] = data.decode('utf-8')
        checks['historical_source_hash:' + name] = digest(data) == lock['sha256']

    root = Path(args.provenance_dir)
    with (root / 'receiver_task_provenance.csv').open(encoding='utf-8', newline='') as f:
        tasks = list(csv.DictReader(f))
    with (root / 'receiver_class_coverage.csv').open(encoding='utf-8', newline='') as f:
        classes = list(csv.DictReader(f))
    missing = [r for r in tasks if r['profile_present'] == 'False']
    checks.update(
        task_matrix_complete=len(tasks) == 98 * 6,
        class_matrix_complete=len(classes) == 98 * 34,
        no_missing_local_profile=not any(r['local_participation'] == 'True' for r in missing),
        missing_task_count=len(missing) == 74,
        missing_task_rows=sum(int(r['test_rows']) for r in missing) == 2902000,
        missing_task_allocation_breakdown=sum(r['dataset_allocated'] == 'False' for r in missing) == 73,
        no_local_class_hidden=not any(int(r['base_rows']) > 0 and r['reachable'] == 'False' for r in classes),
        missing_class_rows=sum(int(r['test_rows']) for r in classes if r['regime'] == 'missing_class_mask') == 2807415,
        full_row_accounting=sum(int(r['test_rows']) for r in classes) == 13505771,
    )

    allocator_file = 'prepare_data/step2_federated_splits.py'
    tree = ast.parse(sources[allocator_file])
    allocator = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ClientAllocator')
    option_uses = [n for n in ast.walk(allocator) if isinstance(n, ast.Attribute)
                   and n.attr == 'old_client_data_prob']
    option = dict(name='old_client_data_prob', source=allocator_file, commit=commit,
                  stores=[n.lineno for n in option_uses if isinstance(n.ctx, ast.Store)],
                  reads=[n.lineno for n in option_uses if isinstance(n.ctx, ast.Load)],
                  dataset_value=provenance['dataset_participation']['old_client_data_prob_metadata'],
                  producer_commit_known=False,
                  finding='CONFIG_OPTION_INERT_IN_AUDITED_ALLOCATOR',
                  impact_on_existing_dataset='NOT_CAUSALLY_ESTABLISHED')
    checks['allocator_option_is_stored_but_never_read'] = bool(option['stores']) and not option['reads']
    breakdown = {}
    for policy in ['BinarySelf', 'MulticlassSelf']:
        full = scope['metrics'][policy]['full_test']
        valid = scope['metrics'][policy]['registered_classes']
        checks['full_metrics_unchanged:' + policy] = full['accuracy'] == completion['metrics'][policy]['accuracy']
        errors = valid['error_counts']
        breakdown[policy] = dict(full_accuracy=full['accuracy'],
            registered_accuracy=valid['accuracy'], registered_samples=valid['samples'],
            registered_oracle_accuracy=valid['oracle_matched_accuracy'],
            registered_macro_f1_34=valid['macro_f1_34'],
            registered_oracle_macro_f1_34=valid['oracle_matched_macro_f1_34'],
            registered_errors=valid['errors'], routing_errors=errors['routing_recoverable'],
            routing_fraction_of_registered_errors=errors['routing_recoverable'] / valid['errors'],
            classifier_errors=errors['classifier_within_coverage'])

    result = dict(passed=all(checks.values()), checks=checks,
        checks_passed=sum(checks.values()), checks_total=len(checks),
        training_commit=commit, findings={
            'coverage': 'PROTOCOL_BEHAVIOR_NOT_PROVEN_BUG',
            'registration_or_restore_loss': 'NOT_OBSERVED_IN_AUDITED_FIXTURE',
            'routing': 'EMPIRICAL_ALGORITHM_LIMITATION_CODE_DEFECT_NOT_ESTABLISHED',
            'rare_class_classifier': 'FAILURE_OBSERVED_CAUSE_NOT_IDENTIFIED',
            'allocator_configuration': option,
            'forgetting': 'NOT_MEASURED',
            'collaboration_negative_transfer': 'NOT_IDENTIFIED'},
        metrics=breakdown,
        limitations=['Receipt checks do not prove every training component is bug-free.',
                    'Original dataset producer commit is unavailable; do not regenerate splits.',
                    'No matched task-end evaluation or no-collaboration control was added.',
                    'Routing accuracy below oracle does not alone prove a software defect.'],
        no_training=True, no_new_inference=True, no_raw_train_or_cal_read=True,
        production_changed=False,
        source_sha256={str(p): digest(Path(p).read_bytes()) for p in
            [args.provenance, args.scope, Path(args.audit_dir) / 'verification.json',
             root / 'receiver_task_provenance.csv', root / 'receiver_class_coverage.csv', __file__]})
    if not result['passed']:
        raise ValueError('Audit evidence mismatch: ' + repr([k for k, v in checks.items() if not v]))
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(passed=True, checks_passed=result['checks_passed'],
                         checks_total=result['checks_total'], metrics=breakdown), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ['provenance', 'scope', 'audit-dir', 'provenance-dir', 'out']:
        parser.add_argument('--' + name, required=True)
    run(parser.parse_args())
