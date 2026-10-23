"""Trace DeNICE coverage from original participation/bootstrap metadata.

No training, model inference, raw historical train/CAL, test reassignment or
router fitting. Reconstruct registration independently, then compare it with
the completed full-test audit. Oracle counts remain diagnostic only.
"""
from __future__ import annotations

import argparse
import ast
import collections
import copy
import csv
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def peers(graph, receiver):
    record = graph.get('alpha_debug', {}).get(str(receiver))
    if record is None:
        return []
    ids, weights = record['group_ids'], record['alphas']
    if len(ids) != len(weights):
        raise ValueError('Graph IDs/alphas length mismatch')
    return [int(cid) for cid, alpha in zip(ids, weights)
            if int(cid) != receiver and float(alpha) > 0]


def csv_rows(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def run(args):
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a new output directory; provenance reports are immutable')
    out.mkdir(parents=True, exist_ok=True)
    audit = Path(args.audit_dir)
    completion, verification = read_json(audit / 'completion.json'), read_json(audit / 'verification.json')
    if not completion.get('completed') or not verification.get('passed'):
        raise ValueError('Requires a completed, verified full-test bottleneck audit')
    lock = completion['protocol']['source_lock']
    roles, metadata = read_json(args.roles), read_json(args.metadata)
    if sha256_file(args.roles) != lock['role_manifest_sha256']:
        raise ValueError('Role manifest differs from original audit')
    if sha256_file(args.metadata) != roles['metadata_sha256']:
        raise ValueError('Dataset metadata differs from original role manifest')
    sources = {}
    with zipfile.ZipFile(args.results) as archive:
        names = ['config.json', 'denice_debug_history.json', 'cluster_history.json']
        objects = {}
        prefixes = set()
        for leaf in names:
            matches = [name for name in archive.namelist() if name.endswith('/' + leaf)]
            if len(matches) != 1:
                raise ValueError(f'Expected one original {leaf}')
            name = matches[0]
            prefixes.add(name.rsplit('/', 1)[0])
            raw = archive.read(name)
            sources[leaf] = dict(member=name, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
            objects[leaf] = json.loads(raw)
        if len(prefixes) != 1:
            raise ValueError('Training metadata does not have a common results root')
        # Bind the history to the exact checkpoint container already audited.
        checkpoint_member = next(iter(prefixes)) + '/checkpoint_task_5_all_rounds.zip'
        digest = hashlib.sha256()
        with archive.open(checkpoint_member) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != lock['checkpoint_sha256']:
            raise ValueError('Training history comes from a different checkpoint output')
    config, debug, graphs = (objects[name] for name in names)
    if config['denice_data_roles_sha256'] != lock['role_manifest_sha256']:
        raise ValueError('Training role lock mismatch')
    if config.get('denice_cl_method') != 'legacy' or config.get('appliance_enabled', False):
        raise ValueError('Requires pure legacy fixture')
    if config.get('denice_canc_mode') != 'paper' or config.get('denice_memory_policy') != 'sketches':
        raise ValueError('Reconstruction is restricted to paper CANC + sketches')
    tasks = {int(task): list(map(int, classes))
             for task, classes in metadata['task_structure']['task_classes'].items()}
    if sorted(tasks) != list(range(6)) or sorted(c for classes in tasks.values() for c in classes) != list(range(34)):
        raise ValueError('Requires six disjoint tasks / 34 classes')
    starts = sorted((event for event in debug if event.get('type') == 'task_start'), key=lambda e: e['task'])
    if [event['task'] for event in starts] != list(range(6)):
        raise ValueError('Missing or duplicate task-start metadata')
    graph_map = {(int(graph['task']), int(graph['round'])): graph for graph in graphs}
    if len(graph_map) != 120 or any((task, round_id) not in graph_map
                                   for task in range(6) for round_id in range(20)):
        raise ValueError('Requires all 120 recorded collaboration rounds')
    inventory = {int(item['receiver']): item for item in read_json(audit / 'coverage_inventory.json')
                 if item['policy'] == 'MulticlassSelf'}
    ids = list(map(int, lock['receivers']))
    if set(ids) != set(inventory):
        raise ValueError('Receiver set mismatch')
    active = {event['task']: set(map(int, event['client_data'])) for event in starts}
    allocated = {int(task): set(map(int, clients))
                 for task, clients in metadata['client_allocation']['task_active_clients'].items()}
    states, joined, last_active, catchups = {}, {}, {}, {}
    history = []
    base_checks = []
    for event in starts:
        task = event['task']
        for bootstrap in event['bootstrap_events']:
            cid, source = int(bootstrap['client_id']), bootstrap['bootstrap_source']
            policy = bootstrap['bootstrap_policy']
            if policy == 'initial_template':
                if cid in states:
                    raise ValueError('Duplicate initial bootstrap')
                states[cid], joined[cid] = {}, task
            elif policy == 'representative_clone':
                if cid in states or int(source) not in states:
                    raise ValueError('Invalid clone chronology')
                states[cid], joined[cid] = copy.deepcopy(states[int(source)]), task
                for entry in states[cid].values():
                    entry['inheritance_chain'].append(dict(task=task, source=int(source), receiver=cid))
            elif policy == 'rejoining_plastic_catch_up':
                missing = list(range(last_active[cid] + 1, task))
                if len(missing) != int(bootstrap['missed_tasks']):
                    raise ValueError('Rejoin missed-task metadata mismatch')
                for skipped in missing:
                    catchups[cid, skipped] = dict(rejoined_at_task=task, source=int(source),
                        source_had_profile=skipped in states[int(source)],
                        source_classes=states[int(source)].get(skipped, {}).get('classes', []),
                        catch_up_param_distance=float(bootstrap['catch_up_param_distance']))
                # Native catch-up synchronizes target-plastic weights, not a
                # detector clone. Do not infer profile creation from weights.
            else:
                raise ValueError(f'Unsupported bootstrap: {policy}')
        for cid in sorted(active[task]):
            data = event['client_data'][str(cid)]
            labels = sorted(map(int, data['labels']))
            base = roles['clients'][str(cid)]['role_class_counts']['base']
            expected = sorted(c for c in tasks[task] if int(base.get(str(c), 0)) > 0)
            count_match = all(int(data['class_hist'].get(str(c), 0)) == int(base.get(str(c), 0))
                              for c in tasks[task])
            base_checks.append(labels == expected and count_match)
            states[cid][task] = dict(classes=labels, root_owner=cid, root_task=task, inheritance_chain=[])
            last_active[cid] = task
        history.append(dict(task=task, active_clients=len(active[task]),
                            bootstrap_policies=dict(collections.Counter(b['bootstrap_policy'] for b in event['bootstrap_events'])),
                            reconstructed_models=len(states)))
    counters, oracle_counts = collections.Counter(), collections.Counter()
    task_rows, class_rows = [], []
    observed_missing_tasks, observed_missing_classes = set(), set()
    terminal_graph = graph_map[5, 19]
    terminal_rates = collections.Counter()
    reconstruct_checks = []
    for cid in ids:
        original = inventory[cid]
        expected = {str(task): entry['classes'] for task, entry in states[cid].items()}
        reconstruct_checks.append(expected == original['declared_classes_by_task']
                                  and sorted(states[cid]) == original['bank'])
        receipt = read_json(audit / f'client_{cid}.json')
        self_cm = receipt['confusion_matrices']['MulticlassSelf']
        global_cm = receipt['confusion_matrices']['OracleGlobalTaskMaskDiagnostic']
        matched_cm = receipt['confusion_matrices']['MulticlassSelf_OracleMatched']
        available_peers = peers(terminal_graph, cid)
        for task, classes in tasks.items():
            entry = states[cid].get(task)
            present = entry is not None
            owned = cid in active[task]
            reason = ('local_participation' if owned else 'exact_clone_inheritance') if present else (
                'participated_but_missing' if owned else
                'prejoin_source_missing' if task < joined[cid] else 'skipped_task_after_join')
            catchup = catchups.get((cid, task))
            row_count = sum(sum(self_cm[c]) for c in classes)
            counters['profile_' + reason + '_pairs'] += 1
            if not present:
                counters['profile_' + reason + '_rows'] += row_count
                observed_missing_tasks.add((cid, task))
                cause = 'not_allocated_to_task' if cid not in allocated[task] else 'allocated_without_usable_base'
                counters['missing_profile_' + cause + '_pairs'] += 1
                counters['missing_profile_' + cause + '_rows'] += row_count
                if catchup:
                    counters['missing_profile_with_later_weight_catchup_pairs'] += 1
                    counters['missing_profile_with_later_weight_catchup_rows'] += row_count
                    if catchup['source_had_profile']:
                        counters['catchup_source_had_missing_profile_pairs'] += 1
                        counters['catchup_source_had_missing_profile_rows'] += row_count
            task_rows.append(dict(receiver=cid, task=task, first_join_task=joined[cid],
                local_participation=owned, profile_present=present, provenance=reason,
                dataset_allocated=cid in allocated[task],
                profile_root_owner=entry['root_owner'] if present else None,
                inheritance_chain=entry['inheritance_chain'] if present else [],
                classes=entry['classes'] if present else [], test_rows=row_count,
                later_catchup=catchup or {}))
            for c in classes:
                base = int(roles['clients'][str(cid)]['role_class_counts']['base'].get(str(c), 0))
                reachable = present and c in entry['classes']
                regime = 'covered' if reachable else 'missing_profile' if not present else 'missing_class_mask'
                birth_peers = peers(graph_map[task, 19], cid)
                birth_owners = [p for p in birth_peers if int(roles['clients'][str(p)]
                                    ['role_class_counts']['base'].get(str(c), 0)) > 0]
                live_owners = [p for p in available_peers if int(roles['clients'][str(p)]
                                    ['role_class_counts']['base'].get(str(c), 0)) > 0]
                live_admitted = [p for p in live_owners if task in states[p] and c in states[p][task]['classes']]
                count, normal_tp = sum(self_cm[c]), self_cm[c][c]
                counters[regime + '_receiver_class_pairs'] += 1
                counters[regime + '_rows'] += count
                if regime == 'missing_class_mask':
                    source_kind = 'local_class_subset' if owned else 'inherited_class_subset'
                    counters['missing_class_mask_' + source_kind + '_pairs'] += 1
                    counters['missing_class_mask_' + source_kind + '_rows'] += count
                oracle_counts[regime + '_global_task_oracle_tp'] += global_cm[c][c]
                oracle_counts[regime + '_matched_oracle_tp'] += matched_cm[c][c]
                if not reachable:
                    observed_missing_classes.add((cid, c))
                    counters['unreachable_with_local_base_pairs'] += int(base > 0)
                    if live_owners:
                        terminal_rates['unreachable_with_live_peer_owner_pairs'] += 1
                        terminal_rates['unreachable_with_live_peer_owner_rows'] += count
                    if live_admitted:
                        terminal_rates['unreachable_with_live_peer_owner_and_route_pairs'] += 1
                        terminal_rates['unreachable_with_live_peer_owner_and_route_rows'] += count
                    if owned and birth_owners:
                        counters['unreachable_locally_active_with_birth_peer_owner_pairs'] += 1
                        counters['unreachable_locally_active_with_birth_peer_owner_rows'] += count
                class_rows.append(dict(receiver=cid, task=task, class_id=c, regime=regime,
                    base_rows=base, profile_root_owner=entry['root_owner'] if present else None,
                    local_active_at_birth=owned, reachable=reachable, test_rows=count,
                    self_correct=normal_tp, matched_oracle_correct=matched_cm[c][c],
                    global_task_oracle_correct=global_cm[c][c],
                    birth_graph_peer_owners=birth_owners, terminal_graph_peer_owners=live_owners,
                    terminal_peer_owners_with_route=live_admitted))
    source_code = {}
    root = Path(__file__).resolve().parents[1]
    for name in ['fed_learning/training/decentralized_denice_il.py',
                 'fed_learning/strategies/decentralized/denice_aggregation.py',
                 'prepare_data/step2_federated_splits.py']:
        result = subprocess.run(['git', 'show', f'{config["git_commit"]}:{name}'], cwd=root,
                                capture_output=True, check=True)
        source_code[name] = dict(commit=config['git_commit'], sha256=hashlib.sha256(result.stdout).hexdigest())
        (out / Path(name).name).write_bytes(result.stdout)
    generator = ast.parse((out / 'step2_federated_splits.py').read_text(encoding='utf-8'))
    allocator = next(node for node in generator.body if isinstance(node, ast.ClassDef)
                     and node.name == 'ClientAllocator')
    probability_reads = [node.lineno for node in ast.walk(allocator) if isinstance(node, ast.Attribute)
                         and node.attr == 'old_client_data_prob' and isinstance(node.ctx, ast.Load)]
    dataset_participation = dict(
        old_client_data_prob_metadata=metadata['config'].get('old_client_data_prob'),
        probability_option_read_lines_in_allocator_at_training_commit=probability_reads,
        allocation_is_observed_not_regenerated=True,
        generator_producer_commit_unknown=True,
        per_task=[dict(task=task, allocated=len(allocated[task]), trained=len(active[task]),
            dropped_since_previous=sorted(allocated[task-1] - allocated[task]) if task else [],
            allocated_without_base=sorted(allocated[task] - active[task])) for task in tasks],
        limitation='Generator source at training commit corroborates metadata; original dataset-generation commit is not recorded.')
    checks = dict(original_checkpoint_bound_to_history=True,
        role_and_metadata_hashes=True, all_local_base_histograms_match=all(base_checks),
        all_terminal_registration_and_profile_keys_reconstructed=all(reconstruct_checks),
        full_test_row_accounting=sum(counters[r + '_rows'] for r in
            ['covered', 'missing_profile', 'missing_class_mask']) == completion['rows'],
        missing_profile_count_matches=sum(1 for row in task_rows if not row['profile_present']) == 74,
        missing_profile_rows_match=counters['missing_profile_rows'] == 2902000,
        missing_class_mask_rows_match=counters['missing_class_mask_rows'] == 2807415,
        all_missing_tasks_correspond_to_recorded_weight_catchup=set(catchups).intersection(
            {(cid, task) for cid in ids for task in tasks}) == observed_missing_tasks,
        missing_locally_trained_class_count_zero=counters['unreachable_with_local_base_pairs'] == 0,
        unseen_predictions_impossible=all(row['self_correct'] == 0 for row in class_rows if not row['reachable']),
        inventory_unreachable_pairs_match=observed_missing_classes ==
            {(cid, int(c)) for cid in ids for c in inventory[cid]['unreachable_classes']},
        full_test_original_receivers_unchanged=set(ids) == set(inventory))
    report = dict(passed=all(checks.values()), checks=checks, rows=completion['rows'], receivers=len(ids),
        receiver_task_pairs=len(task_rows), receiver_class_pairs=len(class_rows),
        reconstructed_receivers=sum(reconstruct_checks), training_commit=config['git_commit'],
        no_training=True, no_inference=True, no_raw_train_or_cal_read=True, no_test_reassignment=True,
        no_foreign_profile_install=True, no_acceptance_claim=True,
        source_hashes=sources, audited_source_code=source_code,
        checkpoint_archive_sha256=lock['checkpoint_sha256'],
        role_manifest_sha256=lock['role_manifest_sha256'],
        dataset_participation=dataset_participation,
        counters=dict(counters), oracle_counts=dict(oracle_counts),
        terminal_peer_metadata_coverage=dict(terminal_rates), per_task_history=history,
        interpretation=dict(
            missing_profiles='Offline at task birth; later plastic-weight catch-up does not clone the detector.',
            missing_class_masks='Local task class subset or inherited source subset; registration is not aggregated.',
            native_fc2_transfer='fc2 pair mask requires receiver AND sender class support; graph membership alone adds no class.',
            peer_owner_is_not_competence='Metadata counts do not certify functional transfer, historical CAL authorization or safety.',
            oracle_is_not_deployable='True-task/global-mask TP only measures latent capacity, not an implementable route.',
            foreign_profile_warning='Partial catch-up does not align all receiver features to donor sketches; blind import is invalid.'))
    csv_rows(out / 'receiver_task_provenance.csv', task_rows)
    csv_rows(out / 'receiver_class_coverage.csv', class_rows)
    write_json(out / 'coverage_provenance.json', report)
    print(json.dumps({k: report[k] for k in ['passed', 'checks', 'rows', 'receivers', 'counters',
                     'oracle_counts', 'terminal_peer_metadata_coverage']}, indent=2), flush=True)
    if not report['passed']:
        raise ValueError('Coverage provenance checks failed; do not infer a repair')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ['results', 'audit-dir', 'roles', 'metadata', 'out']:
        parser.add_argument('--' + name, required=True)
    run(parser.parse_args())
