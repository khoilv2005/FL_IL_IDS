"""Read terminal classifier ranks/masks to validate coverage metadata.

Uses the trusted original checkpoint and existing provenance receipts only.
No optimizer, forward pass, raw historical train/CAL or new test access.
"""
import argparse
import collections
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from appliance.runner import load_input
from audit_denice_coverage_provenance import sha256_file


def run(args):
    provenance_dir = Path(args.provenance_dir)
    provenance = json.loads((provenance_dir / 'coverage_provenance.json').read_text(encoding='utf-8'))
    if not provenance.get('passed'):
        raise ValueError('Provenance audit must pass first')
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    if sha256_file(args.checkpoint) != provenance['checkpoint_archive_sha256']:
        raise ValueError('Requires the exact checkpoint archive bound to provenance')
    ckpt, hashes = load_input(args.checkpoint, 5, 19)
    if ckpt['config']['git_commit'] != provenance['training_commit']:
        raise ValueError('Training commit mismatch')
    rows = list(csv.DictReader((provenance_dir / 'receiver_class_coverage.csv').open(encoding='utf-8')))
    counters, results = collections.Counter(), []
    matches = []
    for row in rows:
        cid, c = int(row['receiver']), int(row['class_id'])
        state = ckpt['client_algorithm_states'].get(cid, ckpt['client_algorithm_states'].get(str(cid)))
        detector = state['context_detector']
        memory = {int(task): values for task, values in detector['activation_memory'].items()}
        classes = {int(task): list(map(int, values)) for task, values in detector['episode_classes'].items()}
        task = int(row['task'])
        admitted = task in memory and len(memory[task]) > 0 and c in classes.get(task, [])
        reachable = row['reachable'] == 'True'
        age = int(state['neuron_ages']['fc2'][c])
        weight_mask = np.asarray(state['connection_masks']['weight_fc2'])[c]
        density = float(np.count_nonzero(weight_mask) / len(weight_mask))
        counters[row['regime'] + '_rank_' + str(age)] += 1
        group = 'young' if age == 0 else 'learner' if age == 1 else 'mature'
        counters[row['regime'] + '_rank_group_' + group + '_test_rows'] += int(row['test_rows'])
        results.append(dict(receiver=cid, class_id=c, regime=row['regime'],
            test_rows=int(row['test_rows']), base_rows=int(row['base_rows']),
            fc2_rank=age, weight_mask_density=density, checkpoint_reachable=admitted))
        matches.append(admitted == reachable)
    checks = dict(all_checkpoint_registration_matches_provenance=all(matches),
        all_unreachable_rows_have_young_classifier_rank=all(row['fc2_rank'] == 0
            for row in results if row['regime'] != 'covered'),
        all_reachable_rows_have_mature_classifier_rank=all(row['fc2_rank'] >= 2
            for row in results if row['regime'] == 'covered'),
        full_receiver_class_matrix=len(results) == provenance['receiver_class_pairs'],
        test_rows_match=sum(row['test_rows'] for row in results) == provenance['rows'])
    report = dict(passed=all(checks.values()), checks=checks, no_inference=True, no_parameter_update=True,
        no_raw_train_or_cal_read=True, checkpoint_hashes=hashes, counters=dict(counters),
        receiver_class_pairs=len(results),
        interpretation='Every unreachable class has fc2 rank 0; no mature class was hidden by registration in this fixture.',
        limitation='Maturity is not a functional competence or ownership certificate. Rank 0 weights may have changed during catch-up.')
    with (out / 'classifier_rank_coverage.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (out / 'classifier_rank_coverage.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    if not report['passed']:
        raise ValueError('Rank/coverage audit checks failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ['checkpoint', 'provenance-dir', 'out']:
        parser.add_argument('--' + name, required=True)
    torch.set_num_threads(2)
    run(parser.parse_args())
