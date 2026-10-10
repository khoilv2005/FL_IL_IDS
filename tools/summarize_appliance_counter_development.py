"""Archive completed counter-guard audits with their limits and source hashes."""
import argparse
import hashlib
import json
from pathlib import Path
from appliance.state import write_json


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def run(a):
    raw = a.root/'contrastive_development_02'; fit = a.root/'fit_counter_pair65_01'
    if not load(raw/'completion.json')['completed'] or not load(fit/'completion.json')['completed']:
        raise ValueError('Incomplete audit must not be reported as completed')
    result = dict(kind='APPLIANCE counter-guard development; not production or final evaluation',
        raw_counter=dict(completion=load(raw/'completion.json'), results=load(raw/'results.json')),
        fixed_FIT_offsets=dict(completion=load(fit/'completion.json'), results=load(fit/'results.json')),
        original_FIT_diagnosis=load(a.root/'counter_fit65_release/result.json'), envelopes={},
        exact_collisions=load(a.root/'feature_collision65_01/result.json'),
        production_enabled=False, native_smoke_authorized=False, installed_patches=0,
        backbone_training=False, xi_changed=False, final_test_reopened=False,
        CAL_recall_gate=.95, CAL_FAR_gate=.001, CAL_break_gate=0,
        no_population_FAR_certificate=True, receipts_simulated=True,
        fixture='Task 3 terminal checkpoint from original results (13); same development roles',
        same_fixture_not_independent_confirmation=True,
        aborted_run_excluded='contrastive_development_01; its partial results are not a completed experiment',
        repeated_pair65_is_not_an_independent_replication=True,
        conclusion='Current guards fail the joint positive-retention and peer-negative-protection objective')
    for name, folder in dict(unit_sketch_diagonal='positive_fit_envelope65_01',
            raw_sketch_diagonal='raw_positive_fit_envelope65_01',
            full_input_diagonal='full_input_fit_envelope65_01',
            full_input_covariance='full_covariance_fit_envelope65_01').items():
        d = a.root/folder
        if not load(d/'completion.json')['completed']:
            raise ValueError(f'Incomplete envelope: {name}')
        result['envelopes'][name] = dict(protocol=load(d/'protocol.json'),
            completion=load(d/'completion.json'), results=load(d/'results.json'),
            lock_sha256=hashlib.sha256((d/'envelope_lock_before_validation.json').read_bytes()).hexdigest())
    paths = ['appliance/contrastive_protection.py', 'appliance/fit_contrastive_protection.py',
             'tools/audit_appliance_provenance_acceptance.py', 'tools/audit_appliance_contrastive_protection.py',
             'tools/audit_appliance_counter_fit_signals.py', 'tools/audit_appliance_positive_fit_envelope.py',
             'tools/audit_appliance_feature_collisions.py', 'tools/summarize_appliance_counter_development.py']
    result['current_source_sha256'] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    result['raw_counter_driver_snapshot_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                   for p in sorted((raw/'source_snapshot').glob('*.py'))}
    result['raw_driver_snapshot_notice'] = ('Raw counter ran without FIT offsets; driver/helper were copied '
        'before adding the optional FIT policy. These file snapshots preserve that executed driver version.')
    write_json(a.out, result)
    print(f'Wrote {a.out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    run(p.parse_args())
