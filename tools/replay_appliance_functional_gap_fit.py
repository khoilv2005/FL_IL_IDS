"""Reapply discovery to sealed FIT counts without rereading CAL or retraining.

Preserves prior communication as measured once, not new replay traffic. Existing
HOLDOUT is reused only for the identical selected registration function; this
is development recomputation, never an independent evaluation.
"""
import argparse
import json
from pathlib import Path

from appliance.functional_gap_discovery import VERSION, select_probes, assess_holdout
from appliance.state import write_json
from fed_learning.data.denice_clean_roles import file_sha256


def run(a):
    a.out.mkdir(parents=True, exist_ok=False)
    source = json.loads(a.source.read_text(encoding='utf-8'))
    if not source['completed'] or source['protocol']['version'] != VERSION:
        raise ValueError('Completed functional discovery source required')
    protocol = dict(version=VERSION, kind='sealed FIT negative-veto recomputation',
        source_sha256=file_sha256(a.source),
        functional_module_sha256=file_sha256(Path('appliance/functional_gap_discovery.py')),
        source_authority={k: source['protocol'][k] for k in
            ('terminal_sha256', 'graph_sha256', 'role_manifest_sha256')},
        source_probe_count=source['probe_count'], selection_role='sealed current CAL-FIT only',
        rule='same-class same-function negative veto from ANY peer is binding for registration',
        current_CAL_reads=0, old_CAL_reads=0, refitting=False,
        earlier_HOLDOUT_reused_not_replication=True, independent=False,
        source_communication_reused_not_charged_as_new=True,
        automatic_install_enabled=False, final_test_opened=False)
    write_json(a.out/'protocol_before_reclassification.json', protocol)
    selected, protection = select_probes(source['probes'])
    results = []
    for probe in selected:
        pair, c = probe['pair'], probe['class_id']
        r = dict(pair=pair, gap=probe['gap'], trial_training_run=False,
            installation_authorized=False, native_smoke_authorized=False)
        if probe['gap'] == 'registration_gap' and protection[c]['current_negative_veto']:
            r.update(decision='skip_registration_current_negative_veto', protection=protection[c])
        elif not probe['observed_receiver_current_risk_pass'] or probe['gap'] == 'unknown_gap':
            r.update(decision='skip_insufficient_or_unsafe_FIT')
        elif probe['gap'] == 'functional_gap':
            r.update(decision='requires_fresh_trial_and_frozen_update_vs_availability_HOLDOUT')
        else:
            old = next((v for v in source['results'] if v['pair'] == pair and
                v.get('binding', {}).get('action') == 'registration'), None)
            if old is None:
                r.update(decision='requires_current_registration_HOLDOUT')
            else:
                if old['binding']['candidate_function'] != protection[c]['availability_function']:
                    raise ValueError('Existing registration function changed')
                assessment = assess_holdout(old['donor_HOLDOUT'], old['receiver_HOLDOUT'], 'registration',
                    dict(status='unknown', independent=False, reason='reused development evidence, no independent cumulative risk receipt'))
                r.update(decision=assessment['decision'], qualification=assessment,
                    donor_HOLDOUT=old['donor_HOLDOUT'], receiver_HOLDOUT=old['receiver_HOLDOUT'],
                    HOLDOUT_reused_not_new=True)
        results.append(r)
    report = dict(completed=True, protocol=protocol, probes=source['probes'],
        selected=selected, current_FIT_protection=protection, results=results,
        metadata_candidates=source['metadata_candidates'], probe_count=source['probe_count'],
        gap_counts=source['gap_counts'], trial_training_runs=0,
        communication=source['communication'],
        total_application_egress_bytes=source['total_application_egress_bytes'],
        communication_measured_in_source_run_not_replay=True,
        new_replay_application_egress_bytes=0,
        setup_capsule_bytes_each_donor=source['setup_capsule_bytes_each_donor'],
        historical_raw_CAL_reads=0, current_CAL_reads=0, BASE_reads=0, validation_reads=0,
        production_install_authorized=False, native_survival_authorized=False,
        production_runner_modified=False, full_training_started=False, final_test_opened=False)
    write_json(a.out/'completion.json', report)
    write_json(a.publish, report)
    print(json.dumps(dict(completed=True, gaps=report['gap_counts'],
        decisions=[(r['pair']['class_id'], r['decision']) for r in results],
        new_CAL_reads=0, native_survival_authorized=False), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'out', 'publish'):
        p.add_argument('--'+name, type=Path, required=True)
    run(p.parse_args())
