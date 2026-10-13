"""Task1 discovery replay: Availability-only first, train only a measured gap.

Checks all legitimate mature/owned donors, uses current CAL-FIT for selection,
then native CAL-HOLDOUT against the correct reference. This is development
replay, not independent acceptance or a production installer.
"""
import argparse
import gc
import io
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from threadpoolctl import threadpool_limits

from appliance.config import Protocol, Rejected
from appliance.current_base_data import CurrentBaseData
from appliance.current_calibration_data import CurrentCalibrationData
from appliance.functional_gap_discovery import (VERSION, RULES as GAP_RULES,
    availability_shadow, metadata_candidates, current_split, endpoint_comparison,
    classify, select_probes, assess_holdout)
from appliance.runner import load_input
from appliance.selector import recorded_graph, lookup
from appliance.state import complete_hash, write_json
from appliance.train_time_transfer import (RULES, donor_proposal, receiver_integration,
    shadow, stratified_cap, update_packet, read_update)
from appliance.transport import Transport
from eval_checkpoint import _make_denice_client_model
from fed_learning.data.denice_clean_roles import file_sha256
from fed_learning.training.checkpoint_state import snapshot_denice_state


def wire_json(transport, sender, receiver, kind, payload):
    packet = json.dumps(payload, sort_keys=True, allow_nan=False).encode('utf-8')
    return json.loads(transport.send(sender, receiver, kind, packet))


def run(a):
    if getattr(a, 'device', 'cpu') != 'cpu':
        raise ValueError('This development replay is CPU-only; GPU/native production integration is not authorized by this tool')
    a.out.mkdir(parents=True, exist_ok=False)
    write_json(a.out/'completion.json', dict(completed=False))
    ckpt, hashes = load_input(a.checkpoint, 1, 19)
    if ckpt['config'].get('denice_cl_method') != 'legacy' or sorted(ckpt['seen_classes']) != list(range(12)):
        raise Rejected('FUNCTIONAL_GAP_LEGACY_TASK1_REQUIRED')
    role_sha = ckpt['config']['denice_data_roles_sha256']
    if file_sha256(a.roles/'role_manifest.json') != role_sha:
        raise Rejected('FUNCTIONAL_GAP_ROLE_CHANGED')
    groups, alphas = recorded_graph(ckpt, 1, 19)
    i = 2
    rv = CurrentCalibrationData(a.cal_store, i, 1, role_sha)
    current = rv.store['task_classes']['1']
    if current != list(range(6, 12)):
        raise Rejected('FUNCTIONAL_GAP_CLASS_SCOPE_CHANGED')
    model, router = _make_denice_client_model(ckpt, i, 'cpu')
    before = complete_hash(model, router)
    authority = {}
    counts = {}
    for owner in set(groups[i]) | {i}:
        alg = lookup(ckpt['client_algorithm_states'], owner)
        alg = alg.get('denice', alg)
        authority[owner] = SimpleNamespace(unit_ranks=alg['neuron_ages'])
        counts[owner] = {str(c): int(rv.manifest['clients'][str(owner)]['role_class_counts']['base'].get(str(c), 0)) for c in current}
    requests, candidates = metadata_candidates(i, model, router, authority, counts,
        groups, alphas, 1, current)
    protocol = dict(version=VERSION, **hashes, role_manifest_sha256=role_sha,
        task=1, round=19, receiver=i, requests=requests, metadata_candidates=candidates,
        discovery='all positive-alpha graph neighbors with current owned BASE and mature target row',
        donor_native_accuracy_filter=False, rules=GAP_RULES, learning_rules_unchanged=RULES,
        primary_inference='native pred_hard binary_cosine, seen0..11',
        probe='Availability-only on owner current CAL-FIT; weights/masks/ranks unchanged',
        selection='registration_gap with current risk pass and no negative veto from any peer first; then functional_gap with risk pass; positive count descending, donor ID tie',
        transfer_reference='Availability-only on identical owner CAL-HOLDOUT rows',
        registration_reference='closed native mask; remains subject to separate risk authorization',
        prior_CAL_and_development_exposure_known=True, independent_evidence_established=False,
        CAL_HOLDOUT_is_offline_development_replay=True, old_risk_status='unknown; no raw old CAL reads',
        automatic_install_enabled=False, production_runner_modified=False,
        final_test_opened=False, strict_population_safety_claim=False,
        failure_substitution_after_HOLDOUT=False,
        registration_aggregates_all_observed_FIT_negative_vetoes=True,
        transfer_HOLDOUT_risk_queries='all already queried current peer endpoints, with new frozen function; old scores cannot certify updated weights',
        communication='one original receiver capsule per queried donor, all requests reuse capsule; count query and result metadata and any trained update',
        transport_budgets='32MiB each direction per donor endpoint simulation; no federation-wide quota claim',
        source_sha256={name: file_sha256(Path(name)) for name in (
            'appliance/functional_gap_discovery.py', 'tools/run_appliance_functional_gap_transfer.py')})
    write_json(a.out/'protocol_before_current_CAL.json', protocol)
    # Build capsule once. No raw reference inputs may be sent to another owner.
    alg = snapshot_denice_state(model, router)
    if alg['context_detector'].get('reference_input_memory'):
        raise Rejected('FUNCTIONAL_GAP_RAW_REFERENCE_MEMORY')
    capsule = dict(config=ckpt['config'], task=1,
        client_model_states={i: {k: v.detach().cpu() for k, v in model.state_dict().items()}},
        client_algorithm_states={i: {'denice': alg}})
    stream = io.BytesIO()
    torch.save(capsule, stream)
    capsule_packet = stream.getvalue()
    transports, replicas, donor_views, donor_fit = {}, {}, {}, {}
    receiver_fit = current_split(rv, 'fit')
    available = {c: availability_shadow(model, router, c, 1) for c in sorted({v['class_id'] for v in candidates})}
    receiver_probe = {c: endpoint_comparison((model, router), pair, receiver_fit, c, list(range(12)), 'cpu')
        for c, pair in available.items()}
    probes = []
    for pair in candidates:
        j, c = pair['donor'], pair['class_id']
        if j not in replicas:
            folder = a.out/f'donor_{j}'
            folder.mkdir()
            transports[j] = Transport(folder/'communication.jsonl', [(i, j), (j, i)],
                Protocol(max_incoming_bytes=32*1024*1024, max_outgoing_bytes=32*1024*1024).validate())
            delivered = transports[j].send(i, j, 'functional_gap_receiver_capsule', capsule_packet)
            replicas[j] = _make_denice_client_model(torch.load(io.BytesIO(delivered), map_location='cpu', weights_only=False), i, 'cpu')
            if complete_hash(*replicas[j]) != before:
                raise Rejected('FUNCTIONAL_GAP_REPLICA_CHANGED')
            donor_views[j] = CurrentCalibrationData(a.cal_store, j, 1, role_sha)
            donor_fit[j] = current_split(donor_views[j], 'fit')
        wire_json(transports[j], i, j, 'functional_gap_request', dict(pair=pair, receiver_function=before))
        donor_available = availability_shadow(*replicas[j], c, 1)
        local_evidence = endpoint_comparison(replicas[j], donor_available, donor_fit[j], c, list(range(12)), 'cpu')
        donor_evidence = wire_json(transports[j], j, i, 'functional_gap_FIT_summary', local_evidence)
        decision = classify(donor_evidence, receiver_probe[c], c)
        record = dict(pair=pair, donor_FIT=donor_evidence, receiver_FIT=receiver_probe[c], **decision)
        probes.append(record)
        write_json(a.out/'FIT_discovery.json', probes)
        print(f'Functional discovery: {i} <- {j}, class={c}, gap={decision["gap"]}, recall={donor_evidence["stats"]["recall"]}, current risk={decision["observed_receiver_current_risk_pass"]}', flush=True)
        del donor_available
        gc.collect()
    # Choose once using FIT. A failed HOLDOUT cannot trigger a new donor trial.
    selected, protection = select_probes(probes)
    write_json(a.out/'current_FIT_protection.json', protection)
    write_json(a.out/'selection_before_BASE_or_HOLDOUT.json', selected)
    results = []
    for probe in selected:
        pair = probe['pair']
        j, c = pair['donor'], pair['class_id']
        record = dict(pair=pair, gap=probe['gap'], trial_training_run=False,
            BASE_reads=0, historical_raw_CAL_reads=0,
            installation_authorized=False, native_smoke_authorized=False)
        if probe['gap'] == 'registration_gap' and protection[c]['current_negative_veto']:
            record.update(decision='skip_registration_current_negative_veto',
                current_HOLDOUT_opened=False, protection=protection[c])
            results.append(record)
            continue
        if not probe['observed_receiver_current_risk_pass'] or probe['gap'] == 'unknown_gap':
            record.update(decision='skip_insufficient_or_unsafe_FIT', current_HOLDOUT_opened=False)
            results.append(record)
            continue
        if probe['gap'] == 'registration_gap':
            reference = (model, router)
            candidate = available[c]
            action = 'registration'
        else:
            # Existing own BASE support is not permission to overwrite a row.
            if pair['receiver_current_owned_BASE']:
                record.update(decision='skip_owned_class_requires_local_registration_review')
                results.append(record)
                continue
            bv = CurrentBaseData(a.base_store, i, 1, role_sha)
            dv = CurrentBaseData(a.base_store, j, 1, role_sha)
            dp = stratified_cap(dv.current_pool(j, 'base', current), current, RULES['cap_BASE_per_class'], 42+j)
            rp = stratified_cap(bv.current_pool(i, 'base', current), current, RULES['cap_BASE_per_class'], 42+i)
            proposed = donor_proposal(replicas[j][0], dp, c, list(range(12)), 'cpu')
            packet = update_packet(proposed['weight'], proposed['bias'], dict(pair=pair,
                receiver_sha256=before, role_manifest_sha256=role_sha, task=1, fit_role='current donor BASE'))
            update = read_update(transports[j].send(j, i, 'functional_gap_trained_readout', packet))
            integrated = receiver_integration(model, rp, c, list(range(12)), update['weight'], update['bias'], 'cpu')
            candidate = shadow(model, router, c, 1, integrated['weight'], integrated['bias'])
            reference = available[c]
            action = 'transfer'
            record.update(trial_training_run=True, BASE_reads=2,
                current_BASE_access=[bv.access_log, dv.access_log], capability_bytes=len(packet))
            folder = a.out/f'trial_receiver_{i}_donor_{j}_class_{c}'
            folder.mkdir()
            torch.save(dict(weight=integrated['weight'], bias=integrated['bias'], metadata=update['metadata']), folder/'trained_row.pt')
            del dp, rp
        binding = dict(pair=pair, action=action, reference_function=complete_hash(*reference),
            candidate_function=complete_hash(*candidate), parameters_frozen=True,
            transfer_reference='availability_only' if action == 'transfer' else 'closed_mask')
        write_json(a.out/f'candidate_{c}_lock_before_HOLDOUT.json', binding)
        # Endpoint emulator: send candidate parameters to donor for HOLDOUT
        # scoring if training occurred; never send donor raw examples back.
        if action == 'transfer':
            post_packet = update_packet(integrated['weight'], integrated['bias'], binding)
            restored = read_update(transports[j].send(i, j, 'integrated_readout_for_donor_HOLDOUT', post_packet))
            donor_candidate = shadow(*replicas[j], c, 1, restored['weight'], restored['bias'])
            donor_reference = availability_shadow(*replicas[j], c, 1)
        else:
            donor_candidate = availability_shadow(*replicas[j], c, 1)
            donor_reference = replicas[j]
        donor_H = endpoint_comparison(donor_reference, donor_candidate, current_split(donor_views[j], 'holdout'), c, list(range(12)), 'cpu')
        donor_H = wire_json(transports[j], j, i, 'functional_gap_HOLDOUT_summary', donor_H)
        receiver_H = endpoint_comparison(reference, candidate, current_split(rv, 'holdout'), c, list(range(12)), 'cpu')
        protected_H = []
        if action == 'transfer':
            # Every previously observed peer-negative risk belongs in the new
            # candidate's qualification, even if another donor was selected.
            for owner in sorted(donor_views):
                if owner == j:
                    continue
                restored = read_update(transports[owner].send(i, owner, 'integrated_readout_for_protected_HOLDOUT', post_packet))
                witness_candidate = shadow(*replicas[owner], c, 1, restored['weight'], restored['bias'])
                witness_reference = availability_shadow(*replicas[owner], c, 1)
                local = endpoint_comparison(witness_reference, witness_candidate,
                    current_split(donor_views[owner], 'holdout'), c, list(range(12)), 'cpu')
                protected_H.append(wire_json(transports[owner], owner, i,
                    'protected_HOLDOUT_summary', local))
                del witness_candidate, witness_reference
        qualification = assess_holdout(donor_H, receiver_H, action,
            dict(status='unknown', independent=False,
                reason='reused Task1 CAL offline replay; no independently qualified cumulative risk evidence'), protected_H)
        record.update(binding=binding, donor_HOLDOUT=donor_H, receiver_HOLDOUT=receiver_H,
            additional_protected_HOLDOUT=protected_H,
            qualification=qualification, current_HOLDOUT_opened=True,
            decision=qualification['decision'])
        results.append(record)
        print(f'Functional discovery class={c}: action={action}, holdout recall={donor_H["stats"]["recall"]}, net rescue={qualification["net_rescue"]}, installation=False', flush=True)
        del donor_candidate, donor_reference
        gc.collect()
    if complete_hash(model, router) != before or any(complete_hash(*p) != before for p in replicas.values()):
        raise Rejected('FUNCTIONAL_GAP_LIVE_OR_REPLICA_MUTATED')
    report = dict(completed=True, protocol=protocol, probes=probes, selected=selected, results=results,
        current_FIT_protection=protection,
        metadata_candidates=len(candidates), probe_count=len(probes),
        gap_counts={name: sum(p['gap'] == name for p in probes) for name in ('registration_gap', 'functional_gap', 'unknown_gap')},
        trial_training_runs=sum(r['trial_training_run'] for r in results),
        communication={str(j): t.summary() for j, t in transports.items()},
        total_application_egress_bytes=sum(t.summary()['application_egress_bytes'] for t in transports.values()),
        setup_capsule_bytes_each_donor=len(capsule_packet), raw_examples_transmitted=0,
        current_CAL_access={str(o): v.access_log for o, v in [(i, rv)] + list(donor_views.items())},
        old_raw_CAL_reads=0, FIT_role_reads=0, validation_reads=0, final_test_opened=False,
        source_receiver_unchanged=True, production_runner_modified=False,
        production_install_authorized=False, native_survival_authorized=False, full_training_started=False)
    write_json(a.out/'completion.json', report)
    write_json(a.publish, report)
    print(json.dumps(dict(completed=True, candidates=len(candidates), gaps=report['gap_counts'],
        trial_training_runs=report['trial_training_runs'], native_survival_authorized=False), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'roles', 'cal-store', 'base-store', 'out', 'publish'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):
        run(a)
