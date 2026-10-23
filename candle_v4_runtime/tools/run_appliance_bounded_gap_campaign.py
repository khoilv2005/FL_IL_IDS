"""Prelocked, metadata-first APPLIANCE functional-gap search (development only).

prepare reads checkpoint/role metadata only. run cannot add cases, donors,
learning configurations or retries after seeing FIT/HOLDOUT outcomes.
"""
import argparse
from collections import Counter
import gc
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from appliance.active_pair_calibration import split_counts
from appliance.config import Protocol, Rejected
from appliance.current_base_data import CurrentBaseData
from appliance.current_calibration_data import CurrentCalibrationData
from appliance.functional_gap_discovery import (RULES as GAP_RULES,
    availability_shadow, current_split, endpoint_comparison, classify,
    select_probes, assess_holdout)
from appliance.runner import load_input
from appliance.selector import lookup, recorded_graph
from appliance.state import complete_hash, write_json
from appliance.train_time_transfer import (RULES, donor_proposal,
    receiver_integration, shadow, stratified_cap, update_packet, read_update)
from appliance.transport import Transport
from eval_checkpoint import _make_denice_client_model
from fed_learning.data.denice_clean_roles import file_sha256
from fed_learning.training.checkpoint_state import snapshot_denice_state


LIMITS = dict(tasks=[1, 2], round=19, excluded_receivers=[2], cases_per_task=3,
    donors_per_case=2, trial_updates=2, application_bytes=128*1024*1024,
    runtime_seconds=900, CAL_rows_per_class=512, seed=42)
SOURCE_FILES = [__file__, 'appliance/functional_gap_discovery.py',
    'appliance/train_time_transfer.py', 'appliance/current_calibration_data.py',
    'appliance/current_base_data.py', 'appliance/evaluate.py', 'eval_checkpoint.py',
    'fed_learning/training/checkpoint_state.py']


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {Path(n).resolve().relative_to(root).as_posix(): file_sha256(Path(n))
        for n in SOURCE_FILES}


def state(ckpt, cid):
    value = lookup(ckpt['client_algorithm_states'], cid)
    return value.get('denice', value)


def clean_endpoint(alg):
    return not (alg.get('adapter_registry') or alg.get('appliance_guarded_head_entries')
        or alg.get('continual_head'))


def metadata(ckpt, manifest, owner, task, classes):
    alg = state(ckpt, owner)
    counts = manifest['clients'][str(owner)]['role_class_counts']
    return dict(owner=owner, task=task,
        base={str(c): int(counts['base'].get(str(c), 0)) for c in classes},
        calibration={str(c): int(counts['calibration'].get(str(c), 0)) for c in classes},
        ranks={str(c): int(alg['neuron_ages']['fc2'][c]) for c in classes},
        router_last_refresh_task=alg['context_detector'].get('router_last_refresh_task'),
        available=list(map(int, alg['context_detector'].get('episode_classes', {}).get(
            task, alg['context_detector'].get('episode_classes', {}).get(str(task), [])))),
        adapter_and_imported_route_free=clean_endpoint(alg))


def prepare(a):
    a.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((a.roles/'role_manifest.json').read_text(encoding='utf-8'))
    role_sha = file_sha256(a.roles/'role_manifest.json')
    store = json.loads((a.cal_store/'calibration_store_manifest.json').read_text(encoding='utf-8'))
    cases, sources, enumeration = [], {}, {}
    for task, path in zip(LIMITS['tasks'], (a.task1, a.task2)):
        ckpt, hashes = load_input(path, task, LIMITS['round'])
        if (ckpt['config'].get('denice_cl_method') != 'legacy' or
                ckpt['config']['denice_data_roles_sha256'] != role_sha):
            raise Rejected('BOUNDED_CAMPAIGN_LEGACY_ROLE_MISMATCH')
        classes = store['task_classes'][str(task)]
        seen = sorted(c for t, cs in store['task_classes'].items() if int(t) <= task for c in cs)
        if sorted(ckpt['seen_classes']) != seen:
            raise Rejected('BOUNDED_CAMPAIGN_SEEN_SCOPE_MISMATCH')
        groups, alphas = recorded_graph(ckpt, task, LIMITS['round'])
        md = {i: metadata(ckpt, manifest, i, task, classes) for i in groups}
        options = []
        for c in sorted(classes):
            for i in sorted(groups):
                r = md[i]
                rf = sum(split_counts(n)[0] for k, n in r['calibration'].items() if int(k) != c)
                rh = sum(split_counts(n)[2] for k, n in r['calibration'].items() if int(k) != c)
                if (i in LIMITS['excluded_receivers'] or not r['adapter_and_imported_route_free']
                        or c in r['available'] or r['base'][str(c)] or r['ranks'][str(c)] != 0
                        or not sum(n for k, n in r['base'].items() if int(k) != c)
                        or min(rf, rh) < GAP_RULES['min_negative']
                        or r['router_last_refresh_task'] != task):
                    continue
                donors = []
                for j in groups[i]:
                    d = md[j]
                    pf, _, ph = split_counts(d['calibration'][str(c)])
                    if (not d['adapter_and_imported_route_free'] or not d['base'][str(c)]
                            or d['ranks'][str(c)] < 2 or d['router_last_refresh_task'] != task
                            or not sum(n for k, n in d['base'].items() if int(k) != c)
                            or min(pf, ph) < GAP_RULES['min_positive']):
                        continue
                    breadth = sum(n > 0 for k, n in d['calibration'].items() if int(k) != c)
                    donors.append(dict(receiver=i, donor=j, class_id=c, task=task,
                        alpha=alphas[i][j], donor_current_owned_BASE=d['base'][str(c)],
                        receiver_current_owned_BASE=0, donor_rank=d['ranks'][str(c)],
                        metadata_positive_FIT=pf, metadata_positive_HOLDOUT=ph,
                        metadata_negative_class_breadth=breadth))
                donors.sort(key=lambda p: (-p['metadata_negative_class_breadth'],
                    -min(p['metadata_positive_FIT'], LIMITS['CAL_rows_per_class']), p['donor']))
                if donors:
                    options.append(dict(task=task, receiver=i, class_id=c, donors=donors,
                        metadata_receiver_negative_FIT=rf, metadata_receiver_negative_HOLDOUT=rh))
        # One case per receiver; distribute classes before taking a second case
        # of the same class. No prediction, target recall or test label ranking.
        chosen, used = [], set()
        for c in sorted(classes):
            for option in options:
                if option['class_id'] == c and option['receiver'] not in used:
                    chosen.append(dict(option, donors=option['donors'][:LIMITS['donors_per_case']]))
                    used.add(option['receiver'])
                    break
            if len(chosen) == LIMITS['cases_per_task']:
                break
        cases.extend(chosen)
        sources[str(task)] = dict(path=str(path.resolve()), **hashes,
            seen=seen, current=classes, xi=ckpt['config'].get('denice_similarity_threshold'),
            eligible_endpoint_count=sum(v['adapter_and_imported_route_free'] for v in md.values()))
        enumeration[str(task)] = dict(structural_receiver_class_options=len(options),
            structurally_eligible_donor_class_pairs=sum(len(o['donors']) for o in options),
            counts_are_not_unique_evidence=True)
        del ckpt
        gc.collect()
    protocol = dict(version='appliance_bounded_gap_campaign_v1', limits=LIMITS,
        sources=sources, cases=cases, metadata_enumeration=enumeration,
        role_sha256=role_sha, cal_store_sha256=file_sha256(a.cal_store/'calibration_store_manifest.json'),
        base_store_sha256=file_sha256(a.base_store/'base_store_manifest.json'),
        source_hashes=source_hashes(), discovery_logic_changed=False,
        learning_rules=RULES, gap_rules=GAP_RULES, metadata_only_preselection=True,
        selection='task ascending, class ascending, first eligible distinct receiver ID; donor negative-class breadth then capped positive FIT count then donor ID',
        trial_selection='first FIT-selected functional gap in locked case order, current risk pass, no selected-peer negative veto; at most two trials',
        stop='exhaust locked cases/trials, total bytes or time; no substitution, no hyperparameter retries',
        transfer_reference='Availability-only on identical owner CAL-HOLDOUT rows',
        broad_graph_safety_established=False, independently_qualified_cumulative_old_risk=False,
        prior_exposure_ledger_complete=False, interpretation='retrospective development, not independent confirmation',
        historical_raw_CAL_reads_authorized=False, automatic_install_enabled=False,
        final_test_opened=False, production_runner_modified=False)
    write_json(a.out/'protocol_before_CAL.json', protocol)
    write_json(a.out/'protocol_lock.json', dict(sha256=file_sha256(a.out/'protocol_before_CAL.json')))
    print('Locked cases:', [(c['task'], c['receiver'], c['class_id'], [p['donor'] for p in c['donors']]) for c in cases], flush=True)


class BudgetTransport(Transport):
    def send(self, sender, receiver, kind, value):
        packet = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True).encode()
        if sum(self.sent.values()) + len(packet) > LIMITS['application_bytes']:
            raise Rejected('CAMPAIGN_TOTAL_BYTE_BUDGET')
        return super().send(sender, receiver, kind, packet)


def wire(transport, sender, receiver, kind, value):
    return json.loads(transport.send(sender, receiver, kind,
        json.dumps(value, sort_keys=True, allow_nan=False).encode()))


def scoped_pool(view, split, classes):
    pool = current_split(view, split)
    pool = stratified_cap(pool, classes, LIMITS['CAL_rows_per_class'], LIMITS['seed']+view.client_id)
    pool['binding'] = dict(pool['binding'], campaign_cap_per_class=LIMITS['CAL_rows_per_class'],
        campaign_selected_rows_sha256=hashlib.sha256(pool['rows'].tobytes()).hexdigest(),
        campaign_selected_unique_rows=len(pool['rows']))
    return pool


def run(a):
    protocol = json.loads((a.out/'protocol_before_CAL.json').read_text(encoding='utf-8'))
    lock = json.loads((a.out/'protocol_lock.json').read_text(encoding='utf-8'))
    if (lock['sha256'] != file_sha256(a.out/'protocol_before_CAL.json') or
            protocol['limits'] != LIMITS or protocol['source_hashes'] != source_hashes()):
        raise Rejected('CAMPAIGN_PROTOCOL_CHANGED')
    for path, expected in ((a.roles/'role_manifest.json', protocol['role_sha256']),
            (a.cal_store/'calibration_store_manifest.json', protocol['cal_store_sha256']),
            (a.base_store/'base_store_manifest.json', protocol['base_store_sha256'])):
        if file_sha256(path) != expected:
            raise Rejected('CAMPAIGN_DATA_AUTHORITY_CHANGED')
    execution = a.out/'execution'
    execution.mkdir(exist_ok=False)
    write_json(execution/'completion.json', dict(completed=False))
    manifest = json.loads((a.roles/'role_manifest.json').read_text(encoding='utf-8'))
    results, probes, accesses, trials = [], [], [], 0
    transport = BudgetTransport(execution/'communication.jsonl', [],
        Protocol(max_incoming_bytes=LIMITS['application_bytes'], max_outgoing_bytes=LIMITS['application_bytes']).validate())
    started = time.monotonic()
    stopped = 'locked_cases_exhausted'
    ckpt, loaded_task = None, None
    try:
        for position, case in enumerate(protocol['cases']):
            if time.monotonic()-started >= LIMITS['runtime_seconds']:
                stopped = 'runtime_budget_before_next_case'
                break
            task, i, c = case['task'], case['receiver'], case['class_id']
            source = protocol['sources'][str(task)]
            if task != loaded_task:
                del ckpt
                gc.collect()
                ckpt, hashes = load_input(Path(source['path']), task, LIMITS['round'])
                if any(hashes[k] != source[k] for k in hashes):
                    raise Rejected('CAMPAIGN_CHECKPOINT_CHANGED')
                loaded_task = task
            groups, alphas = recorded_graph(ckpt, task, LIMITS['round'])
            transport.edges.update((a, b) for a in (i,) for b in groups[i])
            transport.edges.update((b, i) for b in groups[i])
            # Account the metadata exchanges that precede any function capsule.
            for owner in groups[i]:
                wire(transport, i, owner, 'current_class_metadata_request', dict(task=task, classes=source['current']))
                wire(transport, owner, i, 'current_class_metadata_response', metadata(ckpt, manifest, owner, task, source['current']))
            model, router = _make_denice_client_model(ckpt, i, 'cpu')
            original = complete_hash(model, router)
            if not clean_endpoint(state(ckpt, i)):
                raise Rejected('CAMPAIGN_RECEIVER_NOT_CLEAN')
            available = availability_shadow(model, router, c, task)
            rv = CurrentCalibrationData(a.cal_store, i, task, protocol['role_sha256'])
            receiver_F = endpoint_comparison((model, router), available,
                scoped_pool(rv, 'fit', source['current']), c, source['seen'], 'cpu')
            alg = snapshot_denice_state(model, router)
            if alg['context_detector'].get('reference_input_memory'):
                raise Rejected('CAMPAIGN_RAW_REFERENCE_MEMORY')
            capsule = dict(config=ckpt['config'], task=task,
                client_model_states={i: {k: v.detach().cpu() for k, v in model.state_dict().items()}},
                client_algorithm_states={i: {'denice': alg}})
            stream = io.BytesIO()
            torch.save(capsule, stream)
            packet = stream.getvalue()
            replicas, views, local_probes = {}, {}, []
            for pair in case['donors']:
                j = pair['donor']
                if j not in groups[i] or alphas[i].get(j) != pair['alpha']:
                    raise Rejected('CAMPAIGN_GRAPH_CHANGED')
                delivered = transport.send(i, j, 'functional_gap_receiver_capsule', packet)
                replicas[j] = _make_denice_client_model(torch.load(io.BytesIO(delivered), weights_only=False), i, 'cpu')
                if complete_hash(*replicas[j]) != original:
                    raise Rejected('CAMPAIGN_REPLICA_CHANGED')
                views[j] = CurrentCalibrationData(a.cal_store, j, task, protocol['role_sha256'])
                donor_available = availability_shadow(*replicas[j], c, task)
                wire(transport, i, j, 'availability_probe_request', dict(pair=pair, receiver_function=original))
                evidence = endpoint_comparison(replicas[j], donor_available,
                    scoped_pool(views[j], 'fit', source['current']), c, source['seen'], 'cpu')
                evidence = wire(transport, j, i, 'availability_FIT_summary', evidence)
                probe = dict(pair=pair, donor_FIT=evidence, receiver_FIT=receiver_F, **classify(evidence, receiver_F, c))
                probes.append(probe)
                local_probes.append(probe)
                print(f'Bounded {position+1}/{len(protocol["cases"])} task={task} receiver={i} donor={j} class={c}: {probe["gap"]}, recall={evidence["stats"]["recall"]}, current_risk={probe["observed_receiver_current_risk_pass"]}', flush=True)
                del donor_available
            selected, protections = select_probes(local_probes)
            chosen = selected[0]
            protection = protections[c]
            result = dict(case=case, selected_pair=chosen['pair'], gap=chosen['gap'],
                protection=protection, trial_training_run=False, HOLDOUT_opened=False,
                installation_authorized=False, independent_cumulative_risk_status='unknown')
            if protection['current_negative_veto']:
                result['decision'] = 'skip_observed_negative_risk_veto'
            elif not chosen['trial_transfer_requested']:
                result['decision'] = 'registration_gap_no_learning' if chosen['gap'] == 'registration_gap' else 'skip_insufficient_FIT'
            elif trials >= LIMITS['trial_updates']:
                result['decision'] = 'skip_locked_trial_budget'
            elif time.monotonic()-started >= LIMITS['runtime_seconds']:
                result['decision'] = 'skip_runtime_budget'
            else:
                j = chosen['pair']['donor']
                dv = CurrentBaseData(a.base_store, j, task, protocol['role_sha256'])
                bv = CurrentBaseData(a.base_store, i, task, protocol['role_sha256'])
                dp = stratified_cap(dv.current_pool(j, 'base', source['current']), source['current'], RULES['cap_BASE_per_class'], 42+j)
                rp = stratified_cap(bv.current_pool(i, 'base', source['current']), source['current'], RULES['cap_BASE_per_class'], 42+i)
                trials += 1
                proposed = donor_proposal(replicas[j][0], dp, c, source['seen'], 'cpu')
                update = read_update(transport.send(j, i, 'trained_readout', update_packet(proposed['weight'], proposed['bias'], dict(pair=chosen['pair'], receiver_function=original))))
                integrated = receiver_integration(model, rp, c, source['seen'], update['weight'], update['bias'], 'cpu')
                candidate = shadow(model, router, c, task, integrated['weight'], integrated['bias'])
                binding = dict(candidate_function=complete_hash(*candidate), reference_function=complete_hash(*available),
                    parameters_frozen=True, reference='Availability-only', task=task, receiver=i, class_id=c)
                write_json(execution/f'case_{position}_lock_before_HOLDOUT.json', binding)
                torch.save(dict(weight=integrated['weight'], bias=integrated['bias'], binding=binding), execution/f'case_{position}_row.pt')
                post_packet = update_packet(integrated['weight'], integrated['bias'], binding)
                receipts = {}
                for owner, view in views.items():
                    restored = read_update(transport.send(i, owner, 'integrated_row_HOLDOUT_request', post_packet))
                    owner_candidate = shadow(*replicas[owner], c, task, restored['weight'], restored['bias'])
                    owner_reference = availability_shadow(*replicas[owner], c, task)
                    receipt = endpoint_comparison(owner_reference, owner_candidate,
                        scoped_pool(view, 'holdout', source['current']), c, source['seen'], 'cpu')
                    receipts[owner] = wire(transport, owner, i, 'integrated_HOLDOUT_summary', receipt)
                    del owner_reference, owner_candidate
                receiver_H = endpoint_comparison(available, candidate, scoped_pool(rv, 'holdout', source['current']), c, source['seen'], 'cpu')
                qualification = assess_holdout(receipts[j], receiver_H, 'transfer',
                    independent_risk=dict(status='unknown', independent=False, reason='no qualified independent cumulative old-risk receipt authority'),
                    protected_receipts=[r for o, r in receipts.items() if o != j])
                result.update(trial_training_run=True, HOLDOUT_opened=True, binding=binding,
                    donor_HOLDOUT=receipts[j], receiver_HOLDOUT=receiver_H,
                    additional_protected_HOLDOUT=[r for o, r in receipts.items() if o != j],
                    qualification=qualification, decision=qualification['decision'],
                    current_BASE_access=[dv.access_log, bv.access_log],
                    learning_history=dict(donor=proposed['donor_trace'], receiver=integrated['receiver_trace']))
                del dp, rp, candidate
                print(f'Trial task={task} receiver={i} class={c}: {qualification}', flush=True)
            if complete_hash(model, router) != original or any(complete_hash(*p) != original for p in replicas.values()):
                raise Rejected('CAMPAIGN_SOURCE_OR_REPLICA_MUTATED')
            accesses.extend(dict(task=task, owner=o, access_log=v.access_log) for o, v in [(i, rv)]+list(views.items()))
            results.append(result)
            write_json(execution/'results_progress.json', dict(probes=probes, results=results))
            del model, router, available, replicas, views, capsule, packet
            gc.collect()
    except Rejected as exc:
        if 'CAMPAIGN_TOTAL_BYTE_BUDGET' not in str(exc):
            raise
        stopped = 'total_application_byte_budget'
    summary = transport.summary()
    if summary['application_egress_bytes'] != sum(r['bytes'] for r in transport.records):
        raise Rejected('CAMPAIGN_WIRE_ACCOUNTING_MISMATCH')
    report = dict(completed=True, protocol_sha256=lock['sha256'], protocol=protocol,
        cases_completed=len(results), probes=probes, results=results,
        gap_counts=dict(Counter(p['gap'] for p in probes)), trial_training_runs=trials,
        stop_reason=stopped, elapsed_seconds=time.monotonic()-started,
        communication=summary, message_counts=dict(Counter(r['kind'] for r in transport.records)),
        capsule_bytes=sum(r['bytes'] for r in transport.records if r['kind']=='functional_gap_receiver_capsule'),
        metadata_bytes=sum(r['bytes'] for r in transport.records if r['kind'].startswith('current_class_metadata')),
        current_CAL_access=accesses, raw_examples_transmitted=0, historical_raw_CAL_reads=0,
        independent_cumulative_old_risk_established=False, production_install_authorized=False,
        native_survival_authorized=False, full_training_started=False, final_test_opened=False,
        interpretation='Bounded retrospective development search; a negative result is not a theorem about all candidates. Metadata and full setup counted; no socket/TLS cost claim.')
    write_json(execution/'completion.json', report)
    if a.publish:
        write_json(a.publish, report)
    print(json.dumps(dict(completed=True, cases=len(results), gaps=report['gap_counts'],
        trials=trials, bytes=summary['application_egress_bytes'], stop=stopped), indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', required=True, choices=['prepare', 'run'])
    for name in ('task1', 'task2', 'roles', 'cal-store', 'base-store', 'out', 'publish'):
        p.add_argument('--'+name, type=Path, required=name not in ('publish', 'task1', 'task2'))
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    with threadpool_limits(limits=1):
        if args.phase == 'prepare':
            if not args.task1 or not args.task2:
                p.error('prepare requires both --task1 and --task2')
            prepare(args)
        else:
            run(args)
