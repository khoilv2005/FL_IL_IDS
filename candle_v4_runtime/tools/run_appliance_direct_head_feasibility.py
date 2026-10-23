"""Locked three-case direct-head development pilot, no optimizer or replay.

Run as python -m tools.run_appliance_direct_head_feasibility prepare|run ...
CAL arrays stay inside owner endpoints; only bound aggregate receipts cross wire.
"""
import argparse
import base64
import gc
import io
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch
from threadpoolctl import threadpool_limits

from appliance.config import Rejected
from appliance.current_calibration_data import CurrentCalibrationData
from appliance.direct_head_contract import RULES, FIXTURES, VERSION
from appliance.direct_head_codec import export_head, encode, decode
from appliance.direct_head_aggregation import aggregate
from appliance.direct_head_transaction import HeadTransaction
from appliance.direct_head_verification import (scoped_pool, evaluate_functions,
    validate_receipts, select_eta, accept)
from appliance.functional_gap_discovery import availability_shadow
from appliance.runner import load_input
from appliance.selector import lookup, recorded_graph
from appliance.state import complete_hash, write_json, boundary_hash
from appliance.transport import Transport
from eval_checkpoint import _make_denice_client_model
from fed_learning.data.denice_clean_roles import file_sha256
from fed_learning.training.checkpoint_state import snapshot_denice_state

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [Path(__file__).relative_to(ROOT).as_posix()] + [
    'appliance/'+name+'.py' for name in ('direct_head_contract', 'direct_head_codec',
        'direct_head_aggregation', 'direct_head_transaction', 'direct_head_verification',
        'functional_gap_discovery', 'evaluate', 'state', 'current_calibration_data',
        'runner', 'transport', 'selector')] + ['eval_checkpoint.py',
        'fed_learning/training/checkpoint_state.py', 'fed_learning/training/denice_eval.py',
        'fed_learning/models/denice_model.py', 'fed_learning/models/nice_model.py']


def source_hashes():
    return {n: file_sha256(ROOT/n) for n in SOURCES}


def metadata(ckpt, manifest, cid, task):
    alg = lookup(ckpt['client_algorithm_states'], cid)
    alg = alg.get('denice', alg)
    counts = manifest['clients'][str(cid)]['role_class_counts']
    detector = alg['context_detector']
    return dict(owner=cid, task=task, base=counts['base'],
        calibration=counts['calibration'], ranks=list(map(int, alg['neuron_ages']['fc2'])),
        available=detector.get('episode_classes', {}).get(task,
            detector.get('episode_classes', {}).get(str(task), [])),
        clean=not (alg.get('adapter_registry') or alg.get('appliance_guarded_head_entries')
            or alg.get('local_classifier') or alg.get('continual_width', 0)),
        task_freeze_layers=alg.get('task_freeze_layers', []),
        router_mode=detector['router_mode'], refresh_task=detector.get('router_last_refresh_task'))


def prepare(a):
    a.out.mkdir(parents=True, exist_ok=False)
    role_sha = file_sha256(a.roles/'role_manifest.json')
    manifest = json.loads((a.roles/'role_manifest.json').read_text(encoding='utf-8'))
    store_path = a.cal_store/'calibration_store_manifest.json'
    store = json.loads(store_path.read_text(encoding='utf-8'))
    if store['role_manifest_sha256'] != role_sha:
        raise Rejected('DIRECT_HEAD_ROLE_STORE_MISMATCH')
    sources, cases = {}, []
    for task, path in ((1, a.task1), (2, a.task2)):
        ckpt, hashes = load_input(path, task, RULES['round'])
        cfg = ckpt['config']
        if cfg.get('denice_cl_method') != 'legacy' or cfg.get('denice_data_roles_sha256') != role_sha or cfg.get('denice_similarity_threshold') != .8:
            raise Rejected('DIRECT_HEAD_LEGACY_ROLE_XI_MISMATCH')
        if cfg.get('denice_clustering_mode') != 'paper':
            raise Rejected('DIRECT_HEAD_PAPER_GRAPH_REQUIRED')
        seen = sorted(c for t, cs in store['task_classes'].items() if int(t) <= task for c in cs)
        if sorted(ckpt['seen_classes']) != seen:
            raise Rejected('DIRECT_HEAD_CLASS_MAPPING_MISMATCH')
        groups, alphas = recorded_graph(ckpt, task, RULES['round'])
        sources[str(task)] = dict(path=str(path.resolve()), **hashes, seen=seen,
            current=store['task_classes'][str(task)], router=cfg.get('denice_router_mode'))
        for t, receiver, target, donors in FIXTURES:
            if t != task:
                continue
            r = metadata(ckpt, manifest, receiver, task)
            ds = [metadata(ckpt, manifest, j, task) for j in donors]
            reasons = []
            if r['ranks'][target] != 0 or r['base'].get(str(target), 0) or target in r['available']:
                reasons.append('receiver_slot_or_registration_not_eligible')
            if 'fc2' in r['task_freeze_layers']:
                reasons.append('receiver_task_wide_FC2_freeze')
            if not all(m['clean'] and m['refresh_task'] == task for m in (r, *ds)):
                reasons.append('endpoint_dependency_or_router_not_eligible')
            if target not in sources[str(task)]['current']:
                reasons.append('target_not_current')
            for j, d in zip(donors, ds):
                if j not in groups[receiver] or alphas[receiver].get(j, 0) <= 0 or d['ranks'][target] < 2 or not d['base'].get(str(target), 0):
                    reasons.append(f'donor_{j}_not_eligible')
            cases.append(dict(task=task, receiver=receiver, class_id=target, donors=list(donors),
                alphas={str(j): alphas[receiver].get(j, 0.) for j in donors},
                receiver_metadata=r, donor_metadata=ds, eligible=not reasons, skip_reasons=reasons))
        del ckpt
        gc.collect()
    protocol = dict(version=VERSION, rules=RULES, sources=sources, cases=cases,
        role_sha256=role_sha, cal_store_sha256=file_sha256(store_path), source_hashes=source_hashes(),
        controls=['baseline', 'availability', 'mask_only', 'single_1', 'single_2'],
        primary='alpha-only Head-Agg; q=1; eta selected on FIT',
        eta_selection='current risk and adequate support, then worst-donor recall, net rescue, smallest eta',
        selection_scope='fixed development fixtures from plan; no substitutions or donor training',
        risk_policy='empirical_current_only_with_explicit_old_risk_unknown',
        historical_raw_reads=False, test_opened=False, native_runner_modified=False,
        candidate_capsule_policy='one exact receiver reference per selected donor; no raw reference memory',
        prior_exposure='fixtures and current-role development were used in previous audits; not untouched confirmation')
    write_json(a.out/'protocol_before_CAL.json', protocol)
    write_json(a.out/'protocol_lock.json', dict(sha256=file_sha256(a.out/'protocol_before_CAL.json')))
    print('Metadata-only lock:', [(c['task'], c['receiver'], c['class_id'], c['eligible'], c['skip_reasons']) for c in cases], flush=True)


class BudgetTransport(Transport):
    def send(self, sender, receiver, kind, value):
        packet = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, allow_nan=False).encode()
        if sum(self.sent.values())+len(packet) > RULES['application_bytes']:
            raise Rejected('DIRECT_HEAD_TOTAL_BYTE_BUDGET')
        return super().send(sender, receiver, kind, packet)


def wire(t, sender, receiver, kind, value):
    return json.loads(t.send(sender, receiver, kind, value))


def construct(reference, case, packets, specs):
    model, router = reference
    target, task = case['class_id'], case['task']
    weights = {int(k): float(v) for k, v in case['alphas'].items()}
    combined = aggregate(packets, weights)
    funcs = {'baseline': reference, 'availability': availability_shadow(model, router, target, task)}
    for name, spec in specs.items():
        selected = packets if spec['source'] == 'aggregate' else [packets[spec['source']]]
        row = combined if spec['source'] == 'aggregate' else aggregate(selected, weights)
        tx = HeadTransaction(model, router, target, task, case['receiver_metadata']['base'].get(str(target), 0))
        funcs[name] = tx.stage(row, spec['eta'], mask_only=spec.get('mask_only', False))
    return funcs


def endpoint(reference, case, packet_bytes, specs, view, split, seen, device, batch_size):
    # Decode every received head again; no donor model lookup at verification.
    packets = [decode(p) for p in packet_bytes]
    functions = construct(reference, case, packets, specs)
    pool = scoped_pool(view, split)
    receipt = evaluate_functions(functions, pool, case['class_id'], seen, device, batch_size)
    del pool, functions
    return receipt


def raw_reference_forbidden(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('reference_input_memory', 'reference_inputs', 'replay_buffer') and item is not None and len(item):
                raise Rejected('DIRECT_HEAD_RAW_REFERENCE_FORBIDDEN')
            raw_reference_forbidden(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            raw_reference_forbidden(item)


def run(a):
    protocol = json.loads((a.out/'protocol_before_CAL.json').read_text(encoding='utf-8'))
    lock = json.loads((a.out/'protocol_lock.json').read_text(encoding='utf-8'))
    if (file_sha256(a.out/'protocol_before_CAL.json') != lock['sha256'] or
            protocol['version'] != VERSION or protocol['rules'] != RULES or
            protocol['source_hashes'] != source_hashes() or
            file_sha256(a.roles/'role_manifest.json') != protocol['role_sha256'] or
            file_sha256(a.cal_store/'calibration_store_manifest.json') != protocol['cal_store_sha256']):
        raise Rejected('DIRECT_HEAD_LOCK_OR_SOURCE_CHANGED')
    execution = a.out/'execution'
    execution.mkdir(exist_ok=False)  # A rerun cannot silently re-open HOLDOUT.
    t = BudgetTransport(execution/'communication.jsonl', [], SimpleNamespace(
        max_outgoing_bytes=RULES['application_bytes'], max_incoming_bytes=RULES['application_bytes']))
    started, results, accesses = time.monotonic(), [], []
    error, ckpt, cached_task = None, None, None
    def check_time():
        if time.monotonic()-started >= RULES['runtime_seconds']:
            raise Rejected('DIRECT_HEAD_RUNTIME_BUDGET')
    try:
        for position, case in enumerate(protocol['cases']):
            check_time()
            print(f'Direct head case {position+1}/3: task={case["task"]}, receiver={case["receiver"]}, class={case["class_id"]}', flush=True)
            if not case['eligible']:
                results.append(dict(case=case, status='DEFERRED', reason=case['skip_reasons']))
                continue
            task, receiver, target = case['task'], case['receiver'], case['class_id']
            source = protocol['sources'][str(task)]
            if cached_task != task:
                del ckpt
                gc.collect()
                ckpt, hashes = load_input(source['path'], task, RULES['round'])
                if any(source[k] != hashes[k] for k in hashes):
                    raise Rejected('DIRECT_HEAD_CHECKPOINT_CHANGED')
                cached_task = task
            groups, alphas = recorded_graph(ckpt, task, RULES['round'])
            r = _make_denice_client_model(ckpt, receiver, a.device)
            original = complete_hash(*r)
            ds = case['donors']
            t.edges.update((x, y) for j in groups[receiver] for x, y in ((receiver, j), (j, receiver)))
            manifest = json.loads((a.roles/'role_manifest.json').read_text(encoding='utf-8'))
            for j in groups[receiver]:
                wire(t, receiver, j, 'metadata_request', dict(task=task, class_id=target))
                wire(t, j, receiver, 'class_metadata', metadata(ckpt, manifest, j, task))
            authority = dict(role_sha256=protocol['role_sha256'], terminal_sha256=source['terminal_sha256'],
                graph_sha256=source['graph_sha256'], class_mapping=source['current'])
            packets, packet_bytes = [], []
            for j in ds:
                if j not in groups[receiver] or alphas[receiver][j] != case['alphas'][str(j)]:
                    raise Rejected('DIRECT_HEAD_GRAPH_CHANGED')
                donor = _make_denice_client_model(ckpt, j, a.device)
                before = complete_hash(*donor)
                packet = export_head(*donor, j, task, target, authority)
                data = t.send(j, receiver, 'existing_classifier_row', encode(packet))
                decoded = decode(data)
                if complete_hash(*donor) != before:
                    raise Rejected('DIRECT_HEAD_EXPORT_MUTATED_DONOR')
                packets.append(decoded)
                packet_bytes.append(data)
                del donor
            alg = snapshot_denice_state(*r)
            raw_reference_forbidden(alg)
            capsule = dict(config=ckpt['config'], task=task,
                client_model_states={receiver: {k: v.detach().cpu() for k, v in r[0].state_dict().items()}},
                client_algorithm_states={receiver: {'denice': alg}})
            stream = io.BytesIO()
            torch.save(capsule, stream)
            replicas, views = {}, {}
            for j in ds:
                received = t.send(receiver, j, 'cold_receiver_function_reference', stream.getvalue())
                replicas[j] = _make_denice_client_model(torch.load(io.BytesIO(received), map_location='cpu', weights_only=False), receiver, a.device)
                if complete_hash(*replicas[j]) != original:
                    raise Rejected('DIRECT_HEAD_EXACT_REFERENCE_RESTORE')
            for owner in (receiver, *ds):
                views[owner] = CurrentCalibrationData(a.cal_store, owner, task, protocol['role_sha256'])
            fit_specs = {'mask_only': dict(source='aggregate', eta=1., mask_only=True)}
            for prefix, src in (('agg', 'aggregate'), ('single1', 0), ('single2', 1)):
                fit_specs.update({f'{prefix}_{eta:g}': dict(source=src, eta=eta) for eta in RULES['eta_grid']})
            def evaluate(specs, split):
                receipts = {receiver: endpoint(r, case, packet_bytes, specs, views[receiver], split, source['seen'], a.device, a.batch_size)}
                for j in ds:
                    request = wire(t, receiver, j, f'{split}_candidate_request', dict(
                        receiver_reference_function=original, case=case, specs=specs,
                        heads=[base64.b64encode(p).decode('ascii') for p in packet_bytes]))
                    if request['receiver_reference_function'] != complete_hash(*replicas[j]):
                        raise Rejected('DIRECT_HEAD_CACHE_VERSION_MISMATCH')
                    result = endpoint(replicas[j], request['case'],
                        [base64.b64decode(p, validate=True) for p in request['heads']], request['specs'],
                        views[j], split, source['seen'], a.device, a.batch_size)
                    receipts[j] = wire(t, j, receiver, f'{split}_count_receipt', result)
                expected = {name: complete_hash(*fn) for name, fn in construct(r, case, packets, specs).items()}
                validate_receipts(receipts, receiver, ds, task, protocol['role_sha256'], split, expected)
                return receipts, expected
            check_time()
            fit, _ = evaluate(fit_specs, 'fit')
            write_json(execution/f'case_{position}_FIT.json', fit)
            gaps = {j: ('UNKNOWN' if fit[j]['variants']['availability']['stats']['positive'] < RULES['min_positive'] else
                'registration_gap' if fit[j]['variants']['availability']['stats']['recall'] >= RULES['recall'] else 'functional_gap') for j in ds}
            if any(v != 'functional_gap' for v in gaps.values()):
                results.append(dict(case=case, status='DEFERRED', gap=gaps, HOLDOUT_opened=False,
                    reason='both fixed donors must confirm a functional gap on sufficient FIT'))
                for owner, view in views.items():
                    accesses.append(dict(case=position, owner=owner, access_log=view.access_log))
                continue
            eta = select_eta(fit, 'agg', receiver, ds)
            singles = [select_eta(fit, p, receiver, ds) for p in ('single1', 'single2')]
            final_specs = dict(head_agg=dict(source='aggregate', eta=eta),
                mask_only=dict(source='aggregate', eta=eta, mask_only=True),
                single_1=dict(source=0, eta=singles[0]), single_2=dict(source=1, eta=singles[1]))
            final_functions = construct(r, case, packets, final_specs)
            final_lock = dict(case=case, specs=final_specs, function_hashes={n: complete_hash(*fn) for n, fn in final_functions.items()},
                head_checksums=[__import__('hashlib').sha256(p).hexdigest() for p in packet_bytes],
                FIT_gap=gaps, no_HOLDOUT_used_for_selection=True)
            write_json(execution/f'case_{position}_lock_before_HOLDOUT.json', final_lock)
            check_time()
            holdout, _ = evaluate(final_specs, 'holdout')
            write_json(execution/f'case_{position}_HOLDOUT.json', holdout)
            qualification = accept(holdout, receiver, ds)
            tx = HeadTransaction(*r, target, task)
            tx.stage(aggregate(packets, {int(k): v for k, v in case['alphas'].items()}), eta)
            if qualification['status'] == 'EMPIRICAL_CURRENT_PASS':
                tx.verify(qualification)
                committed, provenance = tx.commit()
                if complete_hash(*committed) != final_lock['function_hashes']['head_agg']:
                    raise Rejected('DIRECT_HEAD_COMMIT_HASH')
                # Save exact pilot state for P3 only. Not a packet replay archive.
                torch.save(dict(model_state=committed[0].state_dict(),
                    algorithm_state=snapshot_denice_state(*committed), provenance=provenance),
                    execution/f'case_{position}_accepted_model.pt')
            tx.rollback()
            if complete_hash(*r) != original or any(complete_hash(*fn) != original for fn in replicas.values()):
                raise Rejected('DIRECT_HEAD_REFERENCE_CHANGED')
            results.append(dict(case=case, eta=eta, single_etas=singles,
                packet_bytes=[len(p) for p in packet_bytes], receiver_reference_bytes=len(stream.getvalue()),
                donor_boundary_matches=[p['metadata']['boundary_hash'] == boundary_hash(r[0]) for p in packets],
                HOLDOUT_opened=True, qualification=qualification, status=qualification['status'],
                rollback_exact=True, donor_optimizer_steps=0, receiver_optimizer_steps=0))
            for owner, view in views.items():
                accesses.append(dict(case=position, owner=owner, access_log=view.access_log))
            write_json(execution/'results_progress.json', results)
            print(f'Case {position+1}: {qualification["status"]}, recalls={qualification["required_donor_recalls"]}, gains={qualification["gains"]}', flush=True)
            del r, replicas, views, fit, holdout, final_functions, packets, packet_bytes, capsule
            gc.collect()
    except Exception as exc:
        error = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        summary = dict(version=VERSION, completed_execution=error is None,
            bounded_decision='GO_P3' if any(r['status'] == 'EMPIRICAL_CURRENT_PASS' for r in results) else
                'NO_GO' if error is None and len(results) == len(protocol['cases']) else 'INCOMPLETE',
            results=results, error=error, seconds=time.monotonic()-started,
            communication=t.summary(), communication_by_kind={kind: sum(r['bytes'] for r in t.records if r['kind'] == kind)
                for kind in sorted({r['kind'] for r in t.records})},
            native_DeNICE_round_bytes_measured=False, raw_historical_reads=0, BASE_reads=0,
            test_reads=0, donor_optimizer_steps=0, receiver_optimizer_steps=0,
            old_risk='UNKNOWN', native_runner_modified=False, full_training_authorized=False)
        write_json(execution/'CAL_access.json', accesses)
        write_json(execution/'completion.json', summary)
        print('Completion:', summary['bounded_decision'], 'bytes=', sum(t.sent.values()), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['prepare', 'run'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--roles', type=Path, required=True)
    p.add_argument('--cal-store', type=Path, required=True)
    p.add_argument('--task1', type=Path)
    p.add_argument('--task2', type=Path)
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=256)
    a = p.parse_args()
    if a.batch_size < 1 or a.phase == 'prepare' and (not a.task1 or not a.task2):
        p.error('prepare requires --task1 and --task2; batch size must be positive')
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        prepare(a) if a.phase == 'prepare' else run(a)


if __name__ == '__main__':
    main()
