"""Retrospective graph-support simulation; no assertion of historical delivery.

Existing checkpoints have owned summaries, not per-row aggregation receipts.
Simulated receipts therefore report POTENTIAL coverage only. Validation is
development; test and raw historical CAL are never opened.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from appliance.base_sketch_shield import CurrentBaseSketchShield
from appliance.config import Protocol, Rejected
from appliance.provenance_protection import ProvenanceProtection, export_support, frozen_veto, transfer_support
from appliance.selector import lookup
from appliance.state import digest, write_json
from appliance.transport import Transport


def simulate(ckpt, graph, out, protection_only=False):
    states = {int(i): s.get('denice', s) for i, s in ckpt['client_algorithm_states'].items()}
    shields = {i: CurrentBaseSketchShield.restore(s['appliance_base_sketch_shield_state'])
               for i, s in states.items() if s.get('appliance_base_sketch_shield_state')}
    versions = {i: digest(ckpt['client_model_states'][i]) for i in shields}
    packets = {i: export_support(s, versions[i], digest(states[i]['connection_masks'])) for i, s in shields.items()}
    edges = []
    weights = {}
    for i in shields:
        a = lookup(graph['alpha_debug'], i)
        weights[i] = {int(j): float(w) for j, w in zip(a['group_ids'], a['alphas']) if int(j) != i and w > 0}
        edges += [(j, i) for j in weights[i]]
    wire = Transport(out/'simulated_wire.jsonl', edges,
                     Protocol(max_incoming_bytes=128*1024*1024, max_outgoing_bytes=128*1024*1024).validate())
    ledgers, rows = {}, []
    current = set(shields[next(iter(shields))].provenance[-1]['class_counts'])
    for i, own in shields.items():
        ledger = ProvenanceProtection(own)
        ages = np.asarray(states[i]['neuron_ages']['fc2'])
        required = sorted(int(c) for c in ckpt['seen_classes'] if str(c) not in current and ages[c] >= 2)
        for j, alpha in weights[i].items():
            if j not in shields:
                continue
            donor_ages = np.asarray(states[j]['neuron_ages']['fc2'])
            classes = ([int(c) for c, e in shields[j].memory.entries.items() if e['task'] < ckpt['task']]
                       if protection_only else
                       [c for c in required if donor_ages[c] >= 2 and str(c) in shields[j].memory.entries])
            classes = sorted(classes)
            if not classes:
                continue
            # Terminal mature rows + positive alpha are only an approximation:
            # actual row contribution/mask receipts did not exist in this run.
            receipt = dict(sender=j, receiver=i, task=int(ckpt['task']), round=int(ckpt['final_round_id']),
                kind='protection_only' if protection_only else 'aggregation', alpha=alpha, class_rows=classes,
                sender_model_version=versions[j], receiver_model_version=versions[i],
                dependency_version=packets[j]['dependency_version'])
            receipt['receipt_id'] = digest(receipt)
            transfer_support(ledger, packets[j], receipt, wire, current_task=ckpt['task'], authorized_senders=weights[i])
        coverage = ledger.coverage(required)
        rows.append(dict(receiver=i, required_old_classes=required,
            missing_owned=coverage['owned']['missing_classes'],
            potential_missing_after_peer_support=coverage['missing_protection_classes'],
            peer_origins=coverage['peer_origins']))
        ledgers[i] = ledger
    return ledgers, rows, wire.summary()


def controls(ledgers, out):
    checks = {}
    ledger = next(s for s in ledgers.values() if s.receipts)
    own_before = digest(ledger.own.state())
    state = ledger.state()
    checks['restore_exact'] = ProvenanceProtection.restore(state).state() == state
    checks['peer_not_receiver_owned'] = ledger.coverage([])['inherited_knowledge_is_receiver_owned'] is False
    checks['CAL_not_substituted'] = ledger.coverage([])['CAL_acceptance_substitution'] is False
    record = next(iter(ledger.receipts.values()))
    packet, receipt = ledger.packet_for(record), record['receipt']
    def rejected(name, packet=packet, receipt=receipt, **kwargs):
        try:
            ProvenanceProtection(ledger.own).receive(packet, receipt,
                current_task=kwargs.get('task', receipt['task']),
                authorized_senders=kwargs.get('authorized', [receipt['sender']]))
        except Rejected:
            checks[name] = True
        else:
            checks[name] = False
    rejected('unauthorized_edge', authorized=[])
    rejected('future_summary_or_receipt', task=receipt['task']-1)
    for name, mutate in {
        'CAL_claim': lambda p: p.update(CAL_acceptance_substitution=True),
        'packet_version': lambda p: p.update(version='future'),
        'source_owner': lambda p: p.update(sender=999),
        'dependency_version': lambda p: p.update(dependency_version='other'),
    }.items():
        bad = copy.deepcopy(packet); mutate(bad)
        bad['state_digest'] = digest({k: v for k, v in bad.items() if k != 'state_digest'})
        rejected(name, packet=bad)
    bad = copy.deepcopy(receipt);bad['class_rows'] = []
    bad['receipt_id'] = digest({k: v for k, v in bad.items() if k != 'receipt_id'})
    rejected('graph_without_class_receipt', receipt=bad)
    bad = copy.deepcopy(receipt);bad['alpha'] = 0.
    bad['receipt_id'] = digest({k: v for k, v in bad.items() if k != 'receipt_id'})
    rejected('zero_alpha', receipt=bad)
    before = digest(ledger.state())
    ledger.receive(packet, receipt, current_task=receipt['task'], authorized_senders=[receipt['sender']])
    checks['idempotent_delivery'] = digest(ledger.state()) == before
    required = receipt['class_rows']
    imported = next(c for c in range(34) if c not in required)
    frozen = ledger.freeze(required, imported)
    x = np.random.default_rng(10).normal(size=(64, *ledger.own.memory.sketch.input_shape)).astype(np.float32)
    checks['frozen_restore_label_blind'] = np.array_equal(frozen_veto(frozen, x), ledger.veto(x, required))
    ledger.receipts.clear()
    checks['snapshot_not_mutated_by_later_ledger_update'] = np.array_equal(
        frozen_veto(frozen, x), ProvenanceProtection.restore(state).veto(x, required))
    checks['own_provenance_unchanged'] = digest(ledger.own.state()) == own_before
    # Restore after the mutation control, before development experiments.
    ledger.receipts = ProvenanceProtection.restore(state).receipts
    report = dict(checks=checks, mismatches=sum(not v for v in checks.values()),
                  actual_receipts_verified=False, certificate_installation_authorized=False)
    write_json(out/'controls.json', report)
    if report['mismatches']:
        raise AssertionError(report)
    return report


def development(ckpt, ledgers, a, graph):
    from contextlib import ExitStack
    from appliance.guarded_head import original_local_head
    from appliance.portable_route import ProtectedRoute, transitions
    from appliance.stable_head import stable_signals
    from eval_checkpoint import _make_denice_client_model
    from fed_learning.data.denice_clean_roles import CleanRoleData
    roles = CleanRoleData(a.roles, source_data_dir=a.data)
    metadata = json.loads((roles.source/'metadata.json').read_text(encoding='utf-8'))
    current_classes = metadata['task_structure']['task_classes'][str(ckpt['task'])]
    rows = []
    for cid, ledger in ledgers.items():
        raw = lookup(ckpt['client_algorithm_states'], cid)
        state = raw.get('denice', raw)
        entries = state.get('appliance_guarded_head_entries', {})
        # Predeclared: every patch first installed in this fixture task, not
        # selected using observed test performance or specific class IDs.
        for c, entry in entries.items():
            route = ProtectedRoute.from_packet(entry['packet'])
            if route.metadata['task'] != ckpt['task']:
                continue
            c = int(c)
            model, router = _make_denice_client_model(ckpt, cid, 'cpu', router_mode='multiclass_balanced')
            refs = entry['reference_classes']
            required = sorted(k for k in ckpt['seen_classes'] if k != c and
                              k not in current_classes and
                              int(model.unit_ranks['fc2'][k]) >= 2)
            if a.protection_only:
                required = sorted(set(required) | {k for r in ledger.receipts.values() for k in r['receipt']['class_rows']})
            missing = ledger.coverage(required)['missing_protection_classes']
            if missing:
                rows.append(dict(receiver=cid, class_id=c, skipped='missing potential peer support', missing=missing))
                continue
            frozen = ledger.freeze(required, c)
            receiver = roles.client_role(cid, 'validation')
            donor = roles.client_role(route.metadata['donor'], 'validation')
            pools = [('receiver_validation', receiver, set(ckpt['seen_classes'])),
                     ('donor_validation_positive', donor, {c})]
            # Receiver-local validation alone cannot expose non-owned classes.
            # Query every receipted peer's independently locked validation role
            # for its protected class rows. No class was picked using test errors.
            support = {}
            edge = lookup(graph['alpha_debug'], cid)
            for peer, weight in zip(edge['group_ids'], edge['alphas']):
                if peer == cid or weight <= 0:
                    continue
                raw_peer = lookup(ckpt['client_algorithm_states'], peer)
                ps = raw_peer.get('denice', raw_peer)['appliance_base_sketch_shield_state']
                support[int(peer)] = {int(k) for k, e in ps['memory']['entries'].items() if e['task'] < ckpt['task']}
            # Include old donor-owned classes the receiver has NOT learned too.
            # This diagnoses the actual coverage limit; they do not magically
            # become receipted knowledge or eligible protection evidence.
            for peer, classes in sorted(support.items()):
                pools.append((f'peer_{peer}_validation_negative', roles.client_role(peer, 'validation'), classes))
            for name, (x, y, row_ids), classes in pools:
                chosen = np.concatenate([np.flatnonzero(y == k)[:a.per_class] for k in sorted(classes)])
                x, y, row_ids = x[chosen], y[chosen], row_ids[chosen]
                if not len(y):
                    continue
                before = digest(model.state_dict())
                with ExitStack() as stack:
                    for k, e in entries.items():
                        stack.enter_context(original_local_head(model, int(k), e['backup']))
                    sig = stable_signals(model, router, x, ckpt['seen_classes'], route, refs, a.batch_size, 'cpu')
                if digest(model.state_dict()) != before:
                    raise AssertionError('Development inference mutated model')
                base = sig['local_pred']
                required_own = entry.get('required_old_classes', [])
                own = CurrentBaseSketchShield.restore(entry['shield_at_install'])
                old_hit = (sig['signature_valid'] & (sig['signature_score'] > route.metadata['tau']) &
                           (sig['margin'] > route.metadata['gamma']) & ~own.veto(x, required_own))
                hit = old_hit & ~frozen_veto(frozen, x)
                pred_old = np.where(old_hit, c, base)
                pred_new = np.where(hit, c, base)
                metric = lambda p: dict(activation=int((p == c).sum()),
                    target_rows=int((y == c).sum()), target_correct=int(((y == c) & (p == c)).sum()),
                    false_activation=int(((y != c) & (p == c)).sum()), accounting=transitions(y, base, p))
                rows.append(dict(receiver=cid, class_id=c, pool=name, rows=len(y),
                    class_counts={str(k): int((y == k).sum()) for k in sorted(set(map(int, y)))},
                    row_ids_sha256=hashlib.sha256(row_ids.astype('<i8').tobytes()).hexdigest(),
                    previous_guard_shadow_CPU=metric(pred_old), peer_veto_shadow_CPU=metric(pred_new),
                    vetoed_previous_activations=int((old_hit & ~hit).sum()),
                    certificate_revalidated=False, thresholds_changed=False))
                print(f'Provenance development receiver={cid}, class={c}, {name}: rows={len(y)}, old hit={old_hit.sum()}, new hit={hit.sum()}', flush=True)
                write_json(a.out/'development.json', rows)
            del model, router
    write_json(a.out/'development.json', rows)
    return rows


def run(a):
    a.out.mkdir(parents=True, exist_ok=False)
    write_json(a.out/'completion.json', dict(completed=False))
    ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    task = int(ckpt['task'])
    graphs = json.loads(a.graphs.read_text(encoding='utf-8'))
    graph = next(g for g in graphs if g['task'] == task and g['round'] == ckpt['final_round_id'])
    ledgers, rows, wire = simulate(ckpt, graph, a.out, a.protection_only)
    checks = controls(ledgers, a.out)
    write_json(a.out/'potential_coverage.json', rows)
    write_json(a.out/'simulated_communication.json', wire)
    write_json(a.out/'protocol.json', dict(task=task, checkpoint_sha256=hashlib.file_digest(a.checkpoint.open('rb'),'sha256').hexdigest(),
        graph_sha=digest(graph), historical_knowledge_receipts_present=False,
        simulated_receipt_rule='terminal mature receiver and donor row, donor owned BASE, positive recorded edge',
        no_claim_of_actual_historical_row_contribution=True,
        peer_summary_mode='protection_only: no acquired classifier knowledge claim' if a.protection_only else 'potential classifier-row contribution',
        development_scope='locked validation role, all newly installed task patches, fixed per-class cap',
        validation_per_class_cap=a.per_class, original_thresholds_frozen=True,
        final_test_opened=False, historical_raw_CAL_opened=False, backbone_training=False,
        backend='CPU FP32 shadow analysis; old CUDA certificate NOT revalidated',
        native_smoke_authorized_by_this_audit=False))
    if a.development:
        development(ckpt, ledgers, a, graph)
    summary = dict(completed=True, checks=len(checks['checks']), mismatches=checks['mismatches'],
        development_executed=a.development,
        task=task, receivers=len(rows), missing_owned_pairs=sum(len(r['missing_owned']) for r in rows),
        potential_remaining_missing_pairs=sum(len(r['potential_missing_after_peer_support']) for r in rows),
        initially_uncovered_receivers=sum(bool(r['missing_owned']) for r in rows),
        potential_fully_covered_receivers=sum(not r['potential_missing_after_peer_support'] for r in rows),
        simulated_application_bytes=wire['application_egress_bytes'], production_enabled=False)
    write_json(a.out/'completion.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--graphs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--development', action='store_true')
    p.add_argument('--protection-only', action='store_true', help='Separate negative-support receipt; no classifier row/competence claim')
    p.add_argument('--roles', type=Path)
    p.add_argument('--data', type=Path)
    p.add_argument('--per-class', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=512)
    a = p.parse_args()
    if a.development and (not a.roles or not a.data):
        p.error('Development requires locked roles and original data')
    torch.set_num_threads(4)
    with threadpool_limits(limits=1):
        run(a)
