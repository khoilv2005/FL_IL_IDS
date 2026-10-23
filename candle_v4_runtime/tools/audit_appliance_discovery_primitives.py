"""Focused negative controls for peer protection transport and head prechecks."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from appliance.base_sketch_shield import CurrentBaseSketchShield
from appliance.config import Protocol, Rejected
from appliance.provenance_protection import ProvenanceProtection, export_support, transfer_support, frozen_veto
from appliance.receiver_aware_discovery import head_offer, maturity_precheck, rank_fit_probes
from appliance.state import digest, write_json
from appliance.transport import Transport
from eval_checkpoint import _make_denice_client_model


def run(a):
    a.out.mkdir(parents=True, exist_ok=False)
    checks = {}
    own = CurrentBaseSketchShield.restore(json.loads((a.shields/'receiver_0_shield.json').read_text()))
    donor = CurrentBaseSketchShield.restore(json.loads((a.shields/'receiver_4_shield.json').read_text()))
    ledger = ProvenanceProtection(own)
    packet = export_support(donor, 'sender-model-1', 'dependency-1')
    receipt = dict(sender=4, receiver=0, task=donor.memory.task, round=19, kind='aggregation', alpha=.5,
        class_rows=[min(map(int, donor.memory.entries))], sender_model_version='sender-model-1',
        receiver_model_version='receiver-model-1', dependency_version='dependency-1')
    receipt['receipt_id'] = digest(receipt)
    wire = Transport(a.out/'wire.jsonl', [(4, 0)],
        Protocol(max_incoming_bytes=16*1024*1024, max_outgoing_bytes=16*1024*1024).validate())
    transfer_support(ledger, packet, receipt, wire, current_task=receipt['task'], authorized_senders=[4])
    checks['compressed_wire_restores_exact_source'] = ledger.sources[packet['source']['state_digest']] == packet['source']
    restored = ProvenanceProtection.restore(ledger.state())
    packet2 = export_support(donor, 'sender-model-2', 'dependency-2')
    receipt2 = dict(receipt, round=20, sender_model_version='sender-model-2', dependency_version='dependency-2')
    receipt2['receipt_id'] = digest({k: v for k, v in receipt2.items() if k != 'receipt_id'})
    transfer_support(restored, packet2, receipt2, wire, current_task=receipt['task'], authorized_senders=[4])
    checks['source_cache_survives_resume_without_resend'] = sum(r['kind'] == 'peer_BASE_summary_zlib_v1' for r in wire.records) == 1
    checks['checkpoint_deduplicates_source_geometry'] = len(restored.state()['sources']) == 1 and len(restored.receipts) == 2
    negative = ProvenanceProtection(own)
    safety_receipt = dict(receipt, kind='protection_only')
    safety_receipt['receipt_id'] = digest({k: v for k, v in safety_receipt.items() if k != 'receipt_id'})
    negative.receive(packet, safety_receipt, current_task=receipt['task'], authorized_senders=[4])
    coverage = negative.coverage(receipt['class_rows'])
    checks['negative_support_not_acquired_classifier_knowledge'] = (
        coverage['inherited_knowledge_is_receiver_owned'] is False and
        coverage['peer_evidence_types'][str(receipt['class_rows'][0])] == ['protection_only'] and
        coverage['CAL_acceptance_substitution'] is False)
    fresh = ProvenanceProtection(own); before = fresh.state()
    tiny = Transport(a.out/'quota_wire.jsonl', [(4, 0)], Protocol(max_incoming_bytes=1, max_outgoing_bytes=1).validate())
    try:
        transfer_support(fresh, packet, receipt, tiny, current_task=receipt['task'], authorized_senders=[4])
    except Rejected:
        checks['failed_wire_rolls_back_entire_ledger'] = fresh.state() == before
    else:
        checks['failed_wire_rolls_back_entire_ledger'] = False
    required = receipt['class_rows']; imported = next(c for c in range(34) if c not in required)
    snapshot = restored.freeze(required, imported)
    x = np.random.default_rng(42).normal(size=(64, *own.memory.sketch.input_shape)).astype(np.float32)
    prior = frozen_veto(snapshot, x)
    corrupt_snapshot = copy.deepcopy(snapshot);corrupt_snapshot['implementation_fingerprint'] = 'changed'
    corrupt_snapshot['state_digest'] = digest({k: v for k, v in corrupt_snapshot.items() if k != 'state_digest'})
    try:
        frozen_veto(corrupt_snapshot, x)
    except Rejected:
        checks['veto_implementation_drift_rejected'] = True
    else:
        checks['veto_implementation_drift_rejected'] = False
    restored.receipts.clear()
    checks['frozen_snapshot_survives_live_ledger_change'] = np.array_equal(prior, frozen_veto(snapshot, x))
    ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    receiver, router = _make_denice_client_model(ckpt, 48, 'cpu')
    donor_model, _ = _make_denice_client_model(ckpt, 53, 'cpu')
    offer = head_offer(donor_model, 19, digest(donor_model.state_dict()))
    good = maturity_precheck(receiver, router, offer, 3, ckpt['seen_classes'])
    checks['known_eligible_head_precheck'] = good['eligible'] and good['before_capsule']
    from appliance.dependency_boundary import head_dependency_boundary
    from appliance.closure import effective_linear
    w, _ = effective_linear(receiver, 'fc2')
    boundary = head_dependency_boundary(receiver, torch.cat((torch.tensor(offer['weight'])[None, :], w[good['reference_classes']]), 0))
    changed = copy.deepcopy(receiver)
    changed.unit_ranks['fc1'][boundary['scope']['fc1'][0]] = 1
    bad = maturity_precheck(changed, router, offer, 3, ckpt['seen_classes'])
    checks['young_dependency_rejected_before_capsule'] = not bad['eligible'] and bad['before_capsule']
    corrupt = copy.deepcopy(offer); corrupt['weight'][0] += 1
    try:
        maturity_precheck(receiver, router, corrupt, 3, ckpt['seen_classes'])
    except Rejected:
        checks['head_metadata_corruption_rejected'] = True
    else:
        checks['head_metadata_corruption_rejected'] = False
    prototypes = [dict(donor=d, compatible_positive_lcb=lcb, receiver_FIT_far=far,
        authorizes_installation=False, holdout_predictions_opened=False, selection_predictions_opened=False)
        for d, lcb, far in ((1, .9, .2), (2, .85, 0))]
    checks['receiver_risk_affects_rank'] = rank_fit_probes(prototypes, {1: .99, 2: .8})[0]['donor'] == 2
    prototypes[0]['holdout_predictions_opened'] = True
    try:
        rank_fit_probes(prototypes, {1: .99, 2: .8})
    except Rejected:
        checks['HOLDOUT_ranking_forbidden'] = True
    else:
        checks['HOLDOUT_ranking_forbidden'] = False
    report = dict(completed=True, checks=checks, mismatches=sum(not v for v in checks.values()),
                  communication=wire.summary(), native_training_tested=False, production_enabled=False)
    write_json(a.out/'completion.json', report)
    print(json.dumps(report, indent=2))
    if report['mismatches']:
        raise AssertionError('Discovery primitive controls failed')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--shields', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    run(a)
