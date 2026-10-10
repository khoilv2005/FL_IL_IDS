"""Current FIT-only diagnosis of counter-evidence and old frozen thresholds.

Aggregates remain descriptive: no threshold/donor is selected with these data.
This does not certify an old CUDA guard on the CPU backend or open old raw CAL.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from appliance.base_sketch_shield import CurrentBaseSketchShield
from appliance.contrastive_protection import ContrastiveProtection
from appliance.closure import effective_linear
from appliance.current_calibration_data import CurrentCalibrationData
from appliance.portable_route import ProtectedRoute, receiver_signals
from appliance.receiver_aware_discovery import current_fit
from appliance.selector import lookup
from appliance.state import complete_hash, write_json
from tools.audit_appliance_provenance_acceptance import original_model


def run(a):
    a.out.mkdir(parents=True, exist_ok=False)
    counter = ContrastiveProtection.restore(json.loads(a.counter.read_text()))
    ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    cid, c = counter.ledger.owner, counter.imported_class
    raw = lookup(ckpt['client_algorithm_states'], cid); state = raw.get('denice', raw)
    entries = state['appliance_guarded_head_entries']
    entry = entries.get(c, entries.get(str(c)))
    if entry is None or entry['task'] != ckpt['task']:
        raise ValueError('Diagnostic requires an original fixture-task install')
    route = ProtectedRoute.from_packet(entry['packet'])
    model, router = original_model(ckpt, cid)
    before = complete_hash(model, router)
    view = CurrentCalibrationData(a.calibration_store, route.metadata['donor'], ckpt['task'],
                                  ckpt['config']['denice_data_roles_sha256'])
    pool = current_fit(view); x = pool['X'][pool['y'] == c]
    if not len(x):
        raise ValueError('No donor current FIT positives')
    counter.function(model, route, entry['reference_classes'])
    with torch.no_grad():
        base = receiver_signals(model, router, x, ckpt['seen_classes'], route.metadata['task'], c, a.batch_size, 'cpu')
    w, b = effective_linear(model, 'fc2')
    base['local_best_import_context'] = torch.nn.functional.linear(
        torch.as_tensor(base['imported_features']), w, b).numpy()[:, entry['reference_classes']].max(1)
    sig = route.signals(base, x); h = base['imported_features']; patch = sig['patch_logit']
    old_shield = CurrentBaseSketchShield.restore(entry['shield_at_install'])
    hit = sig['signature_valid'] & (sig['signature_score'] > route.metadata['tau']) & (sig['margin'] > route.metadata['gamma'])
    original_hit = hit & ~old_shield.veto(x, entry['required_old_classes'])
    own = counter.ledger.own.veto(x, counter.local)
    counters = []
    for k in sorted(counter.heads):
        p = counter.heads[k]; offer = p['offer']
        delta = h @ np.asarray(offer['weight'], np.float32) + np.float32(offer['bias']) - patch
        geometry = np.zeros(len(x), bool)
        for source in counter.sources[k]:
            geometry |= source.veto(x, [k])
        blocked = geometry & (delta >= 0)
        counters.append(dict(negative_class=k, source=p['sender'], rows=len(x),
            box_hits=int(geometry.sum()), negative_head_wins=int((delta >= 0).sum()),
            counter_veto=int(blocked.sum()), veto_of_original_hits=int((blocked & original_hit).sum()),
            negative_minus_patch_p05_p50_p95=np.quantile(delta, [.05, .5, .95]).tolist()))
    result = dict(receiver=cid, class_id=c, donor=route.metadata['donor'], FIT_positive_rows=len(x),
        original_guard_hit_shadow_CPU=int(original_hit.sum()),
        expanded_own_veto=int(own.sum()),
        expanded_own_by_class={str(k): int(counter.ledger.own.veto(x, [k]).sum()) for k in counter.local},
        original_required_old_classes=entry['required_old_classes'],
        contrastive_hit_with_original_thresholds=int((hit & ~counter.veto(x, h, patch)).sum()),
        original_hit_lost_to_counter=int((original_hit & counter.veto(x, h, patch)).sum()),
        counter_heads=counters, selection_or_holdout_opened=False, final_test_opened=False,
        no_threshold_selection=True, no_install=True, model_unchanged=before == complete_hash(model, router))
    if not result['model_unchanged']:
        raise AssertionError('FIT diagnosis mutated receiver')
    write_json(a.out/'result.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('checkpoint', 'counter', 'calibration-store', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=512)
    a = p.parse_args(); torch.set_num_threads(4)
    with threadpool_limits(limits=1):
        run(a)
