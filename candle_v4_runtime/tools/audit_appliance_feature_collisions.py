"""Exact preprocessed-input collisions across current FIT and peer validation.

No approximate matching, rounding or test read. A collision establishes only
ambiguity for these audited rows under input-only inference, not a global bound.
No raw vectors or per-row hashes are exported. This tool selects no guard.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from appliance.current_calibration_data import CurrentCalibrationData
from appliance.portable_route import ProtectedRoute
from appliance.receiver_aware_discovery import current_fit
from appliance.selector import lookup
from appliance.state import write_json
from fed_learning.data.denice_clean_roles import CleanRoleData


def hashes(x):
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite preprocessed input')
    # Canonicalize signed zero without rounding any nonzero feature.
    a = np.ascontiguousarray(np.where(x == 0, np.float32(0), x), dtype='<f4').reshape(len(x), -1)
    return Counter(hashlib.sha256(row.tobytes()).digest() for row in a)


def run(a):
    a.out.mkdir(parents=True, exist_ok=False)
    ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    raw = lookup(ckpt['client_algorithm_states'], a.receiver); state = raw.get('denice', raw)
    entries = state['appliance_guarded_head_entries']; c = a.class_id
    entry = entries.get(c, entries.get(str(c)))
    if entry is None or entry['task'] != ckpt['task']:
        raise ValueError('Requires original fixture task install')
    route = ProtectedRoute.from_packet(entry['packet']); donor = route.metadata['donor']
    view = CurrentCalibrationData(a.calibration_store, donor, ckpt['task'], ckpt['config']['denice_data_roles_sha256'])
    fit = current_fit(view); positive = hashes(fit['X'][fit['y'] == c])
    roles = CleanRoleData(a.roles, source_data_dir=a.data)
    x, y, ids = roles.client_role(donor, 'validation')
    valid_positive = hashes(x[np.flatnonzero(y == c)[:a.per_class]])
    graph = next(g for g in json.loads(a.graphs.read_text()) if g['task'] == ckpt['task'] and g['round'] == ckpt['final_round_id'])
    edge = lookup(graph['alpha_debug'], a.receiver)
    rows = []
    for peer, weight in zip(edge['group_ids'], edge['alphas']):
        if peer == a.receiver or weight <= 0:
            continue
        raw = lookup(ckpt['client_algorithm_states'], peer); ps = raw.get('denice', raw)
        classes = sorted(int(k) for k, e in ps['appliance_base_sketch_shield_state']['memory']['entries'].items()
                         if e['task'] < ckpt['task'])
        x, y, ids = roles.client_role(int(peer), 'validation')
        for k in classes:
            ix = np.flatnonzero(y == k)[:a.per_class]
            if not len(ix):
                continue
            negative = hashes(x[ix]); overlap = set(negative) & set(positive)
            val_overlap = set(negative) & set(valid_positive)
            rows.append(dict(peer=int(peer), negative_class=k, negative_rows=len(ix),
                negative_unique_inputs=len(negative), exact_shared_FIT_inputs=len(overlap),
                FIT_positive_rows_with_opposite_label=sum(positive[h] for h in overlap),
                negative_rows_identical_to_positive_FIT=sum(negative[h] for h in overlap),
                exact_shared_validation_inputs=len(val_overlap),
                positive_validation_rows_with_opposite_label=sum(valid_positive[h] for h in val_overlap),
                negative_rows_identical_to_positive_validation=sum(negative[h] for h in val_overlap),
                row_ids_sha256=hashlib.sha256(ids[ix].astype('<i8').tobytes()).hexdigest()))
    result = dict(completed=True, receiver=a.receiver, class_id=c, donor=donor,
        FIT_positive_rows=sum(positive.values()), FIT_unique_inputs=len(positive),
        validation_positive_rows=sum(valid_positive.values()), validation_unique_inputs=len(valid_positive),
        collisions=rows, final_test_opened=False, historical_raw_CAL_opened=False,
        approximate_matching=False, guard_selected=False, preprocessing_modified=False,
        limitation='counts describe only this development fixture, not the full test error bound')
    write_json(a.out/'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'collisions'}, indent=2))
    print(json.dumps([r for r in rows if r['exact_shared_FIT_inputs'] or r['exact_shared_validation_inputs']], indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('checkpoint', 'graphs', 'calibration-store', 'roles', 'data', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--receiver', type=int, required=True); p.add_argument('--class-id', type=int, required=True)
    p.add_argument('--per-class', type=int, default=256)
    run(p.parse_args())
