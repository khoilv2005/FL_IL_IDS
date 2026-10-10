"""Fixed diagonal FIT envelope: a development falsification, never an install.

Mean/variance use donor current CAL-FIT only. Shrinkage=.01 and quantile=.99
are declared before validation. They must not be swept using peer validation.
Original guarded route remains a shadow CPU reference, not a revalidated cert.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from appliance.base_sketch_shield import CurrentBaseSketchShield
from appliance.current_calibration_data import CurrentCalibrationData
from appliance.portable_route import SharedSketch, ProtectedRoute
from appliance.receiver_aware_discovery import current_fit
from appliance.selector import lookup
from appliance.stable_head import stable_signals
from appliance.state import complete_hash, write_json, digest
from fed_learning.data.denice_clean_roles import CleanRoleData
from tools.audit_appliance_provenance_acceptance import original_model


def run(a):
    a.out.mkdir(parents=True, exist_ok=False)
    write_json(a.out/'protocol.json', dict(quantile=.99, diagonal_shrinkage=.01,
        retain_projection_amplitude=a.retain_amplitude,
        full_preprocessed_input=a.full_input,
        covariance='full fixed .01 isotropic shrinkage' if a.full_covariance else 'diagonal',
        variance_floor=float(np.finfo(np.float32).eps**2), validation_sweep=False,
        test_opened=False, production_enabled=False, historical_raw_CAL_opened=False))
    ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    raw = lookup(ckpt['client_algorithm_states'], a.receiver); state = raw.get('denice', raw)
    entries = state['appliance_guarded_head_entries']; c = a.class_id
    entry = entries.get(c, entries.get(str(c)))
    if entry is None or entry['task'] != ckpt['task']:
        raise ValueError('Requires an original fixture-task install')
    route = ProtectedRoute.from_packet(entry['packet']); donor = route.metadata['donor']
    view = CurrentCalibrationData(a.calibration_store, donor, ckpt['task'], ckpt['config']['denice_data_roles_sha256'])
    fit = current_fit(view); pos = fit['y'] == c
    if pos.sum() < 32:
        raise ValueError('Not enough current FIT positive rows')
    m = route.metadata['signature']
    sketch = SharedSketch(tuple(m['input_shape']), m['dimension'], m['preprocessing_sha256'], m['seed'])
    z, valid = sketch.features(fit['X'][pos])
    if not valid.all():
        raise ValueError('Invalid FIT projection')
    features = (lambda x: np.asarray(x, np.float32).reshape(len(x), -1) @ sketch.matrix()) if a.retain_amplitude else (lambda x: sketch.features(x)[0])
    if a.full_input:
        features = lambda x: np.asarray(x, np.float32).reshape(len(x), -1)
    if a.retain_amplitude or a.full_input:
        z = features(fit['X'][pos])
    mean = z.astype(np.float64).mean(0); var = z.astype(np.float64).var(0)
    variance = np.maximum(var + .01*var.mean(), np.finfo(np.float32).eps**2)
    score = lambda features: np.sum((features.astype(np.float64)-mean)**2/variance, axis=1)
    precision = None
    if a.full_covariance:
        covariance = np.cov(z.astype(np.float64), rowvar=False, ddof=0)
        covariance += np.eye(len(mean))*max(.01*var.mean(), float(np.finfo(np.float32).eps**2))
        precision = np.linalg.inv(covariance)
        score = lambda features: np.einsum('ni,ij,nj->n', features.astype(np.float64)-mean,
                                         precision, features.astype(np.float64)-mean)
    radius = float(np.quantile(score(z), .99, method='linear'))
    write_json(a.out/'envelope_lock_before_validation.json', dict(mean=mean, variance=variance, precision=precision, radius=radius,
        full_preprocessed_input=a.full_input, retain_projection_amplitude=a.retain_amplitude,
        quantile=.99, diagonal_shrinkage=.01, FIT_positive_rows=int(pos.sum()),
        donor=donor, class_id=c, role_sha=view.store['role_manifest_sha256'],
        row_ids_sha256=hashlib.sha256(fit['rows'][pos].astype('<i8').tobytes()).hexdigest(),
        shared_sketch=m, validation_opened=False))
    graphs = json.loads(a.graphs.read_text())
    graph = next(g for g in graphs if g['task'] == ckpt['task'] and g['round'] == ckpt['final_round_id'])
    edge = lookup(graph['alpha_debug'], a.receiver)
    pools = [('receiver_validation', a.receiver, set(ckpt['seen_classes'])),
             ('donor_validation_positive', donor, {c})]
    for peer, weight in zip(edge['group_ids'], edge['alphas']):
        if peer == a.receiver or weight <= 0:
            continue
        raw = lookup(ckpt['client_algorithm_states'], peer); ps = raw.get('denice', raw)
        classes = {int(k) for k, e in ps['appliance_base_sketch_shield_state']['memory']['entries'].items()
                   if e['task'] < ckpt['task']}
        if classes:
            pools.append((f'peer_{peer}_validation_negative', int(peer), classes))
    roles = CleanRoleData(a.roles, source_data_dir=a.data)
    model, router = original_model(ckpt, a.receiver); before = complete_hash(model, router)
    shield = CurrentBaseSketchShield.restore(entry['shield_at_install'])
    rows = []
    for name, owner, classes in pools:
        x, y, ids = roles.client_role(owner, 'validation')
        ix = np.concatenate([np.flatnonzero(y == k)[:a.per_class] for k in sorted(classes)])
        x, y, ids = x[ix], y[ix], ids[ix]
        if not len(x):
            continue
        sig = stable_signals(model, router, x, ckpt['seen_classes'], route, entry['reference_classes'], a.batch_size, 'cpu')
        old = sig['signature_valid'] & (sig['signature_score'] > route.metadata['tau']) & (sig['margin'] > route.metadata['gamma'])
        old &= ~shield.veto(x, entry['required_old_classes'])
        _, okay = sketch.features(x)
        projected = features(x)
        new = old & okay & (score(projected) < radius)
        rows.append(dict(pool=name, owner=owner, rows=len(y), target_rows=int((y == c).sum()),
            original_hit=int(old.sum()), envelope_hit=int(new.sum()),
            original_false_activation=int((old & (y != c)).sum()),
            envelope_false_activation=int((new & (y != c)).sum()),
            by_class={str(k):dict(rows=int((y == k).sum()), old=int((old & (y == k)).sum()),
                                   new=int((new & (y == k)).sum())) for k in sorted(set(y))},
            row_ids_sha256=hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest()))
        print(f'FIT envelope receiver={a.receiver} class={c} {name}: old={old.sum()} new={new.sum()}', flush=True)
    if complete_hash(model, router) != before:
        raise AssertionError('Envelope audit mutated model')
    write_json(a.out/'results.json', rows)
    write_json(a.out/'completion.json', dict(completed=True, receiver=a.receiver, class_id=c,
        validation_used_for_parameter_selection=False, installed=False, production_enabled=False,
        retain_projection_amplitude=a.retain_amplitude,
        full_preprocessed_input=a.full_input,
        full_covariance=a.full_covariance,
        final_test_opened=False, CAL_acceptance_evaluated=False, no_HOLDOUT_gate_claim=True,
        model_unchanged=True, envelope_sha=digest(dict(mean=mean, variance=variance, precision=precision, radius=radius))))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('checkpoint', 'graphs', 'calibration-store', 'roles', 'data', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--receiver', type=int, required=True); p.add_argument('--class-id', type=int, required=True)
    p.add_argument('--per-class', type=int, default=256); p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--retain-amplitude', action='store_true', help='Use raw fixed projection; normalization discards magnitude')
    p.add_argument('--full-input', action='store_true', help='Lossless preprocessed feature space; no 16D projection')
    p.add_argument('--full-covariance', action='store_true', help='Preserve feature correlations; fixed .01 shrinkage')
    a = p.parse_args(); torch.set_num_threads(4)
    if a.retain_amplitude and a.full_input:
        p.error('Select at most one feature representation')
    with threadpool_limits(limits=1):
        run(a)
