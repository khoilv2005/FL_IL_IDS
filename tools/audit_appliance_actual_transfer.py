"""Frozen development audit of the actual receiver head + imported guard.

Native donor predictions never screen a candidate. Original held-out FIT panels
are reused as development, not CAL or untouched confirmation. All selection
guards are locked before any actual-transfer EVALUATION forward. No installs.
"""
import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from appliance.base_sketch_shield import CurrentBaseSketchShield
from appliance.closure import effective_linear
from appliance.config import Rejected
from appliance.current_calibration_data import CurrentCalibrationData
from appliance.distributed_calibration import (CalibrationSession, GuardGrid,
    LocalCalibrationEndpoint, current_acceptance)
from appliance.imported_route import ROUTE_RULES, stratified_roles
from appliance.portable_route import (ProtectedRoute, SharedSketch, receiver_signals,
    prototype_summary, fit_support_cosine_floor)
from appliance.selector import lookup
from appliance.stable_head import stable_signals
from appliance.state import complete_hash, digest, write_json
from fed_learning.data.denice_clean_roles import CleanRoleData, file_sha256
from tools.audit_appliance_discriminative_guard import content_hashes
from tools.audit_appliance_donor_eligibility import MIN_ROWS, ROLES, summarize
from tools.audit_appliance_provenance_acceptance import original_model


def choose_guard(sig, keys, slices, receiver, donor, target, floor):
    """Same stable production quantile/tie rule, explicitly DEVELOPMENT only.

    Receiver/donor selection scopes follow the production two endpoints, but
    span their owned development classes here, not pretend-current CAL. Peer
    negative pools are audited separately after this frozen selection.
    """
    own = {o: np.concatenate([np.arange(slices[k].start, slices[k].stop)
            for k in keys if k[0] == o]) for o in (receiver, donor)}
    positive = np.arange(slices[donor, target].start, slices[donor, target].stop)
    if len(own[receiver]) < 32 or len(positive) < 8:
        return dict(status='insufficient_selection', tau=floor, gamma=0., beta=1.)
    q = np.linspace(0, 1, 31)
    def quantiles(name):
        return np.concatenate([np.quantile(sig[name][own[o]], q) for o in (receiver, donor)])
    taus = np.unique(np.clip(np.r_[floor, 1., quantiles('signature_score')], -1, 1))
    taus = taus[taus >= floor]
    gammas = np.unique(np.maximum(0, np.r_[0., quantiles('margin')]))
    receiver_keys = [k for k in keys if k[0] == receiver]
    donor_keys = [k for k in keys if k[0] == donor and k[1] != target]
    endpoint_negatives = receiver_keys + donor_keys
    def count_grid(indices):
        indices = np.asarray(indices, np.int64)
        indices = indices[sig['signature_valid'][indices]]
        # index = number of thresholds strictly below this value. Reverse
        # cumulative histograms therefore preserve the production strict >.
        t = np.searchsorted(taus, sig['signature_score'][indices], side='left')
        g = np.searchsorted(gammas, sig['margin'][indices], side='left')
        hist = np.zeros((len(taus)+1, len(gammas)+1), np.int64)
        np.add.at(hist, (t, g), 1)
        return hist[::-1, ::-1].cumsum(0).cumsum(1)[::-1, ::-1][1:, 1:]
    feasible = np.ones((len(taus), len(gammas)), bool)
    for k in receiver_keys:
        sl = slices[k]; n = sl.stop - sl.start
        if n:
            feasible &= count_grid(np.arange(sl.start, sl.stop))/n <= .001
    negatives = np.concatenate([np.arange(slices[k].start, slices[k].stop) for k in endpoint_negatives])
    broken = np.concatenate([np.arange(slices[k].start, slices[k].stop)[
        sig['local_pred'][slices[k]] == k[1]] for k in endpoint_negatives])
    feasible &= count_grid(broken) == 0
    positives, false_activations = count_grid(positive), count_grid(negatives)
    best = None
    for ti, gi in zip(*np.nonzero(feasible)):
        tau, gamma = float(taus[ti]), float(gammas[gi])
        tp, fp = int(positives[ti, gi]), int(false_activations[ti, gi])
        key = (tp, -fp, -1., gamma, tau)
        if best is None or key > best[0]:
            best = key, dict(status='selected', tau=tau, gamma=gamma, beta=1.,
                positive_rows=len(positive), positive_hits=tp, endpoint_false_activation=fp,
                endpoint_breaks=0, tau_count=len(taus), gamma_count=len(gammas))
    if best is None:
        # tau=1 is always an inactive option; this is an integrity error.
        raise AssertionError('No inactive selection guard')
    return best[1]


def evaluate(sig, keys, slices, donor, target, tau, gamma):
    hit = sig['signature_valid'] & (sig['signature_score'] > tau) & (sig['margin'] > gamma)
    positive = hit[slices[donor, target]]
    negative, missing = [], []
    for o, c in keys:
        if c == target:
            continue
        sl = slices[o, c]
        h = hit[sl]
        negative.append((o, c, h, h & (sig['local_pred'][sl] == c)))
        if len(h) < MIN_ROWS:
            missing.append(dict(owner=o, class_id=c, rows=len(h)))
    result = summarize(positive, negative, missing)
    result['rescue'] = int((positive & (sig['local_pred'][slices[donor, target]] != target)).sum())
    result['forced_head_hits'] = int((sig['margin'][slices[donor, target]] > 0).sum())
    result['no_evidence_is_not_pass'] = True
    return result


def run(a):
    a.out.mkdir(parents=True, exist_ok=False)
    write_json(a.out/'completion.json', dict(completed=False))
    ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    previous = json.loads((a.prepared/'protocol_before_data.json').read_text())
    graph = next(g for g in json.loads(a.graphs.read_text(encoding='utf-8'))
                 if g['task'] == ckpt['task'] and g['round'] == ckpt['final_round_id'])
    roles = CleanRoleData(a.roles, source_data_dir=a.data)
    role_sha = file_sha256(a.roles/'role_manifest.json')
    authority = dict(checkpoint_sha256=file_sha256(a.checkpoint), graph_sha256=digest(graph),
                     role_manifest_sha256=role_sha)
    if (ckpt['task'] != 5 or ckpt['config'].get('denice_cl_method') != 'legacy' or
            any(previous[k] != v for k, v in authority.items()) or
            ckpt['config']['denice_data_roles_sha256'] != role_sha):
        raise ValueError('Prepared audit/checkpoint/graph/roles authority changed')
    states = {int(i): s.get('denice', s) for i, s in ckpt['client_algorithm_states'].items()}
    # Shared imported features can be reused across contexts only without adapters.
    if any(s.get('adapter_registry') for s in states.values()):
        raise ValueError('This cache requires adapter-free legacy models')
    shields = {i: CurrentBaseSketchShield.restore(s['appliance_base_sketch_shield_state'])
               for i, s in states.items()}
    owned = {i: set(map(int, shields[i].memory.entries)) for i in states}
    groups = {}
    for i in states:
        row = lookup(graph['alpha_debug'], i)
        groups[i] = {i} | {int(j) for j, w in zip(row['group_ids'], row['alphas']) if w > 0}
        if not groups[i].issubset(states):
            raise ValueError('Unknown live peer')
        shield = shields[i]
        counts = roles.manifest['clients'][str(i)]['role_class_counts']['base']
        if (shield.owner != i or shield.role_sha != role_sha or
                shield.pp_sha != roles.manifest['metadata_sha256'] or
                any(e['count'] != counts.get(c, 0) for c, e in shield.memory.entries.items())):
            raise ValueError('Owned support provenance changed')
    candidates = json.loads((a.prepared/'graph_candidates_before_scores.json').read_text())
    eligible = [v for v in candidates if v['status'] == 'precheck_pass']
    protocol = dict(authority, kind='Actual receiver-transfer eligibility development audit',
        metadata_candidates=len(candidates), maturity_candidates=len(eligible), task=5, round=19,
        native_donor_screen=False, native_donor_accuracy_used_for_selection=False,
        data_role='locked original FIT panels reused from native audit; development, not CAL',
        untouched_confirmation=False, signature='shared sketch 16D, donor signature_FIT p95 envelope',
        fixed_control='FIT floor + gamma=0 + own BASE veto',
        calibrated_guard='production stable guard quantile/tie rule on receiver/donor DEVELOPMENT selection',
        grid_quantiles=31, confidence_gate=False, recall_gate=.95, far_gate=.001, max_break=0,
        minimum_positive_and_each_owner_class_rows=32,
        scope='owned receiver and live positive-alpha peer classes; unobserved cumulative classes uncertified',
        selection_lock_before_actual_evaluation=True, evaluation_threshold_tuning=False,
        CAL='only Task5 current CAL holdout for frozen development-pass candidates; no threshold retune',
        historical_raw_CAL_reads=0, final_test_reads=0, production_installations=0,
        production_runner_changed=False, certificates_issued=0, backbone_training=False,
        communication_measured=False, CPU_shadow_not_CUDA_certificate_revalidation=True,
        prepared_metadata_sha256=file_sha256(a.prepared/'graph_candidates_before_scores.json'),
        prepared_panels_sha256=file_sha256(a.prepared/'panels_before_predictions.json'),
        script_sha256=file_sha256(Path(__file__)))
    write_json(a.out/'protocol_before_data.json', protocol)
    write_json(a.out/'graph_candidates_before_scores.json', candidates)
    manifest = json.loads((a.prepared/'panels_before_predictions.json').read_text())
    panels = {r: {} for r in ROLES}
    content = {r: set() for r in ROLES}
    for i in sorted(states):
        x, y, rows = roles.client_role(i, 'fit')
        row_index = {int(v): ix for ix, v in enumerate(rows)}
        for p in manifest['panels']:
            if p['owner'] != i:
                continue
            ix = np.asarray([row_index[v] for v in p['row_ids']], np.int64)
            part = x[ix]
            if (len(part) != p['rows'] or not np.all(y[ix] == p['class_id']) or
                    content_hashes(part) != p['content_hashes']):
                raise ValueError('Prepared FIT panel content changed')
            panels[p['role']][i, p['class_id']] = part
            content[p['role']].update(p['content_hashes'])
        print(f'Actual transfer verified FIT owner={i}', flush=True)
    if any(content[r] & content[s] for n, r in enumerate(ROLES) for s in ROLES[n+1:]):
        raise ValueError('Content-role overlap')
    write_json(a.out/'panels_before_predictions.json', manifest)
    sketch = SharedSketch(tuple(ckpt['config']['input_shape']), 16, roles.manifest['metadata_sha256'])
    models, fingerprints, routes = {}, {}, {}
    def model(i):
        if i not in models:
            models[i] = original_model(ckpt, i)
            fingerprints[i] = complete_hash(*models[i])
        return models[i]
    for v in eligible:
        i, j, c = (v[k] for k in ('receiver', 'donor', 'class_id'))
        if (j not in groups[i]-{i} or c in owned[i] or c not in owned[j] or
                int(states[i]['neuron_ages']['fc2'][c]) != 0 or
                int(states[j]['neuron_ages']['fc2'][c]) < 2):
            raise ValueError('Prepared candidate eligibility changed')
        fit = panels['signature_fit'][j, c]
        if len(fit) < 32:
            v['compile'] = dict(status='insufficient_signature_FIT', rows=len(fit))
            continue
        try:
            z, valid = sketch.features(fit)
            proto, var, support = prototype_summary(z, valid)
            floor = fit_support_cosine_floor(support)
        except Rejected as exc:
            v['compile'] = dict(status='insufficient_signature_FIT', reason=exc.reason)
            continue
        w, b = effective_linear(model(j)[0], 'fc2')
        routes[i, j, c] = ProtectedRoute(dict(class_id=c, task=v['task'], signature=sketch.manifest(),
            tau=floor, gamma=0., beta=1.), proto, var, w[c].numpy().copy(), float(b[c]))
        v['compile'] = dict(status='compiled', fit_rows=len(fit), floor=floor,
            signature_sha=digest(dict(prototype=proto, variance=var, support=support)),
            effective_head_sha=digest(dict(weight=w[c], bias=b[c])))
    print(f'Compiled {len(routes)}/{len(eligible)} actual guards; no native filter', flush=True)
    parity = []
    def cache(i, r, records):
        m, router = model(i)
        keys = sorted({(o, c) for o in groups[i] for c in owned[o]})
        parts = [panels[r][k] for k in keys]
        x = np.concatenate(parts)
        slices, cursor = {}, 0
        for k, part in zip(keys, parts):
            slices[k] = slice(cursor, cursor + len(part)); cursor += len(part)
        first = records[0]
        base = receiver_signals(m, router, x, ckpt['seen_classes'], first['task'],
            first['class_id'], a.batch_size, 'cpu')
        w, b = effective_linear(m, 'fc2')
        base['local_unmasked_logits'] = torch.nn.functional.linear(
            torch.from_numpy(base['imported_features']), w, b).numpy()
        veto = shields[i].veto(x, sorted(owned[i]))
        def signals(v):
            route = routes[i, v['donor'], v['class_id']]
            refs = v['precheck']['reference_classes']
            if any(int(m.unit_ranks['fc2'][c]) < 2 for c in refs):
                raise ValueError('Prepared margin references changed')
            view = dict(base, local_best_import_context=base['local_unmasked_logits'][:, refs].max(1))
            result = route.signals(view, x)
            result['signature_valid'] &= ~veto
            return result
        # Numerical equivalence check uses SELECTION only; no EVALUATION tuning.
        if r == 'selection':
            probe = first; n = min(16, len(x)); route = routes[i, probe['donor'], probe['class_id']]
            reference = stable_signals(m, router, x[:n], ckpt['seen_classes'], route,
                probe['precheck']['reference_classes'], a.batch_size, 'cpu')
            fast = signals(probe)
            matches = {k: bool(np.array_equal(reference[k], fast[k][:n]) if k == 'local_pred'
                else np.allclose(reference[k], fast[k][:n], rtol=1e-4, atol=2e-5))
                for k in ('local_pred', 'signature_score', 'margin', 'patch_logit')}
            # Veto is separately applied in the stable registry, not stable_signals.
            reference_hit = (reference['signature_valid'] &
                (reference['signature_score'] > route.metadata['tau']) & (reference['margin'] > 0) & ~veto[:n])
            fast_hit = (fast['signature_valid'][:n] &
                (fast['signature_score'][:n] > route.metadata['tau']) & (fast['margin'][:n] > 0))
            matches['activation'] = bool(np.array_equal(reference_hit, fast_hit))
            parity.append(dict(receiver=i, rows=n, matches=matches,
                max_margin_abs_error=float(np.max(np.abs(reference['margin'] - fast['margin'][:n])))))
            if not all(matches.values()):
                raise AssertionError(f'Cached stable-function mismatch {i}: {matches}')
        return keys, slices, signals
    for r in ('selection', 'evaluation'):
        for i in sorted({v['receiver'] for v in eligible if v['compile']['status'] == 'compiled'}):
            records = [v for v in eligible if v['receiver'] == i and v['compile']['status'] == 'compiled']
            keys, slices, signals = cache(i, r, records)
            for v in records:
                sig = signals(v); j, c = v['donor'], v['class_id']
                floor = v['compile']['floor']
                if r == 'selection':
                    v['guard_lock'] = choose_guard(sig, keys, slices, i, j, c, floor)
                guard = v['guard_lock']
                v[r] = dict(fixed_control=evaluate(sig, keys, slices, j, c, floor, 0.),
                    calibrated=evaluate(sig, keys, slices, j, c, guard['tau'], guard['gamma']))
                if r == 'selection':
                    v['selected_before_evaluation'] = (guard['status'] == 'selected' and
                        v[r]['calibrated']['status'] == 'passed_observed_scope')
                else:
                    v['functional_eligible'] = (v['selected_before_evaluation'] and
                        v[r]['calibrated']['status'] == 'passed_observed_scope')
                v['unobserved_cumulative_classes'] = sorted(set(ckpt['seen_classes']) - {c} -
                    {k for _, k in keys if k != c})
            write_json(a.out/'graph_results.json', candidates)
            print(f'Actual {r.upper()} receiver={i}: {len(records)} guards, '
                f'pass={sum(v[r]["calibrated"]["status"] == "passed_observed_scope" for v in records)}', flush=True)
        if r == 'selection':
            write_json(a.out/'cache_parity.json', parity)
            write_json(a.out/'selection_lock_before_evaluation.json', dict(
                records_sha256=digest(candidates), protocol_sha256=digest(protocol),
                EVALUATION_opened=False, locks=[dict(receiver=v['receiver'], donor=v['donor'],
                class_id=v['class_id'], guard=v['guard_lock'], selected=v['selected_before_evaluation'])
                for v in eligible if 'guard_lock' in v], no_evaluation_fallback=True))
    # Only a fixed guard that passed development can reach current CAL.
    cal_results, access = [], []
    store = json.loads((a.cal_store/'calibration_store_manifest.json').read_text())
    current_classes = store['task_classes']['5']
    for v in eligible:
        if not v.get('functional_eligible'):
            v['current_CAL'] = dict(status='not_opened_development_not_passed', authorized=False)
            continue
        i, j, c = (v[k] for k in ('receiver', 'donor', 'class_id'))
        if c not in current_classes:
            v['current_CAL'] = dict(status='unknown_target_outside_current_task', authorized=False)
            continue
        route = routes[i, j, c]; guard = v['guard_lock']
        route.metadata.update(tau=guard['tau'], gamma=guard['gamma'])
        session = CalibrationSession(i, j, c, 5, 19, tuple(current_classes), fingerprints[i],
            digest(guard), role_sha, 'actual-transfer-development-audit')
        grid = GuardGrid(session, [guard['tau']], [guard['gamma']], [1.])
        write_json(a.out/f'CAL_lock_{i}_{j}_{c}.json', dict(guard=guard, thresholds_frozen=True,
            current_task=5, prototype_source='original FIT; not a new production CAL-FIT compile', holdout_opened=False))
        endpoints = {}
        try:
            for owner in (i, j):
                view = CurrentCalibrationData(a.cal_store, owner, 5, role_sha)
                pool = view.current_pool(owner, 'calibration', current_classes)
                split = stratified_roles(pool, ROUTE_RULES['seed'] + owner)
                ix = split['holdout']; x = pool['X'][ix]
                if len(x):
                    sig = stable_signals(*model(i), x, ckpt['seen_classes'], route,
                        v['precheck']['reference_classes'], a.batch_size, 'cpu')
                    required = sorted(owned[i])
                    sig['signature_valid'] &= ~shields[i].veto(x, required)
                    sig['local_confidence'] = np.zeros(len(x))
                else:
                    sig = dict(signature_valid=np.zeros(0, bool), signature_score=np.zeros(0),
                        margin=np.zeros(0), local_pred=np.zeros(0, np.int64), local_confidence=np.zeros(0))
                empty = dict(y=np.zeros(0, np.int64), row_id=np.zeros(0, np.int64),
                    signals={k: value[:0] for k, value in sig.items()})
                views = dict(fit=empty, selection=empty, holdout=dict(y=pool['y'][ix],
                    row_id=pool['rows'][ix], signals=sig))
                endpoints[owner] = LocalCalibrationEndpoint(owner, session, views)
                endpoints[owner].lock_guard(grid)
                access.extend(view.access_log)
            accepted = current_acceptance(session, grid, endpoints[i].count_packet(grid, 'holdout'),
                                          endpoints[j].count_packet(grid, 'holdout'))
            v['current_CAL'] = dict(status='passed_current_scope' if accepted['passed_current_scope'] else 'failed',
                result=accepted, authorized=False, production_compile_not_reproduced=True)
        except Rejected as exc:
            v['current_CAL'] = dict(status='unknown' if 'INSUFFICIENT' in exc.reason else 'failed',
                reason=exc.reason, authorized=False)
        cal_results.append(dict(receiver=i, donor=j, class_id=c, **v['current_CAL']))
    write_json(a.out/'current_CAL_results.json', dict(results=cal_results, access_log=access))
    # Reuse exact frozen installed-patch shadow results, with sealed source and
    # unchanged checkpoint/panels. This does not refresh any old certificate.
    installed = json.loads((a.prepared/'installed_patch_results.json').read_text())
    write_json(a.out/'installed_patch_shadow.json', dict(source_sha256=file_sha256(
        a.prepared/'installed_patch_results.json'), source_checkpoint_sha256=authority['checkpoint_sha256'],
        results=installed, reused_exact_frozen_shadow=True, new_certificates=0, state_changes=0))
    unchanged = all(complete_hash(*models[i]) == fingerprints[i] for i in models)
    if not unchanged:
        raise AssertionError('Model/router mutated')
    write_json(a.out/'graph_results.json', candidates)
    counts = lambda role, variant: dict(Counter(v[role][variant]['status'] for v in eligible if role in v))
    summary = dict(completed=True, **authority, metadata_candidates=len(candidates),
        maturity_candidates=len(eligible), compiled_actual_guards=len(routes),
        insufficient_signature_FIT=len(eligible)-len(routes),
        actual_function_evaluated_pairs=sum('evaluation' in v for v in eligible),
        native_screen_used=False, fixed_selection=counts('selection', 'fixed_control'),
        fixed_evaluation=counts('evaluation', 'fixed_control'),
        calibrated_selection=counts('selection', 'calibrated'),
        calibrated_evaluation=counts('evaluation', 'calibrated'),
        functional_eligible_pairs=sum(v.get('functional_eligible', False) for v in eligible),
        current_CAL_status=dict(Counter(v['current_CAL']['status'] for v in eligible)),
        all_models_unchanged=unchanged, cache_parity_receivers=len(parity),
        all_parity_checks_passed=all(all(p['matches'].values()) for p in parity),
        raw_historical_CAL_reads=0, final_test_reads=0, installations=0, certificates_issued=0,
        eligible_is_observed_development_not_population_safety=True,
        limitation='Reused FIT development panels; no test claim, no install permission. Small or absent owner-class pools remain UNKNOWN. CAL target restricted to current Task5. Existing CUDA certificates are not revalidated by CPU shadows.')
    write_json(a.out/'completion.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'graphs', 'roles', 'data', 'prepared', 'cal-store', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=512)
    a = p.parse_args(); torch.set_num_threads(4)
    with threadpool_limits(limits=1):
        run(a)
