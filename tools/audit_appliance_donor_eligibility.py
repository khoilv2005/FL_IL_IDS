"""Development-only donor screening and frozen active-patch audit.

Reads the locked original FIT role, never historical CAL or final test. A
native donor screen is conservative and is NOT a transfer-feasibility bound.
Passing it does not authorize installation: transfer and current CAL remain
separate requirements. No model/guard/certificate is changed by this audit.
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
from appliance.patch_lifecycle import route_authorized
from appliance.portable_route import ProtectedRoute, SharedSketch, prototype_summary, fit_support_cosine_floor
from appliance.receiver_aware_discovery import head_offer, maturity_precheck
from appliance.selector import lookup
from appliance.stable_head import stable_signals
from appliance.state import complete_hash, digest, write_json
from fed_learning.data.denice_clean_roles import CleanRoleData, file_sha256
from fed_learning.training.denice_eval import _denice_routed_logits_with_episodes
from tools.audit_appliance_discriminative_guard import content_hashes
from tools.audit_appliance_provenance_acceptance import original_model

SEED = 20261011
MIN_ROWS, CAP = 32, 64
ROLES = ('signature_fit', 'selection', 'evaluation')


def summarize(positive, negative, missing):
    """A missing/undersized protected pool can never be a safe pass."""
    pos = np.asarray(positive, bool)
    per = [dict(owner=o, class_id=c, rows=len(hit), false_activation=int(np.sum(hit)),
                far=float(np.mean(hit)), breaks=int(np.sum(broken)))
           for o, c, hit, broken in negative if len(hit)]
    n = sum(v['rows'] for v in per)
    fp = sum(v['false_activation'] for v in per)
    breaks = sum(v['breaks'] for v in per)
    recall = float(pos.mean()) if len(pos) else None
    bad = (recall is not None and len(pos) >= MIN_ROWS and recall < .95) or any(
        v['far'] > .001 or v['breaks'] for v in per)
    adequate = len(pos) >= MIN_ROWS and bool(per) and not missing
    status = 'failed' if bad else ('passed_observed_scope' if adequate else 'insufficient_evidence')
    return dict(status=status, positive_rows=len(pos), target_hits=int(pos.sum()), recall=recall,
                negative_rows=n, false_activation=fp, far=fp/n if n else None,
                max_owner_class_far=max((v['far'] for v in per), default=None), breaks=breaks,
                missing_or_under32_pools=missing, per_owner_class=per,
                population_FAR_certified=False, installation_authorized=False)


@torch.no_grad()
def native(model, router, x, seen, batch_size):
    flags = [(m, m.training) for m in model.modules()]
    adapters = copy.deepcopy(model.active_adapters)
    try:
        model.eval()
        chunks = []
        for start in range(0, len(x), batch_size):
            logits, _ = _denice_routed_logits_with_episodes(
                model, torch.from_numpy(x[start:start+batch_size]), router, seen, 'cpu',
                inference_policy='pred_hard')
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Nonfinite donor logits')
            chunks.append(logits.argmax(1).numpy())
        return np.concatenate(chunks) if chunks else np.zeros(0, np.int64)
    finally:
        model.active_adapters = adapters
        for m, flag in flags:
            m.training = flag


def run(a):
    a.out.mkdir(parents=True, exist_ok=False)
    write_json(a.out/'completion.json', dict(completed=False))
    ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if ckpt['task'] != 5 or ckpt['config'].get('denice_cl_method') != 'legacy':
        raise ValueError('Legacy APPLIANCE terminal Task5 required')
    graph = next(g for g in json.loads(a.graphs.read_text(encoding='utf-8'))
                 if g['task'] == 5 and g['round'] == ckpt['final_round_id'])
    roles = CleanRoleData(a.roles, source_data_dir=a.data)
    if file_sha256(a.roles/'role_manifest.json') != ckpt['config']['denice_data_roles_sha256']:
        raise ValueError('Role authority changed')
    states = {int(i): s.get('denice', s) for i, s in ckpt['client_algorithm_states'].items()}
    ids = sorted(states)
    shields = {i: CurrentBaseSketchShield.restore(s['appliance_base_sketch_shield_state'])
               for i, s in states.items()}
    owned = {i: set(map(int, shields[i].memory.entries)) for i in ids}
    for i, shield in shields.items():
        if (shield.owner!=i or shield.role_sha!=ckpt['config']['denice_data_roles_sha256'] or
                shield.pp_sha!=roles.manifest['metadata_sha256']):
            raise ValueError('Owned BASE provenance authority changed')
        counts=roles.manifest['clients'][str(i)]['role_class_counts']['base']
        if any(e['count']!=counts.get(c,0) for c,e in shield.memory.entries.items()):
            raise ValueError('Owned BASE support count changed')
    groups = {}
    for i in ids:
        row = lookup(graph['alpha_debug'], i)
        groups[i] = {i} | {int(j) for j, w in zip(row['group_ids'], row['alphas']) if w > 0}
        if not groups[i].issubset(ids):
            raise ValueError('Unknown graph owner')
    protocol = dict(kind='Conservative donor eligibility + frozen active-patch development audit',
        checkpoint_sha256=file_sha256(a.checkpoint), graph_sha256=digest(graph),
        role_manifest_sha256=ckpt['config']['denice_data_roles_sha256'], task=5,
        seed=SEED, cap_per_owner_class_split=CAP, minimum_rows=MIN_ROWS,
        data_role='original backbone-held-out FIT; retrospective development only, not CAL',
        split='canonical FP32 content hash, deterministic three-way split across owners',
        protected_scope='all owned BASE classes of receiver and positive-alpha live neighbors',
        eligibility_screen='native donor target recall >=95%; per-owner/class target false activation <=0.1%',
        transfer_probe='unchanged 16D shared signature, FIT p95 floor, margin>0, own BASE veto',
        active_patch_probe='installed packet/thresholds/references/shield, independent shadow execution',
        native_router='patch-free self view, checkpoint detector, pred_hard; no retrospective multiclass refit',
        thresholds_selected_on_evaluation=False, raw_historical_CAL_opened=False,
        final_test_opened=False, backbone_training=False, production_modified=False,
        outside_graph_candidates=False, current_CAL_acceptance_substituted=False,
        native_screen_is_not_transfer_upper_bound=True, communication_measured=False,
        old_CUDA_certificate_revalidated_on_CPU=False)
    write_json(a.out/'protocol_before_data.json', protocol)
    missing_requests={(i,c) for i in ids for c in ckpt['seen_classes'] if c not in owned[i]}
    reachable=set();edge_count=occupied=immature=0
    for i in ids:
        for j in groups[i]-{i}:
            for c in owned[j]-owned[i]:
                edge_count+=1;reachable.add((i,c))
                if int(states[i]['neuron_ages']['fc2'][c])!=0:occupied+=1
                elif int(states[j]['neuron_ages']['fc2'][c])<2:immature+=1
    write_json(a.out/'coverage_metadata.json',dict(missing_owned_requests=len(missing_requests),
        requests_with_owned_live_donor=len(reachable),requests_without_owned_live_donor=len(missing_requests-reachable),
        owned_donor_edges=edge_count,receiver_output_not_free=occupied,donor_head_not_mature=immature,
        metadata_eligible=edge_count-occupied-immature))
    models = {}
    versions = {}
    def model(i):
        if i not in models:
            models[i] = original_model(ckpt, i)
            versions[i] = complete_hash(*models[i])
        return models[i]
    candidates = []
    metadata_ids = ids
    if a.prepared_run:
        prior = json.loads((a.prepared_run/'protocol_before_data.json').read_text())
        if any(prior[k] != protocol[k] for k in ('checkpoint_sha256','graph_sha256','role_manifest_sha256','seed')):
            raise ValueError('Prepared metadata authority changed')
        candidates = json.loads((a.prepared_run/'graph_candidates_before_scores.json').read_text())
        metadata_ids = []
    for i in metadata_ids:
        m, router = model(i)
        for j in sorted(groups[i]-{i}):
            for c in sorted(owned[j]-owned[i]):
                if int(states[i]['neuron_ages']['fc2'][c]) != 0 or int(states[j]['neuron_ages']['fc2'][c]) < 2:
                    continue
                dm, _ = model(j)
                task = int(shields[j].memory.entries[str(c)]['task'])
                record = dict(receiver=i, donor=j, class_id=c, task=task)
                try:
                    pre = maturity_precheck(m, router, head_offer(dm, c, 'frozen-checkpoint'),
                                            task, ckpt['seen_classes'])
                    record.update(precheck=pre, status='precheck_pass' if pre['eligible'] else 'precheck_rejected')
                except Rejected as e:
                    record.update(status='precheck_rejected', reason=e.reason)
                candidates.append(record)
        print(f'Eligibility metadata receiver={i}: cumulative candidates={len(candidates)}', flush=True)
    write_json(a.out/'graph_candidates_before_scores.json', candidates)
    # Prior feature audit used CAL/BASE/VAL, never the original 8% FIT role.
    # Explicit exclusion also protects against cross-owner content reuse.
    old = json.loads(a.prior_panels.read_text(encoding='utf-8'))
    excluded = {h for p in old['panels'] for h in p['content_hashes']}
    panels = {r: {} for r in ROLES}
    manifest, old_hits = [], 0
    for i in ids:
        x, y, rows = roles.client_role(i, 'fit')
        for c in sorted(owned[i]):
            indices = np.flatnonzero(y == c)
            hashes = content_hashes(x[indices])
            buckets = {r: [] for r in ROLES}
            for ix, h in zip(indices, hashes):
                if h in excluded:
                    old_hits += 1
                    continue
                key = hashlib.sha256(f'{SEED}:{h}'.encode()).hexdigest()
                buckets[ROLES[int(key, 16) % 3]].append((key, int(ix), h))
            for r, values in buckets.items():
                # One content per owner/class: identical contents still share
                # one role across owners and never leak between the splits.
                unique = {}
                for key, ix, h in sorted(values):
                    unique.setdefault(h, (key, ix, h))
                selected = list(unique.values())[:CAP]
                ix = np.array([v[1] for v in selected], np.int64)
                panels[r][i, c] = x[ix]
                manifest.append(dict(owner=i, class_id=c, role=r, rows=len(ix),
                    available_unique=len(unique), row_ids=rows[ix], content_hashes=[v[2] for v in selected]))
        print(f'Eligibility FIT panels owner={i}', flush=True)
    content = {r: {h for p in manifest if p['role'] == r for h in p['content_hashes']} for r in ROLES}
    if any(content[r] & content[s] for n, r in enumerate(ROLES) for s in ROLES[n+1:]):
        raise AssertionError('Content split leakage')
    write_json(a.out/'panels_before_predictions.json', dict(panels=manifest, prior_content_excluded=old_hits,
        content_disjoint=True, heldout_population_independence_not_claimed=True))
    cache = {}
    def predictions(j, r, pairs):
        todo = sorted(set(pairs)-{(o, c) for d, role, o, c in cache if d == j and role == r})
        nonempty = [(key, panels[r][key]) for key in todo if len(panels[r][key])]
        if nonempty:
            x = np.concatenate([v for _, v in nonempty])
            value = native(*model(j), x, ckpt['seen_classes'], a.batch_size)
            cursor = 0
            for key, part in nonempty:
                cache[j, r, *key] = value[cursor:cursor+len(part)]
                cursor += len(part)
        for key in todo:
            cache.setdefault((j, r, *key), np.zeros(0, np.int64))
        return {key: cache[j, r, *key] for key in pairs}
    def negative_keys(i, c, extra=()):
        return sorted({(o, k) for o in groups[i] | set(extra) for k in owned[o] if k != c})
    def donor_screen(i, j, c, r, force_negatives=False):
        positive = predictions(j, r, [(j, c)])[j, c] == c
        if not force_negatives and (len(positive) < MIN_ROWS or float(positive.mean()) < .95):
            result = summarize(positive, [], [])
            result.update(negative_screen_opened=False,
                          early_stop='insufficient_positive' if len(positive)<MIN_ROWS else 'positive_recall_failed')
            return result
        keys = negative_keys(i, c, (j,))
        pred = predictions(j, r, keys+[(j, c)])
        missing = [dict(owner=o, class_id=k, rows=len(panels[r][o, k]))
                   for o, k in keys if len(panels[r][o, k]) < MIN_ROWS]
        negative = [(o, k, pred[o, k] == c, np.zeros(len(pred[o, k]), bool)) for o, k in keys]
        result = summarize(pred[j, c] == c, negative, missing)
        result['negative_screen_opened'] = True
        return result
    # Selection screens are complete and locked before any EVALUATION score.
    selected = []
    for j in sorted({v['donor'] for v in candidates if v['status'] == 'precheck_pass'}):
        records = [v for v in candidates if v['donor'] == j and v['status'] == 'precheck_pass']
        # Batch all scoped owner pools for this donor, avoiding repeated forwards.
        keys = {(j, v['class_id']) for v in records}
        predictions(j, 'selection', sorted(keys))
        for v in records:
            v['selection'] = donor_screen(v['receiver'], j, v['class_id'], 'selection')
            if v['selection']['status'] == 'passed_observed_scope':
                selected.append(v)
        print(f'Donor SELECTION j={j}, candidates={len(records)}, passed={sum(v["selection"]["status"]=="passed_observed_scope" for v in records)}', flush=True)
        write_json(a.out/'graph_results.json', candidates)
    write_json(a.out/'selection_lock_before_evaluation.json', dict(selected=[
        {k: v[k] for k in ('receiver','donor','class_id')} for v in selected],
        all_selection_sha256=digest(candidates), policy_sha256=digest(protocol),
        EVALUATION_opened=False, no_evaluation_fallback=True))
    # Audit all maturity-eligible donors on EVALUATION, even if selection
    # failed; these diagnostics never become additional selected candidates.
    for j in sorted({v['donor'] for v in candidates if v['status'] == 'precheck_pass'}):
        records = [v for v in candidates if v['donor'] == j and v['status'] == 'precheck_pass']
        keys = {(j, v['class_id']) for v in records}
        predictions(j, 'evaluation', sorted(keys))
        for v in records:
            v['evaluation'] = donor_screen(v['receiver'], j, v['class_id'], 'evaluation')
            v['donor_eligible'] = v in selected and v['evaluation']['status'] == 'passed_observed_scope'
        print(f'Donor EVALUATION j={j}, eligible={sum(v["donor_eligible"] for v in records)}', flush=True)
        write_json(a.out/'graph_results.json', candidates)
    # Frozen installed packets are evaluated separately from donor eligibility.
    active_results = []
    for i in ids:
        for key, e in states[i].get('appliance_guarded_head_entries', {}).items():
            c, j = int(key), int(e['donor'])
            scope = e.get('lifecycle_runtime_scope')
            authorized = route_authorized(e, scope)
            # Record every installed patch; seven are active, one suspended.
            route = ProtectedRoute.from_packet(e['packet'])
            shield = CurrentBaseSketchShield.restore(e['shield_at_install'])
            keys = negative_keys(i, c, (j,))
            record = dict(receiver=i, donor=j, class_id=c, installed_task=e['task'],
                lifecycle_state=e['lifecycle_state'], authorized_saved_scope=authorized,
                donor_still_in_live_graph=j in groups[i], packet_sha256=hashlib.sha256(e['packet']).hexdigest(),
                tau=route.metadata['tau'], gamma=route.metadata['gamma'], CPU_shadow_not_certificate_revalidation=True)
            for r in ('selection', 'evaluation'):
                pos_x = panels[r].get((j, c), np.zeros((0, *ckpt['config']['input_shape']), np.float32))
                pool_keys = [(j, c)] + keys
                parts = [panels[r][k] for k in pool_keys]
                lengths = [len(x) for x in parts]
                x = np.concatenate([x for x in parts if len(x)])
                sig = stable_signals(*model(i), x, ckpt['seen_classes'], route,
                                     e['reference_classes'], a.batch_size, 'cpu')
                hit = (sig['signature_valid'] & (sig['signature_score'] > route.metadata['tau']) &
                       (sig['margin'] > route.metadata['gamma']) & ~shield.veto(x, e['required_old_classes']))
                cursor = lengths[0]
                negative = []
                rescue = int((hit[:cursor] & (sig['local_pred'][:cursor] != c)).sum())
                for (o, k), n in zip(keys, lengths[1:]):
                    h = hit[cursor:cursor+n]
                    negative.append((o, k, h, h & (sig['local_pred'][cursor:cursor+n] == k)))
                    cursor += n
                missing = [dict(owner=o, class_id=k, rows=len(panels[r][o, k]))
                           for o, k in keys if len(panels[r][o, k]) < MIN_ROWS]
                result = summarize(hit[:len(pos_x)], negative, missing)
                result['rescue'] = rescue
                result['donor_native'] = donor_screen(i, j, c, r, force_negatives=True)
                record[r] = result
            active_results.append(record)
            write_json(a.out/'installed_patch_results.json', active_results)
            print(f'Frozen patch {i} <- {j}/class{c}: {record["evaluation"]["status"]}', flush=True)
    # Empirical transfer probes only for candidates passing both donor screens.
    # No new threshold search, classifier, install, or CAL acceptance.
    for v in [v for v in candidates if v.get('donor_eligible')]:
        i, j, c = (v[k] for k in ('receiver','donor','class_id'))
        m, router = model(i); dm, _ = model(j)
        fit = panels['signature_fit'][j, c]
        if len(fit) < MIN_ROWS:
            v['transfer'] = dict(status='insufficient_signature_FIT', installation_authorized=False)
            continue
        sketch = SharedSketch(tuple(ckpt['config']['input_shape']), 16, roles.manifest['metadata_sha256'])
        z, valid = sketch.features(fit); proto, var, support = prototype_summary(z, valid)
        w, b = effective_linear(dm, 'fc2')
        route = ProtectedRoute(dict(class_id=c, task=v['task'], signature=sketch.manifest(),
            tau=fit_support_cosine_floor(support), gamma=0., beta=1.), proto, var, w[c].numpy(), float(b[c]))
        required = sorted(owned[i]-{c})
        results = {}
        for r in ('selection','evaluation'):
            keys = negative_keys(i, c)
            parts = [panels[r][j, c]] + [panels[r][k] for k in keys]
            lengths = [len(x) for x in parts]; x = np.concatenate([x for x in parts if len(x)])
            sig = stable_signals(m, router, x, ckpt['seen_classes'], route, v['precheck']['reference_classes'], a.batch_size, 'cpu')
            hit = sig['signature_valid'] & (sig['signature_score']>route.metadata['tau']) & (sig['margin']>0) & ~shields[i].veto(x, required)
            cursor = lengths[0]; negative=[]
            for (o,k), n in zip(keys,lengths[1:]):
                h=hit[cursor:cursor+n]
                negative.append((o,k,h,h & (sig['local_pred'][cursor:cursor+n]==k)));cursor+=n
            missing=[dict(owner=o,class_id=k,rows=len(panels[r][o,k])) for o,k in keys if len(panels[r][o,k])<MIN_ROWS]
            results[r]=summarize(hit[:lengths[0]],negative,missing)
        v['transfer']=dict(results, status='passed_observed_scope' if all(t['status']=='passed_observed_scope' for t in results.values()) else 'not_passed', installation_authorized=False)
        print(f'Transfer diagnostic {i} <- {j}/class{c}: {v["transfer"]["status"]}',flush=True)
        write_json(a.out/'graph_results.json', candidates)
    unchanged = all(complete_hash(*models[i]) == versions[i] for i in models)
    if not unchanged:
        raise AssertionError('Model or router mutated')
    active = [v for v in active_results if v['authorized_saved_scope']]
    counts = Counter(v['status'] for v in candidates)
    summary = dict(completed=True, checkpoint_sha256=protocol['checkpoint_sha256'],
        clients=len(ids), graph_metadata_candidates=len(candidates), maturity_funnel=dict(counts),
        donor_selection_status=dict(Counter(v['selection']['status'] for v in candidates if 'selection' in v)),
        donor_evaluation_status=dict(Counter(v['evaluation']['status'] for v in candidates if 'evaluation' in v)),
        donor_eligible_pairs=sum(v.get('donor_eligible',False) for v in candidates),
        transfer_evaluated_pairs=sum('transfer' in v for v in candidates),
        empirical_transfer_pass_pairs=sum(v.get('transfer',{}).get('status')=='passed_observed_scope' for v in candidates),
        installed_patches=len(active_results), active_saved_scope=len(active),
        active_patch_evaluation_status=dict(Counter(v['evaluation']['status'] for v in active)),
        all_models_unchanged=unchanged, content_splits_disjoint=True,
        raw_historical_CAL_reads=0, final_test_reads=0, production_installations=0,
        new_CAL_certificates=0, thresholds_retuned=False, backbone_training=False,
        limitation='Conservative native donor screen and bounded development panels; zero observed passes is not universal transfer infeasibility. Missing/under32 pools remain unknown. CPU shadows do not revalidate CUDA certificates. Current CAL acceptance required before installation.')
    write_json(a.out/'completion.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','graphs','roles','data','prior-panels','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--prepared-run',type=Path,help='Reuse metadata prechecks only, with exact checkpoint/graph/role/seed checks')
    a=p.parse_args();torch.set_num_threads(4)
    with threadpool_limits(limits=1):run(a)
