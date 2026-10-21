"""Offline two-tier discovery and frozen shadow-transfer diagnostics; never TEST."""
import argparse
from collections import Counter
import copy
import csv
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from appliance.runner import load_input
from appliance.selector import lookup, recorded_graph
from appliance.state import complete_hash
from fed_learning.data.denice_clean_roles import file_sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def save_rows(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def algorithm(ckpt, cid):
    state = lookup(ckpt['client_algorithm_states'], cid)
    return state.get('denice', state)


def coverage(a, ckpt, hashes, roles, store):
    groups, alphas = recorded_graph(ckpt, a.task, 19)
    ids = sorted(groups)
    tasks = {int(t): list(map(int, cs)) for t, cs in store['task_classes'].items()}
    task_of = {c: t for t, cs in tasks.items() for c in cs}
    starts = [e for e in read(a.inputs / 'training_artifacts/denice_debug_history.json')
              if e.get('type') == 'task_start' and int(e['task']) <= a.task]
    owned, counts = {i: set() for i in ids}, {i: {} for i in ids}
    for event in starts:
        for key, row in event['client_data'].items():
            cid = int(key)
            if cid in owned:
                labels = set(map(int, row['labels']))
                owned[cid].update(labels)
                counts[cid].update({int(c): int(n) for c, n in row['class_hist'].items()})
    seen = sorted(c for c, t in task_of.items() if t <= a.task)
    available, ages = {}, {}
    for i in ids:
        state = algorithm(ckpt, i)
        detector = state['context_detector']
        memory = {int(t): mat for t, mat in detector['activation_memory'].items()}
        available[i] = {int(c) for t, cs in detector['episode_classes'].items()
                        if int(t) in memory and len(memory[int(t)]) for c in cs}
        ages[i] = np.asarray(state['neuron_ages']['fc2'])
    global_present = set().union(*owned.values())
    rows, candidates = [], []
    evidence_precheck = Counter()
    for i in ids:
        for c in seen:
            donors = [j for j in ids if j != i and c in owned[j] and ages[j][c] >= 2]
            inside = [j for j in donors if j in groups[i]]
            outside = [j for j in donors if j not in groups[i]]
            missing_data = c in global_present and c not in owned[i]
            missing_route = c not in available[i]
            row = dict(receiver=i, class_id=c, class_task=task_of[c],
                current_class=task_of[c] == a.task, local_BASE_presence=c in owned[i],
                BASE_rows=counts[i].get(c, 0), available=c in available[i],
                fc2_rank=int(ages[i][c]), bitmap_gap=missing_data,
                availability_gap=missing_route, intra_donors=inside, extra_donors=outside,
                bitmap_tier=('intra' if inside else 'extra' if outside else 'none') if missing_data else 'not_requested',
                functional_gap='UNMEASURED')
            rows.append(row)
            if not (missing_data and missing_route and task_of[c] == a.task):
                continue
            own_cal = store['clients'][str(i)][str(a.task)]['class_counts']
            risk_rows = sum(int(n) for label, n in own_cal.items() if int(label) != c)
            preferred = inside or outside
            best_positive = max((int(store['clients'][str(j)][str(a.task)]['class_counts'].get(str(c), 0))
                                 for j in preferred), default=0)
            reason = ('positive_and_negative_below64' if best_positive < 64 and risk_rows < 64 else
                      'positive_below64' if best_positive < 64 else
                      'negative_below64' if risk_rows < 64 else 'metadata_probe_eligible')
            evidence_precheck[reason] += 1
            for tier, pool in [('intra', inside), ('extra', outside if not inside else [])]:
                for j in pool:
                    cal = store['clients'][str(j)][str(a.task)]['class_counts']
                    donor_count = int(cal.get(str(c), 0))
                    receiver_cal = store['clients'][str(i)][str(a.task)]['class_counts']
                    negatives = sum(int(n) for label, n in receiver_cal.items() if int(label) != c)
                    if donor_count >= 64 and negatives >= 64:
                        candidates.append(dict(task=a.task, receiver=i, donor=j, class_id=c, tier=tier,
                            intra_donor_count=len(inside), donor_CAL_count=donor_count,
                            receiver_negative_CAL_count=negatives, alpha=float(alphas[i].get(j, 0)),
                            receiver_rank=int(ages[i][c])))
    selected = []
    for tier in ('intra', 'extra'):
        pool = sorted((c for c in candidates if c['tier'] == tier),
                      key=lambda c: (-c['donor_CAL_count'], -c['receiver_negative_CAL_count'],
                                     c['receiver'], c['class_id'], c['donor']))
        if pool:
            selected.append(pool[0])
    missing = [r for r in rows if r['bitmap_gap']]
    missing_current = [r for r in missing if r['current_class']]
    unavailable = [r for r in rows if r['availability_gap']]
    result = dict(task=a.task, round=19, receivers=len(ids), receiver_class_pairs=len(rows),
        current_classes=tasks[a.task], classes_globally_owned=len(global_present),
        bitmap_gaps=len(missing), bitmap_tiers=dict(Counter(r['bitmap_tier'] for r in missing)),
        current_bitmap_gaps=len(missing_current),
        current_bitmap_tiers=dict(Counter(r['bitmap_tier'] for r in missing_current)),
        availability_gaps=len(unavailable),
        availability_gaps_with_intra=sum(bool(r['intra_donors']) for r in unavailable),
        availability_gaps_extra_only=sum(not r['intra_donors'] and bool(r['extra_donors']) for r in unavailable),
        current_probe_candidates=len(candidates), selected_cases=selected,
        current_gap_evidence_precheck=dict(evidence_precheck),
        graph=dict(mode=ckpt['config']['denice_clustering_mode'],
            xi=ckpt['config']['denice_similarity_threshold'], mean_peers=float(np.mean([len(v) for v in groups.values()])),
            not_a_disjoint_partition=True), checkpoint_hashes=hashes,
        no_functional_competence_inferred_from_metadata=True,
        limitation='Cumulative bitmap counts use local BASE presence history; historical prototypes are not assumed available.')
    save_rows(a.out / 'receiver_class_discovery.csv', rows)
    save_rows(a.out / 'current_probe_candidates.csv', candidates)
    write(a.out / 'coverage.json', result)
    write(a.out / 'probe_plan.json', dict(cases=selected, source_hashes=hashes,
        selection='metadata only: missing local BASE and route; highest donor current CAL support, then negative support/IDs',
        construction_evaluation_split='content SHA256 parity, seed20261022; cap128 per owner/class/partition',
        no_threshold_tuning=True, no_optimizer=True, task=a.task, historical_CAL_reads=False,
        scope='current CAL positives and negatives only; cumulative old-risk remains UNKNOWN',
        prior_exposure='Development source and these checkpoints have prior audits; not untouched confirmation'))
    print('COVERAGE', json.dumps(result), flush=True)
    return result


def digest(row):
    row = np.ascontiguousarray(row, dtype=np.float32)
    return hashlib.sha256(str(row.shape).encode() + row.tobytes()).hexdigest()


def partition_pool(a, store, role_sha, cid, classes):
    from appliance.current_calibration_data import CurrentCalibrationData
    view = CurrentCalibrationData(a.store, cid, a.task, role_sha)
    pool = view.current_pool(cid, 'calibration', classes)
    selected = {'construction': [], 'evaluation': []}
    used, counters = set(), Counter()
    for index, (x, y) in enumerate(zip(pool['X'], pool['y'])):
        h = digest(x)
        if h in used:
            continue
        used.add(h)
        side = 'construction' if int(hashlib.sha256(('20261022:' + h).encode()).hexdigest()[:8], 16) % 2 == 0 else 'evaluation'
        key = (side, int(y))
        if counters[key] < 128:
            counters[key] += 1
            selected[side].append(index)
    return {side: dict(x=pool['X'][ix], y=pool['y'][ix], hashes=[digest(x) for x in pool['X'][ix]])
            for side, ix in selected.items()}


@torch.no_grad()
def predict(model, detector, x, seen, device, task, oracle=False):
    from fed_learning.training.denice_eval import _denice_routed_logits_with_episodes
    predictions, episodes = [], []
    model.eval()
    for start in range(0, len(x), 128):
        inputs = torch.from_numpy(x[start:start + 128]).to(device)
        kwargs = dict(inference_policy='oracle_hard', oracle_episodes=np.full(len(inputs), task)) if oracle else dict(inference_policy='pred_hard')
        logits, routed = _denice_routed_logits_with_episodes(model, inputs, detector, seen, device, **kwargs)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite shadow logits')
        predictions.extend(logits.argmax(1).cpu().tolist())
        episodes.extend(routed.tolist())
    return np.asarray(predictions), np.asarray(episodes)


@torch.no_grad()
def features(model, router, x, task, device):
    active = copy.deepcopy(model.active_adapters)
    try:
        model.eval()
        model.set_active_context(task)
        h = model.penultimate_features(torch.from_numpy(x).to(device)).cpu().numpy()
        binary = router._binarize_per_sample(model, torch.from_numpy(x).to(device))
        return h, binary
    finally:
        model.active_adapters = active


def cosine(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.sum(a * b) / denominator) if denominator else None


def probe(a, ckpt, roles, store, plan):
    from eval_checkpoint import _make_denice_client_model
    classes = list(map(int, store['task_classes'][str(a.task)]))
    seen = sorted(map(int, ckpt['seen_classes']))
    role_sha = file_sha256(a.roles / 'role_manifest.json')
    results = []
    for position, case in enumerate(plan['cases']):
        i, j, c = case['receiver'], case['donor'], case['class_id']
        print(f'PROBE {position+1}/{len(plan["cases"])} receiver={i} donor={j} class={c} tier={case["tier"]}', flush=True)
        receiver_pool = partition_pool(a, store, role_sha, i, classes)
        donor_pool = partition_pool(a, store, role_sha, j, classes)
        construction = donor_pool['construction']
        fit_x = construction['x'][construction['y'] == c]
        # Prefer receiver examples in the panel; supplement target positives from donor.
        records = [(x, int(y), h, i) for x, y, h in zip(receiver_pool['evaluation']['x'], receiver_pool['evaluation']['y'], receiver_pool['evaluation']['hashes'])]
        records += [(x, int(y), h, j) for x, y, h in zip(donor_pool['evaluation']['x'], donor_pool['evaluation']['y'], donor_pool['evaluation']['hashes']) if y == c]
        unique = {}
        for x, y, h, owner in records:
            unique.setdefault(h, (x, y, h, owner))
        records = list(unique.values())
        x = np.stack([r[0] for r in records]); y = np.asarray([r[1] for r in records]); owners = np.asarray([r[3] for r in records])
        construction_hashes = set(construction['hashes']) | set(receiver_pool['construction']['hashes'])
        if construction_hashes & set(unique):
            raise ValueError('Construction/evaluation content overlap')
        if len(fit_x) < 16 or (y == c).sum() < 16 or (y != c).sum() < 16:
            results.append(dict(case=case, status='INSUFFICIENT_UNIQUE_CURRENT_EVIDENCE',
                                construction_target=len(fit_x), positives=int((y == c).sum()), negatives=int((y != c).sum())))
            continue
        receiver, router = _make_denice_client_model(ckpt, i, a.device)
        donor, donor_router = _make_denice_client_model(ckpt, j, a.device)
        receiver.eval(); donor.eval()
        before_hashes = (complete_hash(receiver, router), complete_hash(donor, donor_router))
        base_pred, _ = predict(receiver, router, x, seen, a.device, a.task)
        donor_native, donor_routes = predict(donor, donor_router, x, seen, a.device, a.task)
        donor_forced, _ = predict(donor, donor_router, x, seen, a.device, a.task, oracle=True)
        donor_reference = dict(target_rows=int((y == c).sum()),
            native_target_recall=float((donor_native[y == c] == c).mean()),
            forced_target_recall_DIAGNOSTIC_ONLY=float((donor_forced[y == c] == c).mean()),
            target_task_route_accuracy=float((donor_routes[y == c] == a.task).mean()),
            negative_FAR_on_receiver_panel=float((donor_native[y != c] == c).mean()))
        h_receiver, b_receiver = features(receiver, router, fit_x, a.task, a.device)
        h_donor, b_donor = features(donor, donor_router, fit_x, a.task, a.device)
        h_re, _ = features(receiver, router, x[y == c], a.task, a.device)
        p_r, p_d = h_receiver.mean(0), h_donor.mean(0)
        selected = np.linalg.norm(h_receiver - p_r, axis=1)
        geometry = dict(dimension=int(p_r.size), construction_target=len(fit_x),
            donor_vs_receiver_prototype_cosine=cosine(p_r, p_d),
            prototype_L2=float(np.linalg.norm(p_r-p_d)), receiver_within_class_mean_L2=float(selected.mean()),
            donor_prototype_to_receiver_eval_mean_L2=float(np.linalg.norm(h_re-p_d, axis=1).mean()),
            receiver_prototype_to_receiver_eval_mean_L2=float(np.linalg.norm(h_re-p_r, axis=1).mean()),
            context_dimension=int(b_receiver.shape[1]),
            binary_agreement_same_inputs=float((b_receiver == b_donor).mean()),
            binary_Jaccard_same_inputs=float(np.logical_and(b_receiver, b_donor).sum() /
                                             max(1, np.logical_or(b_receiver, b_donor).sum())),
            binary_receiver_density=float(b_receiver.mean()), binary_donor_density=float(b_donor.mean()),
            binarization_thresholds_equal=router.binarize_thresholds == donor_router.binarize_thresholds,
            stable_feature_masks_equal=np.array_equal(getattr(router, 'stable_feature_mask', None),
                                                     getattr(donor_router, 'stable_feature_mask', None)),
            FC1_prototype_FP32_bytes=int(p_d.nbytes), bitmap_packed_bytes=5,
            one_context_sketch_packed_bytes=int(np.ceil(b_receiver.shape[1]/8)),
            one_context_sketch_current_float32_bytes=int(b_receiver.shape[1]*4))
        geometry['donor_current_episode_memory_rows'] = int(len(donor_router.activation_memory[a.task]))
        geometry['donor_current_episode_memory_current_bytes'] = int(np.asarray(donor_router.activation_memory[a.task]).nbytes)
        metrics = []
        variants = ['baseline', 'availability_only', 'mask_only', 'head_preserve_mask',
                    'head_and_mask', 'head_mask_memory_pool', 'head_mask_memory_mean', 'head_mask_memory_or']
        for name in variants:
            model, det = copy.deepcopy(receiver), copy.deepcopy(router)
            if name != 'baseline':
                det.episode_classes[a.task] = sorted(set(det.episode_classes.get(a.task, [])) | {c})
            if name in ('mask_only', 'head_and_mask', 'head_mask_memory_pool', 'head_mask_memory_mean', 'head_mask_memory_or'):
                model.weight_masks['fc2'][c] = donor.weight_masks['fc2'][c].to(model.weight_masks['fc2']).clone()
                model.bias_masks['fc2'][c] = donor.bias_masks['fc2'][c].to(model.bias_masks['fc2']).clone()
            if name in ('head_preserve_mask', 'head_and_mask', 'head_mask_memory_pool', 'head_mask_memory_mean', 'head_mask_memory_or'):
                with torch.no_grad():
                    model.fc2.weight[c].copy_(donor.fc2.weight[c])
                    model.fc2.bias[c].copy_(donor.fc2.bias[c])
            if name.startswith('head_mask_memory'):
                matrices = [np.asarray(det.activation_memory[t]) for t in [a.task] if t in det.activation_memory and len(det.activation_memory[t])]
                matrices += [np.asarray(donor_router.activation_memory[a.task])]
                bank = np.concatenate(matrices)
                if name.endswith('pool'):
                    det.activation_memory[a.task] = bank.copy()
                else:
                    det.activation_memory[a.task] = ((bank.mean(0) >= .5) if name.endswith('mean') else bank.any(0)).astype(np.float32)[None, :]
            pred, route = predict(model, det, x, seen, a.device, a.task)
            forced, _ = predict(model, det, x, seen, a.device, a.task, oracle=True)
            target, negative = y == c, y != c
            receiver_target = target & (owners == i)
            rescue = int(((base_pred != y) & (pred == y)).sum())
            broken = int(((base_pred == y) & (pred != y)).sum())
            correct_delta = int((pred == y).sum() - (base_pred == y).sum())
            if correct_delta != rescue - broken:
                raise ValueError('Transition accounting mismatch')
            metrics.append(dict(variant=name, rows=len(y), positive_rows=int(target.sum()), negative_rows=int(negative.sum()),
                accuracy=float((pred == y).mean()), target_recall=float((pred[target] == c).mean()),
                receiver_target_rows=int(receiver_target.sum()),
                receiver_target_recall=float((pred[receiver_target] == c).mean()) if receiver_target.any() else None,
                target_precision=float((pred[target] == c).sum()/(pred == c).sum()) if (pred == c).any() else None,
                negative_FAR=float((pred[negative] == c).mean()),
                negative_break=int((negative & (base_pred == y) & (pred != y)).sum()),
                rescue=rescue, breaks=broken, correct_delta=correct_delta,
                task_route_accuracy=float((route == a.task).mean()),
                target_task_route_accuracy=float((route[target] == a.task).mean()),
                negative_classes={str(int(label)): dict(rows=int((y == label).sum()),
                    false_activation=int(((y == label) & (pred == c)).sum()),
                    breaks=int(((y == label) & (base_pred == y) & (pred != y)).sum()))
                    for label in np.unique(y[negative])},
                forced_current_task_recall_DIAGNOSTIC_ONLY=float((forced[target] == c).mean())))
            del model, det
        if before_hashes != (complete_hash(receiver, router), complete_hash(donor, donor_router)):
            raise ValueError('Reference model/router mutated by audit')
        result = dict(case=case, status='COMPLETED_DIAGNOSTIC_NOT_INSTALLATION', geometry=geometry,
            metadata=dict(receiver_support=sorted(roles['clients'][str(i)]['role_class_counts']['base']),
                receiver_rank=int(receiver.unit_ranks['fc2'][c]), donor_rank=int(donor.unit_ranks['fc2'][c])),
            variants=metrics, donor_reference=donor_reference, construction_evaluation_content_overlap=0,
            receiver_reference_unchanged=True, donor_reference_unchanged=True,
            old_class_risk='UNKNOWN: current-task CAL only',
            no_installation=True, no_optimizer=True, prototype_classifier_calibration_not_run=True)
        results.append(result)
        write(a.out / f'probe_{position}_{i}_from_{j}_class_{c}.json', result)
        print('PROBE RESULT', json.dumps(dict(case=case, geometry=geometry, variants=metrics)), flush=True)
        del receiver, router, donor, donor_router
        gc.collect()
        if a.device.startswith('cuda'):
            torch.cuda.empty_cache()
    write(a.out / 'probes.json', dict(completed=True, cases=results))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=ROOT/'audit_denice/appliance_legacy_results11/inputs')
    parser.add_argument('--roles', type=Path, default=ROOT/'audit_denice/appliance_legacy_results11/inputs/roles')
    parser.add_argument('--store', type=Path, default=ROOT/'audit_denice/appliance_current_runtime_data_local/full100_v1/calibration_store')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--task', type=int, default=5)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(20261022)
    started = time.monotonic()
    write(args.out / 'completion.json', dict(completed=False, stage='load_checkpoint'))
    source = args.inputs/f'checkpoint_task_{args.task}_all_rounds.zip'
    print(f'Loading original terminal checkpoint: {source}', flush=True)
    ckpt, hashes = load_input(source, args.task, 19)
    roles, store = read(args.roles/'role_manifest.json'), read(args.store/'calibration_store_manifest.json')
    role_sha = file_sha256(args.roles/'role_manifest.json')
    if (ckpt['config'].get('denice_cl_method') != 'legacy'
            or ckpt['config'].get('denice_router_mode') != 'binary_cosine'
            or ckpt['config'].get('denice_similarity_threshold') != .8
            or ckpt['config'].get('appliance_enabled', False)
            or ckpt['config'].get('denice_appliance_enabled', False)
            or ckpt['config']['denice_data_roles_sha256'] != role_sha
            or store['role_manifest_sha256'] != role_sha
            or read(args.store/'calibration_store_lock.json')['manifest_sha256'] != file_sha256(args.store/'calibration_store_manifest.json')):
        raise ValueError('Expected original legacy/xi=.8/binary checkpoint and matching CAL authority')
    result = coverage(args, ckpt, hashes, roles, store)
    plan = read(args.out/'probe_plan.json')
    plan_hash = file_sha256(args.out/'probe_plan.json')
    cases = probe(args, ckpt, roles, store, plan)
    if file_sha256(args.out/'probe_plan.json') != plan_hash:
        raise ValueError('Probe plan changed')
    write(args.out/'completion.json', dict(completed=True, seconds=time.monotonic()-started,
        task=args.task, coverage=result, probe_cases=len(cases),
        source_checkpoint=str(source), checkpoint_hashes=hashes, probe_plan_sha256=plan_hash,
        script_sha256=file_sha256(__file__), final_TEST_read=False, backbone_training=False,
        historical_CAL_read=False, model_installation=False, development_only=True))
    print(f'Completed: {args.out / "completion.json"}', flush=True)


if __name__ == '__main__':
    main()
