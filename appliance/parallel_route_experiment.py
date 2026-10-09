"""Calibrate a parallel imported route, lock its packet, then diagnose validation.

Offline pairwise simulator, not a deployed transfer or survival experiment.
No final test is read. Validation is an already-used development panel.
"""
import copy
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch

from .closure import effective_linear
from .config import Protocol, Rejected
from .evaluate import Predictor
from .imported_route import (ImportedRoute, ROUTE_RULES, build_signature,
                            encoder_cache_inventory, select_thresholds, stratified_roles)
from .routing_audit import histogram, prior_read, quantiles
from .runner import load_input
from .selector import current_pool, recorded_graph, select_pair
from .state import boundary_hash, changed_state, state_fingerprint, write_json
from .transfer_audit import probe


def metrics(y, pred, baseline, activated, c):
    """Join labels after decisions; one final prediction per row."""
    positive = y == c
    old = ~positive
    correct = pred == y
    base_correct = baseline == y
    rescued = ~base_correct & correct
    broken = base_correct & ~correct
    proposed = pred == c
    return dict(rows=len(y), positive_rows=int(positive.sum()),
        accuracy=float(correct.mean()) if len(y) else None,
        recall=float(correct[positive].mean()) if positive.any() else None,
        precision=float(positive[proposed].mean()) if proposed.any() else None,
        rescue=int(rescued.sum()), break_count=int(broken.sum()),
        break_by_class=histogram(y[broken]), rescue_by_class=histogram(y[rescued]),
        break_class25=int((broken & (y == 25)).sum()),
        activated_rows=int(activated.sum()), activation_rate=float(activated.mean()) if len(y) else None,
        positive_activation_rate=float(activated[positive].mean()) if positive.any() else None,
        false_activation_rows=int((activated & old).sum()),
        false_activation_rate=float(activated[old].mean()) if old.any() else None,
        false_activation_class25=int((activated & (y == 25)).sum()),
        old_accuracy=float(correct[old].mean()) if old.any() else None)


def save_rows(path, columns):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))


def run_imported_route(checkpoint, role_dir, data_dir, out, prior=None, task=4, round_id=19,
                       receiver_id=19, donor_id=71, class_id=24, device='cpu', batch_size=512,
                       allow_cgofed_fixture=False):
    from eval_checkpoint import _make_denice_client_model
    from fed_learning.data.denice_clean_roles import CleanRoleData, file_sha256
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'completion.json', dict(completed_execution=False, passed_feasibility=False))
    try:
        if task not in range(6) or round_id < 0 or batch_size < 1:
            raise ValueError('Invalid experiment scope')
        protocol = Protocol().validate()
        ckpt, hashes = load_input(checkpoint, task, round_id)
        config = ckpt['config']
        method = config.get('denice_cl_method', 'legacy')
        if method != 'legacy' and not (method == 'cgofed' and allow_cgofed_fixture):
            raise Rejected('LEGACY_BASELINE_REQUIRED', method)
        roles = CleanRoleData(role_dir, source_data_dir=data_dir)
        role_hash = file_sha256(roles.root / 'role_manifest.json')
        if config.get('denice_data_roles_sha256') != role_hash:
            raise Rejected('BACKBONE_ROLE_LOCK_MISMATCH')
        if tuple(config['input_shape']) != tuple(roles.manifest['input_shape']):
            raise Rejected('PREPROCESSING_SHAPE_MISMATCH')
        if config.get('denice_max_train_samples_per_client') or config.get('denice_max_clients') not in (None, 100):
            raise Rejected('TRUNCATED_BASE_TRAINING_NOT_SUPPORTED')
        metadata = json.loads((roles.source / 'metadata.json').read_text(encoding='utf-8'))
        task_classes = {int(k): list(map(int, v)) for k, v in metadata['task_structure']['task_classes'].items()}
        classes = task_classes[task]
        seen = sorted({c for t, values in task_classes.items() if t <= task for c in values})
        if sorted(map(int, ckpt['seen_classes'])) != seen or class_id not in classes:
            raise Rejected('CLASS_OR_SEEN_SCOPE_MISMATCH')
        groups, alphas = recorded_graph(ckpt, task, round_id)
        predict = Predictor(seen, device, batch_size)
        if prior is not None:
            previous = json.loads(prior_read(prior, 'protocol_lock.json'))
            for key, value in {**hashes, 'task':task, 'round':round_id, 'role_manifest_sha256':role_hash}.items():
                if previous.get(key) != value:
                    raise Rejected('PRIOR_PROTOCOL_MISMATCH', key)
        root = Path(__file__).resolve().parents[1]
        files = list((root / 'appliance').glob('*.py')) + [root / name for name in (
            'eval_checkpoint.py', 'fed_learning/models/nice_model.py', 'fed_learning/models/denice_model.py',
            'fed_learning/servers/nice_server.py', 'fed_learning/training/denice_eval.py',
            'fed_learning/training/checkpoint_state.py', 'fed_learning/data/denice_clean_roles.py')]
        lock = dict(**ROUTE_RULES, **hashes, task=task, round=round_id,
            receiver=receiver_id, donor=donor_id, class_id=class_id, base_method=method,
            diagnostic_fixture=method != 'legacy', role_manifest_sha256=role_hash,
            xi=config.get('denice_similarity_threshold'), device=device, batch_size=batch_size,
            versions=dict(torch=torch.__version__, numpy=np.__version__),
            source_sha256={p.relative_to(root).as_posix():file_sha256(p) for p in files},
            validation_is_development=True,
            signature_origin='donor calibration FIT positives, encoded at donor by cached receiver encoder',
            selection_origin='scalar calibration-SELECTION feedback from both clients',
            donor_quality_caveat='pair was previously selected; offer recheck uses full current calibration, including internal holdout',
            transfer_protocol='offline simulation; actual peer transport/cache not implemented',
            historical_retention='current task rows only; old task retention and next-round survival unmeasured')
        write_json(out / 'protocol_lock.json', lock)

        def make_model(cid):
            model, router = _make_denice_client_model(ckpt, cid, device)
            state = ckpt['client_model_states'].get(cid, ckpt['client_model_states'].get(str(cid)))
            restored = model.state_dict()
            if set(restored) != set(state) or any(not torch.equal(v.detach().cpu(), state[n]) for n,v in restored.items()):
                raise Rejected('INEXACT_MODEL_RESTORE', str(cid))
            if any(int(ep) > task for ep in router.activation_memory):
                raise Rejected('FUTURE_ROUTER_MEMORY')
            return model, router

        selected, ledger, offers = select_pair(ckpt, roles, task, classes, groups, make_model,
            predict, protocol, (receiver_id, donor_id, class_id), eligibility_path=out / 'pair_eligibility.json')
        write_json(out / 'pair_selection.json', dict(selected=selected, offers=offers, ledger=ledger.to_dict()))
        receiver, router = make_model(receiver_id)
        donor, donor_router = make_model(donor_id)
        if any(getattr(m, 'continual_head', None) is not None or getattr(m, 'local_classifier', None) is not None
               for m in (receiver, donor)):
            raise Rejected('UNSUPPORTED_AUXILIARY_CLASSIFIER')
        original = {name:state_fingerprint(m,r) for name,m,r in (
            ('receiver',receiver,router), ('donor',donor,donor_router))}
        pools = {cid:current_pool(roles, cid, 'calibration', classes) for cid in (receiver_id,donor_id)}
        split = {cid:stratified_roles(pools[cid], ROUTE_RULES['seed'] + cid) for cid in pools}
        manifest = {cid:{role:dict(rows=pools[cid]['rows'][index], class_counts=histogram(pools[cid]['y'][index]),
            missing_current_classes=sorted(set(classes) - set(map(int, pools[cid]['y'][index]))))
            for role,index in roles_for_client.items()} for cid,roles_for_client in split.items()}
        write_json(out / 'calibration_split_manifest.json', manifest)
        fit = split[donor_id]['fit']
        positives = fit[pools[donor_id]['y'][fit] == class_id]
        print(f'Parallel route: build signature at donor, positive FIT rows={len(positives)}', flush=True)
        cached_receiver = copy.deepcopy(receiver)
        prototype, support = build_signature(cached_receiver, pools[donor_id]['X'][positives], batch_size, device)
        del cached_receiver
        weights, biases = effective_linear(donor, 'fc2')
        packet_metadata = dict(kind='parallel_imported_route_experiment', version=ROUTE_RULES['version'],
            class_id=class_id, task=task, receiver=receiver_id, donor=donor_id, tau=1., gamma=0.,
            receiver_feature_hash=boundary_hash(receiver, True), checkpoint_terminal_sha256=hashes,
            role_manifest_sha256=role_hash, calibration_split_sha256=file_sha256(out / 'calibration_split_manifest.json'),
            feature_mode='adapter-free penultimate; receiver coordinates', prototype_support=support,
            head_feature_mode='receiver penultimate in fixed imported task context',
            head_mask='donor effective masked FC2 row', no_legacy_task_prerequisite=True,
            prototype_fit_row_hash=hashlib.sha256(pools[donor_id]['rows'][positives].tobytes()).hexdigest())
        route = ImportedRoute(packet_metadata, prototype, weights[class_id].numpy(), float(biases[class_id]))
        selection_signals = {}
        selection_y = {}
        for cid in pools:
            index = split[cid]['selection']
            selection_signals[cid] = route.signals(receiver, router, pools[cid]['X'][index], seen, batch_size, device)
            selection_y[cid] = pools[cid]['y'][index]
        chosen, feedback = select_thresholds(selection_signals[receiver_id], selection_y[receiver_id],
            selection_signals[donor_id], selection_y[donor_id], class_id)
        save_rows(out / 'calibration_selection_feedback.csv', {k:[row[k] for row in feedback] for k in feedback[0]})
        route.metadata.update(tau=chosen['tau'], gamma=chosen['gamma'])
        packet = route.to_packet()
        (out / 'imported_route.bin').write_bytes(packet)
        route = ImportedRoute.from_packet((out / 'imported_route.bin').read_bytes())
        route.validate(receiver, seen)
        decision_lock = dict(selected=chosen, packet_sha256=hashlib.sha256(packet).hexdigest(),
            packet_bytes=len(packet), packet_kib=len(packet)/1024, packet_metadata=route.metadata,
            validation_loaded=False, inference_requires_labels=False, donor_required_at_inference=False,
            split_manifest_sha256=file_sha256(out / 'calibration_split_manifest.json'),
            protocol_sha256=file_sha256(out / 'protocol_lock.json'))
        write_json(out / 'decision_lock.json', decision_lock)
        decision_hash = file_sha256(out / 'decision_lock.json')
        print(f'LOCK before holdout/validation: tau={chosen["tau"]:.6f}, gamma={chosen["gamma"]:.6f}, packet={len(packet)} bytes', flush=True)
        # The packet has been frozen. No subsequent rejection triggers reselection.
        holdout_metrics = {}
        for cid in pools:
            index = split[cid]['holdout']
            sig = route.signals(receiver, router, pools[cid]['X'][index], seen, batch_size, device)
            decision = route.decisions(sig)
            holdout_metrics[cid] = metrics(pools[cid]['y'][index], decision['pred'], sig['local_pred'], decision['activated'], class_id)
        receiver_holdout = holdout_metrics[receiver_id]
        donor_holdout = holdout_metrics[donor_id]
        holdout_gate = (receiver_holdout['rows'] >= ROUTE_RULES['min_receiver_holdout_rows'] and
            donor_holdout['positive_rows'] >= ROUTE_RULES['min_holdout_positives'] and
            receiver_holdout['rescue'] >= receiver_holdout['break_count'] and donor_holdout['rescue'] > 0)
        write_json(out / 'calibration_holdout.json', dict(metrics=holdout_metrics, passed=holdout_gate,
            no_reselection=True, donor_quality_holdout_independence=False))
        costs = encoder_cache_inventory(receiver, role_hash)
        costs.update(patch_bytes=len(packet), patch_kib=len(packet)/1024,
            cold_encoder_exceeds_original_outbound_budget=costs['cold_encoder_bytes'] > protocol.max_outgoing_bytes,
            original_outbound_budget=protocol.max_outgoing_bytes,
            prototype_raw_inputs_exported=False, actual_transport_verified=False)
        write_json(out / 'communication_inventory.json', costs)
        # Validation is first opened AFTER decision_lock.json and calibration holdout.
        val = {cid:current_pool(roles, cid, 'validation', classes) for cid in pools}
        inputs = np.concatenate([val[cid]['X'] for cid in val])
        labels = np.concatenate([val[cid]['y'] for cid in val])
        origins = np.concatenate([np.full(len(val[cid]['y']), cid) for cid in val])
        rows = np.concatenate([val[cid]['rows'] for cid in val])
        print(f'Frozen validation: {len(inputs)} rows; labels joined only after decisions', flush=True)
        signals = route.signals(receiver, router, inputs, seen, batch_size, device)
        no_guard = route.decisions(signals, guarded=False)
        guarded = route.decisions(signals, guarded=True)
        legacy_candidate, legacy_detector = probe(receiver, router, route.head_weight, route.head_bias, class_id, task)
        legacy = predict.records(legacy_candidate, legacy_detector, inputs)
        if not np.array_equal(legacy['task'], signals['legacy_task']):
            raise RuntimeError('Head-only branch changed legacy routing')
        variants = dict(local_baseline=dict(pred=signals['local_pred'], activated=np.zeros(len(labels), bool)),
            head_legacy=dict(pred=legacy['pred'], activated=(legacy['pred'] == class_id)),
            imported_no_guard=no_guard, imported_guard=guarded)
        # Decision contract verification uses x only and the decoded receiver packet.
        # Exact decisions are saved; no label-aware routing or metric-informed refit occurs.
        validation_metrics = {name:{scope:metrics(labels[mask], value['pred'][mask], signals['local_pred'][mask],
            value['activated'][mask], class_id) for scope,mask in (
                ('pooled',np.ones(len(labels),bool)), ('receiver',origins == receiver_id), ('donor',origins == donor_id))}
            for name,value in variants.items()}
        reproduced = dict(prior_present=prior is not None)
        if prior is not None:
            with np.load(io.BytesIO(prior_read(prior, 'head_only/diagnostic_predictions.npz')), allow_pickle=False) as saved:
                expected = dict(origin_client=origins, row_id=rows, y_true=labels,
                    baseline=signals['local_pred'], probe=legacy['pred'], routed_task=signals['legacy_task'])
                checks = {k:np.array_equal(saved[k],v) for k,v in expected.items()}
                mismatches = {k:int(np.count_nonzero(saved[k] != v)) if saved[k].shape == v.shape else -1 for k,v in expected.items()}
            reproduced.update(checks=checks, mismatches=mismatches)
            if not all(checks.values()):
                write_json(out / 'reproduction.json', reproduced)
                raise Rejected('PRIOR_PREDICTION_REPRODUCTION_FAILED')
        write_json(out / 'reproduction.json', reproduced)
        columns = dict(origin_client=origins, row_id=rows, y_true=labels, **signals,
            no_guard_pred=no_guard['pred'], no_guard_activated=no_guard['activated'],
            guarded_pred=guarded['pred'], guarded_activated=guarded['activated'], head_legacy_pred=legacy['pred'])
        save_rows(out / 'validation_predictions.csv', columns)
        np.savez_compressed(out / 'validation_predictions.npz', **columns)
        source_checks = {name:changed_state(original[name],m,r) for name,m,r in (
            ('receiver',receiver,router), ('donor',donor,donor_router))}
        write_json(out / 'source_state_checks.json', source_checks)
        if not all(v['unchanged'] for v in source_checks.values()):
            raise RuntimeError('Source weights or algorithm state changed')
        if file_sha256(out / 'decision_lock.json') != decision_hash or hashlib.sha256((out / 'imported_route.bin').read_bytes()).hexdigest() != decision_lock['packet_sha256']:
            raise RuntimeError('Frozen route or lock changed after validation')
        guarded_metrics = validation_metrics['imported_guard']
        validation_gate = (guarded_metrics['pooled']['rescue'] > guarded_metrics['pooled']['break_count'] and
            guarded_metrics['receiver']['rescue'] >= guarded_metrics['receiver']['break_count'] and
            guarded_metrics['pooled']['recall'] > validation_metrics['head_legacy']['pooled']['recall'])
        summary = dict(validation_metrics=validation_metrics, calibration_holdout_passed=holdout_gate,
            routing_candidate_gate=holdout_gate and validation_gate, validation_gate=validation_gate,
            selected_thresholds=chosen, costs=costs,
            score_diagnostics={str(c):dict(signature=quantiles(signals['signature_score'][labels == c]),
                margin=quantiles(signals['margin'][labels == c])) for c in sorted(np.unique(labels))},
            no_donor_at_inference=True, inference_label_blind=True, lock_unchanged=True,
            validation_is_development=True, final_test_opened=False,
            historical_retention_unmeasured=True, survival_unmeasured=True, no_install=True,
            not_full_method_accuracy=True, base_method=method, fixture=method != 'legacy')
        write_json(out / 'route_summary.json', summary)
        write_json(out / 'completion.json', dict(completed_execution=True, passed_feasibility=False,
            routing_candidate_gate=summary['routing_candidate_gate'], stage='complete',
            base_method=method, diagnostic_fixture=method != 'legacy', no_install=True,
            final_test_opened=False, selected_tau=chosen['tau'], selected_gamma=chosen['gamma'],
            reason='Pairwise routing only; cold-cache transport, legacy replication and survival remain unverified'))
        for name,value in validation_metrics.items():
            print(f'{name}: pooled acc={value["pooled"]["accuracy"]:.4%}, recall={value["pooled"]["recall"]:.4%}, rescue={value["pooled"]["rescue"]}, break={value["pooled"]["break_count"]}', flush=True)
    except Rejected as exc:
        write_json(out / 'completion.json', dict(completed_execution=True, passed_feasibility=False,
            stage='protocol_rejected', reason=exc.reason, detail=exc.detail, no_install=True))
        print(f'Imported route rejected: {exc.reason}: {exc.detail}', flush=True)
    except Exception as exc:
        write_json(out / 'completion.json', dict(completed_execution=False, passed_feasibility=False,
            stage='execution_error', error_type=type(exc).__name__, detail=str(exc)))
        raise
    return json.loads((out / 'completion.json').read_text(encoding='utf-8'))
