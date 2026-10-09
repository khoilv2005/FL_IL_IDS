"""Frozen-head diagnostic: select guards using historical calibration negatives.

No backbone training, prototype refit, validation selection, or final-test access.
Historical calibration access is retrospective development, not replay-free proof.
"""
import copy
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import torch

from .config import Rejected
from .guarded_head import (GuardedHeadRegistry, HeadContract, compile_guarded_head,
                           install_guarded_head)
from .guarded_head_experiment import combine
from .imported_route import ROUTE_RULES, stratified_roles
from .parallel_route_experiment import metrics
from .portable_route import ProtectedRoute, receiver_signals
from .runner import load_input
from .selector import current_pool, lookup, recorded_graph
from .state import complete_hash, write_json


PAIRS = ((3, 6, 24, 'active_capability'), (19, 71, 24, 'frozen_capability'))
RULES = dict(version='appliance_historical_calibration_v1', task=4, round=19,
    far_budget=.001, min_recall=.95, max_break=0, quantile_points=31,
    selector='max target activations, min all false activations, stricter beta/gamma/tau',
    constraints='receiver pooled AND each receiver-class empirical FAR <= .001; zero pooled break',
    signature_refitted=False, head_refitted=False, router_refitted=False,
    historical_roles='receiver CALIBRATION only; historical task-wise 50/25/25 split',
    current_roles='preserve source locked FIT/SELECTION/HOLDOUT row IDs exactly',
    donor_roles='source current-task CALIBRATION SELECTION/HOLDOUT only',
    no_final_test=True, replay_free_proof=False)


def subset(pool, index, cid):
    return dict(X=pool['X'][index], y=pool['y'][index], rows=pool['rows'][index],
                origin_client=np.full(len(index), cid, dtype=np.int64))


def calibration_views(roles, cid, task_classes, locked, historical, current_task=4):
    """Past-task RNG cannot reassign the frozen prototype's current FIT rows."""
    current = current_pool(roles, cid, 'calibration', task_classes[current_task])
    coordinates = {int(row): index for index, row in enumerate(current['rows'])}
    result = {name: [] for name in ('fit', 'selection', 'holdout')}
    joined = []
    for name in result:
        rows = list(map(int, locked[str(cid)][name]['rows']))
        if len(set(rows)) != len(rows) or any(row not in coordinates for row in rows):
            raise Rejected('LOCKED_CALIBRATION_COORDINATES_CHANGED', str(cid))
        joined.extend(rows)
        result[name].append(subset(current, np.asarray([coordinates[row] for row in rows], dtype=np.int64), cid))
    if len(set(joined)) != len(joined) or set(joined) != set(map(int, current['rows'])):
        raise Rejected('CURRENT_CALIBRATION_SPLIT_OVERLAP_OR_OMISSION', str(cid))
    if historical:
        past = current_pool(roles, cid, 'calibration', [c for t in range(current_task) for c in task_classes[t]])
        for task in range(current_task):
            index = np.flatnonzero(np.isin(past['y'], task_classes[task]))
            pool = subset(past, index, cid)
            split = stratified_roles(pool, ROUTE_RULES['seed'] + cid + 100003 * (task + 1))
            for name, positions in split.items():
                result[name].append(subset(pool, positions, cid))
    result = {name: combine(pools) for name, pools in result.items()}
    all_rows = np.concatenate([p['rows'] for p in result.values()])
    if len(np.unique(all_rows)) != len(all_rows):
        raise Rejected('HISTORICAL_CALIBRATION_SPLIT_OVERLAP', str(cid))
    return result


def select_guard(signals, y, origins, receiver, class_id, old):
    """Prefix counts avoid scanning all rows separately for every beta."""
    y, origins = np.asarray(y), np.asarray(origins)
    rmask, positive = origins == receiver, y == class_id
    if rmask.sum() < 32 or positive.sum() < 8 or np.any(rmask & positive):
        raise Rejected('INSUFFICIENT_HISTORICAL_SELECTION')
    score, margin, conf = [np.asarray(signals[k]) for k in ('signature_score', 'margin', 'local_confidence')]
    if not all(np.isfinite(v).all() for v in (score, margin, conf)):
        raise Rejected('NONFINITE_GUARD_SELECTION')
    q = np.linspace(0, 1, RULES['quantile_points'])
    taus = np.unique(np.clip(np.r_[-1., 1., old['tau'], np.quantile(score, q)], -1, 1))
    gammas = np.unique(np.maximum(0, np.r_[0., old['gamma'], np.quantile(margin, q)]))
    betas = np.unique(np.clip(np.r_[0., 1., old['beta'], np.quantile(conf, q)], 0, 1))
    order = np.argsort(conf, kind='stable')
    cuts = np.searchsorted(conf[order], betas, side='left')  # strict confidence < beta
    receiver_classes = sorted(map(int, np.unique(y[rmask])))
    rows_per_class = {c: int((rmask & (y == c)).sum()) for c in receiver_classes}
    # Activating any non-target correct local prediction is a break.
    masks = [positive, rmask, ~positive, (~positive) & (signals['local_pred'] == y)]
    masks += [rmask & (y == c) for c in receiver_classes]
    masks = [m[order] for m in masks]
    best, examined = None, 0
    for tau in taus:
        hit = signals['signature_valid'] & (score > tau)
        for gamma in gammas:
            candidate = (hit & (margin > gamma))[order]
            counts = [np.r_[0, np.cumsum(candidate & m)][cuts] for m in masks]
            for bindex, beta in enumerate(betas):
                tp, fp_receiver, fp_all, broken = [int(values[bindex]) for values in counts[:4]]
                class_far = {c: int(counts[4+i][bindex])/rows_per_class[c] for i,c in enumerate(receiver_classes)}
                feasible = (broken == 0 and fp_receiver/int(rmask.sum()) <= RULES['far_budget']
                            and all(far <= RULES['far_budget'] for far in class_far.values()))
                examined += 1
                key = (tp, -fp_all, -float(beta), float(gamma), float(tau))
                if feasible and (best is None or key > best[0]):
                    best = (key, dict(tau=float(tau), gamma=float(gamma), beta=float(beta),
                        target_activations=tp, positive_rows=int(positive.sum()), receiver_rows=int(rmask.sum()),
                        receiver_false_activations=fp_receiver, all_false_activations=fp_all,
                        break_count=broken, receiver_far=fp_receiver/int(rmask.sum()),
                        receiver_far_by_class=class_far, feasible=True))
    if best is None:raise Rejected('NO_FEASIBLE_HISTORICAL_GUARD')
    return dict(best[1], examined_candidates=examined, grid_points=RULES['quantile_points'],
                receiver_rows_by_class=rows_per_class, empirical_constraints_only=True)


def summarize(pool, decision, base, receiver, class_id):
    scopes = {'pooled': np.ones(len(pool['y']), bool), 'receiver': pool['origin_client'] == receiver,
              'donor': pool['origin_client'] != receiver}
    report = {name: metrics(pool['y'][m], decision['pred'][m], base['local_pred'][m],
                           decision['activated'][m], class_id) for name,m in scopes.items()}
    report['receiver_by_class'] = {int(c): metrics(pool['y'][m], decision['pred'][m], base['local_pred'][m],
        decision['activated'][m], class_id) for c in np.unique(pool['y'][scopes['receiver']])
        for m in [scopes['receiver'] & (pool['y'] == c)]}
    return report


def run_historical_calibration(checkpoint, data_dir, runtime, out, device='cuda', batch_size=512):
    from eval_checkpoint import _make_denice_client_model
    from fed_learning.data.denice_clean_roles import CleanRoleData, file_sha256
    from fed_learning.training.checkpoint_state import restore_context_detector, snapshot_denice_state
    runtime, out = Path(runtime), Path(out)
    if batch_size < 1:raise ValueError('Positive batch size required')
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'completion.json', dict(completed_execution=False, final_test_opened=False))
    started = time.perf_counter()
    try:
        ckpt, hashes = load_input(checkpoint, 4, 19)
        config = ckpt['config']
        if config.get('denice_cl_method') != 'legacy' or config.get('denice_similarity_threshold') != .8:
            raise Rejected('LEGACY_XI8_REQUIRED')
        roles = CleanRoleData(runtime/'roles', source_data_dir=data_dir)
        role_sha = file_sha256(roles.root/'role_manifest.json')
        if config['denice_data_roles_sha256'] != role_sha:raise Rejected('BACKBONE_ROLES_CHANGED')
        metadata = json.loads((roles.source/'metadata.json').read_text(encoding='utf-8'))
        task_classes = {int(t):list(map(int, labels)) for t,labels in metadata['task_structure']['task_classes'].items()}
        seen = sorted(c for t in range(5) for c in task_classes[t])
        if seen != sorted(map(int, ckpt['seen_classes'])):raise Rejected('SEEN_CLASSES_CHANGED')
        groups, alphas = recorded_graph(ckpt, 4, 19)
        write_json(out/'protocol_lock.json', dict(**RULES, **hashes, pairs=PAIRS,
            role_manifest_sha256=role_sha, device=device, batch_size=batch_size,
            versions=dict(torch=torch.__version__,numpy=np.__version__),
            limitations=['two previously selected pairs, both class24', 'historical calibration retrospective',
                         'validation previously viewed', 'not multi-class/full-round/end-to-end evidence']))
        outcomes = []
        for cid, donor, c, folder in PAIRS:
            target = out/f'receiver_{cid}';target.mkdir()
            source = runtime/folder
            protocol = json.loads((source/'protocol_lock.json').read_text(encoding='utf-8'))
            decisions = json.loads((source/'decision_lock.json').read_text(encoding='utf-8'))
            name = decisions['primary_shared'] if 'primary_shared' in decisions else 'shared16'
            packet = (source/'packets'/f'{name}.bin').read_bytes()
            if hashlib.sha256(packet).hexdigest() != decisions['packet_sha256'][name]:
                raise Rejected('FROZEN_PACKET_CHANGED', str(cid))
            if any(protocol[k] != hashes[k] for k in hashes) or protocol['role_manifest_sha256'] != role_sha:
                raise Rejected('SOURCE_LOCK_CHANGED', str(cid))
            if donor not in groups[cid] or alphas[cid].get(donor, 0) <= 0:raise Rejected('ILLEGAL_DONOR')
            counts = roles.manifest['clients'][str(cid)]['role_class_counts']['base']
            if int(counts.get(str(c), 0)) != 0:raise Rejected('RECEIVER_CLASS_NOT_MISSING')
            model, router = _make_denice_client_model(ckpt, cid, device)
            frozen = lookup(joblib.load(source/'fitted_current_task_routers.joblib'), cid)
            if frozen is None or any(int(t)>4 for t in frozen['activation_memory']):raise Rejected('FROZEN_ROUTER_SCOPE')
            restore_context_detector(router, copy.deepcopy(frozen))
            before = complete_hash(model, router)
            old = ProtectedRoute.from_packet(packet)
            old.validate(model, file_sha256(roles.source/'metadata.json'), seen)
            if any(int(old.metadata[k]) != v for k,v in [('receiver',cid),('donor',donor),('class_id',c),('task',4)]):
                raise Rejected('PACKET_PAIR_CHANGED')
            locked = json.loads((source/'calibration_split_manifest.json').read_text(encoding='utf-8'))
            if file_sha256(source/'calibration_split_manifest.json') != old.metadata['split_manifest_sha256']:
                raise Rejected('SOURCE_SPLIT_CHANGED')
            views = {cid: calibration_views(roles,cid,task_classes,locked,True),
                     donor: calibration_views(roles,donor,task_classes,locked,False)}
            manifest = {i:{role:dict(rows=p['rows'], class_counts={int(k):int(v) for k,v in zip(*np.unique(p['y'],return_counts=True))})
                for role,p in client.items()} for i,client in views.items()}
            write_json(target/'calibration_manifest.json', manifest)
            print(f'Historical calibration receiver={cid}: selection={len(views[cid]["selection"]["y"])}, '
                  f'holdout={len(views[cid]["holdout"]["y"])}, donor={donor}', flush=True)
            selection = combine([views[i]['selection'] for i in (cid,donor)])
            base = receiver_signals(model,router,selection['X'],seen,4,c,batch_size,device)
            signals = old.signals(base,selection['X'])
            chosen = select_guard(signals,selection['y'],selection['origin_client'],cid,c,old.metadata)
            np.savez_compressed(target/'selection_signals.npz',y_true=selection['y'],row_id=selection['rows'],
                origin_client=selection['origin_client'],**{k:signals[k] for k in
                ('signature_score','signature_valid','margin','local_confidence','local_pred')})
            proposed = copy.deepcopy(old)
            proposed.metadata.update(tau=chosen['tau'], gamma=chosen['gamma'], beta=chosen['beta'],
                guard_calibration_version=RULES['version'],
                historical_calibration_manifest_sha256=file_sha256(target/'calibration_manifest.json'))
            new_packet = proposed.packet();(target/'candidate.bin').write_bytes(new_packet)
            write_json(target/'decision_lock.json', dict(selected=chosen,
                original_packet_sha256=hashlib.sha256(packet).hexdigest(),
                candidate_packet_sha256=hashlib.sha256(new_packet).hexdigest(),
                source_router_sha256=file_sha256(source/'fitted_current_task_routers.joblib'),
                holdout_signals_scored=False,validation_loaded=False, head_and_prototype_unchanged=True,
                split_labels_used_for_manifest_only=True))
            del selection, base, signals
            acceptance = combine([views[i]['holdout'] for i in (cid,donor)])
            contract = HeadContract()
            compiled = compile_guarded_head(model,router,new_packet,cid,seen,file_sha256(roles.source/'metadata.json'))
            # Gate before staging so a class-specific failure never commits a row.
            hbase = receiver_signals(model,router,acceptance['X'],seen,4,c,batch_size,device)
            hs = proposed.signals(hbase,acceptance['X'])
            hmetrics = summarize(acceptance,proposed.decisions(hs),hbase,cid,c)
            class_gate = all(v['false_activation_rate'] <= RULES['far_budget']
                             for v in hmetrics['receiver_by_class'].values())
            if class_gate:
                installed, detector, registry, txn = install_guarded_head(model,router,compiled,GuardedHeadRegistry(),
                    acceptance,seen,device,batch_size,contract)
            else:
                installed, detector, registry = model, router, GuardedHeadRegistry()
                txn = dict(status='rejected',applied=False,reason='HISTORICAL_PER_CLASS_ACCEPTANCE_FAILED',
                    source_unchanged=complete_hash(model,router)==before)
            write_json(target/'transaction.json', txn)
            usable = bool(txn['applied'])
            write_json(target/'holdout.json',dict(metrics=hmetrics,base_contract=asdict(contract),
                transaction_committed=txn['applied'],per_class_far_passed=class_gate,capability_enabled=usable,
                receiver_missing_classes=sorted(set(seen)-set(map(int,np.unique(acceptance['y'][acceptance['origin_client']==cid])))),
                no_reselection=True))
            if usable:
                bundle=dict(config=config,seen_classes=seen,receiver=cid,
                    client_model_states={cid:installed.state_dict()},
                    client_algorithm_states={cid:{'denice':snapshot_denice_state(installed,detector)}},
                    guarded_head_entries=registry.entries)
                torch.save(bundle,target/'guarded_receiver.pt')
                # Inspect saved state/certificate in this runtime without a new fit.
                saved=torch.load(target/'guarded_receiver.pt',map_location='cpu',weights_only=False)
                restored,rrouter=_make_denice_client_model(saved,cid,device)
                rregistry=GuardedHeadRegistry();rregistry.entries=saved['guarded_head_entries'];rregistry.sync(restored)
                same=(complete_hash(restored,rrouter)==complete_hash(installed,detector)
                      and rregistry.certificate_current(restored,rrouter,c))
                write_json(target/'restore_checks.json',dict(same_runtime_state_and_certificate_exact=bool(same),
                    cross_runtime_portability_verified=False))
                if not same:raise RuntimeError('Accepted receiver save/restore certificate mismatch')
                del restored,rrouter,rregistry,saved,bundle
            # Validation only opens after threshold and acceptance decisions.
            validation = combine([dict(**{k:p[k] for k in ('X','y','rows')},
                origin_client=np.full(len(p['y']),i,dtype=np.int64)) for i,labels in
                ((cid,seen),(donor,task_classes[4])) for p in [current_pool(roles,i,'validation',labels)]])
            vbase = receiver_signals(model,router,validation['X'],seen,4,c,batch_size,device)
            vs = old.signals(vbase,validation['X'])
            variants = dict(local=dict(pred=vbase['local_pred'],activated=np.zeros(len(validation['y']),bool)),
                frozen_guard=old.decisions(vs), candidate_diagnostic=proposed.decisions(vs))
            deployed = registry.records(installed,detector,validation['X'],seen,device,batch_size) if usable else variants['local']
            if usable and not np.array_equal(deployed['pred'],variants['candidate_diagnostic']['pred']):
                raise RuntimeError('Installed candidate predictions differ from portable route')
            variants['accepted_or_local'] = deployed
            reports = {name:summarize(validation,decision,vbase,cid,c) for name,decision in variants.items()}
            write_json(target/'validation.json',reports)
            # The local variant already supplies local_pred below. Pass it once.
            np.savez_compressed(target/'validation_predictions.npz', y_true=validation['y'],
                row_id=validation['rows'],origin_client=validation['origin_client'],
                **{f'{name}_{field}':value[field] for name,value in variants.items() for field in ('pred','activated')})
            if complete_hash(model,router) != before:raise RuntimeError('Original backbone/router mutated')
            result = dict(receiver=cid,donor=donor,class_id=c,installed=bool(txn['applied']),
                capability_enabled=usable,per_class_holdout_passed=class_gate,
                packet_bytes=len(new_packet),validation=reports['accepted_or_local'],
                passed_full_feasibility=False)
            outcomes.append(result);write_json(target/'completion.json',dict(completed_execution=True,**result))
            print(f'Historical calibration receiver={cid}: enabled={usable}, '
                  f'recall={result["validation"]["pooled"]["recall"]}, '
                  f'FAR={result["validation"]["receiver"]["false_activation_rate"]}',flush=True)
            del model,router,installed,detector,registry,validation,views,acceptance,hbase,hs,vbase,vs,variants
            gc.collect()
            if str(device).startswith('cuda'):torch.cuda.empty_cache()
        result = dict(completed_execution=True,outcomes=outcomes,final_test_opened=False,
            passed_full_feasibility=False,training_performed=False,elapsed_seconds=time.perf_counter()-started)
        write_json(out/'completion.json',result);return result
    except Exception as exc:
        write_json(out/'completion.json',dict(completed_execution=False,final_test_opened=False,
            passed_full_feasibility=False,error_type=type(exc).__name__,error=str(exc)))
        raise
