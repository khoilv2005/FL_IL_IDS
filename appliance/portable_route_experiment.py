"""Accounting + calibration-locked shared-sketch/self-protection pairwise audit."""
import hashlib
import json
import copy
from pathlib import Path

import numpy as np
import torch

from .closure import effective_linear
from .config import Protocol, Rejected
from .evaluate import Predictor
from .imported_route import ImportedRoute, ROUTE_RULES, encoder_cache_inventory, select_thresholds, stratified_roles, unit_rows
from .parallel_route_experiment import metrics, save_rows
from .portable_route import (PORTABLE_RULES, ProtectedRoute, SharedSketch, choose_protection, choose_anchor_beta,
    prototype_summary, receiver_signals, transitions)
from .routing_audit import histogram, quantiles
from .runner import load_input
from .selector import current_pool, recorded_graph, select_pair
from .state import boundary_hash, changed_state, state_fingerprint, write_json
from .transfer_audit import features


def audit_previous_accounting(previous, out):
    previous = Path(previous)
    with np.load(previous / 'validation_predictions.npz',allow_pickle=False) as old:
        y, base, origins = old['y_true'],old['local_pred'],old['origin_client']
        result = {}
        for name,key in [('guarded','guarded_pred'),('no_guard','no_guard_pred'),('head_legacy','head_legacy_pred')]:
            result[name] = {scope:transitions(y[m],base[m],old[key][m]) for scope,m in (
                ('pooled',np.ones(len(y),bool)),('receiver',origins==19),('donor',origins==71))}
    write_json(out / 'previous_accounting.json',dict(source=str(previous),transitions=result,
        explanation='Prior rescue/break were pooled; 77.04% -> 76.35% was receiver-only. All identities hold.'))
    return result


def run_portable_route(checkpoint, role_dir, data_dir, out, previous=None, task=4, round_id=19,
                       receiver_id=19, donor_id=71, class_id=24, device='cpu', batch_size=512,
                       allow_cgofed_fixture=False, router_mode='checkpoint'):
    from eval_checkpoint import _make_denice_client_model
    from fed_learning.data.denice_clean_roles import CleanRoleData, file_sha256
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    write_json(out / 'completion.json',dict(completed_execution=False,passed_feasibility=False))
    try:
        if previous is not None:
            audit_previous_accounting(previous,out)
        if task not in range(6) or round_id < 0 or batch_size < 1 or router_mode not in ('checkpoint','multiclass_balanced'):
            raise ValueError('Invalid experiment scope')
        ckpt,hashes = load_input(checkpoint,task,round_id)
        config = ckpt['config']
        method = config.get('denice_cl_method','legacy')
        if method != 'legacy' and not (method == 'cgofed' and allow_cgofed_fixture):
            raise Rejected('LEGACY_BASELINE_REQUIRED',method)
        roles = CleanRoleData(role_dir,source_data_dir=data_dir)
        role_hash = file_sha256(roles.root / 'role_manifest.json')
        if config.get('denice_data_roles_sha256') != role_hash:
            raise Rejected('BACKBONE_ROLE_LOCK_MISMATCH')
        if tuple(config['input_shape']) != tuple(roles.manifest['input_shape']):
            raise Rejected('PREPROCESSING_SHAPE_MISMATCH')
        if config.get('denice_max_train_samples_per_client') or config.get('denice_max_clients') not in (None,100):
            raise Rejected('TRUNCATED_BASE_TRAINING_NOT_SUPPORTED')
        preprocessing_hash = file_sha256(roles.source / 'metadata.json')
        metadata = json.loads((roles.source / 'metadata.json').read_text(encoding='utf-8'))
        task_classes = {int(k):list(map(int,v)) for k,v in metadata['task_structure']['task_classes'].items()}
        classes = task_classes[task]
        seen = sorted({c for t,v in task_classes.items() if t <= task for c in v})
        if sorted(map(int,ckpt['seen_classes'])) != seen or class_id not in classes:
            raise Rejected('CLASS_OR_SEEN_SCOPE_MISMATCH')
        groups,alphas = recorded_graph(ckpt,task,round_id)
        predict = Predictor(seen,device,batch_size)
        protocol = Protocol().validate()
        root = Path(__file__).resolve().parents[1]
        source_files = list((root / 'appliance').glob('*.py')) + [root/n for n in (
            'eval_checkpoint.py','fed_learning/models/nice_model.py','fed_learning/models/denice_model.py',
            'fed_learning/servers/nice_server.py','fed_learning/training/denice_eval.py',
            'fed_learning/training/checkpoint_state.py','fed_learning/data/denice_clean_roles.py')]
        lock = dict(**PORTABLE_RULES,**hashes,task=task,round=round_id,receiver=receiver_id,donor=donor_id,
            class_id=class_id,base_method=method,fixture=method != 'legacy',
            evaluation_router=router_mode,
            router_refit_source='none' if router_mode=='checkpoint' else 'current task checkpoint activation memory only; no raw data or future-task router',
            role_manifest_sha256=role_hash,preprocessing_sha256=preprocessing_hash,
            xi=config.get('denice_similarity_threshold'),input_shape=config['input_shape'],
            calibration_split='same per-class 50/25/25; seed 20261008 + client ID',
            prototype_fit='donor current calibration FIT positives only',
            threshold_selection='current calibration SELECTION; all variants frozen before HOLDOUT and validation',
            holdout_caveat='previously viewed development role; not independent confirmation',
            simulation_caveat='offline coordinator; actual decentralized feedback transport not implemented',
            device=device,batch_size=batch_size,versions=dict(torch=torch.__version__,numpy=np.__version__),
            source_sha256={p.relative_to(root).as_posix():file_sha256(p) for p in source_files})
        write_json(out / 'protocol_lock.json',lock)
        fitted_routers = {}
        checkpoint_router_hashes = {}

        def make_model(cid):
            model,router = _make_denice_client_model(ckpt,cid,device)
            state = ckpt['client_model_states'].get(cid,ckpt['client_model_states'].get(str(cid)))
            restored = model.state_dict()
            if set(restored) != set(state) or any(not torch.equal(v.detach().cpu(),state[n]) for n,v in restored.items()):
                raise Rejected('INEXACT_MODEL_RESTORE',str(cid))
            if any(int(ep)>task for ep in router.activation_memory):
                raise Rejected('FUTURE_ROUTER_MEMORY')
            if router_mode=='multiclass_balanced':
                from fed_learning.training.checkpoint_state import snapshot_context_detector
                from .state import digest
                import joblib
                checkpoint_router_hashes[cid] = digest(snapshot_context_detector(router))
                weights_before = {n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
                router.router_mode='multiclass_balanced'
                router.train_models(max(router.activation_memory))
                if any(not torch.equal(v.detach().cpu(),weights_before[n]) for n,v in model.state_dict().items()):
                    raise RuntimeError('Multiclass refit mutated backbone weights')
                snap = snapshot_context_detector(router)
                if cid in fitted_routers and digest(snap)!=digest(fitted_routers[cid]):
                    raise RuntimeError('Repeated current-memory router fit changed')
                fitted_routers[cid] = snap
                joblib.dump(fitted_routers,out / 'fitted_current_task_routers.joblib',compress=3)
            return model,router

        pair,ledger,offers = select_pair(ckpt,roles,task,classes,groups,make_model,predict,protocol,
            (receiver_id,donor_id,class_id),eligibility_path=out / 'pair_eligibility.json')
        write_json(out / 'pair_selection.json',dict(selected=pair,offers=offers,ledger=ledger.to_dict()))
        receiver,router = make_model(receiver_id)
        donor,donor_router = make_model(donor_id)
        if any(getattr(m,'continual_head',None) is not None or getattr(m,'local_classifier',None) is not None for m in (receiver,donor)):
            raise Rejected('UNSUPPORTED_AUXILIARY_CLASSIFIER')
        before = {name:state_fingerprint(m,r) for name,m,r in (
            ('receiver',receiver,router),('donor',donor,donor_router))}
        pools = {cid:current_pool(roles,cid,'calibration',classes) for cid in (receiver_id,donor_id)}
        split = {cid:stratified_roles(pools[cid],ROUTE_RULES['seed']+cid) for cid in pools}
        manifest = {cid:{role:dict(rows=pools[cid]['rows'][index],class_counts=histogram(pools[cid]['y'][index]),
            missing_current_classes=sorted(set(classes)-set(map(int,pools[cid]['y'][index]))))
            for role,index in value.items()} for cid,value in split.items()}
        write_json(out / 'calibration_split_manifest.json',manifest)
        split_hash = file_sha256(out / 'calibration_split_manifest.json')
        head,bias = effective_linear(donor,'fc2')
        weight,b = head[class_id].numpy(),float(bias[class_id])
        fit = split[donor_id]['fit']
        positives = fit[pools[donor_id]['y'][fit] == class_id]
        fit_x = pools[donor_id]['X'][positives]
        z,valid = unit_rows(features(receiver,fit_x,None,batch_size,device))
        prototype,variance,support = prototype_summary(z,valid)
        receiver_hash = boundary_hash(receiver,True)
        base_metadata = dict(kind='protected_imported_route',version=PORTABLE_RULES['version'],
            receiver=receiver_id,donor=donor_id,class_id=class_id,task=task,tau=1.,gamma=0.,beta=0.,
            receiver_feature_hash=receiver_hash,checkpoint_hashes=hashes,role_manifest_sha256=role_hash,
            split_manifest_sha256=split_hash,head_scope='donor effective FC2; receiver imported context',
            self_confidence=PORTABLE_RULES['self_confidence'],no_legacy_task_prerequisite=True)
        routes = dict(receiver256=ProtectedRoute(dict(base_metadata,signature=dict(kind='receiver_space',dimension=256),
            support=support),prototype,variance,weight,b))
        for dimension in PORTABLE_RULES['dimensions']:
            sketch = SharedSketch(tuple(config['input_shape']),dimension,preprocessing_hash)
            z,valid = sketch.features(fit_x)
            proto,var,support = prototype_summary(z,valid)
            routes[f'shared{dimension}'] = ProtectedRoute(dict(base_metadata,signature=sketch.manifest(),support=support),proto,var,weight,b)
        base_selection = {cid:receiver_signals(receiver,router,pools[cid]['X'][split[cid]['selection']],
            seen,task,class_id,batch_size,device) for cid in pools}
        selection_y = {cid:pools[cid]['y'][split[cid]['selection']] for cid in pools}
        selections,packet_hashes,packet_sizes = {},{},{}
        (out / 'packets').mkdir()
        # Legacy two-gate reference is recalibrated by the original rule, before validation.
        reference = ImportedRoute(dict(kind='parallel_imported_route_experiment',tau=1.,gamma=0.,
            receiver_feature_hash=receiver_hash,class_id=class_id,task=task),prototype,weight,b)
        reference_selection = {cid:routes['receiver256'].signals(base_selection[cid],
            pools[cid]['X'][split[cid]['selection']]) for cid in pools}
        old_choice,_ = select_thresholds(reference_selection[receiver_id],selection_y[receiver_id],
            reference_selection[donor_id],selection_y[donor_id],class_id)
        reference.metadata.update(tau=old_choice['tau'],gamma=old_choice['gamma'])
        routes['receiver256_anchor_self'] = copy.deepcopy(routes['receiver256'])
        old_packet = reference.to_packet()
        (out / 'packets' / 'receiver256_two_gate.bin').write_bytes(old_packet)
        reference = ImportedRoute.from_packet(old_packet)
        for name,route in routes.items():
            sig = {cid:route.signals(base_selection[cid],pools[cid]['X'][split[cid]['selection']]) for cid in pools}
            if name == 'receiver256_anchor_self':
                chosen,feedback = choose_anchor_beta(sig[receiver_id],selection_y[receiver_id],sig[donor_id],selection_y[donor_id],
                    class_id,old_choice['tau'],old_choice['gamma'])
            else:
                chosen,feedback = choose_protection(sig[receiver_id],selection_y[receiver_id],sig[donor_id],selection_y[donor_id],class_id)
            selections[name] = chosen
            save_rows(out / f'{name}_selection_feedback.csv',{k:[r[k] for r in feedback] for k in feedback[0]})
            route.metadata.update(tau=chosen['tau'],gamma=chosen['gamma'],beta=chosen['beta'])
            packet = route.packet()
            (out / 'packets' / f'{name}.bin').write_bytes(packet)
            routes[name] = ProtectedRoute.from_packet(packet)
            routes[name].validate(receiver,preprocessing_hash,seen)
            packet_hashes[name] = hashlib.sha256(packet).hexdigest()
            packet_sizes[name] = len(packet)
            print(f'LOCK {name}: selection recall={chosen["true_positive"]}/{chosen["positive_rows"]}, '
                f'tau={chosen["tau"]:.6f}, gamma={chosen["gamma"]:.6f}, beta={chosen["beta"]:.9f}',flush=True)
        shared_names = [name for name in routes if name.startswith('shared')]
        primary = sorted(shared_names,key=lambda n:(-selections[n]['true_positive'],
            selections[n]['false_positive_all'],packet_sizes[n],n))[0]
        write_json(out / 'decision_lock.json',dict(selection=selections,primary_shared=primary,
            primary_selected_using='SELECTION counts only; packet size tie-break',packet_sha256=packet_hashes,
            packet_bytes=packet_sizes,reference_two_gate_selection=old_choice,
            reference_sha256=hashlib.sha256(old_packet).hexdigest(),validation_loaded=False,
            holdout_loaded=False,protocol_sha256=file_sha256(out / 'protocol_lock.json')))
        decision_hash = file_sha256(out / 'decision_lock.json')
        router_artifact = out / 'fitted_current_task_routers.joblib'
        router_lock = dict(evaluation_router=router_mode,checkpoint_router_hashes=checkpoint_router_hashes,
            fitted_artifact_sha256=file_sha256(router_artifact) if router_artifact.is_file() else None,
            task=task,fit_completed_before_holdout_validation=True,future_task_memory_used=False)
        write_json(out / 'router_lock.json',router_lock)

        def evaluate(role):
            role_pools = {cid:(dict(X=pools[cid]['X'][split[cid][role]],y=pools[cid]['y'][split[cid][role]],
                rows=pools[cid]['rows'][split[cid][role]]) if role=='holdout' else current_pool(roles,cid,'validation',classes)) for cid in pools}
            x = np.concatenate([role_pools[cid]['X'] for cid in role_pools])
            y = np.concatenate([role_pools[cid]['y'] for cid in role_pools])
            origins = np.concatenate([np.full(len(role_pools[cid]['y']),cid) for cid in role_pools])
            row_ids = np.concatenate([role_pools[cid]['rows'] for cid in role_pools])
            base = receiver_signals(receiver,router,x,seen,task,class_id,batch_size,device)
            ref_sig = routes['receiver256'].signals(base,x)
            variants = dict(local=dict(pred=base['local_pred'],activated=np.zeros(len(y),bool)),
                            receiver256_two_gate=reference.decisions(ref_sig))
            scores = {}
            for name,route in routes.items():
                sig = route.signals(base,x)
                scores[name] = sig
                variants[name] = route.decisions(sig)
                variants[name+'_without_self'] = route.decisions(sig,self_protection=False)
            masks = dict(pooled=np.ones(len(y),bool),receiver=origins==receiver_id,donor=origins==donor_id)
            measured = {name:{scope:dict(**metrics(y[m],value['pred'][m],base['local_pred'][m],value['activated'][m],class_id),
                transition=transitions(y[m],base['local_pred'][m],value['pred'][m])) for scope,m in masks.items()} for name,value in variants.items()}
            from sklearn.metrics import roc_auc_score
            auc = {name:float(roc_auc_score(y==class_id,sig['signature_score'])) for name,sig in scores.items()}
            columns = dict(origin_client=origins,row_id=row_ids,y_true=y,local_pred=base['local_pred'],
                           local_confidence=base['local_confidence'],legacy_task=base['legacy_task'])
            for name,value in variants.items():
                columns[name+'_pred'],columns[name+'_activated'] = value['pred'],value['activated']
            for name,sig in scores.items():
                columns[name+'_score'],columns[name+'_margin'] = sig['signature_score'],sig['margin']
            save_rows(out / f'{role}_predictions.csv',columns)
            np.savez_compressed(out / f'{role}_predictions.npz',**columns)
            accounting = {name:{scope:measured[name][scope]['transition'] for scope in masks} for name in measured}
            write_json(out / f'{role}_accounting.json',accounting)
            if role=='validation' and previous is not None:
                with np.load(Path(previous) / 'validation_predictions.npz',allow_pickle=False) as old:
                    values = dict(y_true=y,origin_client=origins,row_id=row_ids,local_pred=base['local_pred'],
                        guarded_pred=variants['receiver256_two_gate']['pred'])
                    mismatches = {n:int(np.count_nonzero(old[n] != v)) if old[n].shape==v.shape else -1 for n,v in values.items()}
                write_json(out / 'previous_reproduction.json',dict(mismatch_counts=mismatches))
                if any(mismatches.values()):
                    raise Rejected('PREVIOUS_ROUTE_REPRODUCTION_FAILED')
            confidence = {str(c):dict(quantiles=quantiles(base['local_confidence'][y==c]),
                exactly_one=int((base['local_confidence'][y==c]==1).sum())) for c in sorted(np.unique(y))}
            write_json(out / f'{role}_metrics.json',dict(variants=measured,route_auc=auc,local_confidence_by_class=confidence))
            return measured,auc

        holdout,holdout_auc = evaluate('holdout')
        # No retuning or disabling metrics after holdout; validation is diagnostic even when gate fails.
        validation,validation_auc = evaluate('validation')
        def gate(measured,name):
            r,p = measured[name]['receiver'],measured[name]['pooled']
            return r['false_activation_rate'] <= PORTABLE_RULES['receiver_holdout_far_budget'] and r['break_count']==0 and p['rescue']>0
        holdout_gates = {name:gate(holdout,name) for name in routes}
        validation_gates = {name:gate(validation,name) for name in routes}
        costs = {name:dict(packet_bytes=packet_sizes[name],packet_kib=packet_sizes[name]/1024,
            signature_tensor_bytes=routes[name].prototype.nbytes+routes[name].variance.nbytes,
            cold_receiver_encoder_transfer_bytes=0 if name.startswith('shared') else encoder_cache_inventory(receiver,preprocessing_hash)['cold_encoder_bytes'],
            shared_projection=routes[name].metadata['signature'] if name.startswith('shared') else None,
            actual_transport_verified=False,feedback_communication_not_counted=True) for name in routes}
        checks = {name:changed_state(before[name],m,r) for name,m,r in (
            ('receiver',receiver,router),('donor',donor,donor_router))}
        write_json(out / 'source_state_checks.json',checks)
        if not all(v['unchanged'] for v in checks.values()):
            raise RuntimeError('Source weights/router changed')
        if file_sha256(out / 'decision_lock.json') != decision_hash:
            raise RuntimeError('Decision lock mutated after holdout/validation')
        if router_artifact.is_file() and file_sha256(router_artifact)!=router_lock['fitted_artifact_sha256']:
            raise RuntimeError('Frozen fitted routers mutated during evaluation')
        for name,expected in packet_hashes.items():
            if file_sha256(out / 'packets' / f'{name}.bin') != expected:
                raise RuntimeError('Frozen packet mutated')
        primary_gate = holdout_gates[primary] and validation_gates[primary]
        write_json(out / 'portable_summary.json',dict(primary_shared=primary,selections=selections,costs=costs,
            holdout_gates=holdout_gates,validation_gates=validation_gates,routing_candidate_gate=primary_gate,
            validation_metrics=validation,route_auc=validation_auc,holdout_metrics=holdout,
            inference_label_blind=True,no_donor_model_at_inference=True,no_retrain=True,
            evaluation_router=router_mode,
            no_install=True,passed_feasibility=False,final_test_opened=False,
            development_only=True,old_task_retention_unmeasured=True,survival_unmeasured=True,
            sketch_note='input has 39 scalar coordinates: 64/128D expand input, they are not compression of raw input'))
        write_json(out / 'completion.json',dict(completed_execution=True,passed_feasibility=False,
            stage='complete',primary_shared=primary,routing_candidate_gate=primary_gate,no_install=True,
            final_test_opened=False,base_method=method,fixture=method != 'legacy'))
        for name in ['receiver256_two_gate',*routes]:
            p,r = validation[name]['pooled'],validation[name]['receiver']
            print(f'{name}: recall={p["recall"]:.4%}, break={p["break_count"]}, '
                f'receiver FAR={r["false_activation_rate"]:.4%}, AUC={validation_auc.get(name,validation_auc["receiver256"]):.6f}',flush=True)
    except Rejected as exc:
        if hasattr(exc,'audit'):
            write_json(out / 'rejected_donor_offers.json',exc.audit)
        write_json(out / 'completion.json',dict(completed_execution=True,passed_feasibility=False,
            stage='protocol_rejected',reason=exc.reason,detail=exc.detail,no_install=True))
        print(f'Portable route rejected: {exc.reason}: {exc.detail}',flush=True)
    except Exception as exc:
        write_json(out / 'completion.json',dict(completed_execution=False,passed_feasibility=False,
            stage='execution_error',error_type=type(exc).__name__,detail=str(exc)))
        raise
    return json.loads((out / 'completion.json').read_text(encoding='utf-8'))
