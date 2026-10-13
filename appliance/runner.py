"""Checkpoint-based pairwise feasibility, separate from the full training runner."""
import copy
import hashlib
import io
import json
import random
import zipfile
from pathlib import Path
import numpy as np
import torch
from .config import Protocol,Rejected
from .state import complete_hash,digest,write_json,state_fingerprint,changed_state
from .selector import recorded_graph,select_pair,current_pool
from .closure import compile_patch
from .transaction import install,stage
from .registry import Registry
from .transport import Transport
from .evaluate import Predictor,compare,logit_fidelity


def load_input(path,task,round_id):
    from fed_learning.data.denice_clean_roles import file_sha256
    path=Path(path)
    if path.is_dir():root=path;manifest_path=root/'checkpoint_archive_manifest.json'
    elif path.suffix=='.pt':root=path.parent;manifest_path=root/'checkpoint_archive_manifest.json'
    else:root=None;manifest_path=None
    if root is not None:
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        def read(name):
            if Path(name).name!=name:raise Rejected('INVALID_ARCHIVE_MEMBER')
            member=root/name
            if file_sha256(member)!=manifest['checksums'][name]:raise Rejected('CHECKPOINT_CHECKSUM_MISMATCH',name)
            return torch.load(member,map_location='cpu',weights_only=False)
    else:
        with zipfile.ZipFile(path) as archive:manifest=json.loads(archive.read('checkpoint_archive_manifest.json'))
        def read(name):
            with zipfile.ZipFile(path) as archive:data=archive.read(name)
            if hashlib.sha256(data).hexdigest()!=manifest['checksums'][name]:raise Rejected('CHECKPOINT_CHECKSUM_MISMATCH',name)
            return torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)
    terminal=manifest.get('full_terminal_checkpoint')
    if int(manifest.get('task_id',-1))!=task or not terminal:raise Rejected('FULL_FP32_TERMINAL_REQUIRED')
    graph_name=f'checkpoint_task_{task}_round_{round_id}.pt'
    if graph_name not in manifest['checksums']:raise Rejected('GRAPH_ROUND_NOT_ARCHIVED')
    if path.suffix=='.pt' and path.name!=terminal:raise Rejected('NONTERMINAL_CHECKPOINT')
    ckpt=read(terminal);graph=read(graph_name)['cluster']
    if int(ckpt['task'])!=task or int(ckpt['final_round_id'])!=round_id:raise Rejected('TASK_ROUND_MISMATCH')
    if not ckpt.get('client_model_states'):raise Rejected('EMPTY_TERMINAL_STATE')
    for cid,state in ckpt['client_model_states'].items():
        for name,value in state.items():
            if value.is_floating_point() and (value.dtype!=torch.float32 or not torch.isfinite(value).all()):
                raise Rejected('NONFINITE_OR_NON_FP32_TERMINAL',f'{cid}/{name}')
    ckpt['cluster_history']=[graph]
    return ckpt,dict(terminal_sha256=manifest['checksums'][terminal],graph_sha256=manifest['checksums'][graph_name])


def run(checkpoint,role_dir,data_dir,out,task=5,round_id=19,device='cpu',batch_size=512,
        allow_cgofed_fixture=False,manual=None,survival_rows=512,skip_survival=False):
    from eval_checkpoint import _make_denice_client_model
    from fed_learning.data.denice_clean_roles import CleanRoleData,file_sha256
    from fed_learning.training.checkpoint_state import snapshot_denice_state
    from .survival import run_survival
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new output directory; protocol artifacts are immutable')
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'completion.json',dict(completed_execution=False,passed_feasibility=False,stage='load'))
    protocol=Protocol().validate();random.seed(protocol.seed);np.random.seed(protocol.seed);torch.manual_seed(protocol.seed)
    transport=None;selected=None;outcomes={}
    def transaction_event(kind,values):
        with (out/'transaction_log.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(dict(kind=kind,**values))+'\n')
    try:
        if batch_size<1 or task not in range(6) or round_id<0 or survival_rows<0:raise ValueError('Invalid scope/budget')
        ckpt,hashes=load_input(checkpoint,task,round_id);config=ckpt['config']
        roles=CleanRoleData(role_dir,source_data_dir=data_dir)
        if tuple(config['input_shape'])!=tuple(roles.manifest['input_shape']):raise Rejected('PREPROCESSING_SHAPE_MISMATCH')
        role_hash=file_sha256(roles.root/'role_manifest.json')
        if config.get('denice_data_roles_sha256')!=role_hash:raise Rejected('BACKBONE_ROLE_LOCK_MISMATCH')
        method=config.get('denice_cl_method','legacy')
        if method!='legacy' and not (allow_cgofed_fixture and method=='cgofed'):raise Rejected('LEGACY_BASELINE_REQUIRED',method)
        if config.get('denice_max_train_samples_per_client') or config.get('denice_max_clients') not in (None,100):
            raise Rejected('TRUNCATED_BASE_TRAINING_NOT_SUPPORTED')
        metadata=json.loads((roles.source/'metadata.json').read_text(encoding='utf-8'))
        all_classes={int(t):list(map(int,values)) for t,values in metadata['task_structure']['task_classes'].items()}
        classes=all_classes[task];seen=sorted({c for t,values in all_classes.items() if t<=task for c in values})
        if sorted(map(int,ckpt['seen_classes']))!=seen:raise Rejected('SEEN_SCOPE_MISMATCH')
        groups,alphas=recorded_graph(ckpt,task,round_id)
        edges={(i,j) for i,neighbors in groups.items() for j in neighbors}
        edges |= {(j,i) for i,j in list(edges)}
        transport=Transport(out/'traffic_log.jsonl',edges,protocol)
        preprocessing_hash=digest(dict(metadata_sha256=roles.manifest['metadata_sha256'],input_shape=config['input_shape'],dtype='float32'))
        import sklearn
        root=Path(__file__).resolve().parents[1]
        source_files=list((root/'appliance').glob('*.py'))+[root/name for name in (
            'eval_checkpoint.py','fed_learning/models/nice_model.py','fed_learning/models/denice_model.py',
            'fed_learning/clients/nice_client.py','fed_learning/clients/denice_client.py',
            'fed_learning/strategies/incremental/nice.py','fed_learning/strategies/decentralized/denice_aggregation.py',
            'fed_learning/training/checkpoint_state.py','fed_learning/training/denice_eval.py',
            'fed_learning/data/denice_clean_roles.py')]
        lock=dict(protocol=protocol.locked(),**hashes,role_manifest_sha256=role_hash,preprocessing_hash=preprocessing_hash,
            task=task,round=round_id,xi=config.get('denice_similarity_threshold'),base_method=method,
            diagnostic_fixture=method!='legacy',manual_pair=manual,device=device,batch_size=batch_size,
            survival_rows=survival_rows,skip_survival=skip_survival,
            pair_rule='current task only; missing local BASE class; sufficient current receiver acceptance data; positive recorded graph edge; donor Wilson LCB; no test',
            roles='current calibration for offers/acceptance; current BASE for continuation; current validation for diagnostics',
            final_test_opened=False,router='checkpoint router unchanged; no post-hoc historical refit',
            versions=dict(torch=torch.__version__,numpy=np.__version__,sklearn=sklearn.__version__),
            source_sha256={p.relative_to(root).as_posix():file_sha256(p) for p in source_files},
            implementation_scope='shared CNN/BN/GRU boundary; FC1 row closure only; divergent prefix rejects')
        write_json(out/'protocol_lock.json',lock)
        predictor=Predictor(seen,device,batch_size)
        def make_model(cid):
            model,router=_make_denice_client_model(ckpt,cid,device)
            state=ckpt['client_model_states'].get(cid,ckpt['client_model_states'].get(str(cid)))
            restored=model.state_dict()
            if set(restored)!=set(state) or any(not torch.equal(v.detach().cpu(),state[n]) for n,v in restored.items()):
                raise Rejected('INEXACT_MODEL_RESTORE',str(cid))
            if any(int(t)>task for t in router.activation_memory):raise Rejected('FUTURE_ROUTER_MEMORY')
            return model,router
        selected,ledger,offers=select_pair(ckpt,roles,task,classes,groups,make_model,predictor,protocol,manual,transport,eligibility_path=out/'selection_eligibility.json')
        write_json(out/'pair_selection.json',dict(selected=selected,offers=offers,ledger=ledger.to_dict()))
        i=selected['receiver'];j=selected['donor'];c=selected['class_id']
        print(f'APPLIANCE pair: receiver={i}, donor={j}, class={c}, task={task}',flush=True)
        receiver,route=make_model(i);donor,donor_route=make_model(j)
        source_hash=complete_hash(receiver,route)
        source_fingerprint=state_fingerprint(receiver,route)
        acceptance=current_pool(roles,i,'calibration',classes)
        # Deterministic current receiver-local subset; never import donor rows.
        acceptance_order=np.random.default_rng(protocol.seed+i).permutation(len(acceptance['y']))[:protocol.acceptance_max_rows]
        ax=acceptance['X'][acceptance_order];ay=acceptance['y'][acceptance_order]
        write_json(out/'acceptance_manifest.json',dict(origin_client=i,role='calibration',task=task,rows=acceptance['rows'][acceptance_order]))
        # All evaluator raw rows stay outside selection/acceptance/transaction.
        receiver_eval=current_pool(roles,i,'validation',classes);donor_eval=current_pool(roles,j,'validation',classes)
        diagnostic=dict(X=np.concatenate([receiver_eval['X'],donor_eval['X']]),y=np.concatenate([receiver_eval['y'],donor_eval['y']]))
        write_json(out/'diagnostic_manifest.json',dict(role='current validation; offline evaluator only; not receiver calibration',
            origins=[dict(client_id=i,rows=receiver_eval['rows']),dict(client_id=j,rows=donor_eval['rows'])],
            historical_old_task_damage='unmeasured: no historical raw reread',test_rows=0))
        old=predictor(copy.deepcopy(receiver),copy.deepcopy(route),diagnostic['X'])
        state_checks={'baseline_diagnostic':changed_state(source_fingerprint,receiver,route)}
        write_json(out/'receiver_state_checks.json',state_checks)
        if not state_checks['baseline_diagnostic']['unchanged']:raise RuntimeError('Baseline evaluation changed receiver state')
        reports={};survival={}
        for kind in ('head_only','dependency_complete'):
            local_ledger=copy.deepcopy(ledger);registry=Registry()
            transport.send(i,j,'REQUEST',dict(class_id=c,task=task,kind=kind,receiver_base_hash=source_hash))
            transport.send(j,i,'OFFER',dict(counts=selected['donor_counts'],quality_lcb=selected['quality_lcb'],kind=kind))
            report,compiled=compile_patch(receiver,donor,route,donor_route,c,task,seen,i,j,preprocessing_hash,protocol,kind)
            reports[kind]=report
            write_json(out/'closure_report.json',reports)
            print(f'APPLIANCE {kind}: '+('stage' if compiled else f'reject {report["reason"]}'),flush=True)
            if compiled is None:
                state_checks[kind]=changed_state(source_fingerprint,receiver,route)
                write_json(out/'receiver_state_checks.json',state_checks)
                outcomes[kind]=dict(status='rejected',reason=report['reason'],applied=False,rollback_verified=state_checks[kind]['unchanged'])
                transport.send(j,i,'REJECT',dict(kind=kind,reason=report['reason']))
                transaction_event('compiler_reject',dict(kind_name=kind,**outcomes[kind]))
                write_json(out/'outcomes.json',outcomes)
                if not state_checks[kind]['unchanged']:raise RuntimeError('Read-only compiler changed receiver state')
                continue
            branch=out/kind;branch.mkdir()
            packet=transport.send(j,i,'PATCH',compiled.packet)
            (branch/'patch.bin').write_bytes(packet);write_json(branch/'patch_manifest.json',compiled.metadata)
            candidate,candidate_route=stage(receiver,route,compiled,protocol,seen,preprocessing_hash)
            fidelity=logit_fidelity(candidate,donor,ax,task,c,protocol,batch_size)
            write_json(branch/'logit_fidelity.json',fidelity)
            if not fidelity['passed']:
                outcomes[kind]=dict(status='rejected',reason='LOGIT_FIDELITY_FAILED',applied=False,rollback_verified=True)
                transport.send(i,j,'NACK',dict(kind=kind,reason='LOGIT_FIDELITY_FAILED'))
                transaction_event('fidelity_reject',dict(kind_name=kind,**outcomes[kind]))
                write_json(out/'outcomes.json',outcomes);continue
            model,detector,txn=install(receiver,route,compiled,protocol,seen,preprocessing_hash,ax,ay,predictor,local_ledger,registry)
            transaction_event('transaction',dict(kind_name=kind,**txn));outcomes[kind]=txn
            write_json(out/'outcomes.json',outcomes)
            transport.send(i,j,'ACK' if txn['applied'] else 'NACK',dict(kind=kind,patch_id=txn['patch_id'],status=txn['status']))
            if not txn['applied']:continue
            result=predictor.records(model,detector,diagnostic['X'])
            measurements=compare(old,result['pred'],diagnostic['y'],c,ledger.observed,result['task'],task)
            measurements['old_supported_scope']='locally observed classes of current task, excluding imported class; not historical retention'
            write_json(branch/'before_after_metrics.json',measurements)
            nclasses=model.fc2.out_features
            before_cm=np.bincount(diagnostic['y']*nclasses+old,minlength=nclasses*nclasses).reshape(nclasses,nclasses)
            after_cm=np.bincount(diagnostic['y']*nclasses+result['pred'],minlength=nclasses*nclasses).reshape(nclasses,nclasses)
            np.savez_compressed(branch/'confusions.npz',before=before_cm,after=after_cm)
            torch.save(dict(model_state_dict=model.state_dict(),denice_state=snapshot_denice_state(model,detector),
                            ledger=local_ledger.to_dict(),registry=registry.entries),branch/'committed_receiver.pt')
            # Executable version/idempotence checks are part of the requested feasibility experiment.
            committed_hash=complete_hash(model,detector)
            _,_,replay=install(model,detector,compiled,protocol,seen,preprocessing_hash,ax,ay,predictor,local_ledger,registry)
            try:stage(model,detector,compiled,protocol,seen,preprocessing_hash);wrong_base=False
            except Rejected as exc:wrong_base=exc.reason=='STALE_BASE'
            checks=dict(idempotent=replay['status']=='already_committed' and complete_hash(model,detector)==committed_hash,
                        wrong_base_rejected=wrong_base)
            write_json(branch/'transaction_checks.json',checks)
            if not skip_survival:
                survival[kind]=run_survival(model,detector,registry,local_ledger,selected,groups,alphas,make_model,roles,classes,
                    config,device,predictor,diagnostic,old,protocol,batch_size,survival_rows,round_id)
                write_json(branch/'survival_report.json',survival[kind])
            else:survival[kind]=dict(skipped=True,full_training_round_verified=False)
            txn.update(metrics=measurements,fidelity=fidelity,transaction_checks=checks)
        write_json(out/'closure_report.json',reports);write_json(out/'outcomes.json',outcomes)
        write_json(out/'survival_report.json',survival)
        dep=outcomes.get('dependency_complete',{})
        proof=dep.get('applied',False) and dep.get('fidelity',{}).get('passed',False) and all(dep.get('transaction_checks',{}).values())
        write_json(out/'completion.json',dict(completed_execution=True,passed_feasibility=False,functional_pair_proof=bool(proof),
            stage='complete',pair=selected,reason='Initial compiler/continuation milestone; full round and historical retention gate remain unverified',
            diagnostic_fixture=method!='legacy',outcomes={k:v['status'] for k,v in outcomes.items()}))
    except Rejected as exc:
        if hasattr(exc,'audit'):write_json(out/'pair_selection.json',dict(selected=None,offers=exc.audit,reason=exc.reason))
        write_json(out/'completion.json',dict(completed_execution=True,passed_feasibility=False,stage='protocol_or_selection_rejected',
            reason=exc.reason,detail=exc.detail,pair=selected))
        print(f'APPLIANCE rejected: {exc.reason}',flush=True)
    except Exception as exc:
        write_json(out/'completion.json',dict(completed_execution=False,passed_feasibility=False,stage='execution_error',
            error_type=type(exc).__name__,detail=str(exc),pair=selected))
        raise
    finally:
        if transport is not None:write_json(out/'traffic_summary.json',dict(**transport.summary(),
            scope='pair discovery plus both independent ablation branches; not single-branch production cost'))
    return json.loads((out/'completion.json').read_text(encoding='utf-8'))
