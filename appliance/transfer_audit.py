"""Read-only transfer audit: functional probes do not bypass the exact installer.

Receiver calibration alone fits the bridge. Donor validation is evaluator-only.
This is development evidence, never an installed/accepted APPLIANCE patch.
"""
import copy
import hashlib
import json
import struct
import types
from pathlib import Path

import numpy as np
import torch

from .config import Protocol, Rejected
from .closure import compile_patch, effective_linear
from .evaluate import Predictor, compare
from .runner import load_input
from .selector import current_pool, recorded_graph, select_pair
from .state import boundary_hash, changed_state, digest, state_fingerprint, write_json


AUDIT_RULES = dict(
    version='appliance_transfer_audit_v1_1', seed=20261008, ranks=[4, 8, 16],
    calibration_max_rows=256, fit_fraction=.75, ridge_relative=1e-3,
    head_logit_relative_rmse_max=.05, bridge_relative_improvement_min=.10,
    max_current_supported_accuracy_drop=0., max_payload_bytes=32768,
    selection='receiver calibration holdout only; smallest qualifying FP32 rank',
    fp16='quantization ablation only; not selected as primary',
    no_final_test=True, no_install=True, no_full_training=True)


def relative_rmse(a, b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    if not a.size:return None
    return float(np.sqrt(np.mean((a-b)**2))/max(np.sqrt(np.mean(b**2)),1e-12))


def alignment(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    if a.shape!=b.shape:raise Rejected('FEATURE_SHAPE_MISMATCH')
    norms=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1);ok=norms>1e-12
    cos=np.sum(a[ok]*b[ok],axis=1)/norms[ok]
    ac=a-a.mean(axis=0);bc=b-b.mean(axis=0)
    cross=ac.T@bc;aa=ac.T@ac;bb=bc.T@bc
    denom=np.linalg.norm(aa,'fro')*np.linalg.norm(bb,'fro')
    return dict(rows=len(a),width=a.shape[1],row_cosine_mean=float(cos.mean()) if len(cos) else None,
        row_cosine_median=float(np.median(cos)) if len(cos) else None,zero_norm_rows=int((~ok).sum()),
        centered_linear_cka=float(np.sum(cross**2)/denom) if denom>1e-12 else None,
        degenerate_cka=bool(denom<=1e-12),feature_relative_rmse=relative_rmse(a,b),
        interpretation='CKA permits coordinate changes; high CKA does not prove direct head compatibility')


@torch.no_grad()
def features(model,x,task,batch_size,device):
    active=copy.deepcopy(model.active_adapters);modes=[(m,m.training) for m in model.modules()]
    result=[]
    try:
        model.eval();model.set_active_context(task)
        for start in range(0,len(x),batch_size):
            value=model.penultimate_features(torch.as_tensor(x[start:start+batch_size],dtype=torch.float32,device=device))
            if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite penultimate features')
            result.append(value.cpu().numpy())
    finally:
        model.active_adapters=active
        for module,training in modes:module.training=training
    return np.concatenate(result) if result else np.empty((0,model.fc1.out_features),dtype=np.float32)


def fit_bridge(hr,hd,rank):
    """Uncentered reduced-rank ridge; h_D ~= h_R + h_R V.T U.T."""
    x=np.asarray(hr,dtype=np.float64);delta=np.asarray(hd,dtype=np.float64)-x
    gram=x.T@x;ridge=AUDIT_RULES['ridge_relative']*max(float(np.trace(gram)/gram.shape[0]),1e-12)
    mapping=np.linalg.solve(gram+ridge*np.eye(gram.shape[0]),x.T@delta)
    left,singular,right=np.linalg.svd(mapping,full_matrices=False)
    actual=min(rank,len(singular));root=np.sqrt(singular[:actual])
    # mapping = V.T @ U.T in row-vector notation.
    v=(left[:,:actual]*root).T.astype(np.float32)
    u=(right[:actual,:].T*root).astype(np.float32)
    return dict(U=u,V=v,rank=actual,ridge=ridge)


def bridged(h,bridge):
    return h if bridge is None else h+(h@bridge['V'].T)@bridge['U'].T


def probe(receiver,router,weight,bias,c,task,bridge=None):
    """Isolated inference branch; only the imported-context class logit changes.

    Uses the existing per-sample router with an explicitly extended local mask.
    It may overwrite an occupied row FUNCTIONALLY on a clone to measure transfer;
    that is never considered a legal installation.
    """
    model=copy.deepcopy(receiver);detector=copy.deepcopy(router)
    if task not in detector.activation_memory or not detector.episode_classes.get(task):
        raise Rejected('ROUTE_UNAVAILABLE','Audit will not fabricate a task profile')
    detector.episode_classes[task]=sorted(set(map(int,detector.episode_classes[task]))|{c})
    model._appliance_probe_context=None
    activate=model.set_active_context;clear=model.clear_active_adapters
    def set_context(self,episode):
        activate(episode);self._appliance_probe_context=episode
    def clear_context(self):
        clear();self._appliance_probe_context=None
    w=torch.as_tensor(weight,dtype=torch.float32,device=next(model.parameters()).device)
    b=torch.as_tensor(bias,dtype=torch.float32,device=w.device)
    u=torch.as_tensor(bridge['U'],device=w.device) if bridge is not None else None
    v=torch.as_tensor(bridge['V'],device=w.device) if bridge is not None else None
    def forward(self,x):
        h=self.penultimate_features(x)
        out=self._apply_masked_linear(self.dropout(h),self.fc2,'fc2')
        if self._appliance_probe_context==task:
            aligned=h if u is None else h+(h@v.T)@u.T
            out=out.clone();out[:,c]=aligned@w+b
        return out
    model.set_active_context=types.MethodType(set_context,model)
    model.clear_active_adapters=types.MethodType(clear_context,model)
    model.forward=types.MethodType(forward,model)
    return model,detector


def packet(weight,bias,bridge,c,task,receiver_hash,donor_hash,dtype):
    """Separate audit codec; does NOT change appliance_fp32_v1 installer."""
    tensors=dict(head_weight=weight,head_bias=np.asarray(bias),head_weight_mask=np.ones_like(weight),
        head_bias_mask=np.asarray(1.),output_rank=np.asarray(2.))
    if bridge is not None:tensors.update(U=bridge['U'],V=bridge['V'])
    metadata=dict(kind='uninstalled_functional_probe',class_id=c,task=task,wire_dtype=dtype,
        receiver_penultimate_hash=receiver_hash,donor_penultimate_hash=donor_hash,
        route_contract='reuse receiver task profile, extend class mask; adapter follows predicted task',
        head_contract='replace only imported-context class logit; preserve all other class logits',
        rank=None if bridge is None else bridge['rank'],no_install=True)
    entries=[];chunks=[];offset=0
    for name,value in sorted(tensors.items()):
        arr=np.asarray(value,dtype='<f4' if dtype=='fp32' else '<f2',order='C')
        if not np.isfinite(arr).all():raise Rejected('NONFINITE_QUANTIZED_PAYLOAD',name)
        raw=arr.tobytes();chunks.append(raw)
        entries.append(dict(name=name,shape=list(arr.shape),dtype=arr.dtype.str,offset=offset,
                            bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()));offset+=len(raw)
    header=json.dumps(dict(schema='appliance_transfer_probe_v1',metadata=metadata,tensors=entries),
                      sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    value=struct.pack('<I',len(header))+header+b''.join(chunks)
    return value,dict(payload_bytes=len(value),tensor_bytes=offset,manifest_bytes=len(header)+4,
        payload_sha256=hashlib.sha256(value).hexdigest(),payload_kib=len(value)/1024,within_32kib=len(value)<=AUDIT_RULES['max_payload_bytes'],
        bridge_parameters=0 if bridge is None else int(bridge['U'].size+bridge['V'].size),
        head_mask_and_metadata_included=True,route_profile_reused=True,
        protection_metadata='feature boundary hash only; optimizer/pin registry integration not implemented')


def closure_inventory(donor,c,task):
    """Exact head/FC1 row counts; conservative prefix inventory, not minimal DAG.

    Legacy-output FC1 adapter can introduce additional FC1 input rows. Trace its
    masked nonzero U/V paths; linear-input adapter bypasses FC1 into the prefix.
    Prefix recurrent/CNN/adapters are retained in full as a safe upper inventory.
    No claim of a sparse/minimal recurrent closure or an installable payload.
    """
    from .codec import encode
    head,bias=effective_linear(donor,'fc2');w,b=effective_linear(donor,'fc1')
    direct=set(torch.nonzero(head[c]!=0).flatten().tolist());required=set(direct)
    adapter_entries=[];adapter_tensors={}
    for key,meta in donor.adapter_registry.items():
        if int(meta['context_id'])!=task:continue
        module=donor.adapters[key];u=module.U.weight.detach().cpu();v=module.V.weight.detach().cpu()
        im=donor.adapter_input_masks.get(key);om=donor.adapter_output_masks.get(key)
        if im is not None:v=v*im.detach().cpu()[None,:]
        if om is not None:u=u*om.detach().cpu()[:,None]
        if meta['layer_name']=='fc1' and meta.get('mode','legacy_output')=='legacy_output' and direct:
            latent=torch.any(u[sorted(direct)]!=0,dim=0)
            required |= set(torch.nonzero(torch.any(v[latent]!=0,dim=0)).flatten().tolist())
        adapter_entries.append(dict(key=key,mode=meta.get('mode','legacy_output'),parameters=u.numel()+v.numel(),
            selected_fc1_outputs_nonzero=int((u[sorted(direct)]!=0).sum()) if meta['layer_name']=='fc1' and direct else None))
        adapter_tensors[key+'_U']=u.numpy();adapter_tensors[key+'_V']=v.numpy()
    rows=sorted(required);active_weights=int((w[rows]!=0).sum()) if rows else 0
    prefix={name:value.detach().cpu().numpy() for name,value in donor.state_dict().items()
            if not name.startswith(('fc1.','fc2.','adapters.'))}
    prefix.update(adapter_tensors)
    for family in ('weight_masks','bias_masks','gru_connection_masks','adapter_input_masks','adapter_output_masks'):
        for name,value in getattr(donor,family,{}).items():
            if family in ('weight_masks','bias_masks') and name in ('fc1','fc2'):continue
            if family.startswith('adapter_') and name not in {v['key'] for v in adapter_entries}:continue
            prefix[family+'_'+name]=value.detach().cpu().numpy()
    prefix.update(fc1_weight=w[rows].numpy(),fc1_bias=b[rows].numpy(),head_weight=head[c].numpy(),
                  head_bias=np.asarray(bias[c]),fc1_rank=np.asarray(donor.unit_ranks['fc1'])[rows],
                  head_rank=np.asarray(donor.unit_ranks['fc2'][c]))
    meta=dict(kind='conservative_dependency_inventory_NOT_installable',task=task,class_id=c,fc1_rows=rows,
              active_context_adapters=adapter_entries,route_profile_policy='existing receiver profile assumed; router compatibility remains unproven')
    payload=encode(meta,prefix,2**31-1)
    parameter_count=sum(value.numel() for name,value in donor.named_parameters()
        if not name.startswith(('fc1.','fc2.','adapters.')))
    parameter_count+=len(rows)*(donor.fc1.in_features+1)+donor.fc2.in_features+1
    parameter_count+=sum(v['parameters'] for v in adapter_entries)
    return dict(direct_head_fc1_rows=sorted(direct),direct_fc1_count=len(direct),
        dependency_fc1_rows=rows,dependency_fc1_count=len(rows),fc1_width=donor.fc1.out_features,
        fc1_dependency_fraction=len(rows)/donor.fc1.out_features,fc1_effective_nonzero_weights=active_weights,
        fc1_dense_snapshot_bytes=len(rows)*(donor.fc1.in_features+1)*4,
        head_snapshot_bytes=(donor.fc2.in_features+1)*4,active_context_adapters=adapter_entries,
        conservative_parameter_count=parameter_count,conservative_payload_bytes=len(payload),
        conservative_payload_kib=len(payload)/1024,within_32kib=len(payload)<=AUDIT_RULES['max_payload_bytes'],
        minimal_dependency_closure_proven=False,coordinate_remapping_proven=False,
        scope='mask-aware head and FC1 adapter input rows; whole CNN/BN/recurrent prefix retained; effective weights with masks folded in; not a compiled transplant',
        exclusions=['training optimizer slots','receiver protection registry','new router profile','coordinate remapping'],
        no_install=True)



def fixed_context_prediction(model,router,h,seen,task,c=None,weight=None,bias=None,bridge=None):
    """Task-assisted diagnostic only, using one fixed imported context for all rows."""
    from fed_learning.training.denice_eval import _mask_logits_to_classes
    readout,readout_bias=effective_linear(model,'fc2')
    x=torch.as_tensor(h,dtype=torch.float32)
    logits=torch.nn.functional.linear(x,readout,readout_bias)
    allowed=sorted(set(map(int,router.episode_classes.get(task,[])))&set(seen))
    if c is not None:
        aligned=bridged(h,bridge)
        logits[:,c]=torch.as_tensor(aligned@weight+bias,dtype=torch.float32)
        allowed=sorted(set(allowed)|{c})
    return _mask_logits_to_classes(logits,allowed).argmax(1).numpy()


def fidelity(hr,hd,weight,bias,bridge,protocol):
    a=bridged(hr,bridge)@weight+bias;b=hd@weight+bias
    if not len(a):return dict(rows=0,relative_rmse=None,exact_pass=False)
    return dict(rows=len(a),relative_rmse=relative_rmse(a,b),mae=float(np.mean(np.abs(a-b))),
        max_absolute_error=float(np.max(np.abs(a-b))),
        exact_pass=bool(np.allclose(a,b,rtol=protocol.fidelity_rtol,atol=protocol.fidelity_atol)),
        diagnostic_approximate_gate=relative_rmse(a,b)<=AUDIT_RULES['head_logit_relative_rmse_max'],
        context_policy='fixed imported context on both models; not primary inference',
        zero_reference_rms=bool(np.sqrt(np.mean(b.astype(np.float64)**2))<=1e-12))


def run_transfer_audit(checkpoint,role_dir,data_dir,out,task=5,round_id=19,device='cpu',batch_size=512,
                       allow_cgofed_fixture=False,manual=None):
    from eval_checkpoint import _make_denice_client_model
    from fed_learning.data.denice_clean_roles import CleanRoleData,file_sha256
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new audit output directory')
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'completion.json',dict(completed_execution=False,no_install=True,passed_feasibility=False))
    selected=None
    try:
        if batch_size<1 or task not in range(6) or round_id<0:raise ValueError('Invalid audit scope')
        protocol=Protocol().validate();ckpt,hashes=load_input(checkpoint,task,round_id);config=ckpt['config']
        method=config.get('denice_cl_method','legacy')
        if method!='legacy' and not (method=='cgofed' and allow_cgofed_fixture):raise Rejected('LEGACY_BASELINE_REQUIRED',method)
        roles=CleanRoleData(role_dir,source_data_dir=data_dir);role_hash=file_sha256(roles.root/'role_manifest.json')
        if config.get('denice_data_roles_sha256')!=role_hash:raise Rejected('BACKBONE_ROLE_LOCK_MISMATCH')
        if tuple(config['input_shape'])!=tuple(roles.manifest['input_shape']):raise Rejected('PREPROCESSING_SHAPE_MISMATCH')
        if config.get('denice_max_train_samples_per_client') or config.get('denice_max_clients') not in (None,100):
            raise Rejected('TRUNCATED_BASE_TRAINING_NOT_SUPPORTED')
        metadata=json.loads((roles.source/'metadata.json').read_text(encoding='utf-8'))
        task_classes={int(t):list(map(int,v)) for t,v in metadata['task_structure']['task_classes'].items()}
        classes=task_classes[task];seen=sorted({c for t,v in task_classes.items() if t<=task for c in v})
        if sorted(map(int,ckpt['seen_classes']))!=seen:raise Rejected('SEEN_SCOPE_MISMATCH')
        groups,alphas=recorded_graph(ckpt,task,round_id);predict=Predictor(seen,device,batch_size)
        np.random.seed(AUDIT_RULES['seed']);torch.manual_seed(AUDIT_RULES['seed'])
        root=Path(__file__).resolve().parents[1]
        source_files=list((root/'appliance').glob('*.py'))+[root/name for name in (
            'eval_checkpoint.py','fed_learning/models/denice_model.py','fed_learning/models/nice_model.py',
            'fed_learning/training/denice_eval.py','fed_learning/training/checkpoint_state.py',
            'fed_learning/data/denice_clean_roles.py')]
        lock=dict(**AUDIT_RULES,**hashes,base_method=method,diagnostic_fixture=method!='legacy',
            task=task,round=round_id,xi=config.get('denice_similarity_threshold'),role_manifest_sha256=role_hash,
            checkpoint=str(checkpoint),manual_pair=manual,device=device,batch_size=batch_size,
            router='checkpoint router unchanged; per-sample pred_hard; clone mask extended only for imported task',
            fit_data='receiver current calibration only; feature-to-feature regression, no missing-class positives',
            evaluator_data='receiver + donor current validation; offline only, never rank/pair selection',
            primary_result=False,historical_retention='unmeasured',
            source_sha256={p.relative_to(root).as_posix():file_sha256(p) for p in source_files})
        import sklearn
        lock['versions']=dict(torch=torch.__version__,numpy=np.__version__,sklearn=sklearn.__version__)
        write_json(out/'protocol_lock.json',lock)
        def make_model(cid):
            model,router=_make_denice_client_model(ckpt,cid,device)
            state=ckpt['client_model_states'].get(cid,ckpt['client_model_states'].get(str(cid)))
            restored=model.state_dict()
            if set(restored)!=set(state) or any(not torch.equal(v.detach().cpu(),state[n]) for n,v in restored.items()):
                raise Rejected('INEXACT_MODEL_RESTORE',str(cid))
            if any(int(t)>task for t in router.activation_memory):raise Rejected('FUTURE_ROUTER_MEMORY')
            return model,router
        selected,ledger,offers=select_pair(ckpt,roles,task,classes,groups,make_model,predict,protocol,manual,
                                        eligibility_path=out/'selection_eligibility.json')
        write_json(out/'pair_selection.json',dict(selected=selected,offers=offers,ledger=ledger.to_dict()))
        i,j,c=selected['receiver'],selected['donor'],selected['class_id']
        print(f'Transfer audit: receiver={i}, donor={j}, class={c}, method={method}',flush=True)
        receiver,route=make_model(i);donor,donor_route=make_model(j)
        fingerprints={name:state_fingerprint(model,router) for name,model,router in (
            ('receiver',receiver,route),('donor',donor,donor_route))}
        def check_state(stage):
            value=dict(receiver=changed_state(fingerprints['receiver'],receiver,route),
                       donor=changed_state(fingerprints['donor'],donor,donor_route))
            write_json(out/f'state_checks_{stage}.json',value)
            if not all(v['unchanged'] for v in value.values()):raise RuntimeError('Audit mutated source state: '+stage)
        if any(getattr(m,'continual_head',None) is not None or getattr(m,'local_classifier',None) is not None for m in (receiver,donor)):
            raise Rejected('UNSUPPORTED_AUXILIARY_CLASSIFIER')
        structural=dict(same_fc1_shape=tuple(receiver.fc1.weight.shape)==tuple(donor.fc1.weight.shape),
            same_fc2_shape=tuple(receiver.fc2.weight.shape)==tuple(donor.fc2.weight.shape),
            prefix_equal=boundary_hash(receiver,False)==boundary_hash(donor,False),
            penultimate_equal=boundary_hash(receiver,True)==boundary_hash(donor,True),
            receiver_output_rank=int(receiver.unit_ranks['fc2'][c]),
            occupied_slot_blocks_install=bool(receiver.unit_ranks['fc2'][c]!=0),
            diagnostic_overwrite_on_clone_only=True)
        write_json(out/'structural_compatibility.json',structural)
        if not structural['same_fc1_shape'] or not structural['same_fc2_shape']:raise Rejected('ARCHITECTURE_MISMATCH')
        if donor.unit_ranks['fc2'][c]==0 or c not in donor_route.episode_classes.get(task,[]):raise Rejected('DONOR_CLASS_NOT_REPRESENTED')
        inventory=closure_inventory(donor,c,task);write_json(out/'closure_inventory.json',inventory)
        compiler={}
        for kind in ('head_only','dependency_complete'):
            compiler[kind],_=compile_patch(receiver,donor,route,donor_route,c,task,seen,i,j,
                digest(dict(role_manifest_sha256=role_hash,input_shape=config['input_shape'])),protocol,kind)
        write_json(out/'exact_compiler_report.json',compiler)
        check_state('compiler')
        pool=current_pool(roles,i,'calibration',classes)
        order=np.random.default_rng(AUDIT_RULES['seed']+i).permutation(len(pool['y']))[:AUDIT_RULES['calibration_max_rows']]
        if len(order)<protocol.acceptance_min_rows:raise Rejected('INSUFFICIENT_RECEIVER_ACCEPTANCE_DATA')
        split=max(1,min(len(order)-8,int(len(order)*AUDIT_RULES['fit_fraction'])))
        fit_idx=order[:split];hold_idx=order[split:]
        write_json(out/'calibration_manifest.json',dict(client_id=i,task=task,role='calibration',
            bridge_fit_rows=pool['rows'][fit_idx],holdout_rows=pool['rows'][hold_idx],
            counts_by_class={int(v):int((pool['y'][order]==v).sum()) for v in np.unique(pool['y'][order])},
            missing_class_positive_rows=int((pool['y'][order]==c).sum()),
            fit_holdout_disjoint=not bool(set(map(int,pool['rows'][fit_idx]))&set(map(int,pool['rows'][hold_idx])))))
        hr=features(receiver,pool['X'][order],task,batch_size,device)
        hd=features(donor,pool['X'][order],task,batch_size,device)
        write_json(out/'fc1_alignment.json',dict(all_receiver_calibration=alignment(hr,hd),
            fit=alignment(hr[:split],hd[:split]),holdout=alignment(hr[split:],hd[split:]),
            context='fixed imported context; same receiver inputs through frozen donor and receiver'))
        head,bias=effective_linear(donor,'fc2');weight=head[c].numpy();bias_value=float(bias[c])
        hold_x=pool['X'][hold_idx];hold_y=pool['y'][hold_idx]
        baseline=predict(copy.deepcopy(receiver),copy.deepcopy(route),hold_x)
        reval=current_pool(roles,i,'validation',classes);deval=current_pool(roles,j,'validation',classes)
        eval_x=np.concatenate([reval['X'],deval['X']]);eval_y=np.concatenate([reval['y'],deval['y']])
        write_json(out/'diagnostic_manifest.json',dict(role='current validation, offline evaluator only',
            origins=[dict(client_id=i,rows=reval['rows']),dict(client_id=j,rows=deval['rows'])],
            receiver_rows=len(reval['y']),donor_rows=len(deval['y']),final_test_rows=0,
            not_receiver_acceptance=True,not_used_to_select_bridge=True,
            old_damage_scope='current locally observed classes; historical retention unmeasured'))
        eval_baseline=predict(copy.deepcopy(receiver),copy.deepcopy(route),eval_x)
        donor_normal=predict(copy.deepcopy(donor),copy.deepcopy(donor_route),eval_x)
        ehr=features(receiver,eval_x,task,batch_size,device);ehd=features(donor,eval_x,task,batch_size,device)
        fixed_before=fixed_context_prediction(receiver,route,ehr,seen,task)
        summaries={};qualifying=[]
        def measure(name,bridge,dtype='fp32'):
            if dtype=='fp16':
                w=weight.astype(np.float16).astype(np.float32);b=float(np.float16(bias_value))
                used=None if bridge is None else {**bridge,**{k:bridge[k].astype(np.float16).astype(np.float32) for k in ('U','V')}}
            else:w,b,used=weight,bias_value,bridge
            folder=out/name;folder.mkdir()
            wire,size=packet(w,b,used,c,task,boundary_hash(receiver,True),boundary_hash(donor,True),dtype)
            (folder/'diagnostic_patch.bin').write_bytes(wire)
            candidate,detector=probe(receiver,route,w,b,c,task,used)
            held=predict.records(candidate,detector,hold_x)
            held_metrics=compare(baseline,held['pred'],hold_y,c,ledger.observed,held['task'],task)
            fid=fidelity(hr[split:],hd[split:],w,b,used,protocol)
            observed=held_metrics['old_supported_accuracy']
            damage_ok=observed['rows']>0 and observed['after']>=observed['before']
            scores=dict(name=name,dtype=dtype,rank=None if used is None else used['rank'],
                size=size,calibration_holdout=dict(metrics=held_metrics,logit_fidelity=fid,
                    feature_alignment=alignment(bridged(hr[split:],used),hd[split:])),
                receiver_current_supported_damage_gate=bool(damage_ok),missing_positive_gain_verified=False,
                no_install=True)
            result=predict.records(candidate,detector,eval_x)
            metrics=compare(eval_baseline,result['pred'],eval_y,c,ledger.observed,result['task'],task)
            origins=np.concatenate([np.full(len(reval['y']),i),np.full(len(deval['y']),j)])
            row_ids=np.concatenate([reval['rows'],deval['rows']])
            np.savez_compressed(folder/'diagnostic_predictions.npz',origin_client=origins,row_id=row_ids,
                y_true=eval_y,baseline=eval_baseline,probe=result['pred'],donor=donor_normal,routed_task=result['task'])
            def origin_metrics(start,end):
                return compare(eval_baseline[start:end],result['pred'][start:end],eval_y[start:end],c,
                               ledger.observed,result['task'][start:end],task)
            scores['offline_validation']=dict(metrics=metrics,receiver_only=origin_metrics(0,len(reval['y'])),
                donor_only=origin_metrics(len(reval['y']),len(eval_y)),
                logit_fidelity=fidelity(ehr,ehd,w,b,used,protocol),not_used_for_selection=True)
            fixed_after=fixed_context_prediction(receiver,route,ehr,seen,task,c,w,b,used)
            np.savez_compressed(folder/'fixed_context_predictions.npz',origin_client=origins,row_id=row_ids,
                y_true=eval_y,baseline=fixed_before,probe=fixed_after)
            fixed_metrics=compare(fixed_before,fixed_after,eval_y,c,ledger.observed,np.full(len(eval_y),task),task)
            scores['fixed_import_context_diagnostic']=dict(metrics=fixed_metrics,
                task_identifier_assisted=True,not_primary_inference=True,not_used_for_selection=True,
                description='all current validation rows forced through imported context; separates function transfer from router reachability')
            scores['offline_validation']['missing_class_router_reachability_ceiling']=(
                float(((eval_y==c)&(result['task']==task)).sum()/max(1,(eval_y==c).sum())))
            if dtype=='fp16':
                float_logits=bridged(hr[split:],bridge)@weight+bias_value
                quant_logits=bridged(hr[split:],used)@w+b
                scores['quantization_relative_rmse']=relative_rmse(quant_logits,float_logits)
            write_json(folder/'metrics.json',scores);summaries[name]=scores
            print(f'Audit {name}: {size["payload_kib"]:.2f} KiB, holdout logit RRMSE={fid["relative_rmse"]:.6f}',flush=True)
            del candidate,detector
            return scores
        first=measure('head_only',None)
        measure('head_only_fp16',None,'fp16')
        head_ok=(first['calibration_holdout']['logit_fidelity']['diagnostic_approximate_gate']
                 and not first['calibration_holdout']['logit_fidelity']['zero_reference_rms']
                 and first['receiver_current_supported_damage_gate'] and first['size']['within_32kib'])
        bridge_enabled=not head_ok
        if bridge_enabled:
            for rank in AUDIT_RULES['ranks']:
                bridge=fit_bridge(hr[:split],hd[:split],rank)
                result=measure(f'bridge_r{rank}',bridge)
                measure(f'bridge_r{rank}_fp16',bridge,'fp16')
                before=first['calibration_holdout']['logit_fidelity']['relative_rmse']
                after=result['calibration_holdout']['logit_fidelity']['relative_rmse']
                improved=after<=(1-AUDIT_RULES['bridge_relative_improvement_min'])*before
                result['selection_gate']=dict(payload_within_budget=result['size']['within_32kib'],
                    relative_improvement=bool(improved),approximate_fidelity=(result['calibration_holdout']['logit_fidelity']['diagnostic_approximate_gate']
                        and not result['calibration_holdout']['logit_fidelity']['zero_reference_rms']),
                    current_supported_damage=result['receiver_current_supported_damage_gate'])
                write_json(out/f'bridge_r{rank}'/'metrics.json',result)
                if all(result['selection_gate'].values()):qualifying.append(rank)
        choice='head_only' if head_ok else (f'bridge_r{min(qualifying)}' if qualifying else 'no_qualified_portable_probe')
        # Even a selected probe lacks a receiver-positive gate and a survival proof.
        decision=dict(calibration_selected_probe=choice,bridge_attempted=bridge_enabled,
            rule=AUDIT_RULES['selection'],selection_used_donor_validation=False,
            compatible_for_install=False,passed_feasibility=False,no_install=True,
            structural=structural,positive_quality_at_receiver='unknown: missing local positives',
            historical_damage='unmeasured',survives_next_round='unmeasured',
            closure_minimality='not proven; conservative inventory only',
            next_step='legacy replication and functional/survival gate; do not deploy based on this fixture')
        write_json(out/'decision.json',decision);write_json(out/'summary.json',summaries)
        check_state('final')
        write_json(out/'completion.json',dict(completed_execution=True,passed_feasibility=False,no_install=True,
            diagnostic_fixture=method!='legacy',base_method=method,stage='complete',pair=selected,
            selected_probe=choice,bridge_attempted=bridge_enabled,final_test_opened=False,
            reason='Read-only development transfer evidence; no patch committed or survival/legacy proof'))
    except Rejected as exc:
        if hasattr(exc,'audit'):write_json(out/'pair_selection.json',dict(selected=None,offers=exc.audit,reason=exc.reason))
        write_json(out/'completion.json',dict(completed_execution=True,passed_feasibility=False,no_install=True,
            stage='protocol_or_selection_rejected',reason=exc.reason,detail=exc.detail,pair=selected))
        print(f'Transfer audit rejected: {exc.reason}: {exc.detail}',flush=True)
    except Exception as exc:
        write_json(out/'completion.json',dict(completed_execution=False,passed_feasibility=False,no_install=True,
            stage='execution_error',error_type=type(exc).__name__,detail=str(exc),pair=selected))
        raise
    return json.loads((out/'completion.json').read_text(encoding='utf-8'))
