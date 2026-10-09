"""Matched, receiver-only AMP diagnostics; no full federation or test access."""
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
import torch

from .config import Rejected
from .guarded_head import GuardedHeadRegistry, compile_guarded_head, install_guarded_head, put_head
from .guarded_head_experiment import combine, locked_pool
from .native_lifecycle_experiment import load_continuation
from .selector import lookup, current_pool
from .state import complete_hash, digest, rng_snapshot, restore_rng, tensor_hash, write_json


VARIANTS = (
    dict(name='native_amp', amp=True, init_scale=65536., fp32_head=False),
    dict(name='amp_scale1024', amp=True, init_scale=1024., fp32_head=False),
    dict(name='amp_fp32_head', amp=True, init_scale=65536., fp32_head=True),
    dict(name='full_fp32', amp=False, init_scale=1., fp32_head=False),
)


def tensor_stats(value):
    x=value.detach(); finite=torch.isfinite(x); v=x[finite].double()
    return dict(shape=list(x.shape), dtype=str(x.dtype), elements=x.numel(),
        nan=int(torch.isnan(x).sum()), posinf=int(torch.isposinf(x).sum()),
        neginf=int(torch.isneginf(x).sum()), finite=int(finite.sum()),
        finite_max_abs=float(v.abs().max()) if v.numel() else None,
        finite_l2=float(torch.linalg.vector_norm(v)) if v.numel() else None)


def gradient_stats(model):
    result={}
    for name,p in model.named_parameters():
        if p.grad is None:continue
        result[name]=tensor_stats(p.grad)
        if name in ('fc2.weight','fc2.bias'):
            result[name]['rows']={str(i):tensor_stats(row) for i,row in enumerate(p.grad)}
    return result


def parameter_delta(model,before):
    result={}
    for name,p in model.named_parameters():
        delta=p.detach().cpu()-before[name]
        if bool(torch.count_nonzero(delta)) or not bool(torch.isfinite(delta).all()):
            result[name]=dict(**tensor_stats(delta), changed=int(torch.count_nonzero(delta)))
    return result


class Trace:
    def __init__(self,out,model,fp32_head):
        self.out=Path(out); self.out.mkdir(parents=True,exist_ok=True)
        self.model=model; self.events=[]; self.phase=0; self.batch=0
        self.step_base=None; self.optimizer=None
        self.fp32_head=bool(fp32_head)

    def record(self,stage,**fields):
        self.events.append(dict(stage=stage, phase=self.phase, batch=self.batch, **fields))
        write_json(self.out/'gradient_trace.json',self.events)

    def attach_forward(self):
        # Observe the existing masked linear calls, without another forward or
        # RNG draw. Only the declared FP32-head variant changes arithmetic.
        original=self.model._apply_masked_linear
        def linear(x,layer,name):
            if name=='fc2' and self.fp32_head:
                with torch.autocast(device_type=x.device.type,enabled=False):
                    value=original(x.float(),layer,name)
            else:value=original(x,layer,name)
            self.record('masked_linear',layer=name,input=tensor_stats(x),output=tensor_stats(value))
            return value
        self.model._apply_masked_linear=linear

    def factory(self,parameters,lr,registry):
        self.optimizer=torch.optim.Adam(parameters,lr=lr)
        if registry is not None:registry.protect(self.model,self.optimizer)
        self.record('phase_after_pruning_and_protection',
            model_parameters=digest({n:p.detach() for n,p in self.model.named_parameters()}),
            backbone_parameters=digest({n:p.detach() for n,p in self.model.named_parameters() if not n.startswith('fc2.')}),
            fc2_ranks=self.model.unit_ranks['fc2'],
            freeze_masks=self.model.freeze_masks,
            fc2_effective_weight=tensor_stats(self.model.fc2.weight*self.model.weight_masks['fc2']),
            head_row24=tensor_stats(self.model.fc2.weight[24]),
            fc2_weight_mask_active_by_row=self.model.weight_masks['fc2'].sum(dim=1),
            fc2_bias_mask=self.model.bias_masks['fc2'])
        return self.optimizer

    def observe(self,stage,model,context):
        fields={k:(tensor_stats(v) if torch.is_tensor(v) else v) for k,v in context.items()
                if k not in ('inputs','labels')}
        if stage=='forward':
            fields['input_sha256']=tensor_hash(context['inputs'])
            fields['label_sha256']=tensor_hash(context['labels'])
            fields['class_histogram']={str(int(c)):int(n) for c,n in zip(*torch.unique(context['labels'],return_counts=True))}
            self.step_base={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
        if stage in ('scaled_backward','unscaled_backward','before_clipping','after_clipping'):
            fields['gradients']=gradient_stats(model)
        if stage in ('optimizer_step','after_step_protection') and self.step_base is not None:
            fields['parameter_delta_since_forward']=parameter_delta(model,self.step_base)
            fields['parameters_finite']=all(bool(torch.isfinite(p).all()) for p in model.parameters())
        self.record(stage,**fields)
        if stage=='after_step_protection':self.batch+=1


def run_amp_audit(checkpoint,data_dir,runtime_dir,out,device='cuda',phases=3,audit_batch_size=512):
    from eval_checkpoint import _make_denice_client_model
    from fed_learning.data.denice_clean_roles import CleanRoleData,file_sha256
    from fed_learning.factories.client_factory import create_client
    from fed_learning.training.checkpoint_state import restore_context_detector
    from fed_learning.training.decentralized_denice_il import _prepare_client_task,_split_local_validation
    from fed_learning.strategies.incremental import get_incremental_strategy
    from fed_learning.strategies.incremental.denice_novelty import NoveltyEstimator
    from fed_learning.strategies.incremental.denice_variants import variant_config
    from fed_learning.clients.denice_client import normalize_denice_imbalance_config
    from fed_learning.strategies.incremental.nice import update_freeze_masks
    from .runner import load_input

    runtime,out=Path(runtime_dir),Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new AMP audit output directory')
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'completion.json',dict(completed_execution=False,stage='prepare',final_test_opened=False))
    if not device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('Run the paired AMP diagnostic on Kaggle GPU; CPU cannot reproduce FP16 CUDA overflow')
    if not 1<=int(phases)<=3:raise ValueError('This locked local audit uses 1 to 3 phases')
    global_rng=rng_snapshot()
    try:
        source=runtime/'active_capability'
        protocol=json.loads((source/'protocol_lock.json').read_text(encoding='utf-8'))
        pair=json.loads((source/'pair_lock.json').read_text(encoding='utf-8'))
        cid,donor,c=map(int,(pair[k] for k in ('receiver','donor','class_id')))
        if (cid,donor,c)!=(3,6,24):raise Rejected('AMP_LOCKED_PAIR_CHANGED')
        if not json.loads((source/'completion.json').read_text(encoding='utf-8'))['installed']:
            raise Rejected('AMP_ACCEPTED_FROZEN_CAPABILITY_REQUIRED')
        packet=(source/'packets/shared16.bin').read_bytes()
        decision=json.loads((source/'decision_lock.json').read_text(encoding='utf-8'))
        packet_sha=hashlib.sha256(packet).hexdigest()
        if packet_sha!=decision['packet_sha256']['shared16']:raise Rejected('AMP_FROZEN_PACKET_CHANGED')
        state,continuation_sha=load_continuation(checkpoint,protocol['terminal_sha256'])
        config=copy.deepcopy(state['config'])
        if config.get('denice_cl_method')!='legacy' or config.get('denice_similarity_threshold')!=.8:
            raise Rejected('AMP_LEGACY_XI8_REQUIRED')
        roles=CleanRoleData(runtime/'roles',source_data_dir=data_dir)
        if file_sha256(roles.root/'role_manifest.json')!=protocol['role_manifest_sha256']:
            raise Rejected('AMP_ROLE_LOCK_CHANGED')
        terminal,_=load_input(checkpoint,4,19)
        model,router=_make_denice_client_model(terminal,cid,device)
        del terminal
        snapshots=joblib.load(source/'fitted_current_task_routers.joblib')
        frozen_router=lookup(snapshots,cid)
        if not frozen_router:raise Rejected('AMP_FROZEN_ROUTER_MISSING')
        restore_context_detector(router,frozen_router)
        source_hash=complete_hash(model,router)
        split=json.loads((source/'calibration_split_manifest.json').read_text(encoding='utf-8'))
        acceptance=combine([locked_pool(roles,i,list(range(24,30)),split[str(i)]['holdout']['rows']) for i in (cid,donor)])
        compiled=compile_guarded_head(model,router,packet,cid,list(range(30)),protocol['preprocessing_sha256'])
        installed,detector,registry,transaction=install_guarded_head(model,router,compiled,GuardedHeadRegistry(),
            acceptance,list(range(30)),device,audit_batch_size)
        write_json(out/'frozen_install_transaction.json',transaction)
        if not transaction['applied']:raise Rejected('AMP_FROZEN_INSTALL_REJECTED',str(transaction))
        if complete_hash(model,router)!=source_hash:raise RuntimeError('Original source mutated during install')
        del model,router
        pool=current_pool(roles,cid,'base',[30,31,32,33])
        X,y=torch.from_numpy(pool['X']),torch.from_numpy(pool['y'])
        train_x,train_y,held_x,held_y=_split_local_validation(X,y,float(config.get('denice_validation_fraction',0.)),
            int(config.get('seed',42))+50000+cid)
        if (len(X),len(train_y))!=(25,23) or torch.unique(train_y).tolist()!=[33]:
            raise Rejected('AMP_LOCKED_TASK5_DATA_CHANGED')
        if int(config['batch_size'])!=2048 or int(config.get('nice_phase_epochs',1))!=1:
            raise Rejected('AMP_ORIGINAL_LOCAL_SCHEDULE_CHANGED')
        seed=20261008
        np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        import random
        random.seed(seed)
        client=create_client(cid,train_x,train_y,{**config,'algorithm':'denice'})
        client.set_task_data(train_x,train_y,5,[30,31,32,33]);client.setup_for_gpu(installed,device)
        client.X_validation,client.y_validation=held_x,held_y
        trainer=get_incremental_strategy('denice',**{k:v for k,v in config.items() if k!='algorithm'})
        novelty_state=lookup(state['novelty_states'],cid,{})
        novelty=NoveltyEstimator(layer_weights=novelty_state.get('layer_weights',trainer.novelty_layer_weights))
        novelty.load_state(novelty_state)
        prep=_prepare_client_task(cid=cid,task_id=5,num_tasks=6,new_classes=[30,31,32,33],
            model=installed,client=client,trainer=trainer,config=config,device=torch.device(device),
            context_detector=detector,novelty_estimator=novelty,prev_ages=lookup(state['prev_ages'],cid),
            old_ref_bank=lookup(state['old_ref_banks'],cid,{}),old_ref_loss_baseline=lookup(state['old_ref_loss_baselines'],cid))
        installed.local_classifier=None
        registry.protect(installed)
        common_rng=rng_snapshot()
        control=copy.deepcopy(installed)
        put_head(control,c,registry.entries[c]['backup'])
        for name in ('imported_registry','appliance_guarded_head_entries'):
            if hasattr(control,name):delattr(control,name)
        update_freeze_masks(control)
        common_hash=complete_hash(installed,detector)
        # All tensor differences must be confined to the imported output row.
        for name,value in installed.state_dict().items():
            changed=value!=control.state_dict()[name];allowed=torch.zeros_like(changed)
            if name in ('fc2.weight','fc2.bias'):allowed[c]=True
            if bool((changed&~allowed).any()):raise RuntimeError(f'Unmatched preparation tensor: {name}')
        write_json(out/'protocol_lock.json',dict(version='appliance_amp_audit_v1',receiver=cid,donor=donor,class_id=c,
            source_terminal_sha256=protocol['terminal_sha256'],source_continuation_sha256=continuation_sha,
            packet_sha256=packet_sha,role_manifest_sha256=protocol['role_manifest_sha256'],
            phases=int(phases),variants=VARIANTS,base_rows=len(X),native_training_rows=len(train_y),
            base_held_rows=len(held_y),training_batch_size=config['batch_size'],rng_seed=seed,
            prepared_model_hash=common_hash,paired_preparation='prepare patched Task5 once, clone, restore original class24 row/masks/rank and remove registry for control',
            scope='receiver3 local NICE phases only; no aggregation, router refresh, final test or held-out validation loaded',
            calibration_use='reinstall the frozen accepted packet only; no prototype/threshold fitting',
            limitation='common post-prepare fixture, not an exact replay of federation round RNG; three local phases are not three full rounds',
            main_training_numeric_policy_changed=False))
        write_json(out/'runtime_manifest.json',json.loads((runtime/'runtime_manifest.json').read_text(encoding='utf-8')))
        del state,client,trainer,novelty,acceptance,pool,X,y
        trials=[]
        for variant in VARIANTS:
            for patched in (False,True):
                name=variant['name']+('_patched' if patched else '_control')
                print(f'AMP paired audit: {name}, receiver={cid}, train rows={len(train_y)}',flush=True)
                restore_rng(common_rng)
                candidate=copy.deepcopy(installed if patched else control)
                reg=GuardedHeadRegistry() if patched else None
                if reg is not None:reg.entries=candidate.appliance_guarded_head_entries;reg.sync(candidate)
                trial_client=create_client(cid,train_x,train_y,{**config,'algorithm':'denice'})
                trial_client.set_task_data(train_x,train_y,5,[30,31,32,33]);trial_client.setup_for_gpu(candidate,device)
                trial_trainer=get_incremental_strategy('denice',**{k:v for k,v in config.items() if k!='algorithm'})
                trial_trainer.set_task(5,prep['plan']['new_local_classes'])
                if variant['amp']:trial_client._nice_grad_scaler=torch.amp.GradScaler('cuda',init_scale=variant['init_scale'])
                trace=Trace(out/name,candidate,variant['fp32_head']);trace.attach_forward()
                base={n:p.detach().cpu().clone() for n,p in candidate.named_parameters()}
                summary=dict(name=name,patched=patched,variant=variant,phases=[],completed=False)
                start=time.perf_counter()
                try:
                    for phase in range(int(phases)):
                        trace.phase=phase
                        hooks=dict(optimizer_factory=lambda parameters,lr:trace.factory(parameters,lr,reg),
                            training_diagnostic=trace.observe)
                        if reg is not None:
                            hooks.update(gradient_filter=lambda:reg.gradient_filter(candidate),
                                after_optimizer_step_update=lambda current:reg.protect(current,trace.optimizer))
                        result=trial_client.train(trainer=trial_trainer,epochs=max(1,int(config.get('nice_phase_epochs',1))),
                            batch_size=int(config['batch_size']),lr=float(config.get('learning_rate',.001)),global_params=None,
                            is_last_task=True,phase_offset=phase,max_phases_override=1,amp_enabled=variant['amp'],
                            denice_elastic_strength=0.,continual_controls=variant_config(config),local_replay=None,
                            **normalize_denice_imbalance_config(config),**hooks)
                        phase_result={k:result[k] for k in ('loss','optimizer_steps','skipped_optimizer_steps','amp')}
                        phase_result['loss_finite']=math.isfinite(phase_result['loss'])
                        if not phase_result['loss_finite']:phase_result['loss']=None
                        summary['phases'].append(phase_result)
                        del result
                        print(f'  phase={phase}: steps={phase_result["optimizer_steps"]}, skipped={phase_result["skipped_optimizer_steps"]}, '
                            f'scale={phase_result["amp"]["final_scale"]}',flush=True)
                    summary['completed']=True
                except FloatingPointError as exc:
                    summary['error']=dict(type=type(exc).__name__,detail=str(exc))
                finally:
                    summary.update(seconds=time.perf_counter()-start,optimizer_steps=sum(p['optimizer_steps'] for p in summary['phases']),
                        skipped_optimizer_steps=sum(p['skipped_optimizer_steps'] for p in summary['phases']),
                        total_parameter_delta_including_pruning=parameter_delta(candidate,base),
                        head_protected_exact=reg.head_matches(candidate,c) if reg is not None else None,
                        parameters_finite=all(bool(torch.isfinite(p).all()) for p in candidate.parameters()))
                    write_json(out/name/'summary.json',summary)
                trials.append(summary)
                del candidate,trial_client,trial_trainer,trace,reg,base
                torch.cuda.empty_cache()
        # Verify batches and post-pruning backbone state across paired models.
        matched=[]
        for variant in VARIANTS:
            a=json.loads((out/(variant['name']+'_control')/'gradient_trace.json').read_text())
            b=json.loads((out/(variant['name']+'_patched')/'gradient_trace.json').read_text())
            a_forward=[(e['phase'],e['input_sha256'],e['label_sha256']) for e in a if e['stage']=='forward']
            b_forward=[(e['phase'],e['input_sha256'],e['label_sha256']) for e in b if e['stage']=='forward']
            a_phase=[e['backbone_parameters'] for e in a if e['stage']=='phase_after_pruning_and_protection']
            b_phase=[e['backbone_parameters'] for e in b if e['stage']=='phase_after_pruning_and_protection']
            common=min(len(a_forward),len(b_forward))
            matched.append(dict(variant=variant['name'],batches_identical=common>0 and a_forward[:common]==b_forward[:common],
                compared_batches=common,control_attempts=len(a_forward),patched_attempts=len(b_forward),
                unequal_attempt_counts_are_numerical_failure_not_pairing_error=True,
                first_phase_backbone_identical=bool(a_phase and b_phase and a_phase[0]==b_phase[0]),
                subsequent_backbone_comparison='not required after different numerical updates'))
        if complete_hash(installed,detector)!=common_hash:raise RuntimeError('Common prepared fixture mutated')
        write_json(out/'paired_integrity.json',matched)
        passed=all(e['batches_identical'] and e['first_phase_backbone_identical'] for e in matched)
        result=dict(completed_execution=True,paired_integrity_passed=passed,source_unchanged=True,
            final_test_opened=False,full_survival_certified=False,main_training_numeric_policy_changed=False,
            trials=[{k:t[k] for k in ('name','completed','optimizer_steps','skipped_optimizer_steps','parameters_finite','head_protected_exact')} for t in trials])
        write_json(out/'completion.json',result)
        if not passed:raise RuntimeError('Paired audit inputs/pre-pruning backbone mismatch; interpret trace before comparing variants')
        return result
    except Exception as exc:
        write_json(out/'completion.json',dict(completed_execution=False,error_type=type(exc).__name__,detail=str(exc),
            final_test_opened=False,full_survival_certified=False,main_training_numeric_policy_changed=False))
        raise
    finally:restore_rng(global_rng)
