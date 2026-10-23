"""Bounded real native T2 training with isolated V4 draft experts; no CAL/TEST."""
import argparse
from collections import Counter
from copy import deepcopy
import json
import pickle
from pathlib import Path
import sys
from unittest.mock import patch

from pilot_candle_v4_standalone import stage_current, CleanRoleData, write, file_sha256
import numpy as np
import torch
from fed_learning.training.candle_v4_shadow import digest, ShadowService
from fed_learning.training.decentralized_denice_il import run_decentralized_denice_il, _load_denice_continuation_state
from fed_learning.data.denice_clean_roles import CleanRoleIncrementalDataLoader
import fed_learning.training.decentralized_denice_il as native_runner

ROOT = Path(__file__).resolve().parents[1]


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    data = ROOT/'audit_denice/appliance_transfer_results9/inputs/dataset'
    roles_path = ROOT/'audit_denice/appliance_legacy_results11/inputs/roles'
    roles = CleanRoleData(roles_path, source_data_dir=data)
    access=[]; stage=out/'current_data'; stage.mkdir()
    for owner in (16,30):
        for role in ('base','fit'):
            stage_current(roles,owner,role,list(range(12,18)),stage,4096 if role=='base' else None,2,access)
    archive=ROOT/'audit_denice/appliance_legacy_results11/inputs/checkpoint_task_1_all_rounds.zip'
    state=_load_denice_continuation_state(str(archive))
    state['rng_state']['torch_cuda']=state['rng_state'].get('torch_cuda', [])[:torch.cuda.device_count()]
    source_config=deepcopy(state['config'])
    keep={16,30}
    state['client_ids']=sorted(keep)
    for key in ('client_model_states','client_algorithm_states','novelty_states','prev_ages',
                'old_ref_banks','old_ref_loss_baselines','last_active_task','client_runtime_states','local_replay_states'):
        if key in state: state[key]={int(k):v for k,v in state[key].items() if int(k) in keep}
    state['history']['round_metrics']=[]
    state['history']['task_metrics']=[]
    for key in ('cluster_history','adapter_history','debug_history'): state[key]=[]
    if state.get('appliance_service_state'): raise ValueError('Source must be V1 without APPLIANCE')
    config=deepcopy(source_config)
    config.update(data_dir=str(data),denice_clean_roles_dir=str(roles_path),task_start=2,task_end=2,
        denice_evaluation_data_role='validation',denice_post_task_eval=False,denice_eval_final_round=False,
        denice_eval_final_task_only=True,denice_eval_local_validation=False,eval_every=999999,
        denice_save_round_artifacts=False,round_checkpoint_every=None,save_continuation_every_task=False,
        denice_archive_checkpoints=False,denice_max_train_samples_per_client=None,denice_max_clients=None,
        candle_v4_native_fixture=dict(version='candle_v4_bounded_native_v1',client_ids=[16,30]),
        device='cuda',batch_size=2048)
    state['config']=deepcopy(config)
    torch.save(state,out/'native_seed.pt')
    write(out/'config.json',config)
    write(out/'preparation_access.json',access)
    write(out/'protocol_lock.json',dict(version='candle_v4_bounded_native_v1',task=2,rounds=20,
        receiver=30,donor=16,target=13,xi=.8,native_source='V1 T1 terminal; pruned to two existing clients',
        fixture_limitation='Two-client engineering/development branch, not original 100-client graph or fresh training',
        BASE='current T2 deduped capped4096/class, then original native 10% local validation',
        expert_BASE='post-local-validation current client.X_train, cap4096/class',
        primary='joint_class_balanced',controls=['donor_only','joint_unbalanced'],
        native_budget='20 native rounds, original optimizer/epochs, batch2048',
        expert_budget='44 quota/round, max88 causal catch-up, 128 positive+128 negative/step',
        CAL_opened=False,TEST_opened=False,ACTIVE=False,
        roles_sha256=file_sha256(roles_path/'role_manifest.json'),
        source_archive_sha256=file_sha256(archive),seed_sha256=file_sha256(out/'native_seed.pt'),
        source_cuda_rng_mapping='Restore available original CUDA streams only; local single GPU engineering branch',
        input_authority='source metadata + common preprocessing benchmark; separate scaler artifact unavailable'))


def run(out, name, enabled, resume=None, stop=None):
    config=json.loads((out/'config.json').read_text())
    target=out/name; target.mkdir(parents=True,exist_ok=bool(resume))
    config.update(output_dir=str(target),resume_output_dir=str(target),
                  resume_state_path=str(resume or out/'native_seed.pt'),
                  candle_v4_stop_after_round=[2,stop if stop is not None else 19])
    if enabled:
        config['candle_v4_shadow']=dict(mode='DRAFT',task=2,receiver=30,donor=16,target=13,
            current_classes=list(range(12,18)),input_sha256=json.loads((out/'protocol_lock.json').read_text())['roles_sha256'],
            fit_paths={str(o):str(out/'current_data'/f'client_{o}_task_2_fit.npz') for o in (16,30)},
            wire_budget=256*1024**2)
    reads=[]
    def current(self,cid,task):
        if task!=2 or cid not in (16,30): raise ValueError('Off-task native data access')
        path=out/'current_data'/f'client_{cid}_task_2_base.npz'
        with np.load(path,allow_pickle=False) as z: x=torch.from_numpy(z['X'].copy()); y=torch.from_numpy(z['y'].copy())
        reads.append(dict(owner=cid,task=task,role='BASE',sha256=file_sha256(path)))
        return x,y
    def forbidden(*a,**k): raise ValueError('Unscoped raw role / CAL / TEST / evaluation loader access forbidden')
    original_aggregate=native_runner._aggregate_round
    def measured_aggregate(**kwargs):
        ids=kwargs['client_ids']; models=kwargs['models']; capsules=kwargs['capsules']
        packets={cid:dict(weights={k:v.detach().cpu().clone() for k,v in models[cid].state_dict().items()},
                    masks=models[cid].get_masks_state(),ages=models[cid].get_neuron_ages_state(),
                    adapters=models[cid].get_adapter_registry_state()) for cid in ids}
        result=original_aggregate(**kwargs)
        capsule_bytes=sum(len(pickle.dumps(dict(sender=j,receiver=i,capsule=capsules[j]),protocol=5))
                          for i in ids for j in ids if i!=j)
        parameter_bytes=sum(len(pickle.dumps(dict(sender=j,receiver=i,state=packets[j]),protocol=5))
                            for i,group in result['groups'].items() for j in group if j!=i)
        result['v4_native_wire']=dict(capsule_bytes=capsule_bytes,parameter_bytes=parameter_bytes,
            total=capsule_bytes+parameter_bytes,
            policy='Application serialization model: all-to-all capsules + dense neighbor weights/masks/ages/adapters; no transport headers')
        return result
    with patch.object(CleanRoleIncrementalDataLoader,'get_client_data',current), \
         patch.object(CleanRoleData,'client_role',forbidden), \
         patch.object(native_runner,'_aggregate_round',measured_aggregate), \
         patch.object(CleanRoleIncrementalDataLoader,'get_test_data',forbidden), \
         patch.object(CleanRoleIncrementalDataLoader,'get_full_test_data',forbidden):
        result=run_decentralized_denice_il(config)
    write(target/f'current_reads_{"resume" if resume else "start"}.json',reads)
    write(target/'result.json',result)


def summarize(out):
    states={name:torch.load(out/name/'continuation_state_latest.pt',map_location='cpu',weights_only=False)
            for name in ('shadow','baseline','resumed')}
    def native(state):
        return digest({k:state[k] for k in ('client_model_states','client_algorithm_states','rng_state')})
    same=native(states['shadow'])==native(states['baseline'])
    resumed=native(states['shadow'])==native(states['resumed'])
    a=states['shadow']['candle_v4_shadow_state']; b=states['resumed']['candle_v4_shadow_state']
    exact=digest(a)==digest(b)
    graphs={name:digest([{k:r.get(k) for k in ('groups','alpha_debug','labels')} for r in state['cluster_history']])
            for name,state in states.items()}
    checks=dict(native_shadow_on_off_exact=same,native_resume_exact=resumed,expert_resume_exact=exact,
                graph_on_off_exact=graphs['shadow']==graphs['baseline'],graph_resume_exact=graphs['shadow']==graphs['resumed'])
    checks['native_parameters_finite']=all(torch.isfinite(v).all().item()
        for state in states.values() for model in state['client_model_states'].values()
        for v in model.values() if isinstance(v,torch.Tensor))
    native_wire=[r['v4_native_wire'] for r in states['baseline']['cluster_history']]
    report=dict(completed=all(checks.values()),checks=checks,steps=a['steps'],pending=a['pending'],
        application_wire_bytes=a['bytes'],by_kind=a['by_kind'],messages=a['messages'],rounds=a['history'],
        native_application_serialization_bytes=sum(r['total'] for r in native_wire),
        native_wire_rounds=native_wire,
        modeled_total_bytes=a['bytes']+sum(r['total'] for r in native_wire),
        native_traffic='Dense application serialization model, not real transport. Three expert branches run together; current-base verification shared.',
        CAL_opened=False,TEST_opened=False,ACTIVE=False)
    write(out/'report.json',report)
    if not report['completed']: raise RuntimeError(f'Native shadow invariants failed: {checks}')
    print(json.dumps({k:v for k,v in report.items() if k!='rounds'},indent=2),flush=True)


def timing_control(out):
    """Train at task end with identical samples/optimizer work; no extra threshold search."""
    from types import SimpleNamespace
    from fed_learning.training.decentralized_denice_il import _split_local_validation
    terminal=torch.load(out/'shadow/continuation_state_latest.pt',map_location='cpu',weights_only=False)
    saved=terminal['candle_v4_shadow_state']; service=ShadowService(saved['spec'])
    data={}
    for owner in (16,30):
        with np.load(out/'current_data'/f'client_{owner}_task_2_base.npz',allow_pickle=False) as z:
            x=torch.from_numpy(z['X'].copy()); y=torch.from_numpy(z['y'].copy())
        x,y,_,_=_split_local_validation(x,y,.1,42+20000+owner)
        data[owner]=service.data(SimpleNamespace(X_train=x,y_train=y))
    for step in range(saved['steps']):
        for name in service.branches:
            service.train_step(name,data,step,2,19)
    exact={name:dict(parameters_exact=digest(model.state_dict())==digest(saved['branches'][name]['model']),
                    optimizer_exact=digest(optimizer.state_dict())==digest(saved['branches'][name]['optimizer']))
           for name,(model,optimizer) in service.branches.items()}
    write(out/'end_task_control.json',dict(completed=all(all(r.values()) for r in exact.values()),
        steps=saved['steps'],branches=exact,application_training_bytes=service.bytes,
        interpretation='Same fixed current BASE, sample order and work yield identical endpoint; round-wise timing alone provides no final accuracy gain here. Earlier results are DRAFT diagnostics, not deployed utility.'))


def no_peer_probe(out):
    saved=torch.load(out/'shadow/continuation_state_latest.pt',map_location='cpu',weights_only=False)['candle_v4_shadow_state']
    from fed_learning.training.candle_v4_shadow import make_expert
    model=make_expert(123); models={30:model}; detectors={30:None}
    service=ShadowService(saved['spec'])
    rng=torch.random.get_rng_state().clone()
    first=service.after_round(2,0,models,detectors,{},[30],{'alpha_debug':{}})
    restored=ShadowService(saved['spec'],service.state_dict())
    late=restored.after_round(2,1,models,detectors,{},[30],{'alpha_debug':{}})
    same_rng=torch.equal(rng,torch.random.get_rng_state())
    from types import SimpleNamespace
    late_service=ShadowService(saved['spec'],service.state_dict())
    client_views={}
    for owner in (16,30):
        with np.load(out/'current_data'/f'client_{owner}_task_2_base.npz',allow_pickle=False) as z:
            client_views[owner]=SimpleNamespace(X_train=torch.from_numpy(z['X'].copy()),y_train=torch.from_numpy(z['y'].copy()))
    # Controlled edge introduced at round1; no FIT/base-quality assertion in this probe.
    with patch.object(late_service,'evaluate',return_value={'controlled_probe':True}):
        catchup=late_service.after_round(2,1,models,detectors,client_views,[16,30],
            {'alpha_debug':{30:{'group_ids':[30,16],'alphas':[.5,.5]}}})
    expired=restored.after_round(3,0,models,detectors,{},[30],{'alpha_debug':{}})
    rejected=False
    try:restored.after_round(2,2,models,detectors,{},[30],{'alpha_debug':{}})
    except ValueError:rejected=True
    checks=dict(no_edge_zero_steps=first['total_steps']==late['total_steps']==0,
        no_edge_zero_bytes=late['total_wire_bytes']==0,quota_carries_to88=late['pending']==88,
        task_transition_expires=expired['status']=='EXPIRED' and restored.pending==0,
        late_peer_88_fresh_updates=catchup['updates']==88 and catchup['pending']==0,
        expired_generation_rejected=rejected,native_CPU_RNG_unchanged=same_rng)
    write(out/'no_peer_probe.json',dict(completed=all(checks.values()),checks=checks,
        scope='Controlled service conformance probe, not a naturally absent edge in the two-client native graph'))


@torch.no_grad()
def endpoint(out):
    import hashlib
    from eval_checkpoint import _make_denice_client_model
    from fed_learning.training.denice_eval import _denice_routed_logits_with_episodes
    from fed_learning.training.decentralized_denice_il import _split_local_validation
    state=torch.load(out/'shadow/continuation_state_latest.pt',map_location='cpu',weights_only=False)
    saved=state['candle_v4_shadow_state']; service=ShadowService(saved['spec'],saved)
    base,detector=_make_denice_client_model(state,30,'cuda')
    xs=[]; ys=[]; owners=[]; seen=set(); duplicates=0; counts={}
    for owner in (16,30):
        with np.load(out/'current_data'/f'client_{owner}_task_2_base.npz',allow_pickle=False) as z:
            bx=torch.from_numpy(z['X'].copy()); by=torch.from_numpy(z['y'].copy())
        _,by,_,_=_split_local_validation(bx,by,.1,42+20000+owner)
        counts[owner]=dict(Counter(by.tolist()))
        with np.load(saved['spec']['fit_paths'][str(owner)],allow_pickle=False) as z:
            x=torch.from_numpy(z['X'].copy()); y=torch.from_numpy(z['y'].copy())
        for row,label in zip(x,y):
            key=hashlib.sha256(row.numpy().tobytes()).hexdigest()
            if key in seen: duplicates+=1; continue
            seen.add(key); xs.append(row); ys.append(label); owners.append(owner)
    x=torch.stack(xs); y=torch.stack(ys); predicted=[]
    for start in range(0,len(x),512):
        logits,_=_denice_routed_logits_with_episodes(base,x[start:start+512].cuda(),detector,list(range(18)),
                                                     'cuda',inference_policy='pred_hard')
        predicted.append(logits.argmax(1).cpu())
    baseline=torch.cat(predicted); before=baseline==y; rows={}
    for name,(model,_) in service.branches.items():
        score=torch.sigmoid(model(x)).flatten(); rows[name]={}
        for threshold_name,receipt in saved['history'][-1]['FIT'][name].items():
            threshold=receipt['threshold']; pred=torch.where(score>=threshold,13,baseline); after=pred==y
            rows[name][threshold_name]=dict(correct=int(after.sum()),accuracy=float(after.float().mean()),
                target_correct=int(((pred==13)&(y==13)).sum()),
                false_activations=int(((score>=threshold)&(y!=13)).sum()),
                rescue=int((~before&after).sum()),break_count=int((before&~after).sum()),
                changed_wrong=int((~before&~after&(pred!=baseline)).sum()))
    write(out/'endpoint_unique.json',dict(rows=len(y),removed_cross_owner_duplicates=duplicates,
        baseline_correct=int(before.sum()),baseline_accuracy=float(before.float().mean()),
        post_native_local_validation_BASE_counts=counts,metrics=rows,
        scope='Pooled unique current T2 FIT only; no old-task or full-test accuracy claim'))


if __name__=='__main__':
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'): stream.reconfigure(encoding='utf-8',errors='replace')
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=['prepare','shadow','baseline','split','resume','summarize','timing-control','no-peer','endpoint'],required=True)
    a=p.parse_args(); out=a.out.resolve()
    if a.mode=='prepare':prepare(out)
    elif a.mode=='timing-control':timing_control(out)
    elif a.mode=='no-peer':no_peer_probe(out)
    elif a.mode=='endpoint':endpoint(out)
    elif a.mode=='summarize':summarize(out)
    elif a.mode=='shadow':run(out,'shadow',True)
    elif a.mode=='baseline':run(out,'baseline',False)
    elif a.mode=='split':run(out,'resumed',True,stop=8)
    elif a.mode=='resume':run(out,'resumed',True,resume=out/'resumed/continuation_state_latest.pt')
