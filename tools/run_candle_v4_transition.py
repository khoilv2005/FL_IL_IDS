"""One registered 4-client T2->T3 smoke; one-shot CAL, no full TEST or retuning."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import sys
from unittest.mock import patch

from pilot_candle_v4_standalone import stage_current, CleanRoleData, file_sha256, write
import numpy as np
import torch
import fed_learning.training.decentralized_denice_il as runner
from fed_learning.training.candle_v4_bank import ExpertBank, VERSION, fuse, score_matrix
from fed_learning.training.candle_v4_shadow import digest
from fed_learning.data.denice_clean_roles import CleanRoleIncrementalDataLoader

ROOT=Path(__file__).resolve().parents[1]
OWNERS=[8,16,18,30]


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    data=ROOT/'audit_denice/appliance_transfer_results9/inputs/dataset'
    roles=ROOT/'audit_denice/appliance_legacy_results11/inputs/roles'
    source=ROOT/'audit_denice/appliance_legacy_results11/inputs/checkpoint_task_1_all_rounds.zip'
    state=runner._load_denice_continuation_state(str(source))
    state['rng_state']['torch_cuda']=state['rng_state']['torch_cuda'][:torch.cuda.device_count()]
    state['client_ids']=OWNERS
    for key in ('client_model_states','client_algorithm_states','novelty_states','prev_ages','old_ref_banks',
                'old_ref_loss_baselines','last_active_task','client_runtime_states','local_replay_states'):
        if key in state:state[key]={int(k):v for k,v in state[key].items() if int(k) in OWNERS}
    for key in ('round_metrics','task_accuracies','task_forgetting'):state['history'][key]=[]
    for key in ('cluster_history','adapter_history','debug_history'):state[key]=[]
    config=deepcopy(state['config'])
    config.update(task_start=2,task_end=3,rounds_per_task=8,data_dir=str(data),denice_clean_roles_dir=str(roles),
        denice_evaluation_data_role='validation',denice_post_task_eval=False,denice_eval_final_round=False,
        denice_eval_local_validation=False,eval_every=999999,denice_save_round_artifacts=False,
        round_checkpoint_every=None,save_continuation_every_task=True,denice_archive_checkpoints=False,
        denice_max_train_samples_per_client=None,denice_max_clients=None,device='cuda',batch_size=2048,
        candle_v4_native_fixture=dict(version='candle_v4_transition_smoke_v1',client_ids=OWNERS))
    state['config']=deepcopy(config)
    torch.save(state,out/'seed.pt'); write(out/'config.json',config)
    protocol=dict(version=VERSION,receiver=18,donor=16,targets=[13,17],owners=OWNERS,
        tasks=[2,3],rounds_per_task=8,expert_updates='44 current-task quota/round, max88 catch-up; no forced edge',
        native_data='current BASE deduped capped4096/class, original 10% native local-validation split',
        expert='39->64->1, Adam .001, balanced joint, clip5; no architecture/optimizer change',
        threshold_rule='Locked before CAL: donor FIT recall95 order statistic, no CAL threshold search',
        arbitration='largest (sigmoid-threshold)/(1-threshold), class-ID tie break, native fallback',
        gate=dict(positive_unique_min=32,negative_unique_min=32,recall_min=.95,
            observed_FAR_max=.001,per_observed_negative_class_FAR_max=.001,negative_break_max=0,
            atomic_bank_acceptance=True,net_gain_required=True),
        risk_policy='Empirical current CAL scope; old-positive receipts carry only with unchanged independent raw-input experts. New-task negatives can suspend; no strict historical safety claim.',
        CAL_independence='NOT independent confirmation: same role lineage used by earlier APPLIANCE studies; V4 pilots themselves did not open CAL',
        historical_evidence=['docs/APPLIANCE_GOAL_STATUS_20261009_VI.md: Task2 CAL study',
                             'docs/APPLIANCE_DIRECT_HEAD_DEVELOPMENT_20261010_VI.md: CAL-H already development'],
        communication_budget=dict(bank_total=128*1024**2,per_receiver_class=64*1024**2,
            modeled_native_plus_bank=2*1024**3,full_seed_incremental_projection_cap=64*1024**3),
        source_archive_sha256=file_sha256(source),roles_sha256=file_sha256(roles/'role_manifest.json'),
        CAL_opens='Once per current task, after finalized native task; never reopened in later tasks',
        CAL_content_exclusion='Dedup within each owner; remove current BASE/FIT content across all four owners before acceptance; counts gates apply after exclusion',
        TEST_opened=False,full_training='Conditional on actual two ACTIVE experts, transition/retention/risk/budget gates; otherwise NO-GO')
    write(out/'protocol_lock.json',protocol)
    # Lock precedes all newly materialized BASE/FIT and CAL outcomes.
    stage(out,2)


def stage(out, task):
    config=json.loads((out/'config.json').read_text()); path=out/'evidence'/f'task_{task}';path.mkdir(parents=True,exist_ok=True)
    roles=CleanRoleData(config['denice_clean_roles_dir'],source_data_dir=config['data_dir']); log=[]
    for owner in OWNERS:
        for role in ('base','fit'):
            target=path/f'client_{owner}_{role}.npz'
            if target.exists():continue
            created=stage_current(roles,owner,role,list(range(task*6,task*6+6)),path,
                                  4096 if role=='base' else None,task,log)
            created.rename(target)
    if log:write(path/'preparation_access.json',log)


def spec(out):
    config=json.loads((out/'config.json').read_text()); protocol=json.loads((out/'protocol_lock.json').read_text())
    return dict(version=VERSION,policy='empirical_current_CAL',receiver=18,owners=OWNERS,
        experts=[dict(target=c,donor=16,birth_task=2,current_classes=list(range(12,18))) for c in (13,17)],
        input_sha256=protocol['roles_sha256'],data_dir=config['data_dir'],roles_dir=config['denice_clean_roles_dir'],
        stage_dir=str(out/'evidence'),per_expert_budget=64*1024**2,bank_budget=128*1024**2)


def run(out, enabled):
    name='bank' if enabled else 'baseline'; target=out/name;target.mkdir(exist_ok=False)
    config=json.loads((out/'config.json').read_text())
    config.update(output_dir=str(target),resume_output_dir=str(target),resume_state_path=str(out/'seed.pt'))
    if enabled:config['candle_v4_expert_bank']=spec(out)
    source_paths=['tools/run_candle_v4_transition.py',
        'candle_v4_runtime/fed_learning/training/candle_v4_bank.py',
        'candle_v4_runtime/fed_learning/training/candle_v4_shadow.py',
        'candle_v4_runtime/fed_learning/training/decentralized_denice_il.py',
        'candle_v4_runtime/fed_learning/data/denice_clean_roles.py']
    source_lock={p:file_sha256(ROOT/p) for p in source_paths}
    lock_path=out/'execution_source_lock.json'
    if lock_path.exists():
        if json.loads(lock_path.read_text())!=source_lock:raise ValueError('Smoke source changed after execution lock')
    else:write(lock_path,source_lock)
    accesses=[]
    def current(self,owner,task):
        if owner not in OWNERS or task not in (2,3):raise ValueError('Off-scope training data')
        if not (out/'evidence'/f'task_{task}'/f'client_{owner}_base.npz').exists():stage(out,task)
        path=out/'evidence'/f'task_{task}'/f'client_{owner}_base.npz'
        with np.load(path,allow_pickle=False) as z:x=torch.from_numpy(z['X'].copy());y=torch.from_numpy(z['y'].copy())
        accesses.append(dict(owner=owner,task=task,role='current_BASE',sha256=file_sha256(path)))
        return x,y
    def forbidden(*args,**kwargs):raise ValueError('Unscoped raw role/evaluation access forbidden')
    aggregate=runner._aggregate_round
    def measured(**kwargs):
        ids=kwargs['client_ids'];models=kwargs['models'];capsules=kwargs['capsules']
        packets={o:dict(weights={k:v.detach().cpu() for k,v in models[o].state_dict().items()},
            masks=models[o].get_masks_state(),ages=models[o].get_neuron_ages_state(),
            adapters=models[o].get_adapter_registry_state()) for o in ids}
        packet_sizes={o:len(pickle.dumps(packets[o],protocol=5)) for o in ids}
        capsule_sizes={o:len(pickle.dumps(capsules[o],protocol=5)) for o in ids}
        result=aggregate(**kwargs)
        result['v4_native_wire']=dict(capsules=sum(capsule_sizes[o]*(len(ids)-1) for o in ids),
            parameters=sum(packet_sizes[j] for o,group in result['groups'].items() for j in group if j!=o))
        return result
    with patch.object(CleanRoleIncrementalDataLoader,'get_client_data',current), \
         patch.object(CleanRoleData,'client_role',forbidden),patch.object(runner,'_aggregate_round',measured), \
         patch.object(CleanRoleIncrementalDataLoader,'get_test_data',forbidden), \
         patch.object(CleanRoleIncrementalDataLoader,'get_full_test_data',forbidden):
        result=runner.run_decentralized_denice_il(config)
    write(target/'current_training_reads.json',accesses);write(target/'result.json',runner._json_safe(result))


def summarize(out):
    snapshots={}
    for task in (2,3):
        snapshots[task]={name:torch.load(out/name/f'continuation_state_task_{task}.pt',map_location='cpu',weights_only=False)
                         for name in ('bank','baseline')}
    native_keys=('client_model_states','client_algorithm_states','rng_state')
    checks={f'native_isolation_task_{t}':digest({k:s['bank'][k] for k in native_keys})==digest({k:s['baseline'][k] for k in native_keys})
            for t,s in snapshots.items()}
    a=snapshots[2]['bank']['candle_v4_shadow_state']; b=snapshots[3]['bank']['candle_v4_shadow_state']
    restored=ExpertBank(spec(out),b)
    checks['bank_checkpoint_restore_exact']=digest(b)==digest(restored.state_dict())
    checks['expert_retention_weights_exact']=all(digest(a['engines'][c]['branches']['joint_class_balanced']['model'])==
        digest(b['engines'][c]['branches']['joint_class_balanced']['model']) for c in (13,17))
    checks['no_old_expert_training_task3']=all(a['engines'][c]['steps']==b['engines'][c]['steps'] for c in (13,17))
    checks['two_experts_retained']=len(b['engines'])==2
    checks['current_only_training_reads']=all(r['role']=='current_BASE' and r['task'] in (2,3)
        for r in json.loads((out/'bank/current_training_reads.json').read_text()))
    # Exercise collision arbitration independently of CAL outcomes; never fake an install.
    pred,count=fuse(np.array([12,12,12]),np.array([[.99,.98],[.8,.9],[.2,.1]]),[.9,.9],[13,17])
    checks['collision_resolution']=pred.tolist()==[13,17,12] and count.tolist()==[2,1,0]
    g=torch.Generator().manual_seed(991); x=torch.randn(64,39,generator=g)
    p1=ExpertBank(spec(out),b); p2=ExpertBank(spec(out),b)
    q1=p1.infer(np.full(64,12),x);q2=p2.infer(np.full(64,12),x)
    checks['inference_restore_exact']=all(np.array_equal(x,y) for x,y in zip(q1,q2))
    checks['closed_CAL_task2_rejected']=False
    try:restored.role(16,2,'calibration')
    except ValueError:checks['closed_CAL_task2_rejected']=True
    native_bytes=sum(sum(r['v4_native_wire'].values()) for r in snapshots[3]['bank']['cluster_history'])
    protocol=json.loads((out/'protocol_lock.json').read_text())
    checks['bank_budget']=b['bytes']<=protocol['communication_budget']['bank_total']
    checks['per_class_budget']=all(n<=protocol['communication_budget']['per_receiver_class'] for n in b['per_class_bytes'].values())
    checks['total_modeled_budget']=native_bytes+b['bytes']<=protocol['communication_budget']['modeled_native_plus_bank']
    active_two=all(b['status'][c]=='ACTIVE' for c in (13,17))
    accepted=len(b['acceptances'])==2 and all(r['outcome']=='PASS' for r in b['acceptances'])
    decision='GO_FULL_SEED42' if all(checks.values()) and active_two and accepted else 'NO_GO_FULL'
    write(out/'report.json',dict(completed_execution=True,engineering_pass=all(checks.values()),checks=checks,
        active_gate_pass=accepted,two_ACTIVE=active_two,decision=decision,status=b['status'],
        gates=b['acceptances'],rounds=b['history'],expert_steps={c:e['steps'] for c,e in b['engines'].items()},
        bank_bytes=b['bytes'],per_receiver_class_bytes=b['per_class_bytes'],by_kind=b['by_kind'],
        native_modeled_bytes=native_bytes,combined_modeled_bytes=native_bytes+b['bytes'],
        CAL_confirmation_independent=False,TEST_opened=False,
        scope='One registered 4-client transition development smoke; no strict old-class or full-system quality guarantee'))
    print(decision,checks,flush=True)


if __name__=='__main__':
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8',errors='replace')
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=['prepare','bank','baseline','summarize'],required=True)
    a=p.parse_args();out=a.out.resolve()
    if a.mode=='prepare':prepare(out)
    elif a.mode=='summarize':summarize(out)
    else:run(out,a.mode=='bank')
