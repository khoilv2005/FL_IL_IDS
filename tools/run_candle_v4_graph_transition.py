"""Lock graph-aware candidates before training; one bounded T2->T3 development smoke."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch

from pilot_candle_v4_standalone import stage_current,CleanRoleData,file_sha256,write
import numpy as np
import torch
import run_candle_v4_transition as previous
from fed_learning.data.denice_clean_roles import CleanRoleIncrementalDataLoader
from fed_learning.training.candle_v4_preflight import select_fixture,VERSION as PREFLIGHT_VERSION
from fed_learning.training.candle_v4_bank import VERSION as BANK_VERSION
from fed_learning.training.candle_v4_shadow import digest
import fed_learning.training.decentralized_denice_il as runner

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'audit_denice/appliance_transfer_results9/inputs/dataset'
ROLES=ROOT/'audit_denice/appliance_legacy_results11/inputs/roles'
SOURCE=ROOT/'audit_denice/appliance_legacy_results11/inputs/checkpoint_task_1_all_rounds.zip'


def registration(out):
    out.mkdir(parents=True,exist_ok=False)
    write(out/'selection_rule_lock.json',dict(version=PREFLIGHT_VERSION,task=2,seed=42,xi=.8,
        clock='native prepared T2, before first local/expert optimizer step',
        selection='receiver ID, class pair IDs, donor tuple IDs, filler IDs ascending after eligibility',
        BASE_min=32,FIT_positive_count_min=32,CAL_positive_count_min=32,owner_negative_CAL_count_min=32,
        owner_count=4,targets=2,optimizer='Unchanged balanced joint MLP39->64->1 Adam .001 clip5',
        threshold='donor FIT recall95 order statistic fixed before CAL; no CAL search',
        gate='recall>=.95; unique positive>=32; negatives>=32; observed FAR/per-negative-class FAR<=.001; break=0; atomic bank; net gain>0',
        smoke_rounds_per_task=8,tasks=[2,3],full_rule='Two ACTIVE still required; unchanged from previous protocol',
        budgets=dict(bank=128*1024**2,receiver_class=64*1024**2,total=2*1024**3),
        no_FIT_CAL_outcomes_for_selection=True,no_future_graphs=True,
        CAL_confirmation_independent=False,TEST_opened=False))


def preflight(out):
    registration(out)
    state=runner._load_denice_continuation_state(str(SOURCE))
    state['rng_state']['torch_cuda']=state['rng_state']['torch_cuda'][:torch.cuda.device_count()]
    for key in ('round_metrics','task_accuracies','task_forgetting'):state['history'][key]=[]
    for key in ('cluster_history','adapter_history','debug_history'):state[key]=[]
    config=deepcopy(state['config'])
    config.update(data_dir=str(DATA),denice_clean_roles_dir=str(ROLES),task_start=2,task_end=3,rounds_per_task=8,
        denice_evaluation_data_role='validation',denice_post_task_eval=False,denice_eval_final_round=False,
        denice_eval_local_validation=False,eval_every=999999,denice_save_round_artifacts=False,
        round_checkpoint_every=None,save_continuation_every_task=True,denice_archive_checkpoints=False,
        denice_max_train_samples_per_client=None,denice_max_clients=None,device='cuda',batch_size=2048,
        candle_v4_native_fixture=dict(version='candle_v4_graph_transition_v1',client_ids=list(range(100))),
        candle_v4_graph_preflight_only=True)
    state['config']=deepcopy(config);torch.save(state,out/'preflight_seed.pt');del state
    target=out/'native_preflight';target.mkdir()
    config.update(output_dir=str(target),resume_output_dir=str(target),resume_state_path=str(out/'preflight_seed.pt'))
    authority=CleanRoleData(ROLES,source_data_dir=DATA);access=[]
    stage=out/'evidence/task_2';stage.mkdir(parents=True)
    def current(self,owner,task):
        if task!=2:raise ValueError('Future/historical data forbidden in graph preflight')
        counts=authority.manifest['clients'][str(owner)]['role_class_counts']['base']
        if not sum(n for c,n in counts.items() if 12<=int(c)<18):
            return torch.empty(0,39),torch.empty(0,dtype=torch.long)
        path=stage/f'client_{owner}_base.npz'
        if not path.exists():
            created=stage_current(authority,owner,'base',list(range(12,18)),stage,4096,2,access)
            created.rename(path)
        with np.load(path,allow_pickle=False) as z:return torch.from_numpy(z['X'].copy()),torch.from_numpy(z['y'].copy())
    def forbidden(*a,**k):raise ValueError('Only current BASE authorized for graph preflight')
    with patch.object(CleanRoleIncrementalDataLoader,'get_client_data',current), \
         patch.object(CleanRoleData,'client_role',forbidden), \
         patch.object(CleanRoleIncrementalDataLoader,'get_test_data',forbidden):
        result=runner.run_decentralized_denice_il(config)
    write(out/'preflight_BASE_access.json',access);write(out/'preflight_execution.json',result)
    raw=json.loads((target/'prepared_graph.json').read_text())
    metadata={}
    for key,train_counts in raw['current_train_BASE_counts'].items():
        owner=int(key);row=authority.manifest['clients'][key]['role_class_counts']
        metadata[owner]={name:{int(c):n for c,n in values.items() if 12<=int(c)<18}
            for name,values in dict(owned_BASE=row['base'],current_train_BASE=train_counts,
                                   FIT_counts=row['fit'],CAL_counts=row['calibration']).items()}
    fixture=select_fixture(raw['graph'],metadata,list(range(12,18)))
    fixture['graph_sha256']=file_sha256(target/'prepared_graph.json')
    fixture['rule_lock_sha256']=file_sha256(out/'selection_rule_lock.json')
    fixture['preflight_cost']=raw['cost'];fixture['metadata']=metadata
    write(out/'candidate_preflight.json',fixture)
    print('CANDIDATE PREFLIGHT',fixture['status'],flush=True)
    if fixture['status']!='METADATA_FEASIBLE':return
    freeze(out,fixture)


def freeze(out,fixture):
    owners=fixture['owners'];prepared=out/'native_preflight/prepared_state.pt'
    state=runner._load_denice_continuation_state(str(prepared))
    state['client_ids']=owners
    for key in ('client_model_states','client_algorithm_states','novelty_states','prev_ages','old_ref_banks',
                'old_ref_loss_baselines','last_active_task','client_runtime_states','local_replay_states'):
        if key in state:state[key]={int(k):v for k,v in state[key].items() if int(k) in owners}
    boundary=state['round_state'];boundary['active_ids']=owners
    for key in ('canc_plans','ref_data','ref_labels','client_scalers'):
        boundary[key]={int(k):v for k,v in boundary[key].items() if int(k) in owners}
    graph=boundary['cluster_summary']; graph['alpha_debug']={o:graph['alpha_debug'][o] for o in owners}
    config=deepcopy(state['config']);config['candle_v4_graph_preflight_only']=False
    config['candle_v4_native_fixture']['client_ids']=owners
    state['config']=deepcopy(config);torch.save(state,out/'seed.pt')
    write(out/'config.json',config)
    protocol=dict(fixture={k:v for k,v in fixture.items() if k not in ('candidate_rows','metadata')},
        receiver=fixture['receiver'],experts=fixture['experts'],owners=owners,seed=42,
        task_start=2,task_end=3,rounds=8,prepared_current_task_optimizer_steps=0,
        rules=json.loads((out/'selection_rule_lock.json').read_text()),
        seed_sha256=file_sha256(out/'seed.pt'),source_sha256=file_sha256(prepared),
        roles_sha256=file_sha256(ROLES/'role_manifest.json'),selection_rule_sha256=file_sha256(out/'selection_rule_lock.json'),
        candidate_preflight_sha256=file_sha256(out/'candidate_preflight.json'),
        CAL_confirmation_independent=False,no_prior_CAL_outcomes_consulted=True,TEST_opened=False)
    write(out/'protocol_lock.json',protocol)
    # Only after fixture lock may selected owners' FIT be materialized.
    stage_selected(out,2)
    print('LOCKED FIXTURE',fixture['receiver'],fixture['experts'],owners,flush=True)


def stage_selected(out,task):
    fixture=json.loads((out/'protocol_lock.json').read_text());owners=fixture['owners']
    authority=CleanRoleData(ROLES,source_data_dir=DATA);directory=out/'evidence'/f'task_{task}'
    directory.mkdir(parents=True,exist_ok=True);log=[]
    for owner in owners:
        for role in ('base','fit'):
            path=directory/f'client_{owner}_{role}.npz'
            if path.exists():continue
            created=stage_current(authority,owner,role,list(range(task*6,task*6+6)),directory,
                                  4096 if role=='base' else None,task,log)
            created.rename(path)
    if log:write(directory/'selected_role_preparation.json',log)


def bank_spec(out):
    lock=json.loads((out/'protocol_lock.json').read_text())
    return dict(version=BANK_VERSION,policy='empirical_current_CAL',receiver=lock['receiver'],owners=lock['owners'],
        experts=[dict(**entry,birth_task=2,current_classes=list(range(12,18))) for entry in lock['experts']],
        input_sha256=lock['roles_sha256'],data_dir=str(DATA),roles_dir=str(ROLES),stage_dir=str(out/'evidence'),
        per_expert_budget=64*1024**2,bank_budget=128*1024**2)


def run(out,enabled,resume=False):
    lock=json.loads((out/'protocol_lock.json').read_text())
    if file_sha256(out/'seed.pt')!=lock['seed_sha256']:raise ValueError('Prepared seed changed')
    if file_sha256(out/'candidate_preflight.json')!=lock['candidate_preflight_sha256']:raise ValueError('Candidate selection changed')
    previous.OWNERS=lock['owners'];previous.spec=bank_spec;previous.stage=stage_selected
    code=['tools/run_candle_v4_graph_transition.py','candle_v4_runtime/fed_learning/training/candle_v4_preflight.py']
    hashes={p:file_sha256(ROOT/p) for p in code};path=out/'graph_implementation_lock.json'
    if path.exists() and json.loads(path.read_text())!=hashes:raise ValueError('Preflight/smoke implementation changed')
    if not path.exists():write(path,hashes)
    previous.run(out,enabled,resume_path=out/'bank/continuation_state_latest.pt' if resume else None)


if __name__=='__main__':
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8',errors='replace')
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--mode',choices=['preflight','bank','baseline','resume-bank'],required=True)
    args=parser.parse_args();out=args.out.resolve()
    if args.mode=='preflight':preflight(out)
    else:run(out,args.mode in ('bank','resume-bank'),resume=args.mode=='resume-bank')
