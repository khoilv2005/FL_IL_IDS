"""Paired small multi-client, three-task diagnostics around the production runner.

Same cached real-data panels for peer/self-only and independent training seeds.
No oracle labels enter training, capsule construction or clustering.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
from zipfile import ZipFile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from sklearn.metrics import f1_score
from tools.diagnose_denice_real_client import sample_npz
from fed_learning.clients.denice_client import DeNICEClient
from fed_learning.training import decentralized_denice_il as runner
from fed_learning.training.denice_eval import _denice_routed_logits_with_episodes
from fed_learning.strategies.decentralized.denice_aggregation import build_compatible_mask
from eval_checkpoint import _make_denice_client_model
from fed_learning.training.checkpoint_state import snapshot_denice_state

POLICIES = ['backbone_nomask','oracle_hard_no_adapter','oracle_hard','pred_hard']


def prepare_panel(archive, output, ids):
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'client_ids': ids, 'train_indices': {}, 'sampling_seed': 2026,
                'train_per_class': 64, 'test_per_class': 16, 'class_count': 18}
    with ZipFile(archive) as z:
        for cid in ids:
            x,y,indices = sample_npz(z.read(f'100-clients/client_{cid}_train.npz'),'train',64,2026+cid,18)
            np.savez_compressed(output/f'client_{cid}.npz', x=x.numpy(), y=y.numpy())
            manifest['train_indices'][str(cid)] = indices
        x,y,indices = sample_npz(z.read('100-clients/global_test_data.npz'),'test',16,2027,18)
        np.savez_compressed(output/'test.npz', x=x.numpy(), y=y.numpy())
        manifest['test_indices'] = indices
    manifest['panel_hashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.glob('*.npz'))}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


def run(panel, output, seed, mode, rounds, counterfactual=False, resume_state_path=None, task_end=2, extra_config=None):
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((panel/'manifest.json').read_text())
    data = {}
    for cid in manifest['client_ids']:
        with np.load(panel/f'client_{cid}.npz',allow_pickle=False) as f:
            data[cid] = (torch.from_numpy(f['x']),torch.from_numpy(f['y']))
    with np.load(panel/'test.npz',allow_pickle=False) as f:
        tx,ty = torch.from_numpy(f['x']),torch.from_numpy(f['y'])
    class Loader:
        input_shape = (39,1)
        task_classes = {t:list(range(6*t,min(6*t+6,34))) for t in range(6)}
        def __init__(self, data_dir): pass
        def get_num_tasks(self): return 6
        def get_task_classes(self,t): return self.task_classes[t]
        def get_all_client_ids(self): return sorted(data)
        def get_client_data(self,cid,t):
            x,y=data[cid]; m=(y//6==t)
            return x[m],y[m]
        def get_test_data(self,t,cumulative=True):
            m=ty<6*(t+1)
            return tx[m],ty[m]
    cfg = deepcopy(json.loads(Path('audit_denice/artifact_evidence.json').read_text())['full_d2']['42']['config'])
    cfg.update(data_dir=str(panel),output_dir=str(output/'training'),resume_output_dir=str(output/'training'),
               random_seed=seed,seed=seed,num_clients=len(data),denice_max_clients=len(data),
               task_start=0,task_end=task_end,rounds_per_task=rounds,nice_phase_epochs=1,batch_size=32,
               eval_every=9999,denice_post_task_eval=False,denice_aggregation_mode=mode,
               denice_collaboration_guard_mode='off',denice_shared_context_eval=False,
               denice_min_free_capacity_ratio=0.1,save_resume_after_task=True,resume_state_path=resume_state_path,
               round_checkpoint_every=rounds)
    cfg.update(extra_config or {})
    contexts, records, group_events = {}, [], []
    task = -1
    def evaluate(cid, model, detector):
        # Copies and RNG preservation keep observation outside training behavior.
        rng=runner._snapshot_rng_state()
        try:
            model,detector=deepcopy(model).eval(),deepcopy(detector)
            mask=ty<6*(task+1); x,y=tx[mask],ty[mask]
            support=np.unique(data[cid][1].numpy())
            metrics, raw = {}, {}
            for policy in POLICIES:
                logits,routes=_denice_routed_logits_with_episodes(model,x,detector,list(range(6*(task+1))),
                            'cpu',inference_policy=policy,oracle_episodes=y.numpy()//6)
                pred=logits.argmax(1).numpy(); metrics[policy]={};raw[policy]=logits.cpu()
                for old in range(task+1):
                    for subset in ['all','supported']:
                        allowed=[c for c in range(6*old,6*old+6) if subset=='all' or c in support]
                        m=np.isin(y.numpy(),allowed)
                        metrics[policy][f'{old}_{subset}']={
                            'n':int(m.sum()),'accuracy':float(np.mean(pred[m]==y.numpy()[m])),
                            'macro_f1_task_labels':float(f1_score(y.numpy()[m],pred[m],labels=allowed,average='macro',zero_division=0)),
                            'route_accuracy':float(np.mean(np.asarray(routes)[m]==old)) if routes is not None else None}
            with torch.no_grad():
                features=model.get_context_activations_per_sample(tx[ty<6])['gru'].cpu()
            return metrics,features,raw
        finally:
            runner._restore_rng_state(rng)
    def measure(cid,stage,round_id=None):
        ctx=contexts[cid]
        metrics,features,logits=evaluate(cid,ctx['model'],ctx['context_detector'])
        row={'task':task,'client':cid,'stage':stage,'round':round_id,'metrics':metrics,
             'capacity':runner._capacity_debug(ctx['model'])}
        records.append(row)
        return row,features,logits
    def snapshot(model):
        state=runner._cpu_state_dict(model)
        return state,build_compatible_mask(state,model.unit_ranks),model.unit_ranks['gru'].copy()
    def compare(before, model, old_features, new_features):
        state,masks,ages=before;after=runner._cpu_state_dict(model)
        changes=[]
        for key,value in state.items():
            if key not in after: continue
            frozen=masks[key]==0
            if frozen.any():changes.append(float((after[key].double()-value.double())[frozen].abs().max()))
        mature=ages>=2
        return {'protected_row_storage_max_change':max(changes,default=0),
                'mature_gru_feature_max_change':float((new_features-old_features)[:,mature].abs().max()) if mature.any() else None}
    original_prepare=runner._prepare_client_task
    def prepare(**kw):
        nonlocal task
        task=kw['task_id'];result=original_prepare(**kw);contexts[kw['cid']]=kw
        measure(kw['cid'],'prepare')
        return result
    original_train=DeNICEClient.train
    def train(client,*a,**kw):
        cid=client.client_id; rid=kw['phase_offset']
        _,features,_=measure(cid,'before_local',rid);before=snapshot(client.model)
        result=original_train(client,*a,**kw)
        row,after,_=measure(cid,'after_local',rid);row['transition']=compare(before,client.model,features,after)
        return result
    original_agg=runner._aggregate_round
    def aggregate(**kw):
        previous={cid:(snapshot(kw['models'][cid]),measure(cid,'before_aggregation')) for cid in kw['client_ids']}
        result=original_agg(**kw)
        group_events.append(runner._json_safe({'task':task,**result}))
        for cid in kw['client_ids']:
            row,features,_=measure(cid,'after_aggregation')
            row['transition']=compare(previous[cid][0],kw['models'][cid],previous[cid][1][1],features)
            if counterfactual and task>0:
                rng=runner._snapshot_rng_state()
                try:
                    row['counterfactuals']={}
                    for intervention in ['refresh_router','restore_preagg_gru','restore_preagg_old_adapters']:
                        model,detector=deepcopy(kw['models'][cid]),deepcopy(contexts[cid]['context_detector'])
                        old_state=previous[cid][0][0]
                        if intervention=='refresh_router':
                            model.eval()
                            detector.refresh_activation_memory(model,task_id=task)
                        elif intervention=='restore_preagg_gru':
                            model.gru.load_state_dict({k[4:]:v for k,v in old_state.items() if k.startswith('gru.')})
                        else:
                            for key,meta in model.adapter_registry.items():
                                if meta['context_id']<task:
                                    prefix=f'adapters.{key}.'
                                    values={k[len(prefix):]:v for k,v in old_state.items() if k.startswith(prefix)}
                                    if values:model.adapters[key].load_state_dict(values)
                        metrics,_,_=evaluate(cid,model,detector)
                        row['counterfactuals'][intervention]=metrics
                finally:runner._restore_rng_state(rng)
        print(f'GROUP seed={seed} mode={mode} task={task}: peers={result.get("peer_aggregated_client_count")} policy={result.get("effective_policy")}',flush=True)
        return result
    original_age=runner.increase_unit_ranks
    def age(model):
        cid=next(c for c,k in contexts.items() if k['model'] is model)
        measure(cid,'before_aging');result=original_age(model)
        row,_,logits=measure(cid,'after_aging')
        rng=runner._snapshot_rng_state()
        try:
            ckpt={'config':cfg,'client_model_states':{cid:runner._cpu_state_dict(model)},
                  'client_algorithm_states':{cid:snapshot_denice_state(model,contexts[cid]['context_detector'])}}
            fresh,router=_make_denice_client_model(ckpt,cid,'cpu')
            _,_,again=evaluate(cid,fresh,router)
            row['reload_max_logit_error']=max(float((logits[p]-again[p]).abs().max()) for p in POLICIES)
            assert row['reload_max_logit_error']==0
        finally:runner._restore_rng_state(rng)
        return result
    completed=False
    try:
        with patch.object(runner,'IncrementalDataLoader',Loader),patch.object(runner,'_prepare_client_task',prepare), \
             patch.object(DeNICEClient,'train',train),patch.object(runner,'_aggregate_round',aggregate), \
             patch.object(runner,'increase_unit_ranks',age):
            runner.run_decentralized_denice_il(cfg)
        completed=True
    finally:
        report={'completed':completed,'seed':seed,'mode':mode,'counterfactuals_enabled':counterfactual,'config':cfg,'panel_manifest':manifest,
                'records':records,'groups':group_events,
                'scope':'Five selected clients active on all three tasks; fixed stratified subsets, not population benchmark.'}
        (output/'transitions.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--seeds',nargs='+',type=int,default=[42,43])
    parser.add_argument('--rounds',type=int,default=3)
    args=parser.parse_args()
    args.output_root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    panel=args.output_root/'panel'
    prepare_panel(args.archive,panel,[15,26,34,78,98])
    for seed in args.seeds:
        for mode in ['self_only','peer']:
            run(panel,args.output_root/f'{seed}_{mode}',seed,mode,args.rounds)


if __name__=='__main__':main()
