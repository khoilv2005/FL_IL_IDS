"""Controlled continuation probes using NICE local updates and locked graph deltas."""
from collections import OrderedDict
import copy
import numpy as np
import torch
from .selector import current_pool
from .state import complete_hash


def local_update(model,router,cid,roles,classes,task,round_id,config,device,batch_size,
                 row_cap,registry=None,audit_optimizer_delta=False):
    from fed_learning.clients.denice_client import DeNICEClient
    from fed_learning.strategies.incremental.nice import NICETrainer
    pool=current_pool(roles,cid,'base',classes)
    available=len(pool['y'])
    if row_cap:pool={**pool,'X':pool['X'][:row_cap],'y':pool['y'][:row_cap],'rows':pool['rows'][:row_cap]}
    if not len(pool['y']):return dict(rows=0,available_rows=available,optimizer_steps=0,skipped='no current BASE rows')
    epochs=max(1,int(config.get('nice_phase_epochs',1)))
    trainer=NICETrainer(tau=float(config.get('nice_tau',.95)),max_phases=1,phase_epochs=epochs)
    trainer.set_task(task,classes)
    client=DeNICEClient(cid,torch.from_numpy(pool['X']),torch.from_numpy(pool['y']),max_phases=1,phase_epochs=epochs)
    client.set_task_data(client.X_train,client.y_train,task,classes);client.setup_for_gpu(model,device)
    model.set_active_context(task)
    optimizers=[]
    step_snapshots={};step_audits=[]
    def optimizer_factory(parameters,lr):
        optimizer=torch.optim.Adam(parameters,lr=lr);optimizers.append(optimizer);return optimizer
    def protect():
        if registry is not None:
            registry.protect(model,optimizers[-1] if optimizers else None)
            # The exact dependency registry pins the prefix. A guarded-head
            # registry protects only FC2 and must allow upstream/BN drift.
            if getattr(registry,'freeze_bn',True):
                for module in model.modules():
                    if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
    def forward(x):protect();return model.forward_output(x)
    def before_step(current):
        if not audit_optimizer_delta:return
        step_snapshots.clear()
        step_snapshots.update({name:p.detach().clone() for name,p in current.named_parameters()})
    def after_step(current):
        protect()
        if not audit_optimizer_delta:return
        changed={}
        for name,p in current.named_parameters():
            delta=p.detach()-step_snapshots[name]
            count=int(torch.count_nonzero(delta))
            if count:changed[name]=dict(elements=count,max_absolute=float(delta.abs().max()))
        step_audits.append(dict(changed_parameters=changed,
            changed_elements=sum(v['elements'] for v in changed.values()),
            nonzero_gradient_elements=sum(int(torch.count_nonzero(p.grad)) for p in current.parameters() if p.grad is not None)))
        step_snapshots.clear()
    result=client.train(trainer,epochs,batch_size,float(config.get('learning_rate',.001)),
        phase_offset=round_id+1,max_phases_override=1,is_last_task=task==5,amp_enabled=False,
        continual_controls={'method':'legacy'},optimizer_factory=optimizer_factory,
        supervised_forward=forward,gradient_filter=(lambda:registry.gradient_filter(model)) if registry else model.reset_frozen_gradients,
        before_optimizer_step=before_step if audit_optimizer_delta else None,
        after_optimizer_step_update=after_step)
    protect()
    report=dict(rows=len(pool['y']),available_rows=available,optimizer_steps=result['optimizer_steps'],loss=result['loss'],
                phase_epochs=epochs,scope='one legacy NICE phase, FP32; current task only',bounded=len(pool['y'])<available,
                original_method_matched=config.get('denice_cl_method','legacy')=='legacy',router_refresh_performed=False)
    if audit_optimizer_delta:
        report.update(optimizer_delta_audit=step_audits,
            successful_steps_with_parameter_change=sum(v['changed_elements']>0 for v in step_audits))
    return report


def run_survival(model,router,registry,ledger,pair,groups,alphas,make_model,roles,classes,
                 config,device,predict,diagnostic_pool,old_pred,protocol,batch_size,row_cap,round_id):
    from fed_learning.strategies.decentralized.denice_aggregation import age_aware_aggregate,AggregationConfig
    from .evaluate import compare
    cid=pair['receiver'];neighbors=groups[cid];task=pair['task'];c=pair['class_id']
    baseline=OrderedDict((n,v.detach().cpu().clone()) for n,v in model.state_dict().items())
    before=complete_hash(model,router)
    deltas=[];ages=[];labels=[];weights=[];updates=[]
    # Each neighbor trains on its own current BASE shard; no pooling into receiver.
    for donor in neighbors:
        print(f'APPLIANCE survival neighbor: client={donor}',flush=True)
        neighbor,route=make_model(donor)
        start={n:v.detach().cpu().clone() for n,v in neighbor.state_dict().items()}
        update=local_update(neighbor,route,donor,roles,classes,task,round_id,config,device,batch_size,row_cap)
        updates.append(dict(client_id=donor,**update))
        delta=OrderedDict((n,v.detach().cpu()-start[n]) for n,v in neighbor.state_dict().items() if v.is_floating_point())
        deltas.append(delta);ages.append(copy.deepcopy(neighbor.unit_ranks))
        labels.append(sorted({c for values in route.episode_classes.values() for c in values}))
        weights.append(alphas[cid][donor]);del neighbor,route
    receiver_update=local_update(model,router,cid,roles,classes,task,round_id,config,device,batch_size,row_cap,registry)
    self_delta=OrderedDict((n,v.detach().cpu()-baseline[n]) for n,v in model.state_dict().items() if v.is_floating_point())
    deltas.append(self_delta);ages.append(copy.deepcopy(model.unit_ranks))
    labels.append(sorted({c for values in router.episode_classes.values() for c in values}));weights.append(alphas[cid][cid])
    def measure():
        valid=registry.validate(model,router,ledger)
        record=predict.records(model,router,diagnostic_pool['X'])
        return dict(registry_valid=all(valid.values()),entries=valid,state_hash=complete_hash(model,router),
            metrics=compare(old_pred,record['pred'],diagnostic_pool['y'],c,ledger.observed,record['task'],task),
            historical_old_task_damage='unmeasured')
    local=measure()
    # This stress replay fixes the recorded graph. It does not recompute capsules,
    # CANC/age merge or future graph selection and is not a full federation round.
    cfg=AggregationConfig(eta=float(config.get('denice_eta_agg',config.get('eta_agg',1.))),protect_mature=True)
    aggregated=age_aware_aggregate(baseline,model.unit_ranks,deltas,np.asarray(weights),cfg,
        neighbor_ages=ages,neighbor_labels=labels,target_labels=labels[-1])
    model.load_state_dict(aggregated,strict=True);registry.protect(model)
    combined=measure()
    return dict(initial_state_hash=before,receiver_local_update=receiver_update,neighbor_updates=updates,
        after_local=local,after_local_then_aggregation=combined,
        aggregation='retained age_aware_aggregate; local deltas; recorded positive graph alphas; dependency projection',
        full_training_round_verified=False,limitation='No CANC, router refresh, age merge or next-round graph recomputation; continuation stress probe only')
