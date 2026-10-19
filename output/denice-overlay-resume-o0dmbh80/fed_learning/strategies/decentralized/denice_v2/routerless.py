"""Opt-in routerless cluster class-incremental prediction and loss contract.

Task boundaries remain known to the training scheduler. Inference accepts only
x; no task selection, activation memory, context detector or donor predictor.
"""
import numpy as np
import torch
from torch.nn import functional as F

from fed_learning.strategies.incremental.denice import DeNICETrainer
from .aggregation import _validate_contribution,topology_hash
from .contracts import ClusterState,state_hash
from .native import capture_native,load_native
from .state import snapshot,restore


def capture_routerless(model,**identity):
    router = dict(router_mode='routerless_v1',context_layers=(),stable_feature_mask=np.zeros(0,bool),
        binarize_thresholds=None,activation_memory={},episode_classes={})
    return capture_native(model,router_state=router,routing_dependencies=(),**identity)


def training_classes(plan):
    plan.validate()
    if plan.reference.algorithm_state.get('classification_objective') != 'cumulative_registered_ce_v1':
        raise ValueError('Not a cumulative class training plan')
    return tuple(sorted(set(plan.reference.algorithm_state['seen_classes']) |
        {c for counts in plan.base_counts.values() for c in counts}))


def masked_logits(logits,classes):
    if not classes or tuple(sorted(set(classes))) != tuple(classes) or any(type(c) is not int or not 0 <= c < logits.shape[1] for c in classes):
        raise ValueError('Invalid active classes')
    mask = torch.zeros(logits.shape[1],dtype=torch.bool,device=logits.device)
    mask[list(classes)] = True
    return logits.masked_fill(~mask,float('-inf'))


class CumulativeClassTrainer(DeNICETrainer):
    def __init__(self,classes,**kwargs):
        super().__init__(**kwargs)
        self.active_classes = tuple(classes)

    def compute_loss(self,model,output,target,global_params=None,**kwargs):
        if not set(target.detach().cpu().tolist()) <= set(self.active_classes):
            raise ValueError('Target outside authorized cumulative classes')
        # Slice the denominator as well as targets, so unlearned logits have
        # neither probability mass nor gradient. Old registered logits compete.
        indices = torch.tensor(self.active_classes,device=output.device,dtype=torch.long)
        lookup = torch.full((output.shape[1],),-1,device=output.device,dtype=torch.long)
        lookup[indices] = torch.arange(len(indices),device=output.device)
        weights = kwargs.get('class_weights')
        if weights is not None:
            weights = weights.to(output)[indices]
        return F.cross_entropy(output[:,indices],lookup[target.long()],weight=weights)


def register_learned_classes(state,plan,contributions):
    """Commit registrations only from authorized successful optimizer receipts."""
    state.validate();plan.validate()
    if (state.router_state['router_mode'] != 'routerless_v1' or state.task != plan.reference.task
        or state.round != plan.target_round or state.cluster_uid != plan.reference.cluster_uid
        or state.members != plan.reference.members or topology_hash(state) != plan.topology_hash):
        raise ValueError('Invalid routerless registration round')
    actual = {}
    for packet in contributions:
        cid = packet.event.client_id
        if cid in actual:
            raise ValueError('Duplicate registration contributor')
        actual[cid] = tuple(_validate_contribution(plan,packet))
    if set(actual) != set(state.members):
        raise ValueError('Missing registration receipts')
    labels = tuple(sorted({c for cs in actual.values() for c in cs}))
    if not labels:
        raise ValueError('No learned class to register')
    payload = snapshot(state)
    history = payload['algorithm_state']['learned_classes_by_task']
    history[state.task] = tuple(sorted(set(history.get(state.task,())) | set(labels)))
    payload['algorithm_state']['seen_classes'] = tuple(sorted({c for cs in history.values() for c in cs}))
    candidate = ClusterState.create(**{k:v for k,v in payload.items() if k not in ('model_spec','routing_space_hash')})
    return candidate,dict(task=state.task,round=state.round,trained_classes=labels,
        seen_classes=candidate.algorithm_state['seen_classes'],contributors=actual,
        weights_unchanged=state_hash(candidate.model_state)==state_hash(state.model_state),
        router_memory_rows=0,routing_dependency_parameters=0)


class RouterlessPredictor:
    def __init__(self,state,device='cpu'):
        self.state = restore(snapshot(state))
        if self.state.router_state['router_mode'] != 'routerless_v1' or not self.state.algorithm_state['seen_classes']:
            raise ValueError('Routerless predictor requires learned classes')
        self.model,self.device = load_native(self.state,device).eval(),device

    def predict(self,x):
        """x only, one local network, cumulative class mask then argmax."""
        self.state.validate()
        with torch.no_grad():
            logits = masked_logits(self.model(x.to(self.device)),self.state.algorithm_state['seen_classes'])
        return logits.argmax(1).cpu().numpy(),logits.cpu().numpy()
