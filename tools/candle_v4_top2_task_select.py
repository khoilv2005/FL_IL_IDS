"""Input-only task reranking; frozen LR class mass, single-task scoped inference."""
import numpy as np
import torch

from eval_candle_v4_task_round19_topk import candidates
from fed_learning.training.candle_v4_bank import base_predictions
from fed_learning.training.candle_v4_continuous_runtime import ContinuousRoutingView,predictions

POLICY='top2_class_probability_mass_v1'
PROTOTYPE_POLICY='top2_receiver_router_prototype_cosine_v1'


def task_mass(memory,features):
    """Marginalize the existing LR class probabilities; never fit or calibrate."""
    if len(memory.fitted_classes)==1:
        classes=memory.fitted_classes;p=np.ones((len(features),1))
    else:
        classes=memory.classifier.classes_
        p=memory.classifier.predict_proba(memory.scaler.transform(features))
    mass=np.zeros((len(features),6),dtype=np.float64)
    for j,c in enumerate(classes):mass[:,int(c)//6]+=p[:,j]
    if not np.isfinite(mass).all() or not np.allclose(mass.sum(1),1.):raise ValueError('Invalid frozen class probabilities')
    return mass


def choose_task(top2,mass):
    """Strict comparison, keeping production Top1 on ties or absent runner-up."""
    first=top2[:,0];second=top2[:,1];rows=np.arange(len(first));valid=second>=0
    change=valid&(mass[rows,np.maximum(second,0)]>mass[rows,first])
    return np.where(change,second,first)


def task_prototype_similarity(memory,features):
    """Receiver-space class centroids; standardized cosine, no donor alignment."""
    classes=sorted(memory.memory)
    prototypes=np.stack([memory.memory[c].mean(0) for c in classes])
    p=memory.scaler.transform(prototypes);z=memory.scaler.transform(features)
    p=p/np.maximum(np.linalg.norm(p,axis=1,keepdims=True),np.finfo(np.float32).eps)
    z=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),np.finfo(np.float32).eps)
    similarity=z@p.T;values=np.full((len(features),6),-np.inf)
    for j,c in enumerate(classes):values[:,c//6]=np.maximum(values[:,c//6],similarity[:,j])
    return values


class FixedChoiceView(ContinuousRoutingView):
    def __init__(self,native,memory,choice):
        super().__init__(native,memory);self.choice=np.asarray(choice,dtype=np.int64)
    def predict_episodes_batch(self,features):
        if len(features)!=len(self.choice):raise ValueError('Fixed route batch mismatch')
        return self.choice.copy()
    def predict_episodes_with_scores(self,features):
        return self.predict_episodes_batch(features),None


def predict_selected(service,cid,model,det,inputs,seen,rule='mass'):
    """No labels. Two candidate tasks, exactly one adapter/mask/expert scope."""
    fallback,top1=predictions(service,cid,model,det,inputs,seen)
    baseline,_=service.infer(cid,'coverage',fallback,inputs,predicted_tasks=top1)
    memory=service.continuous_routers[cid];view=ContinuousRoutingView(det,memory)
    device=next(model.parameters()).device
    with torch.no_grad():acts=model.get_context_activations_per_sample(inputs.to(device))
    features=view.binarize_layer_activations({k:v.cpu().numpy() for k,v in acts.items()})
    check,scores=memory.predict(features)
    if not np.array_equal(top1,check):raise ValueError('Top1 route parity failed')
    top2=candidates(top1,scores,set(memory.class_to_task.values()))
    if rule not in ('mass','prototype'):raise ValueError('Unregistered task selection rule')
    scores=task_mass(memory,features) if rule=='mass' else task_prototype_similarity(memory,features)
    choice=choose_task(top2,scores);switched=np.flatnonzero(choice!=top1)
    result=baseline.copy()
    if len(switched):
        selected_inputs=inputs[torch.as_tensor(switched)]
        chosen=choice[switched]
        native,routes=base_predictions(model,FixedChoiceView(det,memory,chosen),selected_inputs,seen,True)
        if not np.array_equal(routes,chosen):raise ValueError('Candidate-specific adapter route failed')
        result[switched]=service.infer(cid,'coverage',native,selected_inputs,predicted_tasks=chosen)[0]
    return dict(top1=baseline,top2=result),top2,choice
