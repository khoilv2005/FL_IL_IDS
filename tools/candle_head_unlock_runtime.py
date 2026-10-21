"""Experimental, label-blind imported-head scoring with one local model.

No donor inference, router-memory merge, true class or task input. Context ID is
fixed capability metadata. This module does not authorize installation/training.
"""
import copy
import numpy as np
import torch
from fed_learning.training.denice_eval import _denice_routed_logits_with_episodes


def predict_packet(model, router, x, packet, seen_classes, device='cuda', batch_size=256):
    target, context = int(packet['class_id']), int(packet['context_id'])
    spec, head = packet['spec'], packet['head']
    gate = spec['gate']
    if spec.get('score') != 'raw' or gate not in ('donor_proto', 'receiver_proto'):
        raise ValueError('This frozen runtime supports the registered cosine-prototype/raw-head variants')
    if target not in seen_classes or context not in router.episode_classes:
        raise ValueError('Capability outside declared class/task state')
    if any(target in values for values in router.episode_classes.values()):
        raise ValueError('Expected an unregistered imported class; preserve legacy fallback explicitly')
    if model.continual_head is not None:
        raise ValueError('Residual classifier not supported by this head-only runtime')
    weight = torch.as_tensor(head['w'], dtype=model.fc2.weight.dtype, device=device)
    bias = float(head['b'])
    prototype_name = 'donor' if gate == 'donor_proto' else 'receiver'
    prototype = torch.as_tensor(packet['prototypes'][prototype_name], dtype=weight.dtype, device=device)
    if weight.shape != (model.fc2.in_features,) or prototype.shape != weight.shape:
        raise ValueError('Head/prototype feature dimensions differ')
    if not torch.isfinite(weight).all() or not torch.isfinite(prototype).all():
        raise ValueError('Nonfinite capability')
    allowed = sorted((set(router.episode_classes[context]) & set(seen_classes)) | {target})
    modes = [(m,m.training) for m in model.modules()]
    active = copy.deepcopy(model.active_adapters)
    predictions, activated = [], []
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0,len(x),batch_size):
                inputs = torch.as_tensor(x[start:start+batch_size],dtype=weight.dtype,device=device)
                legacy, _ = _denice_routed_logits_with_episodes(
                    model,inputs,router,list(seen_classes),device,inference_policy='pred_hard')
                fallback = legacy.argmax(1)
                model.set_active_context(context)
                h = model.penultimate_features(inputs)
                current = model._apply_masked_linear(h,model.fc2,'fc2')
                imported = h @ weight + bias
                current[:,target] = imported
                within = torch.as_tensor(allowed,device=device)[current[:,allowed].argmax(1)] == target
                score = (h @ prototype)/(h.norm(dim=1)*prototype.norm()).clamp_min(1e-8)
                use = (score >= float(spec['tau'])) & within
                if spec['gamma'] is not None:
                    use &= imported-legacy.max(1).values >= float(spec['gamma'])
                pred = fallback.clone();pred[use] = target
                predictions.append(pred.cpu().numpy());activated.append(use.cpu().numpy())
        return np.concatenate(predictions),np.concatenate(activated)
    finally:
        model.active_adapters = active
        for module,training in modes:module.training = training
