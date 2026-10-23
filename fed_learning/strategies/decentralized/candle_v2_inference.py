"""Versioned CANDLE V2 inference policies over the same frozen local model.

Old checkpoints retain task-free inference unless the caller explicitly chooses
the saved-router repair. Routing uses input activations, never a sample label.
"""
import numpy as np
import torch

from .candle_v2 import features, logits, state

TASKFREE = 'taskfree_cosine_v1'
ROUTED = 'binary_cosine_registered_v1'
POLICIES = (TASKFREE, ROUTED)
PRIMARY = {TASKFREE: 'TaskFreeCosineSelf', ROUTED: 'BinaryCosineRegisteredSelf'}


def resolve_policy(config, override=None):
    policy = override if override is not None else config.get('candle_v2_inference_policy', TASKFREE)
    if policy not in POLICIES:
        raise ValueError(f'Unsupported CANDLE V2 inference policy: {policy}')
    return policy


def features_and_context(model, x):
    """Same adapter-free masked CNN/GRU/FC1 computation, one backbone forward."""
    if model.training:
        raise ValueError('CANDLE inference requires model.eval()')
    if model.adapter_registry or model.active_adapters or model.continual_head is not None:
        raise ValueError('CANDLE V2 inference requires an adapter-free local model')
    if x.ndim == 2:
        x = x.unsqueeze(-1)
    cnn = x.permute(0, 2, 1)
    acts = {}
    for layer in ('conv1', 'conv2', 'conv3'):
        number = layer[-1]
        cnn = model._apply_masked_conv(cnn, getattr(model, layer),
            getattr(model, 'bn'+number), getattr(model, 'pool'+number), layer)
        acts[layer] = cnn.abs().mean(dim=2)
    gru, _ = model._run_gru(x)
    gru = gru[:, -1, :] * model.weight_masks['gru'].to(gru)
    acts['gru'] = gru.abs()
    h = model.relu(model._apply_masked_linear(torch.cat([cnn.reshape(len(x), -1), gru], dim=1),
                                             model.fc1, 'fc1'))
    active = torch.as_tensor(np.asarray(model.unit_ranks['fc1']) > 0, device=h.device)
    return h * active.to(h.dtype), acts


@torch.no_grad()
def predict(model, x, config, router=None, policy=None):
    """Return logits, optional predicted episodes, and explicit empty-scope fallback.

    The router's locally owned episode_classes are not the class authorization
    source: CANDLE can have successfully trained imported classes in that task.
    Use only available classes and their saved class_tasks mapping instead.
    """
    policy = resolve_policy(config, policy)
    if model.training:
        raise ValueError('CANDLE inference requires model.eval()')
    value = state(model)
    available = sorted(set(map(int, value['available'])))
    if not available:
        raise ValueError('Checkpoint has no trained CANDLE V2 availability')
    if policy == TASKFREE:
        output = logits(model, features(model, x), available, config['candle_v2_logit_scale'])
        return dict(logits=output, unrouted_logits=output, episodes=None,
                    fallback=np.zeros(len(x), dtype=bool), policy=policy)
    if router is None or router.router_mode != 'binary_cosine':
        raise ValueError('Registered routing requires the saved binary_cosine detector')
    if not any(len(memory) for memory in router.activation_memory.values()):
        raise ValueError('Registered routing requires nonempty saved activation memory')
    mapping = {int(c):int(t) for c,t in value['class_tasks'].items()}
    if any(c not in mapping or not 0 <= mapping[c] <= int(value['task']) for c in available):
        raise ValueError('Available class has missing or invalid saved task provenance')
    h, acts = features_and_context(model, x)
    output = logits(model, h, available, config['candle_v2_logit_scale'])
    binary = router.binarize_layer_activations({name:act.cpu().numpy() for name,act in acts.items()})
    episodes = np.asarray(router.predict_episodes_batch(binary), dtype=np.int64)
    if episodes.shape != (len(x),) or (episodes < 0).any():
        raise ValueError('Saved router returned invalid episode IDs')
    valid = torch.zeros_like(output, dtype=torch.bool)
    fallback = np.zeros(len(x), dtype=bool)
    for episode in np.unique(episodes):
        rows = np.flatnonzero(episodes == episode)
        allowed = [c for c in available if mapping[c] == int(episode)]
        if not allowed:
            # No authorized class in selected scope: retain original task-free
            # prediction, explicitly counted by evaluators. Never open a class.
            allowed = available
            fallback[rows] = True
        row_ids = torch.as_tensor(rows, device=output.device)
        class_ids = torch.as_tensor(allowed, device=output.device)
        valid[row_ids[:, None], class_ids[None, :]] = True
    return dict(logits=output.masked_fill(~valid, -1e4), unrouted_logits=output, episodes=episodes,
                fallback=fallback, policy=policy)
