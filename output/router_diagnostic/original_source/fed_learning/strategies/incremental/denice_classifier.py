"""Client-local balanced discriminant readout for incremental DENICE.

Inspired by Deep SLDA (Hayes & Kanan, CVPRW 2020) and decoupled classifier
learning (Kang et al., ICLR 2020). This is a replay-refitted, class-balanced
shrinkage LDA, not a reproduction of streaming SLDA. No test data, central
teacher, peer statistics, or updates to protected model parameters are used.
"""
from contextlib import contextmanager
import math
import time

import torch
import torch.nn.functional as F

from .denice_replay import inference_statistics


def classifier_config(config):
    result = {
        'enabled': bool(config.get('denice_classifier_enabled', False)),
        'per_class': config.get('denice_classifier_per_class', 128),
        'batch_size': config.get('denice_classifier_batch_size', 256),
        'shrinkage': float(config.get('denice_classifier_shrinkage', .1)),
        'temperature': float(config.get('denice_classifier_temperature', 1.)),
        'validation_select': bool(config.get('denice_classifier_validation_select', False)),
        'validation_per_class': config.get('denice_classifier_validation_per_class', 32),
    }
    for key in ('per_class', 'batch_size', 'validation_per_class'):
        value = result[key]
        if isinstance(value, bool) or int(value) != value or value <= 0:
            raise ValueError(f'denice_classifier_{key} must be a positive integer.')
        result[key] = int(value)
    if not 0 < result['shrinkage'] <= 1 or not math.isfinite(result['temperature']) or result['temperature'] <= 0:
        raise ValueError('Classifier shrinkage must be in (0,1], temperature finite and positive.')
    return result


@contextmanager
def backbone_features(model):
    """A common feature space across tasks, independent of predicted context."""
    active = dict(model.active_adapters)
    try:
        model.clear_active_adapters()
        with inference_statistics(model):
            yield
    finally:
        model.active_adapters = active


@torch.no_grad()
def encode_features(model, inputs, batch_size):
    device = next(model.parameters()).device
    with backbone_features(model):
        return torch.cat([
            F.normalize(model.penultimate_features(inputs[i:i+batch_size].to(device)).float(), dim=1).cpu()
            for i in range(0, len(inputs), batch_size)
        ])


def fit_balanced_lda(features, labels, shrinkage=.1, temperature=1.):
    """Equal class priors and equal-class pooled within-class covariance.

    Fit in float64, store only an affine discriminant in float32. Isotropic
    trace-scaled shrinkage makes undersampled/singular covariance invertible.
    """
    x = F.normalize(features.detach().cpu().double(), dim=1)
    y = labels.detach().cpu().long()
    if x.ndim != 2 or len(x) != len(y) or not len(y) or not torch.isfinite(x).all():
        raise ValueError('Classifier needs finite 2D features and matching nonempty labels.')
    if not 0 < shrinkage <= 1 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Invalid LDA shrinkage/temperature.')
    classes = torch.unique(y, sorted=True)
    means, covariances, counts = [], [], []
    for label in classes:
        group = x[y == label]
        mean = group.mean(0)
        centered = group - mean
        means.append(mean)
        counts.append(len(group))
        if len(group) > 1:
            covariances.append(centered.T @ centered / (len(group) - 1))
    means = torch.stack(means)
    dim = x.shape[1]
    covariance = (torch.stack(covariances).mean(0) if covariances
                  else torch.eye(dim, dtype=x.dtype))
    scale = covariance.diag().mean().clamp_min(1e-6)
    regularized = (1 - shrinkage) * covariance + shrinkage * scale * torch.eye(dim, dtype=x.dtype)
    weights = torch.linalg.solve(regularized, means.T)
    bias = -.5 * (means * weights.T).sum(1)
    return {'version': 1, 'classes': classes, 'weight': weights.float(),
            'bias': bias.float(), 'temperature': float(temperature),
            'counts': counts, 'feature_dim': dim, 'shrinkage': float(shrinkage)}


@torch.no_grad()
def fit_local_classifier(model, memory, inputs, labels, config, *, task_id, client_id,
                         detector=None, validation_inputs=None, validation_labels=None):
    """Refit on train-only current samples + old private exemplars.

    A separate RNG prevents this readout from changing backbone training RNG.
    For recurring classes, current observations replace their old fit examples.
    Old classes are re-encoded with the current backbone, avoiding stale means.
    """
    cfg = classifier_config(config)
    if not cfg['enabled']:
        return None
    start = time.perf_counter()
    generator = torch.Generator().manual_seed(
        int(config.get('random_seed', config.get('seed', 42))) + 100003 * int(task_id) + int(client_id))
    labels = labels.detach().cpu().long()
    current = set(labels.tolist())
    entries = {} if memory is None else memory.entries
    classes = sorted(current | set(entries))
    if not classes:
        raise ValueError('Cannot fit local classifier without local observations.')
    all_features, all_labels = [], []
    holdout_x, holdout_y = [], []
    for label in classes:
        if label in current:
            index = torch.nonzero(labels == label, as_tuple=False).flatten()
            index = index[torch.randperm(len(index), generator=generator)[:cfg['per_class']]]
            selected = inputs[index]
        else:
            source = entries[label]['x']
            index = torch.randperm(len(source), generator=generator)
            if cfg['validation_select'] and len(index) >= 2:
                held_count = min(cfg['validation_per_class'], max(1, len(index) // 5))
                holdout_x.append(source[index[:held_count]])
                holdout_y.append(torch.full((held_count,), label, dtype=torch.long))
                index = index[held_count:]
            index = index[:cfg['per_class']]
            selected = source[index]
        all_features.append(encode_features(model, selected, cfg['batch_size']))
        all_labels.append(torch.full((len(selected),), label, dtype=torch.long))
    state = fit_balanced_lda(torch.cat(all_features), torch.cat(all_labels),
                             cfg['shrinkage'], cfg['temperature'])
    state.update(task_id=int(task_id), client_id=int(client_id))
    # Plain algorithm state: deliberately NOT a registered model parameter or
    # buffer, so model state_dict/capsule aggregation cannot exchange the head.
    model.local_classifier = state
    selection = {'selected_weight': 1., 'reason': 'validation_selection_disabled'}
    if cfg['validation_select']:
        if detector is None:
            raise ValueError('Validation selection requires the client context detector.')
        if validation_labels is not None and len(validation_labels):
            validation_labels = validation_labels.detach().cpu().long()
            for label in sorted(current):
                index = torch.nonzero(validation_labels == label, as_tuple=False).flatten()
                index = index[torch.randperm(len(index), generator=generator)[:cfg['validation_per_class']]]
                if len(index):
                    holdout_x.append(validation_inputs[index].detach().cpu())
                    holdout_y.append(validation_labels[index])
        if holdout_y:
            selection = select_readout_weight(model, detector, torch.cat(holdout_x),
                                               torch.cat(holdout_y), classes, cfg['batch_size'])
        else:
            selection = {'selected_weight': 0., 'reason': 'no_disjoint_validation_samples'}
        state['blend_weight'] = selection['selected_weight']
        state['selection'] = selection
    return {'method': 'balanced_shrinkage_lda', 'classes': classes,
            'fit_counts': dict(zip(classes, state['counts'])),
            'selection': selection,
            'seconds': time.perf_counter() - start,
            'bytes': sum(v.numel()*v.element_size() for v in state.values() if torch.is_tensor(v))}


def blend_readout_logits(lda, hard, weight):
    if weight <= 0:
        return hard
    if weight >= 1:
        return lda
    return torch.logaddexp(F.log_softmax(lda, dim=1) + math.log(weight),
                          F.log_softmax(hard, dim=1) + math.log1p(-weight))


@torch.no_grad()
def select_readout_weight(model, detector, inputs, labels, classes, batch_size):
    """Fixed grid, class-balanced validation accuracy; ties prefer baseline.

    This is validation-selected probability mixing, NOT the BiC algorithm.
    The old exemplars here were excluded from LDA fitting by the caller.
    """
    from fed_learning.training.denice_eval import _denice_routed_logits_with_episodes
    lda, hard = [], []
    device = next(model.parameters()).device
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i+batch_size].to(device)
        lda.append(local_classifier_logits(model, batch, classes).cpu())
        hard.append(_denice_routed_logits_with_episodes(
            model, batch, detector, classes, str(device), route_mode='hard')[0].cpu())
    lda, hard = torch.cat(lda), torch.cat(hard)
    scores = []
    for weight in (0., .25, .5, .75, 1.):
        prediction = blend_readout_logits(lda, hard, weight).argmax(1)
        score = torch.stack([(prediction[labels == c] == c).float().mean()
                             for c in torch.unique(labels)]).mean().item()
        scores.append({'weight': weight, 'balanced_accuracy': score})
    best = max(scores, key=lambda item: item['balanced_accuracy'])
    return {'selected_weight': best['weight'], 'reason': 'local_disjoint_validation',
            'scores': scores, 'validation_counts': {int(c): int((labels == c).sum())
                                                   for c in torch.unique(labels)}}


@torch.no_grad()
def local_classifier_logits(model, inputs, seen_classes):
    state = getattr(model, 'local_classifier', None)
    if state is None:
        raise ValueError('local_lda requires a fitted DENICE local classifier; use a task-end checkpoint or route_mode=hard.')
    if state.get('version') != 1:
        raise ValueError('Unsupported DENICE classifier state version.')
    with backbone_features(model):
        features = F.normalize(model.penultimate_features(inputs).float(), dim=1)
    logits = (features @ state['weight'].to(features.device) + state['bias'].to(features.device)) / state['temperature']
    classes = state['classes'].to(features.device)
    known = torch.isin(classes, torch.tensor(seen_classes, device=features.device))
    if not known.any():
        raise ValueError('Local classifier has no classes in the requested seen-class support.')
    # Log probabilities on the locally supported label space: ensemble callers
    # can normalize as before; unknown classes receive zero probability.
    scores = F.log_softmax(logits[:, known], dim=1)
    output = torch.full((len(inputs), model.num_classes), -1e9, device=features.device)
    output[:, classes[known]] = scores
    return output


def herding_indices(features, count):
    """Greedy normalized-mean exemplar approximation inspired by iCaRL.

    Order matters: every prefix approximates the full candidate mean. Never
    choose the same candidate twice, including duplicate/constant features.
    """
    features = F.normalize(features.detach().cpu().float(), dim=1)
    count = min(int(count), len(features))
    target = features.mean(0)
    running = torch.zeros_like(target)
    used = torch.zeros(len(features), dtype=torch.bool)
    selected = []
    for step in range(count):
        distances = ((running + features) / (step + 1) - target).square().sum(1)
        distances[used] = float('inf')
        index = int(distances.argmin())
        selected.append(index)
        used[index] = True
        running += features[index]
    return torch.tensor(selected, dtype=torch.long)
