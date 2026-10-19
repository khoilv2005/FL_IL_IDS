"""Client-local validation of peer updates, with episodic retention constraints.

This is a DENICE adaptation of personalized mixing, not an APFL reproduction.
No validation inputs, replay inputs or chosen weights leave the receiver.
"""
import math

import torch
import torch.nn.functional as F

from fed_learning.strategies.incremental.denice_replay import inference_statistics


def transfer_config(config):
    result = {'enabled': bool(config.get('denice_transfer_enabled', False))}
    for key, default in [('validation_limit', 256), ('memory_per_class', 16), ('batch_size', 128)]:
        value = config.get('denice_transfer_' + key, default)
        if isinstance(value, bool) or int(value) != value or value <= 0:
            raise ValueError('denice_transfer_' + key + ' must be a positive integer')
        result[key] = int(value)
    if result['enabled'] and (config.get('denice_aggregation_update_mode') != 'local_delta'
                              or config.get('denice_age_merge_policy', 'none') != 'none'):
        raise ValueError('Transfer selection requires local_delta and age_merge_policy=none')
    return result


@torch.no_grad()
def select_peer_transfer(model, local_state, peer_state, x, y, memory, controls, *, seed=0):
    """Line-search parameters; preserve all local buffers and protected entries.

    Current validation uses natural frequencies, never class-balanced accuracy.
    Each old class must retain its replay accuracy AND CE versus the post-local
    model. Replay is a training-memory proxy, not an independent validation set.
    Scores use current active adapters and pre-routing logits; no oracle routing.
    """
    if x is None or y is None or len(y) == 0:
        return local_state, {'weight': 0., 'reason': 'no_local_validation', 'scores': []}
    generator = torch.Generator().manual_seed(int(seed))
    index = torch.randperm(len(y), generator=generator)[:controls['validation_limit']]
    groups = [('validation', x[index], y[index])]
    if memory is not None:
        for label, entry in sorted(memory.entries.items()):
            index = torch.randperm(len(entry['y']), generator=generator)[:controls['memory_per_class']]
            groups.append(('memory_' + str(label), entry['x'][index], entry['y'][index]))
    parameters = set(dict(model.named_parameters()))
    device = next(model.parameters()).device

    def state_at(weight):
        return {name: (torch.lerp(value, peer_state[name].to(value), weight)
                       if weight and name in parameters and value.is_floating_point()
                       else value) for name, value in local_state.items()}

    def score():
        values = {}
        for name, inputs, labels in groups:
            loss, correct = 0., 0
            for start in range(0, len(labels), controls['batch_size']):
                target = labels[start:start + controls['batch_size']].to(device).long()
                logits = model(inputs[start:start + controls['batch_size']].to(device)).float()
                loss += float(F.cross_entropy(logits, target, reduction='sum'))
                correct += int((logits.argmax(1) == target).sum())
            values[name] = {'ce': loss / len(labels), 'correct': correct, 'count': len(labels)}
        return values

    rows = []
    chosen_weight = 0.
    try:
        with inference_statistics(model):
            model.load_state_dict(local_state, strict=True)
            baseline = score()
            if not all(math.isfinite(v['ce']) for v in baseline.values()):
                raise ValueError('Nonfinite local baseline in DENICE transfer selection')
            best = baseline['validation']['ce']
            rows.append({'weight': 0., 'eligible': True, 'metrics': baseline})
            for weight in (.25, .5, 1.):
                model.load_state_dict(state_at(weight), strict=True)
                metrics = score()
                eligible = all(math.isfinite(v['ce']) and v['ce'] <= baseline[name]['ce'] + 1e-7
                               and v['correct'] >= baseline[name]['correct']
                               for name, v in metrics.items())
                rows.append({'weight': weight, 'eligible': eligible, 'metrics': metrics})
                # Strict improvement: ties keep the smaller/local update.
                if eligible and metrics['validation']['ce'] < best - 1e-7:
                    best, chosen_weight = metrics['validation']['ce'], weight
    finally:
        model.load_state_dict(local_state, strict=True)
    return state_at(chosen_weight), {'weight': chosen_weight, 'reason': 'local_validation_and_memory',
                                     'scores': rows}
