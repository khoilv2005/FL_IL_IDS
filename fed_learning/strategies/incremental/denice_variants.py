"""Explicit, mutually exclusive continual-learning experiments within DENICE."""
import math


def preserve_local_mature(aggregated, local, ages):
    """Local EWC/DER updates survive; protected coordinates never take peer deltas."""
    import torch
    from fed_learning.strategies.decentralized.denice_aggregation import build_compatible_mask
    masks = build_compatible_mask(local, ages)
    for name in aggregated:
        aggregated[name] = torch.where(masks[name].to(aggregated[name].device).bool(),
                                       aggregated[name], local[name].to(aggregated[name].device))
    return aggregated


def variant_config(config):
    result = {
        'method': str(config.get('denice_cl_method', 'legacy')).lower(),
        'train_mature': bool(config.get('denice_cl_train_mature', True)),
        'logit_scope': config.get('denice_cl_logit_scope', 'all'),
        'alpha': float(config.get('denice_der_alpha', .5)),
        'beta': float(config.get('denice_der_beta', .5)),
        'reduction': config.get('denice_der_reduction', 'sum'),
        'ewc_lambda': float(config.get('denice_ewc_lambda', 100.)),
        'ewc_mode': config.get('denice_ewc_mode', 'separate'),
        'fisher_samples': config.get('denice_ewc_fisher_samples', 128),
        'fisher_labels': config.get('denice_ewc_fisher_labels', 'model'),
        'decay': float(config.get('denice_ewc_decay', 1.)),
    }
    if result['method'] not in ('legacy', 'der', 'derpp', 'ewc'):
        raise ValueError('denice_cl_method must be legacy, der, derpp or ewc')
    for key in ('alpha', 'beta', 'ewc_lambda', 'decay'):
        if not math.isfinite(result[key]) or result[key] < 0:
            raise ValueError(f'Invalid continual-learning control: {key}')
    if result['decay'] > 1 or result['logit_scope'] not in ('all', 'seen'):
        raise ValueError('Invalid decay or logit scope')
    if result['reduction'] not in ('sum', 'mean') or result['ewc_mode'] not in ('separate', 'online'):
        raise ValueError('Invalid DER reduction or EWC mode')
    if result['fisher_labels'] not in ('model', 'empirical'):
        raise ValueError('Fisher labels must be model or empirical')
    n = result['fisher_samples']
    if isinstance(n, bool) or int(n) != n or n < 1:
        raise ValueError('Fisher samples must be a positive integer')
    result['fisher_samples'] = int(n)
    if result['method'] != 'legacy':
        if config.get('denice_plasticity_enabled', False) or config.get('denice_continual_width', 0):
            raise ValueError('DER/EWC experiments cannot mix with experimental plasticity/residual head')
        if config.get('denice_classifier_enabled', False) or config.get('denice_transfer_enabled', False):
            raise ValueError('Disable classifier/transfer ablations for DER/EWC experiments')
        capacity = int(config.get('denice_replay_capacity', 0))
        if result['method'] in ('der', 'derpp') and capacity < 1:
            raise ValueError('DER requires a positive replay capacity')
        if result['method'] == 'ewc' and (capacity or config.get('denice_router_replay_enabled', False)):
            raise ValueError('The EWC-only version must disable raw replay and router replay')
    return result


def variant_preset(name):
    if name not in ('der', 'derpp', 'ewc'):
        raise ValueError('DENICE_VARIANT must be der, derpp or ewc')
    replay = name != 'ewc'
    return dict(algorithm='denice', mode='decentralized', denice_cl_method=name,
                denice_cl_train_mature=True, denice_cl_logit_scope='all',
                denice_der_alpha=.5, denice_der_beta=.5, denice_der_reduction='sum',
                denice_ewc_lambda=100., denice_ewc_mode='separate',
                denice_ewc_fisher_samples=128, denice_ewc_fisher_labels='model', denice_ewc_decay=1.,
                denice_replay_capacity=1024 if replay else 0, denice_replay_batch_size=32,
                denice_replay_ce_weight=0., denice_replay_logit_weight=0., denice_replay_calibration_weight=0.,
                denice_replay_selection='priority', denice_router_replay_enabled=False,
                denice_memory_policy='local_replay' if replay else 'sketches',
                denice_plasticity_enabled=False, denice_continual_width=0,
                denice_classifier_enabled=False, denice_transfer_enabled=False,
                denice_shared_context_eval=False, denice_eval_route_mode='hard',
                denice_calibrate_plastic_bn=True, denice_eval_final_round=True,
                denice_eval_local_validation=True)
