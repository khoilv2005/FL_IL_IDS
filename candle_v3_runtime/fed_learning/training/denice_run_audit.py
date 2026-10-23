"""Small, dependency-free checks for reproducible DENICE experiments."""


def method_manifest(config):
    overlay = bool(config.get('denice_appliance_enabled', False))
    service = bool(config.get('appliance_enabled', False))
    teaching = bool(config.get('candle_v3_teaching', {}).get('enabled', False))
    transfer_mode = config.get('candle_v3_teaching', {}).get('transfer_mode', 'receiver_head_learning')
    if teaching and (overlay or service):
        raise ValueError('CANDLE V3 teaching cannot be combined with APPLIANCE implementations')
    if overlay and service:
        raise ValueError('Choose only one APPLIANCE implementation')
    method = (('candle_v3_direct_fc2' if transfer_mode == 'direct_fc2' else 'candle_v3_peer_teaching') if teaching else
              'row_overlay' if overlay else 'appliance_service' if service else 'denice_base')
    expected = config.get('denice_expected_method')
    if expected is not None and expected != method:
        raise ValueError(f'Expected method {expected!r}, effective method is {method!r}')
    return {
        'effective_method': method,
        'candle_v3_teaching_enabled': teaching,
        'candle_v3_transfer_mode': transfer_mode if teaching else None,
        'denice_appliance_enabled': overlay,
        'appliance_enabled': service,
        'denice_router_mode': config.get('denice_router_mode', 'binary_cosine'),
        'denice_amp_enabled': bool(config.get('denice_amp_enabled', True)),
        'denice_amp_initial_scale': config.get('denice_amp_initial_scale'),
        'denice_eval_require_full_coverage': bool(config.get('denice_eval_require_full_coverage', True)),
        'denice_eval_max_clients': config.get('denice_eval_max_clients'),
        'denice_eval_max_samples': config.get('denice_eval_max_samples'),
        'denice_clustering_mode': config.get('denice_clustering_mode'),
    }


def optimizer_health(result, *, fail_on_zero=False):
    steps = int(result.get('optimizer_steps', 0))
    skipped = int(result.get('skipped_optimizer_steps', 0))
    health = {'optimizer_steps': steps, 'skipped_optimizer_steps': skipped,
              'zero_update': steps == 0 and skipped > 0,
              'skipped_fraction': skipped / max(1, steps + skipped),
              'amp': result.get('amp', {})}
    if fail_on_zero and health['zero_update']:
        raise FloatingPointError('Client completed no optimizer update; inspect optimizer_health.jsonl')
    return health
