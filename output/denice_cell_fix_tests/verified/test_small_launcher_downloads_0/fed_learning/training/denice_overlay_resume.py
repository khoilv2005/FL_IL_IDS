"""Task-boundary recovery of client-private row-overlay state in simulation."""
from copy import deepcopy


def snapshot(protocol):
    return {'version': 1, 'knowledge': deepcopy(protocol.knowledge),
            'counts': deepcopy(protocol.counts),
            'banks': {cid: {label: (x.detach().cpu().clone(), y.detach().cpu().clone())
                            for label, (x, y) in validator.bank.items()}
                      for cid, validator in protocol.validators.items()}}


def restore(protocol, state, loader, split, limit):
    """Old checkpoints require explicit consent to reconstruct from identical NPZs.

    Historical sample histograms detect many dataset mismatches, but cannot
    certify unchanged features or row order. No test data is accessed.
    """
    from fed_learning.strategies.decentralized.denice_appliance import LocalValidator
    saved = state['config']
    if not saved.get('denice_appliance_enabled'):
        raise ValueError('Cannot resume row overlay from a non-overlay checkpoint')
    if state['meta'].get('boundary', 'task') != 'task':
        raise ValueError('Row overlay recovery currently requires a task boundary')
    defaults = {'seed': 42, 'random_seed': 42, 'denice_validation_fraction': 0.,
                'denice_max_clients': None, 'denice_max_train_samples_per_client': None,
                'appliance_bank_per_class': 64, 'num_clients': 100,
                'total_classes': 34, 'base_classes': 6, 'classes_per_task': 6}
    for key, default in defaults.items():
        if saved.get(key, default) != protocol.config.get(key, default):
            raise ValueError(f'Overlay resume partition configuration changed: {key}')
    payload = state.get('denice_appliance_state')
    if payload is not None:
        if payload.get('version') != 1:
            raise ValueError('Unsupported row-overlay continuation state version')
        if not (set(payload['banks']) == set(payload['knowledge']) == set(payload['counts'])):
            raise ValueError('Incomplete row-overlay client state')
        protocol.knowledge = deepcopy(payload['knowledge'])
        protocol.counts = deepcopy(payload['counts'])
        for cid, bank in payload['banks'].items():
            validator = LocalValidator(protocol.config)
            validator.bank = deepcopy(bank)
            protocol.validators[cid] = validator
        return 'checkpoint'
    if not protocol.config.get('denice_overlay_rebuild_private_state', False):
        raise ValueError('Old overlay checkpoint lacks private state. Use the identical dataset '
                         'and explicitly enable denice_overlay_rebuild_private_state.')
    history = {int(item['task']): item['client_data'] for item in state.get('debug_history', [])
               if item.get('type') == 'task_start'}
    cap = saved.get('denice_max_train_samples_per_client')
    cap = None if cap is None else max(1, int(cap))
    max_clients = saved.get('denice_max_clients')
    for task in range(int(state['meta']['completed_task']) + 1):
        if task not in history:
            raise ValueError(f'Missing historical data audit for task {task}')
        observed = set()
        for cid in loader.get_all_client_ids():
            x, y = loader.get_client_data(cid, task)
            if cap is not None and len(y) > cap:
                x, y = limit(x, y, cap, seed=int(saved.get('random_seed', saved.get('seed', 42))) + 10000 * task + int(cid))
            if not len(y) or (max_clients is not None and len(observed) >= int(max_clients)):
                continue
            observed.add(int(cid))
            audit = history[task].get(cid, history[task].get(str(cid)))
            labels, counts = y.unique(return_counts=True)
            hist = {int(c): int(n) for c, n in zip(labels, counts)}
            if audit is None or audit['num_samples'] != len(y) or hist != {int(c): int(n) for c, n in audit['class_hist'].items()}:
                raise ValueError(f'Dataset audit mismatch: task {task}, client {cid}')
            _, train_y, val_x, val_y = split(x, y, float(saved['denice_validation_fraction']),
                                            int(saved.get('seed', 42)) + task * 10000 + int(cid))
            protocol.add_local_task(cid, val_x, val_y, train_y)
        if observed != {int(cid) for cid in history[task]}:
            raise ValueError(f'Historical active client mismatch at task {task}')
        print(f'  Restored private overlay banks: task {task}, {len(observed)} clients', flush=True)
    return 'reconstructed_from_identical_dataset_required'
