"""Local capacity scheduling and Fisher-guided maturation for DENICE.

This is an experimental composition, not a reproduction of DEN or EWC.
Only model parameters participate in the existing peer protocol; anchors stay local.
"""
import math
import numpy as np
import torch


def plasticity_config(config):
    controls = dict(enabled=bool(config.get('denice_plasticity_enabled', False)),
                    frontload=float(config.get('denice_capacity_frontload', 1.5)),
                    mature_fraction=float(config.get('denice_mature_fraction', .8)),
                    strength=float(config.get('denice_elastic_strength', 100.)),
                    decay=float(config.get('denice_elastic_decay', .9)))
    if not all(math.isfinite(v) for k, v in controls.items() if k != 'enabled'):
        raise ValueError('DENICE plasticity controls must be finite')
    if controls['frontload'] < 1 or not 0 < controls['mature_fraction'] <= 1:
        raise ValueError('frontload must be >=1 and mature_fraction in (0,1]')
    if controls['strength'] < 0 or not 0 <= controls['decay'] <= 1:
        raise ValueError('elastic strength must be >=0 and decay in [0,1]')
    return controls


def capacity_plan(model, classes, task_id, num_tasks, controls):
    """Reserve budget uses declared horizon, never future inputs or test scores."""
    remaining = max(1, int(num_tasks) - int(task_id))
    allocation = {}
    for layer, ranks in model.unit_ranks.items():
        if layer == 'fc2':
            continue
        free = int(np.count_nonzero(ranks == 0))
        old_budget = int(model.capacity_per_class.get(layer, math.ceil(len(ranks) / model.num_classes))) * len(classes)
        budget = max(old_budget, math.ceil(free * min(1., controls['frontload'] / remaining))) if classes else 0
        allocation[layer] = min(free, budget)
    # Exhausted layers expand through existing input adapters, without resetting
    # old coordinates or changing shapes exchanged by peers.
    adapters = [layer for layer in ('conv3', 'gru', 'fc1')
                if classes and allocation.get(layer, 0) == 0]
    return dict(layers={}, adapters_to_add=adapters, recycle_layers=[],
                freeze_low_layers=False, novelty=0., action='Allocate' if classes else 'Reuse',
                controller='local_horizon_fisher', reserve_to_promote={},
                adaptive_allocation=allocation, remaining_tasks=remaining)


def allocate_capacity(model, plan):
    allocated = {}
    for layer, count in plan['adaptive_allocation'].items():
        free = np.flatnonzero(model.unit_ranks[layer] == 0)
        selected = free[:count]
        model.unit_ranks[layer][selected] = 1
        allocated[layer] = len(selected)
    return allocated


def consolidate(model, fisher, controls):
    """Graduate important units; keep a small, regularized plastic population."""
    previous = getattr(model, 'elastic_state', {})
    merged = {}
    for name, parameter in model.named_parameters():
        if name.startswith('adapters.'):
            continue  # Historical adapters already have a separate lifecycle.
        value = torch.as_tensor(fisher.get(name, torch.zeros_like(parameter, device='cpu'))).detach().cpu().float()
        if value.shape != parameter.shape or not torch.isfinite(value).all() or (value < 0).any():
            raise ValueError(f'Invalid consolidation importance: {name}')
        old = previous.get('importance', {}).get(name)
        merged[name] = value.clone() if old is None else value + controls['decay'] * old
    audit = {}
    for layer, ranks in model.unit_ranks.items():
        learners = np.flatnonzero(ranks == 1)
        ranks[ranks >= 2] += 1
        names = [n for n in merged if n.startswith(layer + '.') and 'weight' in n]
        score = torch.zeros(len(ranks))
        for name in names:
            rows = merged[name].reshape(merged[name].shape[0], -1).mean(1)
            if len(rows) == len(ranks):
                score += rows
            elif layer == 'gru' and len(rows) == 3 * len(ranks):
                score += rows.reshape(3, len(ranks)).mean(0)
        count = len(learners)
        # With no importance evidence, use the original hard consolidation.
        if layer != 'fc2' and count and float(score[learners].sum()) > 0:
            count = math.ceil(count * controls['mature_fraction'])
        ordered = learners[np.argsort(-score[learners].numpy(), kind='stable')]
        ranks[ordered[:count]] = 2
        audit[layer] = dict(graduated=count, retained_plastic=len(learners) - count)
    model.elastic_state = dict(
        importance=merged,
        anchor={name: p.detach().cpu().clone() for name, p in model.named_parameters() if name in merged})
    return audit


def elastic_loss_factory(model, strength):
    """One device copy per local train call, released afterwards (100-client safe)."""
    state = getattr(model, 'elastic_state', {})
    if not strength or not state:
        return None
    from fed_learning.strategies.decentralized.denice_aggregation import build_compatible_mask
    parameters = dict(model.named_parameters())
    ages = {name: np.where(rank == 1, 1, 2) for name, rank in model.unit_ranks.items()}
    masks = build_compatible_mask(state['importance'], ages)
    terms = []
    denominator = 0.
    for name, importance in state['importance'].items():
        p = parameters[name]
        weights = importance * masks[name].cpu()
        total = float(weights.sum())
        if total > 0:
            terms.append((p, state['anchor'][name].to(p.device), weights.to(p.device)))
            denominator += total
    if not terms:
        return None
    def loss():
        return strength * sum((w * (p.float() - anchor).square()).sum()
                              for p, anchor, w in terms) / denominator
    return loss
