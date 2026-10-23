"""Class-balanced synchronous head SGD; frozen features remain at their owner."""
from collections import Counter
from dataclasses import asdict
import torch
import torch.nn.functional as F
from .head_aggregation import aggregate_head
from .snapshot import digest


def head_permissions(reference, targets):
    model = reference.model
    if len(set(targets)) != len(targets) or not targets:
        raise ValueError('Distinct target rows required')
    if any(c < 0 or c >= model.num_classes or model.unit_ranks['fc2'][c] != 0 for c in targets):
        raise ValueError('Only unallocated FC2 rows may be acquired in the first variant')
    # Use matured FC1 coordinates only. Existing mature connections are never reopened.
    mature = torch.as_tensor(model.unit_ranks['fc1'] >= 2, dtype=torch.bool)
    if not mature.any():
        raise ValueError('No consolidated feature coordinates available')
    return mature


def initial_head(reference, targets, permissions):
    model = reference.model
    rows = torch.as_tensor(targets, device=model.fc2.weight.device)
    weight = model.fc2.weight.detach().index_select(0, rows).cpu()
    weight *= permissions
    bias = model.fc2.bias.detach().index_select(0, rows).cpu().unsqueeze(1)
    return torch.cat([weight, bias], dim=1)


def participant_delta(panel, head, *, targets, allowed, class_weights, permissions,
                      rate, round_id, batch_size):
    labels = panel['labels']
    keep = torch.isin(labels, torch.as_tensor(allowed))
    z, logits, labels = panel['features'][keep], panel['logits'][keep], labels[keep]
    if not len(labels):
        return None
    target_index = [allowed.index(c) for c in targets]
    y = torch.as_tensor([allowed.index(int(c)) for c in labels])
    weights = torch.as_tensor([class_weights[int(c)] for c in labels], dtype=torch.float32)
    mass = float(weights.sum())
    phi = head.detach().clone().requires_grad_(True)
    gradient = torch.zeros_like(phi)
    loss_sum = 0.
    for start in range(0, len(labels), batch_size):
        end = start + batch_size
        scores = logits[start:end, allowed].clone()
        replacement = F.linear(z[start:end], phi[:, :-1] * permissions, phi[:, -1])
        scores[:, target_index] = replacement
        weighted = F.cross_entropy(scores, y[start:end], reduction='none') * weights[start:end]
        loss = weighted.sum() / mass
        grad, = torch.autograd.grad(loss, phi)
        gradient.add_(grad)
        loss_sum += float(weighted.detach().sum())
    gradient[:, :-1] *= permissions
    if not torch.isfinite(gradient).all():
        raise FloatingPointError('Nonfinite head gradient')
    return dict(owner=panel['owner'], snapshot=panel['identity']['reference_sha256'],
                head_reference=digest(head), round=int(round_id), targets=list(targets),
                delta=-rate*gradient, mass=mass, used_samples=len(labels),
                excluded_samples=int((~keep).sum()), loss=loss_sum/mass)


def learn_head(reference, panels, targets, *, config, ledger, include_receiver=True):
    """One full local gradient per head round, same starting phi everywhere.

    Global inverse-class-frequency weights are fixed from BASE counts only.
    Weighting each local gradient by its mass is exactly the gradient of that
    pooled empirical objective. No Adam states or multiple local steps are mixed.
    """
    permissions = head_permissions(reference, targets)
    allowed = sorted(set(reference.detector.episode_classes.get(reference.identity.task, [])) | set(targets))
    selected = [p for p in panels if include_receiver or p['owner'] != reference.identity.receiver]
    for panel in selected:
        if panel['identity'] != asdict(reference.identity):
            raise ValueError('Feature cache belongs to another reference')
    counts = Counter()
    for panel in selected:
        summary = dict(Counter(int(c) for c in panel['labels'].tolist() if int(c) in allowed))
        if panel['owner'] != reference.identity.receiver:
            summary = ledger.transmit(panel['owner'], reference.identity.receiver,
                                     'verification_receipt', dict(role='BASE_counts', counts=summary))['counts']
        counts.update(summary)
    if any(counts[c] == 0 for c in targets):
        raise ValueError('Target lacks current BASE positives')
    weights = {c: 1./n for c,n in counts.items()}
    phi = initial_head(reference, targets, permissions)
    history = []
    for round_id in range(config.head_rounds):
        packets = []
        for panel in selected:
            owner = panel['owner']
            local_phi = phi
            if owner != reference.identity.receiver:
                packet = ledger.transmit(reference.identity.receiver, owner, 'head_reference',
                    dict(identity=asdict(reference.identity), targets=list(targets), round=round_id,
                         head=phi, permissions=permissions, allowed=allowed, class_weights=weights))
                local_phi = packet['head']
            delta = participant_delta(panel, local_phi, targets=targets, allowed=allowed,
                class_weights=weights, permissions=permissions, rate=config.learning_rate,
                round_id=round_id, batch_size=config.batch_size)
            if delta is None:
                continue
            if owner != reference.identity.receiver:
                delta = ledger.transmit(owner, reference.identity.receiver, 'head_delta', delta)
            packets.append(delta)
        phi = aggregate_head(phi, packets)
        history.append(dict(round=round_id, participants=[{k:v for k,v in p.items()
                            if k not in ('delta',)} for p in packets]))
    return dict(head=phi, targets=list(targets), permissions=permissions, history=history,
                include_receiver=include_receiver, allowed=allowed)
