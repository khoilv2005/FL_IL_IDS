"""Reduce only deltas produced from the identical receiver/head reference."""
import torch
from .snapshot import digest


def aggregate_head(reference, packets):
    if not packets:
        raise ValueError('No head contributions')
    total = sum(float(p['mass']) for p in packets)
    if not total > 0:
        raise ValueError('No contribution mass')
    reference_hash = digest(reference)
    owners = set()
    delta = torch.zeros_like(reference)
    signature = None
    for packet in packets:
        if packet['owner'] in owners:
            raise ValueError('Duplicate participant contribution')
        owners.add(packet['owner'])
        key = (packet['snapshot'], packet['head_reference'], packet['round'], tuple(packet['targets']))
        if signature is None:
            signature = key
        if key != signature or packet['head_reference'] != reference_hash:
            raise ValueError('Stale or incompatible head contribution')
        update = packet['delta'].to(reference)
        mass = float(packet['mass'])
        if update.shape != reference.shape or mass <= 0 or not torch.isfinite(update).all():
            raise ValueError('Invalid head delta or contribution mass')
        delta.add_(update, alpha=mass/total)
    result = reference + delta
    if not torch.isfinite(result).all():
        raise FloatingPointError('Nonfinite aggregate head')
    return result
