"""Lossless mask-aware application messages, bound to the existing round plan.

Zero updates with a true mask still contribute to aggregation denominators.
Consequently masks are packed independently of values, never inferred from nnz.
This codec is a simulation transport boundary, not a network implementation.
"""
from dataclasses import asdict
import json

import numpy as np

from .aggregation import LocalContribution, _validate_contribution
from .contracts import array, decode, encode
from .knowledge import LocalLearningEvent


def _serialize(value):
    return json.dumps(encode(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def pack_contribution(plan, packet):
    """Choose the smaller complete message: legacy dense or packed masks/values."""
    _validate_contribution(plan, packet)
    dense = _serialize(dict(codec='dense_v1', packet=asdict(packet)))
    tensors = {}
    for name, value in packet.deltas.items():
        delta = array(value)
        mask = array(packet.masks[name]).astype(bool).reshape(-1)
        tensors[name] = dict(shape=tuple(delta.shape),
            mask=np.packbits(mask, bitorder='little'),
            values=delta.reshape(-1)[mask].copy())
    metadata = dict(event=asdict(packet.event), plan_hash=packet.plan_hash,
        initial_reference_hash=packet.initial_reference_hash,
        initial_topology_hash=packet.initial_topology_hash)
    compact = _serialize(dict(codec='masked_v1', metadata=metadata, tensors=tensors))
    return compact if len(compact) < len(dense) else dense


def unpack_contribution(plan, message):
    """Reconstruct, then enforce the same provenance/permission checks as dense."""
    payload = decode(json.loads(message))
    if payload['codec'] == 'dense_v1':
        fields = payload['packet']
    elif payload['codec'] == 'masked_v1':
        fields = dict(payload['metadata'])
        deltas, masks = {}, {}
        if set(payload['tensors']) != set(plan.parameter_permissions):
            raise ValueError('Compact tensor schema mismatch')
        for name, record in payload['tensors'].items():
            shape = tuple(plan.reference.model_state[name].shape)
            size = int(np.prod(shape))
            packed, values = record['mask'], record['values']
            if (tuple(record['shape']) != shape or not isinstance(packed, np.ndarray)
                    or packed.dtype != np.uint8 or packed.shape != ((size + 7) // 8,)):
                raise ValueError('Invalid compact shape/mask')
            bits = np.unpackbits(packed, bitorder='little')
            if bits[size:].any():
                raise ValueError('Nonzero mask padding')
            mask = bits[:size].astype(bool)
            if (not isinstance(values, np.ndarray) or values.ndim != 1
                    or values.dtype.kind != 'f' or len(values) != int(mask.sum())):
                raise ValueError('Invalid compact values')
            delta = np.zeros(size, dtype=values.dtype)
            delta[mask] = values
            deltas[name], masks[name] = delta.reshape(shape), mask.reshape(shape)
        fields.update(deltas=deltas, masks=masks)
    else:
        raise ValueError('Unknown contribution codec')
    fields['event'] = LocalLearningEvent(**fields['event'])
    packet = LocalContribution(**fields)
    _validate_contribution(plan, packet)
    return packet


def transport_contributions(plan, packets):
    """Exercise serialization and decoding before reduction, report exact bytes."""
    received, sizes = [], {}
    for packet in packets:
        message = pack_contribution(plan, packet)
        received.append(unpack_contribution(plan, message))
        sizes[packet.event.client_id] = len(message)
    return received, sizes
