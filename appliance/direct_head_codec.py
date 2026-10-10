"""Bounded JSON header + exact little-endian FP32 + packed masks; no pickle."""
import hashlib
import json
import struct
import numpy as np

from .config import Rejected
from .direct_head_contract import binding, validate_packet

MAGIC = b'APH1'
MAX_BYTES = 65536


def export_head(model, router, donor, task, target, authority):
    packet = dict(metadata=binding(model, router, donor, task, target, authority),
        weight=model.fc2.weight[target].detach().cpu().numpy().astype('<f4', copy=True),
        bias=float(model.fc2.bias[target].detach().cpu()),
        mask=model.weight_masks['fc2'][target].detach().cpu().numpy().copy(),
        bias_mask=float(model.bias_masks['fc2'][target].detach().cpu()))
    return validate_packet(packet)


def encode(packet):
    validate_packet(packet)
    h = json.dumps(packet['metadata'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    floats = np.concatenate((np.asarray(packet['weight'], dtype='<f4'),
                             np.asarray([packet['bias']], dtype='<f4'))).astype('<f4')
    bits = np.append(np.asarray(packet['mask'], np.uint8), int(packet['bias_mask']))
    body = MAGIC + struct.pack('<I', len(h)) + h + floats.tobytes() + np.packbits(bits, bitorder='little').tobytes()
    data = body + hashlib.sha256(body).digest()
    if len(data) > MAX_BYTES:
        raise Rejected('DIRECT_HEAD_PACKET_TOO_LARGE')
    return data


def decode(data):
    if not isinstance(data, bytes) or not 40 <= len(data) <= MAX_BYTES or data[:4] != MAGIC:
        raise Rejected('DIRECT_HEAD_BAD_ENVELOPE')
    if hashlib.sha256(data[:-32]).digest() != data[-32:]:
        raise Rejected('DIRECT_HEAD_CHECKSUM')
    try:
        n = struct.unpack('<I', data[4:8])[0]
        h = json.loads(data[8:8+n])
        d = h['architecture']['in_features']
        if type(d) is not int or not 1 <= d <= 8192:
            raise ValueError('dimension')
        expected = 8+n+4*(d+1)+(d+8)//8+32
        if len(data) != expected:
            raise ValueError('length')
        vals = np.frombuffer(data, dtype='<f4', count=d+1, offset=8+n).copy()
        bits = np.unpackbits(np.frombuffer(data[8+n+4*(d+1):-32], np.uint8), bitorder='little')
        if bits[d+1:].any():
            raise ValueError('noncanonical padding')
        return validate_packet(dict(metadata=h, weight=vals[:-1], bias=float(vals[-1]),
            mask=bits[:d].copy(), bias_mask=int(bits[d])))
    except (KeyError, ValueError, TypeError, struct.error) as exc:
        raise Rejected('DIRECT_HEAD_MALFORMED_PACKET', str(exc)) from exc
