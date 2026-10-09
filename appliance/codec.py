"""Bounded FP32 tensor packet; metadata is JSON and tensor bytes are non-executable."""
import hashlib
import json
import math
import struct
import numpy as np
from .config import Rejected


def encode(metadata,tensors,max_bytes):
    entries=[];parts=[];offset=0
    for name in sorted(tensors):
        array=np.asarray(tensors[name],dtype='<f4',order='C')
        if not np.isfinite(array).all():raise Rejected('NONFINITE_PAYLOAD',name)
        raw=array.tobytes();parts.append(raw)
        entries.append(dict(name=name,shape=list(array.shape),dtype='<f4',offset=offset,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        offset+=len(raw)
    header=json.dumps(dict(schema='appliance_fp32_v1',metadata=metadata,tensors=entries),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    packet=struct.pack('<I',len(header))+header+b''.join(parts)
    if len(packet)>max_bytes:raise Rejected('PATCH_BUDGET_EXCEEDED',f'{len(packet)} > {max_bytes}')
    return packet


def decode(packet,max_bytes,expected_hash=None):
    if len(packet)>max_bytes or len(packet)<4:raise Rejected('INVALID_PACKET_SIZE')
    if expected_hash and hashlib.sha256(packet).hexdigest()!=expected_hash:raise Rejected('HASH_OR_SIGNATURE_FAIL')
    size=struct.unpack('<I',packet[:4])[0]
    if size>len(packet)-4:raise Rejected('INVALID_HEADER')
    try:header=json.loads(packet[4:4+size].decode())
    except (ValueError,UnicodeError) as exc:raise Rejected('INVALID_HEADER') from exc
    if header.get('schema')!='appliance_fp32_v1':raise Rejected('INVALID_SCHEMA')
    raw=memoryview(packet)[4+size:];values={};offset=0
    for entry in header['tensors']:
        shape=entry['shape'];name=entry['name']
        if (name in values or entry['dtype']!='<f4' or len(shape)>4
                or any(type(v)!=int or v<0 for v in shape)):
            raise Rejected('INVALID_TENSOR_SCHEMA')
        n=math.prod(shape)*4
        if entry['offset']!=offset or entry['bytes']!=n or offset+n>len(raw):raise Rejected('INVALID_TENSOR_RANGE')
        data=raw[offset:offset+n]
        if hashlib.sha256(data).hexdigest()!=entry['sha256']:raise Rejected('HASH_OR_SIGNATURE_FAIL')
        array=np.frombuffer(data,dtype='<f4').reshape(shape).copy()
        if not np.isfinite(array).all():raise Rejected('NONFINITE_PAYLOAD')
        values[name]=array;offset+=n
    if offset!=len(raw):raise Rejected('UNDECLARED_PAYLOAD')
    return header['metadata'],values
