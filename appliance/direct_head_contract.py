"""Direct-head pilot contract. Metadata authorizes a probe, never proves recall."""
import math
import numpy as np

from .config import Rejected
from .state import digest, boundary_hash, complete_hash

VERSION = 'appliance_direct_head_v1'
RULES = dict(eta_grid=[.25, .5, 1.], min_positive=32, min_negative=32,
    recall=.95, FAR=.001, negative_break=0, application_bytes=128*1024*1024,
    runtime_seconds=900, cap_per_class=512, seed=42, q=1., round=19,
    receiver_permission='unowned rank-0 reserve only; fixed-allocation gradient freeze is not maturity; task-wide FC2 freeze rejects')
FIXTURES = ((1, 3, 7, (54, 51)), (2, 1, 13, (74, 34)), (2, 3, 14, (74, 26)))


def architecture(model):
    return dict(type=f'{type(model).__module__}.{type(model).__qualname__}',
        input_shape=[model.seq_length, model.num_features],
        in_features=model.fc2.in_features, out_features=model.fc2.out_features,
        architecture_version=int(getattr(model, 'architecture_version', 1)))


def clean_endpoint(model):
    if (getattr(model, 'adapter_registry', {}) or getattr(model, 'active_adapters', {})
            or getattr(model, 'appliance_guarded_head_entries', {})
            or getattr(model, 'continual_head', None) is not None
            or getattr(model, 'local_classifier', None) is not None):
        raise Rejected('DIRECT_HEAD_AUXILIARY_FUNCTION_UNSUPPORTED')


def binding(model, router, owner, task, target, authority):
    clean_endpoint(model)
    if type(owner) is not int or owner < 0 or type(task) is not int or task < 0:
        raise Rejected('DIRECT_HEAD_ID_INVALID')
    if type(target) is not int or not 0 <= target < model.fc2.out_features:
        raise Rejected('DIRECT_HEAD_TARGET_INVALID')
    return dict(protocol=VERSION, donor=owner, task=task, class_id=target,
        architecture=architecture(model), authority=authority,
        rank=int(model.unit_ranks['fc2'][target]),
        boundary_hash=boundary_hash(model), function_hash=complete_hash(model, router))


def validate_packet(packet):
    m = packet['metadata']
    if m['protocol'] != VERSION or m['rank'] < 2:
        raise Rejected('DIRECT_HEAD_VERSION_OR_MATURITY')
    if type(m['donor']) is not int or m['donor'] < 0 or type(m['task']) is not int or m['task'] < 0:
        raise Rejected('DIRECT_HEAD_PACKET_ID')
    d = m['architecture']['in_features']
    c = m['class_id']
    if type(d) is not int or d <= 0 or type(c) is not int or not 0 <= c < m['architecture']['out_features']:
        raise Rejected('DIRECT_HEAD_PACKET_SHAPE')
    w, mask = np.asarray(packet['weight']), np.asarray(packet['mask'])
    if w.shape != (d,) or mask.shape != (d,) or not np.isfinite(w).all():
        raise Rejected('DIRECT_HEAD_PACKET_SHAPE_OR_FINITE')
    if not np.isin(mask, (0, 1)).all() or packet['bias_mask'] not in (0, 1):
        raise Rejected('DIRECT_HEAD_MASK_NOT_BINARY')
    if not math.isfinite(float(packet['bias'])):
        raise Rejected('DIRECT_HEAD_BIAS_NONFINITE')
    for name in ('boundary_hash', 'function_hash'):
        if not isinstance(m[name], str) or not m[name]:
            raise Rejected('DIRECT_HEAD_MISSING_FUNCTION_BINDING')
    return packet


def compatible(packets):
    if not packets:
        raise Rejected('DIRECT_HEAD_NO_PACKETS')
    for p in packets:
        validate_packet(p)
    keys = ('protocol', 'task', 'class_id', 'architecture', 'authority')
    signatures = {digest({k: p['metadata'][k] for k in keys}) for p in packets}
    donors = [p['metadata']['donor'] for p in packets]
    if len(signatures) != 1 or len(set(donors)) != len(donors):
        raise Rejected('DIRECT_HEAD_INCOMPATIBLE_OR_DUPLICATE')
    # Boundary equality is reported, not assumed: empirical shadow verification
    # must establish usefulness on the receiver's feature space.
    return packets[0]['metadata']
