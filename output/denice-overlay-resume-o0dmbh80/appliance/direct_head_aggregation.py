"""Absolute effective weights, with a separate contributor/application mask."""
import numpy as np
from .config import Rejected
from .direct_head_contract import compatible


def aggregate(packets, alphas):
    metadata = compatible(packets)
    d = metadata['architecture']['in_features']
    numerator, denominator = np.zeros(d+1, np.float64), np.zeros(d+1, np.float64)
    mix = []
    for p in packets:
        donor = p['metadata']['donor']
        a = float(alphas[donor])  # q=1 fixed primary, no HOLDOUT quality tuning.
        if not np.isfinite(a) or a <= 0:
            raise Rejected('DIRECT_HEAD_ALPHA_INVALID')
        masks = np.append(p['mask'], p['bias_mask']).astype(np.float64)
        values = np.append(p['weight'], p['bias']).astype(np.float64)
        numerator += a*masks*values
        denominator += a*masks
        mix.append(dict(donor=donor, alpha=a, q=1., contributed=int(masks.sum())))
    values = np.zeros(d+1, np.float64)
    np.divide(numerator, denominator, out=values, where=denominator > 0)
    if not np.isfinite(values).all():
        raise Rejected('DIRECT_HEAD_AGGREGATE_NONFINITE')
    return dict(weight=values[:-1].astype(np.float32), bias=float(np.float32(values[-1])),
        contributed=denominator[:-1] > 0, bias_contributed=bool(denominator[-1] > 0),
        denominator=denominator, mix=mix, metadata=metadata)
