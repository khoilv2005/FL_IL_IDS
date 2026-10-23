"""Frozen receiver reference and canonical effective inference features."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import pickle
import numpy as np
import torch
from fed_learning.training.checkpoint_state import snapshot_denice_state
from .contracts import SnapshotIdentity


def digest(value):
    h = hashlib.sha256()
    def add(item):
        if isinstance(item, torch.Tensor):
            value = item.detach().cpu().contiguous()
            h.update(str((str(value.dtype), tuple(value.shape))).encode())
            h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            h.update(str((str(item.dtype), item.shape)).encode())
            h.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            h.update(b'dict')
            for key in sorted(item, key=lambda k: (type(k).__name__, str(k))):
                add(key); add(item[key])
        elif isinstance(item, (list, tuple)):
            h.update(str((type(item).__name__, len(item))).encode())
            for child in item:
                add(child)
        else:
            h.update(pickle.dumps(item, protocol=5))
    add(value)
    return h.hexdigest()


def reference_payload(model, detector):
    algorithm = snapshot_denice_state(model, detector)
    # Session audit/ownership metadata do not define the mathematical predictor.
    algorithm.pop('candle_v3_state', None)
    return dict(model_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                algorithm_state=algorithm, input_shape=(model.seq_length, model.num_features),
                num_classes=int(model.num_classes))


@dataclass
class FrozenReceiver:
    model: object
    detector: object
    identity: SnapshotIdentity
    payload: dict


def freeze_receiver(model, detector, receiver, task, preprocessing_sha256, device):
    cloned = deepcopy(model).to(device)
    cloned.set_active_context(int(task))
    cloned.eval()
    for parameter in cloned.parameters():
        parameter.requires_grad_(False)
    if getattr(cloned, 'continual_head', None) is not None:
        raise ValueError('Peer head teaching requires the frozen V1 classifier contract')
    route = deepcopy(detector)
    payload = reference_payload(cloned, route)
    identity = SnapshotIdentity(int(receiver), int(task), digest(payload), preprocessing_sha256)
    return FrozenReceiver(cloned, route, identity, payload)


def canonical_features(model, inputs):
    if model.training or any(module.training for module in model.modules()):
        raise ValueError('Canonical teaching features require eval mode, including BN/dropout')
    return model.penultimate_features(inputs)


@torch.no_grad()
def encode_panel(reference, inputs, labels, owner, batch_size):
    model = reference.model
    device = next(model.parameters()).device
    features, logits = [], []
    for start in range(0, len(labels), batch_size):
        x = inputs[start:start+batch_size].to(device)
        z = canonical_features(model, x)
        out = model._apply_masked_linear(z, model.fc2, 'fc2')
        if not torch.isfinite(z).all() or not torch.isfinite(out).all():
            raise FloatingPointError('Nonfinite frozen receiver feature/logit')
        features.append(z.cpu()); logits.append(out.cpu())
    return dict(owner=int(owner), identity=asdict(reference.identity),
                features=torch.cat(features) if features else torch.empty(0, model.fc2.in_features),
                logits=torch.cat(logits) if logits else torch.empty(0, model.num_classes),
                labels=labels.detach().cpu().long())
