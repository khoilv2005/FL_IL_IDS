"""Current BASE sketches computed by the receiver extractor and thresholds."""
from copy import deepcopy
from dataclasses import asdict
import numpy as np
import torch
from .snapshot import digest


@torch.no_grad()
def receiver_sketches(reference, inputs, labels, owner, *, batch_size, per_class):
    detector = reference.detector
    if detector.router_mode != 'binary_cosine' or not detector.binarize_thresholds:
        raise ValueError('Teaching requires initialized receiver binary_cosine thresholds')
    rows = []
    for c in sorted(set(labels.tolist())):
        rows.extend(torch.nonzero(labels == c).flatten()[:per_class].tolist())
    bits = []
    device = next(reference.model.parameters()).device
    for start in range(0, len(rows), batch_size):
        bits.append(detector._binarize_per_sample(reference.model, inputs[rows[start:start+batch_size]].to(device)))
    if not bits:
        raise ValueError('No current BASE router samples')
    values = np.concatenate(bits).astype(np.uint8)
    if not np.isin(values, (0, 1)).all():
        raise ValueError('Router sketches must be binary')
    return dict(owner=int(owner), identity=asdict(reference.identity), bits=values,
                class_counts={int(c):int((labels[rows] == c).sum()) for c in set(labels[rows].tolist())},
                thresholds_sha256=digest(detector.binarize_thresholds))


def stage_router(reference, targets, packets):
    detector = deepcopy(reference.detector)
    historical = {t:digest(v) for t,v in detector.activation_memory.items()
                  if int(t) != reference.identity.task}
    threshold_hash = digest(detector.binarize_thresholds)
    arrays, owners = [], set()
    for packet in packets:
        if packet['owner'] in owners:
            raise ValueError('Duplicate router participant')
        owners.add(packet['owner'])
        if packet['identity'] != asdict(reference.identity) or packet['thresholds_sha256'] != threshold_hash:
            raise ValueError('Sketches use another receiver feature/threshold version')
        values = np.asarray(packet['bits'])
        if values.ndim != 2 or not len(values) or not np.isin(values, (0,1)).all():
            raise ValueError('Invalid per-sample binary memory')
        arrays.append(values)
    if not arrays or len({a.shape[1] for a in arrays}) != 1:
        raise ValueError('No compatible router sketches')
    episode = reference.identity.task
    detector.activation_memory[episode] = np.concatenate(arrays).astype(np.float32)
    detector.context_masks[episode] = detector._get_context_mask(reference.model)
    detector.episode_classes[episode] = sorted(set(detector.episode_classes.get(episode, [])) | set(targets))
    detector.train_models(episode)
    detector.mark_router_fresh(task_id=episode, round_id=None)
    if digest(detector.binarize_thresholds) != threshold_hash:
        raise ValueError('Historical thresholds changed during teaching')
    if any(digest(detector.activation_memory[t]) != h for t,h in historical.items()):
        raise ValueError('Historical router memory changed')
    return detector
