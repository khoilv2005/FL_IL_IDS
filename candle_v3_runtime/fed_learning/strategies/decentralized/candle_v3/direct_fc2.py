"""Direct absolute FC2 row packets; no donor optimizer or receiver-row averaging."""
from dataclasses import asdict
import numpy as np
import torch
from .snapshot import digest

ROW_SCHEMA = 'candle_v3_direct_fc2_row_v1'


class RowCompatibilityError(ValueError):
    """Expected candidate incompatibility, not a corrupted packet."""


def export_row(model, *, donor, task, class_id, announcement):
    c = int(class_id)
    if (announcement.client != int(donor) or announcement.task != int(task) or
            c not in announcement.current_classes or c not in announcement.local_seen_classes):
        raise ValueError('Row export requires actual current local training provenance')
    record = getattr(model, 'candle_v3_state', {})
    local = record.get('local_seen', {}).get(str(int(task)), {}).get(str(c), {})
    if local.get('positive_optimizer_rounds', 0) <= 0 or model.unit_ranks['fc2'][c] < 2:
        raise ValueError('Donor row has no current successful optimizer evidence or is not mature')
    weight = model.fc2.weight[c].detach().cpu().float().contiguous()
    bias = model.fc2.bias[c].detach().cpu().float().reshape(1)
    wm = model.weight_masks['fc2'][c].detach().cpu().numpy()
    bm = float(model.bias_masks['fc2'][c])
    if not np.isin(wm, (0,1)).all() or bm not in (0.,1.):
        raise ValueError('Direct row packet requires binary connection masks')
    values = torch.cat([weight,bias]).numpy().astype('<f4').tobytes()
    if not torch.isfinite(torch.cat([weight,bias])).all():
        raise FloatingPointError('Nonfinite donor row')
    packet = dict(schema=ROW_SCHEMA, donor=int(donor), task=int(task), class_id=c,
        architecture=announcement.architecture, preprocessing_sha256=announcement.preprocessing_sha256,
        input_dim=weight.numel(), row_fp32=values,
        weight_mask=np.packbits(wm.astype(np.uint8), bitorder='little').tobytes(), bias_mask=int(bm),
        donor_rank=int(model.unit_ranks['fc2'][c]),
        donor_feature_sha256=digest(dict(
            tensors={k:v for k,v in model.state_dict().items() if not k.startswith('fc2.')},
            masks={k:v for k,v in model.get_masks_state().items() if k not in ('weight_fc2','bias_fc2')},
            bn_frozen={name:getattr(model,name,{}) for name in
                       ('_bn_frozen_units','_bn_running_mean_frozen','_bn_running_var_frozen')},
            structural_protection=bool(model.structural_protection),
            task=int(task), adapter_mode=model.adapter_mode,
            adapter_registry=model.get_adapter_registry_state())),
        local_evidence=dict(local), source_row_sha256=digest([weight,bias,wm,bm]))
    packet['packet_sha256'] = digest(packet)
    return packet


def decode_row(packet):
    if packet.get('schema') != ROW_SCHEMA:
        raise ValueError('Unknown direct-row schema')
    body = {k:v for k,v in packet.items() if k != 'packet_sha256'}
    if digest(body) != packet.get('packet_sha256'):
        raise ValueError('Direct row packet checksum mismatch')
    width = packet['input_dim']
    if not isinstance(width, int) or width <= 0 or len(packet['row_fp32']) != (width+1)*4:
        raise ValueError('Invalid FP32 row size')
    if len(packet['weight_mask']) != (width+7)//8 or packet['bias_mask'] not in (0,1):
        raise ValueError('Invalid packed connection masks')
    values = torch.from_numpy(np.frombuffer(packet['row_fp32'],dtype='<f4').copy())
    mask = torch.from_numpy(np.unpackbits(np.frombuffer(packet['weight_mask'],dtype=np.uint8),
                                        bitorder='little')[:width].copy()).bool()
    if not torch.isfinite(values).all() or packet['donor_rank'] < 2:
        raise ValueError('Invalid/missing mature donor row')
    return values, mask


def aggregate_rows(reference, targets, packets, *, quality):
    """Normalize by contributors per coordinate; receiver is never a donor.

    Single-donor rows and masks are copied exactly, with no random receiver blend.
    For multi-donor aggregation, without a contributor preserve receiver state. Reject rather
    than silently clipping connections onto unconsolidated receiver features.
    Native-neighborhood membership is checked by the service, not assumed safe.
    """
    model = reference.model
    rows, masks, bias_masks, provenance = [], [], [], {}
    for c in targets:
        if model.unit_ranks['fc2'][c] != 0:
            raise ValueError('Direct transfer cannot overwrite an allocated receiver row')
        contributors = [p for p in packets if p['class_id'] == c]
        if not contributors or len({p['donor'] for p in contributors}) != len(contributors):
            raise ValueError('Missing or duplicate direct-row contributor')
        if len(contributors) > 1 and len({p['donor_feature_sha256'] for p in contributors}) != 1:
            raise RowCompatibilityError('Multi-donor rows have different feature frames; no alignment evidence')
        weight_sum = torch.zeros(model.fc2.in_features)
        normalizer = torch.zeros_like(weight_sum)
        bias_sum, bias_normalizer = 0.,0.
        donors = []
        for packet in contributors:
            if (packet['donor'] == reference.identity.receiver or packet['task'] != reference.identity.task or
                packet['preprocessing_sha256'] != reference.identity.preprocessing_sha256 or
                packet['architecture'] != f'denice:{getattr(model,"architecture_version",1)}:fc2={tuple(model.fc2.weight.shape)}' or
                packet['input_dim'] != model.fc2.in_features):
                raise ValueError('Incompatible direct row reference')
            values, mask = decode_row(packet)
            q = float(quality.get((packet['donor'],c), 0.))
            if not np.isfinite(q) or not 0 < q <= 1:
                raise ValueError('Positive current FIT quality required; missing evidence is not a weight')
            weight_sum += q*mask*values[:-1]
            normalizer += q*mask
            if packet['bias_mask']:
                bias_sum += q*float(values[-1]); bias_normalizer += q
            donors.append(dict(donor=packet['donor'], quality=q, packet_sha256=packet['packet_sha256'],
                               feature_sha256=packet['donor_feature_sha256']))
        keep = normalizer > 0
        result = model.fc2.weight[c].detach().cpu().float().clone()
        permission = model.weight_masks['fc2'][c].detach().cpu().bool().clone()
        if len(contributors) == 1:
            values, donor_mask = decode_row(contributors[0])
            result = values[:-1].clone()  # exact absolute row, including inactive coordinates
            permission = donor_mask.clone()
        else:
            result[keep] = weight_sum[keep]/normalizer[keep]
            permission[keep] = True
        mature = torch.as_tensor(model.unit_ranks['fc1'] >= 2)
        if bool((permission & (result != 0) & ~mature).any()):
            raise RowCompatibilityError('Direct row would depend on unprotected receiver feature coordinates')
        if len(contributors) == 1:
            bias = float(decode_row(contributors[0])[0][-1])
            bias_mask = contributors[0]['bias_mask']
        else:
            bias = bias_sum/bias_normalizer if bias_normalizer else float(model.fc2.bias[c])
            bias_mask = 1 if bias_normalizer else int(model.bias_masks['fc2'][c])
        rows.append(torch.cat([result,torch.tensor([bias])]))
        masks.append(permission); bias_masks.append(bias_mask); provenance[c] = donors
    return dict(head=torch.stack(rows), permissions=torch.stack(masks), bias_masks=bias_masks,
                targets=list(targets), history=[], include_receiver=False,
                provenance=provenance, receiver_row_weight=0., optimizer_steps=0,
                identity=asdict(reference.identity), mode='direct_fc2')
