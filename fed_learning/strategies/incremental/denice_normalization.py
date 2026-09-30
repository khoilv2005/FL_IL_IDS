"""Layerwise population BN calibration for plastic channels only, from local train data."""
import torch
from .denice_replay import inference_statistics


def balanced_calibration_inputs(inputs, labels, replay, *, limit=1024, seed=0):
    """Current + private historical classes with equal counts; no global RNG use."""
    labels = labels.detach().cpu().long()
    entries = getattr(replay, 'entries', {})
    classes = sorted(set(labels.tolist()) | set(entries))
    if not classes or not entries:
        return inputs
    if limit < len(classes):
        raise ValueError('BN calibration limit must cover all local classes')
    generator = torch.Generator().manual_seed(int(seed))
    quota = limit // len(classes)
    selected = []
    for cls in classes:
        current = torch.where(labels == cls)[0]
        # Select indices first; do not copy the client's entire dataset.
        if len(current):
            indices = current[torch.randint(len(current), (quota,), generator=generator)]
            selected.append(inputs[indices.to(inputs.device)].detach().cpu())
        else:
            old = entries[cls]['x']
            indices = torch.randint(len(old), (quota,), generator=generator)
            selected.append(old[indices].detach().cpu())
    return torch.cat(selected)


@torch.no_grad()
def calibrate_plastic_bn(model, inputs, *, limit=1024, batch_size=128, seed=0):
    if len(inputs) < 2:
        return {'sample_count': 0, 'updated_channels': {}}
    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(len(inputs), generator=generator)[:limit]
    values = inputs[indices]
    device = next(model.parameters()).device
    snapshots = {name: buffer.clone() for name, buffer in model.named_buffers()}
    audit = {}
    try:
        with inference_statistics(model):
            # Earlier BN layers use their new population statistics while
            # collecting the next layer; mature channels keep original stats.
            for layer, bn_name in model.BN_LAYER_MAP.items():
                bn = getattr(model, bn_name)
                plastic = torch.as_tensor(model.unit_ranks[layer] == 1, device=device)
                if not plastic.any():
                    continue
                total, square, count = None, None, 0

                def collect(module, args):
                    nonlocal total, square, count
                    value = args[0].double()
                    axes = (0,) + tuple(range(2, value.ndim))
                    first, second = value.sum(axes), value.square().sum(axes)
                    total = first if total is None else total + first
                    square = second if square is None else square + second
                    count += value.numel() // value.shape[1]

                handle = bn.register_forward_pre_hook(collect)
                try:
                    for start in range(0, len(values), batch_size):
                        model(values[start:start+batch_size].to(device))
                finally:
                    handle.remove()
                if count <= 1:
                    continue
                mean = total / count
                variance = ((square / count - mean.square()).clamp_min(0) * count / (count - 1))
                if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
                    raise ValueError('Nonfinite local BN calibration statistics')
                bn.running_mean[plastic] = mean.to(bn.running_mean)[plastic]
                bn.running_var[plastic] = variance.to(bn.running_var)[plastic]
                audit[layer] = int(plastic.sum())
    except Exception:
        for name, buffer in model.named_buffers():
            buffer.copy_(snapshots[name])
        raise
    return {'sample_count': len(values), 'updated_channels': audit}
