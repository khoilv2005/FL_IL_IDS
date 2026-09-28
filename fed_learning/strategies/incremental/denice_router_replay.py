"""Rebuild all local episode sketches in one current feature space from replay."""
from copy import deepcopy
import numpy as np
import torch

from .denice_replay import inference_statistics


def router_replay_config(config):
    result = {'enabled': bool(config.get('denice_router_replay_enabled', False))}
    for key, default in [('per_class', 64), ('batch_size', 128)]:
        value = config.get('denice_router_replay_' + key, default)
        if isinstance(value, bool) or int(value) != value or value <= 0:
            raise ValueError('Invalid denice_router_replay_' + key)
        result[key] = int(value)
    if result['enabled'] and (config.get('denice_memory_policy') != 'local_replay'
                              or int(config.get('denice_replay_capacity', 0)) <= 0
                              or config.get('denice_shared_context_eval', False)):
        raise ValueError('Router replay requires private local_replay and local context evaluation')
    return result


@torch.no_grad()
def refresh_replay_router(detector, model, memory, x, y, task_id, controls, *, seed=0, round_id=None):
    """Atomic refresh; never mix old sketches with a newly calibrated encoder.

    A joining client without exemplars for inherited episodes keeps its old
    detector. There is no transfer of raw memory or invention of missing data.
    The original structural-protection mask is preserved for CANC recycling.
    """
    episodes = sorted(set(detector.activation_memory) | {int(task_id)})
    if len(episodes) <= 1:
        return {'updated': False, 'reason': 'single_episode'}
    generator = torch.Generator().manual_seed(int(seed))
    entries = {} if memory is None else memory.entries
    inputs = {}
    for episode in episodes:
        chunks = []
        for label in detector.episode_classes.get(episode, []):
            if episode == task_id:
                indices = torch.nonzero(y.detach().cpu() == int(label), as_tuple=False).flatten()
                selected = indices[torch.randperm(len(indices), generator=generator)[:controls['per_class']]]
                if len(selected):
                    chunks.append(x[selected].detach().cpu())
            elif int(label) in entries:
                values = entries[int(label)]['x']
                selected = torch.randperm(len(values), generator=generator)[:controls['per_class']]
                if len(selected):
                    chunks.append(values[selected].detach().cpu())
        if not chunks:
            return {'updated': False, 'reason': 'missing_episode_replay', 'episode': episode}
        inputs[episode] = torch.cat(chunks)
    layers = ('conv1', 'conv2', 'conv3', 'gru')
    mask = np.concatenate([np.asarray(model.unit_ranks[layer]) > 0 for layer in layers])
    if not mask.any():
        raise ValueError('Replay router requires allocated feature units')
    device = next(model.parameters()).device
    acts = {}
    with inference_statistics(model):
        for episode, values in inputs.items():
            chunks = {layer: [] for layer in layers}
            for start in range(0, len(values), controls['batch_size']):
                features = model.get_context_activations_per_sample(values[start:start+controls['batch_size']].to(device))
                for layer in layers:
                    chunks[layer].append(features[layer].detach().float().cpu().numpy())
            acts[episode] = {layer: np.concatenate(chunks[layer]) for layer in layers}
    # Fit a single calibration using the same bounded sample bank as all sketches.
    thresholds, offset = {}, 0
    for layer in layers:
        values = np.concatenate([features[layer] for features in acts.values()])
        selected = values[:, mask[offset:offset + values.shape[1]]]
        offset += values.shape[1]
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite features in replay router')
        thresholds[layer] = float(selected.mean() + selected.std()) if selected.size else 0.
    candidate = deepcopy(detector)
    candidate.routing_feature_mask = mask
    candidate.binarize_thresholds = thresholds
    candidate.activation_memory = {episode: candidate.binarize_layer_activations(features)
                                   for episode, features in acts.items()}
    candidate.context_masks = {episode: mask.copy() for episode in acts}
    candidate.retain_reference_inputs = False
    candidate.reference_input_memory = {}
    candidate.calibration_provenance = 'private_replay_current_encoder'
    candidate.train_models(max(episodes))
    candidate.mark_router_fresh(task_id=task_id, round_id=round_id)
    detector.__dict__.update(candidate.__dict__)
    return {'updated': True, 'reason': 'all_local_episodes_reencoded',
            'episode_samples': {episode: len(values) for episode, values in inputs.items()},
            'feature_count': int(mask.sum())}
