"""Explicit class-mask policies for self-only DeNICE inference.

`local` preserves legacy evaluation. `global_task` is an experimental coverage
variant: the *predicted* native task selects its globally declared seen classes.
It does not claim ownership, import weights, add profiles, or certify safety.
"""
from __future__ import annotations

import copy


def detector_with_class_mask_policy(detector, policy, task_classes, seen_classes, num_classes):
    if policy == 'local':
        return detector
    if policy != 'global_task':
        raise ValueError(f'Unknown DeNICE class-mask policy: {policy}')
    if not task_classes:
        raise ValueError('global_task requires the training task-class mapping')
    seen = {int(c) for c in seen_classes}
    if any(c < 0 or c >= num_classes for c in seen):
        raise ValueError('Invalid seen class ID')
    mapping = {int(task): sorted({int(c) for c in classes}) for task, classes in task_classes.items()}
    declared = [c for classes in mapping.values() for c in classes]
    if len(set(declared)) != len(declared) or any(c < 0 or c >= num_classes for c in declared):
        raise ValueError('Task-class mapping must contain unique, valid class IDs')
    if not seen.issubset(declared):
        raise ValueError('Seen classes missing from the declared task mapping')
    episodes = {int(task) for task in detector.episode_classes}
    episodes.update(int(task) for task, memory in detector.activation_memory.items() if len(memory))
    if not episodes.issubset(mapping):
        raise ValueError('Native task lacks a declared global class mapping')
    if any(not seen.intersection(mapping[task]) for task in episodes):
        raise ValueError('Cannot open a task containing only unseen classes')
    # Preserve the router episode-key set, including native fallback entries.
    # New profiles must never be synthesized by a mask-only policy.
    shadow = copy.deepcopy(detector)
    shadow.episode_classes = {task: sorted(seen.intersection(mapping[task])) for task in episodes}
    if max(shadow.episode_classes, default=0) != max(detector.episode_classes, default=0):
        raise ValueError('Mask policy would change the router episode index range')
    return shadow


def routed_self_with_class_mask_policy(model, inputs, detector, seen_classes, device,
                                      *, class_mask_policy='local', task_classes=None):
    """Predict from x only: no y_true, oracle task or donor models accepted."""
    from fed_learning.training.denice_eval import _denice_routed_logits_with_episodes
    selected = detector_with_class_mask_policy(detector, class_mask_policy, task_classes,
                                               seen_classes, int(model.num_classes))
    return _denice_routed_logits_with_episodes(model, inputs, selected, seen_classes, device,
                                               inference_policy='pred_hard')
