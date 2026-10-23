"""A receiver-only imported class route, calibrated separately from validation.

This experimental side branch does not relax the exact transplant installer.
The donor provides a head and a summary in the receiver's frozen feature space;
inference needs the receiver and the serialized packet only.
"""
from dataclasses import dataclass
import copy
import hashlib

import numpy as np
import torch

from .codec import decode, encode
from .closure import effective_linear
from .config import Rejected
from .state import boundary_hash
from .transfer_audit import features


ROUTE_RULES = dict(
    version='appliance_parallel_route_v1', seed=20261008,
    signature='unit centroid of L2-normalized adapter-free receiver penultimate features',
    calibration_split=[.50, .25, .25], min_fit_positives=32,
    min_selection_positives=8, min_holdout_positives=8,
    min_receiver_selection_rows=8, min_receiver_holdout_rows=8,
    tau_quantile_points=101, gamma_quantile_points=101,
    receiver_selection_max_false_activation=0.,
    receiver_holdout_max_accuracy_drop=0.,
    selection='maximize donor-positive recall, then minimize all-negative false activations; ties prefer larger gamma then tau',
    no_guard_ablation='same locked tau/prototype; gamma=0 (ordinary class-head competition)',
    margin_space='local supported classes in fixed imported context, not cross-context logits',
    max_patch_bytes=32768, no_bridge=True, no_neural_router=True,
    final_test_opened=False, no_install=True,
)


def unit_rows(values):
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-12
    return values / np.maximum(norms, 1e-12), valid


def stratified_roles(pool, seed):
    """Split each class independently. Rare classes are reported, not invented."""
    result = dict(fit=[], selection=[], holdout=[])
    rng = np.random.default_rng(seed)
    for label in sorted(np.unique(pool['y'])):
        indices = rng.permutation(np.flatnonzero(pool['y'] == label))
        n = len(indices)
        if n == 1:
            counts = (1, 0, 0)
        elif n == 2:
            counts = (1, 1, 0)
        else:
            fit = min(n - 2, max(1, n // 2))
            selection = min(n - fit - 1, max(1, n // 4))
            counts = (fit, selection, n - fit - selection)
        start = 0
        for role, count in zip(result, counts):
            result[role].extend(indices[start:start + count].tolist())
            start += count
    for role in result:
        result[role] = np.asarray(sorted(result[role]), dtype=np.int64)
    joined = np.concatenate(list(result.values()))
    if len(joined) != len(pool['y']) or len(np.unique(joined)) != len(joined):
        raise RuntimeError('Calibration split overlap or omission')
    return result


def build_signature(receiver_at_donor, donor_fit_inputs, batch_size, device):
    """Runs at donor with cached receiver encoder; no donor raw inputs exported."""
    h = features(receiver_at_donor, donor_fit_inputs, None, batch_size, device)
    normalized, valid = unit_rows(h)
    if int(valid.sum()) < ROUTE_RULES['min_fit_positives']:
        raise Rejected('INSUFFICIENT_SIGNATURE_SUPPORT')
    prototype = normalized[valid].mean(axis=0)
    norm = np.linalg.norm(prototype)
    if norm <= 1e-12 or not np.isfinite(prototype).all():
        raise Rejected('INVALID_IMPORTED_PROTOTYPE')
    return (prototype / norm).astype(np.float32), dict(
        input_rows=len(h), usable_rows=int(valid.sum()), feature_width=h.shape[1],
        zero_norm_rows=int((~valid).sum()), raw_input_exported=False,
    )


@dataclass
class ImportedRoute:
    metadata: dict
    prototype: np.ndarray
    head_weight: np.ndarray
    head_bias: float

    def to_packet(self):
        tensors = dict(prototype=self.prototype, head_weight=self.head_weight,
                       head_bias=np.asarray(self.head_bias, dtype=np.float32),
                       head_weight_mask=np.ones_like(self.head_weight),
                       head_bias_mask=np.asarray(1., dtype=np.float32))
        return encode(self.metadata, tensors, ROUTE_RULES['max_patch_bytes'])

    @classmethod
    def from_packet(cls, value):
        metadata, tensors = decode(value, ROUTE_RULES['max_patch_bytes'])
        if metadata.get('kind') != 'parallel_imported_route_experiment':
            raise Rejected('INVALID_IMPORTED_ROUTE_KIND')
        for key in ('tau', 'gamma'):
            if not np.isfinite(metadata[key]):
                raise Rejected('INVALID_IMPORTED_ROUTE_THRESHOLD', key)
        if metadata['gamma'] < 0:
            raise Rejected('NEGATIVE_MARGIN_GUARD')
        if set(tensors) != {'prototype', 'head_weight', 'head_bias', 'head_weight_mask', 'head_bias_mask'}:
            raise Rejected('INVALID_IMPORTED_ROUTE_TENSORS')
        if (tensors['head_bias'].shape != () or tensors['head_bias_mask'].shape != () or
                tensors['head_weight_mask'].shape != tensors['head_weight'].shape or
                not np.all(tensors['head_weight_mask'] == 1) or tensors['head_bias_mask'] != 1):
            raise Rejected('INVALID_IMPORTED_ROUTE_MASK')
        return cls(metadata, tensors['prototype'], tensors['head_weight'], float(tensors['head_bias']))

    def validate(self, model, seen):
        c = int(self.metadata['class_id'])
        if c not in seen or not 0 <= c < model.fc2.out_features:
            raise Rejected('IMPORTED_CLASS_OUTSIDE_SCOPE')
        if boundary_hash(model, True) != self.metadata['receiver_feature_hash']:
            raise Rejected('IMPORTED_ROUTE_ENCODER_DRIFT')
        if (self.prototype.shape != (model.fc1.out_features,) or
                self.head_weight.shape != (model.fc2.in_features,)):
            raise Rejected('IMPORTED_ROUTE_SHAPE_MISMATCH')
        if not np.isfinite(self.prototype).all() or not np.isfinite(self.head_weight).all():
            raise Rejected('NONFINITE_IMPORTED_ROUTE')
        if not np.isclose(np.linalg.norm(self.prototype), 1., rtol=1e-5, atol=1e-6):
            raise Rejected('NONUNIT_IMPORTED_PROTOTYPE')

    @torch.no_grad()
    def signals(self, model, router, inputs, seen, batch_size, device):
        """Label-blind inference, receiver + packet only. Returns no labels."""
        from fed_learning.training.denice_eval import (
            _denice_routed_logits_with_episodes, _mask_logits_to_classes,
        )
        self.validate(model, seen)
        task = int(self.metadata['task'])
        c = int(self.metadata['class_id'])
        if not router.episode_classes.get(task):
            raise Rejected('IMPORTED_CONTEXT_LOCAL_SUPPORT_UNAVAILABLE')
        allowed = sorted((set(map(int, router.episode_classes[task])) & set(seen)) - {c})
        if not allowed:
            raise Rejected('NO_LOCAL_CLASS_FOR_MARGIN_REFERENCE')
        modes = [(m, m.training) for m in model.modules()]
        active = copy.deepcopy(model.active_adapters)
        result = []
        local_weights, local_bias = effective_linear(model, 'fc2')
        has_import_adapter = any(int(meta['context_id']) == task
                                 for meta in model.adapter_registry.values())
        try:
            model.eval()
            for start in range(0, len(inputs), batch_size):
                x = torch.as_tensor(inputs[start:start + batch_size], dtype=torch.float32, device=device)
                legacy_logits, episode = _denice_routed_logits_with_episodes(
                    model, x, router, seen, device, inference_policy='pred_hard')
                model.clear_active_adapters()
                h = model.penultimate_features(x).cpu().numpy()
                normalized, valid = unit_rows(h)
                similarity = normalized @ self.prototype
                if has_import_adapter:
                    model.set_active_context(task)
                    imported = model.penultimate_features(x).cpu().numpy()
                else:
                    imported = h
                local = torch.nn.functional.linear(torch.as_tensor(imported), local_weights, local_bias)
                local_best = _mask_logits_to_classes(local, allowed).max(1).values.numpy()
                patch_logit = imported @ self.head_weight + self.head_bias
                margin = patch_logit - local_best
                if not (np.isfinite(similarity).all() and np.isfinite(margin).all()):
                    raise FloatingPointError('Nonfinite imported-route signals')
                result.append(dict(local_pred=legacy_logits.argmax(1).cpu().numpy(),
                    legacy_task=np.asarray(episode), signature_score=similarity,
                    signature_valid=valid, patch_logit=patch_logit,
                    local_best_import_context=local_best, margin=margin))
        finally:
            model.active_adapters = active
            for module, training in modes:
                module.training = training
        if not result:
            return {key:np.empty(0) for key in ('local_pred', 'legacy_task', 'signature_score',
                    'signature_valid', 'patch_logit', 'local_best_import_context', 'margin')}
        return {key:np.concatenate([part[key] for part in result]) for key in result[0]}

    def decisions(self, signals, guarded=True):
        """No y_true, donor model, or legacy-task prerequisite for override."""
        tau = float(self.metadata['tau'])
        gamma = float(self.metadata['gamma']) if guarded else 0.
        signature_hit = signals['signature_valid'] & (signals['signature_score'] > tau)
        activated = signature_hit & (signals['margin'] > gamma)
        pred = np.where(activated, int(self.metadata['class_id']), signals['local_pred'])
        return dict(pred=pred, signature_hit=signature_hit, activated=activated)


def select_thresholds(receiver_signals, receiver_y, donor_signals, donor_y, class_id):
    """Scalar feedback search on calibration-selection, never validation/holdout."""
    # Selection simulator aggregates counts only; it does not train on pooled raw x.
    scores = np.concatenate([receiver_signals['signature_score'], donor_signals['signature_score']])
    margins = np.concatenate([receiver_signals['margin'], donor_signals['margin']])
    valid = np.concatenate([receiver_signals['signature_valid'], donor_signals['signature_valid']])
    labels = np.concatenate([receiver_y, donor_y])
    receiver_rows = np.arange(len(labels)) < len(receiver_y)
    positive = labels == class_id
    if int(positive.sum()) < ROUTE_RULES['min_selection_positives']:
        raise Rejected('INSUFFICIENT_SELECTION_POSITIVES')
    if len(receiver_y) < ROUTE_RULES['min_receiver_selection_rows']:
        raise Rejected('INSUFFICIENT_RECEIVER_SELECTION')
    if np.any(receiver_y == class_id):
        raise Rejected('RECEIVER_IMPORTED_POSITIVES_UNEXPECTED')
    tau_values = np.unique(np.concatenate([[-1., 1.], np.quantile(
        scores, np.linspace(0, 1, ROUTE_RULES['tau_quantile_points']))]))
    gamma_values = np.unique(np.concatenate([[0.], np.maximum(0., np.quantile(
        margins, np.linspace(0, 1, ROUTE_RULES['gamma_quantile_points'])))]))
    feedback = []
    best = None
    for tau in tau_values:
        hit = valid & (scores > tau)
        for gamma in gamma_values:
            active = hit & (margins > gamma)
            tp = int((active & positive).sum())
            fp_receiver = int((active & receiver_rows).sum())
            fp_all = int((active & ~positive).sum())
            feasible = fp_receiver == 0  # locked zero receiver false activation gate
            row = dict(tau=float(tau), gamma=float(gamma), true_positive=tp,
                positive_rows=int(positive.sum()), false_positive_receiver=fp_receiver,
                receiver_rows=len(receiver_y), false_positive_all=fp_all, feasible=feasible)
            feedback.append(row)
            key = (tp, -fp_all, float(gamma), float(tau))
            if feasible and (best is None or key > best[0]):
                best = key, row
    if best is None:
        raise Rejected('NO_FEASIBLE_CALIBRATION_GATE')
    selected = dict(best[1], receiver_labels_used='negative feedback only',
                    donor_labels_used='current calibration class support',
                    threshold_grid_source='selection signals only; fixed quantile grid rule')
    return selected, feedback


def encoder_cache_inventory(model, preprocessing_hash):
    """Count a cold receiver-encoder transfer, not silently assume free caching."""
    tensors = {}
    integers = {}
    for name, value in model.state_dict().items():
        if name.startswith(('fc2.', 'adapters.', 'continual_head.')):
            continue
        if value.is_floating_point():
            tensors['state_' + name] = value.detach().cpu().numpy()
        else:
            integers[name] = value.detach().cpu().tolist()
    for family in ('weight_masks', 'bias_masks', 'gru_connection_masks'):
        for name, value in getattr(model, family, {}).items():
            if name != 'fc2':
                tensors[family + '_' + name] = value.detach().cpu().numpy()
    metadata = dict(kind='receiver_encoder_cache_inventory_NOT_patch',
        input_shape=[model.seq_length, model.num_features],
        feature_mode='adapter-free penultimate', preprocessing_hash=preprocessing_hash,
        structural_protection=model.structural_protection, integer_buffers=integers,
        receiver_feature_hash=boundary_hash(model, True))
    value = encode(metadata, tensors, 2**31 - 1)
    return dict(cold_encoder_bytes=len(value), cold_encoder_kib=len(value)/1024,
        cold_encoder_sha256=hashlib.sha256(value).hexdigest(),
        cache_existing_in_real_protocol_verified=False,
        note='offline simulator has the receiver model; full cold cache cost is reported separately from patch bytes')
