"""Shared deterministic routing sketches and a calibrated self-protection gate.

No learned/shared backbone is sent to construct a sketch prototype. This module
is an offline, receiver-only inference experiment; it is not an installer.
"""
from dataclasses import dataclass
import copy
import hashlib

import numpy as np
import torch

from .codec import decode, encode
from .closure import effective_linear
from .config import Rejected
from .imported_route import unit_rows
from .state import boundary_hash


PORTABLE_RULES = dict(version='appliance_shared_sketch_protection_v1', seed=20261008,
    dimensions=[16,32,64,128], receiver_reference_width=256, quantile_grid_points=31,
    sketch='fixed seeded Rademacher projection of preprocessed flattened input; prefix dimensions',
    shared_projection_seed=20261008, projection_algorithm='numpy PCG64 integers -> {-1,+1} FP32',
    prototype='normalized mean of unit donor calibration-FIT sketches',
    variance='elementwise variance of unit FIT sketches; descriptive payload, not scored in V1',
    self_confidence='max softmax of legacy-routed local logits; float64; temperature=1',
    activation='cosine>tau AND same-import-context head margin>gamma AND local confidence<beta',
    receiver_selection_far_budget=.001, receiver_holdout_far_budget=.001,
    beta_range=[0.,1.], gamma_min=0., signature_dtype='float32', max_patch_bytes=32768,
    anchor_control='receiver reference with V1 selection-derived tau/gamma fixed; choose beta only from selection',
    selection='maximize donor-positive activation; then fewer negative activations; then smaller beta, larger gamma, larger tau',
    candidate_choice='selection only: maximize positive activations, fewer all-negative activations, smaller packet bytes; tie name',
    no_retrain=True, no_install=True, final_test_opened=False,
    validation_is_development=True, calibration_holdout_previously_observed=True)


def transitions(y, baseline, pred):
    y, baseline, pred = map(np.asarray, (y,baseline,pred))
    if y.shape != baseline.shape or y.shape != pred.shape:
        raise ValueError('Accounting row alignment changed')
    b, p = baseline == y, pred == y
    changed = baseline != pred
    cells = dict(rescue=int((~b&p).sum()), break_count=int((b&~p).sum()),
        wrong_wrong=int((~b&~p).sum()), correct_correct=int((b&p).sum()))
    if sum(cells.values()) != len(y):
        raise RuntimeError('Transition partition is not exhaustive')
    delta = int(p.sum()) - int(b.sum())
    if delta != cells['rescue'] - cells['break_count']:
        raise RuntimeError('Accuracy / rescue / break identity failed')
    return dict(**cells, rows=len(y), before_correct=int(b.sum()), after_correct=int(p.sum()),
        before_accuracy=float(b.mean()) if len(y) else None,
        after_accuracy=float(p.mean()) if len(y) else None,
        delta_correct=delta, delta_accuracy=delta/len(y) if len(y) else None,
        identity_exact=True, changed_wrong=int((changed&~b&~p).sum()),
        harmless_changed=int((changed&b&p).sum()), unchanged_rows=int((~changed).sum()))


@dataclass(frozen=True)
class SharedSketch:
    input_shape: tuple
    dimension: int
    preprocessing_sha256: str
    seed: int = PORTABLE_RULES['shared_projection_seed']

    def matrix(self):
        if self.dimension not in PORTABLE_RULES['dimensions']:
            raise Rejected('UNSUPPORTED_SKETCH_DIMENSION')
        rng = np.random.Generator(np.random.PCG64(self.seed))
        # All four experiments use prefixes of the same fixed matrix.
        bits = rng.integers(0,2,size=(int(np.prod(self.input_shape)),max(PORTABLE_RULES['dimensions'])),dtype=np.int8)
        return (bits[:,:self.dimension].astype(np.float32)*2-1)/np.sqrt(np.float32(self.dimension))

    def manifest(self):
        matrix = self.matrix()
        return dict(kind='shared_sketch', input_shape=list(self.input_shape), dimension=self.dimension,
            seed=self.seed, preprocessing_sha256=self.preprocessing_sha256,
            projection_algorithm=PORTABLE_RULES['projection_algorithm'],
            projection_sha256=hashlib.sha256(matrix.tobytes()).hexdigest(),
            local_generated_matrix_bytes=matrix.nbytes, encoder_transfer_bytes=0,
            input_width=int(np.prod(self.input_shape)))

    def features(self, inputs):
        inputs = np.asarray(inputs,dtype=np.float32)
        if tuple(inputs.shape[1:]) != self.input_shape or not np.isfinite(inputs).all():
            raise Rejected('SHARED_SKETCH_INPUT_MISMATCH')
        projected = inputs.reshape(len(inputs),int(np.prod(self.input_shape))) @ self.matrix()
        return unit_rows(projected)


def prototype_summary(features, valid):
    if int(valid.sum()) < 32:
        raise Rejected('INSUFFICIENT_PORTABLE_SIGNATURE_SUPPORT')
    z = features[valid]
    mean = z.mean(axis=0)
    norm = np.linalg.norm(mean)
    if not np.isfinite(mean).all() or norm <= 1e-12:
        raise Rejected('INVALID_PORTABLE_PROTOTYPE')
    prototype = (mean/norm).astype(np.float32)
    variance = z.var(axis=0).astype(np.float32)
    radius = float(np.quantile(1-z@prototype,.95))
    return prototype, variance, dict(usable_rows=len(z), zero_norm_rows=int((~valid).sum()), fit_radius_p95=radius)


def fit_support_cosine_floor(support):
    """Predeclare the donor-FIT 95% radius; no selection/old-data tuning.

    This bounds the prototype region. It is not an old-class FAR certificate.
    """
    radius=float(support['fit_radius_p95'])
    if int(support['usable_rows'])<32 or not np.isfinite(radius) or not -1e-5<=radius<=2.00001:
        raise Rejected('INVALID_FIT_SUPPORT_ENVELOPE')
    return float(np.clip(1-radius,-1.,1.))


@torch.no_grad()
def receiver_signals(model, router, inputs, seen, task, class_id, batch_size, device):
    """Shared head/local signals; no labels, donor or routing signature needed."""
    from fed_learning.training.denice_eval import _denice_routed_logits_with_episodes, _mask_logits_to_classes
    modes = [(m,m.training) for m in model.modules()]
    active = copy.deepcopy(model.active_adapters)
    allowed = sorted((set(map(int,router.episode_classes.get(task,[]))) & set(seen)) - {class_id})
    if not allowed:
        raise Rejected('NO_LOCAL_MARGIN_REFERENCE')
    local_weights, bias = effective_linear(model,'fc2')
    has_adapter = any(int(meta['context_id']) == task for meta in model.adapter_registry.values())
    parts = []
    try:
        model.eval()
        for start in range(0,len(inputs),batch_size):
            x = torch.as_tensor(inputs[start:start+batch_size],dtype=torch.float32,device=device)
            logits, ep = _denice_routed_logits_with_episodes(model,x,router,seen,device,inference_policy='pred_hard')
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Nonfinite local logits')
            confidence = torch.softmax(logits.double(),dim=1).max(1).values.cpu().numpy()
            model.clear_active_adapters()
            h = model.penultimate_features(x).cpu().numpy()
            normalized, valid = unit_rows(h)
            if has_adapter:
                model.set_active_context(task)
                imported = model.penultimate_features(x).cpu().numpy()
            else:
                imported = h
            local = torch.nn.functional.linear(torch.as_tensor(imported),local_weights,bias)
            best = _mask_logits_to_classes(local,allowed).max(1).values.numpy()
            parts.append(dict(local_pred=logits.argmax(1).cpu().numpy(), legacy_task=np.asarray(ep),
                local_confidence=confidence, receiver_signature_features=normalized,
                receiver_signature_valid=valid, imported_features=imported,
                local_best_import_context=best, local_unmasked_logits=local.numpy()))
    finally:
        model.active_adapters = active
        for module,flag in modes:
            module.training = flag
    if not parts:
        raise Rejected('EMPTY_PROTECTED_ROUTE_INPUT')
    return {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}


@dataclass
class ProtectedRoute:
    metadata: dict
    prototype: np.ndarray
    variance: np.ndarray
    head_weight: np.ndarray
    head_bias: float

    def packet(self):
        return encode(self.metadata,dict(prototype=self.prototype,variance=self.variance,
            head_weight=self.head_weight,head_bias=np.asarray(self.head_bias,np.float32),
            head_weight_mask=np.ones_like(self.head_weight),head_bias_mask=np.asarray(1.,np.float32)),
            PORTABLE_RULES['max_patch_bytes'])

    @classmethod
    def from_packet(cls, value):
        metadata,tensors = decode(value,PORTABLE_RULES['max_patch_bytes'])
        required = {'prototype','variance','head_weight','head_bias','head_weight_mask','head_bias_mask'}
        if metadata.get('kind') != 'protected_imported_route' or set(tensors) != required:
            raise Rejected('INVALID_PROTECTED_ROUTE_PACKET')
        if (tensors['head_bias'].shape != () or tensors['head_bias_mask'].shape != () or
                tensors['head_bias_mask'] != 1 or not np.all(tensors['head_weight_mask'] == 1) or
                tensors['head_weight_mask'].shape != tensors['head_weight'].shape):
            raise Rejected('INVALID_PROTECTED_HEAD_MASK')
        if not (np.isfinite([metadata[k] for k in ('tau','gamma','beta')]).all() and
                -1 <= metadata['tau'] <= 1 and metadata['gamma'] >= 0 and 0 <= metadata['beta'] <= 1):
            raise Rejected('INVALID_PROTECTED_THRESHOLDS')
        scope=metadata.get('margin_reference_scope','import_context')
        if scope not in ('import_context','all_locally_mature'):raise Rejected('UNKNOWN_MARGIN_REFERENCE_SCOPE')
        if scope=='all_locally_mature':
            references=metadata.get('margin_reference_classes',[])
            if (not references or len(references)!=len(set(references)) or metadata['class_id'] in references
                    or any(type(c) is not int or c<0 or c>=34 for c in references)):
                raise Rejected('INVALID_LOCAL_MARGIN_REFERENCE')
        if 'fit_support_min_cosine' in metadata:
            floor=fit_support_cosine_floor(metadata['support'])
            if metadata['fit_support_min_cosine']!=floor or metadata['tau']<floor:
                raise Rejected('FIT_SUPPORT_ENVELOPE_WEAKENED')
        return cls(metadata,tensors['prototype'],tensors['variance'],tensors['head_weight'],float(tensors['head_bias']))

    def validate(self, model, preprocessing_sha256, seen):
        if self.metadata['class_id'] not in seen or boundary_hash(model,True) != self.metadata['receiver_feature_hash']:
            raise Rejected('PROTECTED_HEAD_BOUNDARY_MISMATCH')
        if self.metadata.get('margin_reference_scope')=='all_locally_mature':
            classes=self.metadata['margin_reference_classes']
            if not set(classes).issubset(seen) or any(model.unit_ranks['fc2'][c]<2 for c in classes):
                raise Rejected('LOCAL_MARGIN_REFERENCE_CHANGED')
        signature = self.metadata['signature']
        if signature['kind'] == 'shared_sketch':
            sketch = SharedSketch(tuple(signature['input_shape']),signature['dimension'],preprocessing_sha256,signature['seed'])
            if sketch.manifest() != signature:
                raise Rejected('SHARED_SKETCH_VERSION_MISMATCH')
            width = signature['dimension']
        elif signature['kind'] == 'receiver_space':
            width = model.fc1.out_features
        else:
            raise Rejected('UNSUPPORTED_PROTECTED_SIGNATURE')
        if (self.prototype.shape != (width,) or self.variance.shape != (width,) or
                self.head_weight.shape != (model.fc2.in_features,) or (self.variance < 0).any() or
                not np.isclose(np.linalg.norm(self.prototype),1,rtol=1e-5,atol=1e-6)):
            raise Rejected('INVALID_PROTECTED_SIGNATURE_SHAPE')

    def signals(self, base, inputs):
        """Label-blind. Serialized packet + receiver signals + x, no donor model."""
        signature = self.metadata['signature']
        if signature['kind'] == 'shared_sketch':
            sketch = SharedSketch(tuple(signature['input_shape']),signature['dimension'],
                                  signature['preprocessing_sha256'],signature['seed'])
            z,valid = sketch.features(inputs)
        else:
            z,valid = base['receiver_signature_features'],base['receiver_signature_valid']
        score = np.clip(z@self.prototype,-1.,1.)
        patch_logit = base['imported_features']@self.head_weight + self.head_bias
        reference=base['local_best_import_context']
        scope=self.metadata.get('margin_reference_scope','import_context')
        if scope=='all_locally_mature':
            classes=self.metadata['margin_reference_classes']
            logits=np.asarray(base['local_unmasked_logits'])
            if (not classes or len(classes)!=len(set(classes)) or self.metadata['class_id'] in classes
                    or any(not isinstance(c,int) or c<0 or c>=logits.shape[1] for c in classes)):
                raise Rejected('INVALID_LOCAL_MARGIN_REFERENCE')
            reference=logits[:,classes].max(1)
        elif scope!='import_context':raise Rejected('UNKNOWN_MARGIN_REFERENCE_SCOPE')
        margin = patch_logit - reference
        if not np.isfinite(margin).all() or not np.isfinite(score).all():
            raise FloatingPointError('Nonfinite protected route signals')
        return dict(local_pred=base['local_pred'], local_confidence=base['local_confidence'],
            signature_score=score, signature_valid=valid, margin=margin, patch_logit=patch_logit)

    def decisions(self, signals, self_protection=True):
        floor=self.metadata.get('fit_support_min_cosine')
        if floor is not None and (not np.isfinite(floor) or not -1<=floor<=1 or self.metadata['tau']<floor):
            raise Rejected('FIT_SUPPORT_ENVELOPE_WEAKENED')
        route = signals['signature_valid'] & (signals['signature_score'] > self.metadata['tau'])
        margin = signals['margin'] > self.metadata['gamma']
        uncertain = signals['local_confidence'] < self.metadata['beta']
        active = route & margin & (uncertain if self_protection else True)
        return dict(pred=np.where(active,self.metadata['class_id'],signals['local_pred']),activated=active,
            signature_hit=route, margin_hit=margin, self_gate_hit=uncertain)


def choose_protection(receiver, receiver_y, donor, donor_y, class_id):
    """Selection-only constrained grid. Does not accept validation/holdout."""
    score = np.concatenate([receiver['signature_score'],donor['signature_score']])
    margin = np.concatenate([receiver['margin'],donor['margin']])
    confidence = np.concatenate([receiver['local_confidence'],donor['local_confidence']])
    valid = np.concatenate([receiver['signature_valid'],donor['signature_valid']])
    y = np.concatenate([receiver_y,donor_y])
    pos = y == class_id
    if pos.sum() < 8 or len(receiver_y) < 8 or np.any(receiver_y == class_id):
        raise Rejected('INSUFFICIENT_PROTECTED_SELECTION')
    rmask = np.arange(len(y)) < len(receiver_y)
    q = np.linspace(0,1,PORTABLE_RULES['quantile_grid_points'])
    taus = np.unique(np.clip(np.r_[-1.,1.,np.quantile(score,q)],-1.,1.))
    gammas = np.unique(np.maximum(0,np.r_[0.,np.quantile(margin,q)]))
    betas = np.unique(np.clip(np.r_[0.,1.,np.quantile(confidence,q)],0.,1.))
    feedback, best = [], None
    for tau in taus:
        hit = valid & (score > tau)
        for gamma in gammas:
            candidate = hit & (margin > gamma)
            for beta in betas:
                active = candidate & (confidence < beta)
                tp, fp_receiver, fp_all = int((active&pos).sum()),int((active&rmask).sum()),int((active&~pos).sum())
                feasible = fp_receiver/len(receiver_y) <= PORTABLE_RULES['receiver_selection_far_budget']
                row = dict(tau=float(tau),gamma=float(gamma),beta=float(beta),true_positive=tp,
                    positive_rows=int(pos.sum()),false_positive_receiver=fp_receiver,receiver_rows=len(receiver_y),
                    false_positive_all=fp_all,far_receiver=fp_receiver/len(receiver_y),feasible=feasible)
                feedback.append(row)
                key = (tp,-fp_all,-float(beta),float(gamma),float(tau))
                if feasible and (best is None or key > best[0]):
                    best = key,row
    return dict(best[1],far_budget=PORTABLE_RULES['receiver_selection_far_budget'],
        has_positive_gain=best[1]['true_positive'] > 0), feedback


def choose_anchor_beta(receiver, receiver_y, donor, donor_y, class_id, tau, gamma):
    """Direct self-gate ablation; never weaken the original routing/margin gate."""
    score = np.concatenate([receiver['signature_score'],donor['signature_score']])
    margin = np.concatenate([receiver['margin'],donor['margin']])
    valid = np.concatenate([receiver['signature_valid'],donor['signature_valid']])
    conf = np.concatenate([receiver['local_confidence'],donor['local_confidence']])
    y = np.concatenate([receiver_y,donor_y])
    pos,rmask = y==class_id,np.arange(len(y)) < len(receiver_y)
    beta_values = np.unique(np.clip(np.r_[0.,1.,np.quantile(conf,np.linspace(0,1,PORTABLE_RULES['quantile_grid_points']))],0,1))
    best,feedback = None,[]
    candidate = valid & (score > tau) & (margin > gamma)
    for beta in beta_values:
        active = candidate & (conf < beta)
        tp,fp,all_fp = int((active&pos).sum()),int((active&rmask).sum()),int((active&~pos).sum())
        feasible = fp/len(receiver_y) <= PORTABLE_RULES['receiver_selection_far_budget']
        row = dict(tau=float(tau),gamma=float(gamma),beta=float(beta),true_positive=tp,positive_rows=int(pos.sum()),
            false_positive_receiver=fp,receiver_rows=len(receiver_y),false_positive_all=all_fp,
            far_receiver=fp/len(receiver_y),feasible=feasible)
        feedback.append(row)
        key = (tp,-all_fp,-float(beta))
        if feasible and (best is None or key > best[0]):
            best = key,row
    return dict(best[1],far_budget=PORTABLE_RULES['receiver_selection_far_budget'],
        has_positive_gain=best[1]['true_positive']>0,anchor_tau_gamma_fixed=True),feedback
