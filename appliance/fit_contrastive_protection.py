"""FIT-calibrated negative-head offsets; no old CAL or new backbone learning.

Cross-context donor heads have incomparable offsets even within one model.
Subtract a fixed donor-current-FIT 99th percentile from each negative-vs-patch
margin. CAL selection/holdout and independent development still decide whether
the resulting empirical guard is usable. No old-class FAR proof is inferred.
"""
import copy
import hashlib
import inspect
import numpy as np
import torch
from .config import Rejected
from .contrastive_protection import ContrastiveProtection
from .dependency_boundary import head_dependency_boundary
from .closure import effective_linear
from .portable_route import receiver_signals
from .provenance_protection import ProvenanceProtection, checked, sealed
from .state import digest

VERSION = 'appliance_current_FIT_p99_negative_margin_v1'
POLICY = 'own hard veto; peer box AND negative-minus-imported logit >= fixed FIT-p99 offset'


def feature_binding(model, route, refs, heads):
    w, b = effective_linear(model, 'fc2')
    queries = [torch.as_tensor(route.head_weight).reshape(1, -1), w[refs]]
    queries += [torch.as_tensor(heads[c]['offer']['weight'], dtype=w.dtype).reshape(1, -1) for c in sorted(heads)]
    boundary = head_dependency_boundary(model, torch.cat(queries, 0))
    if not boundary['all_dependencies_mature']:
        raise Rejected('FIT_COUNTER_NONMATURE_DEPENDENCY')
    return digest(dict(prefix=boundary['canonical_value_function_sha'], refs=refs,
                       ref_weight=w[refs], ref_bias=b[refs], patch_weight=route.head_weight,
                       patch_bias=route.head_bias, negative_heads=heads,
                       torch_version=torch.__version__, backend=next(model.parameters()).device.type))


class FitContrastiveProtection(ContrastiveProtection):
    def __init__(self, ledger, required, imported_class, negative_heads, offsets, fit_evidence):
        super().__init__(ledger, required, imported_class, negative_heads)
        self.offsets = {int(c): float(v) for c, v in offsets.items()}
        self.fit_evidence = copy.deepcopy(fit_evidence)
        fields = {'role', 'owner', 'task', 'class_id', 'positive_rows', 'row_ids_sha256', 'partition_sha256',
                  'role_manifest_sha256', 'feature_binding', 'quantile', 'minimum_offset', 'selection_opened',
                  'holdout_opened'}
        e = self.fit_evidence
        if (set(self.offsets) != set(self.heads) or any(not np.isfinite(v) or v < 0 for v in self.offsets.values()) or
                set(e) != fields or e['role'] != 'donor current CAL-FIT only' or
                type(e['owner']) is not int or e['owner'] == self.ledger.owner or
                type(e['task']) is not int or e['task'] != self.ledger.own.memory.task or
                e['class_id'] != imported_class or type(e['positive_rows']) is not int or e['positive_rows'] < 32 or
                e['quantile'] != .99 or e['minimum_offset'] != 0 or
                e['role_manifest_sha256'] != self.ledger.own.role_sha or
                e['selection_opened'] is not False or e['holdout_opened'] is not False or
                any(type(e[k]) is not str or not e[k] for k in
                    ('row_ids_sha256', 'partition_sha256', 'feature_binding'))):
            raise Rejected('FIT_COUNTER_OFFSET_EVIDENCE_CHANGED')
        if (len(e['row_ids_sha256']) != 64 or len(e['partition_sha256']) != 64 or
                not any(r['receipt']['sender'] == e['owner'] for r in self.ledger.receipts.values())):
            raise Rejected('FIT_COUNTER_DONOR_OR_ROWS_UNBOUND')

    @classmethod
    @torch.no_grad()
    def fit(cls, counter, model, router, route, refs, fit_x, fit_rows, view, seen, batch_size):
        from .current_calibration_data import CurrentCalibrationData
        if (not isinstance(view, CurrentCalibrationData) or view.client_id != route.metadata['donor'] or
                view.task != route.metadata['task'] or len(fit_x) != len(fit_rows) or len(fit_x) < 32):
            raise Rejected('FIT_COUNTER_CURRENT_FIT_AUTHORITY_REQUIRED')
        # Caller's rows must exactly equal the authenticated donor's current positive FIT.
        from .receiver_aware_discovery import current_fit
        actual = current_fit(view); pos = actual['y'] == counter.imported_class
        if not np.array_equal(fit_rows, actual['rows'][pos]) or not np.array_equal(fit_x, actual['X'][pos]):
            raise Rejected('FIT_COUNTER_ROWS_NOT_CURRENT_POSITIVE_FIT')
        counter.function(model, route, refs)
        base = receiver_signals(model, router, fit_x, seen, route.metadata['task'],
                                route.metadata['class_id'], batch_size, 'cpu')
        sig = route.signals(base, fit_x); h = base['imported_features']; patch = sig['patch_logit']
        offsets = {}
        for c in sorted(counter.heads):
            offer = counter.heads[c]['offer']
            delta = h @ np.asarray(offer['weight'], np.float32) + np.float32(offer['bias']) - patch
            if not np.isfinite(delta).all():
                raise Rejected('FIT_COUNTER_NONFINITE_CALIBRATION')
            offsets[c] = max(0., float(np.quantile(delta, .99, method='linear')))
        evidence = dict(role='donor current CAL-FIT only', owner=view.client_id, task=view.task,
            class_id=counter.imported_class, positive_rows=len(fit_x),
            row_ids_sha256=hashlib.sha256(np.asarray(fit_rows, dtype='<i8').tobytes()).hexdigest(),
            partition_sha256=view.current_pool(view.client_id, 'calibration', view.store['task_classes'][str(view.task)])['partition_sha256'],
            role_manifest_sha256=view.store['role_manifest_sha256'],
            feature_binding=feature_binding(model, route, refs, counter.heads),
            quantile=.99, minimum_offset=0, selection_opened=False, holdout_opened=False)
        return cls(counter.ledger, counter.required, counter.imported_class, counter.heads, offsets, evidence)

    def state(self):
        base = {k: v for k, v in super().state().items() if k != 'state_digest'}
        return sealed(dict(base, version=VERSION, policy=POLICY,
                           offsets={str(c): v for c, v in self.offsets.items()}, fit_evidence=self.fit_evidence))

    @classmethod
    def restore(cls, state):
        b = checked(state, VERSION)
        fields = {'version', 'ledger', 'required', 'imported_class', 'negative_heads', 'policy',
                  'population_FAR_certified', 'installs_classifier_knowledge', 'offsets', 'fit_evidence'}
        if (set(b) != fields or b['policy'] != POLICY or b['population_FAR_certified'] is not False or
                b['installs_classifier_knowledge'] is not False):
            raise Rejected('FIT_COUNTER_STATE_SCHEMA_CHANGED')
        return cls(ProvenanceProtection.restore(b['ledger']), b['required'], b['imported_class'],
                   {int(c): p for c, p in b['negative_heads'].items()}, b['offsets'], b['fit_evidence'])

    def function(self, model, route, refs):
        if self.fit_evidence['owner'] != route.metadata['donor'] or self.fit_evidence['task'] != route.metadata['task']:
            raise Rejected('FIT_COUNTER_CALIBRATING_DONOR_CHANGED')
        binding = feature_binding(model, route, refs, self.heads)
        if binding != self.fit_evidence['feature_binding']:
            raise Rejected('FIT_COUNTER_RECEIVER_FUNCTION_CHANGED')
        base = super().function(model, route, refs)
        implementation = {n: hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
                          for n, fn in [('binding', feature_binding), ('fit', self.fit), ('function', self.function),
                                        ('veto', self.veto), ('state', self.state), ('restore', self.restore),
                                        ('constructor', self.__init__)]}
        return dict(base, fingerprint=digest(dict(base=base, state=self.state(), implementation=implementation)))

    def veto(self, x, features, patch_logit):
        h = np.asarray(features, np.float32); patch = np.asarray(patch_logit)
        if h.ndim != 2 or h.shape[0] != len(x) or patch.shape != (len(x),):
            raise Rejected('FIT_COUNTER_INPUT_ALIGNMENT_CHANGED')
        if not np.isfinite(h).all() or not np.isfinite(patch).all():
            raise Rejected('FIT_COUNTER_NONFINITE_SIGNALS')
        veto = self.ledger.own.veto(x, self.local)
        for c in sorted(self.heads):
            offer = self.heads[c]['offer']
            delta = h @ np.asarray(offer['weight'], np.float32) + np.float32(offer['bias']) - patch
            if not np.isfinite(delta).all():
                raise Rejected('FIT_COUNTER_NONFINITE_NEGATIVE_LOGIT')
            box_hit = np.zeros(len(x), bool)
            for source in self.sources[c]:
                box_hit |= source.veto(x, [c])
            veto |= box_hit & (delta >= self.offsets[c])
        return veto
