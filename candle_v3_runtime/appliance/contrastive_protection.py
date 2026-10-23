"""Experimental negative-head evidence for overlapping peer BASE sketches.

Own BASE veto stays unconditional. A peer sketch hit blocks an imported head
only when a receipted mature negative head beats it in the receiver's feature
space. This is empirical counter-evidence, NOT a finite-support safety proof.
It never emits a negative class prediction or establishes receiver ownership.
Old certificates must not be relabelled with this policy.
"""
import copy
import hashlib
import inspect
import numpy as np
import torch
from .base_sketch_shield import CurrentBaseSketchShield
from .closure import effective_linear
from .config import Rejected
from .dependency_boundary import head_dependency_boundary
from .portable_route import receiver_signals
from .provenance_protection import ProvenanceProtection, checked, sealed, veto_implementation
from .receiver_aware_discovery import VERSION as OFFER_VERSION
from .state import digest

VERSION = 'appliance_peer_negative_head_counter_evidence_v1'


class ContrastiveProtection:
    def __init__(self, ledger, required, imported_class, negative_heads):
        self.ledger = ProvenanceProtection.restore(ledger.state())
        self.required = sorted(set(required))
        self.imported_class = imported_class
        if type(imported_class) is not int or not 0 <= imported_class < 34 or imported_class in self.required:
            raise Rejected('CONTRASTIVE_POSITIVE_CLASS_CANNOT_BE_NEGATIVE')
        if self.ledger.coverage(self.required)['missing_protection_classes']:
            raise Rejected('CONTRASTIVE_SUPPORT_MISSING')
        self.local = [c for c in self.required if str(c) in self.ledger.own.memory.entries]
        peer_classes = {c for c in self.required if any(c in r['receipt']['class_rows']
                        for r in self.ledger.receipts.values())}
        if set(negative_heads) != peer_classes:
            raise Rejected('CONTRASTIVE_NEGATIVE_HEAD_COVERAGE_CHANGED')
        self.heads = copy.deepcopy(negative_heads)
        self.sources = {}
        for c, packet in self.heads.items():
            if set(packet) != {'sender', 'class_id', 'offer', 'source_id', 'source_head_mature'}:
                raise Rejected('CONTRASTIVE_HEAD_SCHEMA_CHANGED')
            offer = packet['offer']
            if (set(offer) != {'version', 'class_id', 'model_version', 'weight', 'bias', 'digest'} or
                    offer['version'] != OFFER_VERSION or offer['class_id'] != c or packet['class_id'] != c or
                    packet['source_head_mature'] is not True or
                    offer['digest'] != digest({k: v for k, v in offer.items() if k != 'digest'})):
                raise Rejected('CONTRASTIVE_HEAD_OR_MATURITY_CHANGED')
            records = [r for r in self.ledger.receipts.values()
                       if r['receipt']['sender'] == packet['sender'] and c in r['receipt']['class_rows'] and
                       r['source_id'] == packet['source_id'] and
                       r['receipt']['sender_model_version'] == offer['model_version']]
            if not records:
                raise Rejected('CONTRASTIVE_HEAD_RECEIPT_BINDING_CHANGED')
            w = np.asarray(offer['weight'], np.float32)
            if w.ndim != 1 or not np.isfinite(w).all() or not np.isfinite(offer['bias']):
                raise Rejected('CONTRASTIVE_NONFINITE_OR_INVALID_HEAD')
            source_ids = {r['source_id'] for r in self.ledger.receipts.values()
                          if c in r['receipt']['class_rows']}
            self.sources[c] = [CurrentBaseSketchShield.restore(self.ledger.sources[sid]) for sid in sorted(source_ids)]

    def state(self):
        return sealed(dict(version=VERSION, ledger=self.ledger.state(), required=self.required,
            imported_class=self.imported_class,
            negative_heads={str(c): p for c, p in self.heads.items()},
            policy='own hard veto; peer box AND negative logit >= imported logit',
            population_FAR_certified=False, installs_classifier_knowledge=False))

    @classmethod
    def restore(cls, state):
        b = checked(state, VERSION)
        expected = {'version', 'ledger', 'required', 'imported_class', 'negative_heads', 'policy',
                    'population_FAR_certified', 'installs_classifier_knowledge'}
        if (set(b) != expected or b['policy'] != 'own hard veto; peer box AND negative logit >= imported logit' or
                b['population_FAR_certified'] is not False or b['installs_classifier_knowledge'] is not False):
            raise Rejected('CONTRASTIVE_STATE_SCHEMA_CHANGED')
        return cls(ProvenanceProtection.restore(b['ledger']), b['required'], b['imported_class'],
                   {int(c): p for c, p in b['negative_heads'].items()})

    def function(self, model, route, refs):
        if route.metadata['class_id'] != self.imported_class or not refs:
            raise Rejected('CONTRASTIVE_ROUTE_OR_REFERENCE_CHANGED')
        w, b = effective_linear(model, 'fc2')
        if any(int(model.unit_ranks['fc2'][c]) < 2 for c in refs):
            raise Rejected('CONTRASTIVE_NONMATURE_LOCAL_REFERENCE')
        queries = [torch.as_tensor(route.head_weight).reshape(1, -1), w[refs]]
        for c in sorted(self.heads):
            q = torch.as_tensor(self.heads[c]['offer']['weight'], dtype=w.dtype).reshape(1, -1)
            if q.shape[1] != model.fc2.in_features:
                raise Rejected('CONTRASTIVE_HEAD_DIMENSION_CHANGED')
            queries.append(q)
        boundary = head_dependency_boundary(model, torch.cat(queries, 0))
        if not boundary['all_dependencies_mature']:
            raise Rejected('CONTRASTIVE_NONMATURE_DEPENDENCY')
        body = dict(policy=VERSION, snapshot=self.state()['state_digest'],
            boundary=boundary['canonical_value_function_sha'], reference_classes=refs,
            reference_weight=w[refs], reference_bias=b[refs], patch_weight=route.head_weight,
            patch_bias=route.head_bias, signature=route.metadata['signature'],
            tau=route.metadata['tau'], gamma=route.metadata['gamma'], torch_version=torch.__version__,
            backend=next(model.parameters()).device.type, numpy_version=np.__version__,
            veto_implementation=veto_implementation(),
            implementation={n: hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
                for n, fn in [('function', self.function), ('signals', self.signals),
                              ('receiver_signals', receiver_signals), ('veto', self.veto),
                              ('constructor', self.__init__), ('restore', self.restore)]})
        return dict(fingerprint=digest(body), all_dependencies_mature=True,
                    negative_head_count=len(self.heads), extra_classifier_outputs=0)

    def veto(self, x, features, patch_logit):
        h = np.asarray(features, np.float32)
        patch = np.asarray(patch_logit)
        if h.ndim != 2 or h.shape[0] != len(x) or patch.shape != (len(x),):
            raise Rejected('CONTRASTIVE_INPUT_ALIGNMENT_CHANGED')
        if not np.isfinite(h).all() or not np.isfinite(patch).all():
            raise Rejected('CONTRASTIVE_NONFINITE_SIGNALS')
        veto = self.ledger.own.veto(x, self.local)
        for c in sorted(self.heads):
            offer = self.heads[c]['offer']
            negative = h @ np.asarray(offer['weight'], np.float32) + np.float32(offer['bias'])
            if not np.isfinite(negative).all():
                raise Rejected('CONTRASTIVE_NONFINITE_NEGATIVE_LOGIT')
            box_hit = np.zeros(len(x), bool)
            for source in self.sources[c]:
                box_hit |= source.veto(x, [c])
            veto |= box_hit & (negative >= patch)
        return veto

    @torch.no_grad()
    def signals(self, model, router, x, seen, route, refs, batch_size, device):
        self.function(model, route, refs)
        with torch.autocast(device_type=next(model.parameters()).device.type, enabled=False):
            base = receiver_signals(model, router, x, seen, route.metadata['task'],
                                    route.metadata['class_id'], batch_size, device)
        w, b = effective_linear(model, 'fc2')
        h = base['imported_features']
        base['local_best_import_context'] = torch.nn.functional.linear(torch.as_tensor(h), w, b).numpy()[:, refs].max(1)
        sig = route.signals(base, x)
        sig['signature_valid'] &= ~self.veto(x, h, sig['patch_logit'])
        sig['local_confidence'] = np.zeros(len(x), np.float64)
        return sig
