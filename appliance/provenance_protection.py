"""Portable peer BASE support, separate from ownership and CAL certification.

Packets come from a trusted simulator endpoint. Checksums detect corruption;
they are not signatures against a malicious sender. A positive graph edge alone
is insufficient: the aggregation/bootstrap caller must supply a knowledge
receipt identifying class rows and the model versions involved.

The separate ``protection_only`` receipt records delivery of finite negative
support geometry, not a classifier row transfer. Its class_rows identify source
support classes; it must never be treated as acquired classifier competence.
"""
import copy
import hashlib
import inspect
import json
import zlib
import numpy as np
from .base_sketch_shield import CurrentBaseSketchShield, sketch_with_error
from .portable_route import SharedSketch
from .config import Rejected
from .state import digest

VERSION = 'appliance_peer_BASE_protection_v1'


def veto_implementation():
    return digest(dict(numpy_version=np.__version__,
        own_veto=hashlib.sha256(inspect.getsource(CurrentBaseSketchShield.veto).encode()).hexdigest(),
        numerical_enclosure=hashlib.sha256(inspect.getsource(sketch_with_error).encode()).hexdigest(),
        sketch_features=hashlib.sha256(inspect.getsource(SharedSketch.features).encode()).hexdigest(),
        sketch_projection=hashlib.sha256(inspect.getsource(SharedSketch.matrix).encode()).hexdigest(),
        peer_veto=hashlib.sha256(inspect.getsource(ProvenanceProtection.veto).encode()).hexdigest(),
        frozen_veto=hashlib.sha256(inspect.getsource(frozen_veto).encode()).hexdigest()))


def sealed(body):
    return dict(copy.deepcopy(body), state_digest=digest(body))


def checked(value, version):
    body = {k: copy.deepcopy(v) for k, v in value.items() if k != 'state_digest'}
    if body.get('version') != version or value.get('state_digest') != digest(body):
        raise Rejected('PEER_PROTECTION_CHECKSUM_OR_VERSION_CHANGED')
    return body


def export_support(shield, sender_model_version, dependency_version):
    """Only owned summaries are exported; no automatic transitive relaying."""
    source = CurrentBaseSketchShield.restore(shield.state())
    if any(type(v) is not str or not v for v in (sender_model_version, dependency_version)):
        raise Rejected('PEER_PROTECTION_MODEL_BINDING_REQUIRED')
    return sealed(dict(version=VERSION, sender=source.owner,
        sender_model_version=sender_model_version, dependency_version=dependency_version,
        source=source.state(), source_role='peer-owned BASE support; not receiver-owned; not CAL',
        CAL_acceptance_substitution=False, population_FAR_certified=False))


class ProvenanceProtection:
    """Receiver-local receipt ledger; frozen snapshots can veto without labels.

    The union protects finite support from participating origins. It does not
    establish recall, competence or safety on unseen populations. A snapshot
    belongs to a NEW guard and must pass unchanged CAL acceptance before install.
    Existing certificates must never be rewritten to add this veto.
    """
    def __init__(self, own):
        self.own = CurrentBaseSketchShield.restore(own.state())
        self.receipts = {}
        self.sources = {}

    @property
    def owner(self):
        return self.own.owner

    def receive(self, packet, receipt, *, current_task, authorized_senders):
        body = checked(packet, VERSION)
        if set(body) != {'version', 'sender', 'sender_model_version', 'dependency_version',
                         'source', 'source_role', 'CAL_acceptance_substitution', 'population_FAR_certified'}:
            raise Rejected('PEER_PROTECTION_PACKET_SCHEMA_CHANGED')
        source = CurrentBaseSketchShield.restore(body['source'])
        fields = {'sender', 'receiver', 'task', 'round', 'kind', 'alpha', 'class_rows',
                  'sender_model_version', 'receiver_model_version', 'dependency_version', 'receipt_id'}
        if set(receipt) != fields or receipt['receipt_id'] != digest({k: v for k, v in receipt.items() if k != 'receipt_id'}):
            raise Rejected('PEER_PROTECTION_KNOWLEDGE_RECEIPT_CHANGED')
        if (type(current_task) is not int or not 0 <= current_task <= 5 or
                type(receipt['task']) is not int or receipt['task'] != current_task or type(receipt['round']) is not int or receipt['round'] < 0 or
                source.memory.task > current_task):
            raise Rejected('PEER_PROTECTION_FUTURE_OR_STALE_TASK')
        if (receipt['receiver'] != self.owner or receipt['sender'] == self.owner or
                receipt['sender'] not in authorized_senders or source.owner != receipt['sender'] or
                body['sender'] != receipt['sender'] or receipt['kind'] not in ('aggregation', 'bootstrap', 'protection_only') or
                not np.isfinite(receipt['alpha']) or receipt['alpha'] <= 0):
            raise Rejected('PEER_PROTECTION_UNAUTHORIZED_SOURCE')
        if (source.role_sha != self.own.role_sha or source.pp_sha != self.own.pp_sha or
                source.memory.sketch.manifest() != self.own.memory.sketch.manifest()):
            raise Rejected('PEER_PROTECTION_ROLE_OR_FEATURE_SPACE_CHANGED')
        if (body['sender_model_version'] != receipt['sender_model_version'] or
                body['dependency_version'] != receipt['dependency_version'] or
                any(type(receipt[k]) is not str or not receipt[k] for k in
                    ('sender_model_version', 'receiver_model_version', 'dependency_version')) or
                body['source_role'] != 'peer-owned BASE support; not receiver-owned; not CAL' or
                body['CAL_acceptance_substitution'] is not False or body['population_FAR_certified'] is not False):
            raise Rejected('PEER_PROTECTION_EVIDENCE_BINDING_CHANGED')
        classes = receipt['class_rows']
        if (not classes or classes != sorted(set(classes)) or
                any(type(c) is not int or not 0 <= c < 34 or str(c) not in source.memory.entries for c in classes)):
            raise Rejected('PEER_PROTECTION_CLASS_RECEIPT_REQUIRED')
        source_id = body['source']['state_digest']
        self.sources[source_id] = copy.deepcopy(body['source'])
        header = {k: copy.deepcopy(v) for k, v in body.items() if k != 'source'}
        record = dict(header=header, source_id=source_id, receipt=copy.deepcopy(receipt))
        key = receipt['receipt_id']
        if key in self.receipts and self.receipts[key] != record:
            raise Rejected('PEER_PROTECTION_RECEIPT_REPLAY_CHANGED')
        self.receipts[key] = record
        return key

    def packet_for(self, record):
        return sealed(dict(record['header'], source=copy.deepcopy(self.sources[record['source_id']])))

    def advance_own(self, own):
        updated = CurrentBaseSketchShield.restore(own.state())
        if (updated.owner != self.owner or updated.role_sha != self.own.role_sha or
                updated.pp_sha != self.own.pp_sha or updated.memory.sketch.manifest() != self.own.memory.sketch.manifest() or
                updated.memory.task <= self.own.memory.task or
                updated.provenance[:len(self.own.provenance)] != self.own.provenance or
                any(updated.memory.entries.get(c) != e for c, e in self.own.memory.entries.items())):
            raise Rejected('PEER_PROTECTION_OWN_HISTORY_REWRITTEN')
        self.own = updated

    def coverage(self, required):
        owned = self.own.coverage(required)
        origins = {str(c): [] for c in required}
        evidence_types = {str(c): [] for c in required}
        for record in self.receipts.values():
            receipt = record['receipt']
            for c in required:
                if c in receipt['class_rows']:
                    origins[str(c)].append(receipt['sender'])
                    evidence_types[str(c)].append(receipt['kind'])
        missing = [c for c in owned['missing_classes'] if not origins[str(c)]]
        return dict(owned=owned, peer_origins={c: sorted(set(v)) for c, v in origins.items()},
            peer_evidence_types={c: sorted(set(v)) for c, v in evidence_types.items()},
            missing_protection_classes=missing, CAL_acceptance_substitution=False,
            inherited_knowledge_is_receiver_owned=False, population_FAR_certified=False)

    def freeze(self, required, imported_class):
        required = sorted(set(required))
        if type(imported_class) is not int or not 0 <= imported_class < 34 or imported_class in required:
            raise Rejected('PEER_PROTECTION_POSITIVE_CLASS_CANNOT_BE_NEGATIVE')
        if self.coverage(required)['missing_protection_classes']:
            raise Rejected('PEER_PROTECTION_SUPPORT_MISSING')
        # Independent snapshot: later discovery cannot mutate an installed guard.
        return sealed(dict(version=VERSION, ledger=self.state(), required=required,
                           imported_class=imported_class, policy='union finite owned and receipted peer BASE veto',
                           implementation_fingerprint=veto_implementation(),
                           CAL_acceptance_substitution=False))

    def veto(self, x, required):
        if self.coverage(required)['missing_protection_classes']:
            raise Rejected('PEER_PROTECTION_SUPPORT_MISSING')
        local = [c for c in required if str(c) in self.own.memory.entries]
        result = self.own.veto(x, local)
        for record in self.receipts.values():
            supported = [c for c in required if c in record['receipt']['class_rows']]
            if supported:
                source = CurrentBaseSketchShield.restore(self.sources[record['source_id']])
                result |= source.veto(x, supported)
        return result

    def state(self):
        used = {r['source_id'] for r in self.receipts.values()}
        return sealed(dict(version=VERSION, own=self.own.state(), receipts=copy.deepcopy(self.receipts),
                           sources={k: copy.deepcopy(self.sources[k]) for k in sorted(used)}))

    @classmethod
    def restore(cls, state):
        body = checked(state, VERSION)
        if set(body) != {'version', 'own', 'receipts', 'sources'}:
            raise Rejected('PEER_PROTECTION_LEDGER_SCHEMA_CHANGED')
        obj = cls(CurrentBaseSketchShield.restore(body['own']))
        if set(body['sources']) != {r['source_id'] for r in body['receipts'].values()}:
            raise Rejected('PEER_PROTECTION_SOURCE_CACHE_CHANGED')
        for key, record in body['receipts'].items():
            if set(record) != {'header', 'source_id', 'receipt'}:
                raise Rejected('PEER_PROTECTION_RECEIPT_SCHEMA_CHANGED')
            receipt = record['receipt']
            source = body['sources'][record['source_id']]
            if source.get('state_digest') != record['source_id']:
                raise Rejected('PEER_PROTECTION_SOURCE_VERSION_CHANGED')
            packet = sealed(dict(record['header'], source=source))
            accepted = obj.receive(packet, receipt, current_task=receipt['task'],
                                   authorized_senders=[receipt['sender']])
            if key != accepted:
                raise Rejected('PEER_PROTECTION_RECEIPT_KEY_CHANGED')
        return obj


def frozen_veto(snapshot, x):
    body = checked(snapshot, VERSION)
    if (set(body) != {'version', 'ledger', 'required', 'imported_class', 'policy', 'CAL_acceptance_substitution', 'implementation_fingerprint'} or
            body['policy'] != 'union finite owned and receipted peer BASE veto' or
            body['implementation_fingerprint'] != veto_implementation() or
            body['CAL_acceptance_substitution'] is not False or
            type(body['imported_class']) is not int or not 0 <= body['imported_class'] < 34 or
            body['imported_class'] in body['required']):
        raise Rejected('PEER_PROTECTION_FROZEN_POLICY_CHANGED')
    return ProvenanceProtection.restore(body['ledger']).veto(x, body['required'])


def transfer_support(ledger, packet, receipt, transport, *, current_task, authorized_senders):
    """Lossless compressed source cache + small model/row receipt on each edge.

    Cache is receiver-local and survives ledger checkpoint/restore. Failed
    sends leave the ledger unchanged; any bytes already sent remain accounted.
    No cryptographic authentication is claimed for the simulated endpoints.
    """
    source_id = packet['source']['state_digest']
    key = receipt['receipt_id']
    old_source = ledger.sources.get(source_id)
    old_record = ledger.receipts.get(key)
    # Validate source, edge, class rows and version BEFORE sending bytes.
    ledger.receive(packet, receipt, current_task=current_task, authorized_senders=authorized_senders)
    try:
        sender, receiver = receipt['sender'], receipt['receiver']
        if old_source is None:
            raw = json.dumps(packet['source'], sort_keys=True, separators=(',', ':')).encode()
            if len(raw) > 16*1024*1024:
                raise Rejected('PEER_PROTECTION_SOURCE_TOO_LARGE')
            encoded = transport.send(sender, receiver, 'peer_BASE_summary_zlib_v1', zlib.compress(raw, 6))
            decoded = zlib.decompress(encoded)
            source = json.loads(decoded)
            if source != packet['source']:
                raise Rejected('PEER_PROTECTION_WIRE_SOURCE_CHANGED')
        else:
            source = old_source
        header = {k: v for k, v in packet.items() if k != 'source'}
        raw = transport.send(sender, receiver, 'peer_BASE_model_row_receipt_v1',
                             dict(packet_header=header, source_id=source_id, receipt=receipt))
        delivered = json.loads(raw)
        ledger.receive(dict(delivered['packet_header'], source=source), delivered['receipt'],
                       current_task=current_task, authorized_senders=authorized_senders)
    except Exception:
        if old_record is None:
            ledger.receipts.pop(key, None)
        else:
            ledger.receipts[key] = old_record
        if old_source is None:
            ledger.sources.pop(source_id, None)
        else:
            ledger.sources[source_id] = old_source
        raise
    return key
