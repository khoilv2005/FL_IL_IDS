"""Append-only empirical CAL scope grants for an unchanged imported function.

The original certificate is immutable. A later task, a BASE veto, a router
prediction or a class inventory alone never grants scope. Only current-task
CAL HOLDOUT aggregate receipts may grant their actually observed classes.
Packets are checksummed application messages in the existing trusted transport
simulation, not authenticated signatures or population FAR guarantees.
"""
import copy
import hashlib
import json

from .config import Rejected

VERSION = 'appliance_cumulative_current_CAL_v2'
RECEIPT_VERSION = 'appliance_current_negative_receipt_v2'
MAX_PACKET_BYTES = 65536


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def checked_receipt(value, entry):
    fields = {'version', 'receiver', 'owner', 'task', 'class_id', 'patch_id',
              'initial_certificate_sha256', 'guard_function_fingerprint',
              'guard_declaration_sha256', 'role_manifest_sha256', 'partition_sha256',
              'coordinate_sha256', 'current_classes', 'counts', 'source_role',
              'historical_raw_reads', 'thresholds_retuned', 'receipt_id'}
    if not isinstance(value, dict) or set(value) != fields:
        raise Rejected('CUMULATIVE_CAL_RECEIPT_SCHEMA')
    body = {k: v for k, v in value.items() if k != 'receipt_id'}
    try:
        body_sha = sha(body)
    except (ValueError, TypeError):
        raise Rejected('CUMULATIVE_CAL_RECEIPT_SCHEMA') from None
    if value['version'] != RECEIPT_VERSION or value['receipt_id'] != body_sha:
        raise Rejected('CUMULATIVE_CAL_RECEIPT_HASH')
    certificate = entry['lifecycle_certificate']
    if (value['receiver'] != entry['receiver'] or value['class_id'] != entry['class_id'] or
            value['patch_id'] != entry['patch_id'] or
            value['initial_certificate_sha256'] != entry['lifecycle_certificate_sha256'] or
            value['guard_function_fingerprint'] != entry['guard_function_fingerprint'] or
            value['guard_declaration_sha256'] != sha(entry['guard_declaration']) or
            value['role_manifest_sha256'] != certificate['initial_acceptance']['role_manifest_sha256'] or
            value['source_role'] != 'current owner-local CAL HOLDOUT aggregates' or
            value['historical_raw_reads'] != 0 or value['thresholds_retuned'] is not False):
        raise Rejected('CUMULATIVE_CAL_RECEIPT_BINDING')
    if (type(value['task']) is not int or not entry['task'] <= value['task'] <= 5 or
            type(value['owner']) is not int or value['owner'] < 0 or
            not isinstance(value['current_classes'], list) or not value['current_classes'] or
            any(type(c) is not int or not 0 <= c < 34 for c in value['current_classes']) or
            len(set(value['current_classes'])) != len(value['current_classes'])):
        raise Rejected('CUMULATIVE_CAL_RECEIPT_TASK')
    for name in ('partition_sha256', 'coordinate_sha256'):
        if (type(value[name]) is not str or len(value[name]) != 64 or
                any(c not in '0123456789abcdef' for c in value[name])):
            raise Rejected('CUMULATIVE_CAL_RECEIPT_PROVENANCE')
    counts = value['counts']
    if not isinstance(counts, dict):
        raise Rejected('CUMULATIVE_CAL_RECEIPT_COUNTS')
    for label, count in counts.items():
        if (type(label) is not str or not label.isdigit() or str(int(label)) != label or
                int(label) not in value['current_classes'] or int(label) == entry['class_id'] or
                not isinstance(count, dict) or set(count) != {'rows', 'activated', 'break_count'} or
                any(type(n) is not int or n < 0 for n in count.values()) or
                not 0 <= count['break_count'] <= count['activated'] <= count['rows'] or
                count['rows'] == 0):
            raise Rejected('CUMULATIVE_CAL_RECEIPT_COUNTS')
    return copy.deepcopy(value)


def encode_receipt(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if len(data) > MAX_PACKET_BYTES:
        raise Rejected('CUMULATIVE_CAL_RECEIPT_BUDGET')
    return data


def decode_receipt(packet, entry):
    if not isinstance(packet, bytes) or len(packet) > MAX_PACKET_BYTES:
        raise Rejected('CUMULATIVE_CAL_RECEIPT_BUDGET')
    try:
        return checked_receipt(json.loads(packet.decode()), entry)
    except (ValueError, UnicodeError) as exc:
        raise Rejected('CUMULATIVE_CAL_RECEIPT_SCHEMA') from exc


def event_classes(receipts):
    # Each endpoint/class must pass individually. Never dilute a failed
    # endpoint with many zero-activation endpoints, nor sum duplicate receipts.
    classes = set()
    total = 0
    for receipt in receipts:
        for label, count in receipt['counts'].items():
            if count['activated'] / count['rows'] > .001 or count['break_count']:
                raise Rejected('CUMULATIVE_CAL_NEW_PROTECTION_CONFLICT')
            classes.add(int(label))
            total += count['rows']
    if total < 32:
        raise Rejected('CUMULATIVE_CAL_INSUFFICIENT_NEGATIVES')
    return classes


def validated_scope(entry):
    """Validate the entire grant chain; callers still check the original seal."""
    original = entry['lifecycle_certificate']['scope']
    classes = set(original['classes'])
    chain = entry.get('cumulative_CAL_certificate')
    if chain is None:
        return copy.deepcopy(original)
    if (not isinstance(chain, dict) or
            set(chain) != {'version', 'initial_certificate_sha256', 'events', 'head_sha256'} or
            chain['version'] != VERSION or
            chain['initial_certificate_sha256'] != entry['lifecycle_certificate_sha256'] or
            not isinstance(chain['events'], list)):
        raise Rejected('CUMULATIVE_CAL_CHAIN_SCHEMA')
    parent = entry['lifecycle_certificate_sha256']
    seen_sources = set()
    last_task = entry['task']
    for event in chain['events']:
        if (not isinstance(event, dict) or
                set(event) != {'parent_sha256', 'task', 'receipts', 'granted_classes', 'event_sha256'} or
                event['parent_sha256'] != parent or type(event['task']) is not int or
                not last_task <= event['task'] <= 5 or not isinstance(event['receipts'], list)):
            raise Rejected('CUMULATIVE_CAL_CHAIN_ORDER')
        if event['event_sha256'] != sha({k: v for k, v in event.items() if k != 'event_sha256'}):
            raise Rejected('CUMULATIVE_CAL_CHAIN_HASH')
        receipts = [checked_receipt(r, entry) for r in event['receipts']]
        if any(r['task'] != event['task'] for r in receipts):
            raise Rejected('CUMULATIVE_CAL_CHAIN_TASK')
        if len({tuple(r['current_classes']) for r in receipts}) != 1:
            raise Rejected('CUMULATIVE_CAL_CHAIN_TASK_CLASSES')
        for r in receipts:
            key = (r['owner'], r['task'], r['partition_sha256'])
            if key in seen_sources:
                raise Rejected('CUMULATIVE_CAL_RECEIPT_REPLAY')
            seen_sources.add(key)
        granted = sorted(event_classes(receipts) - classes)
        if not granted or event['granted_classes'] != granted:
            raise Rejected('CUMULATIVE_CAL_UNSUPPORTED_GRANT')
        classes.update(granted)
        parent, last_task = event['event_sha256'], event['task']
    if chain['head_sha256'] != parent:
        raise Rejected('CUMULATIVE_CAL_CHAIN_HEAD')
    return dict(domain_id=original['domain_id'], classes=sorted(classes))


def append_current_evidence(entry, receipts, function_current):
    """Atomic append; does not issue new positive certificates or change guards."""
    if function_current is not True or entry.get('new_conflict_latched', False):
        raise Rejected('CUMULATIVE_CAL_FUNCTION_OR_CONFLICT')
    previous = validated_scope(entry)
    receipts = [checked_receipt(r, entry) for r in receipts]
    if not receipts or len({r['task'] for r in receipts}) != 1:
        raise Rejected('CUMULATIVE_CAL_CURRENT_TASK_REQUIRED')
    sources = {(r['owner'], r['task'], r['partition_sha256'])
               for event in entry.get('cumulative_CAL_certificate', {}).get('events', [])
               for r in event['receipts']}
    for receipt in receipts:
        key = (receipt['owner'], receipt['task'], receipt['partition_sha256'])
        if key in sources:
            raise Rejected('CUMULATIVE_CAL_RECEIPT_REPLAY')
        sources.add(key)
    granted = sorted(event_classes(receipts) - set(previous['classes']))
    if not granted:
        return dict(issued=False, reason='no_new_CAL_classes', scope=previous)
    pending = copy.deepcopy(entry.get('cumulative_CAL_certificate', dict(version=VERSION,
        initial_certificate_sha256=entry['lifecycle_certificate_sha256'], events=[],
        head_sha256=entry['lifecycle_certificate_sha256'])))
    event = dict(parent_sha256=pending['head_sha256'], task=receipts[0]['task'],
                 receipts=receipts, granted_classes=granted)
    event['event_sha256'] = sha(event)
    pending['events'].append(event)
    pending['head_sha256'] = event['event_sha256']
    candidate = dict(entry, cumulative_CAL_certificate=pending)
    current = validated_scope(candidate)
    entry['cumulative_CAL_certificate'] = pending
    return dict(issued=True, granted_classes=granted, scope=current,
                event_sha256=event['event_sha256'], positive_retested=False,
                initial_certificate_modified=False, thresholds_changed=False,
                historical_raw_reads=0, population_FAR_claim=False)
