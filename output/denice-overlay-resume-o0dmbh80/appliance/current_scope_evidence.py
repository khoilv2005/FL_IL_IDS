"""Current owner-local negative CAL witnesses for immutable installed guards.

Raw samples stay at the witness endpoint. Only model capsules and aggregate
receipts cross the simulated transport. Witness selection uses current shard
counts and the live graph; it never uses HOLDOUT outcomes to substitute peers.
"""
import copy
import hashlib
import io

import numpy as np
import torch

from .active_pair_calibration import split_counts
from .config import Rejected
from .current_calibration_data import CurrentCalibrationData
from .cumulative_certificate import (RECEIPT_VERSION, checked_receipt, decode_receipt,
                                     encode_receipt, sha, validated_scope)
from .guarded_head import original_local_head
from .imported_route import ROUTE_RULES, stratified_roles
from .portable_route import ProtectedRoute
from .base_sketch_shield import CurrentBaseSketchShield
from .stable_head import StableHeadRegistry, stable_signals


def plan_current_witnesses(receiver, entry, task, current_classes, groups, alphas, views, max_peers=4):
    if type(max_peers) is not int or max_peers < 0:
        raise Rejected('INVALID_SCOPE_PEER_BUDGET')
    effective = set(validated_scope(entry)['classes'])
    missing = set(current_classes) - effective - {entry['class_id']}
    used={(r['owner'],r['task'],r['partition_sha256'])
          for e in entry.get('cumulative_CAL_certificate',{}).get('events',[]) for r in e['receipts']}
    def unused(owner):
        v=views[owner]
        return (owner,task,v.store['clients'][str(owner)][str(task)]['sha256']) not in used
    owners = []
    if receiver in views and views[receiver].task == task and unused(receiver):
        owners.append(receiver)
    peers = [i for i in groups.get(receiver, []) if i != receiver and
             alphas.get(receiver, {}).get(i, 0) > 0 and i in views and views[i].task == task and unused(i)]
    def supported(owner):
        counts = views[owner].store['clients'][str(owner)][str(task)]['class_counts']
        return {c for c in missing if split_counts(int(counts[str(c)]))[2] > 0}
    remaining = set(missing)
    if owners:
        remaining -= supported(receiver)
    chosen = []
    while remaining and peers and len(chosen) < max_peers:
        owner = min(peers, key=lambda i: (-len(supported(i) & remaining), i))
        covered = supported(owner) & remaining
        if not covered:
            break
        chosen.append(owner)
        peers.remove(owner)
        remaining -= covered
    return dict(receiver=receiver, task=task, owners=owners + chosen,
                missing_current_classes=sorted(missing), uncovered_by_planned_witnesses=sorted(remaining),
                peer_budget=max_peers, selection='metadata-only greedy class coverage; owner ID tie',
                historical_raw_reads=0, holdout_opened=False,
                effective_classes_before=sorted(effective))


class CurrentNegativeEndpoint:
    def __init__(self, owner, view):
        if not isinstance(view, CurrentCalibrationData) or owner != view.client_id:
            raise Rejected('SCOPE_CURRENT_OWNER_AUTHORITY_REQUIRED')
        self.owner, self.view = owner, view

    def receipt(self, model, router, class_id, seen, batch_size):
        """CAL labels are used for local counters only, after guard scoring."""
        registry = StableHeadRegistry()
        registry.entries = getattr(model, 'appliance_guarded_head_entries', {})
        entry = registry.entries[class_id]
        if not registry.certificate_current(model, router, class_id):
            raise Rejected('SCOPE_WITNESS_FUNCTION_CHANGED')
        classes = self.view.store['task_classes'][str(self.view.task)]
        pool = self.view.current_pool(self.owner, 'calibration', classes)
        held = stratified_roles(pool, ROUTE_RULES['seed'] + self.owner)['holdout']
        x, labels, rows = pool['X'][held], pool['y'][held], pool['rows'][held]
        device = str(next(model.parameters()).device)
        # Score the unchanged guard counterfactually even while the route is
        # suspended. This neither activates it nor modifies acceptance state.
        route = ProtectedRoute.from_packet(entry['packet'])
        with original_local_head(model, class_id, entry['backup']):
            signals = stable_signals(model, router, x, seen, route, entry['reference_classes'],
                                     batch_size, device) if len(x) else None
        shield = CurrentBaseSketchShield.restore(entry['shield_at_install'])
        hit = ((signals['signature_valid'] & (signals['signature_score'] > route.metadata['tau']) &
                (signals['margin'] > route.metadata['gamma']) & ~shield.veto(x, entry['required_old_classes']))
               if signals else np.zeros(0, bool))
        counts = {}
        for c in sorted(map(int, np.unique(labels))):
            if c == class_id:
                continue
            mask = labels == c
            counts[str(c)] = dict(rows=int(mask.sum()), activated=int(hit[mask].sum()),
                break_count=int((hit & mask & (signals['local_pred'] == labels)).sum()))
        value = dict(version=RECEIPT_VERSION, receiver=entry['receiver'], owner=self.owner,
            task=self.view.task, class_id=class_id, patch_id=entry['patch_id'],
            initial_certificate_sha256=entry['lifecycle_certificate_sha256'],
            guard_function_fingerprint=entry['guard_function_fingerprint'],
            guard_declaration_sha256=sha(entry['guard_declaration']),
            role_manifest_sha256=self.view.store['role_manifest_sha256'],
            partition_sha256=pool['partition_sha256'],
            coordinate_sha256=hashlib.sha256(np.asarray(rows, dtype='<i8').tobytes()).hexdigest(),
            current_classes=list(classes), counts=counts,
            source_role='current owner-local CAL HOLDOUT aggregates',
            historical_raw_reads=0, thresholds_retuned=False)
        value['receipt_id'] = sha(value)
        return encode_receipt(checked_receipt(value, entry))


def collect_current_receipts(receiver, class_id, model, router, config, seen, plan, views, transport):
    """Each witness evaluates the receiver capsule on its own current partition."""
    from fed_learning.training.checkpoint_state import snapshot_denice_state
    from eval_checkpoint import _make_denice_client_model
    entry = model.appliance_guarded_head_entries[class_id]
    receipts = []
    payload = None
    for owner in plan['owners']:
        endpoint = CurrentNegativeEndpoint(owner, views[owner])
        if owner == receiver:
            current, detector = model, router
        else:
            if payload is None:
                algorithm = snapshot_denice_state(model, router)
                if algorithm['context_detector'].get('reference_input_memory'):
                    raise Rejected('RAW_REFERENCE_MEMORY_IN_SCOPE_CAPSULE')
                capsule = dict(config=copy.deepcopy(config), task=views[owner].task,
                    client_model_states={receiver: {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}},
                    client_algorithm_states={receiver: {'denice': algorithm}})
                buffer = io.BytesIO()
                torch.save(capsule, buffer)
                payload = buffer.getvalue()
            delivered = transport.send(receiver, owner, 'current_scope_function_capsule', payload)
            current, detector = _make_denice_client_model(torch.load(io.BytesIO(delivered),
                map_location='cpu', weights_only=False), receiver, str(next(model.parameters()).device))
        packet = endpoint.receipt(current, detector, class_id, seen, int(config.get('appliance_batch_size', 512)))
        if owner != receiver:
            packet = transport.send(owner, receiver, 'current_scope_negative_receipt', packet)
        receipts.append(decode_receipt(packet, entry))
        if owner != receiver:
            del current, detector
    return receipts
