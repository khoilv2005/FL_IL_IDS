"""Bounded end-of-task peer sessions. All fitting/verification is current-only."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from .communication import WireLedger, CommunicationBudgetExceeded
from .contracts import TeachingConfig
from .discovery import select_teachers
from .head_learning import learn_head, head_permissions, initial_head
from .knowledge import announce, state
from .router_teaching import receiver_sketches, stage_router
from .snapshot import digest, encode_panel, freeze_receiver
from .transaction import stage_head, commit
from .verification import predict, accounting, acceptance


def _append(path, row):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, allow_nan=False)+'\n')


def _available_detector(reference, targets):
    detector = deepcopy(reference.detector)
    t = reference.identity.task
    detector.episode_classes[t] = sorted(set(detector.episode_classes.get(t, [])) | set(targets))
    return detector


def run_task_teaching(*, config, task, models, detectors, active_ids, data, seen_classes,
                      preprocessing_sha256, output_dir, device, allowed_edges, native_neighbors=None):
    settings = TeachingConfig.from_config(config)
    if not settings.enabled:
        return dict(enabled=False)
    ids = sorted(set(int(c) for c in active_ids))
    if set(allowed_edges) != set(ids):
        raise ValueError('Teaching graph must explicitly declare every active receiver')
    if any(set(peers)-set(ids) for peers in allowed_edges.values()):
        raise ValueError('Teaching graph includes undeclared peers')
    if settings.transfer_mode == 'direct_fc2' and native_neighbors is None:
        raise ValueError('Direct FC2 requires the actual native neighborhood, not sentinel cluster labels')
    # Task-boundary checkpoints persist completed sessions. A partial task is
    # replayed from its previous boundary; in-session resume is not claimed.
    saved_wire = [event for owner in ids for event in state(models[owner], owner)['wire_events']
                  if event['task'] == int(task)]
    if saved_wire:
        if len(saved_wire) != 1:
            raise ValueError('Ambiguous restored teaching ledger')
        ledger = WireLedger.from_state(saved_wire[0]['ledger_state'])
    else:
        ledger = WireLedger(int(settings.communication_budget_mib*1024**2))
    announcements = {}
    for owner in ids:
        _,labels = data.base[owner]
        announcements[owner] = announce(models[owner], detectors[owner], owner, task, labels, preprocessing_sha256)
    started, receipts = time.monotonic(), []
    for receiver in ids:
        record = state(models[receiver], receiver)
        if int(task) in record['completed_tasks']:
            continue
        try:
            local = announcements[receiver]
            peers = {}
            eligible_peers = set(allowed_edges[receiver]) - {receiver}
            if settings.transfer_mode == 'direct_fc2':
                eligible_peers &= set(native_neighbors.get(receiver, native_neighbors.get(str(receiver), [])))
            for donor in sorted(eligible_peers):
                peers[donor] = ledger.transmit(donor, receiver, 'announcement', announcements[donor])
            eligible = sorted(c for peer in peers.values() for c in peer.current_classes
                              if c not in local.current_classes and models[receiver].unit_ranks['fc2'][c] == 0)
            selection = select_teachers(receiver, {receiver:local, **peers}, eligible_peers,
                                        eligible, max_donors=settings.max_donors,
                                        max_targets=settings.max_targets)
            if not selection.targets:
                receipts.append(dict(receiver=receiver, status='SKIP', reason='no complementary candidate'))
                record['completed_tasks'].append(int(task))
                continue
            receipt = _session(receiver, selection, models, detectors, data, task, seen_classes,
                               preprocessing_sha256, settings, ledger, device, announcements)
            receipts.append(receipt)
            record = state(models[receiver], receiver)
            record['completed_tasks'].append(int(task))
            record['evidence'][str(task)] = receipt
            _append(Path(output_dir)/'candle_v3_sessions.jsonl', receipt)
            print(f"CANDLE V3 task={task} receiver={receiver} targets={list(selection.targets)} "
                  f"status={receipt['status']} teaching_bytes={ledger.bytes_sent}", flush=True)
        except CommunicationBudgetExceeded as exc:
            receipts.append(dict(receiver=receiver, status='DEFERRED', reason=str(exc)))
            # No evidence was turned into a PASS. Continue native training next task.
            for pending in ids:
                if task not in state(models[pending], pending)['completed_tasks']:
                    state(models[pending], pending)['completed_tasks'].append(int(task))
            break
    report = dict(task=int(task), receipts=receipts, elapsed_seconds=time.monotonic()-started,
                  wire=ledger.summary(), settings=asdict(settings),
                  historical_risk='unverified outside observed current CAL',
                  checkpoint_boundary='task: completed sessions are not replayed on task-boundary resume')
    _append(Path(output_dir)/'candle_v3_task_teaching.jsonl', report)
    # Persist ledgers in algorithm state as well as the human-readable JSONL.
    if ids:
        record = state(models[ids[0]], ids[0])
        record['wire_events'] = [event for event in record['wire_events'] if event['task'] != int(task)]
        record['wire_events'].append(dict(task=task, ledger_state=ledger.state_dict()))
    return report


def _session(receiver, selection, models, detectors, data, task, seen_classes,
             preprocessing_sha256, settings, ledger, device, announcements):
    reference = freeze_receiver(models[receiver], detectors[receiver], receiver, task, preprocessing_sha256, device)
    targets, donors = list(selection.targets), list(selection.donors)
    participants = [receiver]+donors
    session_id = digest(dict(reference=asdict(reference.identity), targets=targets, donors=donors,
                             settings=asdict(settings)))
    base = dict(receiver=receiver, task=task, session_id=session_id, targets=targets, donors=donors,
                snapshot=asdict(reference.identity), scope='observed current CAL only',
                transfer_mode=settings.transfer_mode)
    if not reference.detector.binarize_thresholds:
        return dict(base, status='UNKNOWN', reason='Receiver router thresholds are not initialized')
    if not np.any(reference.model.unit_ranks['fc1'] >= 2):
        return dict(base, status='UNKNOWN', reason='No consolidated FC1 coordinates for teaching')
    for donor in donors:
        ledger.transmit(receiver, donor, 'receiver_reference', dict(identity=asdict(reference.identity),
                        reference=reference.payload))
    available = _available_detector(reference, targets)
    fit_counts = {c:0 for c in targets}
    fit_correct = {c:0 for c in targets}
    for donor in donors:
        x,y = data.get(donor, 'fit', task=task, per_class=settings.fit_per_class)
        prediction,_ = predict(reference.model, available, x, seen_classes=seen_classes, batch_size=settings.batch_size)
        labels = y.numpy()
        receipt = {c:dict(n=int((labels==c).sum()), correct=int(((labels==c)&(prediction==c)).sum())) for c in targets}
        receipt = ledger.transmit(donor, receiver, 'verification_receipt', dict(role='fit', counts=receipt))
        for c, row in receipt['counts'].items():
            fit_counts[c] += row['n']; fit_correct[c] += row['correct']
    base['availability_fit'] = {c:dict(n=fit_counts[c], correct=fit_correct[c]) for c in targets}
    if any(n < settings.min_fit_positive for n in fit_counts.values()):
        return dict(base, status='UNKNOWN', reason='Insufficient current FIT positives')
    if any(fit_correct[c]/fit_counts[c] >= settings.recall_gate for c in targets):
        # Do not train already-functional classes or select a fallback using CAL.
        return dict(base, status='REGISTRATION_GAP',
                    reason='Availability-only already meets recall for at least one target; no weight trial')
    if settings.transfer_mode == 'direct_fc2':
        return _direct_session(base, reference, selection, models, detectors, data, seen_classes,
                               settings, ledger, announcements)
    panels, sketches = [], []
    for owner in participants:
        x,y = data.get(owner, 'base', task=task, per_class=settings.base_per_class)
        if not len(y):
            continue
        panels.append(encode_panel(reference, x, y, owner, settings.batch_size))
        packet = receiver_sketches(reference, x, y, owner, batch_size=settings.batch_size,
                                   per_class=settings.sketch_per_class)
        if owner != receiver:
            packet = ledger.transmit(owner, receiver, 'router_sketches', packet)
        sketches.append(packet)
    if not any(p['owner'] == receiver for p in panels):
        return dict(base, status='UNKNOWN', reason='No current receiver BASE negatives')
    staged_router = stage_router(reference, targets, sketches)
    permissions = head_permissions(reference, targets)
    mask_only_model = stage_head(reference, dict(targets=targets, permissions=permissions,
                                                head=initial_head(reference, targets, permissions)))
    full = learn_head(reference, panels, targets, config=settings, ledger=ledger, include_receiver=True)
    donor_only = learn_head(reference, panels, targets, config=settings, ledger=ledger, include_receiver=False)
    full_model = stage_head(reference, full)
    donor_model = stage_head(reference, donor_only)
    conditions = dict(availability_only=(reference.model, available),
                      router_only=(reference.model, staged_router),
                      mask_only=(mask_only_model, staged_router),
                      donor_only=(donor_model, staged_router), full=(full_model, staged_router))
    return _verify_and_commit(base, reference, conditions, staged_router, participants, data,
        seen_classes, settings, ledger, models, detectors,
        training_details=dict(head_training=full['history'], donor_only_training=donor_only['history']))


def _direct_session(base, reference, selection, models, detectors, data, seen_classes,
                    settings, ledger, announcements):
    from .direct_fc2 import export_row, aggregate_rows, RowCompatibilityError
    receiver, task = base['receiver'], base['task']
    targets, donors = base['targets'], base['donors']
    packets, quality, quality_evidence = [], {}, {}
    event_start = len(ledger.events)
    for donor in donors:
        donor_targets = [c for c in targets if c in announcements[donor].current_classes]
        local_seen = getattr(models[donor], 'candle_v3_state', {}).get('local_seen', {}).get(str(task), {})
        if any(local_seen.get(str(c), {}).get('positive_optimizer_rounds', 0) <= 0
               or models[donor].unit_ranks['fc2'][c] < 2 for c in donor_targets):
            return dict(base, status='UNKNOWN', reason='Selected donor lacks current trained/mature row evidence')
        x,y = data.get(donor, 'fit', task=task, per_class=settings.fit_per_class)
        prediction,_ = predict(models[donor], detectors[donor], x, seen_classes=seen_classes,
                                batch_size=settings.batch_size)
        labels = y.numpy()
        evidence = {c:dict(n=int((labels==c).sum()), correct=int(((labels==c)&(prediction==c)).sum()))
                    for c in donor_targets}
        evidence = ledger.transmit(donor, receiver, 'verification_receipt',
                                    dict(role='donor_native_FIT_quality', counts=evidence))['counts']
        quality_evidence[donor] = evidence
        for c in donor_targets:
            if evidence[c]['n'] < settings.min_fit_positive:
                return dict(base, status='UNKNOWN', reason='Insufficient donor quality FIT',
                            donor_quality=quality_evidence)
            # Native ranking is not a final transfer eligibility gate. For one
            # donor no averaging weight is needed; for multiple use measured quality.
            quality[(donor,c)] = evidence[c]['correct']/max(evidence[c]['n'],1)
            packet = export_row(models[donor], donor=donor, task=task, class_id=c,
                                announcement=announcements[donor])
            packets.append(ledger.transmit(donor, receiver, 'fc2_row', packet))
    if len(donors) == 1:
        quality = {(donors[0],c):1. for c in targets}
    elif any(q <= 0 for q in quality.values()):
        return dict(base, status='UNKNOWN', reason='No positive quality weight for multi-donor aggregation',
                    donor_quality=quality_evidence)
    try:
        imported = aggregate_rows(reference, targets, packets, quality=quality)
    except RowCompatibilityError as exc:
        return dict(base, status='INCOMPATIBLE', reason=str(exc), donor_quality=quality_evidence)
    sketches = []
    participants = [receiver]+donors
    for owner in participants:
        x,y = data.get(owner, 'base', task=task, per_class=settings.base_per_class)
        if not len(y):
            continue
        packet = receiver_sketches(reference, x, y, owner, batch_size=settings.batch_size,
                                   per_class=settings.sketch_per_class)
        if owner != receiver:
            packet = ledger.transmit(owner, receiver, 'router_sketches', packet)
        sketches.append(packet)
    staged_router = stage_router(reference, targets, sketches)
    full_model = stage_head(reference, imported)
    row_index = torch.as_tensor(targets, device=reference.model.fc2.weight.device)
    initial = torch.cat([reference.model.fc2.weight.detach().index_select(0,row_index).cpu(),
                         reference.model.fc2.bias.detach().index_select(0,row_index).cpu().unsqueeze(1)],dim=1)
    mask_only = stage_head(reference, dict(imported, head=initial))
    available = _available_detector(reference, targets)
    conditions = dict(availability_only=(reference.model, available),
                      router_only=(reference.model, staged_router),
                      mask_only=(mask_only, staged_router),
                      direct_native_router=(full_model, available),
                      full=(full_model, staged_router))
    details = dict(optimizer_steps=0, head_training=[], donor_only_training=[],
        receiver_row_weight=0., native_neighborhood_only=True, row_provenance=imported['provenance'],
        donor_quality=quality_evidence,
        raw_row_parameter_bytes=sum(len(p['row_fp32']) for p in packets),
        packed_row_mask_bytes=sum(len(p['weight_mask'])+1 for p in packets),
        serialized_row_message_bytes=sum(e['bytes'] for e in ledger.events[event_start:] if e['kind']=='fc2_row'),
        communication_note='Reference for FIT/CAL and receiver-space sketches remains counted; row payload is not total traffic')
    return _verify_and_commit(base, reference, conditions, staged_router, participants, data,
        seen_classes, settings, ledger, models, detectors, training_details=details)


def _verify_and_commit(base, reference, conditions, staged_router, participants, data,
                       seen_classes, settings, ledger, models, detectors, *, training_details):
    receiver, task = base['receiver'], base['task']
    targets, session_id = base['targets'], base['session_id']
    metrics = {name:{} for name in conditions}
    # Deduplicate CAL content across participants before counting pooled evidence.
    used = set()
    for owner in participants:
        x,y = data.get(owner, 'calibration', task=task, per_class=settings.cal_per_class)
        identities = [digest([x[row], y[row]]) for row in range(len(y))]
        if owner != receiver:
            identities = ledger.transmit(owner, receiver, 'verification_receipt',
                dict(role='CAL_content_ids', session_id=session_id, content_ids=identities))['content_ids']
        keep = []
        for row,identity in enumerate(identities):
            if identity not in used:
                used.add(identity); keep.append(row)
        x,y = x[keep],y[keep]
        baseline,_ = predict(reference.model, reference.detector, x, seen_classes=seen_classes,
                             batch_size=settings.batch_size)
        if owner != receiver:
            # Receiver reference is already cached at this peer; send only the candidate changes.
            ledger.transmit(receiver, owner, 'verification_request', dict(session_id=session_id,
                snapshot=asdict(reference.identity), candidate_rows={name:dict(weight=model.fc2.weight[targets].detach().cpu(),
                    bias=model.fc2.bias[targets].detach().cpu(),
                    masks=model.weight_masks['fc2'][targets].detach().cpu(),
                    bias_masks=model.bias_masks['fc2'][targets].detach().cpu())
                    for name,(model,_route) in conditions.items()},
                targets=targets, router=staged_router,
                unique_cal_rows=keep,
                protocol='locked conditions including mask-only; current CAL only'))
        local_receipt = {}
        for name,(model,route) in conditions.items():
            prediction,episodes = predict(model, route, x, seen_classes=seen_classes, batch_size=settings.batch_size)
            local_receipt[name] = accounting(y.numpy(), baseline, prediction, episodes, targets=targets, task=task)
        if owner != receiver:
            local_receipt = ledger.transmit(owner, receiver, 'verification_receipt', local_receipt)
        for name,row in local_receipt.items():
            metrics[name][owner] = row
    decision = acceptance(metrics['full'], metrics, config=settings, receiver=receiver, targets=targets)
    receipt = dict(base, **decision, controls=metrics, **training_details,
                   fit_role='FIT discovery only',
                   training_role=('none; current BASE used for router sketches only' if base['transfer_mode']=='direct_fc2'
                                  else 'current BASE only'), acceptance_role='current CAL only')
    if receipt['status'] == 'PASS':
        commit(models[receiver], detectors[receiver], reference, conditions['full'][0], staged_router, owner=receiver, receipt=receipt)
    return receipt
