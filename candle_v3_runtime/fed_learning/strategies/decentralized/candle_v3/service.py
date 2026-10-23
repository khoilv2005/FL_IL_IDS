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
                      preprocessing_sha256, output_dir, device, allowed_edges):
    settings = TeachingConfig.from_config(config)
    if not settings.enabled:
        return dict(enabled=False)
    ids = sorted(set(int(c) for c in active_ids))
    if set(allowed_edges) != set(ids):
        raise ValueError('Teaching graph must explicitly declare every active receiver')
    if any(set(peers)-set(ids) for peers in allowed_edges.values()):
        raise ValueError('Teaching graph includes undeclared peers')
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
            for donor in sorted(set(allowed_edges[receiver]) - {receiver}):
                peers[donor] = ledger.transmit(donor, receiver, 'announcement', announcements[donor])
            eligible = sorted(c for peer in peers.values() for c in peer.current_classes
                              if c not in local.current_classes and models[receiver].unit_ranks['fc2'][c] == 0)
            selection = select_teachers(receiver, {receiver:local, **peers}, allowed_edges[receiver],
                                        eligible, max_donors=settings.max_donors,
                                        max_targets=settings.max_targets)
            if not selection.targets:
                receipts.append(dict(receiver=receiver, status='SKIP', reason='no complementary candidate'))
                record['completed_tasks'].append(int(task))
                continue
            receipt = _session(receiver, selection, models, detectors, data, task, seen_classes,
                               preprocessing_sha256, settings, ledger, device)
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
             preprocessing_sha256, settings, ledger, device):
    reference = freeze_receiver(models[receiver], detectors[receiver], receiver, task, preprocessing_sha256, device)
    targets, donors = list(selection.targets), list(selection.donors)
    participants = [receiver]+donors
    session_id = digest(dict(reference=asdict(reference.identity), targets=targets, donors=donors))
    base = dict(receiver=receiver, task=task, session_id=session_id, targets=targets, donors=donors,
                snapshot=asdict(reference.identity), scope='observed current CAL only')
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
                snapshot=asdict(reference.identity), full_head=full['head'], donor_head=donor_only['head'],
                targets=targets, permissions=full['permissions'], router=staged_router,
                unique_cal_rows=keep,
                protocol='five locked conditions including mask-only; current CAL only'))
        local_receipt = {}
        for name,(model,route) in conditions.items():
            prediction,episodes = predict(model, route, x, seen_classes=seen_classes, batch_size=settings.batch_size)
            local_receipt[name] = accounting(y.numpy(), baseline, prediction, episodes, targets=targets, task=task)
        if owner != receiver:
            local_receipt = ledger.transmit(owner, receiver, 'verification_receipt', local_receipt)
        for name,row in local_receipt.items():
            metrics[name][owner] = row
    decision = acceptance(metrics['full'], metrics, config=settings, receiver=receiver, targets=targets)
    receipt = dict(base, **decision, controls=metrics,
                   head_training=full['history'], donor_only_training=donor_only['history'],
                   fit_role='FIT discovery only', training_role='current BASE only', acceptance_role='current CAL only')
    if receipt['status'] == 'PASS':
        commit(models[receiver], detectors[receiver], reference, full_model, staged_router, owner=receiver, receipt=receipt)
    return receipt
