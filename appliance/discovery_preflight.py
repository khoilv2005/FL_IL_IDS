"""Metadata-only receiver checks before any donor choice or CAL HOLDOUT read."""
from .active_pair_calibration import split_counts
from .config import Rejected


def receiver_preflight(cid, model, view, shield, seen):
    task = view.task
    classes = view.store['task_classes'][str(task)]
    required = sorted(int(c) for c in seen if c not in classes and int(model.unit_ranks['fc2'][c]) >= 2)
    missing = shield.coverage(required)['missing_classes']
    counts = view.store['clients'][str(cid)][str(task)]['class_counts']
    negative_counts = [sum(split_counts(int(n))[i] for n in counts.values()) for i in range(3)]
    reason = ('receiver_already_has_capability' if getattr(model, 'appliance_guarded_head_entries', {}) else
              'missing_owned_old_BASE_support' if missing else
              'insufficient_current_CAL_splits' if min(negative_counts) < 32 else None)
    return dict(receiver=cid, task=task, eligible=reason is None, reason=reason,
                required_old_classes=required, missing_owned_classes=missing,
                current_negative_split_counts=negative_counts, raw_CAL_opened=False,
                inherited_mature_rows_are_owned_evidence=False)


def schedule_requests(requests, offers, groups, alphas, task, round_id, preflight,
                      max_transactions=16, max_receivers_per_class=8):
    for value in (max_transactions, max_receivers_per_class):
        if type(value) is not int or value < 1:
            raise Rejected('INVALID_DISCOVERY_TRANSACTION_BUDGET')
    selected, used, coverage, rejected = [], set(), {}, []
    # Layered rounds across classes prevent one populous class consuming the
    # whole callback budget. Every choice uses FIT and metadata, never HOLDOUT.
    queues = {}
    for c in sorted({r['class_id'] for r in requests}):
        options = []
        for request in sorted((r for r in requests if r['class_id'] == c), key=lambda r: r['receiver']):
            cid = request['receiver']
            gate = preflight[cid]
            if not gate['eligible']:
                rejected.append(dict(receiver=cid, class_id=c, reason=gate['reason']))
                continue
            if request['task'] != task:
                raise Rejected('REQUEST_TASK_CHANGED')
            candidates = [o for o in offers if o['class_id'] == c and o['task'] == task and
                          o['donor'] in groups[cid] and alphas[cid].get(o['donor'], 0) > 0]
            if candidates:
                offer = min(candidates, key=lambda o: (-o['quality_lcb'], o['donor']))
                options.append(dict(receiver=cid, donor=offer['donor'], class_id=c, task=task,
                    round=round_id, selection_scope='current_local_fit', quality_lcb=offer['quality_lcb'],
                    donor_model_fingerprint=offer['model_fingerprint'],
                    rule='metadata preflight; class round-robin; receiver ID; donor FIT LCB; no HOLDOUT fallback'))
        queues[c] = options
        coverage[c] = dict(eligible_current_receivers=len(options), selected_receivers=0)
    while len(selected) < max_transactions:
        added = False
        for c, options in queues.items():
            if coverage[c]['selected_receivers'] >= max_receivers_per_class:
                continue
            while options and options[0]['receiver'] in used:
                options.pop(0)
            if options:
                pair = options.pop(0)
                selected.append(pair)
                used.add(pair['receiver'])
                coverage[c]['selected_receivers'] += 1
                added = True
                if len(selected) == max_transactions:
                    break
        if not added:
            break
    return selected, coverage, rejected
