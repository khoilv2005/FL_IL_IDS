"""Non-oracle routed evaluation and empirical current-scope acceptance."""
from collections import defaultdict
import numpy as np
import torch
from fed_learning.training.denice_eval import _denice_routed_logits_with_episodes


@torch.no_grad()
def predict(model, detector, inputs, *, seen_classes, batch_size):
    device = next(model.parameters()).device
    prediction, routes = [], []
    for start in range(0, len(inputs), batch_size):
        logits, episode = _denice_routed_logits_with_episodes(model, inputs[start:start+batch_size].to(device),
            detector, seen_classes, device, inference_policy='pred_hard')
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite routed verification logits')
        prediction.extend(logits.argmax(1).cpu().tolist())
        routes.extend(np.asarray(episode, dtype=np.int64).tolist() if episode is not None
                      else [-1] * len(logits))
    return np.asarray(prediction, dtype=np.int64), np.asarray(routes, dtype=np.int64)


def accounting(labels, baseline, candidate, routes, *, targets, task):
    y = np.asarray(labels, dtype=np.int64)
    if any(len(v) != len(y) for v in (baseline, candidate, routes)):
        raise ValueError('Metrics must use the same pool')
    correct_before, correct_after = baseline == y, candidate == y
    positive = np.isin(y, targets)
    negative = ~positive
    activated = np.isin(candidate, targets)
    rescue = int((~correct_before & correct_after).sum())
    broken = int((correct_before & ~correct_after).sum())
    if int(correct_after.sum()) - int(correct_before.sum()) != rescue - broken:
        raise RuntimeError('Rescue/break accounting mismatch')
    by_class = {}
    for c in sorted(set(y.tolist())):
        rows = y == c
        by_class[int(c)] = dict(n=int(rows.sum()), correct=int(correct_after[rows].sum()),
            false_activation=int((rows & negative & activated).sum()),
            break_count=int((rows & negative & correct_before & ~correct_after).sum()))
    return dict(n=len(y), positive=int(positive.sum()), negative=int(negative.sum()),
        correct=int(correct_after.sum()), baseline_correct=int(correct_before.sum()), rescue=rescue, break_count=broken,
        negative_break=int((negative & correct_before & ~correct_after).sum()),
        changed_wrong=int((~correct_before & ~correct_after & (candidate != baseline)).sum()),
        harmless=int((correct_before & correct_after).sum()),
        target_correct=int((positive & correct_after).sum()),
        false_activation=int((negative & activated).sum()),
        target_predictions=int(activated.sum()), route_correct=int((routes == task).sum()), by_class=by_class)


def acceptance(owner_metrics, control_metrics, *, config, receiver, targets):
    """Evidence is empirical/current only. Missing evidence never counts as PASS."""
    totals = defaultdict(int)
    reasons, unknown = [], []
    receiver_negatives = 0
    for owner, row in owner_metrics.items():
        for key in ('n', 'positive', 'negative', 'correct', 'baseline_correct', 'rescue',
                    'break_count', 'negative_break', 'target_correct', 'false_activation'):
            totals[key] += row[key]
        if owner == receiver:
            receiver_negatives += row['negative']
        for c, value in row['by_class'].items():
            if int(c) not in targets and value['n'] and value['false_activation']/value['n'] > config.far_gate:
                reasons.append(f'owner={owner}/class={c}: FAR exceeded')
    if totals['positive'] < config.min_cal_positive:
        unknown.append('Insufficient positive CAL')
    if totals['negative'] < config.min_cal_negative or receiver_negatives < config.min_cal_negative:
        unknown.append('Insufficient receiver/current negative CAL')
    target_rows = {c: [r['by_class'][c] for r in owner_metrics.values() if c in r['by_class']] for c in targets}
    for c, rows in target_rows.items():
        count = sum(r['n'] for r in rows)
        recall = sum(r['correct'] for r in rows)/max(count,1)
        if count < config.min_cal_positive:
            unknown.append(f'class={c}: insufficient positive CAL')
        if recall < config.recall_gate:
            reasons.append(f'class={c}: recall below gate')
    if totals['false_activation']/max(totals['negative'],1) > config.far_gate:
        reasons.append('Pooled FAR exceeded')
    if totals['negative_break']:
        reasons.append('Negative break detected')
    if totals['correct'] < totals['baseline_correct']:
        reasons.append('Candidate reduces current-pool accuracy')
    candidate_recall = totals['target_correct']/max(totals['positive'],1)
    for name in ('availability_only', 'router_only', 'mask_only'):
        rows = control_metrics.get(name, {})
        count = sum(r['positive'] for r in rows.values())
        if count != totals['positive'] or set(rows) != set(owner_metrics):
            raise ValueError('Control and candidate evidence pools differ')
        old_recall = sum(r['target_correct'] for r in rows.values())/max(count,1)
        if candidate_recall - old_recall < config.min_gain:
            reasons.append(f'No locked minimum gain over {name}')
    return dict(status='UNKNOWN' if unknown else 'FAIL' if reasons else 'PASS',
                reasons=reasons, missing_evidence=unknown, totals=dict(totals),
                scope='observed current CAL only', historical_safety='unverified',
                statistical_far_certificate=False)
