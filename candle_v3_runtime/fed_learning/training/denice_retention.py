"""Matched-client retention diagnostics; never used to optimize on test data."""
import math


def retention_summary(history, current, task_id):
    previous = {}
    for row in history:
        if int(row['task']) >= task_id:
            continue
        for cid, client in row.get('per_client', {}).items():
            for episode, metric in client.get('per_task', {}).items():
                value = metric.get('accuracy')
                if value is not None and math.isfinite(value) and metric.get('sample_count', 0) > 0:
                    key = (int(cid), int(episode))
                    previous[key] = max(previous.get(key, value), value)
    by_task, forgetting = {}, []
    for cid, client in current.get('per_client', {}).items():
        for episode, metric in client.get('per_task', {}).items():
            value = metric.get('accuracy')
            if value is None or not math.isfinite(value) or metric.get('sample_count', 0) <= 0:
                continue
            episode = int(episode)
            by_task.setdefault(episode, []).append(value)
            key = (int(cid), episode)
            if episode < task_id and key in previous:
                forgetting.append(previous[key] - value)
    return {'avg_forgetting': sum(forgetting) / len(forgetting) if forgetting else None,
            'forgetting_matched_client_task_count': len(forgetting),
            'per_task_accuracy': {ep: sum(values) / len(values) for ep, values in by_task.items()},
            'forgetting_definition': 'mean_previous_best_minus_current_on_matched_client_task_pairs'}
