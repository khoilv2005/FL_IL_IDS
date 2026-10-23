"""Read-only audit of completed CANDLE V2 Kaggle outputs; no model inference/training."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--baseline', type=Path, default=ROOT/'audit_denice/appliance_legacy_results11/inputs/legacy_self_task_5/completion.json')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    baseline = json.loads(a.baseline.read_text(encoding='utf-8'))
    with zipfile.ZipFile(a.archive) as archive:
        matches = [n for n in archive.namelist() if n.endswith('/candle_v2_completion.json')]
        if len(matches) != 1:
            raise ValueError('Expected one CANDLE V2 output root')
        prefix = matches[0][:-len('candle_v2_completion.json')]
        checksums = {}
        def read_bytes(name):
            data = archive.read(prefix+name)
            checksums[name] = hashlib.sha256(data).hexdigest()
            return data
        def read(name):
            return json.loads(read_bytes(name))
        def lines(name):
            return [json.loads(row) for row in read_bytes(name).decode('utf-8').splitlines() if row]
        completion = read('candle_v2_completion.json')
        duplicate = read('candle_v2_self_task_5/completion.json')
        if completion != duplicate or not completion['completed']:
            raise ValueError('Evaluation not complete or duplicated receipts disagree')
        protocol = read('candle_v2_protocol.json')
        config = read('config.json')
        cm = np.asarray(read('candle_v2_self_task_5/confusion_matrix.json'), np.int64)
        clients = read('candle_v2_self_task_5/per_client.json')
        train = lines('candle_v2_training.jsonl')
        health = lines('optimizer_health.jsonl')
        communication = lines('candle_v2_communication.jsonl')
        n, correct = int(cm.sum()), int(np.trace(cm))
        if n != completion['test_rows'] or n != completion['evaluated_rows']:
            raise ValueError('Evaluation row accounting mismatch')
        f1 = np.divide(2*np.diag(cm), cm.sum(0)+cm.sum(1), out=np.zeros(34), where=cm.sum(0)+cm.sum(1)>0)
        if not np.isclose(correct/n, completion['metrics']['accuracy']) or not np.isclose(f1.mean(), completion['metrics']['macro_f1_34']):
            raise ValueError('Metrics disagree with confusion matrix')
        per_task = []
        for t in range(6):
            classes = list(range(t*6, min(t*6+6, 34)))
            rows = [r for r in train if r['task'] == t]
            h = [r for r in health if r['task'] == t]
            comm = [r for r in communication if r['task'] == t]
            first = [r for r in rows if r['round'] == 0]
            last = [r for r in rows if r['round'] == 19]
            if sorted(set(r['round'] for r in rows)) != list(range(20)) or len(comm) != 20:
                raise ValueError('Incomplete task/round logs')
            unique = {(r['client_id'], c) for r in rows if r['optimizer_steps'] > 0 for c in r['imported_classes']}
            total = int(cm[classes].sum())
            per_task.append(dict(task=t, classes=classes, eval_rows=total,
                correct=int(np.diag(cm)[classes].sum()), accuracy=float(np.diag(cm)[classes].sum()/total),
                predicted_into_task=int(cm[:, classes].sum()), predictions_to_last_task=int(cm[np.ix_(classes, list(range(30, 34)))].sum()),
                active_clients=len(set(r['client_id'] for r in rows)), train_records=len(rows),
                round0_real_CE=float(np.mean([r['real_CE'] for r in first])),
                round19_real_CE=float(np.mean([r['real_CE'] for r in last])),
                round19_losses={k:float(np.mean([r['losses'][k] for r in last])) for k in ('alignment','virtual_CE','preservation')},
                zero_step_records=sum(r['optimizer_steps']==0 for r in rows),
                optimizer_steps=sum(r['optimizer_steps'] for r in h), skipped_steps=sum(r['skipped_optimizer_steps'] for r in h),
                successful_virtual_training_pairs=len(unique), virtual_training_records=sum(bool(r['imported_classes']) for r in rows),
                prototype_JSON_payload_bytes=sum(r['prototype_JSON_payload_bytes'] for r in comm)))
        old_rows = int(cm[:30].sum())
        old_correct = int(np.diag(cm)[:30].sum())
        predictions = cm.sum(0)
        baseline_rows = baseline['metrics']['BinarySelf']['class_counts']
        if baseline_rows != cm.sum(1).tolist():
            raise ValueError('V1/V2 class supports differ')
        b = baseline['metrics']['BinarySelf']
        comparison = dict(baseline='Historical results11 DeNICE legacy / CANDLE V1 config, BinarySelf',
            V1_accuracy=b['accuracy'], V2_accuracy=correct/n,
            accuracy_delta_pp=100*(correct/n-b['accuracy']),
            V1_macro_f1=b['macro_f1_34'], V2_macro_f1=float(f1.mean()),
            macro_f1_delta_pp=100*(float(f1.mean())-b['macro_f1_34']),
            same_partition_receipt=baseline['partition']==completion['partition'],
            same_class_supports=True,same_role_manifest=protocol['role_manifest_sha256']=='b89cd6868f4185d3307f1bbed56dafd170a10c73a1752496aeb944b037d98a15',
            limitation='Historical comparator, no multi-seed or controlled component ablation')
        result = dict(completed=True, archive=str(a.archive), archive_bytes=a.archive.stat().st_size,
            source_member_sha256=checksums, training_protocol=protocol, requested_config=config,
            evaluation=dict(metrics=completion['metrics'],correct=correct, rows=n,
                mean_client_macro_f1=completion['mean_client_macro_f1'],label_coverage=completion['label_coverage'],
                covered_rows=completion['covered_rows'], accuracy_on_available_rows=correct/completion['covered_rows'],
                errors_within_available=completion['covered_rows']-correct,
                unavailable_rows=n-completion['covered_rows'],
                old_task_rows=old_rows,old_task_correct=old_correct,old_task_accuracy=old_correct/old_rows,
                task5_true_fraction=float(cm[30:].sum()/n), task5_prediction_fraction=float(cm[:,30:].sum()/n),
                class32_prediction_fraction=float(predictions[32]/n),
                top_predicted_classes=sorted([dict(class_id=c,predictions=int(v)) for c,v in enumerate(predictions)], key=lambda r:-r['predictions']),
                client_accuracy_quantiles=np.quantile([r['accuracy'] for r in clients],[0,.25,.5,.75,1]).tolist(),
                available_class_count_quantiles=np.quantile([len(r['available_classes']) for r in clients],[0,.25,.5,.75,1]).tolist(),
                test_runtime_seconds=completion['runtime_seconds'], route_accuracy=None),
            tasks=per_task, comparison=comparison,
            optimizer=dict(successful_steps=sum(r['optimizer_steps'] for r in health),
                skipped_steps=sum(r['skipped_optimizer_steps'] for r in health),
                zero_step_records=sum(r['optimizer_steps']==0 for r in train),
                nonfinite_gradient_reports=sum(bool(r.get('amp',{}).get('nonfinite_gradient_parameters_after_clipping')) for r in health)),
            communication=dict(prototype_JSON_payload_bytes=sum(r['prototype_JSON_payload_bytes'] for r in communication),
                full_wire_bytes_known=False),
            all120_rounds_logged=True,confusion_metrics_recomputed=True,
            diagnoses=dict(observed='Severe last-task prediction bias on cumulative TEST',
                source_identified_limit='Preservation KL normalizes only over old classes on current inputs, so does not constrain old-vs-new probability mass',
                causality='Not isolated by an ablation; actual within-task representation retention still requires checkpoint diagnostics'),
            no_model_inference=True,no_retraining=True,no_production_code_changes=True)
        write(a.out/'audit_summary.json', result)
        for name in ('candle_v2_completion.json','candle_v2_protocol.json','candle_v2_training.jsonl','candle_v2_communication.jsonl','optimizer_health.jsonl','config.json'):
            (a.out/name).write_bytes(archive.read(prefix+name))
    print(json.dumps(dict(accuracy=correct/n,macro_f1=float(f1.mean()),rows=n,comparison=comparison,
        task5_prediction_fraction=result['evaluation']['task5_prediction_fraction'],
        old_accuracy=old_correct/old_rows),indent=2))


if __name__ == '__main__':
    main()
