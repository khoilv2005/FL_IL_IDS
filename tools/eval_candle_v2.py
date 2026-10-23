"""Terminal CANDLE V2 evaluation with an explicitly versioned inference policy."""
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from appliance.runner import load_input
from eval_checkpoint import _make_denice_client_model
from fed_learning.data.denice_clean_roles import CleanRoleData, file_sha256
from fed_learning.data.incremental_loader import IncrementalDataLoader
from fed_learning.strategies.decentralized.candle_v2 import state, validate
from fed_learning.strategies.decentralized.candle_v2_inference import predict, resolve_policy, PRIMARY, TASKFREE
from fed_learning.training.decentralized_denice_il import _partition_test_data_by_client


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def statistics(cm):
    correct = np.diag(cm)
    support = cm.sum(1)
    predicted = cm.sum(0)
    f1 = np.divide(2 * correct, support + predicted, out=np.zeros(34), where=support+predicted > 0)
    recall = np.divide(correct, support, out=np.zeros(34), where=support > 0)
    precision = np.divide(correct, predicted, out=np.zeros(34), where=predicted > 0)
    return dict(samples=int(cm.sum()), accuracy=float(correct.sum()/cm.sum()),
                macro_f1_34=float(f1.mean()), macro_recall_34=float(recall.mean()),
                macro_precision_34=float(precision.mean()), per_class_f1=f1.tolist(),
                per_class_recall=recall.tolist(), class_counts=support.tolist())


@torch.no_grad()
def run(checkpoint, roles_dir, data_dir, out, device='cuda', batch_size=4096,
        inference_policy=None, expected_taskfree_confusion=None, validation_panel=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    ckpt, source = load_input(Path(checkpoint), 5, 19)
    config = ckpt['config']
    if not config.get('candle_v2_enabled'):
        raise ValueError('Expected a CANDLE V2 checkpoint')
    validate(config)
    policy = resolve_policy(config, inference_policy)
    roles = CleanRoleData(roles_dir, source_data_dir=data_dir)
    if config['denice_data_roles_sha256'] != file_sha256(roles.root/'role_manifest.json'):
        raise ValueError('Evaluation role identity differs from training')
    ids = sorted(map(int, ckpt['client_model_states']))
    evaluation_role = 'validation_diagnostic' if validation_panel is not None else 'full_TEST'
    if validation_panel is not None and expected_taskfree_confusion is not None:
        raise ValueError('Cannot compare a validation panel with the original full TEST confusion')
    write(out/'pipeline_lock.json', dict(method='CANDLE V2', source=source,
        primary=PRIMARY[policy], inference_policy=policy, scale=config['candle_v2_logit_scale'],
        task=5, round=19, oracle=False, donor_inference=False, task_router=policy != TASKFREE,
        policy_override=inference_policy is not None,
        training_inference_policy=resolve_policy(config),
        inference_module_sha256=file_sha256(ROOT/'fed_learning/strategies/decentralized/candle_v2_inference.py'),
        evaluator_sha256=file_sha256(__file__),
        class_authorization='Saved CANDLE available + class_tasks; never true task at inference',
        evaluation_role=evaluation_role,
        validation_panel_sha256=file_sha256(validation_panel) if validation_panel is not None else None,
        test_partition=('V1-matched disjoint receiver shards, each global row exactly once'
                        if validation_panel is None else 'Diagnostic validation shards, each panel row exactly once'),
        receivers=ids, role_manifest_sha256=config['denice_data_roles_sha256']))
    # The lock is written before evaluation data access. Development mode never
    # instantiates the original TEST loader or accesses global_test_data.npz.
    if validation_panel is not None:
        with np.load(validation_panel, allow_pickle=False) as panel:
            x = torch.from_numpy(np.asarray(panel['x'], dtype=np.float32))
            y = torch.from_numpy(np.asarray(panel['y'], dtype=np.int64))
    else:
        loader = IncrementalDataLoader(str(roles.source))
        x, y = loader.get_full_test_data()
    if not np.array_equal(np.unique(y.numpy()), np.arange(34)):
        raise ValueError('Expected the original 34-class full TEST')
    expected = len(y)
    shards, partition = _partition_test_data_by_client(
        x, y, ids, seed=int(config.get('seed', 42)) + 104729*5, lazy_features=True)
    del x, y
    if validation_panel is None:
        del loader
    gc.collect()
    cm = np.zeros((34, 34), dtype=np.int64)
    baseline_cm = np.zeros_like(cm)
    rescue = 0
    broken = 0
    client_rows = []
    evaluated = 0
    covered = 0
    started = time.monotonic()
    route_correct = 0
    route_rows = 0
    fallback_rows = 0
    label_task = {int(c):int(t) for t,classes in roles.manifest.get('task_classes', {}).items() for c in classes}
    if not label_task:
        # Official dataset metadata defines all tasks, including labels that a
        # receiver has never learned. Used only for reporting route accuracy.
        metadata = json.loads((roles.source/'metadata.json').read_text(encoding='utf-8'))
        label_task = {int(c):int(t) for t,classes in metadata['task_structure']['task_classes'].items() for c in classes}
    for pos, cid in enumerate(ids, 1):
        model, router = _make_denice_client_model(ckpt, cid, device)
        model.eval()
        if model.adapter_registry:
            raise ValueError('Task-free CANDLE V2 cannot silently select context-specific adapters')
        shard = shards[cid]
        labels = shard['y_test']
        inputs = shard['X_test']
        local_cm = np.zeros_like(cm)
        local_baseline_cm = np.zeros_like(cm)
        local_rescue = 0
        local_broken = 0
        available = state(model)['available']
        local_covered = 0
        local_route_correct = 0
        local_route_rows = 0
        local_fallback = 0
        for start in range(0, len(labels), batch_size):
            truth = labels[start:start+batch_size].cpu().numpy().astype(np.int64)
            batch = inputs[start:start+batch_size].to(device)
            prediction = predict(model, batch, config, router=router, policy=policy)
            pred = prediction['logits'].argmax(1).cpu().numpy()
            base_pred = prediction['unrouted_logits'].argmax(1).cpu().numpy()
            local_baseline_cm += np.bincount(truth*34 + base_pred, minlength=34*34).reshape(34, 34)
            local_rescue += int(((base_pred != truth) & (pred == truth)).sum())
            local_broken += int(((base_pred == truth) & (pred != truth)).sum())
            if prediction['episodes'] is not None:
                actual_tasks = np.asarray([label_task[int(c)] for c in truth])
                local_route_correct += int((prediction['episodes'] == actual_tasks).sum())
                local_route_rows += len(truth)
            local_fallback += int(prediction['fallback'].sum())
            local_cm += np.bincount(truth*34 + pred, minlength=34*34).reshape(34, 34)
            local_covered += int(np.isin(truth, available).sum())
            evaluated += len(truth)
            if start % (100*batch_size) == 0:
                print(f'CANDLE V2 {evaluation_role} {evaluated}/{expected}; receiver={cid}', flush=True)
        covered += local_covered
        route_correct += local_route_correct
        route_rows += local_route_rows
        fallback_rows += local_fallback
        cm += local_cm
        baseline_cm += local_baseline_cm
        rescue += local_rescue
        broken += local_broken
        client_rows.append(dict(client_id=cid, available_classes=available,
                                covered_rows=local_covered,
                                paired_taskfree_metrics=statistics(local_baseline_cm),
                                rescue=local_rescue, breaks=local_broken,
                                route_accuracy=local_route_correct/local_route_rows if local_route_rows else None,
                                empty_scope_fallback_rows=local_fallback, **statistics(local_cm)))
        print(f'Evaluated receiver {pos}/{len(ids)}: accuracy={client_rows[-1]["accuracy"]:.4%}', flush=True)
        del model, router
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if evaluated != expected or cm.sum() != expected:
        raise ValueError('Full TEST row accounting mismatch')
    if int(np.trace(cm)-np.trace(baseline_cm)) != rescue-broken:
        raise ValueError('Paired rescue/break accounting mismatch')
    baseline_match = None
    if expected_taskfree_confusion is not None:
        baseline_match = np.array_equal(baseline_cm, np.asarray(expected_taskfree_confusion, dtype=np.int64))
        if not baseline_match:
            write(out/'taskfree_baseline_mismatch.json', dict(
                computed=baseline_cm.tolist(), expected=np.asarray(expected_taskfree_confusion).tolist()))
            raise ValueError('Task-free baseline differs from original results; inspect source/data before claiming a repair')
    write(out/'confusion_matrix.json', cm.tolist())
    write(out/'paired_taskfree_confusion_matrix.json', baseline_cm.tolist())
    write(out/'per_client.json', client_rows)
    result = dict(completed=True, experiment='CANDLE V2', primary=PRIMARY[policy], inference_policy=policy,
        metrics=statistics(cm), test_rows=expected if validation_panel is None else None,
        evaluation_role=evaluation_role, input_rows=expected, evaluated_rows=evaluated,
        label_coverage=covered/expected, covered_rows=covered,
        mean_client_accuracy=float(np.mean([r['accuracy'] for r in client_rows])),
        mean_client_macro_f1=float(np.mean([r['macro_f1_34'] for r in client_rows])),
        route_accuracy=route_correct/route_rows if route_rows else None,
        route_accuracy_reason='Saved input-driven binary router' if route_rows else 'Task-free inference; no task prediction',
        empty_scope_fallback_rows=fallback_rows,
        paired_taskfree_control=statistics(baseline_cm), rescue=rescue, breaks=broken,
        original_taskfree_confusion_matches=baseline_match,
        partition=partition, runtime_seconds=time.monotonic()-started, source=source,
        limitation='Engineering run, not a population safety certificate or untouched confirmation')
    write(out/'completion.json', result)
    return result


if __name__ == '__main__':
    import argparse
    from fed_learning.strategies.decentralized.candle_v2_inference import POLICIES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--roles', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--inference-policy', choices=POLICIES)
    parser.add_argument('--validation-panel', help='Explicit offline diagnostic; never reads final TEST')
    args = parser.parse_args()
    run(args.checkpoint, args.roles, args.data_dir, args.out, args.device,
        args.batch_size, args.inference_policy, validation_panel=args.validation_panel)
