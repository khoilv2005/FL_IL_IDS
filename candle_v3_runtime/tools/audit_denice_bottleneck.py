"""Read-only, full-test bottleneck audit of pure legacy DeNICE.

Oracle policies are diagnostics. No fitting, raw train/CAL reads, or transfer.
The fast path is restricted to adapter-free models and checked against the
native evaluator and every saved full-test prediction. Numerical disagreement
is reported, never hidden or used to silently change the source protocol.
"""
import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
import torch
import sklearn
from threadpoolctl import threadpool_limits

from appliance.runner import load_input
from appliance.state import complete_hash
from eval_checkpoint import _make_denice_client_model
from fed_learning.data.denice_clean_roles import file_sha256
from fed_learning.training.checkpoint_state import restore_context_detector
from fed_learning.training.denice_eval import (
    _allowed_classes_for_episode, _denice_routed_logits_with_episodes,
)

POLICIES = ('BinarySelf', 'MulticlassSelf')
BUCKETS = ('correct', 'coverage_unreachable', 'classifier_within_coverage', 'routing_recoverable')
RECORD_DTYPE = np.dtype([('client_id', 'u1'), ('global_test_row', '<u4'),
                         ('y_true', 'u1'), ('BinarySelf', 'u1'), ('MulticlassSelf', 'u1')])


def write_json(path, value):
    partial = path.with_suffix(path.suffix + '.partial')
    partial.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    partial.replace(path)


def metrics(cm):
    n = int(cm.sum())
    correct = cm.diagonal()
    denominator = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * correct, denominator, out=np.zeros(34), where=denominator > 0)
    return dict(samples=n, correct=int(correct.sum()), accuracy=float(correct.sum() / n),
                macro_f1_34=float(f1.mean()), class_counts=cm.sum(1).tolist(),
                recall=np.divide(correct, cm.sum(1), out=np.zeros(34), where=cm.sum(1) > 0).tolist())


def cm_add(cm, truth, pred):
    cm += np.bincount(truth.astype(int) * 34 + pred, minlength=34 * 34).reshape(34, 34)


def stage_records(inputs, n):
    path = inputs / 'records.npy'
    manifest_path = inputs / 'records_manifest.json'
    source_hash = file_sha256(inputs / 'test_predictions.csv.gz')
    if path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['source_sha256'] != source_hash or manifest['rows'] != n:
            raise ValueError('Cached receiver assignment does not match saved predictions')
        if file_sha256(path) != manifest['records_sha256']:
            raise ValueError('Cached record checksum changed')
        return np.load(path, mmap_mode='r'), manifest
    partial = path.with_suffix('.npy.partial')
    array = np.lib.format.open_memmap(partial, mode='w+', dtype=RECORD_DTYPE, shape=(n,))
    offset = 0
    for frame in pd.read_csv(inputs / 'test_predictions.csv.gz', chunksize=200_000):
        count = len(frame)
        if offset + count > n or list(frame.columns) != list(RECORD_DTYPE.names):
            raise ValueError('Unexpected prediction row count/schema')
        for name in RECORD_DTYPE.names:
            values = frame[name].to_numpy()
            maximum = n - 1 if name == 'global_test_row' else (99 if name == 'client_id' else 33)
            if (values < 0).any() or (values > maximum).any():
                raise ValueError(f'Invalid saved {name}')
            array[name][offset:offset + count] = values
        offset += count
    if offset != n:
        raise ValueError('Saved predictions do not cover full test')
    array.flush()
    del array
    partial.replace(path)
    array = np.load(path, mmap_mode='r')
    client_ids = array['client_id']
    changes = np.r_[0, np.flatnonzero(client_ids[1:] != client_ids[:-1]) + 1, n]
    ranges = {str(int(client_ids[start])): [int(start), int(end)] for start, end in zip(changes[:-1], changes[1:])}
    if len(ranges) != len(changes) - 1:
        raise ValueError('Original receiver records are not contiguous')
    seen = np.zeros(n, dtype=bool)
    for start in range(0, n, 200_000):
        rows = array['global_test_row'][start:start + 200_000]
        if len(np.unique(rows)) != len(rows) or seen[rows].any():
            raise ValueError('Repeated original global row')
        seen[rows] = True
    if not seen.all():
        raise ValueError('Missing original global row')
    manifest = dict(source_sha256=source_hash, records_sha256=file_sha256(path), rows=n,
                    receiver_ranges=ranges, partition='exact original saved row-to-receiver assignment')
    write_json(manifest_path, manifest)
    return array, manifest


def summarize_inventory(cid, detector, class_counts, task_of_class, base_class_counts):
    bank = sorted(int(t) for t, values in detector.activation_memory.items() if len(values))
    if not bank or any(t not in range(6) for t in bank):
        raise ValueError('Invalid native route bank')
    masks = np.zeros((6, 34), dtype=bool)
    fallback_tasks = []
    for task in bank:
        masks[task, _allowed_classes_for_episode(detector, task, list(range(34)), 34)] = True
        if not detector.episode_classes.get(task):
            fallback_tasks.append(task)
    union = masks.any(0)
    stable = getattr(detector, 'routing_feature_mask', None)
    if stable is None:
        stable = getattr(detector, 'stable_feature_mask', None)
    prototypes = np.array([np.mean(detector.activation_memory[t], axis=0) for t in bank])
    norm = np.linalg.norm(prototypes, axis=1)
    similarities = prototypes @ prototypes.T / np.maximum(norm[:, None] * norm[None, :], 1e-12)
    cross = similarities[np.triu_indices(len(bank), 1)]
    true_present = np.isin(task_of_class, bank)
    true_mask = masks[task_of_class, np.arange(34)]
    locally_trained = np.array([int(base_class_counts.get(str(c), 0)) > 0 for c in range(34)])
    item = dict(receiver=cid, bank=bank, classes_by_task={t: np.flatnonzero(masks[t]).tolist() for t in bank},
                fallback_all_seen_tasks=fallback_tasks, declared_classes_by_task=detector.episode_classes,
                unreachable_classes=np.flatnonzero(~union).tolist(),
                unreachable_rows=int(class_counts[~union].sum()),
                locally_trained_but_unreachable_classes=np.flatnonzero(locally_trained & ~union).tolist(),
                locally_trained_but_unreachable_rows=int(class_counts[locally_trained & ~union].sum()),
                unowned_unreachable_rows=int(class_counts[~locally_trained & ~union].sum()),
                reachable_without_local_base_classes=np.flatnonzero(~locally_trained & union).tolist(),
                missing_true_task_rows=int(class_counts[~true_present].sum()),
                true_task_class_mask_missing_rows=int(class_counts[~true_mask].sum()),
                routing_features_total=int(prototypes.shape[1]),
                routing_features_enabled=int(np.count_nonzero(stable)) if stable is not None else int(prototypes.shape[1]),
                mean_cross_task_prototype_cosine=float(cross.mean()) if len(cross) else None,
                memory_rows={t: len(detector.activation_memory[t]) for t in bank},
                router_state_fresh=detector.router_state_fresh,
                router_stale_reason=detector.router_stale_reason)
    return item, bank, masks


@torch.no_grad()
def run(args):
    inputs, legacy, out = Path(args.inputs), Path(args.legacy), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source_lock = json.loads((legacy / 'pipeline_lock.json').read_text())
    if source_lock['training_method'] != 'legacy' or source_lock['cgofed'] or source_lock['cme']:
        raise ValueError('Pure DeNICE legacy input required')
    for path, expected in ((Path(args.checkpoint), source_lock['checkpoint_sha256']),
                           (legacy / 'frozen_routers.joblib', source_lock['router_sha256'])):
        if file_sha256(path) != expected:
            raise ValueError(f'Source checksum mismatch: {path}')
    x, y = np.load(inputs / 'X_test.npy', mmap_mode='r'), np.load(inputs / 'y_test.npy', mmap_mode='r')
    metadata = json.loads((inputs / 'metadata.json').read_text())
    role_manifest = json.loads(Path(args.roles).joinpath('role_manifest.json').read_text())
    if file_sha256(inputs / 'metadata.json') != role_manifest['metadata_sha256']:
        raise ValueError('Dataset metadata differs from original training roles')
    if file_sha256(Path(args.roles) / 'role_manifest.json') != source_lock['role_manifest_sha256']:
        raise ValueError('Data role lock mismatch')
    n = len(y)
    if len(x) != n or n != 13_505_771 or not np.array_equal(np.unique(y), np.arange(34)):
        raise ValueError('Expected original full 34-class test, without a sample cap')
    records, record_manifest = stage_records(inputs, n)
    if sorted(map(int, record_manifest['receiver_ranges'])) != source_lock['receivers']:
        raise ValueError('Receiver set changed')
    protocol = dict(kind='DeNICE pure legacy full-test diagnostic', source_lock=source_lock,
                    rows=n, classes=list(range(34)), records_sha256=record_manifest['records_sha256'],
                    test_x_sha256=file_sha256(inputs / 'X_test.npy'), test_y_sha256=file_sha256(inputs / 'y_test.npy'),
                    audit_source_sha256=file_sha256(__file__), device=args.device, torch=str(torch.__version__),
                    numpy=np.__version__, sklearn=sklearn.__version__, joblib=joblib.__version__,
                    batch_size=args.batch_size, oracle_labels='diagnostics only, no prediction fitting',
                    historical_train_read=False, historical_cal_read=False, parameter_update=False,
                    legal_routes='nonempty activation_memory task entries; native mask and native fallback',
                    oracle_matched='true task only if legal; otherwise keep normal selected task',
                    best_allowed='any legal route can predict the true class; diagnostic upper bound',
                    global_task_oracle='outside legal local policy: true task and global task class mask; diagnostic only',
                    mask_only_diagnostic='same predicted task, global classes for that task; weights/ranks unchanged; not production repair',
                    exclusive_error_order=list(BUCKETS),
                    fast_path='adapter-free native one-forward logits; reuse logits across legal route masks',
                    native_equivalence_checks='first full batch of every receiver, both policies',
                    saved_prediction_comparison='every full-test row, both policies',
                    numerical_disagreement_budget_fraction=.0001,
                    limitation='Previously inspected development test source; not untouched confirmation')
    lock_path = out / 'protocol.json'
    if lock_path.exists() and json.loads(lock_path.read_text()) != protocol:
        raise ValueError('Immutable audit protocol changed; use a new output directory')
    write_json(lock_path, protocol)
    write_json(out / 'completion.json', dict(completed=False, stage='load_checkpoint'))
    ckpt, checkpoint_hashes = load_input(args.checkpoint, 5, 19)
    if ckpt['config'].get('denice_cl_method', 'legacy') != 'legacy':
        raise ValueError('Unexpected training method')
    task_of_class = np.full(34, -1, dtype=int)
    for task, classes in metadata['task_structure']['task_classes'].items():
        task_of_class[list(map(int, classes))] = int(task)
    if (task_of_class < 0).any():
        raise ValueError('Incomplete class-task mapping')
    snapshots = joblib.load(legacy / 'frozen_routers.joblib')
    inventory, clients = [], []
    start_time = time.monotonic()
    for position, cid in enumerate(source_lock['receivers'], 1):
        result_path = out / f'client_{cid}.json'
        if result_path.exists() and not args.inventory_only:
            saved = json.loads(result_path.read_text())
            clients.append(saved)
            inventory.extend(saved['inventory'])
            continue
        begin, end = record_manifest['receiver_ranges'][str(cid)]
        shard = records[begin:end]
        class_counts = np.bincount(shard['y_true'], minlength=34)
        model, original = _make_denice_client_model(ckpt, cid, args.device)
        if model.adapter_registry or model.continual_head is not None or getattr(model, 'appliance_guarded_head_entries', {}):
            raise ValueError('Fast oracle cache requires a pure adapter-free legacy model')
        detectors = {p: copy.deepcopy(original) for p in POLICIES}
        before, masks, banks, local_inventory = {}, {}, {}, []
        for policy, detector in detectors.items():
            restore_context_detector(detector, snapshots[cid][policy])
            before[policy] = complete_hash(model, detector)
            base_counts = role_manifest['clients'][str(cid)]['role_class_counts']['base']
            item, banks[policy], masks[policy] = summarize_inventory(cid, detector, class_counts, task_of_class, base_counts)
            item['policy'] = policy
            inventory.append(item)
            local_inventory.append(item)
        write_json(out / 'coverage_inventory.json', inventory)
        if args.inventory_only:
            del model, detectors, original
            continue
        cms = {p + suffix: np.zeros((34, 34), dtype=np.int64) for p in POLICIES
               for suffix in ('', '_Saved', '_OracleMatched', '_MaskOnlyDiagnostic')}
        cms['AllClassesDiagnostic'] = np.zeros((34, 34), dtype=np.int64)
        cms['OracleGlobalTaskMaskDiagnostic'] = np.zeros((34, 34), dtype=np.int64)
        tables = {p: np.zeros((34, 4), dtype=np.int64) for p in POLICIES}
        route_cms = {p: np.zeros((6, 6), dtype=np.int64) for p in POLICIES}
        discrepancies = {p: dict(saved=0, native=0, native_rows=0) for p in POLICIES}
        oracle_correct = {p: 0 for p in POLICIES}
        mask_only_transitions = {p: np.zeros((2, 4), dtype=np.int64) for p in POLICIES}
        route_label_missing = {p: 0 for p in POLICIES}
        mask_label_missing = {p: 0 for p in POLICIES}
        for offset in range(0, len(shard), args.batch_size):
            data = shard[offset:offset + args.batch_size]
            rows, truth = data['global_test_row'], data['y_true'].astype(int)
            if not np.array_equal(y[rows], truth):
                raise ValueError('Saved label/global-row alignment differs from original test')
            batch = torch.as_tensor(np.array(x[rows], copy=True), device=args.device, dtype=torch.float32)
            logits, activations = model.get_output_and_context_activations(batch)
            if not torch.isfinite(logits).all():
                raise ValueError(f'Nonfinite classifier logits for receiver {cid}')
            logits = logits.cpu().numpy()
            activations = {name: value.cpu().numpy() for name, value in activations.items()}
            cm_add(cms['AllClassesDiagnostic'], truth, logits.argmax(1))
            true_task = task_of_class[truth]
            global_pred = np.empty(len(data), dtype=int)
            for task in np.unique(true_task):
                selector = true_task == task
                allowed = task_of_class == task
                global_pred[selector] = np.where(allowed, logits[selector], -np.inf).argmax(1)
            cm_add(cms['OracleGlobalTaskMaskDiagnostic'], truth, global_pred)
            for policy, detector in detectors.items():
                binary = detector.binarize_layer_activations(activations)
                route, _ = detector.predict_episodes_with_scores(binary)
                route = np.asarray(route, dtype=int)
                if not np.isin(route, banks[policy]).all():
                    raise ValueError('Native selected task outside defined legal bank')
                actions = np.stack([np.where(masks[policy][task], logits, -np.inf).argmax(1)
                                    for task in banks[policy]], axis=1)
                bank_index = {task: k for k, task in enumerate(banks[policy])}
                selected = np.array([bank_index[int(t)] for t in route])
                pred = actions[np.arange(len(data)), selected]
                mask_only = np.empty(len(data), dtype=int)
                for task in np.unique(route):
                    selector = route == task
                    mask_only[selector] = np.where(task_of_class == task, logits[selector], -np.inf).argmax(1)
                saved = data[policy].astype(int)
                discrepancies[policy]['saved'] += int(np.count_nonzero(saved != pred))
                if offset == 0:
                    native_logits, native_route = _denice_routed_logits_with_episodes(
                        model, batch, detector, list(range(34)), args.device, inference_policy='pred_hard')
                    native = native_logits.argmax(1).cpu().numpy()
                    discrepancies[policy]['native'] += int(np.count_nonzero(native != pred))
                    discrepancies[policy]['native_rows'] += len(data)
                    if not np.array_equal(native_route, route):
                        raise ValueError('One-forward routing differs from native route')
                matched_route = np.where(np.isin(true_task, banks[policy]), true_task, route)
                matched = actions[np.arange(len(data)), [bank_index[int(t)] for t in matched_route]]
                best = (actions == truth[:, None]).any(1)
                covered = masks[policy].any(0)[truth]
                correct = pred == truth
                control_correct = mask_only == truth
                transition = np.where(correct, np.where(control_correct, 0, 1),
                                      np.where(control_correct, 2, 3))
                mask_only_transitions[policy] += np.bincount(covered.astype(int) * 4 + transition,
                    minlength=8).reshape(2, 4)
                bucket = np.where(correct, 0, np.where(~covered, 1, np.where(~best, 2, 3)))
                # Exact additive identity on this fixed action bank.
                if np.any(correct & ~best):
                    raise RuntimeError('Normal prediction omitted from legal route bank')
                tables[policy] += np.bincount(truth * 4 + bucket, minlength=34 * 4).reshape(34, 4)
                route_cms[policy] += np.bincount(true_task * 6 + route, minlength=36).reshape(6, 6)
                oracle_correct[policy] += int(best.sum())
                route_label_missing[policy] += int(np.count_nonzero(~np.isin(true_task, banks[policy])))
                mask_label_missing[policy] += int(np.count_nonzero(~masks[policy][true_task, truth]))
                for suffix, prediction in (('', pred), ('_Saved', saved), ('_OracleMatched', matched),
                                           ('_MaskOnlyDiagnostic', mask_only)):
                    cm_add(cms[policy + suffix], truth, prediction)
            if offset and offset % (args.batch_size * 100) == 0:
                print(f'Bottleneck receiver={cid}: {offset}/{len(shard)}; full test={begin + offset}/{n}', flush=True)
        for policy, detector in detectors.items():
            if complete_hash(model, detector) != before[policy]:
                raise RuntimeError('Audit mutated weights, masks, registry or frozen router')
        item = dict(receiver=cid, rows=len(shard), inventory=local_inventory,
                    metrics={p: metrics(cm) for p, cm in cms.items()},
                    confusion_matrices={p: cm.tolist() for p, cm in cms.items()},
                    buckets_by_class={p: table.tolist() for p, table in tables.items()},
                    route_confusion={p: cm.tolist() for p, cm in route_cms.items()},
                    best_allowed_correct=oracle_correct, reproduction=discrepancies,
                    mask_only_transitions_by_original_coverage={p: table.tolist()
                        for p, table in mask_only_transitions.items()},
                    missing_true_task=route_label_missing, missing_true_task_class_mask=mask_label_missing,
                    state_unchanged=True)
        write_json(result_path, item)
        clients.append(item)
        print(f'Completed receiver {position}/{len(source_lock["receivers"])} ({cid}); '
              f'rows={end}/{n}; seconds={time.monotonic() - start_time:.1f}', flush=True)
        del model, detectors, original
        gc.collect()
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
    if args.inventory_only:
        write_json(out / 'inventory_completion.json', dict(completed=True, inference=False, inventory=inventory))
        return
    aggregate = {p: np.zeros((34, 34), dtype=np.int64) for p in clients[0]['confusion_matrices']}
    for client in clients:
        for policy, cm in client['confusion_matrices'].items():
            aggregate[policy] += np.array(cm)
    result = dict(completed=True, protocol=protocol, checkpoint_hashes=checkpoint_hashes,
                  rows=sum(c['rows'] for c in clients), receivers=len(clients),
                  metrics={p: metrics(cm) for p, cm in aggregate.items()}, policies={},
                  all_state_unchanged=all(c['state_unchanged'] for c in clients),
                  seconds_this_invocation=time.monotonic() - start_time)
    for policy in POLICIES:
        table = sum(np.array(c['buckets_by_class'][policy]) for c in clients)
        route_cm = sum(np.array(c['route_confusion'][policy]) for c in clients)
        reproduced = {key: sum(c['reproduction'][policy][key] for c in clients)
                      for key in ('saved', 'native', 'native_rows')}
        best_correct = sum(c['best_allowed_correct'][policy] for c in clients)
        transitions = sum(np.array(c['mask_only_transitions_by_original_coverage'][policy]) for c in clients)
        if (transitions[:, 2].sum() - transitions[:, 1].sum()
                != aggregate[policy + '_MaskOnlyDiagnostic'].diagonal().sum() - aggregate[policy].diagonal().sum()):
            raise RuntimeError('Mask-only rescue/break accounting failed')
        if table.sum() != n or table[:, 0].sum() + table[:, 3].sum() != best_correct:
            raise RuntimeError('Error decomposition identity failed')
        result['policies'][policy] = dict(
            best_allowed_route_accuracy=best_correct / n,
            route_accuracy=float(route_cm.diagonal().sum() / n),
            buckets={bucket: int(table[:, k].sum()) for k, bucket in enumerate(BUCKETS)},
            buckets_by_class=table.tolist(), route_confusion=route_cm.tolist(), reproduction=reproduced,
            mask_only_transitions_by_original_coverage=transitions.tolist(),
            reproduction_within_locked_numerical_budget=(reproduced['saved'] / n <= .0001
                and reproduced['native'] / reproduced['native_rows'] <= .0001),
            dominant_error=BUCKETS[1 + int(table.sum(0)[1:].argmax())],
            missing_true_task=sum(c['missing_true_task'][policy] for c in clients),
            missing_true_task_class_mask=sum(c['missing_true_task_class_mask'][policy] for c in clients))
    if result['rows'] != n or len(clients) != 98:
        raise RuntimeError('Incomplete full test audit')
    write_json(out / 'completion.json', result)
    print(json.dumps({p: result['policies'][p] for p in POLICIES}), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ('checkpoint', 'legacy', 'roles', 'inputs', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--inventory-only', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('Batch size must be positive')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    with threadpool_limits(limits=1):
        run(args)


if __name__ == '__main__':
    main()
