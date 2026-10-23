"""Native API replay of the fixed mask-only control, no fitting or label inputs."""
import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from appliance.runner import load_input
from appliance.state import complete_hash
from eval_checkpoint import _make_denice_client_model
from fed_learning.training.checkpoint_state import restore_context_detector
from fed_learning.training.denice_eval import _allowed_classes_for_episode, _denice_routed_logits_with_episodes
from fed_learning.strategies.incremental.denice_class_availability import (
    detector_with_class_mask_policy, routed_self_with_class_mask_policy,
)


@torch.no_grad()
def run(args):
    source = Path(args.audit_dir)
    completion = json.loads((source / 'completion.json').read_text())
    if not completion.get('completed'):
        raise ValueError('Finish the full frozen bottleneck audit first')
    root = Path(args.inputs)
    metadata = json.loads((root / 'metadata.json').read_text())
    task_classes = {int(t): list(map(int, classes))
                    for t, classes in metadata['task_structure']['task_classes'].items()}
    x = np.load(root / 'X_test.npy', mmap_mode='r')
    records = np.load(root / 'records.npy', mmap_mode='r')
    ranges = json.loads((root / 'records_manifest.json').read_text())['receiver_ranges']
    snapshots = joblib.load(Path(args.legacy) / 'frozen_routers.joblib')
    ckpt, _ = load_input(args.checkpoint, 5, 19)
    receipts = []
    for cid in completion['protocol']['source_lock']['receivers']:
        begin, end = ranges[str(cid)]
        rows = records[begin:min(end, begin + 512)]['global_test_row']
        inputs = torch.as_tensor(np.array(x[rows], copy=True), device=args.device, dtype=torch.float32)
        model, detector = _make_denice_client_model(ckpt, cid, args.device)
        restore_context_detector(detector, snapshots[cid]['MulticlassSelf'])
        before = complete_hash(model, detector)
        shadow = detector_with_class_mask_policy(detector, 'global_task', task_classes, list(range(34)), 34)
        logits, acts = model.get_output_and_context_activations(inputs)
        acts = {name: value.cpu().numpy() for name, value in acts.items()}
        raw = logits.cpu().numpy()
        route, _ = detector.predict_episodes_with_scores(detector.binarize_layer_activations(acts))
        shadow_route, _ = shadow.predict_episodes_with_scores(shadow.binarize_layer_activations(acts))
        expected = np.empty(len(inputs), dtype=int)
        for task in np.unique(route):
            chosen = route == task
            mask = np.isin(np.arange(34), task_classes[int(task)])
            expected[chosen] = np.where(mask, raw[chosen], -np.inf).argmax(1)
        actual, actual_route = routed_self_with_class_mask_policy(
            model, inputs, detector, list(range(34)), args.device,
            class_mask_policy='global_task', task_classes=task_classes)
        actual = actual.argmax(1).cpu().numpy()
        # Separate diagnostic integrity replay. Labels enter only this oracle
        # comparison, after the real mask-only prediction has been produced.
        task_of_class = {c: task for task, classes in task_classes.items() for c in classes}
        truth = records[begin:min(end, begin + 512)]['y_true']
        true_task = np.array([task_of_class[int(c)] for c in truth])
        oracle_checks = {}
        for policy in ('BinarySelf', 'MulticlassSelf'):
            oracle_detector = copy.deepcopy(detector)
            restore_context_detector(oracle_detector, snapshots[cid][policy])
            predicted, _ = oracle_detector.predict_episodes_with_scores(
                oracle_detector.binarize_layer_activations(acts))
            bank = sorted(int(task) for task, memory in oracle_detector.activation_memory.items() if len(memory))
            cached = {}
            native_actions_match = True
            for task in bank:
                allowed = _allowed_classes_for_episode(oracle_detector, task, list(range(34)), 34)
                cached[task] = np.where(np.isin(np.arange(34), allowed), raw, -np.inf).argmax(1)
                oracle, _ = _denice_routed_logits_with_episodes(model, inputs, oracle_detector,
                    list(range(34)), args.device, inference_policy='oracle_hard',
                    oracle_episodes=np.full(len(inputs), task))
                native_actions_match &= np.array_equal(cached[task], oracle.argmax(1).cpu().numpy())
            matched_route = np.where(np.isin(true_task, bank), true_task, predicted)
            cached_matched = np.array([cached[int(task)][index] for index, task in enumerate(matched_route)])
            native_matched, _ = _denice_routed_logits_with_episodes(model, inputs, oracle_detector,
                list(range(34)), args.device, inference_policy='oracle_hard', oracle_episodes=matched_route)
            oracle_checks[policy + '_native_oracle_actions'] = bool(native_actions_match)
            oracle_checks[policy + '_native_matched_oracle'] = bool(np.array_equal(
                cached_matched, native_matched.argmax(1).cpu().numpy()))
        checks = dict(native_matches_fixed_counterfactual=np.array_equal(expected, actual),
                      router_unchanged=np.array_equal(route, actual_route) and np.array_equal(route, shadow_route),
                      original_state_unchanged=complete_hash(model, detector) == before,
                      profiles_unchanged=set(shadow.activation_memory) == set(detector.activation_memory),
                      local_default_identity=detector_with_class_mask_policy(
                          detector, 'local', None, list(range(34)), 34) is detector)
        checks.update(oracle_checks)
        receipts.append(dict(receiver=cid, rows=len(rows), checks={k: bool(v) for k, v in checks.items()}))
        del model, detector, shadow
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
    report = dict(passed=all(all(r['checks'].values()) for r in receipts),
                  receivers=len(receipts), rows=sum(r['rows'] for r in receipts), receipts=receipts,
                  real_prediction_labels_read=False, oracle_integrity_labels_read=True,
                  fitting=False, donor_inference=False,
                  production_default='local', variant='global_task; experimental coverage policy only')
    Path(args.out).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'receipts'}, indent=2))
    if not report['passed']:
        raise ValueError('Class-mask variant native replay failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ('checkpoint', 'legacy', 'inputs', 'audit-dir', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    with threadpool_limits(limits=1):
        run(args)
