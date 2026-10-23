"""Small real-data transition probe around the production DeNICE runner.

Does not edit training code. Sampling is stratified and deliberately small;
reported scores are diagnostics, not the historical experiment's benchmark.
"""
import argparse
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from sklearn.metrics import f1_score
from fed_learning.clients.denice_client import DeNICEClient
from fed_learning.models.denice_model import DeNICEModel
from fed_learning.servers.nice_server import ContextDetector
from fed_learning.training import decentralized_denice_il as runner
from fed_learning.training.checkpoint_state import snapshot_denice_state, restore_denice_state
from fed_learning.training.denice_eval import _denice_routed_logits_with_episodes


def sample_npz(payload, split, quota, seed, num_classes=12):
    """Read labels then stream features; never allocate the full global test X."""
    with ZipFile(io.BytesIO(payload)) as z:
        with z.open(f'y_{split}.npy') as stream:
            y = np.load(stream, allow_pickle=False)
        rng = np.random.default_rng(seed)
        indices = np.sort(np.concatenate([
            rng.choice(np.flatnonzero(y == c), min(quota, int((y == c).sum())), replace=False)
            for c in range(num_classes) if (y == c).any()
        ]))
        labels = y[indices].copy()
        with z.open(f'X_{split}.npy') as stream:
            version = np.lib.format.read_magic(stream)
            reader = {(1,0): np.lib.format.read_array_header_1_0,
                      (2,0): np.lib.format.read_array_header_2_0}[version]
            shape, fortran, dtype = reader(stream)
            if fortran or dtype.hasobject:
                raise ValueError('Expected C-order numeric features')
            width = int(np.prod(shape[1:]))
            selected = np.empty((len(indices), *shape[1:]), dtype=dtype)
            for start in range(0, shape[0], 65536):
                count = min(65536, shape[0]-start)
                raw = stream.read(count*width*dtype.itemsize)
                block = np.frombuffer(raw, dtype=dtype).reshape(count, *shape[1:])
                lo, hi = np.searchsorted(indices, [start, start+count])
                selected[lo:hi] = block[indices[lo:hi]-start]
            if not np.isfinite(selected).all():
                raise ValueError('Nonfinite selected features')
    return torch.from_numpy(selected).float(), torch.from_numpy(labels).long(), indices.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--client', type=int, default=34)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--train-per-class', type=int, default=64)
    parser.add_argument('--test-per-class', type=int, default=32)
    parser.add_argument('--no-observation', action='store_true', help='Control run: observe only final state')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError('Choose a new output directory; existing diagnostics are never overwritten')
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(1)
    print('Sampling fixed real train/test panels...', flush=True)
    with ZipFile(args.archive) as z:
        train_x, train_y, train_indices = sample_npz(
            z.read(f'100-clients/client_{args.client}_train.npz'), 'train', args.train_per_class, 2026)
        test_x, test_y, test_indices = sample_npz(
            z.read('100-clients/global_test_data.npz'), 'test', args.test_per_class, 2027)
        metadata = json.loads(z.read('100-clients/metadata.json'))
    print(f'Selected train={len(train_y)}, test={len(test_y)}', flush=True)
    np.savez_compressed(args.output_dir/'sampled_panel.npz', X_train=train_x.numpy(),
                        y_train=train_y.numpy(), X_test=test_x.numpy(), y_test=test_y.numpy())
    support = sorted(set(train_y.tolist()))
    class Loader:
        input_shape = (39,1)
        task_classes = {int(t):cs for t,cs in metadata['task_structure']['task_classes'].items()}
        def __init__(self, data_dir): pass
        def get_num_tasks(self): return 6  # Task 1 is NOT the original terminal task.
        def get_task_classes(self, task): return self.task_classes[task]
        def get_all_client_ids(self): return [args.client]
        def get_client_data(self, cid, task):
            mask = torch.isin(train_y, torch.tensor(self.task_classes[task]))
            return train_x[mask], train_y[mask]
        def get_test_data(self, task, cumulative=True):
            mask = test_y < 6*(task+1)
            return test_x[mask], test_y[mask]
    prior = json.loads(Path('audit_denice/artifact_evidence.json').read_text())
    config = deepcopy(prior['full_d2']['42']['config'])
    config.update(data_dir='in_memory_stratified_real_archive_panel',
                  output_dir=str(args.output_dir/'training'), resume_output_dir=str(args.output_dir/'training'),
                  random_seed=42, seed=42, num_clients=1, denice_max_clients=1,
                  task_start=0, task_end=1, rounds_per_task=args.rounds,
                  nice_phase_epochs=1, batch_size=32, eval_batch_size=128,
                  eval_every=9999, denice_post_task_eval=False,
                  denice_aggregation_mode='self_only', denice_collaboration_guard_mode='off',
                  denice_shared_context_eval=False, denice_min_free_capacity_ratio=0.1,
                  save_resume_after_task=True, resume_state_path=None, round_checkpoint_every=args.rounds)
    records = []
    current = {}
    previous_gru = None
    def score(model, detector, task):
        rows, logits_by_policy = {}, {}
        mask = test_y < (task+1)*6
        x, y = test_x[mask], test_y[mask]
        for policy in ['backbone_nomask', 'oracle_hard', 'pred_hard']:
            logits, routes = _denice_routed_logits_with_episodes(
                model, x, detector, list(range((task+1)*6)), 'cpu',
                inference_policy=policy, oracle_episodes=(y.numpy()//6))
            pred = logits.argmax(1).numpy()
            logits_by_policy[policy] = logits.clone()
            rows[policy] = {}
            for old in range(task+1):
                for name, allowed in [('all_classes', list(range(6*old,6*old+6))),
                                      ('local_support', [c for c in support if c//6 == old])]:
                    selected = np.isin(y.numpy(), allowed)
                    rows[policy][f'task{old}_{name}'] = {
                        'n': int(selected.sum()), 'accuracy': float(np.mean(pred[selected] == y.numpy()[selected])),
                        'macro_f1': float(f1_score(y.numpy()[selected], pred[selected], labels=allowed, average='macro', zero_division=0)),
                        'route_accuracy': float(np.mean(np.asarray(routes)[selected] == old)) if routes is not None else None,
                    }
        return rows, logits_by_policy
    def observe(stage, reload=False):
        nonlocal previous_gru
        if not current: return
        if args.no_observation and stage != 'final_task_boundary': return
        rng = runner._snapshot_rng_state()
        try:
            model, detector = current['model'], current['context_detector']
            clone, router = deepcopy(model).eval(), deepcopy(detector)
            metrics, logits = score(clone, router, current['task_id'])
            with torch.no_grad():
                gru = clone.get_context_activations_per_sample(test_x[test_y<6])['gru'].cpu()
            row = {'stage': stage, 'task': current['task_id'], 'metrics': metrics,
                   'output_ages': model.unit_ranks['fc2'].tolist(), 'active_adapters': dict(model.active_adapters),
                   'capacity': runner._capacity_debug(model),
                   'task0_gru_max_change_since_previous': None if previous_gru is None else float((gru-previous_gru).abs().max())}
            previous_gru = gru.clone()
            if reload:
                buffer = io.BytesIO()
                torch.save({'weights': model.state_dict(), 'algorithm': snapshot_denice_state(model, detector)}, buffer)
                buffer.seek(0)
                saved = torch.load(buffer, weights_only=False)
                fresh, fresh_router = DeNICEModel((39,1),34).eval(), ContextDetector()
                restore_denice_state(fresh, fresh_router, saved['algorithm'])
                fresh.load_state_dict(saved['weights'])
                _, fresh_logits = score(fresh, fresh_router, current['task_id'])
                row['reload_max_abs_logit_error'] = max(float((logits[p]-fresh_logits[p]).abs().max()) for p in logits)
                assert row['reload_max_abs_logit_error'] == 0
            records.append(row)
            print(f"PROBE task={row['task']} {stage}: oracle old="
                  f"{metrics['oracle_hard']['task0_local_support']['accuracy']:.4f}, "
                  f"pred old={metrics['pred_hard']['task0_local_support']['accuracy']:.4f}", flush=True)
        finally:
            runner._restore_rng_state(rng)
    prepare = runner._prepare_client_task
    def wrapped_prepare(**kwargs):
        value = prepare(**kwargs)
        current.update(kwargs)
        observe('after_task_prepare')
        return value
    local = DeNICEClient.train
    def wrapped_train(client, *a, **kw):
        observe(f"round{kw['phase_offset']}_before_local")
        result = local(client,*a,**kw)
        observe(f"round{kw['phase_offset']}_after_local")
        return result
    def wrap_operation(original, name):
        def wrapped(*a, **kw):
            observe('before_'+name)
            result = original(*a, **kw)
            observe('after_'+name, reload=name == 'aging')
            return result
        return wrapped
    try:
        with patch.object(runner,'IncrementalDataLoader',Loader), \
             patch.object(runner,'_prepare_client_task',wrapped_prepare), \
             patch.object(DeNICEClient,'train',wrapped_train), \
             patch.object(runner,'_aggregate_round',wrap_operation(runner._aggregate_round,'aggregation')), \
             patch.object(runner,'_enforce_minimum_free_capacity',wrap_operation(runner._enforce_minimum_free_capacity,'reserve')), \
             patch.object(runner,'increase_unit_ranks',wrap_operation(runner.increase_unit_ranks,'aging')):
            runner.run_decentralized_denice_il(config)
        observe('final_task_boundary', reload=True)
    finally:
        report = {'client_id': args.client, 'train_support': support,
                  'completed': bool(records and records[-1]['stage'] == 'final_task_boundary'),
                  'observer_enabled_during_training': not args.no_observation,
                  'train_source_indices': train_indices, 'test_source_indices': test_indices,
                  'train_source': f'client_{args.client}_train.npz', 'test_source': 'global_test_data.npz',
                  'config': config, 'records': records,
                  'scope': 'One-client real-data stratified subset. Self-only aggregation; original six-task semantics, stopped after task 1. Not a full-run benchmark.'}
        (args.output_dir/'transitions.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
