"""P1 one-round real current-BASE smoke; no TEST/P2/full-run integration."""
import argparse
import json
from pathlib import Path
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from fed_learning.data.denice_clean_roles import CleanRoleData, file_sha256
from fed_learning.models.denice_model import DeNICEModel
from fed_learning.strategies.decentralized.denice_v2.contracts import state_hash
from fed_learning.strategies.decentralized.denice_v2.native import CurrentBase, capture_native, run_native_round
from fed_learning.strategies.decentralized.denice_v2.normalization import array


def copy_member(z, name, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return  # CleanRoleData subsequently checks locked checksums.
    target.write_bytes(z.read(name))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-zip', required=True)
    parser.add_argument('--dataset-zip', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--partition', default='artifacts/denice_v2_task0_partition_dry_run_20261011.json')
    parser.add_argument('--clients', type=int, nargs=2, default=[9, 30])
    parser.add_argument('--per-class', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    started = time.monotonic()
    out, cache = Path(args.out), Path(args.cache)
    if out.exists() or args.per_class < 2 or len(set(args.clients)) != 2:
        raise ValueError('New receipt path and two distinct clients required')
    partition = json.loads(Path(args.partition).read_text(encoding='utf-8'))
    natural = [c for c in partition['clusters'] if set(args.clients) <= set(c['members'])]
    if partition['task'] != 0 or partition['threshold'] != .8 or len(natural) != 1:
        raise ValueError('Smoke members must belong to one locked Task0 baseline cluster')
    with zipfile.ZipFile(args.results_zip) as z:
        for name in ('role_manifest.json', 'role_lock.json', *[f'indices/client_{cid}.npz' for cid in args.clients]):
            copy_member(z, 'denice_clean_roles/' + name, cache / 'roles' / name)
    with zipfile.ZipFile(args.dataset_zip) as z:
        for name in ('metadata.json', *[f'client_{cid}_train.npz' for cid in args.clients]):
            copy_member(z, '100-clients/' + name, cache / 'dataset' / name)
    roles = CleanRoleData(cache / 'roles', cache / 'dataset')
    if not json.loads((cache / 'roles/role_lock.json').read_text(encoding='utf-8')).get('locked_before_backbone_training'):
        raise ValueError('BASE roles were not locked before backbone training')
    if file_sha256(cache / 'roles/role_manifest.json') != partition['manifest_sha256']:
        raise ValueError('Partition used different BASE roles')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    bases = {}
    for cid in sorted(args.clients):
        x, y, rows = roles.client_role(cid, 'base')
        selected = np.concatenate([np.flatnonzero(y == c)[:args.per_class] for c in range(6)])
        bases[cid] = CurrentBase(cid, 0, torch.from_numpy(x[selected]), torch.from_numpy(y[selected]),
            rows[selected], state_hash(dict(source=roles.manifest['clients'][str(cid)]['source_sha256'],
                                           indices=roles.manifest['clients'][str(cid)]['index_sha256'])))
    shape = tuple(roles.manifest['input_shape'])
    model = DeNICEModel(shape, 34)
    model.allocation_policy = 'class_blocks'
    layers = ('conv1', 'conv2', 'conv3', 'gru')
    router = dict(router_mode='binary_cosine', context_layers=layers, stable_feature_mask=np.zeros(548, bool),
                  binarize_thresholds=None, activation_memory={}, episode_classes={})
    # No router history exists yet. One actual parameter is a declared protected
    # dependency witness for P1, NOT a claimed P2 historical dependency closure.
    state = capture_native(model, members=args.clients, coordinator_id=min(args.clients),
        cluster_uid='p1-task0-subcluster-smoke', router_state=router, routing_dependencies=('gru.weight_hh_l1',))
    original_hash = state.fingerprint()
    settings = dict(seed=args.seed, batch_size=16, epochs=1, lr=.001, optimizer_family='adam', weight_decay=0., momentum=0., device=args.device)
    print('P1 native smoke: current BASE only; clients', args.clients, 'rows', {c: len(b.y) for c, b in bases.items()}, flush=True)
    result, plan, packets, receipt = run_native_round(state, bases, tuple(range(6)), **settings)
    counts = {cid: base.validate(tuple(range(6))) for cid, base in bases.items()}
    unique_classes = [c for c in range(6) if sum(c in row for row in counts.values()) == 1]
    checks = dict(reference_unchanged=state.fingerprint() == original_hash,
        common_start=all(w['initial_reference_hash'] == plan.reference_hash and
                         w['initial_topology_hash'] == plan.topology_hash for w in receipt['workers']),
        optimizer_changed_all_workers=all(w['steps'] > 0 and sum(w['changed_coordinates'].values()) > 0 for w in receipt['workers']),
        unique_contributor_fc2_transferred=bool(unique_classes),
        declared_routing_dependency_preserved=all(torch.equal(state.model_state[k], result.state.model_state[k]) for k in state.routing_dependencies),
        sequential_bn=len(receipt['bn_collector_calls']) == 3 and all(
            call['input_model_hash'] == step['input_model_hash'] for call, step in zip(receipt['bn_collector_calls'], receipt['aggregation']['bn'])),
        bn_chain=all(a['output_model_hash'] == b['input_model_hash'] for a, b in zip(receipt['aggregation']['bn'], receipt['aggregation']['bn'][1:])),
        broadcast_state=receipt['broadcast_state_identical'], broadcast_logits=receipt['broadcast_logits_identical'])
    details = {}
    for c in unique_classes:
        packet = next(p for p in packets if c in dict(p.event.successful_class_counts))
        expected = array(plan.reference.model_state['fc2.weight'][c]).astype(np.float64) + packet.deltas['fc2.weight'][c]
        actual = result.state.model_state['fc2.weight'][c].numpy()
        correct = np.array_equal(actual, expected.astype(actual.dtype))
        changed = bool(np.any(packet.deltas['fc2.weight'][c]))
        checks['unique_contributor_fc2_transferred'] &= correct and changed
        details[c] = dict(contributor=packet.event.client_id, exact_aggregate=correct, changed=changed)
    report = dict(completed=True, passed=all(checks.values()), checks=checks,
        scope='P1 native one round on bounded subset of a natural Task0 cluster',
        task=0, round=0, baseline_xi=.8, baseline_partition_K=partition['K'],
        natural_cluster=natural[0], members=sorted(args.clients), source_manifest_sha256=partition['manifest_sha256'],
        per_class_cap=args.per_class, counts=counts, unique_class_checks=details, settings=settings,
        receipts=receipt, elapsed_seconds=time.monotonic()-started,
        test_read=False, historical_data_used=False, raw_rows_retained_in_receipt=False,
        router_history_empty=True, router_consistency_claim=False,
        mature_protection_test='separate controlled native unit tests; no historical neurons in fresh Task0 smoke',
        production_runner_integrated=False, accuracy_claim=False)
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root/'fed_learning/clients/nice_client.py', root/'fed_learning/clients/denice_client.py',
               root/'fed_learning/models/nice_model.py', root/'fed_learning/models/denice_model.py',
               *sorted((root/'fed_learning/strategies/decentralized/denice_v2').glob('*.py'))]
    report['source_sha256'] = {p.relative_to(root).as_posix(): file_sha256(p) for p in sources}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'], checks=checks, elapsed=report['elapsed_seconds']), indent=2), flush=True)
    if not report['passed']:
        raise SystemExit('P1 native smoke failed')


if __name__ == '__main__':
    main()
