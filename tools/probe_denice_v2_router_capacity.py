"""P2 preflight on post-aggregation Task0 BASE; abort native transition at NO-GO."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from fed_learning.data.denice_clean_roles import CleanRoleData, file_sha256
from fed_learning.models.denice_model import DeNICEModel
from fed_learning.strategies.decentralized.denice_v2.contracts import state_hash
from fed_learning.strategies.decentralized.denice_v2.native import CurrentBase, capture_native, run_native_round
from fed_learning.strategies.decentralized.denice_v2.router import probe_router_capacity, stage_cluster_router


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True, help='Verified cache prepared by run_denice_v2_native_smoke.py')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    if out.exists():
        raise ValueError('Use a new receipt path')
    started = time.monotonic()
    cache = Path(args.cache)
    roles = CleanRoleData(cache/'roles', cache/'dataset')
    if not json.loads((cache/'roles/role_lock.json').read_text()).get('locked_before_backbone_training'):
        raise ValueError('Roles not locked before training')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    bases = {}
    for cid in (9, 30):
        x, y, rows = roles.client_role(cid, 'base')
        selected = np.concatenate([np.flatnonzero(y == c)[:32] for c in range(6)])
        bases[cid] = CurrentBase(cid, 0, torch.from_numpy(x[selected]), torch.from_numpy(y[selected]), rows[selected],
            state_hash(dict(source=roles.manifest['clients'][str(cid)]['source_sha256'], indices=roles.manifest['clients'][str(cid)]['index_sha256'])))
    model = DeNICEModel(tuple(roles.manifest['input_shape']), 34)
    router = dict(router_mode='binary_cosine', context_layers=('conv1','conv2','conv3','gru'),
        stable_feature_mask=np.zeros(548, bool), binarize_thresholds=None, activation_memory={}, episode_classes={})
    initial = capture_native(model, members=(9,30), coordinator_id=9, cluster_uid='p2-native-capacity-preflight',
        router_state=router, routing_dependencies=('gru.weight_hh_l1',))
    print('P2 preflight: one bounded Task0 BASE round, then dependency capacity gate', flush=True)
    result, plan, packets, p1 = run_native_round(initial, bases, tuple(range(6)), seed=42, batch_size=16, epochs=1, lr=.001)
    before = result.state.fingerprint()
    probe = probe_router_capacity(result.state)
    proposal = None
    if probe['capacity']['passed']:
        proposal = stage_cluster_router(result.state, plan, packets, bases)
        proposal.commit()
    # Never bypass the locked capacity limit to manufacture a two-task smoke.
    report = dict(completed=True, task=0, task0_postaggregation_state_hash=before,
        decision=probe['capacity']['decision'], probe=probe, base_manifest_sha256=file_sha256(cache/'roles/role_manifest.json'),
        members=[9,30], rows={cid:len(b.y) for cid,b in bases.items()},
        feature_policy='allocated_units_all_four_native_context_layers', policy_selected_before_probe=True,
        native_task_transition_started=False, production_router_committed=proposal is not None,
        p2_pass=False, p1_checks=p1, original_state_unchanged=result.state.fingerprint()==before,
        no_historical_raw_read=True, test_read=False, threshold_tuning=False,
        elapsed_seconds=time.monotonic()-started)
    root=Path(__file__).resolve().parents[1]
    paths=[Path(__file__), *sorted((root/'fed_learning/strategies/decentralized/denice_v2').glob('*.py'))]
    report['source_sha256']={path.relative_to(root).as_posix():file_sha256(path) for path in paths}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(decision=report['decision'],capacity=probe['capacity'],selected=probe['selected_features']),indent=2),flush=True)


if __name__ == '__main__':
    main()
