"""Lock production integration evidence and build the external Kaggle runtime."""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits

from appliance.stable_head import StableHeadRegistry, POLICY
from appliance.state import digest, write_json
from eval_checkpoint import _make_denice_client_model
from fed_learning.training.decentralized_denice_il import _load_denice_continuation_state


def run(a):
    independent = json.loads((a.smoke / 'controlled_integration_independent_audit.json').read_text())
    verification = json.loads((a.verification / 'completion.json').read_text())
    if not independent['completed'] or not verification['completed'] or verification['mismatches']:
        raise ValueError('Automatic integration/true uninterrupted comparison is not complete')
    final = _load_denice_continuation_state(str(a.smoke / 'native' / 'continuation_state_task_4.pt'))
    service = final['appliance_service_state']
    history = service['rounds']
    transactions = history[0]['transactions']
    committed = [r for r in transactions if r['transaction']['applied']]
    checks = {**verification['checks'], **independent['checks']}
    checks.update(automatic_patch_not_prepared=all(not r.get('prepared_packet_used', True) for r in transactions),
        rollback_verified=all(r['transaction'].get('rollback_verified') for r in transactions if not r['transaction']['applied']),
        full_federation_checkpoint=len(final['client_ids']) >= 89,
        no_historical_CAL=all(r.get('historical_CAL_reads', 0) == 0 for r in transactions),
        small_CAL_never_stops_training=final['meta']['completed_task'] == 4,
        heads_and_certificates_survive_every_callback=all(r['head_exact_after'] and r['certificate_current']
            for row in history for r in row['lifecycle']))
    lifecycle = []
    for item in committed:
        cid, c = item['pair']['receiver'], item['pair']['class_id']
        model, router = _make_denice_client_model(final, cid, 'cpu')
        reg = StableHeadRegistry()
        reg.entries = model.appliance_guarded_head_entries
        entry = reg.entries[c]
        seal = entry['lifecycle_certificate_sha256']
        carried = reg.update_lifecycle(model, router, c, entry['lifecycle_certificate']['scope'], 4, new_cal_rows=10)
        suspended = reg.update_lifecycle(model, router, c, model.appliance_runtime_scope, 4, new_cal_rows=10)
        checks[f'carry_and_suspend_{cid}'] = (carried['state'] == 'CARRY_FORWARD' and
            suspended['state'] == 'SUSPENDED' and entry['lifecycle_certificate_sha256'] == seal and reg.head_matches(model, c))
        lifecycle.append(dict(receiver=cid, class_id=c, immutable_seal=seal,
            certified_scope=entry['lifecycle_certificate']['scope'], carry=carried['state'],
            cumulative_state=suspended['state'], cumulative_reason=suspended['reason']))
    del final
    gc.collect()
    boundary = _load_denice_continuation_state(str(a.smoke / 'round0_branch_seed.pt'))
    checks['strict_round_boundary_load'] = boundary['round_state']['next_round'] == 1 and boundary['meta']['boundary'] == 'round'
    del boundary
    kinds = {}
    for record in service['communication']:
        kinds[record['kind']] = kinds.get(record['kind'], 0) + record['bytes']
    checks['setup_and_discovery_counted'] = (kinds.get('current_receiver_function_capsule', 0) > 0 and
        kinds.get('discovery_requests', 0) > 0 and kinds.get('discovery_offers', 0) > 0)
    paths = list(Path('appliance').glob('*.py'))
    paths += [Path(p) for p in ('fed_learning/training/decentralized_denice_il.py',
        'fed_learning/training/checkpoint_state.py', 'fed_learning/training/denice_checkpoint_archive.py',
        'fed_learning/training/denice_eval.py', 'fed_learning/data/denice_clean_roles.py',
        'fed_learning/clients/nice_client.py', 'tools/eval_denice_legacy_self.py')]
    result = dict(version='appliance_production_integration_smoke_v1', completed=all(checks.values()),
        checks=checks, comparisons=len(checks), mismatches=sum(not v for v in checks.values()),
        guard=POLICY, xi=.8, method='legacy', automatic_commits=len(committed),
        automatic_rejections=len(transactions)-len(committed),
        native_task=4, native_rounds=2, native_active_clients=89, native_max_samples_per_client=256,
        native_backend='CPU FP32', Task3_origin='original terminal checkpoint, actual graph, fresh automatic compile',
        BASE_summary_fixture='owned chronological BASE only; retrospectively seeded for late-start smoke',
        communication=dict(application_egress_bytes=sum(kinds.values()), by_kind=kinds,
            socket_TLS_overhead_measured=False, baseline_aggregation_separate=True),
        lifecycle=lifecycle, historical_CAL_runtime_reads=0, final_test_opened=False,
        full_campaign_started=False, cumulative_imported_route_authorization_verified=False,
        selection_policy='existing one receiver/class current FIT scheduler; single capability/receiver',
        limitation='Integration PASS, but current-only certificates suspend for cumulative test outside certified scope; no full-test benefit claimed',
        source_sha256={p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))})
    write_json(a.report, result)
    if not result['completed']:
        raise AssertionError(checks)
    print(json.dumps({k:result[k] for k in ('completed','comparisons','mismatches','automatic_commits','communication','limitation')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke',type=Path,required=True)
    p.add_argument('--verification',type=Path,required=True)
    p.add_argument('--report',type=Path,default=Path('artifacts/appliance_production_integration_smoke.json'))
    a=p.parse_args()
    torch.set_num_threads(4)
    with threadpool_limits(limits=1):run(a)
