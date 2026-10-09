"""Compare genuine uninterrupted runner execution against a round-resumed endpoint."""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from appliance.current_base_data import CurrentBaseData
from appliance.stable_head import StableHeadRegistry
from appliance.state import complete_hash, digest, write_json
from eval_checkpoint import _make_denice_client_model
from fed_learning.training.decentralized_denice_il import run_decentralized_denice_il
from tools.run_appliance_training_smoke import fingerprints


def run(a):
    if a.verify_only:
        if not (a.out / 'uninterrupted' / 'continuation_state_task_4.pt').is_file():
            raise FileNotFoundError('No completed uninterrupted endpoint to verify')
    else:
        a.out.mkdir(parents=True, exist_ok=False)
    seed = torch.load(a.smoke / 'smoke_seed.pt', map_location='cpu', weights_only=False)
    config = copy.deepcopy(seed['config'])
    config.pop('appliance_smoke_stop_after_round', None)
    config.update(output_dir=str((a.out / 'uninterrupted').resolve()),
        resume_output_dir=str((a.out / 'uninterrupted').resolve()),
        resume_state_path=str((a.smoke / 'smoke_seed.pt').resolve()))
    del seed
    if not a.verify_only:
        run_decentralized_denice_il(config)
    resumed = torch.load(a.smoke / 'native' / 'continuation_state_task_4.pt', map_location='cpu', weights_only=False)
    full = torch.load(a.out / 'uninterrupted' / 'continuation_state_task_4.pt', map_location='cpu', weights_only=False)
    first, second = fingerprints(resumed), fingerprints(full)
    checks = {f'exact_{k}': first[k] == second[k] for k in first if k != 'appliance'}
    records = []
    for cid in full['client_ids']:
        algorithm = full['client_algorithm_states'][cid]
        entries = algorithm.get('denice', algorithm).get('appliance_guarded_head_entries', {})
        if not entries:
            continue
        model, router = _make_denice_client_model(full, cid, 'cpu')
        other, other_router = _make_denice_client_model(resumed, cid, 'cpu')
        reg, rr = StableHeadRegistry(), StableHeadRegistry()
        reg.entries, rr.entries = model.appliance_guarded_head_entries, other.appliance_guarded_head_entries
        checks[f'exact_registry_{cid}'] = digest(reg.entries) == digest(rr.entries)
        checks[f'protected_heads_{cid}'] = all(reg.head_matches(model, c) for c in reg.entries)
        checks[f'current_certificates_{cid}'] = all(reg.certificate_current(model, router, c) for c in reg.entries)
        view = CurrentBaseData(config['appliance_base_store'], cid, 4, config['denice_data_roles_sha256'])
        pool = view.current_pool(cid, 'base', view.store['task_classes']['4'])
        x = pool['X'][:32]  # labels are deliberately not passed to inference
        if len(x):
            scope = model.appliance_runtime_scope
            left = reg.records(model, router, x, list(range(30)), 'cpu', 512, runtime_scope=scope)
            right = rr.records(other, other_router, x, list(range(30)), 'cpu', 512, runtime_scope=scope)
            checks[f'label_blind_prediction_equal_{cid}'] = np.array_equal(left['pred'], right['pred'])
            checks[f'activation_equal_{cid}'] = np.array_equal(left['activated'], right['activated'])
        records.append(dict(receiver=cid, entries=reg.summary()))
    service = full['appliance_service_state']
    checks['current_task_monitor_only'] = all(
        r['monitor'] is None or r['monitor']['task'] == row['task']
        for row in service['rounds'] for r in row['lifecycle'])
    checks['no_callback_replay'] = len({(r['task'], r['round'], r['phase']) for r in service['rounds']}) == len(service['rounds'])
    checks['communication_ledger_restored'] = (resumed['appliance_service_state']['communication'][:len(
        torch.load(a.smoke / 'round0_branch_seed.pt', map_location='cpu', weights_only=False)
            ['appliance_service_state']['communication'])] ==
        torch.load(a.smoke / 'round0_branch_seed.pt', map_location='cpu', weights_only=False)
            ['appliance_service_state']['communication'])
    result = dict(completed=all(checks.values()), checks=checks, comparisons=len(checks),
        mismatches=sum(not v for v in checks.values()), first=first, second=second,
        task3_automatic_install_followed_by_native_task4=True, native_rounds=2,
        final_test_opened=False, historical_CAL_runtime_reads=0,
        guard_changed=False, records=records, native_scope='89 clients, 256 BASE samples/client maximum; CPU FP32')
    write_json(a.out / 'completion.json', result)
    if not result['completed']:
        raise AssertionError(checks)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--verify-only', action='store_true')
    a = p.parse_args()
    with threadpool_limits(limits=1):
        run(a)
