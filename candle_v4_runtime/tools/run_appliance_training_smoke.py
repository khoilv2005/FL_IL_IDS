"""Two-task runner smoke plus controlled automatic-install integration test.

No manual patches. The late-start smoke seeds numerical BASE summaries only
from owned chronological BASE partitions; CAL runtime opens Task3/4 only.
It is a bounded integration check, not a fresh six-task scientific run.
"""
import argparse
import copy
import gc
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from appliance.base_sketch_shield import CurrentBaseSketchShield
from appliance.current_base_data import CurrentBaseData
from appliance.native_lifecycle_experiment import load_continuation
from appliance.portable_route import SharedSketch
from appliance.runner import load_input
from appliance.state import complete_hash, digest, write_json
from appliance.training_service import ApplianceTrainingService
from eval_checkpoint import _make_denice_client_model
from fed_learning.data.denice_clean_roles import CleanRoleData, file_sha256
from fed_learning.training.checkpoint_state import snapshot_denice_state
from fed_learning.training.decentralized_denice_il import run_decentralized_denice_il


def config_for(saved, a, out):
    config = copy.deepcopy(saved)
    for k in ('appliance_current_native_manifest', 'appliance_native_probe_manifest'):
        config.pop(k, None)
    config.update(data_dir=str(a.data_dir.resolve()), denice_clean_roles_dir=str(a.roles.resolve()),
        output_dir=str(out.resolve()), resume_output_dir=str(out.resolve()), task_end=4,
        appliance_enabled=True, appliance_base_store=str((a.store / 'base_store').resolve()),
        appliance_calibration_store=str((a.store / 'calibration_store').resolve()),
        appliance_batch_size=512, appliance_application_domain='cumulative_dataset',
        appliance_integration_smoke=True,
        rounds_per_task=2, denice_max_train_samples_per_client=256, denice_amp_enabled=False,
        denice_post_task_eval=False, denice_eval_final_round=False, denice_eval_last_round_only=False,
        denice_eval_terminal_state_only=True, denice_eval_local_validation=False,
        denice_evaluation_data_role='validation', eval_every=999999,
        round_checkpoint_every=None, denice_archive_checkpoints=False,
        denice_save_round_artifacts=False, save_continuation_every_task=True,
        denice_cme_after_each_task=False, denice_cme_tasks=[])
    if config.get('denice_cl_method') != 'legacy' or config['denice_similarity_threshold'] != .8:
        raise ValueError('Smoke requires the original legacy xi=.8 checkpoint')
    if getattr(a,'scope_mode',None):
        config['appliance_scope_mode']=a.scope_mode
    return config


def seed_base_summaries(state, a, task):
    """Explicit late-start fixture setup. No historical CAL, no head changes."""
    role_sha = file_sha256(a.roles / 'role_manifest.json')
    for cid in state['client_ids']:
        view = CurrentBaseData(a.store / 'base_store', int(cid), 0, role_sha)
        shield = CurrentBaseSketchShield(view, SharedSketch(tuple(view.store['input_shape']), 16,
                                                          view.store['metadata_sha256']))
        for t in range(task + 1):
            if t:
                view.advance(t)
            shield.observe_current(view)
        alg = state['client_algorithm_states'][cid]
        alg = alg.get('denice', alg)
        alg['appliance_base_sketch_shield_state'] = shield.state()
        print(f'Smoke setup own BASE summaries client={cid} through_task={task}; CAL unread', flush=True)


def fingerprints(state):
    return dict(models=digest(state['client_model_states']), algorithms=digest(state['client_algorithm_states']),
        rng=digest(state['rng_state']), novelty=digest(state['novelty_states']),
        prev_ages=digest(state['prev_ages']), old_refs=digest(state['old_ref_banks']),
        last_active=digest(state['last_active_task']),
        # UUID sessions and measured seconds are not prediction state. Exact
        # state comparison is appropriate when both branches resume the same
        # already-committed seed and no new acceptance session is opened.
        appliance=digest(state.get('appliance_service_state')))


def controlled(a, out):
    """The exact production callback on actual mature Task3 endpoints/graph."""
    ckpt, hashes = load_input(a.checkpoint_dir / 'checkpoint_task_3_all_rounds.zip', 3, 19)
    ids = sorted(map(int, ckpt['client_model_states']))
    # Evaluation terminal checkpoint has no continuation bookkeeping; seed
    # only BASE summary provenance in its algorithm payload.
    ckpt['client_ids'] = ids
    seed_base_summaries(ckpt, a, 2)
    config = config_for(ckpt['config'], a, out)
    models, routers, clients = {}, {}, {}
    for cid in ids:
        models[cid], routers[cid] = _make_denice_client_model(ckpt, cid, 'cpu')
        clients[cid] = SimpleNamespace(model=models[cid])
    service = ApplianceTrainingService(config)
    service.begin_task(3, ids, models)
    before = {cid: digest(models[cid].state_dict()) for cid in ids}
    # Recorded graph helper uses the original terminal group; adapt back to the
    # same live runner cluster payload without inventing edges or weights.
    from appliance.selector import recorded_graph
    groups, alphas = recorded_graph(ckpt, 3, 19)
    cluster = dict(groups={cid: list(alphas[cid]) for cid in ids},
        alpha_debug={cid: dict(group_ids=list(alphas[cid]), alphas=list(alphas[cid].values())) for cid in ids})
    result = service.after_refresh(3, 19, models, routers, clients, ids, cluster, list(range(24)), 'task_finalized')
    commits = [r for r in result['transactions'] if r['transaction']['applied']]
    rejects = [r for r in result['transactions'] if not r['transaction']['applied']]
    checks = dict(automatic_install=len(commits) > 0,
        atomic_client_model_commit=all(clients[p['pair']['receiver']].model is models[p['pair']['receiver']] for p in commits),
        rejected_sources_unchanged=all(p['transaction'].get('rollback_verified') and
            before[p['pair']['receiver']] == digest(models[p['pair']['receiver']].state_dict()) for p in rejects),
        actual_graph_positive_donors=all(p['pair']['donor'] in groups[p['pair']['receiver']] for p in result['transactions']),
        registry_certificates_protected=all(service.registry(models[p['pair']['receiver']]).head_matches(
            models[p['pair']['receiver']], p['pair']['class_id']) for p in commits),
        no_historical_CAL=all(v.task == 3 and all(e['task'] == 3 for e in v.access_log) for v in service.views.values()),
        setup_accounted=service.communication_summary()['by_kind'].get('current_receiver_function_capsule', 0) > 0,
        no_manually_prepared_packet=all(not r.get('prepared_packet_used', True) for r in result['transactions']))
    if config.get('appliance_scope_mode')=='appliance_empirical_current_CAL_v2':
        from appliance.patch_lifecycle import route_authorized
        from appliance.imported_route import ROUTE_RULES,stratified_roles
        activation=[]
        for item in commits:
            cid,donor,c=(item['pair'][k] for k in ('receiver','donor','class_id'))
            reg=service.registry(models[cid]);e=reg.entries[c]
            # Same already-used locked current donor CAL HOLDOUT: functional
            # activation verification only, never independent test evidence.
            pool=service.views[donor].current_pool(donor,'calibration',
                service.views[donor].store['task_classes']['3'])
            held=stratified_roles(pool,ROUTE_RULES['seed']+donor)['holdout']
            held=held[pool['y'][held]==c]
            result_pred=reg.records(models[cid],routers[cid],pool['X'][held],list(range(24)),
                'cpu',512,runtime_scope=service.runtime_scope(list(range(24))))
            activation.append(dict(receiver=cid,class_id=c,rows=len(held),
                activated=int(result_pred['activated'].sum()),
                correct=int((result_pred['pred']==pool['y'][held]).sum()),
                authorized=route_authorized(e,service.runtime_scope(list(range(24)))),
                seal_unchanged=e['current_acceptance']==e['lifecycle_certificate']['initial_acceptance']))
        checks['empirical_routes_authorized']=all(r['authorized'] for r in activation)
        checks['actual_imported_activation']=bool(activation) and all(r['activated']>0 for r in activation)
        checks['initial_acceptance_immutable']=all(r['seal_unchanged'] for r in activation)
        write_json(out/'empirical_activation.json',dict(records=activation,
            independent_test=False,source='same locked current donor CAL HOLDOUT; endpoint functional verification'))
    # A controlled rejected transaction goes through the real staging service:
    # deliberately request a second capability, which V1 explicitly rejects.
    if commits:
        from appliance.current_installation import install_current_pair
        p = commits[0]['pair']; cid, donor = p['receiver'], p['donor']
        source = complete_hash(models[cid], routers[cid])
        _, _, _, rejection = install_current_pair(p, models[cid], routers[cid], models[donor], routers[donor],
            config, service.views[cid], service.views[donor], service.shields[cid], groups, alphas, list(range(24)),
            service.registry(models[cid]).entries[p['class_id']]['lifecycle_certificate']['scope'], out / 'controlled_rejection')
        checks['controlled_rejection_rollback'] = (not rejection['transaction']['applied'] and
            rejection['transaction'].get('rollback_verified') and complete_hash(models[cid], routers[cid]) == source)
    snapshots = dict(config=config, client_ids=ids,
        client_model_states={cid: {k: v.detach().cpu().clone() for k, v in models[cid].state_dict().items()} for cid in ids},
        client_algorithm_states={cid: {'denice': snapshot_denice_state(models[cid], routers[cid])} for cid in ids},
        appliance_service_state=service.state_dict())
    torch.save(snapshots, out / 'controlled_automatic_endpoint.pt')
    restored = torch.load(out / 'controlled_automatic_endpoint.pt', map_location='cpu', weights_only=False)
    rr = ApplianceTrainingService(config, restored['appliance_service_state'])
    checks['service_save_restore'] = digest(rr.state_dict()) == digest(service.state_dict())
    checks['model_registry_restore'] = all(complete_hash(*_make_denice_client_model(restored, cid, 'cpu')) ==
        complete_hash(models[cid], routers[cid]) for cid in ids)
    write_json(out / 'completion.json', dict(completed=all(checks.values()), checks=checks,
        commits=len(commits), rejections=len(rejects), **hashes, communication=service.communication_summary(),
        BASE_summary_seed_retrospective=True, historical_CAL_reads=0, final_test_opened=False))
    if not all(checks.values()):
        raise AssertionError(checks)
    return snapshots


def run(a):
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    write_json(a.out / 'completion.json', dict(completed=False))
    if a.controlled_source:
        source = torch.load(a.controlled_source, map_location='cpu', weights_only=False)
        original, _ = load_input(a.checkpoint_dir / 'checkpoint_task_3_all_rounds.zip', 3, 19)
        service_state = source['appliance_service_state']
        transactions = service_state['rounds'][-1]['transactions']
        rejected = [t['pair']['receiver'] for t in transactions if not t['transaction']['applied']]
        checks = dict(rejected_weights_exact=all(digest(source['client_model_states'][cid]) ==
            digest(original['client_model_states'][cid]) for cid in rejected),
            actual_automatic_commits=sum(t['transaction']['applied'] for t in transactions) == 2,
            setup_accounted=any(r['kind']=='current_receiver_function_capsule' for r in service_state['communication']))
        write_json(a.out / 'controlled_integration_independent_audit.json', dict(checks=checks,
            completed=all(checks.values()), note='Previous checker compared intentional runtime-scope metadata; physical rollback checked independently'))
        if not all(checks.values()):
            raise AssertionError(checks)
        del original
    else:
        source = controlled(a, a.out / 'controlled')
    gc.collect()
    archive = a.checkpoint_dir / 'checkpoint_task_3_all_rounds.zip'
    import zipfile
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('checkpoint_archive_manifest.json'))
    state, sha = load_continuation(archive, manifest['checksums'][manifest['full_terminal_checkpoint']], 3)
    for cid in source['client_ids']:
        state['client_model_states'][cid] = source['client_model_states'][cid]
        state['client_algorithm_states'][cid] = source['client_algorithm_states'][cid]
    state['appliance_service_state'] = source['appliance_service_state']
    del source
    config = config_for(state['config'], a, a.out / 'native')
    state['config'] = copy.deepcopy(config)
    torch.save(state, a.out / 'smoke_seed.pt')
    del state
    original_role = CleanRoleData.client_role
    reads = []

    def base_only(self, cid, role, *args, **kwargs):
        if role != 'base':
            raise AssertionError(f'Unscoped runtime role forbidden: {role}')
        reads.append(dict(client=int(cid), role=role))
        return original_role(self, cid, role, *args, **kwargs)

    config.update(resume_state_path=str((a.out / 'smoke_seed.pt').resolve()),
                  appliance_smoke_stop_after_round=[4, 0])
    with patch.object(CleanRoleData, 'client_role', base_only):
        run_decentralized_denice_il(config)
    latest = a.out / 'native' / 'continuation_state_latest.pt'
    paused = torch.load(latest, map_location='cpu', weights_only=False)
    torch.save(paused, a.out / 'round0_branch_seed.pt')
    del paused
    config.pop('appliance_smoke_stop_after_round')
    config['resume_state_path'] = str((a.out / 'round0_branch_seed.pt').resolve())
    with patch.object(CleanRoleData, 'client_role', base_only):
        run_decentralized_denice_il(config)
    first = torch.load(a.out / 'native' / 'continuation_state_task_4.pt', map_location='cpu', weights_only=False)
    fp = fingerprints(first)
    del first
    # Compare the resumed branch with a genuinely uninterrupted two-round run
    # from the same automatic Task3 seed, not two identical resumed branches.
    config.update(output_dir=str((a.out / 'uninterrupted').resolve()),
        resume_output_dir=str((a.out / 'uninterrupted').resolve()),
        resume_state_path=str((a.out / 'smoke_seed.pt').resolve()))
    with patch.object(CleanRoleData, 'client_role', base_only):
        run_decentralized_denice_il(config)
    final = torch.load(a.out / 'uninterrupted' / 'continuation_state_task_4.pt', map_location='cpu', weights_only=False)
    fp2 = fingerprints(final)
    # Timings/session UUIDs are reported separately, not model equality.
    checks = {k: fp[k] == fp2[k] for k in fp if k != 'appliance'}
    service = final['appliance_service_state']
    active4 = set(final['cluster_history'][-1]['groups'])
    current = all(s['task'] == 4 for cid, s in service['calibration_views'].items() if cid in active4)
    checks.update(task3_to4_completed=final['meta']['completed_task'] == 4,
        round_resume_skips_preparation=sum(r['task'] == 4 and r['round'] == 0 for r in service['rounds']) == 1,
        current_CAL_authority=current, no_unscoped_CAL=all(r['role'] == 'base' for r in reads))
    result = dict(completed=all(checks.values()), checks=checks, fingerprints=fp2,
        historical_CAL_runtime_reads=0, final_test_opened=False, task_schedule=[3, 4], rounds_per_task=2,
        training_samples_per_client=256, guard_unchanged=True,
        BASE_summary_seed_retrospective=True, fresh_six_task_campaign_verified=False)
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
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--controlled-source', type=Path,
        help='Resume the automatic Task3 endpoint; independently check old rollback before native Task4')
    p.add_argument('--scope-mode',choices=('initial_scope_v1','appliance_empirical_current_CAL_v2'),
        default='initial_scope_v1')
    p.add_argument('--store', type=Path, default=Path('audit_denice/appliance_current_runtime_data_local/full100_v1'))
    p.add_argument('--checkpoint-dir', type=Path, default=Path('audit_denice/appliance_legacy_results11/inputs'))
    p.add_argument('--roles', type=Path, default=Path('audit_denice/appliance_historical_calibration_local/_runtime_5f5bbe06cee041de/roles'))
    p.add_argument('--data-dir', type=Path, default=Path('audit_denice/appliance_transfer_results9/inputs/dataset'))
    a = p.parse_args()
    with threadpool_limits(limits=1):
        run(a)
