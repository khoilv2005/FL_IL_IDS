"""Locked P2-Shallow native T0->T1 feasibility; current roles, never TEST."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from fed_learning.data.denice_clean_roles import CleanRoleData, file_sha256
from fed_learning.models.denice_model import DeNICEModel
from fed_learning.strategies.decentralized.denice_v2.contracts import encode, decode, state_hash
from fed_learning.strategies.decentralized.denice_v2.knowledge import ClassKnowledgeLedger
from fed_learning.strategies.decentralized.denice_v2.native import (
    CurrentBase, capture_native, consolidate_native_task, load_native, run_native_round)
from fed_learning.strategies.decentralized.denice_v2.router import (
    ClusterRouter, probe_router_capacity, stage_cluster_router)
from fed_learning.strategies.decentralized.denice_v2.state import handoff, restore, snapshot


def prepare_current_roles(roles, cache):
    """Dataset preparation before model initialization, not an online replay bank.

    Source NPZs contain multiple tasks. Split them once offline; the native
    transition reads only its current-task files, including validation.
    """
    directory = cache/'p2_shallow_current_roles'
    directory.mkdir(parents=True,exist_ok=True)
    files = {}
    for cid in (9,30):
        for role,cap in (('base',32),('validation',64)):
            x,y,rows = roles.client_role(cid,role)
            for task in (0,1):
                chosen = np.concatenate([np.flatnonzero(y == c)[:cap] for c in range(task*6,task*6+6)])
                path = directory/f'task_{task}_client_{cid}_{role}.npz'
                np.savez(path,x=x[chosen],y=y[chosen],row_ids=rows[chosen])
                files[path.name] = file_sha256(path)
            del x,y,rows
    return directory,files


def current_rows(prepared, cid, role, task, cap):
    directory,hashes = prepared
    path = directory/f'task_{task}_client_{cid}_{role}.npz'
    if file_sha256(path) != hashes[path.name]:
        raise ValueError('Prepared current-role file changed')
    with np.load(path,allow_pickle=False) as data:
        x,y,rows = data['x'],data['y'],data['row_ids']
    if not set(y.tolist()) <= set(range(task*6,task*6+6)):
        raise ValueError('Historical/future rows in current role')
    if any(n > cap for n in np.unique(y,return_counts=True)[1]):
        raise ValueError('Current role exceeds locked cap')
    return torch.from_numpy(x),torch.from_numpy(y),rows


def current_bases(prepared, task):
    data = {}
    for cid in (9, 30):
        x, y, rows = current_rows(prepared, cid, 'base', task, 32)
        data[cid] = CurrentBase(cid, task, x, y, rows,
            prepared[1][f'task_{task}_client_{cid}_base.npz'])
    return data


def validation_bits(prepared, state):
    predictor = ClusterRouter(state)
    banks, classes = [], []
    for cid in state.members:
        x, y, rows = current_rows(prepared, cid, 'validation', state.task, 64)
        if len(x):
            banks.append(predictor.predict(x)[3])
            classes.extend(y.tolist())
        del x, y, rows
    if not banks:
        raise ValueError('No current VALIDATION evidence')
    return np.concatenate(banks), {int(c): int(classes.count(c)) for c in sorted(set(classes))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out, cache = Path(args.out), Path(args.cache)
    if out.exists():
        raise ValueError('Use a new receipt path')
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    roles = CleanRoleData(cache/'roles', cache/'dataset')
    if not json.loads((cache/'roles/role_lock.json').read_text())['locked_before_backbone_training']:
        raise ValueError('Roles not locked')
    input_shape = tuple(roles.manifest['input_shape'])
    prepared = prepare_current_roles(roles,cache)
    del roles
    protocol = Path(__file__).resolve().parents[1]/'docs/DENICE_V2_P2_SHALLOW_PROTOCOL_20261012_VI.md'
    report = dict(protocol_sha256=file_sha256(protocol), feature_policy='conv1_only_v1',
        test_read=False, historical_raw_replay=False, threshold_tuning=False, seed=42,
        rounds_per_task=1, training_base_cap_per_class=32, validation_cap_per_class=64,
        membership_fixture=[9, 30], scheduler_full_campaign=False, checks={},
        role_preparation=dict(phase='offline_before_model_initialization',source_npz_multitask=True,
            prepared_file_sha256=prepared[1],online_task1_reads_only_current_files=True))
    router = dict(router_mode='binary_cosine', context_layers=('conv1','conv2','conv3','gru'),
        stable_feature_mask=np.zeros(548,bool), binarize_thresholds=None, activation_memory={}, episode_classes={})
    state = capture_native(DeNICEModel(input_shape,34),
        members=(9,30),coordinator_id=9,cluster_uid='p2-shallow-t0',router_state=router,
        routing_dependencies=('bn1.num_batches_tracked',))
    bases = current_bases(prepared,0)
    print('P2-Shallow: native Task0 round',flush=True)
    result, plan, packets, training = run_native_round(state,bases,tuple(range(6)),seed=42)
    report['task0_training'] = training
    probe = probe_router_capacity(result.state,'conv1_only_v1')
    report['task0_capacity'] = probe
    report['checks']['capacity_preflight'] = probe['capacity']['passed']
    if probe['capacity']['passed']:
        proposal0 = stage_cluster_router(result.state,plan,packets,bases,feature_policy='conv1_only_v1')
        state0 = proposal0.commit()
        report['task0_router'] = proposal0.receipt
        bits0, counts0 = validation_bits(prepared,state0)
        report['validation_class_counts_task0'] = counts0
        historical_hash = state_hash(state0.router_state['activation_memory'][0])
        thresholds_hash = state_hash(state0.router_state['binarize_thresholds'])
        ledger = ClassKnowledgeLedger({t: tuple(range(t*6,min(t*6+6,34))) for t in range(6)})
        for p in packets:
            ledger.record(p.event,at_task=0,at_round=0)
        consolidated = consolidate_native_task(state0)
        next_state = handoff([consolidated],members=(9,30),cluster_uid='p2-shallow-t1',task=1,ledger=ledger)
        # Release current T0 inputs before any T1 read/training.
        del bases, plan, packets, result, proposal0
        gc.collect()
        bases = current_bases(prepared,1)
        witness = torch.cat([b.x[:4] for b in bases.values()])
        old_features = load_native(next_state).eval().get_context_activations_per_sample(witness)['conv1']
        print('P2-Shallow: native Task1 round (client9 zero BASE)',flush=True)
        result, plan, packets, training = run_native_round(next_state,bases,tuple(range(6,12)),seed=42)
        report['task1_training'] = training
        report['task1_capacity'] = probe_router_capacity(plan.reference,'conv1_only_v1')
        report['checks']['other_layers_learner_capacity'] = all(
            report['task1_capacity']['capacity']['per_layer'][k]['currently_trainable_after'] > 0
            for k in ('conv2','conv3','gru'))
        proposal1 = stage_cluster_router(result.state,plan,packets,bases,feature_policy='conv1_only_v1')
        final = proposal1.commit()
        report['task1_router'] = proposal1.receipt
        new_features = load_native(final).eval().get_context_activations_per_sample(witness)['conv1']
        report['checks']['conv1_current_witness_exact'] = torch.equal(old_features,new_features)
        report['checks']['routing_dependency_hash_exact'] = final.routing_space_hash == state0.routing_space_hash
        report['checks']['historical_bank_exact'] = state_hash(final.router_state['activation_memory'][0]) == historical_hash
        report['checks']['thresholds_exact'] = state_hash(final.router_state['binarize_thresholds']) == thresholds_hash
        report['checks']['native_optimizer_task1'] = training['workers'][1]['steps'] > 0
        report['checks']['empty_client_zero_step'] = training['workers'][0]['steps'] == 0
        bits1, counts1 = validation_bits(prepared,final)
        report['validation_class_counts_task1'] = counts1
        predictor = ClusterRouter(final)
        bits = np.concatenate([bits0,bits1])
        truth = np.concatenate([np.zeros(len(bits0),int),np.ones(len(bits1),int)])
        task_pred, scores = predictor.detector.predict_episodes_with_scores(bits)
        confusion = [[int(((truth==t)&(task_pred==p)).sum()) for p in (0,1)] for t in (0,1)]
        recalls = [(task_pred[truth==t]==t).mean().item() for t in (0,1)]
        report['routing_discrimination'] = dict(confusion=confusion,recall_by_task=recalls,
            balanced_task_accuracy=float(np.mean(recalls)), evidence='T0 frozen VALIDATION bits + current T1 VALIDATION bits',
            historical_raw_reencoded=False, gate_balanced_accuracy=.60,gate_min_task_recall=.25,
            predicted_task_counts={t:int((task_pred==t).sum()) for t in (0,1)})
        report['checks']['router_discrimination'] = min(recalls) >= .25 and float(np.mean(recalls)) >= .60
        checkpoint = cache/'p2_shallow_model_router_state.json'
        checkpoint.write_text(json.dumps(encode(snapshot(final))),encoding='utf-8')
        restored = restore(decode(json.loads(checkpoint.read_text(encoding='utf-8'))))
        report['checkpoint_restore'] = dict(format='typed_json_cluster_state',
            file_sha256=file_sha256(checkpoint),bytes=checkpoint.stat().st_size,
            scope='model_router_state_only_not_full_optimizer_resume')
        before, after = predictor.predict(witness), ClusterRouter(restored).predict(witness)
        report['checks']['restore_predictions_exact'] = all(np.array_equal(a,b) for a,b in zip(before,after))
        report['checks']['restore_state_exact'] = final.fingerprint() == restored.fingerprint()
        report['restored_state_hash'] = restored.fingerprint()
    report['completed'] = True
    report['decision'] = 'GO' if all(report['checks'].values()) else 'NO_GO'
    report['p2_shallow_pass'] = report['decision'] == 'GO'
    report['full_training_started'] = False
    report['elapsed_seconds'] = time.monotonic()-started
    report['base_manifest_sha256'] = file_sha256(cache/'roles/role_manifest.json')
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__),protocol,*sorted((root/'fed_learning/strategies/decentralized/denice_v2').glob('*.py'))]
    report['source_sha256'] = {p.relative_to(root).as_posix():file_sha256(p) for p in paths}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(decision=report['decision'],checks=report['checks'],
        routing=report.get('routing_discrimination')),indent=2),flush=True)


if __name__ == '__main__':
    main()
