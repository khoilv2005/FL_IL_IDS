"""Explicit legacy Task4 import. No raw data, backbone or expert optimization."""
from copy import deepcopy
import json
from pathlib import Path
import pickle
import sys
import zipfile

from candle_v4_historical_repair import SOURCE, copy_member, member, sha256, repair_memories

TASK4_SHA = '1336e1d91f4301bac73b78005c9ecd195bbd3ec36b9d4832e7b3b63967bed075'
OLD = 'candle_v4_continuous_class_router_v7'
NEW = 'candle_v4_continuous_bootstrap_v8'


def extract_snapshots(archive_path, cache):
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    paths = {}
    with zipfile.ZipFile(archive_path) as outer:
        lock = json.loads(outer.read(member(outer, 'candle_v4_run_lock.json')))
        if lock['source_commit'] != SOURCE or lock['version'] != OLD:
            raise ValueError('Expected original continuous results16 source/protocol')
        for task in range(5):
            nested = cache / f'input_task_{task}.zip'
            copy_member(outer, member(outer, f'checkpoint_task_{task}_all_rounds.zip'), nested)
            try:
                with zipfile.ZipFile(nested) as inner:
                    manifest = json.loads(inner.read('checkpoint_archive_manifest.json'))
                    name = f'continuation_state_task_{task}.pt'
                    if (manifest.get('task_id') != task or manifest.get('completed_rounds') != list(range(20))
                            or manifest.get('continuation_checkpoint') != name):
                        raise ValueError('Incomplete historical task boundary')
                    paths[task] = copy_member(inner, name, cache/name, manifest['checksums'][name])
            finally:
                nested.unlink(missing_ok=True)
    if sha256(paths[4]) != TASK4_SHA:
        raise ValueError('Not the locked results16 Task4 continuation')
    return paths, lock


def migrate_state(state, snapshots, spec):
    """Mutate only service protocol, routing memory and explicit provenance."""
    from fed_learning.training.candle_v4_shadow import digest
    meta = state['meta']; saved = state['candle_v4_shadow_state']
    if (meta.get('completed_task') != 4 or meta.get('resume_from_task') != 5
            or meta.get('boundary') != 'task' or state.get('round_state') is not None):
        raise ValueError('Task4 completed boundary required')
    if saved['version'] != OLD or spec['version'] != NEW or spec.get('evaluation_mode') != 'v4_only':
        raise ValueError('Unsupported explicit migration')
    oldspec = saved['spec']
    allowed = {'version', 'data_dir', 'output_dir', 'evaluation_mode'}
    if {k:v for k,v in oldspec.items() if k not in allowed} != {k:v for k,v in spec.items() if k not in allowed}:
        raise ValueError('Training/activation budget differs from original run')
    immutable = {k:digest(v) for k,v in state.items() if k not in ('config', 'candle_v4_shadow_state')}
    expert_before = digest(saved['engines'])
    staged, provenance = repair_memories(state, snapshots)
    wire = sum(len(pickle.dumps(m.state_dict(), protocol=5)) for m in staged.values())
    for cid, memory in staged.items(): saved['continuous_routers'][cid] = memory.state_dict()
    saved.update(version=NEW, spec=deepcopy(spec), continuous_bootstrap_receipts=[])
    saved['bytes']['continuous_bootstrap'] = 0  # Future exact-clone traffic only.
    saved['bytes']['historical_migration'] = wire
    for certs in saved['certificates'].values():
        for cert in certs.values():
            if cert['version'] != OLD: raise ValueError('Unexpected expert certificate protocol')
            cert['version'] = NEW
    state['config']['candle_v4_full_service'] = deepcopy(spec)
    for key, before in immutable.items():
        if digest(state[key]) != before: raise ValueError('Migration changed '+key)
    if digest(saved['engines']) != expert_before: raise ValueError('Expert/Adam state changed')
    return dict(version='candle_v4_task4_explicit_migration_v1', original_source=SOURCE,
                input_task4_sha256=TASK4_SHA, repaired_clients=len(staged), provenance=provenance,
                historical_migration_serialized_bytes=wire, native_and_rng_unchanged=True,
                expert_and_optimizer_unchanged=True, historical_raw_data_read=False,
                exact_legacy_resume=False, resume_task=5, resume_round=0)


def migrate_archive(archive, output, config, spec):
    import sklearn
    import torch
    if sklearn.__version__ != '1.6.1': raise RuntimeError('Migration requires pinned sklearn1.6.1')
    output = Path(output)
    if any(output.iterdir()): raise ValueError('Migration requires an empty output directory')
    cache = Path('/kaggle/temp/candle_v4_task4_migration')
    paths, lock = extract_snapshots(archive, cache)
    if lock['config']['candle_v4_full_service']['input_sha256'] != spec['input_sha256']:
        raise ValueError('Original dataset metadata changed')
    snapshots = {t:torch.load(p, map_location='cpu', weights_only=False, mmap=True) for t,p in paths.items()}
    state = snapshots[4]
    receipt = migrate_state(state, snapshots, spec)
    # Preserve historical configuration; only paths/service identity need to change.
    state['config'].update(data_dir=config['data_dir'], output_dir=config['output_dir'],
                           resume_output_dir=config['output_dir'])
    target = output/'continuation_state_task_4_migrated.pt'
    partial = target.with_suffix('.pt.partial')
    torch.save(state, partial); partial.replace(target)
    receipt['migrated_checkpoint_sha256'] = sha256(target)
    (output/'task4_migration_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(f'Task4 migration PASS: {receipt["repaired_clients"]} repaired routers; resume T5/R0.', flush=True)
    return target
