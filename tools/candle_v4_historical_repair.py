"""Historical memory repair utilities; no raw data, backbone or expert training."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np


SOURCE='37a277c8f6ae85e74c9553d4d30e5e27f052ca93'
FINAL_SHA='715deb2ebb60991860324a2deb31b57ab1257f6a5a1c686ff17f320c816a303e'


def sha256(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def member(archive,suffix):
    matches=[n for n in archive.namelist() if n==suffix or n.endswith('/'+suffix)]
    if len(matches)!=1:raise ValueError('Expected one archive member: '+suffix)
    return matches[0]


def copy_member(archive,name,target,expected=None):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and expected and sha256(target)==expected:return target
    partial=target.with_suffix(target.suffix+'.partial')
    with archive.open(name) as src,partial.open('wb') as dst:shutil.copyfileobj(src,dst,8*1024*1024)
    if expected and sha256(partial)!=expected:raise ValueError('Checkpoint checksum failed: '+name)
    partial.replace(target);return target


def prepare_archives(results16,results17,cache):
    """Extract only task continuations; never expand20 full round models/task."""
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(results17) as archive:
        lock=json.loads(archive.read(member(archive,'candle_v4_run_lock.json')))
        completion=json.loads(archive.read(member(archive,'candle_v4_completion.json')))
        if lock['source_commit']!=SOURCE or not completion.get('completed'):
            raise ValueError('Expected completed results17 from the locked continuous V4 run')
        final=copy_member(archive,member(archive,'continuation_state_latest.pt'),cache/'continuation_state_task_5.pt',FINAL_SHA)
        old=copy_member(archive,member(archive,'candle_v4_final_test.json'),cache/'original_final_test.json')
    paths={5:final}
    with zipfile.ZipFile(results16) as archive:
        lock16=json.loads(archive.read(member(archive,'candle_v4_run_lock.json')))
        if lock16!=lock:raise ValueError('results16 and results17 are different runs')
        for task in range(5):
            target=cache/f'continuation_state_task_{task}.pt'
            receipt=cache/f'task_{task}_extraction.json'
            if target.exists() and receipt.exists() and sha256(target)==json.loads(receipt.read_text())['sha256']:
                paths[task]=target;continue
            nested=cache/f'input_task_{task}.zip'
            copy_member(archive,member(archive,f'checkpoint_task_{task}_all_rounds.zip'),nested)
            with zipfile.ZipFile(nested) as inner:
                manifest=json.loads(inner.read('checkpoint_archive_manifest.json'))
                expected_name=f'continuation_state_task_{task}.pt'
                if manifest.get('task_id')!=task or manifest.get('completed_rounds')!=list(range(20)) or manifest.get('continuation_checkpoint')!=expected_name:
                    raise ValueError('Incomplete task-boundary checkpoint')
                expected=manifest['checksums'][expected_name]
                paths[task]=copy_member(inner,expected_name,target,expected)
                receipt.write_text(json.dumps(dict(task=task,sha256=expected)),encoding='utf-8')
            # Only this utility's newly extracted temporary nested archive.
            nested.unlink()
            print('Verified historical continuation T'+str(task),flush=True)
    return paths,old


def context(state,cid):
    a=state['client_algorithm_states'][cid];return a.get('denice',a)['context_detector']


def lineage_from_state(state):
    rows={int(cid):dict(local_tasks=[],events=[]) for cid in state['client_model_states']}
    for record in state['debug_history']:
        for cid,data in record.get('client_data',{}).items():
            if data.get('num_samples',0):rows[int(cid)]['local_tasks'].append(int(record['task']))
        for event in record.get('bootstrap_events',[]):
            rows[int(event['client_id'])]['events'].append(dict(task=int(record['task']),**event))
    return rows


def find_origin(lineage,cid,task,visited=()):
    if cid in visited:raise ValueError('Cycle in exact-clone lineage')
    row=lineage[cid]
    if task in row['local_tasks']:return cid,task,[*visited,cid]
    event=next((e for e in row['events'] if e['bootstrap_policy']=='representative_clone'),None)
    if event is None or task>=event['task'] or event['param_distance_to_source']!=0:
        raise ValueError('No exact historical feature origin')
    return find_origin(lineage,int(event['bootstrap_source']),task,(*visited,cid))


def repair_memories(final,snapshots):
    """All clients or fail: preserve unaffected estimators, fit missing banks only."""
    from fed_learning.servers.continuous_class_router import ContinuousClassMemory
    from fed_learning.training.candle_v4_shadow import digest
    from threadpoolctl import threadpool_limits
    lineage=lineage_from_state(final);staged={};provenance={}
    saved=final['candle_v4_shadow_state']['continuous_routers']
    if set(saved)!=set(final['client_model_states']):raise ValueError('Incomplete final router/model state')
    for cid,raw in sorted(saved.items()):
        original=ContinuousClassMemory.restore(raw);native=context(final,cid)
        native_tasks={int(t) for t,v in native['activation_memory'].items() if len(v)}
        missing=sorted(native_tasks-set(original.class_to_task.values()))
        if not missing:continue
        mem=deepcopy(original.state_dict());entries=[]
        for task in missing:
            owner,t,chain=find_origin(lineage,cid,task);ck=snapshots[t]
            donor=ck['candle_v4_shadow_state']['continuous_routers'][owner];nd=context(ck,owner)
            if not np.array_equal(original.indices,donor['indices']):raise ValueError('Feature indices differ')
            for key in ('stable_feature_mask','routing_feature_mask','binarize_thresholds'):
                if digest(native.get(key))!=digest(nd.get(key)):raise ValueError('Incompatible '+key)
            if (digest(native['activation_memory'][task])!=digest(nd['activation_memory'][task]) or
                    native['episode_classes'][task]!=nd['episode_classes'][task]):
                raise ValueError('Historical memory is not the inherited native task state')
            added=[]
            for c,v in donor['memory'].items():
                if c//6==task:
                    if c in mem['memory']:raise ValueError('Do not overwrite local memory')
                    mem['memory'][c]=v.copy();mem['class_to_task'][c]=task;added.append(int(c))
            if not added:raise ValueError('Empty historical source class memory')
            entries.append(dict(task=task,source_client=owner,snapshot_task=t,clone_chain=chain,
                classes=added,source_memory_sha256=digest({c:donor['memory'][c] for c in added}),
                native_sketches_thresholds_coordinates_equal=True))
        repaired=ContinuousClassMemory(original.per_class)
        for key in ('indices','full_width','latest_task','memory','class_to_task'):setattr(repaired,key,deepcopy(mem[key]))
        with threadpool_limits(limits=2):repaired.fit(repaired.latest_task)
        if set(repaired.class_to_task.values())!=native_tasks:raise ValueError('Repair left missing/extra task support')
        staged[cid]=repaired
        provenance[cid]=dict(original_tasks=sorted(set(original.class_to_task.values())),
            repaired_tasks=sorted(native_tasks),imports=entries,
            before_feature_bytes=original.stats()['feature_bytes'],after_feature_bytes=repaired.stats()['feature_bytes'])
    return staged,provenance
