"""Audit missing retrospective profiles without fitting or granting task access."""
import argparse
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import torch


def unique_member(archive, name):
    matches=[n for n in archive.namelist() if Path(n).name==name]
    if len(matches)!=1: raise ValueError(f'Expected one {name}, got {len(matches)}')
    return matches[0]


def audit(results_zip, matched_zip, out, data_dir=None):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(results_zip) as z:
        debug=json.loads(z.read(unique_member(z,'denice_debug_history.json')))
        checkpoint=torch.load(io.BytesIO(z.read(unique_member(z,'checkpoint_task_5_round_19.pt'))),
                              map_location='cpu',weights_only=False)
    with zipfile.ZipFile(matched_zip) as z:
        lifecycle=pd.read_csv(z.open('profile_lifecycle_audit.csv'))
        protocol=json.loads(z.read('protocol.json'))
    config=checkpoint['config']
    if not str(config.get('git_commit','')).startswith('03b9b53'):
        raise ValueError('This audit expects original training commit 03b9b53')
    starts={int(r['task']):r for r in debug if r.get('type')=='task_start'}
    if set(starts)!=set(range(6)):
        raise ValueError('Incomplete task-start history; cannot establish absence of participation')
    missing=lifecycle[(lifecycle.local_class_count>0)&~lifecycle.continuous_profile_present]
    rows=[]
    for pair in missing.itertuples():
        cid,task=int(pair.client_id),int(pair.task)
        alg=checkpoint['client_algorithm_states']
        state=alg.get(cid,alg.get(str(cid)))
        state=state.get('denice',state)
        detector=state['context_detector']
        completed=[int(t['task_id']) for t in (state.get('cgofed_projection_state') or {}).get('tasks',[])]
        refreshed=detector.get('router_last_refresh_task')
        evidence=task in completed or refreshed==task
        participation=[t for t,r in starts.items() if str(cid) in r['client_data']]
        recorded=starts[task]['client_data'].get(str(cid))
        events=[dict(task=t,**event) for t,r in starts.items()
                for event in r.get('bootstrap_events',[]) if int(event['client_id'])==cid]
        clone=next((event for event in events if event['bootstrap_policy']=='representative_clone'),None)
        raw_rows=None; usable_rows=None; raw_status='not_read_original_shard_not_supplied'
        if data_dir:
            path=Path(data_dir)/f'client_{cid}_train.npz'
            if path.exists():
                with np.load(path) as shard: labels=shard['y_train']
                task_classes=protocol['task_classes'][str(task)]
                supported=detector['episode_classes'].get(task,[])
                raw_rows=int(np.isin(labels,task_classes).sum())
                usable_rows=int(np.isin(labels,sorted(set(task_classes)&set(supported))).sum())
                raw_status='read_original_npz_labels'
            else: raw_status='original_shard_missing'
        # Proven absence before client creation is stronger than merely finding
        # no local completed-task record in a final checkpoint.
        if (clone and task<int(clone['task']) and participation and min(participation)>task
                and recorded is None and not evidence):
            reason='A'; detail='pre_join_task_binary_memory_inherited_via_representative_clone'
        elif recorded and not evidence:
            reason='B'; detail='local_task_data_recorded_but_evaluator_participation_evidence_missing'
        elif evidence and usable_rows==0:
            reason='C'; detail='eligible_task_but_no_usable_local_training_rows'
        elif evidence and usable_rows is not None and usable_rows>0:
            reason='D'; detail='eligible_task_with_usable_rows_but_profile_absent_requires_fitter_audit'
        else:
            reason='UNRESOLVED'; detail='insufficient_participation_or_raw_row_evidence'
        rows.append(dict(client_id=cid,task_id=task,local_class_count=int(pair.local_class_count),
            binary_memory_present=bool(pair.binary_memory_present),
            cgofed_completed_task=task in completed,router_last_refresh_task=refreshed,
            evaluator_evidence_present=evidence,continuous_profile_present=bool(pair.continuous_profile_present),
            first_local_task=min(participation) if participation else None,
            local_participation_at_task=recorded is not None,
            logged_rows_at_task=recorded['num_samples'] if recorded else None,
            bootstrap_task=clone['task'] if clone else None,
            bootstrap_policy=clone['bootstrap_policy'] if clone else None,
            bootstrap_source=clone.get('bootstrap_source') if clone else None,
            rejoining_events=json.dumps([e for e in events if e['bootstrap_policy']=='rejoining_plastic_catch_up']),
            local_shard_task_rows=raw_rows,available_rows=usable_rows,raw_rows_status=raw_status,
            feature_version_status='not_applicable_no_retrospective_profile_created',
            reason=reason,reason_detail=detail))
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'profile_provenance_pairs.csv',index=False)
    # Correct display labels using presence, without changing original metrics.
    lifecycle['status']=np.where(lifecycle.continuous_profile_present,'profile_present',
                                  'needs_participation_provenance_audit')
    lifecycle.to_csv(out/'profile_lifecycle_corrected.csv',index=False)
    summary=dict(training_commit=config['git_commit'],pairs=len(frame),clients=int(frame.client_id.nunique()),
        counts={key:int((frame.reason==key).sum()) for key in ('A','B','C','D','UNRESOLVED')},
        original_shards_read=bool(data_dir),
        limitation='Absent historical task_start membership is not a direct count of rows in a current raw shard',
        fit_or_training_performed=False)
    (out/'profile_provenance_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-zip',required=True); p.add_argument('--matched-zip',required=True)
    p.add_argument('--out',required=True); p.add_argument('--data-dir')
    a=p.parse_args(); audit(a.results_zip,a.matched_zip,a.out,a.data_dir)
