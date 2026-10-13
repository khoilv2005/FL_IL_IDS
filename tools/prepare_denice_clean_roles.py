"""Prepare BASE 80%, META 10%, VALIDATION 10% views without copying raw data.

META is subdivided into 2% calibration and 8% fit. Equal input content across
clients has one role; conflicting labels are excluded. SQLite bounds RAM while
global per-class quotas keep rare-class coverage explicit. Final test is unread.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fed_learning.data.denice_clean_roles import file_sha256
from tools.denice_competence_gate import content_hash


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def prepare(source,out,seed=20261006):
    source=Path(source).resolve();out=Path(out).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Use an empty role directory; never overwrite a locked split')
    out.mkdir(parents=True,exist_ok=True);(out/'indices').mkdir()
    metadata=json.loads((source/'metadata.json').read_text(encoding='utf-8'))
    classes={int(t):list(map(int,values)) for t,values in metadata['task_structure']['task_classes'].items()}
    labels=sorted({c for values in classes.values() for c in values})
    if labels!=list(range(34)) or sorted(classes)!=list(range(6)):raise ValueError('Expected six tasks and classes 0..33')
    files=sorted(source.glob('client_*_train.npz'),key=lambda p:int(p.stem.split('_')[1]))
    if not files:raise ValueError('No original client training shards')
    manifest=dict(completed=False,source_data_dir=str(source),split_seed=seed,content_disjoint=True,
        split_unit='global unique float32 input content, stratified by class',
        requested_fractions={'base':.8,'calibration':.02,'fit':.08,'validation':.1},
        meta_total_fraction=.1,final_test_read=False,final_test_coverage_verified=False,
        global_coverage={},clients={},input_shape=None,metadata_sha256=file_sha256(source/'metadata.json'))
    write_json(out/'role_manifest.json',manifest)
    db=sqlite3.connect(out/'content_roles.sqlite')
    try:
        db.execute('CREATE TABLE content (digest TEXT PRIMARY KEY, label INTEGER, split_key TEXT, conflict INTEGER DEFAULT 0, role TEXT)')
        for path in files:
            cid=int(path.stem.split('_')[1]);checksum=file_sha256(path)
            with np.load(path,allow_pickle=False) as data:x=data['X_train'];y=data['y_train']
            if len(x)!=len(y) or not np.isin(y,labels).all():raise ValueError(f'Invalid shard {cid}')
            shape=list(x.shape[1:])
            if manifest['input_shape'] is not None and manifest['input_shape']!=shape:raise ValueError('Inconsistent input shapes')
            manifest['input_shape']=shape
            for start in range(0,len(y),4096):
                batch=[]
                for row,label in zip(x[start:start+4096],y[start:start+4096]):
                    digest=content_hash(row);split_key=hashlib.sha256(f'{seed}:{digest}'.encode()).hexdigest()
                    batch.append((digest,int(label),split_key))
                db.executemany('INSERT INTO content(digest,label,split_key) VALUES(?,?,?) ON CONFLICT(digest) DO UPDATE SET conflict=MAX(conflict,content.label!=excluded.label)',batch)
            db.commit();manifest['clients'][str(cid)]=dict(source_sha256=checksum,source_rows=len(y))
            print(f'Role scan client={cid}, rows={len(y)}',flush=True)
        db.execute('CREATE INDEX content_class_sort ON content(label,split_key)')
        for label in labels:
            n=db.execute('SELECT COUNT(*) FROM content WHERE label=? AND conflict=0',(label,)).fetchone()[0]
            if n<4:
                # Preserve disjointness without requiring every class in every role.
                # Prefer BASE, then fit, then validation; calibration may be empty.
                base=int(n>=1);cal=0;fit=int(n>=2);val=int(n>=3)
                print(f'Class {label}: only {n} unique inputs; missing roles are report-only.',flush=True)
            else:
                base=max(1,min(n-3,math.floor(.8*n)));cal=max(1,min(n-base-2,math.floor(.02*n)))
                fit=max(1,min(n-base-cal-1,math.floor(.9*n)-base-cal)); val=n-base-cal-fit
            counts={'base':base,'calibration':cal,'fit':fit,'validation':val}
            db.execute('WITH ranked AS (SELECT digest,ROW_NUMBER() OVER (ORDER BY split_key) AS rank FROM content WHERE label=? AND conflict=0) UPDATE content SET role=(SELECT CASE WHEN rank<=? THEN \'base\' WHEN rank<=? THEN \'calibration\' WHEN rank<=? THEN \'fit\' ELSE \'validation\' END FROM ranked WHERE ranked.digest=content.digest) WHERE label=? AND conflict=0',
                (label,base,base+cal,base+cal+fit,label))
            manifest['global_coverage'][str(label)]=counts
        db.commit()
        allocation=metadata.get('client_allocation',{});active=allocation.get('task_active_clients',{})
        joins=allocation.get('client_join_task',{})
        eligible={label:[] for label in labels};task_of={c:t for t,values in classes.items() for c in values}
        eligible_holdout={r:{label:[] for label in labels} for r in ('calibration','fit','validation')}
        for path in files:
            cid=int(path.stem.split('_')[1]);entry=manifest['clients'][str(cid)]
            if file_sha256(path)!=entry['source_sha256']:raise ValueError('Source shard changed during preparation')
            with np.load(path,allow_pickle=False) as data:x=data['X_train'];y=data['y_train']
            rows={r:[] for r in ('base','calibration','fit','validation')};hist={r:{} for r in rows}
            for index,(value,label) in enumerate(zip(x,y)):
                record=db.execute('SELECT role FROM content WHERE digest=?',(content_hash(value),)).fetchone()
                role=record[0] if record else None
                if role:
                    rows[role].append(index);key=str(int(label));hist[role][key]=hist[role].get(key,0)+1
            index_file=out/'indices'/f'client_{cid}.npz'
            np.savez_compressed(index_file,**{r:np.asarray(v,dtype=np.int64) for r,v in rows.items()})
            entry.update(index_sha256=file_sha256(index_file),role_rows={r:len(v) for r,v in rows.items()},role_class_counts=hist)
            for label in labels:
                task=task_of[label]
                participates=(cid in list(map(int,active.get(str(task),active.get(task,[]))))) if active else (str(cid) in joins and int(joins[str(cid)])<=task)
                if participates and hist['base'].get(str(label),0)>0:
                    eligible[label].append(cid)
                    for role in eligible_holdout:
                        if hist[role].get(str(label),0)>0:eligible_holdout[role][label].append(cid)
            print(f'Role indices client={cid}: {entry["role_rows"]}',flush=True)
        missing=[label for label,clients in eligible.items() if not clients]
        missing_base=missing
        if missing:print(f'No scheduled BASE expert supports classes {missing}; report-only, continuing.',flush=True)
        missing_holdout={}
        for role,coverage in eligible_holdout.items():
            missing=[label for label,clients in coverage.items() if not clients]
            missing_holdout[role]=missing
            if missing:print(f'No BASE-supported origin supplies {role} for classes {missing}; report-only, continuing.',flush=True)
        manifest.update(completed=True,eligible_base_experts={str(c):v for c,v in eligible.items()},
            eligible_holdout_origins={r:{str(c):v for c,v in coverage.items()} for r,coverage in eligible_holdout.items()},
            class_coverage_policy='report_only',missing_eligible_base_classes=missing_base,
            missing_eligible_holdout_classes=missing_holdout,
            conflicting_content_excluded=db.execute('SELECT COUNT(*) FROM content WHERE conflict=1').fetchone()[0],
            unique_content_by_role={r:db.execute('SELECT COUNT(*) FROM content WHERE role=?',(r,)).fetchone()[0] for r in ('base','calibration','fit','validation')},
            reachable_candidate_prediction_coverage='requires post-training validation; not inferable from input coverage')
        write_json(out/'role_manifest.json',manifest)
        write_json(out/'role_lock.json',dict(manifest_sha256=file_sha256(out/'role_manifest.json'),locked_before_backbone_training=True))
        print('Clean roles locked; class coverage recorded with report-only policy.',flush=True)
    finally:db.close()
    # The locked index views and original-shard checksums suffice for use and
    # reproducibility. Keep the temporary content database only on failure.
    (out/'content_roles.sqlite').unlink()
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',required=True,type=Path);parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--seed',type=int,default=20261006);args=parser.parse_args()
    prepare(args.data_dir,args.out,args.seed)
