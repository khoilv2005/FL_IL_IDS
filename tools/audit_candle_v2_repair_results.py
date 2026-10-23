"""Read-only audit of paired full TEST outputs from the CANDLE V2 router repair."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def metrics(cm):
    support=cm.sum(1);predicted=cm.sum(0);correct=np.diag(cm)
    f1=np.divide(2*correct,support+predicted,out=np.zeros(34),where=support+predicted>0)
    recall=np.divide(correct,support,out=np.zeros(34),where=support>0)
    return dict(rows=int(cm.sum()),correct=int(correct.sum()),accuracy=float(correct.sum()/cm.sum()),
        macro_f1_34=float(f1.mean()),per_class_recall=recall.tolist(),class_counts=support.tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--original',type=Path,required=True)
    p.add_argument('--v1',type=Path,default=ROOT/'audit_denice/appliance_legacy_results11/inputs/legacy_self_task_5/completion.json')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    hashes={}
    with zipfile.ZipFile(a.archive) as archive:
        matches=[n for n in archive.namelist() if n.endswith('/repair_comparison.json')]
        if len(matches)!=1:raise ValueError('Expected exactly one repair run')
        prefix=matches[0][:-len('repair_comparison.json')]
        def read(name):
            raw=archive.read(prefix+name);hashes[name]=hashlib.sha256(raw).hexdigest()
            (a.out/name).write_bytes(raw)
            return json.loads(raw)
        completion=read('completion.json');comparison=read('repair_comparison.json')
        lock=read('pipeline_lock.json');clients=read('per_client.json')
        cm=np.asarray(read('confusion_matrix.json'),dtype=np.int64)
        base=np.asarray(read('paired_taskfree_confusion_matrix.json'),dtype=np.int64)
        if cm.shape!=(34,34) or base.shape!=(34,34) or (cm<0).any() or (base<0).any():
            raise ValueError('Invalid confusion matrices')
        nested_name=prefix.rstrip('/')+'.zip'
        nested_matches=None
        if nested_name in archive.namelist():
            with zipfile.ZipFile(io.BytesIO(archive.read(nested_name))) as nested:
                nested_matches=all(hashlib.sha256(nested.read(n)).hexdigest()==h for n,h in hashes.items())
                if not nested_matches:raise ValueError('Nested output ZIP differs from extracted results')
    with zipfile.ZipFile(a.original) as original:
        names=[n for n in original.namelist() if n.endswith('/candle_v2_completion.json')]
        if len(names)!=1:raise ValueError('Original CANDLE run ambiguous')
        previous=json.loads(original.read(names[0]));oldprefix=names[0][:-len('candle_v2_completion.json')]
        original_cm=np.asarray(json.loads(original.read(oldprefix+'candle_v2_self_task_5/confusion_matrix.json')))
    v1=json.loads(a.v1.read_text(encoding='utf-8'))
    if not completion['completed'] or not comparison['completed'] or completion['evaluation_role']!='full_TEST':
        raise ValueError('Full evaluation not completed')
    if lock['oracle'] or lock['donor_inference'] or lock['inference_policy']!='binary_cosine_registered_v1':
        raise ValueError('Unexpected inference protocol')
    m,b=metrics(cm),metrics(base)
    for key in ('accuracy','macro_f1_34'):
        if not np.isclose(m[key],completion['metrics'][key]) or not np.isclose(b[key],completion['paired_taskfree_control'][key]):
            raise ValueError('Metrics disagree with saved confusion')
    n=m['rows']
    if n!=b['rows'] or n!=completion['test_rows'] or n!=completion['evaluated_rows'] or not np.array_equal(cm.sum(1),base.sum(1)):
        raise ValueError('Sample accounting mismatch')
    if not np.array_equal(base,original_cm) or completion['partition']!=previous['partition']:
        raise ValueError('Original baseline/partition mismatch')
    if completion['source']!=previous['source'] or lock['source']!=completion['source']:
        raise ValueError('Checkpoint source identity changed')
    partition=completion['partition']
    if not partition['all_rows_assigned_once'] or partition['assigned_row_count']!=n:
        raise ValueError('Invalid disjoint partition receipt')
    if m['correct']-b['correct']!=completion['rescue']-completion['breaks']:
        raise ValueError('Rescue/break accounting mismatch')
    if sum(c['samples'] for c in clients)!=n or sum(round(c['accuracy']*c['samples']) for c in clients)!=m['correct']:
        raise ValueError('Per-client accounting mismatch')
    if sum(c['rescue'] for c in clients)!=completion['rescue'] or sum(c['breaks'] for c in clients)!=completion['breaks']:
        raise ValueError('Per-client transitions mismatch')
    if sorted(c['client_id'] for c in clients)!=lock['receivers'] or len(set(c['client_id'] for c in clients))!=len(clients):
        raise ValueError('Receiver membership mismatch')
    if not np.isclose(sum(round(c['route_accuracy']*c['samples']) for c in clients)/n,completion['route_accuracy']):
        raise ValueError('Per-client route accounting mismatch')
    tasks=[]
    for t in range(6):
        classes=list(range(t*6,min((t+1)*6,34)))
        rows=int(cm[classes].sum())
        tasks.append(dict(task=t,rows=rows,baseline_accuracy=float(np.diag(base)[classes].sum()/rows),
            repaired_accuracy=float(np.diag(cm)[classes].sum()/rows)))
    historical={name:dict(accuracy=value['accuracy'],macro_f1_34=value['macro_f1_34'],
        repair_accuracy_delta_pp=100*(m['accuracy']-value['accuracy']),
        repair_macro_f1_delta_pp=100*(m['macro_f1_34']-value['macro_f1_34'])) for name,value in v1['metrics'].items()}
    result=dict(completed=True,archive=str(a.archive),archive_bytes=a.archive.stat().st_size,
        archive_sha256=hashlib.sha256(a.archive.read_bytes()).hexdigest(),source_member_sha256=hashes,
        baseline=b,repaired=m,route_accuracy=completion['route_accuracy'],
        rescue=completion['rescue'],breaks=completion['breaks'],net_correct=m['correct']-b['correct'],
        delta_accuracy_pp=100*(m['accuracy']-b['accuracy']),
        delta_macro_f1_pp=100*(m['macro_f1_34']-b['macro_f1_34']),
        per_task=tasks,pipeline_lock=lock,mean_client_macro_f1=completion['mean_client_macro_f1'],
        label_coverage=completion['label_coverage'],empty_scope_fallback_rows=completion['empty_scope_fallback_rows'],
        prediction_fraction_T5=dict(baseline=float(base[:,30:].sum()/n),repaired=float(cm[:,30:].sum()/n)),
        original_confusion_exact_match=True,original_partition_exact_match=True,nested_output_zip_matches=nested_matches,
        original_checkpoint_identity_matches=True,
        V1_partition_matches=v1['partition']==completion['partition'],V1_historical_comparison=historical,
        runtime_seconds=completion['runtime_seconds'],no_training=True,no_new_model_inference=True,
        limitation='Post-development repair evaluation, historical V1 comparator, one seed; no isolated prototype contribution or untouched confirmation')
    (a.out/'audit_summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('baseline','repaired','route_accuracy','rescue','breaks','net_correct','V1_historical_comparison')},indent=2))


if __name__=='__main__':main()
