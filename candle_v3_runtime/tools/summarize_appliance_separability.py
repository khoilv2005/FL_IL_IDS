"""Archive completed, locked feature probes without raw development inputs."""
import argparse
import json
import platform
from pathlib import Path
import numpy as np
import torch
from appliance.state import write_json
from fed_learning.data.denice_clean_roles import file_sha256


def run(root,out):
    load=lambda name:json.loads((root/name).read_text(encoding='utf-8'))
    result=load('completion.json')
    checks=result['controls']
    if not result['completed'] or checks['historical_CAL_runtime_reads']!=0 or not all(v for k,v in checks.items() if k!='historical_CAL_runtime_reads'):
        raise ValueError('Incomplete audit or integrity failure')
    for f,sha in result['source_sha256'].items():
        if file_sha256(Path(f))!=sha:raise ValueError('Executed audit code changed')
    files=['protocol_before_data.json','fit_support_lock.json','probe_models_before_selection.json',
           'development_panels_before_scores.json','configuration_lock_before_evaluation.json','completion.json']
    panels=load('development_panels_before_scores.json')
    heads=result['donor_compatibility']['evaluation']['all_pool_details']
    hard=[v for v in heads if v['class_id']==13]
    artifact=dict(kind='APPLIANCE frozen feature separability development audit',
        protocol=load('protocol_before_data.json'),completion=result,
        configuration_lock=load('configuration_lock_before_evaluation.json'),
        panel_scope=[{k:v for k,v in p.items() if k not in ('row_ids','content_hashes')} for p in panels['panels']],
        panel_missing_classes=panels['missing_classes'],panel_exclusions=panels['exclusions'],
        prior_content_matches_excluded=sum(v['prior_content_excluded'] for v in panels['exclusions']),
        fit_content_matches_excluded=sum(v['training_content_excluded'] for v in panels['exclusions']),
        donor_native_class13=dict(rows=sum(v['rows'] for v in hard),
            predicted20=sum(v['donor_predicted_class_histogram'].get('20',0) for v in hard),
            routed_task3=sum(v['donor_task_histogram'].get('3',0) for v in hard),
            correct=sum(v['donor_native_correct'] for v in hard)),
        file_sha256={name:file_sha256(root/name) for name in files},
        runtime=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,device='cpu'),
        audit_root=str(root),instrumentation_reruns_not_independent_seeds=True,
        no_algorithm_or_threshold_policy_changed_between_reruns=True,
        decision='No deployment acceptance for 65 <- 88/class20 under current mechanism',
        scope_of_conclusion='Fixed linear probes and this donor/pair only; not universal nonlinear inseparability',
        production_runner_changed=False,native_smoke_authorized=False)
    snapshot=root/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for f in ['tools/audit_appliance_feature_separability.py','tools/summarize_appliance_separability.py']:
        (snapshot/Path(f).name).write_bytes(Path(f).read_bytes())
    write_json(out,artifact)
    print(f'Archived {out}')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.root,a.out)
