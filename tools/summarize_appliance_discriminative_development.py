"""Archive one completed linear-guard audit, including locks and limitations."""
import argparse
import hashlib
import json
from pathlib import Path
from appliance.state import write_json


def run(root, out):
    load=lambda name:json.loads((root/name).read_text(encoding='utf-8'))
    completion=load('completion.json')
    if not completion['completed'] or not all(completion['controls'].values()):
        raise ValueError('Completed audit and passing integrity controls required')
    files=['appliance/discriminative_guard.py','tools/audit_appliance_discriminative_guard.py',
           'tools/summarize_appliance_discriminative_development.py']
    results=completion['results']
    negatives=[r for r in results if 'negative_validation' in r['pool']]
    artifact=dict(kind='APPLIANCE discriminative linear guard; development only',
        protocol=load('protocol_before_data.json'),guard=load('linear_guard_before_SELECTION.json'),
        threshold_lock=load('threshold_lock_before_HOLDOUT.json'),completion=completion,
        peer_negative_rows=sum(r['rows'] for r in negatives),
        original_route_false_activation=sum(r['original_route_false_activations'] for r in negatives),
        linear_guard_false_activation=sum(r['false_activations'] for r in negatives),
        peer_class13_false_activation=sum(r['per_class'].get('13',{}).get('false_activation',0) for r in negatives),
        all_evaluation_content_overlap=sum(r['exact_content_overlap_with_guard_training'] for r in results),
        source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        lock_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
            ['protocol_before_data.json','linear_guard_before_SELECTION.json','threshold_lock_before_HOLDOUT.json']},
        audit_source=str(root),installation_authorized=False,production_enabled=False,
        repetitions_are_instrumentation_reruns_not_independent_seeds=True,
        aborted_instrumentation_run_excluded='discriminative_guard65_03: empty content-hash slice; no completed evaluation',
        current_CAL_holdout_previously_observed=True,
        main_conclusion='Linear guard preserves current CAL recall but fails peer-negative protection; no native/full run')
    snapshot=root/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for f in files:(snapshot/Path(f).name).write_bytes(Path(f).read_bytes())
    write_json(out,artifact)
    print(f'Archived {out}')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.root,a.out)
