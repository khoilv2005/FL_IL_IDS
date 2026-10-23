"""Publish compact receipts for the bounded native shadow development run."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(out):
    root=Path(__file__).resolve().parents[1]
    read=lambda name:json.loads((out/name).read_text(encoding='utf-8'))
    report=read('report.json'); timing=read('end_task_control.json'); probe=read('no_peer_probe.json')
    if not all(x['completed'] for x in (report,timing,probe)):raise ValueError('Incomplete engineering receipts')
    if len(report['rounds'])!=20 or report['steps']!=880:raise ValueError('Bounded schedule incomplete')
    manifest=json.loads((root/'candle_v4_runtime/v1_clone_manifest.json').read_text())
    changed=[]
    for entry in manifest['files']:
        path=root/'candle_v4_runtime'/entry['path']
        if sha(path)!=entry['sha256']:changed.append(entry['path'])
    expected=['fed_learning/data/denice_clean_roles.py','fed_learning/training/decentralized_denice_il.py']
    if sorted(changed)!=sorted(expected):raise ValueError(f'Unexpected V1 clone mutations: {changed}')
    files=['candle_v4_runtime/'+p for p in expected]+[
        'candle_v4_runtime/fed_learning/training/candle_v4_shadow.py',
        'tools/run_candle_v4_native_shadow.py','tools/summarize_candle_v4_native_shadow.py']
    receipt=dict(experiment='CANDLE V4 bounded native DRAFT integration',engineering_pass=True,
        quality_status='Receiver false activation reduction reproduced; no endpoint accuracy advantage over donor-only or matched end-task learning',
        protocol=read('protocol_lock.json'),integration=report,timing_control=timing,no_peer_probe=probe,
        endpoint_unique=read('endpoint_unique.json'),preparation_access=read('preparation_access.json'),
        checkpoint_boundary='T2/R19 after aggregation/router refresh and shadow hook, before task finalization',
        runtime_extension=dict(original_V1_files=manifest['source_files'],original_files_unchanged=365,
            changed_private_clone_files=changed,new_module='fed_learning/training/candle_v4_shadow.py',
            root_V1_modified=False,original_clone_manifest_preserved=True),
        source_sha256={name:sha(root/name) for name in files},
        local_receipt_sha256={name:sha(out/name) for name in (
            'report.json','end_task_control.json','no_peer_probe.json','endpoint_unique.json','protocol_lock.json')},
        limitations=['Fixed development pair selected in earlier work, not blind discovery',
            'Native graph rebuilt on two existing clients, not full 100-client graph',
            'Preparation decodes original NPZ shards to select current BASE/FIT; runtime reads only staged T2 files',
            'Current-task evidence only; historical retention and multi-expert safety unverified',
            'FIT reused for diagnostics and threshold control, not independent CAL acceptance',
            'Traffic is an explicit dense application serialization model, not measured network traffic',
            'Three trial branches share current-base verification; combined bytes are not a production single-branch benchmark',
            'No CAL, TEST, ACTIVE or full-training run'])
    target=root/'artifacts/candle_v4_native_shadow_20261023.json'
    target.write_bytes((json.dumps(receipt,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8'))
    print(target)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)
