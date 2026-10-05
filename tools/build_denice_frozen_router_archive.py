"""Package existing fitted donor routers verbatim; never fit or regenerate them."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import warnings
import zipfile
import torch


def build(routers,reference,output):
    with zipfile.ZipFile(routers) as source,zipfile.ZipFile(reference) as ref:
        protocol=json.loads(source.read('protocol.json'))
        gate=json.loads(source.read('integration_gate.json'))
        manifest=json.loads(ref.read('profile_manifest.json'))
        if not gate['passed'] or gate['prediction_mismatches'] or gate['route_mismatches']:
            raise ValueError('Exact completed multiclass integration artifact required')
        if not protocol['training_commit'].startswith('03b9b53'):
            raise ValueError('Wrong checkpoint')
        entries={};payloads={}
        for cid in protocol['client_ids']:
            name=f'router_states/client_{cid}_context.pt';payload=source.read(name)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                state=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
            if state['router_mode']!='multiclass_balanced' or not state['multiclass_episodes']:
                raise ValueError('Expected a fitted multiclass_balanced router')
            if len(state['multiclass_episodes'])>1 and state['multiclass_router'] is None:
                raise ValueError('Missing learned router')
            if any(len(v) for v in state.get('reference_input_memory',{}).values()):
                raise ValueError('Archive must not publish raw reference samples')
            payloads[name]=payload
            entries[str(cid)]=dict(path=name,sha256=hashlib.sha256(payload).hexdigest(),
                encoder_hash=manifest[str(cid)]['encoder_hash'])
        metadata=dict(training_commit=protocol['training_commit'],
            checkpoint_file_sha256=protocol['checkpoint_file_sha256'],client_ids=protocol['client_ids'],
            task_classes=protocol['task_classes'],routers=entries,raw_reference_rows=0,
            fitted_router_bytes_copied_verbatim=True,fit_calls=0,
            source_integration_commit=protocol['evaluation_commit'])
        output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as target:
            for name,payload in sorted(payloads.items()):target.writestr(name,payload)
            target.writestr('manifest.json',json.dumps(metadata,indent=2)+'\n')
        print(output,output.stat().st_size,'bytes')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--routers',required=True);parser.add_argument('--reference',required=True)
    parser.add_argument('--output',default='artifacts/denice_multiclass_routers_03b9b53.zip')
    args=parser.parse_args();build(args.routers,args.reference,args.output)
