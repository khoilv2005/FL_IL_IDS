"""Metadata reachability and fixed-pair availability for every original native round."""
import argparse
import gc
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

from pilot_candle_v4_standalone import reachability,write
import torch


def run(checkpoint,manifest,classes):
    records=[]; union=set(); union_evidence=set(); all_gaps=set()
    with zipfile.ZipFile(checkpoint) as archive:
        index=json.loads(archive.read('checkpoint_archive_manifest.json'))
        if int(index['task_id'])!=2: raise ValueError('Task 2 archive required')
        for round_id in range(20):
            name=f'checkpoint_task_2_round_{round_id}.pt'; raw=archive.read(name)
            checksum=hashlib.sha256(raw).hexdigest()
            if checksum!=index['checksums'][name]: raise ValueError('Graph checkpoint checksum mismatch')
            packet=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
            graph=packet['cluster']; groups={}; weights={}
            if graph.get('similarity_threshold')!=.8: raise ValueError('Expected native xi=.8')
            for key,row in graph['alpha_debug'].items():
                owner=int(key); ids=list(map(int,row['group_ids'])); alpha=list(map(float,row['alphas']))
                if len(ids)!=len(alpha) or len(set(ids))!=len(ids) or any(not math.isfinite(v) or v<0 for v in alpha):
                    raise ValueError('Malformed graph')
                groups[owner]=[j for j,v in zip(ids,alpha) if j!=owner and v>0]
                weights[owner]={j:v for j,v in zip(ids,alpha) if j!=owner and v>0}
            coverage=reachability(manifest,classes,groups)
            for row in coverage['rows']:
                key=(row['receiver'],row['class_id']); all_gaps.add(key)
                if row['donors']: union.add(key)
                if row['evidence_count_eligible_donors']: union_evidence.add(key)
            record={k:v for k,v in coverage.items() if k not in ('rows','scope')}
            record.update(round=round_id,graph_payload_sha256=checksum,
                pair_30_from_16_available=16 in groups.get(30,[]),pair_alpha=weights.get(30,{}).get(16,0.))
            records.append(record)
            print('GRAPH',round_id,'reachable',record['reachable_gaps'],'/',record['metadata_gaps'],
                  'pair',record['pair_30_from_16_available'],flush=True)
            del packet,raw,graph; gc.collect()
    available=[r['round'] for r in records if r['pair_30_from_16_available']]
    return dict(completed=True,task=2,rounds=records,pair=dict(receiver=30,donor=16,target=13,
        available_rounds=available,first_available_round=min(available) if available else None,
        fraction=len(available)/20),union_reachable_gaps=len(union),union_evidence_gaps=len(union_evidence),
        union_metadata_gaps=len(all_gaps),
        scope='Metadata ownership gaps and positive native-alpha edges only; no functional competence or acceptance claim',
        CAL_examples_opened=False,TEST_opened=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--roles',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=False)
    raw=(args.roles/'role_manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=json.loads((args.roles/'role_lock.json').read_text())['manifest_sha256']:
        raise ValueError('Role manifest is not locked')
    manifest=json.loads(raw)
    result=run(args.checkpoint,manifest,list(range(12,18)))
    write(args.out/'report.json',result)
    print('PAIR AVAILABILITY',result['pair'],flush=True)


if __name__=='__main__': main()
