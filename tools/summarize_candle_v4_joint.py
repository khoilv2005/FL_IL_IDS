"""Publish compact joint-pilot metrics; no fitting, calibration or winner selection."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from pilot_candle_v4_standalone import write
import numpy as np
import sklearn
import torch

ROOT=Path(__file__).resolve().parents[1]


def safe_observed(row):
    donor=row['donor']; pooled=row['pooled_unique']
    classes=[v for c,v in pooled['by_class'].items() if int(c)!=13]
    owner_classes=[v for owner in ('donor','receiver') for c,v in row[owner]['by_class'].items() if int(c)!=13]
    return (donor['recall']>=.95 and pooled['false_activation_rate']<=.001
            and all(v['activated']/v['n']<=.001 for v in classes)
            and all(v['activated']/v['n']<=.001 for v in owner_classes)
            and pooled['negative_break']==0 and pooled['rescue']>pooled['break_count'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',type=Path,required=True)
    parser.add_argument('--coverage',type=Path,required=True)
    args=parser.parse_args()
    report=json.loads((args.pilot/'report.json').read_text()); coverage=json.loads(args.coverage.read_text())
    if not report['completed'] or report['CAL_opened'] or report['TEST_opened'] or report['ACTIVE']:
        raise ValueError('Unexpected pilot state')
    branches={}
    for name,row in report['branches'].items():
        if row['optimizer_steps']!=880 or row['examples_processed']!=225280: raise ValueError('Work budget mismatch')
        first={condition:next((r['round'] for r in row['history'] if r['metrics'] and safe_observed(r['metrics'][condition])),None)
               for condition in ('fixed_0p5','donor_FIT_recall95')}
        curve=[]
        for r in row['history']:
            item={k:v for k,v in r.items() if k!='metrics'}
            if r['metrics']:
                item['observed_FIT']={c:dict(donor_recall=m['donor']['recall'],donor_FP=m['donor']['false_positive'],
                    receiver_FP=m['receiver']['false_positive'],receiver_class12_FP=m['receiver']['by_class']['12']['activated'],
                    pooled_rescue=m['pooled_unique']['rescue'],pooled_break=m['pooled_unique']['break_count'],
                    pooled_changed_wrong=m['pooled_unique']['changed_wrong'],
                    donor_positive_score_margin=m['donor']['positive_score_margin']) for c,m in r['metrics'].items()}
            curve.append(item)
        branches[name]={k:v for k,v in row.items() if k not in ('history','final')}
        branches[name].update(final_metrics=row['final']['metrics'],round_curve=curve,
            first_round_meeting_observed_FIT_criteria=first,
            first_round_scope='Descriptive FIT reuse, frozen legacy base; not activation, independent evidence or native early-utility validation')
    if len({r['initial_sha256'] for r in branches.values()})!=1: raise ValueError('Initialization mismatch')
    if len({r['positive_sample_trace_sha256'] for r in branches.values()})!=1: raise ValueError('Positive sampling mismatch')
    if branches['joint_class_balanced']['BASE_sample_trace_sha256']!=branches['joint_unbalanced']['BASE_sample_trace_sha256']:
        raise ValueError('Balancing ablation sampling differs')
    compact={k:v for k,v in report.items() if k!='branches'}
    compact['branches']=branches
    compact['publication_provenance']=dict(original_report_sha256=hashlib.sha256((args.pilot/'report.json').read_bytes()).hexdigest(),
        pilot_script_sha256_lf=hashlib.sha256((ROOT/'tools/pilot_candle_v4_joint.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        core_v1_clone_manifest_sha256=hashlib.sha256((ROOT/'candle_v4_runtime/v1_clone_manifest.json').read_bytes()).hexdigest(),
        python=sys.version,torch=str(torch.__version__),numpy=np.__version__,sklearn=sklearn.__version__)
    write(ROOT/'artifacts/candle_v4_joint_pilot_20261023.json',compact)
    coverage['publication_provenance']=dict(original_report_sha256=hashlib.sha256(args.coverage.read_bytes()).hexdigest(),
        role_manifest_sha256=report['protocol']['role_manifest_sha256'],
        audit_script_sha256_lf=hashlib.sha256((ROOT/'tools/audit_candle_v4_round_coverage.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest())
    write(ROOT/'artifacts/candle_v4_round_coverage_20261023.json',coverage)
    print('PUBLISHED COMPACT PILOT',len(branches),'branches')
    for name,r in branches.items():
        print(name,'first observed FIT criteria',r['first_round_meeting_observed_FIT_criteria'],'bytes',r['wire']['application_wire_bytes'])


if __name__=='__main__': main()
