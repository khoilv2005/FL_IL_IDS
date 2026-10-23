"""Publish the registered coverage experiment; summaries only, no sample reads."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform

from pilot_candle_v4_standalone import REPO,write,file_sha256
import torch
import numpy as np
import sklearn


def main(out):
    report=json.loads((out/'report.json').read_text());training=report['training']
    checks=deepcopy(training['checks'])
    primary='primary_FIT_select95';classes=sorted(map(int,training['negative_plans']))
    untouched=[c for c in classes if not training['negative_plans'][str(c)]]
    checks['uncovered_only_changes_affected_experts']=all(training['final_hashes']['coverage_aware'][str(c)]==
        training['final_hashes']['current'][str(c)] for c in untouched)
    checks['ablation_metrics_exact']=report['FIT_metrics']['current']==report['FIT_metrics']['no_extra_ablation']
    checks['native_remote_reference_exact']=report['native_remote_reference_exact']
    checks['all_coverage_steps_have_128_positive_128_negative']=all((r['positives'],r['negatives'])==(128,128)
        for rows in training['coverage_traces'].values() for r in rows)
    round_summaries={}
    for c,rows in training['coverage_traces'].items():
        round_summaries[c]=[dict(round=round_id,steps=sum(r['round']==round_id for r in rows),
            extra_samples=sum(r['extra_negative_samples'] for r in rows if r['round']==round_id),
            nonzero_extra_gradient_steps=sum(r['extra_negative_gradient_norm']>0 for r in rows if r['round']==round_id),
            selected_sources=next(r['selected'] for r in rows if r['round']==round_id)) for round_id in range(8)]
    compact=deepcopy(report);compact['training'].pop('coverage_traces')
    compact['training']['coverage_round_summaries']=round_summaries
    current=report['FIT_metrics']['current'][primary]['pooled_unique']
    aware=report['FIT_metrics']['coverage_aware'][primary]['pooled_unique']
    comparison=dict(assessment_unique_rows=current['rows'],accuracy_before=current['accuracy'],accuracy_after=aware['accuracy'],
        additional_correct=aware['correct']-current['correct'],break_before=current['break_count'],break_after=aware['break_count'],
        target_metrics={str(c):dict(recall_before=current['targets'][str(c)]['recall'],recall_after=aware['targets'][str(c)]['recall'],
            false_activations_before=current['targets'][str(c)]['false_activations'],
            false_activations_after=aware['targets'][str(c)]['false_activations']) for c in classes})
    hypothesis=dict(FP_reduced=any(aware['targets'][str(c)]['false_activations']<current['targets'][str(c)]['false_activations'] for c in classes),
        pooled_target_recalls_not_lower=all(aware['targets'][str(c)]['recall']>=current['targets'][str(c)]['recall'] for c in classes),
        required_donor_recalls_not_lower=all(report['FIT_metrics']['coverage_aware'][primary]['owners'][str(e['donor'])]['targets'][str(e['target'])]['recall']>=
            report['FIT_metrics']['current'][primary]['owners'][str(e['donor'])]['targets'][str(e['target'])]['recall'] for e in report['protocol']['experts']),
        candidate_accuracy_not_lower=aware['correct']>=current['correct'],break_not_increased=aware['break_count']<=current['break_count'])
    if not all(checks.values()):raise AssertionError(checks)
    result=dict(experiment='CANDLE V4 coverage-aware negatives controlled development',
        checks=checks,primary_comparison=comparison,development_hypothesis_criteria=hypothesis,
        development_hypothesis_supported=all(hypothesis.values()),experiment_receipt=compact,
        local_report_sha256=file_sha256(out/'report.json'),protocol_lock_sha256=file_sha256(out/'protocol_lock.json'),
        summary_source_sha256=file_sha256(Path(__file__)),runtime=dict(python=platform.python_version(),
            torch=str(torch.__version__),numpy=np.__version__,sklearn=sklearn.__version__),
        decision='Development evidence only; no CAL/ACTIVE/full authorization',
        interpretation=['Ablation exactly reproduces original model weights and assessment predictions',
            'Only expert with uncovered negative classes changes weights; peer2 supplies class13 negatives by a generic metadata rule',
            'Primary +101 unique correct; fixed0.5 secondary +6 net with9 gained/3 lost owner events',
            'Benefits combine newly covered negative class and redistribution of fixed class-balanced negative budget',
            'FIT reused development lineage; partition is not independent confirmation',
            'The three prior rejected CAL samples were not opened, trained on or used for threshold selection; no claim they are fixed'])
    target=REPO/'artifacts/candle_v4_coverage_development_20261023.json'
    write(target,result)
    print(json.dumps(dict(checks=checks,primary_comparison=comparison,hypothesis=hypothesis),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out.resolve())
