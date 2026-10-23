"""Check cached audit counts and selector equivalence using synthetic inputs."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from appliance.distributed_calibration import (CalibrationSession, GuardGrid,
    LocalCalibrationEndpoint, select_distributed_guard)
from appliance.state import digest, write_json
from tools.audit_appliance_actual_transfer import choose_guard
from tools.audit_appliance_donor_eligibility import summarize


def controls():
    results = []
    for seed in range(12):
        rng = np.random.default_rng(seed)
        keys = [(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)]
        slices = {k: slice(n*32, (n+1)*32) for n, k in enumerate(keys)}
        y = np.concatenate([np.full(32, c) for _, c in keys])
        sig = dict(signature_score=np.round(rng.uniform(-1, 1, len(y)), 1),
            margin=np.round(rng.normal(size=len(y)), 1),
            signature_valid=rng.random(len(y)) > .1,
            local_pred=rng.integers(0, 3, len(y)), local_confidence=np.zeros(len(y)))
        floor = -.4
        session = CalibrationSession(0, 1, 1, 0, 0, (0, 1, 2), 'synthetic',
            'synthetic', 'synthetic', f'control-{seed}')
        endpoints = {}
        for owner in (0, 1):
            ix = np.concatenate([np.arange(slices[k].start, slices[k].stop)
                                 for k in keys if k[0] == owner])
            local = dict(y=y[ix], row_id=np.arange(len(ix)), signals={k:v[ix] for k,v in sig.items()})
            empty = dict(y=np.zeros(0, np.int64), row_id=np.zeros(0, np.int64),
                         signals={k:v[:0] for k,v in sig.items()})
            endpoints[owner] = LocalCalibrationEndpoint(owner, session,
                dict(fit=empty, selection=local, holdout=empty))
        proposed = GuardGrid.from_quantiles(session, [endpoints[o].quantiles() for o in (0,1)], floor)
        grid = GuardGrid(session, proposed.taus, proposed.gammas, [1.])
        expected = select_distributed_guard(session, grid,
            endpoints[0].count_packet(grid), endpoints[1].count_packet(grid))
        actual = choose_guard(sig, keys, slices, 0, 1, 1, floor)
        fields = ('tau', 'gamma', 'beta')
        match = all(actual[k] == expected[k] for k in fields)
        if not match or actual['positive_hits'] != expected['selection_target_activations']:
            raise AssertionError(f'Production selector mismatch at synthetic seed {seed}')
        results.append(dict(seed=seed, thresholds_match=match,
            positive_hits_match=actual['positive_hits']==expected['selection_target_activations']))
    for name, pos, neg, missing in (
        ('no_negatives', np.ones(64,bool), [], []),
        ('negative_under32',np.ones(64,bool),[(0,2,np.zeros(4,bool),np.zeros(4,bool))],
            [dict(owner=0,class_id=2,rows=4)]),
        ('positive_under32',np.ones(4,bool),[(0,2,np.zeros(64,bool),np.zeros(64,bool))],[])):
        value=summarize(pos,neg,missing)
        if value['status']!='insufficient_evidence':
            raise AssertionError(f'Missing evidence passed {name}')
        results.append(dict(control=name,missing_evidence_not_pass=True))
    return results


def verify(root):
    summary=json.loads((root/'completion.json').read_text())
    if not summary['completed']:raise ValueError('Audit incomplete')
    records=json.loads((root/'graph_results.json').read_text())
    scored=[v for v in records if 'evaluation' in v]
    if (len(scored)!=summary['actual_function_evaluated_pairs'] or
            len(scored)+summary['insufficient_signature_FIT']!=summary['maturity_candidates']):
        raise AssertionError('Funnel denominator mismatch')
    lock=json.loads((root/'selection_lock_before_evaluation.json').read_text())
    selected_state=copy.deepcopy(records)
    for v in selected_state:
        for name in ('evaluation','functional_eligible','current_CAL'):
            v.pop(name,None)
    if digest(selected_state)!=lock['records_sha256']:
        raise AssertionError('Post-selection state changed beyond evaluation/CAL fields')
    lookup={(v['receiver'],v['donor'],v['class_id']):v for v in lock['locks']}
    for v in scored:
        if lookup[v['receiver'],v['donor'],v['class_id']]['guard']!=v['guard_lock']:
            raise AssertionError('Evaluation changed selection guard')
        for role in ('selection','evaluation'):
            for variant in ('fixed_control','calibrated'):
                s=v[role][variant];p=s['per_owner_class']
                if (sum(x['rows'] for x in p)!=s['negative_rows'] or
                        sum(x['false_activation'] for x in p)!=s['false_activation'] or
                        sum(x['breaks'] for x in p)!=s['breaks'] or
                        any(x['breaks']>x['false_activation'] for x in p) or
                        not 0<=s['target_hits']<=s['positive_rows'] or
                        s['rescue']>s['target_hits']):
                    raise AssertionError('Metric accounting mismatch')
                if s['status']=='passed_observed_scope' and (s['positive_rows']<32 or
                    not s['negative_rows'] or s['missing_or_under32_pools']):
                    raise AssertionError('Missing evidence certified')
    return dict(metric_records_recomputed=len(scored)*4, guard_locks_unchanged=True,
        funnel_accounted=True, no_unknown_promoted_to_pass=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--run',type=Path)
    a=p.parse_args()
    result=dict(synthetic_controls=controls(), **(verify(a.run) if a.run else {}))
    write_json(a.out,result)
    print(json.dumps(result,indent=2))
