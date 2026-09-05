"""Summarize matched multi-client transitions; no training or tuning."""
import argparse
import json
from pathlib import Path
from statistics import mean


def weighted_metric(rows, policy, task, subset, metric='accuracy'):
    values=[r['metrics'][policy][f'{task}_{subset}'] for r in rows]
    return sum(v['n']*v[metric] for v in values)/sum(v['n'] for v in values)


def summarize(d):
    result={'seed':d['seed'],'mode':d['mode'],'matrices':{},'operation_effects':[],
            'peer_client_events':sum(g['peer_aggregated_client_count'] for g in d['groups']),
            'round_count':len(d['groups']),
            'mean_peer_alpha':mean(g['peer_alpha_sum_stats']['mean'] for g in d['groups']),
            'policies':{p:sum(g['effective_policy']==p for g in d['groups']) for p in {g['effective_policy'] for g in d['groups']}},
            'reload_max_logit_error':max(r.get('reload_max_logit_error',0) for r in d['records'])}
    policies=list(d['records'][-1]['metrics'])
    for policy in policies:
        result['matrices'][policy]={}
        for subset in ['all','supported']:
            result['matrices'][policy][subset]=[
                [weighted_metric([r for r in d['records'] if r['stage']=='after_aging' and r['task']==t],policy,k,subset)
                 for k in range(t+1)] for t in range(3)]
    before={}
    for row in d['records']:
        if row['stage'].startswith('before_'):
            before[(row['task'],row['client'],row['stage'][7:])]=row
        if row['stage'].startswith('after_'):
            operation=row['stage'][6:]
            previous=before[(row['task'],row['client'],operation)]
            result['operation_effects'].append({
                'task':row['task'],'client':row['client'],'operation':operation,'round':row['round'],
                'transition':row.get('transition'),
                'task0_delta_pp':{p:100*(row['metrics'][p]['0_supported']['accuracy']-previous['metrics'][p]['0_supported']['accuracy']) for p in policies},
                'current_task_delta_pp':{p:100*(row['metrics'][p][f"{row['task']}_supported"]['accuracy']-previous['metrics'][p][f"{row['task']}_supported"]['accuracy']) for p in policies}})
    result['transition_extrema']={operation:{
        'max_protected_row_storage_change':max((r['transition']['protected_row_storage_max_change'] for r in result['operation_effects'] if r['operation']==operation and r['transition']),default=None),
        'max_mature_gru_feature_change':max((r['transition']['mature_gru_feature_max_change'] or 0 for r in result['operation_effects'] if r['operation']==operation and r['transition']),default=None),
        'mean_current_oracle_delta_pp':mean(r['current_task_delta_pp']['oracle_hard'] for r in result['operation_effects'] if r['operation']==operation),
        'mean_old_oracle_delta_pp':mean(r['task0_delta_pp']['oracle_hard'] for r in result['operation_effects'] if r['operation']==operation and r['task']>0),
    } for operation in ['local','aggregation','aging']}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    runs={}
    panel=None
    for path in sorted(args.root.glob('*/transitions.json')):
        d=json.loads(path.read_text())
        if not d['completed']:raise ValueError(f'Incomplete run: {path}')
        if panel is None:panel=d['panel_manifest']
        assert panel==d['panel_manifest'],'Different sampled panels'
        runs[path.parent.name]=summarize(d)
    paired={}
    for seed in sorted({r['seed'] for r in runs.values()}):
        if f'{seed}_peer' not in runs or f'{seed}_self_only' not in runs:continue
        p,s=runs[f'{seed}_peer'],runs[f'{seed}_self_only']
        paired[seed]={policy:{subset:[100*(a-b) for a,b in zip(p['matrices'][policy][subset][-1],s['matrices'][policy][subset][-1])]
                             for subset in ['all','supported']} for policy in p['matrices']}
    out={'scope':'Sample-count-weighted accuracy across selected client/task panels; paired modes, fixed data.',
         'runs':runs,'paired_peer_minus_self_final_pp':paired}
    cf_path=args.root/'42_peer_counterfactual/transitions.json'
    if cf_path.exists():
        cf=json.loads(cf_path.read_text())
        rows=[r for r in cf['records'] if r.get('counterfactuals')]
        out['counterfactuals']={'event_count':len(rows),'scope':'Equal-weight mean over 30 client/round events at tasks 1–2, NOT final-model improvement.',
            'effects':{variant:{policy:{
                'mean_old_task0_delta_pp':mean(100*(r['counterfactuals'][variant][policy]['0_supported']['accuracy']-r['metrics'][policy]['0_supported']['accuracy']) for r in rows),
                'mean_current_task_delta_pp':mean(100*(r['counterfactuals'][variant][policy][str(r['task'])+'_supported']['accuracy']-r['metrics'][policy][str(r['task'])+'_supported']['accuracy']) for r in rows),
            } for policy in ['oracle_hard','pred_hard']} for variant in rows[0]['counterfactuals']}}
    (args.root/'analysis.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({'paired':paired,'runs':{k:{'peer_events':v['peer_client_events'],
        'peer_alpha':v['mean_peer_alpha'],'final_E3':v['matrices']['oracle_hard']['supported'][-1],
        'final_E4':v['matrices']['pred_hard']['supported'][-1],'extrema':v['transition_extrema']} for k,v in runs.items()}},indent=2))


if __name__=='__main__':main()
