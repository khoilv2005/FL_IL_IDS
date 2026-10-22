"""Summarize saved diagnostic scores without further model or data access."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnostics', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--training-log', type=Path)
    p.add_argument('--plot', type=Path)
    a = p.parse_args()
    d = json.loads((a.diagnostics/'completion.json').read_text(encoding='utf-8'))
    rows = d['records']
    stages, retention = [], []
    for task in range(6):
        records = [r for r in rows if r['task']==task]
        if not records:
            continue
        n = sum(r['native']['rows'] for r in records)
        counts = dict(correct=0, coverage=0, cross_task_competition=0, within_task_error=0)
        for r in records:
            with np.load(a.diagnostics/f'T{task}_client_{r["client_id"]}_scores.npz') as f:
                y, pred, oracle = f['y'], f['pred'], f['oracle']
            covered = np.isin(y,r['available'])
            counts['correct'] += int((pred==y).sum())
            counts['coverage'] += int((~covered).sum())
            counts['cross_task_competition'] += int((covered&(pred!=y)&(oracle==y)).sum())
            counts['within_task_error'] += int((covered&(pred!=y)&(oracle!=y)).sum())
        if sum(counts.values()) != n:
            raise ValueError('Nonexclusive error decomposition')
        stages.append(dict(task=task, clients=len(records), prediction_rows=n,
            native_accuracy=sum(r['native']['correct'] for r in records)/n,
            task_oracle_available_accuracy=sum(r['oracle_available']['correct'] for r in records)/n,
            task_oracle_all_rows_accuracy=sum(r['oracle_all_task_rows']['correct'] for r in records)/n,
            raw_affine_accuracy=sum(r['raw_affine']['correct'] for r in records)/n,
            label_coverage=sum(r['covered_rows'] for r in records)/n,
            decomposition_counts=counts,
            cross_task_fraction_of_errors=counts['cross_task_competition']/(n-counts['correct']),
            old_margin_positive_fraction_mean=float(np.mean([r['old_input_new_minus_old']['positive_fraction'] for r in records if r['old_input_new_minus_old']])) if task else None))
    final = {r['client_id']:r for r in rows if r['task']==5}
    for birth_task in range(5):
        pairs = [r for r in rows if r['task']==birth_task and r['client_id'] in final]
        totals = dict(rows=0, birth_correct=0, final_native_correct=0, final_birth_mask_correct=0,
                      identical_predictions=0, row_magnitude_scaling_residual_max=0.)
        for r in pairs:
            cid = r['client_id']
            allowed = r['by_true_task'][birth_task]['available']
            with np.load(a.diagnostics/f'T{birth_task}_client_{cid}_scores.npz') as f:
                ix = f['y']//6==birth_task
                y = f['y'][ix]; before = f['cosine'][ix][:,allowed]
            with np.load(a.diagnostics/f'T5_client_{cid}_scores.npz') as f:
                ix = f['y']//6==birth_task
                after = f['cosine'][ix][:,allowed]; pred = f['pred'][ix]
            if not allowed:
                continue
            bp = np.asarray(allowed)[before.argmax(1)]
            ap = np.asarray(allowed)[after.argmax(1)]
            totals['rows'] += len(y)
            totals['birth_correct'] += int((bp==y).sum())
            totals['final_native_correct'] += int((pred==y).sum())
            totals['final_birth_mask_correct'] += int((ap==y).sum())
            totals['identical_predictions'] += int((ap==bp).sum())
            factor = (before*after).sum(1)/np.maximum((before*before).sum(1),1e-12)
            residual = np.abs(after-before*factor[:,None]).max()
            totals['row_magnitude_scaling_residual_max'] = max(totals['row_magnitude_scaling_residual_max'],float(residual))
        n = totals['rows']
        retention.append(dict(birth_task=birth_task, paired_clients=len(pairs), **totals,
            birth_oracle_accuracy=totals['birth_correct']/n,
            final_native_accuracy=totals['final_native_correct']/n,
            final_birth_mask_accuracy=totals['final_birth_mask_correct']/n,
            birth_vs_final_prediction_agreement=totals['identical_predictions']/n))
    competence = []
    if a.training_log:
        training = [json.loads(line) for line in a.training_log.read_text(encoding='utf-8').splitlines() if line]
        local = {(r['task'],r['client_id']):r['local_classes'] for r in training if r['round']==19}
        for task in range(6):
            counts = {k:[0,0] for k in ('owned','available_not_owned')}
            for r in rows:
                if r['task'] != task:
                    continue
                current = r['by_true_task'][task]
                metric = current['oracle_available']
                for c in current['available']:
                    key = 'owned' if c in local[task,r['client_id']] else 'available_not_owned'
                    counts[key][0] += metric['per_class_correct'][c]
                    counts[key][1] += metric['class_counts'][c]
            competence.append(dict(task=task, **{k:dict(correct=v[0],rows=v[1],recall=v[0]/v[1] if v[1] else None) for k,v in counts.items()}))
    result = dict(completed=True, source='Frozen checkpoint development panel, 64 unique rows/class',
        model_evaluations=len(rows), unique_panel_rows=d['panel_rows'], stages=stages,
        paired_birth_task_retention=retention, no_training=True, no_fitting=True, no_final_TEST=True,
        current_task_competence=competence,
        current_task_competence_caveat='Nonowned available does not isolate prototype gain from native aggregation or inherited knowledge; task oracle diagnostic only',
        structural=dict(zero_feature_prediction_rows=sum(r['zero_feature_rows'] for r in rows),
            zero_effective_available_rows=sum(len(r['zero_effective_available_rows']) for r in rows),
            T0_old_FC2_effective_max_change=max(r['T0_reference']['old_effective_row_max_abs_change'] for r in rows if 'T0_reference' in r),
            T0_old_FC2_connection_mask_changes=sum(r['T0_reference']['old_connection_mask_changed'] for r in rows if 'T0_reference' in r)),
        caveat='Balanced development panel and repeated predictions across models; not final TEST accuracy or deployment oracle')
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    if a.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        with plt.rc_context({'font.size':10,'svg.hashsalt':'candle-v2-checkpoint-debug-v1'}):
            fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
            axes[0].plot([s['task'] for s in stages],[100*s['native_accuracy'] for s in stages],'o-',label='Task-free inference')
            axes[0].plot([s['task'] for s in stages],[100*s['task_oracle_available_accuracy'] for s in stages],'o-',label='Oracle task mask (diagnostic)')
            axes[0].set(xlabel='Terminal checkpoint task',ylabel='Balanced development accuracy (%)',ylim=(0,100),xticks=range(6),title='Same panel, cumulative seen classes')
            axes[0].legend();axes[0].grid(alpha=.2)
            ix=np.arange(len(retention));width=.25
            for offset,key,label in [(-width,'birth_oracle_accuracy','At task completion'),(0,'final_birth_mask_accuracy','T5, same task mask'),(width,'final_native_accuracy','T5, task-free')]:
                axes[1].bar(ix+offset,[100*r[key] for r in retention],width=width,label=label)
            axes[1].set(xticks=ix,xticklabels=[f'T{r["birth_task"]}' for r in retention],ylim=(0,100),ylabel='Paired panel accuracy (%)',title='Preserved ranking within each old task')
            axes[1].legend(fontsize=8);axes[1].grid(axis='y',alpha=.2)
            fig.suptitle('CANDLE V2 checkpoint diagnostics — development only, not full TEST')
            fig.savefig(a.plot,dpi=180)
            fig.savefig(a.plot.with_suffix('.svg'),metadata={'Date':None})
            svg = a.plot.with_suffix('.svg')
            svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
            plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
