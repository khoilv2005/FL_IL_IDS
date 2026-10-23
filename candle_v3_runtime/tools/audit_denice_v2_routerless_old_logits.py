"""Inference-only old-mask diagnostic and complete old/new margin distribution.

Never fit a threshold, edit model weights, regenerate training, or read TEST.
T0 can be an existing checkpoint or the original archived baseline receipt.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from fed_learning.data.denice_clean_roles import file_sha256
from fed_learning.strategies.decentralized.denice_v2.contracts import decode,state_hash
from fed_learning.strategies.decentralized.denice_v2.native import load_native
from fed_learning.strategies.decentralized.denice_v2.state import restore
from tools.run_denice_v2_routerless_smoke import metrics,read_current


def masked_argmax(logits,classes):
    ids=np.asarray(classes,dtype=np.int64)
    if (logits.ndim!=2 or not len(ids) or len(set(ids.tolist()))!=len(ids)
        or (ids<0).any() or (ids>=logits.shape[1]).any() or not np.isfinite(logits[:,ids]).all()):
        raise ValueError('Invalid diagnostic logits/classes')
    return ids[logits[:,ids].argmax(1)]


def margin_summary(values):
    q=(0,.01,.05,.10,.25,.50,.75,.90,.95,.99,1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Invalid margin distribution')
    return dict(rows=len(values),mean=float(values.mean()),std_population=float(values.std()),
        quantiles={str(p):float(np.quantile(values,p)) for p in q},positive_fraction=float((values>0).mean()),
        zero_fraction=float((values==0).mean()))


def ranking_auc(negative,positive):
    # Probability a new sample's margin exceeds an old sample's margin.
    # No threshold selection; ties get half credit.
    differences=positive[:,None]-negative[None,:]
    return float(((differences>0)+.5*(differences==0)).mean())


def panel(prepared,task):
    chunks=[read_current(prepared,cid,'validation',task) for cid in (9,30)]
    x=torch.cat([p[0] for p in chunks])
    y=np.concatenate([p[1].numpy() for p in chunks])
    ids=[(cid,int(row)) for cid,p in zip((9,30),chunks) for row in p[2]]
    return x,y,ids


def load_checkpoint(path):
    return restore(decode(json.loads(path.read_text(encoding='utf-8'))))


def raw_logits(model,x,batch_size=64):
    with torch.no_grad():
        return np.concatenate([model(batch).cpu().numpy() for batch in x.split(batch_size)])


def write_ecdf_svg(path,old,new):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # All empirical points, full range plus a clearly labeled display zoom.
    # No smoothing, threshold sweep or classifier fitting.
    combined=np.concatenate([old,new])
    maximum=float(combined.max())
    with plt.rc_context({'font.size':10,'svg.hashsalt':'denice-routerless-old-logits-v1'}):
        fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True,layout='constrained')
        for axis in axes:
            for values,label,color in ((old,f'Old validation (n={len(old)})','#2563eb'),
                                       (new,f'New validation (n={len(new)})','#dc2626')):
                values=np.sort(values)
                x=np.concatenate([[0],values,[maximum*1.02]])
                y=np.concatenate([[0],np.arange(1,len(values)+1)/len(values),[1]])
                axis.step(x,y,where='post',label=label,color=color,lw=1.8)
            axis.axvline(0,color='gray',linestyle='--',lw=1)
            axis.grid(alpha=.25)
            axis.set_ylim(0,1.02)
            axis.set_xlabel('max(new logits) - max(old logits)')
        axes[0].set_xlim(-maximum*.015,maximum*1.02)
        axes[0].set_title('Complete observed margin range')
        axes[0].set_ylabel('Empirical cumulative fraction')
        axes[0].legend(loc='lower right')
        axes[1].set_xlim(-.05,float(np.quantile(combined,.95))*1.04)
        axes[1].set_title('Display zoom: pooled 95th percentile')
        fig.suptitle('T1 checkpoint: old/new margin distributions (diagnostic only)')
        fig.savefig(path,metadata={'Date':None})
        path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
        fig.savefig(path.with_suffix('.png'),dpi=170)
        plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',required=True)
    p.add_argument('--pilot-receipt',required=True)
    p.add_argument('--checkpoint0',help='Optional existing T0 checkpoint; never regenerated')
    p.add_argument('--checkpoint1',help='Default existing cache/routerless_v1_model_state.json')
    p.add_argument('--out',required=True)
    args=p.parse_args()
    cache,out=Path(args.cache),Path(args.out)
    csv_path=out.with_name(out.stem+'_rows.csv')
    svg_path=out.with_name(out.stem+'_margin_ecdf.svg')
    if any(path.exists() for path in (out,csv_path,svg_path,svg_path.with_suffix('.png'))):
        raise ValueError('Use fresh diagnostic output paths')
    pilot_path=Path(args.pilot_receipt)
    pilot=json.loads(pilot_path.read_text(encoding='utf-8'))
    prepared=(cache/'routerless_v1_current_roles',pilot['role_preparation_sha256'])
    path1=Path(args.checkpoint1) if args.checkpoint1 else cache/'routerless_v1_model_state.json'
    if file_sha256(path1)!=pilot['checkpoint']['sha256']:
        raise ValueError('T1 differs from original pilot checkpoint')
    state=load_checkpoint(path1)
    if state.task!=1 or state.router_state['router_mode']!='routerless_v1':
        raise ValueError('Expected frozen T1 routerless checkpoint')
    before=state.fingerprint()
    torch.set_num_threads(2)
    model=load_native(state).eval()
    old=tuple(pilot['tasks']['0']['registration']['seen_classes'])
    full=tuple(state.algorithm_state['seen_classes'])
    new=tuple(c for c in full if c not in old)
    if old!=(1,2,3,4,5) or not new:
        raise ValueError('Wrong original pilot class registry')
    x0,y0,ids0=panel(prepared,0);x1,y1,ids1=panel(prepared,1)
    z0,z1=raw_logits(model,x0),raw_logits(model,x1)
    pred_old=masked_argmax(z0,old);pred_full=masked_argmax(z0,full)
    m0=z0[:,new].max(1)-z0[:,old].max(1)
    m1=z1[:,new].max(1)-z1[:,old].max(1)
    old_only=metrics(y0,pred_old,6);cumulative_old=metrics(y0,pred_full,12)
    original0=pilot['tasks']['0']['development']
    original1=pilot['tasks']['1']['development']
    checks=dict(old_panel_rows_match=len(y0)==original0['rows']==374,
        old_panel_support_match=all(old_only['per_class'][c]['support']==original0['per_class'][str(c)]['support'] for c in range(6)),
        t1_full_confusion_matches_original=cumulative_old['confusion']==original1['old']['confusion'],
        old_margin_median_matches=bool(np.isclose(np.median(m0),pilot['cross_task_competition']['median_new_minus_old_logit_on_old'],rtol=1e-6,atol=1e-7)),
        new_margin_median_matches=bool(np.isclose(np.median(m1),pilot['cross_task_competition']['median_new_minus_old_logit_on_new'],rtol=1e-6,atol=1e-7)),
        state_unmodified=before==state.fingerprint(),
        model_unmodified=state_hash(state.model_state)==state_hash({k:v.detach().cpu() for k,v in model.state_dict().items()}))
    baseline=dict(accuracy=original0['accuracy'],correct=int(sum(original0['confusion'][i][i] for i in range(6))),
        source='archived_original_T0_receipt_not_reexecuted',checkpoint_available=False,
        per_class=original0['per_class'])
    if args.checkpoint0:
        path0=Path(args.checkpoint0)
        state0=load_checkpoint(path0)
        if state0.task!=0 or tuple(state0.algorithm_state['seen_classes'])!=old:
            raise ValueError('Wrong optional T0 checkpoint')
        measured0=metrics(y0,masked_argmax(raw_logits(load_native(state0).eval(),x0),old),6)
        if measured0['confusion']!=original0['confusion']:
            raise ValueError('T0 does not reproduce original pilot')
        baseline.update(source='existing_T0_checkpoint_inference',checkpoint_available=True,sha256=file_sha256(path0),measurement=measured0)
    out.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    for panel_name,truth,ids,z,margin in (('old',y0,ids0,z0,m0),('new',y1,ids1,z1,m1)):
        a,b=masked_argmax(z,old),masked_argmax(z,full)
        for index,(cid,rowid) in enumerate(ids):
            rows.append(dict(panel=panel_name,client_id=cid,source_row_id=rowid,true_class=int(truth[index]),
                old_only_pred=int(a[index]),full_pred=int(b[index]),old_best_logit=float(z[index,old].max()),
                new_best_logit=float(z[index,new].max()),new_minus_old_margin=float(margin[index])))
    with csv_path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    write_ecdf_svg(svg_path,m0,m1)
    weights=model.fc2.weight.detach().cpu().numpy()*model.weight_masks['fc2'].cpu().numpy()
    norms=np.linalg.norm(weights,axis=1)
    old_norm,new_norm=float(norms[list(old)].mean()),float(norms[list(new)].mean())
    report=dict(completed=True,diagnostic_only=True,oracle_old_class_mask=True,official_inference_changed=False,
        training_runs=0,threshold_fit=False,weight_alignment_applied=False,distillation=False,test_read=False,
        source_pilot_sha256=file_sha256(pilot_path),t1_checkpoint_sha256=file_sha256(path1),
        old_classes=old,new_classes=new,full_classes=full,baseline_t0=baseline,
        t1_old_only=old_only,t1_full_on_old=cumulative_old,
        old_only_accuracy_change_pp=100*(old_only['accuracy']-baseline['accuracy']),
        old_present_macro_recall_t0=float(np.mean([original0['per_class'][str(c)]['recall'] for c in old])),
        old_present_macro_recall_t1_old_only=float(np.mean([old_only['per_class'][c]['recall'] for c in old])),
        old_only_recall_change_pp={c:100*(old_only['per_class'][c]['recall']-original0['per_class'][str(c)]['recall']) for c in range(6)},
        old_only_correct_but_full_wrong=int(((pred_old==y0)&(pred_full!=y0)).sum()),
        margin_old=margin_summary(m0),margin_new=margin_summary(m1),
        margin_ranking_auc_new_vs_old=ranking_auc(m0,m1),
        classifier_norms=dict(per_class={c:float(norms[c]) for c in full},old_mean=old_norm,new_mean=new_norm,
            old_mean_over_new_mean=old_norm/new_norm,alignment_not_applied=True),
        checks=checks,integrity_pass=all(checks.values()),
        evidence_limits=['T0 checkpoint was not persisted in original pilot; original baseline/confusion are used unless an existing T0 checkpoint is supplied',
            'No T0/T1 feature-drift comparison possible without original T0 checkpoint',
            'Oracle old-only accuracy is diagnostic, not a deployable V2 metric; development panel already observed'],
        outputs=dict(rows_csv=csv_path.name,rows_csv_sha256=file_sha256(csv_path),ecdf_svg=svg_path.name,ecdf_svg_sha256=file_sha256(svg_path),
            ecdf_png=svg_path.with_suffix('.png').name,ecdf_png_sha256=file_sha256(svg_path.with_suffix('.png'))),
        source_sha256={p.relative_to(Path(__file__).resolve().parents[1]).as_posix():file_sha256(p)
            for p in [Path(__file__).resolve(),Path(__file__).resolve().with_name('run_denice_v2_routerless_smoke.py'),
                *sorted((Path(__file__).resolve().parents[1]/'fed_learning/strategies/decentralized/denice_v2').glob('*.py'))]})
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(t0_baseline=baseline['accuracy'],t1_old_only=old_only['accuracy'],t1_full_on_old=cumulative_old['accuracy'],
        change_pp=report['old_only_accuracy_change_pp'],margin_auc=report['margin_ranking_auc_new_vs_old'],integrity_pass=report['integrity_pass']),indent=2),flush=True)


if __name__=='__main__':
    main()
