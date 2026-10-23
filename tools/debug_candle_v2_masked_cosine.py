"""Single locked score ablation: project feature normalization to each FC2 row's mask."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from appliance.runner import load_input
from eval_checkpoint import _make_denice_client_model
from fed_learning.data.denice_clean_roles import file_sha256
from fed_learning.strategies.decentralized.candle_v2 import state,features,logits
from tools.debug_candle_v2_checkpoints import write,fingerprint,argmax,metrics


def masked_cosine(model,h,scale):
    # Binary masks define the actual input space of each classifier row.
    active=torch.as_tensor(np.asarray(model.unit_ranks['fc1'])>0,device=h.device)
    mask=model.weight_masks['fc2'].to(h)*active.to(h.dtype)[None,:]
    if not bool(((mask==0)|(mask==1)).all()):
        raise ValueError('This score contract requires binary masks')
    weight=model.fc2.weight*mask
    weight_norm=weight.float().norm(dim=1).clamp_min(1e-6)
    projected_feature_norm=torch.sqrt(torch.clamp(h.float().square()@mask.float().T,min=0)).clamp_min(1e-6)
    return float(scale)*(h.float()@weight.float().T)/(projected_feature_norm*weight_norm[None,:])


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--panel',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    write(a.out/'protocol_before_data.json',dict(version='per_row_masked_cosine_diagnostic_v1',
        script_sha256=file_sha256(__file__),panel_sha256=file_sha256(a.panel),
        change='Only feature norm: ||h|| becomes ||M_c*h|| for each row c',
        no_weights_changed=True,no_masks_changed=True,no_availability_changed=True,
        no_true_task_at_inference=True,no_threshold_selection=True,
        no_training=True,no_fitting=True,no_final_TEST=True,
        decision='Descriptive ablation on previously exposed development only; never silently replace production inference'))
    torch.set_num_threads(2);started=time.monotonic()
    panel=np.load(a.panel);x,y=panel['x'],panel['y']
    ckpt,source=load_input(a.checkpoint,5,19)
    rows=[];baseline_cm=np.zeros((34,34),np.int64);candidate_cm=baseline_cm.copy()
    for cid in sorted(map(int,ckpt['client_model_states'])):
        model,_=_make_denice_client_model(ckpt,cid,a.device);before=fingerprint(model)
        baseline=[];candidate=[];delta=[]
        for start in range(0,len(x),512):
            h=features(model,torch.from_numpy(x[start:start+512]).to(a.device))
            old=logits(model,h,range(34),ckpt['config']['candle_v2_logit_scale'])
            new=masked_cosine(model,h,ckpt['config']['candle_v2_logit_scale'])
            if not torch.isfinite(new).all():
                raise FloatingPointError('Nonfinite projected cosine')
            baseline.append(old.cpu().numpy());candidate.append(new.cpu().numpy())
        available=state(model)['available']
        bp=argmax(np.concatenate(baseline),available);cp=argmax(np.concatenate(candidate),available)
        rescue=int(((bp!=y)&(cp==y)).sum());broken=int(((bp==y)&(cp!=y)).sum())
        if int((cp==y).sum()-(bp==y).sum())!=rescue-broken:
            raise ValueError('Rescue/break accounting mismatch')
        row=dict(client_id=cid,baseline=metrics(y,bp),candidate=metrics(y,cp),rescue=rescue,breaks=broken,
            baseline_predictions_into_T5=int((bp>=30).sum()),candidate_predictions_into_T5=int((cp>=30).sum()),
            old_accuracy_baseline=float((bp[y<30]==y[y<30]).mean()),
            old_accuracy_candidate=float((cp[y<30]==y[y<30]).mean()),
            new_accuracy_baseline=float((bp[y>=30]==y[y>=30]).mean()),
            new_accuracy_candidate=float((cp[y>=30]==y[y>=30]).mean()))
        if fingerprint(model)!=before:
            raise ValueError('Frozen model was mutated')
        rows.append(row)
        np.add.at(baseline_cm,(y,bp),1);np.add.at(candidate_cm,(y,cp),1)
        write(a.out/f'client_{cid}.json',row)
        print(f'receiver={cid} baseline={row["baseline"]["accuracy"]:.2%} per-row={row["candidate"]["accuracy"]:.2%} rescue={rescue} break={broken}',flush=True)
        del model
    def pooled(cm):
        diag=np.diag(cm);support=cm.sum(1);pred=cm.sum(0)
        return dict(accuracy=float(diag.sum()/cm.sum()),
            macro_f1=float(np.divide(2*diag,support+pred,out=np.zeros(34),where=support+pred>0).mean()),
            per_class_recall=(diag/support).tolist())
    result=dict(completed=True,source=source,model_evaluations=len(rows),unique_panel_rows=len(y),
        baseline=pooled(baseline_cm),candidate=pooled(candidate_cm),
        baseline_confusion=baseline_cm.tolist(),candidate_confusion=candidate_cm.tolist(),
        rescue=sum(r['rescue'] for r in rows),breaks=sum(r['breaks'] for r in rows),rows=rows,
        no_training=True,no_fitting=True,no_final_TEST=True,seconds=time.monotonic()-started,
        caveat='Development score ablation; no acceptance claim or claim of improved full TEST')
    write(a.out/'completion.json',result)
    print(json.dumps({k:result[k] for k in ('baseline','candidate','rescue','breaks','seconds')}),flush=True)
    gc.collect()


if __name__=='__main__':
    main()
