"""Attribute frozen T5 cosine scores to T0 and subsequently allocated FC1 coordinates."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from appliance.runner import load_input
from eval_checkpoint import _make_denice_client_model
from fed_learning.strategies.decentralized.candle_v2 import state
from tools.debug_candle_v2_checkpoints import write, fingerprint, infer, argmax, distribution
from fed_learning.data.denice_clean_roles import file_sha256


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--panel',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--clients',type=int,nargs='+',default=[4,9,65,98])
    p.add_argument('--device',default='cuda')
    a = p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    write(a.out/'protocol_before_data.json',dict(clients=a.clients,source_tasks=[0,5],
        no_training=True,no_fitting=True,no_final_TEST=True,
        counterfactual='Dropping post-T0 coordinates is a diagnostic, not input-dependent deployment',
        script_sha256=file_sha256(__file__),panel_sha256=file_sha256(a.panel)))
    torch.set_num_threads(2)
    started=time.monotonic()
    panel=np.load(a.panel);x,y=panel['x'],panel['y']
    reference={}
    ckpt,src0=load_input(a.inputs/'checkpoint_task_0_all_rounds.zip',0,19)
    for cid in a.clients:
        model,_=_make_denice_client_model(ckpt,cid,a.device)
        h,s,_=infer(model,x[y<6],ckpt['config']['candle_v2_logit_scale'],512)
        reference[cid]=dict(h=h,active=np.asarray(model.unit_ranks['fc1'])>0,
            available=state(model)['available'],scores=s)
        del model
    del ckpt;gc.collect();torch.cuda.empty_cache()
    ckpt,src5=load_input(a.inputs/'checkpoint_task_5_all_rounds.zip',5,19)
    rows=[]
    for cid in a.clients:
        model,_=_make_denice_client_model(ckpt,cid,a.device)
        before=fingerprint(model)
        scale=ckpt['config']['candle_v2_logit_scale']
        h,s,_=infer(model,x,scale,512)
        active=np.asarray(model.unit_ranks['fc1'])>0
        old=reference[cid]['active'];added=active&~old
        w=(model.fc2.weight*model.weight_masks['fc2'].to(model.fc2.weight)).cpu().numpy()*active
        hn=h/np.maximum(np.linalg.norm(h,axis=1,keepdims=True),1e-6)
        wn=w/np.maximum(np.linalg.norm(w,axis=1,keepdims=True),1e-6)
        old_score=float(scale)*(hn[:,old]@wn[:,old].T)
        added_score=float(scale)*(hn[:,added]@wn[:,added].T)
        residual=np.abs(s-old_score-added_score).max()
        if residual>1e-4:
            raise ValueError('Score decomposition does not reconstruct original')
        allowed=state(model)['available']
        native=argmax(s,allowed);counterfactual=argmax(old_score,allowed)
        old_ix=y<6;old_h=h[old_ix][:,old];ref_h=reference[cid]['h'][:,old]
        ref_available=reference[cid]['available']
        winner=native[old_ix]
        selected=np.arange(old_ix.sum())
        row=dict(client_id=cid,old_FC1_coordinates=int(old.sum()),added_FC1_coordinates=int(added.sum()),
            old_feature_max_abs_change=float(np.abs(old_h-ref_h).max()),
            old_feature_mean_abs_change=float(np.abs(old_h-ref_h).mean()),
            original_score_reconstruction_max_error=float(residual),
            T0_accuracy=float((argmax(reference[cid]['scores'],ref_available)==y[old_ix]).mean()),
            T5_old_within_T0_mask_accuracy=float((argmax(s[old_ix],ref_available)==y[old_ix]).mean()),
            T5_old_native_accuracy=float((native[old_ix]==y[old_ix]).mean()),
            T5_old_drop_added_coordinates_accuracy=float((counterfactual[old_ix]==y[old_ix]).mean()),
            T5_new_native_accuracy=float((native[y>=30]==y[y>=30]).mean()),
            T5_new_drop_added_coordinates_accuracy=float((counterfactual[y>=30]==y[y>=30]).mean()),
            added_feature_norm_over_total_on_old=distribution(np.linalg.norm(h[old_ix][:,added],axis=1)/np.maximum(np.linalg.norm(h[old_ix],axis=1),1e-8)),
            winning_score_on_old=dict(total_mean=float(s[old_ix][selected,winner].mean()),
                old_coordinate_contribution_mean=float(old_score[old_ix][selected,winner].mean()),
                added_coordinate_contribution_mean=float(added_score[old_ix][selected,winner].mean())))
        if fingerprint(model)!=before:
            raise ValueError('Inference mutated model')
        row['fingerprint']=before
        rows.append(row)
        print(json.dumps(row),flush=True)
        del model
    write(a.out/'completion.json',dict(completed=True,source_T0=src0,source_T5=src5,rows=rows,
        no_training=True,no_fitting=True,no_final_TEST=True,seconds=time.monotonic()-started))


if __name__=='__main__':
    main()
