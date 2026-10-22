"""Verify the one-forward repair on the old panel and a content-disjoint validation tail."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from appliance.runner import load_input
from eval_checkpoint import _make_denice_client_model
from fed_learning.data.denice_clean_roles import CleanRoleData,file_sha256
from fed_learning.strategies.decentralized.candle_v2 import features
from fed_learning.strategies.decentralized.candle_v2_inference import predict,ROUTED,features_and_context
from tools.debug_candle_v2_checkpoints import write,panel,content_hash,fingerprint


class ExcludedValidation:
    def __init__(self,roles,excluded):
        self.roles=roles;self.manifest=roles.manifest;self.excluded=excluded

    def client_role(self,cid,role):
        if role!='validation':raise ValueError('Only validation is permitted')
        x,y,rows=self.roles.client_role(cid,role)
        keep=np.asarray([content_hash(v) not in self.excluded for v in x])
        return x[keep],y[keep],rows[keep]


def score(y,pred,baseline):
    support=np.bincount(y,minlength=34)
    cm=np.bincount(y*34+pred,minlength=34**2).reshape(34,34)
    d=np.diag(cm);fp=cm.sum(0)
    correct=int((pred==y).sum());old_correct=int((baseline==y).sum())
    rescue=int(((baseline!=y)&(pred==y)).sum());broken=int(((baseline==y)&(pred!=y)).sum())
    if correct-old_correct!=rescue-broken:raise ValueError('Accounting mismatch')
    return dict(rows=len(y),correct=correct,accuracy=correct/len(y),rescue=rescue,breaks=broken,
        confusion=cm.tolist(),macro_f1=float(np.divide(2*d,support+fp,out=np.zeros(34),where=support+fp>0).mean()),
        old_correct=int((pred[y<30]==y[y<30]).sum()),new_correct=int((pred[y>=30]==y[y>=30]).sum()))


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--router-control',type=Path,required=True)
    p.add_argument('--roles',type=Path,default=ROOT/'audit_denice/appliance_legacy_results11/inputs/roles')
    p.add_argument('--data',type=Path,default=ROOT/'audit_denice/appliance_transfer_results9/inputs/dataset')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    started=time.monotonic()
    write(a.out/'protocol_before_data.json',dict(policy=ROUTED,no_training=True,no_fitting=True,no_final_TEST=True,
        reference_panel_sha256=file_sha256(a.reference/'development_panel.npz'),
        inference_module_sha256=file_sha256(ROOT/'fed_learning/strategies/decentralized/candle_v2_inference.py'),
        script_sha256=file_sha256(__file__),tail_cap_per_class=64,
        tail='Unique validation content excluding all original panel hashes; shortages retained and reported',
        empty_scope='Counted original-taskfree fallback; never authorize new classes',
        exposure='Same previously exposed development source; content-disjoint is not independent confirmation'))
    reference=np.load(a.reference/'development_panel.npz');rx,ry=reference['x'],reference['y']
    excluded={content_hash(v) for v in rx}
    roles=CleanRoleData(a.roles,source_data_dir=a.data)
    tail_dir=a.out/'tail';tail_dir.mkdir()
    tx,ty=panel(ExcludedValidation(roles,excluded),tail_dir,64)
    if any(content_hash(v) in excluded for v in tx):raise ValueError('Panel overlap')
    ckpt,source=load_input(a.inputs/'checkpoint_task_5_all_rounds.zip',5,19)
    if ckpt['config']['denice_data_roles_sha256']!=file_sha256(a.roles/'role_manifest.json'):
        raise ValueError('Role identity mismatch')
    records=[];feature_error=0.;context_error=0.;reference_disagreement=0
    for cid in sorted(map(int,ckpt['client_model_states'])):
        model,router=_make_denice_client_model(ckpt,cid,a.device)
        before=fingerprint(model);rb=hashlib.sha256(pickle.dumps(router)).hexdigest()
        # Verify fused implementation against both canonical forward paths.
        sample=torch.from_numpy(rx[:16]).to(a.device)
        h,acts=features_and_context(model,sample)
        feature_error=max(feature_error,float((h-features(model,sample)).abs().max()))
        canonical=model.get_context_activations_per_sample(sample)
        context_error=max(context_error,max(float((acts[k]-canonical[k]).abs().max()) for k in acts))
        row=dict(client_id=cid)
        for name,x,y in [('reference',rx,ry),('tail',tx,ty)]:
            base=[];repaired=[];episodes=[];fallback=0
            for i in range(0,len(x),512):
                rec=predict(model,torch.from_numpy(x[i:i+512]).to(a.device),ckpt['config'],router=router,policy=ROUTED)
                base.append(rec['unrouted_logits'].argmax(1).cpu().numpy())
                repaired.append(rec['logits'].argmax(1).cpu().numpy())
                episodes.append(rec['episodes']);fallback+=int(rec['fallback'].sum())
            bp=np.concatenate(base);rp=np.concatenate(repaired);ep=np.concatenate(episodes)
            if name=='reference':
                old=np.load(a.reference/f'T5_client_{cid}_scores.npz')
                control=np.load(a.router_control/f'client_{cid}_predictions.npz')
                mismatch=int((bp!=old['pred']).sum())+int((rp!=control['registered']).sum())
                reference_disagreement+=mismatch
                if mismatch:raise ValueError('Repair differs from saved-router control or original baseline')
            row[name]=dict(baseline=score(y,bp,bp),repaired=score(y,rp,bp),
                route_accuracy=float((ep==y//6).mean()),fallback_rows=fallback)
        if before!=fingerprint(model) or rb!=hashlib.sha256(pickle.dumps(router)).hexdigest():
            raise ValueError('Repair mutated frozen state')
        records.append(row);write(a.out/f'client_{cid}.json',row)
        print(f'client={cid} tail baseline={row["tail"]["baseline"]["accuracy"]:.2%} repaired={row["tail"]["repaired"]["accuracy"]:.2%}',flush=True)
        del model,router
    if feature_error>1e-6 or context_error>1e-6:raise ValueError('Single-forward features differ')
    summaries={}
    for name,y in [('reference',ry),('tail',ty)]:
        scope=dict(unique_inputs=len(y),class_counts=np.bincount(y,minlength=34).tolist(),
            model_evaluations=len(records),prediction_rows=len(y)*len(records),
            route_accuracy=float(np.mean([r[name]['route_accuracy'] for r in records])),
            fallback_rows=sum(r[name]['fallback_rows'] for r in records))
        for method in ('baseline','repaired'):
            cm=sum(np.asarray(r[name][method]['confusion']) for r in records)
            d=np.diag(cm);support=cm.sum(1);pred=cm.sum(0)
            scope[method]=dict(accuracy=float(d.sum()/cm.sum()),
                macro_f1=float(np.divide(2*d,support+pred,out=np.zeros(34),where=support+pred>0).mean()),
                old_accuracy=float(d[:30].sum()/support[:30].sum()),
                new_accuracy=float(d[30:].sum()/support[30:].sum()),
                confusion=cm.tolist(),rescue=sum(r[name][method]['rescue'] for r in records),
                breaks=sum(r[name][method]['breaks'] for r in records))
        summaries[name]=scope
    result=dict(completed=True,policy=ROUTED,source=source,summary=summaries,
        feature_forward_max_error=feature_error,context_forward_max_error=context_error,
        reference_prediction_disagreements=reference_disagreement,tail_reference_content_overlap=0,
        no_training=True,no_fitting=True,no_final_TEST=True,seconds=time.monotonic()-started)
    write(a.out/'completion.json',result)
    print(json.dumps({k:{m:v[m]['accuracy'] for m in ('baseline','repaired')} for k,v in summaries.items()}),flush=True)


if __name__=='__main__':main()
