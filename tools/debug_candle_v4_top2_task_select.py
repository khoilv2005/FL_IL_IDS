"""Locked mass-reranker diagnostic on the existing TEST development panel."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import json
import argparse
from pathlib import Path
import sys
import warnings

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'candle_v4_runtime'))
from candle_v4_top2_task_select import predict_selected,POLICY,PROTOTYPE_POLICY
from eval_checkpoint import _make_denice_client_model
from fed_learning.training.candle_v4_original_train import OriginalTrainService
from fed_learning.training.candle_v4_shadow import digest
from fed_learning.training.candle_v4_terminal_test import statistics
from fed_learning.training.decentralized_denice_il import _partition_test_data_by_client


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--rule',choices=('mass','prototype'),default='mass');args=parser.parse_args()
    warnings.filterwarnings('ignore',message='Trying to unpickle estimator')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ck=torch.load(ROOT/'audit_denice/candle_v4_results20_20261027/continuation_state_latest.pt',map_location='cpu',weights_only=False,mmap=True)
    service=OriginalTrainService(ck['candle_v4_shadow_state']['spec'],ck['candle_v4_shadow_state'])
    bank_before=digest(service.state_dict())
    with np.load(r'C:\Users\khoak\Downloads\candle_v1_multiclass_audit\global_test_data.npz') as z:
        x=torch.from_numpy(z['X_test']);y=torch.from_numpy(z['y_test']).long()
    ids=sorted(ck['client_model_states']);shards,_=_partition_test_data_by_client(x,y,ids,seed=523687,lazy_features=True)
    cms={k:np.zeros((34,34),dtype=np.int64) for k in ['top1','top2']};clients={}
    totals=dict(rows=0,switched=0,rescue=0,break_count=0,route_rescue=0,route_break=0,top1_route_correct=0,selected_route_correct=0)
    for pos,cid in enumerate(ids,1):
        shard=shards[cid];rng=np.random.default_rng(20261028+cid)
        idx=np.sort(rng.choice(len(shard['y_test']),2048,replace=False));inputs=shard['X_test'][idx];truth=shard['y_test'][idx].numpy()
        model,det=_make_denice_client_model(ck,cid,'cuda');model.eval();before=digest(model.state_dict())
        local={k:np.zeros_like(v) for k,v in cms.items()};counts={k:0 for k in totals}
        for lo in range(0,len(truth),512):
            xx=inputs[lo:lo+512].reshape(-1,39).cpu();yy=truth[lo:lo+512]
            with torch.no_grad():pred,top2,chosen=predict_selected(service,cid,model,det,xx,list(range(34)),rule=args.rule)
            for k,p in pred.items():np.add.at(local[k],(yy,p),1)
            first=top2[:,0];a=first==yy//6;b=chosen==yy//6
            values=dict(rows=len(yy),switched=int((first!=chosen).sum()),
                rescue=int(((pred['top2']==yy)&(pred['top1']!=yy)).sum()),
                break_count=int(((pred['top2']!=yy)&(pred['top1']==yy)).sum()),
                route_rescue=int((b&~a).sum()),route_break=int((a&~b).sum()),
                top1_route_correct=int(a.sum()),selected_route_correct=int(b.sum()))
            for k,v in values.items():counts[k]+=v
        if before!=digest(model.state_dict()):raise ValueError('Native weights changed')
        for k in cms:cms[k]+=local[k]
        for k,v in counts.items():totals[k]+=v
        clients[cid]=dict(counts=counts,metrics={k:statistics(cm) for k,cm in local.items()})
        del model,det,shards[cid]
        if pos%20==0:print(f'Top2 task selection {pos}/100',flush=True)
    if bank_before!=digest(service.state_dict()):raise ValueError('Bank changed')
    old=json.loads((ROOT/'audit_denice/candle_v4_results20_top2_20261028/audit.json').read_text())
    if cms['top1'].tolist()!=old['metrics']['baseline']['confusion_matrix']:raise ValueError('Panel Top1 no longer matches')
    result=dict(completed=True,policy=POLICY if args.rule=='mass' else PROTOTYPE_POLICY,counts=totals,metrics={k:statistics(cm) for k,cm in cms.items()},clients=clients,
        panel_seed=20261028,rows_per_client=2048,input_only_predictions=True,no_training=True,no_refit=True,
        no_threshold_tuning=True,frozen_state_verified=True,exact_top1_prior_panel_match=True,
        scientific_status='Repeated TEST development panel; not independent confirmation; predeclared rule, no parameter sweep')
    out=ROOT/('audit_denice/candle_v4_top2_task_select_20261029' if args.rule=='mass' else 'audit_denice/candle_v4_top2_prototype_select_20261029');out.mkdir(parents=True,exist_ok=True)
    (out/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(counts=totals,metrics={k:{m:v[m] for m in ['accuracy','macro_f1']} for k,v in result['metrics'].items()}),indent=2))


if __name__=='__main__':main()
