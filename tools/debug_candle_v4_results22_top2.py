"""Bounded frozen 2x2 attribution of actual Top2 union failures. No tuning."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import json
from pathlib import Path
import sys
import warnings
import zipfile

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'candle_v4_runtime')]
import eval_candle_v4_task_round19_topk as evaluation
from eval_checkpoint import _make_denice_client_model
from fed_learning.training.candle_v4_original_train import OriginalTrainService
from fed_learning.training.candle_v4_shadow import digest
from fed_learning.training.candle_v4_terminal_test import statistics
from fed_learning.training.decentralized_denice_il import _partition_test_data_by_client


def main():
    warnings.filterwarnings('ignore',message='Trying to unpickle estimator')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    info=[]
    for n in (20,21):
        with zipfile.ZipFile(Path(r'C:\Users\khoak\Downloads')/f'results ({n}).zip') as z:
            item=z.getinfo(next(n for n in z.namelist() if n.endswith('/continuation_state_latest.pt')))
            info.append((item.CRC,item.file_size))
    if info[0]!=info[1]:raise ValueError('Need results21 checkpoint; results20 differs')
    ck=torch.load(ROOT/'audit_denice/candle_v4_results20_20261027/continuation_state_latest.pt',map_location='cpu',weights_only=False,mmap=True)
    service=OriginalTrainService(ck['candle_v4_shadow_state']['spec'],ck['candle_v4_shadow_state'])
    bank_before=digest(service.state_dict())
    with np.load(r'C:\Users\khoak\Downloads\candle_v1_multiclass_audit\global_test_data.npz') as z:
        x=torch.from_numpy(z['X_test']);y=torch.from_numpy(z['y_test']).long()
    ids=sorted(ck['client_model_states']);shards,_=_partition_test_data_by_client(x,y,ids,seed=523687,lazy_features=True)
    names=['N1_E1','N2_E1','N1_E2','N2_E2','native_top1','native_top2']
    cms={name:np.zeros((34,34),dtype=np.int64) for name in names}
    counts={name:dict(rescue=0,break_count=0) for name in names}
    reasons=dict(new_second_task_expert=0,native_expansion_without_expert=0,other=0)
    source=digest(ck['client_model_states']);clients={};original=evaluation.fuse_candidates
    for position,cid in enumerate(ids,1):
        shard=shards[cid];rng=np.random.default_rng(20261028+cid)
        idx=np.sort(rng.choice(len(shard['y_test']),size=2048,replace=False))
        xx=shard['X_test'][idx];yy=shard['y_test'][idx].numpy()
        model,det=_make_denice_client_model(ck,cid,'cuda');model.eval();before=digest(model.state_dict())
        local={name:np.zeros_like(cm) for name,cm in cms.items()}
        for lo in range(0,len(yy),512):
            trace=[]
            def capture(native,scores,classes,births,selected):
                trace.append((native.copy(),scores,classes,births,selected.copy()))
                return original(native,scores,classes,births,selected)
            evaluation.fuse_candidates=capture
            try:
                with torch.no_grad():pred,selected,_=evaluation.predict_pair(service,cid,model,det,xx[lo:lo+512].reshape(-1,39).cpu(),list(range(34)))
            finally:evaluation.fuse_candidates=original
            (n1,scores,classes,births,s1),(n2,_,_,_,s2)=trace
            branches=dict(N1_E1=pred['top1'],N2_E2=pred['top2'],native_top1=n1,native_top2=n2)
            branches['N2_E1']=original(n2,scores,classes,births,s1)[0]
            branches['N1_E2']=original(n1,scores,classes,births,s2)[0]
            truth=yy[lo:lo+512];baseok=branches['N1_E1']==truth
            for name,p in branches.items():
                np.add.at(local[name],(truth,p),1)
                counts[name]['rescue']+=int(((p==truth)&~baseok).sum())
                counts[name]['break_count']+=int(((p!=truth)&baseok).sum())
            broken=(branches['N2_E2']!=truth)&baseok
            active=(s2[:,:,None]==np.asarray(births)[None,None,:]).any(1)&(scores>=.5)
            fire=active.any(1)
            newexpert=broken&fire&(branches['N2_E2']//6==s2[:,1])
            native=broken&~fire
            reasons['new_second_task_expert']+=int(newexpert.sum())
            reasons['native_expansion_without_expert']+=int(native.sum())
            reasons['other']+=int((broken&~newexpert&~native).sum())
        if before!=digest(model.state_dict()):raise ValueError('Native changed')
        for name in names:cms[name]+=local[name]
        clients[cid]={name:dict(accuracy=statistics(cm)['accuracy'],macro_f1=statistics(cm)['macro_f1']) for name,cm in local.items()}
        del model,det,shards[cid]
        if position%20==0:print(f'Top2 attribution {position}/100',flush=True)
    if bank_before!=digest(service.state_dict()) or source!=digest(ck['client_model_states']):raise ValueError('Frozen state changed')
    result=dict(completed=True,panel_rows=204800,panel_seed=20261028,rows_per_client=2048,
        predictions_use_true_labels=False,metrics={k:statistics(cm) for k,cm in cms.items()},
        changes_vs_top1=counts,break_attribution=reasons,clients=clients,
        no_training=True,no_fit=True,no_threshold_tuning=True,frozen_state_verified=True,
        checkpoint20_21_same_crc_and_size=True,local_sklearn_coefficients_only_not_refit=True)
    target=ROOT/'audit_denice/candle_v4_results22_20261029/panel_attribution.json'
    target.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(metrics={k:{a:v[a] for a in ['accuracy','macro_f1']} for k,v in result['metrics'].items()},
                         changes=counts,break_attribution=reasons),indent=2))


if __name__=='__main__':main()
