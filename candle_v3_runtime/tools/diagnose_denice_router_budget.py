"""Paired 20/100 reference router ablation on frozen task-3 backbones."""
import json
from pathlib import Path
import sys
from zipfile import ZipFile
from copy import deepcopy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from tools.diagnose_denice_real_client import sample_npz
from tools.diagnose_denice_router import score
from eval_checkpoint import _make_denice_client_model
from fed_learning.training.denice_delta_checkpoint import load_denice_checkpoint


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    root=Path('audit_denice/results3')
    out=root/'router_budget'
    out.mkdir(exist_ok=False)
    ckpt=load_denice_checkpoint(str(root/'checkpoints/checkpoint_task_3_round_19.pt'))
    ids=json.loads((root/'probe_summary.json').read_text())['fixed_client_ids']
    with np.load(root/'probe_panel.npz') as p:
        tx=torch.from_numpy(p['x']);ty=p['y']//6
    result={'scope':'Frozen backbone, same 15 clients and 768 test examples; local original train data only',
            'seeds':[2028,2029],'budgets':[20,100],'validation':'disjoint source indices; up to 20/class reserved first; not unseen by backbone',
            'caveat':'Fresh local bank differs from stored historical/inherited bank; compare 20 vs 100 within each seed',
            'rows':[],'complete':False}
    with ZipFile(r'C:\Users\khoak\Downloads\archive.zip') as archive:
        for cid in ids:
            model,det=_make_denice_client_model(ckpt,cid,'cpu')
            payload=archive.read(f'100-clients/client_{cid}_train.npz')
            test_binary=det._binarize_per_sample(model,tx)
            for seed in result['seeds']:
                sx,sy,source=sample_npz(payload,'train',120,seed+cid,num_classes=24)
                sy=sy.numpy();source=np.asarray(source)
                rng=np.random.default_rng(seed+cid)
                banks={20:[],100:[]};validation=[]
                for c in range(24):
                    pool=rng.permutation(np.flatnonzero(sy==c))
                    # Keep at least one fit example where any class data exists.
                    nv=min(20,len(pool)//5)
                    validation.extend(pool[:nv].tolist())
                    rest=pool[nv:]
                    for budget in banks:banks[budget].extend(rest[:budget].tolist())
                assert set(banks[20])<=set(banks[100])
                assert not set(validation)&set(banks[100])
                chunks=[det._binarize_per_sample(model,sx[i:i+256]) for i in range(0,len(sx),256)]
                binary=np.concatenate(chunks)
                for budget,indices in banks.items():
                    clf=LogisticRegression(max_iter=1000,class_weight='balanced').fit(binary[indices],sy[indices]//6)
                    probs=clf.predict_proba(test_binary);pred=clf.classes_[probs.argmax(1)]
                    row={'client':cid,'seed':seed,'budget':budget,'fit_count':len(indices),
                         'validation_count':len(validation),'fit_source_indices':source[indices].tolist(),
                         'validation_source_indices':source[validation].tolist(),
                         'class_counts':{c:int((sy[indices]==c).sum()) for c in range(24)},
                         'fit':score(clf.predict(binary[indices]),sy[indices]//6),
                         'validation':score(clf.predict(binary[validation]),sy[validation]//6),
                         'test':score(pred,ty),'test_predictions':pred.tolist(),
                         'test_wrong_confidence':float(probs.max(1)[pred!=ty].mean())}
                    result['rows'].append(row)
                a,b=result['rows'][-2:]
                print(cid,seed,'20:',round(a['test']['accuracy']*100,2),'100:',round(b['test']['accuracy']*100,2),flush=True)
            (out/'evidence.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    result['complete']=True
    result['summary']={}
    for seed in result['seeds']:
        paired=[]
        for cid in ids:
            a,b=[r for r in result['rows'] if r['client']==cid and r['seed']==seed]
            paired.append(b['test']['accuracy']-a['test']['accuracy'])
        result['summary'][seed]={'mean_test_gain_pp':float(np.mean(paired)*100),'clients_improved':sum(d>0 for d in paired),
            'budgets':{b:{key:float(np.mean([r[key]['accuracy'] for r in result['rows'] if r['seed']==seed and r['budget']==b]))
                         for key in ['fit','validation','test']} for b in result['budgets']}}
    (out/'evidence.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['summary'],indent=2),flush=True)


if __name__=='__main__':main()
