"""Check paired banks and downstream classification using saved router choices."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from sklearn.metrics import f1_score
from eval_checkpoint import _make_denice_client_model
from fed_learning.training.denice_delta_checkpoint import load_denice_checkpoint
from fed_learning.training.denice_eval import _denice_routed_logits_with_episodes


@torch.no_grad()
def main():
    root=Path('audit_denice/results3')
    evidence=json.loads((root/'router_budget/evidence.json').read_text())
    assert evidence['complete'] and len(evidence['rows'])==60
    target=root/'router_budget/downstream.json'
    if target.exists():raise FileExistsError(target)
    ids=sorted({r['client'] for r in evidence['rows']})
    for cid in ids:
        for seed in evidence['seeds']:
            a,b=sorted([r for r in evidence['rows'] if r['client']==cid and r['seed']==seed],key=lambda r:r['budget'])
            assert a['validation_source_indices']==b['validation_source_indices']
            assert set(a['fit_source_indices'])<=set(b['fit_source_indices'])
            assert not set(b['fit_source_indices']) & set(b['validation_source_indices'])
            for r in [a,b]:
                assert len(set(r['fit_source_indices']))==r['fit_count']
                assert sum(r['class_counts'].values())==r['fit_count']
    torch.set_num_threads(1)
    ckpt=load_denice_checkpoint(str(root/'checkpoints/checkpoint_task_3_round_19.pt'))
    with np.load(root/'probe_panel.npz') as p:x=torch.from_numpy(p['x']);y=p['y']
    rows=[]
    for cid in ids:
        model,router=_make_denice_client_model(ckpt,cid,'cpu')
        choices=[]
        for ep in range(4):
            # Force each possible episode to cache its adapter + mask decision.
            # Actual routes below come only from the fitted router, never labels.
            logits,_=_denice_routed_logits_with_episodes(model,x,router,list(range(24)),'cpu',
                inference_policy='oracle_hard',oracle_episodes=np.full(len(y),ep))
            choices.append(logits.argmax(1).numpy())
        choices=np.stack(choices)
        for r in evidence['rows']:
            if r['client']!=cid:continue
            routes=np.asarray(r['test_predictions'])
            assert np.all((routes>=0)&(routes<4))
            pred=choices[routes,np.arange(len(y))]
            rows.append({'client':cid,'seed':r['seed'],'budget':r['budget'],
                'accuracy':float(np.mean(pred==y)),
                'f1_macro':float(f1_score(y,pred,labels=list(range(24)),average='macro',zero_division=0))})
        print('Verified downstream client',cid,flush=True)
    summary={}
    for seed in evidence['seeds']:
        summary[seed]={}
        for budget in evidence['budgets']:
            selected=[r for r in rows if r['seed']==seed and r['budget']==budget]
            summary[seed][budget]={k:float(np.mean([r[k] for r in selected])) for k in ['accuracy','f1_macro']}
    result={'paired_indices_verified':True,'classification_uses_router_predictions_only':True,
            'scope':'Same fixed 15 clients; local routing; 768 samples; exploratory reused test panel',
            'summary':summary,'rows':rows}
    target.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
