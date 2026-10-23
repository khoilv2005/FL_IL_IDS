"""Frozen saved-router control with native versus CANDLE class registration."""
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
from fed_learning.training.denice_eval import _route_episodes
from fed_learning.strategies.decentralized.candle_v2 import state
from fed_learning.data.denice_clean_roles import file_sha256
from tools.debug_candle_v2_checkpoints import write,metrics,argmax,fingerprint


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--diagnostics',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    write(a.out/'protocol_before_data.json',dict(version='saved_binary_router_control_v1',
        script_sha256=file_sha256(__file__),panel_sha256=file_sha256(a.diagnostics/'development_panel.npz'),
        methods=['original_taskfree','saved_binary_router_native_episode_classes','saved_binary_router_CANDLE_available_by_class_tasks'],
        empty_scope_policy='Explicit task-free fallback, counted',
        no_training=True,no_fitting=True,no_true_task_at_inference=True,no_final_TEST=True))
    panel=np.load(a.diagnostics/'development_panel.npz');x,y=panel['x'],panel['y']
    ckpt,source=load_input(a.checkpoint,5,19);rows=[];started=time.monotonic()
    for cid in sorted(map(int,ckpt['client_model_states'])):
        model,router=_make_denice_client_model(ckpt,cid,a.device)
        if router.router_mode!='binary_cosine':raise ValueError('Expected saved binary_cosine')
        before=fingerprint(model);router_before=hashlib.sha256(pickle.dumps(router)).hexdigest()
        reference=json.loads((a.diagnostics/f'T5_client_{cid}.json').read_text(encoding='utf-8'))
        if before!=reference['model_fingerprint_before_after']:raise ValueError('Model identity differs from cached scores')
        saved=np.load(a.diagnostics/f'T5_client_{cid}_scores.npz')
        if not np.array_equal(y,saved['y']):raise ValueError('Panel row order differs')
        scores=saved['cosine'];baseline=argmax(scores,state(model)['available'])
        episodes=np.concatenate([_route_episodes(model,torch.from_numpy(x[i:i+512]).to(a.device),router) for i in range(0,len(x),512)])
        local=baseline.copy();registered=baseline.copy();fallback_local=0;fallback_registered=0
        available=set(map(int,state(model)['available']))
        mapping={int(c):int(t) for c,t in state(model)['class_tasks'].items()}
        for ep in np.unique(episodes):
            ix=episodes==ep
            classes=sorted(available&set(map(int,router.episode_classes.get(int(ep),[]))))
            if classes:local[ix]=argmax(scores[ix],classes)
            else:fallback_local+=int(ix.sum())
            classes=sorted(c for c in available if mapping[c]==ep)
            if classes:registered[ix]=argmax(scores[ix],classes)
            else:fallback_registered+=int(ix.sum())
        if before!=fingerprint(model) or router_before!=hashlib.sha256(pickle.dumps(router)).hexdigest():
            raise ValueError('Inference mutated model/router')
        record=dict(client_id=cid,route_accuracy=float((episodes==y//6).mean()),
            taskfree=metrics(y,baseline),native_episode_classes=metrics(y,local),class_tasks=metrics(y,registered),
            fallback_native=fallback_local,fallback_registered=fallback_registered,
            registered_rescue=int(((baseline!=y)&(registered==y)).sum()),
            registered_break=int(((baseline==y)&(registered!=y)).sum()))
        rows.append(record);write(a.out/f'client_{cid}.json',record)
        np.savez_compressed(a.out/f'client_{cid}_predictions.npz',episodes=episodes,native=local,registered=registered)
        print(f'client={cid} taskfree={record["taskfree"]["accuracy"]:.2%} native={record["native_episode_classes"]["accuracy"]:.2%} registered={record["class_tasks"]["accuracy"]:.2%} route={record["route_accuracy"]:.2%}',flush=True)
        del model,router
    result=dict(completed=True,source=source,rows=rows,model_evaluations=len(rows),
        summary={key:dict(accuracy=float(np.mean([r[key]['accuracy'] for r in rows])),
                          correct=sum(r[key]['correct'] for r in rows),prediction_rows=sum(r[key]['rows'] for r in rows))
                 for key in ('taskfree','native_episode_classes','class_tasks')},
        route_accuracy=float(np.mean([r['route_accuracy'] for r in rows])),
        fallback_registered=sum(r['fallback_registered'] for r in rows),
        rescue=sum(r['registered_rescue'] for r in rows),breaks=sum(r['registered_break'] for r in rows),
        no_training=True,no_fitting=True,no_final_TEST=True,seconds=time.monotonic()-started)
    write(a.out/'completion.json',result);print(json.dumps(result['summary']),flush=True)


if __name__=='__main__':main()
