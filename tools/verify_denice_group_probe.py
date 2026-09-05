"""Fresh-process heldout verification of trusted small-run group checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from eval_checkpoint import _make_denice_client_model
from fed_learning.training.denice_eval import _denice_routed_logits_with_episodes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    manifest=json.loads((args.root/'panel/manifest.json').read_text())
    for name,digest in manifest['panel_hashes'].items():
        assert hashlib.sha256((args.root/'panel'/name).read_bytes()).hexdigest()==digest
    with np.load(args.root/'panel/test.npz',allow_pickle=False) as f:
        x,y=torch.from_numpy(f['x']),torch.from_numpy(f['y'])
    support={}
    for cid in manifest['client_ids']:
        with np.load(args.root/f'panel/client_{cid}.npz',allow_pickle=False) as f:support[cid]=np.unique(f['y'])
    checked=[]
    for path in sorted(args.root.glob('*/transitions.json')):
        d=json.loads(path.read_text())
        assert d['completed']
        assert d['panel_manifest']==manifest
        count=0
        fingerprints=[]
        for task in range(3):
            ckpt=torch.load(path.parent/f'training/checkpoint_task_{task}.pt',map_location='cpu',weights_only=False)
            digest=hashlib.sha256()
            for client_id,state in sorted(ckpt['client_model_states'].items()):
                for name,value in sorted(state.items()):
                    digest.update(f'{client_id}/{name}/{value.dtype}/{tuple(value.shape)}'.encode())
                    digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            fingerprints.append(digest.hexdigest())
            for cid in manifest['client_ids']:
                row=next(r for r in d['records'] if r['task']==task and r['client']==cid and r['stage']=='after_aging')
                model,router=_make_denice_client_model(ckpt,cid,'cpu')
                selected=y<6*(task+1); hx,hy=x[selected],y[selected]
                for policy in row['metrics']:
                    logits,routes=_denice_routed_logits_with_episodes(model,hx,router,list(range(6*(task+1))),
                                'cpu',inference_policy=policy,oracle_episodes=hy.numpy()//6)
                    pred=logits.argmax(1).numpy()
                    for old in range(task+1):
                        for subset in ['all','supported']:
                            mask=(hy.numpy()//6==old)
                            if subset=='supported':mask &= np.isin(hy.numpy(),support[cid])
                            acc=float(np.mean(pred[mask]==hy.numpy()[mask]))
                            assert acc==row['metrics'][policy][f'{old}_{subset}']['accuracy'],(path,task,cid,policy)
                            count+=1
            del ckpt
        checked.append({'run':path.parent.name,'exact_accuracy_comparisons':count,'task_model_tensor_hashes':fingerprints})
        print(checked[-1],flush=True)
    by_name={row['run']:row for row in checked}
    repeat_verified=None
    if '42_peer' in by_name and '42_peer_counterfactual' in by_name:
        repeat_verified=by_name['42_peer']['task_model_tensor_hashes']==by_name['42_peer_counterfactual']['task_model_tensor_hashes']
        assert repeat_verified, 'Diagnostic counterfactual changed actual training'
    result={'panel_hashes_verified':True,'fresh_process_checkpoint_accuracy_matches_live':True,
            'counterfactual_observer_preserved_all_task_model_tensors':repeat_verified,'runs':checked}
    (args.root/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()
