"""Direct full-forward cross-check of the matrix-replacement readout."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import sys,json,pathlib,copy
import torch,numpy as np
from threadpoolctl import threadpool_limits


def main():
    root=pathlib.Path(r'C:\Users\khoak\Downloads\candle_v1_multiclass_audit')
    out=pathlib.Path('audit_denice/candle_v4_multiclass_scoped_20261024')
    sys.path.insert(0,str(root/'source'))
    from eval_checkpoint import _make_denice_client_model
    from fed_learning.training.decentralized_denice_il import _partition_test_data_by_client
    from fed_learning.training.denice_eval import _denice_routed_logits_with_episodes
    from fed_learning.training.candle_v4_shadow import make_expert
    from fed_learning.training.candle_v4_bank import fuse
    torch.set_num_threads(2);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    d=torch.load(r'C:\Users\khoak\Downloads\candle_v4_debug\continuation_state_latest.pt',map_location='cpu',weights_only=False)
    ref=json.loads((out/'comparison.json').read_text());bank=d['candle_v4_shadow_state']
    with np.load(root/'global_test_data.npz') as a:x=torch.from_numpy(a['X_test']);y=torch.from_numpy(a['y_test']).long()
    shards,_=_partition_test_data_by_client(x,y,sorted(d['client_ids']),seed=523687,lazy_features=True)
    receipt=dict(passed=False,clients={},note='Direct full native multiclass forward plus scoped expert fusion; no reused class confusion cells')
    for cid in [9,71]:
        model,det=_make_denice_client_model(d,cid,'cuda');model.eval();det.router_mode='multiclass_balanced'
        with threadpool_limits(limits=2):det.train_models(max(det.activation_memory))
        keys=sorted(k for k in bank['entries'] if k[0]==cid and bank['status']['coverage'][k]=='ACTIVE')
        classes=np.array([k[1] for k in keys]);experts=[]
        for k in keys:
            e=make_expert(42);e.load_state_dict(bank['engines']['coverage'][k]['branches']['joint_class_balanced']['model']);e.eval();experts.append(e)
        cms={k:np.zeros((34,34),dtype=np.int64) for k in ['native','scoped']}
        with torch.no_grad(),threadpool_limits(limits=2):
            for start in range(0,len(shards[cid]['y_test']),512):
                inputs=shards[cid]['X_test'][start:start+512];truth=shards[cid]['y_test'][start:start+512].numpy()
                logits,routes=_denice_routed_logits_with_episodes(model,inputs.to('cuda'),det,list(range(34)),'cuda',inference_policy='pred_hard')
                native=logits.argmax(1).cpu().numpy();scores=np.column_stack([torch.sigmoid(e(inputs.reshape(-1,39))).flatten().numpy() for e in experts])
                scoped,_=fuse(native,np.where(classes[None,:]//6==routes[:,None],scores,-1.),[.5]*len(classes),classes)
                for arm,pred in [('native',native),('scoped',scoped)]:np.add.at(cms[arm],(truth,pred),1)
        check={arm:int(np.abs(cm-np.asarray(ref['per_client'][str(cid)][arm]['confusion_matrix'])).sum()) for arm,cm in cms.items()}
        receipt['clients'][str(cid)]=check;assert all(v==0 for v in check.values()),check
        print(cid,check,flush=True)
    receipt['passed']=True;(out/'direct_forward_verification.json').write_text(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
