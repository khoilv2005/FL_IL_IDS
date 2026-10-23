"""Frozen-checkpoint router diagnostics; all refits use training references only."""
import json
from pathlib import Path
import sys
from copy import deepcopy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from eval_checkpoint import _make_denice_client_model
from fed_learning.training.denice_delta_checkpoint import load_denice_checkpoint
from fed_learning.servers.nice_server import ContextDetector
from fed_learning.training.checkpoint_state import restore_context_detector


def score(pred,y):
    return {'accuracy':float(np.mean(pred==y)),
            'per_task':[float(np.mean(pred[y==t]==t)) for t in range(4)],
            'confusion':confusion_matrix(y,pred,labels=list(range(4))).tolist()}


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    root=Path('audit_denice/results3')
    out=root/'router_diagnostic_v2.json'
    if out.exists():raise FileExistsError(out)
    ckpt=load_denice_checkpoint(str(root/'checkpoints/checkpoint_task_3_round_19.pt'))
    with np.load(root/'probe_panel.npz') as p:x=torch.from_numpy(p['x']);labels=p['y'];y=labels//6
    signatures={}
    for client,state in ckpt['client_algorithm_states'].items():
        detector=ContextDetector()
        restore_context_detector(detector,state.get('denice',state)['context_detector'])
        signatures[int(client)]=detector.calibration_signature()
    groups=ckpt['cluster']['groups']
    pooling={}
    for client in signatures:
        group=set(map(int,groups.get(str(client),groups.get(client,[client]))))|{client}
        sigs={signatures[g] for g in group if g in signatures}
        pooling[client]={'group_size':len(group),'compatible':None not in sigs and len(sigs)==1}
    ids=json.loads((root/'probe_summary.json').read_text())['fixed_client_ids']
    rows=[]
    for cid in ids:
        model,det=_make_denice_client_model(ckpt,cid,'cpu')
        def acts(data):
            return {k:v.numpy() for k,v in model.get_context_activations_per_sample(data).items()}
        test_acts=acts(x)
        test_binary=det.binarize_layer_activations(test_acts)
        episodes=sorted(det.reference_input_memory)
        raw=torch.from_numpy(np.concatenate([det.reference_input_memory[e] for e in episodes])).float()
        ry=np.concatenate([np.full(len(det.reference_input_memory[e]),e) for e in episodes])
        stored=np.concatenate([det.activation_memory[e] for e in episodes])
        ref_acts=acts(raw)
        fresh=det.binarize_layer_activations(ref_acts)
        pred,probs=det.predict_episodes_with_scores(test_binary)
        router=det.multiclass_router
        manual=router.classes_[(test_binary@router.coef_.T+router.intercept_).argmax(1)]
        row={'client':cid,'reference_counts':{e:len(det.reference_input_memory[e]) for e in episodes},
             'thresholds':det.binarize_thresholds,'signature':det.calibration_signature(),
             'manual_linear_predict_agreement':float(np.mean(manual==pred)),
             'reference_bit_mismatch':float(np.mean(fresh!=stored)),
             'variable_binary_features':int((fresh.std(0)>0).sum()),'feature_count':fresh.shape[1],
             'saved_reference_score':score(det.predict_episodes_batch(stored),ry),
             'saved_test_score':score(pred,y),
             'wrong_confidence_mean':float(probs.max(1)[pred!=y].mean()),
             'per_class_route_accuracy':{c:float(np.mean(pred[labels==c]==c//6)) for c in range(24)}}
        # Same feature encoding, train-only refit: checks refresh/estimator effects.
        binary_lr=LogisticRegression(max_iter=1000,class_weight='balanced').fit(fresh,ry)
        row['fresh_binary_test_score']=score(binary_lr.predict(test_binary),y)
        # Continuous features test information lost in binarization; scaler sees references only.
        names=['conv1','conv2','conv3','gru']
        continuous=np.concatenate([ref_acts[k] for k in names],axis=1)
        continuous_test=np.concatenate([test_acts[k] for k in names],axis=1)
        clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight='balanced'))
        clf.fit(continuous,ry)
        row['continuous_reference_score']=score(clf.predict(continuous),ry)
        row['continuous_test_score']=score(clf.predict(continuous_test),y)
        rows.append(row)
        out.write_text(json.dumps({'scope':'15 fixed clients; same 768 test samples; no test fitting; frozen backbone',
                                   'pooling_compatibility':pooling,
                                   'rows':rows,'complete':len(rows)==len(ids)},indent=2),encoding='utf-8')
        print(cid,'ref',round(row['saved_reference_score']['accuracy'],3),'test',round(row['saved_test_score']['accuracy'],3),
              'fresh',round(row['fresh_binary_test_score']['accuracy'],3),'continuous',round(row['continuous_test_score']['accuracy'],3),flush=True)


if __name__=='__main__':main()
