"""Reproduction gate for the opt-in production ContextDetector routing path."""
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from eval_checkpoint import _make_denice_client_model
from fed_learning.servers.nice_server import ContextDetector
from fed_learning.training.checkpoint_state import snapshot_context_detector, restore_context_detector
from fed_learning.strategies.incremental.denice_tip_router import encoder_fingerprint
from tools.eval_denice_tip_router import classify, write_json


def run_multiclass_integration(ckpt,shards,classes,ids,out,device,diagnostic_input,batch_size=512):
    source = Path(diagnostic_input)
    if source.is_dir():
        reference = pd.read_csv(source/'predictions.csv')
        manifest = json.loads((source/'profile_manifest.json').read_text())
        protocol = json.loads((source/'protocol.json').read_text())
    else:
        with zipfile.ZipFile(source) as z:
            reference = pd.read_csv(z.open('predictions.csv'))
            manifest = json.loads(z.read('profile_manifest.json'))
            protocol = json.loads(z.read('protocol.json'))
    if protocol['client_ids'] != ids or not protocol['training_commit'].startswith('03b9b53'):
        raise ValueError('Expected the original baseline panel/checkpoint')
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    state_dir=out/'router_states'; state_dir.mkdir(exist_ok=True)
    seen=sorted({c for values in classes.values() for c in values})
    mapping={c:t for t,values in classes.items() for c in values}
    rows,frames=[],[]
    for index,cid in enumerate(ids,1):
        model,detector=_make_denice_client_model(ckpt,cid,device)
        signature=encoder_fingerprint(model)
        if signature != manifest[str(cid)]['encoder_hash']:
            raise ValueError(f'{cid}: encoder fingerprint mismatch')
        x=shards[cid]['X_test']; y=shards[cid]['y_test'].numpy()
        prior=reference[reference.client_id==cid].sort_values('sample_in_shard').reset_index(drop=True)
        if (not np.array_equal(prior.global_test_row,shards[cid]['sample_ids'])
                or not np.array_equal(prior.y_true,y)):
            raise ValueError(f'{cid}: original sample identities differ')
        legacy,legacy_route=classify(model,detector,x,seen,device,batch_size)
        if not (np.array_equal(legacy,prior.Router) and np.array_equal(legacy_route,prior.Router_task)):
            raise ValueError(f'{cid}: legacy reproduction failed')
        detector.router_mode='multiclass_balanced'
        detector.train_models(max(detector.activation_memory))
        # This is the normal pred_hard pipeline, not oracle_hard with forced IDs.
        prediction,route=classify(model,detector,x,seen,device,batch_size)
        path=state_dir/f'client_{cid}_context.pt'
        torch.save(snapshot_context_detector(detector),path)
        restored=ContextDetector()
        restore_context_detector(restored,torch.load(path,map_location='cpu',weights_only=False))
        again,again_route=classify(model,restored,x,seen,device,batch_size)
        if not (np.array_equal(prediction,again) and np.array_equal(route,again_route)):
            raise ValueError(f'{cid}: serialized router restore changed predictions')
        if encoder_fingerprint(model) != signature:
            raise RuntimeError(f'{cid}: inference mutated classifier state')
        truth_task=np.asarray([mapping[int(v)] for v in y])
        rows.append(dict(client_id=cid,samples=len(y),legacy_accuracy=float((legacy==y).mean()),
            reference_accuracy=float((prior.Multiclass==y).mean()),
            integrated_accuracy=float((prediction==y).mean()),
            integrated_f1_macro=float(f1_score(y,prediction,average='macro',zero_division=0)),
            route_accuracy_global=float((route==truth_task).mean()),
            prediction_mismatches=int((prediction!=prior.Multiclass).sum()),
            route_mismatches=int((route!=prior.Multiclass_task).sum()),restore_exact=True))
        frames.append(pd.DataFrame(dict(client_id=cid,global_test_row=prior.global_test_row,
            y_true=y,true_task=truth_task,prediction=prediction,predicted_task=route,
            reference_prediction=prior.Multiclass,reference_task=prior.Multiclass_task)))
        pd.DataFrame(rows).to_csv(out/'integration_per_client.csv',index=False)
        print(f'Integration {index}/{len(ids)} cid={cid}: accuracy={rows[-1]["integrated_accuracy"]:.6f}, '
              f'class mismatches={rows[-1]["prediction_mismatches"]}, task mismatches={rows[-1]["route_mismatches"]}',flush=True)
        del model,detector,restored
        if device=='cuda': torch.cuda.empty_cache()
    metrics=pd.DataFrame(rows); predictions=pd.concat(frames,ignore_index=True)
    predictions.to_csv(out/'integration_predictions.csv',index=False)
    observed=float(metrics.integrated_accuracy.mean()); expected=float(metrics.reference_accuracy.mean())
    result=dict(router_mode='multiclass_balanced',client_mean_accuracy=observed,
        reference_client_mean_accuracy=expected,delta_pp=100*(observed-expected),
        pooled_accuracy=float((predictions.prediction==predictions.y_true).mean()),
        pooled_macro_f1=float(f1_score(predictions.y_true,predictions.prediction,labels=seen,average='macro',zero_division=0)),
        route_accuracy_global=float((predictions.predicted_task==predictions.true_task).mean()),
        prediction_mismatches=int(metrics.prediction_mismatches.sum()),
        route_mismatches=int(metrics.route_mismatches.sum()),
        restore_exact=True,tolerance_pp=0.1,
        passed=bool(abs(expected-0.27768144439771997)<=0.001 and abs(observed-expected)<=0.001))
    write_json(out/'integration_gate.json',result)
    if not result['passed']:
        raise RuntimeError('Multiclass normal-pipeline reproduction gate failed; inspect outputs')
    return result
