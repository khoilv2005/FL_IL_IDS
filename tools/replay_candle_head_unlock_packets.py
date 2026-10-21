"""Replay fixed primary and preregistered q=.01 ablation without fitting."""
import argparse
import copy
import io
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from appliance.runner import load_input
from appliance.state import complete_hash
from fed_learning.data.denice_clean_roles import CleanRoleData,file_sha256
from fed_learning.training.checkpoint_state import snapshot_denice_state
from tools.audit_candle_v1_two_tier import read,write
from tools.audit_candle_v1_positive_quantile_gate import evaluation_panel
from tools.audit_candle_v1_head_unlock import cache,candidate_prediction,stats
from tools.candle_head_unlock_runtime import predict_packet


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--inputs',type=Path,default=ROOT/'audit_denice/appliance_legacy_results11/inputs')
    p.add_argument('--roles',type=Path,default=ROOT/'audit_denice/appliance_legacy_results11/inputs/roles')
    p.add_argument('--data',type=Path,default=ROOT/'audit_denice/appliance_transfer_results9/inputs/dataset')
    p.add_argument('--device',default='cuda')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    started=time.monotonic();from eval_checkpoint import _make_denice_client_model
    ckpt,source=load_input(a.inputs/'checkpoint_task_4_all_rounds.zip',4,19)
    roles=CleanRoleData(a.roles,source_data_dir=a.data);seen=sorted(map(int,ckpt['seen_classes']))
    debug=read(a.inputs/'training_artifacts/denice_debug_history.json');records=[]
    write(a.out/'protocol_before_replay.json',dict(experiment=str(a.experiment),
        cases=[7,65],quantiles=[.05,.01],selection='Replay previously fixed primary and q=.01 ablation; do not choose thresholds here',
        final_TEST=False,no_optimizer=True,script_sha256=file_sha256(__file__),
        runtime_sha256=file_sha256(ROOT/'tools/candle_head_unlock_runtime.py')))
    for cid in [7,65]:
        result=read(a.experiment/f'receiver_{cid}/result.json')
        primary_packet=read(a.experiment/f'receiver_{cid}/primary_packet.json')
        model,router=_make_denice_client_model(ckpt,cid,a.device);model.eval()
        original=complete_hash(model,router)
        learned=set()
        for e in debug:
            if e.get('type')=='task_start' and e['task']<=4 and str(cid) in e['client_data']:
                learned.update(map(int,e['client_data'][str(cid)]['labels']))
        panel=evaluation_panel(roles,cid,11,learned,set())
        cached=cache(model,router,panel['x'],seen,a.device)
        alg=snapshot_denice_state(model,router)
        if alg['context_detector'].get('reference_input_memory'):
            raise ValueError('Raw reference memory must not enter a receiver capsule')
        capsule=dict(config=ckpt['config'],task=4,
            client_model_states={cid:{k:v.detach().cpu() for k,v in model.state_dict().items()}},
            client_algorithm_states={cid:{'denice':alg}})
        stream=io.BytesIO();torch.save(capsule,stream);setup_bytes=len(stream.getvalue())
        for q in [.05,.01]:
            row=next(r for r in result['ablations'] if r['head']==result['primary']['head']
                     and r['spec']['gate']==result['primary']['spec']['gate'] and r['quantile']==q)
            packet=copy.deepcopy(primary_packet);packet['spec']=row['spec'];packet['quantile']=q
            packet['authorization']='development replay only; q=.01 is a secondary ablation'
            expected=candidate_prediction(cached,dict(w=np.asarray(packet['head']['w'],np.float32),b=packet['head']['b']),
                {k:np.asarray(v,np.float32) for k,v in packet['prototypes'].items()},packet['spec'])
            actual,activated=predict_packet(model,router,panel['x'],packet,seen,a.device)
            if not np.array_equal(expected,actual):raise ValueError('Label-blind runtime/cache prediction mismatch')
            metrics=stats(actual,panel['y'],cached['pred'])
            if metrics!=row['metrics']:raise ValueError('Replay differs from locked ablation result')
            # Serialize only the head and the prototype actually used at inference.
            meta=dict(class_id=24,context_id=4,spec=packet['spec'],quantile=q,source=source)
            proto_name='donor' if packet['spec']['gate']=='donor_proto' else 'receiver'
            encoded=io.BytesIO()
            np.savez_compressed(encoded,weight=np.asarray(packet['head']['w'],np.float32),
                bias=np.asarray(packet['head']['b'],np.float32),prototype=np.asarray(packet['prototypes'][proto_name],np.float32),
                metadata=np.frombuffer(json.dumps(meta,sort_keys=True).encode(),dtype=np.uint8))
            leaf=f'receiver_{cid}_q_{str(q).replace(".","p")}'
            (a.out/(leaf+'.npz')).write_bytes(encoded.getvalue());write(a.out/(leaf+'.json'),packet)
            records.append(dict(receiver=cid,quantile=q,primary=q==.05,metrics=metrics,
                label_blind_runtime_matches_locked_predictions=True,activated=int(activated.sum()),
                receiver_function_reference_serialized_bytes=setup_bytes,
                capability_npz_bytes=len(encoded.getvalue()),raw_head_and_prototype_FP32_bytes=2052,
                serialized_inventory_not_full_network_traffic=True))
            print('REPLAY',cid,q,'recall',metrics['recall'],'break',metrics['breaks'],
                'capability',len(encoded.getvalue()),'setup',setup_bytes,flush=True)
        if original!=complete_hash(model,router):raise ValueError('Runtime mutated source model/router')
    write(a.out/'completion.json',dict(completed=True,seconds=time.monotonic()-started,records=records,
        final_TEST_read=False,no_fitting=True,reference_states_unchanged=True))
    print('Completed',a.out/'completion.json',flush=True)


if __name__=='__main__':main()
