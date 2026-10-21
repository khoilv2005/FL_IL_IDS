"""Frozen-checkpoint multiclass disambiguation and receiver-target-free thresholds."""
import argparse
from collections import Counter
import copy
import gc
import hashlib
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from appliance.runner import load_input
from appliance.state import complete_hash
from fed_learning.data.denice_clean_roles import CleanRoleData, file_sha256
from tools.audit_candle_v1_two_tier import read, write, digest, features, partition_pool
from tools.audit_candle_v1_head_unlock import cache, normalized_dot
from tools.candle_head_unlock_runtime import predict_packet

RULES = dict(version='candle_multiclass_prototype_v1', task=4, round=19,
    cases=[dict(receiver=7, targets={24:11, 29:3}),
           dict(receiver=23, targets={24:21, 25:13, 26:96}),
           dict(receiver=65, targets={24:11})],
    case_selection='Receiver7 continuity, receiver23 largest minimum donor CAL count among >=3 actual intra gaps; receiver65 extra control',
    threshold_primary='max(donor self-cosine mean minus 2 std, nextafter(max receiver current negative cosine,+inf))',
    threshold_secondary=['donor_only', 'receiver_negative_only'],
    calibration='Task4 CAL construction content partition only; no receiver target features for threshold',
    prototype='Donor CAL-FIT FC1 mean; no receiver-space prototype in new primary',
    head_secondary='Diagonal moment transform on same donor-positive FIT using receiver function; costs reference',
    methods=['prototype_cosine','direct_head_cosine','adjusted_head_cosine','adjusted_head_raw'],
    decision='Among gated imported candidates choose method score argmax; no candidate => exact native fallback',
    evaluation='Validation role only, max256 global unique rows/class; exclude previous validation panels by content',
    broad_panel_classes=list(range(34)),future_classes=[30,31,32,33],
    strict_gate=dict(min_recall_each_target=.95,FAR=.001,negative_break=0),
    no_optimizer=True,no_final_TEST=True,no_memory_merge=True,
    exposure='Development sources previously audited; content-disjoint from prior panels is not untouched confirmation')


def content_pool(x, y, permitted=None, excluded=None, cap=None):
    used=set(excluded or []);counts=Counter();rows=[];labels=[];hashes=[]
    for value,label in zip(x,y):
        label=int(label)
        if permitted is not None and label not in permitted:continue
        if cap is not None and counts[label]>=cap:continue
        key=digest(value)
        if key in used:continue
        used.add(key);counts[label]+=1;rows.append(value);labels.append(label);hashes.append(key)
    return dict(x=np.stack(rows) if rows else np.empty((0,*x.shape[1:]),np.float32),
                y=np.asarray(labels,np.int64),hashes=hashes)


def learned_classes(debug,cid):
    return set(int(c) for event in debug if event.get('type')=='task_start' and event['task']<=4
               for c in event['client_data'].get(str(cid),{}).get('labels',[]))


def build_panel(roles,debug,out):
    # Explicitly exclude the actual first-256 validation panels used previously,
    # including duplicate content that might appear under another owner.
    excluded=set();loaded={};access=[]
    def load(cid):
        if cid not in loaded:
            loaded[cid]=roles.client_role(cid,'validation')[:2];access.append(cid)
        return loaded[cid]
    for cid,labels in [(7,learned_classes(debug,7)-{24}),
                       (65,learned_classes(debug,65)-{24}),(11,{24})]:
        x,y=load(cid)
        for c in labels:
            excluded.update(digest(v) for v in x[np.flatnonzero(y==c)[:256]])
    selected=[];all_used=set(excluded);per_class={};owners={}
    for c in range(34):
        candidates=sorted(map(int,roles.manifest['clients']),key=lambda cid:(
            -roles.manifest['clients'][str(cid)]['role_class_counts']['validation'].get(str(c),0),cid))
        count=0
        for cid in candidates:
            n=roles.manifest['clients'][str(cid)]['role_class_counts']['validation'].get(str(c),0)
            if not n:continue
            x,y=load(cid)
            pool=content_pool(x,y,{c},all_used,256-count)
            if len(pool['y']):
                selected.append(pool);all_used.update(pool['hashes']);count+=len(pool['y'])
                owners.setdefault(str(c),[]).append(dict(owner=cid,rows=len(pool['y'])))
            if count>=256:break
        per_class[str(c)]=count
        if not count:raise ValueError(f'No independent validation rows for class {c}')
        print('Validation panel class',c,'rows',count,flush=True)
    panel=dict(x=np.concatenate([p['x'] for p in selected]),y=np.concatenate([p['y'] for p in selected]),
               hashes=[h for p in selected for h in p['hashes']])
    write(out/'panel_receipt.json',dict(per_class=per_class,owners=owners,prior_content_excluded=len(excluded),
        overlap_prior_validation=0,role='validation',role_shards_read=access,
        content_hashes=panel['hashes'],future_classes_diagnostic_only=[30,31,32,33]))
    return panel


def metrics(pred,y,baseline,targets):
    pos=np.isin(y,targets);neg=~pos;correct=pred==y;old=baseline==y
    rescue=(~old)&correct;broken=old&(~correct);fa=neg&np.isin(pred,targets)
    def subset(mask):
        n=int(mask.sum());nn=int((mask&neg).sum())
        return dict(rows=n,accuracy=float(correct[mask].mean()) if n else None,
            rescue=int((rescue&mask).sum()),breaks=int((broken&mask).sum()),
            negatives=nn,false_activation=int((fa&mask).sum()),
            FAR=float(fa[mask&neg].mean()) if nn else None)
    result=subset(np.ones(len(y),bool));result['per_target']={str(c):dict(rows=int((y==c).sum()),
        recall=float((pred[y==c]==c).mean())) for c in targets}
    result['per_negative_class']={str(c):subset(y==c) for c in sorted(set(y)-set(targets))}
    result['cumulative_through_T4']=subset(y<30);result['future_class_stress_only']=subset(y>=30)
    result['empirical_gate']=all(v['recall']>=.95 for v in result['per_target'].values()) and result['FAR']<=.001 and result['breaks']==0
    if int(correct.sum()-old.sum())!=result['rescue']-result['breaks']:raise ValueError('Accounting mismatch')
    return result


def choose(scores,gated,targets,fallback):
    result=fallback.copy();active=gated.any(1)
    result[active]=np.asarray(targets)[np.where(gated,scores,-np.inf).argmax(1)[active]]
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--inputs',type=Path,default=ROOT/'audit_denice/appliance_legacy_results11/inputs')
    p.add_argument('--roles',type=Path,default=ROOT/'audit_denice/appliance_legacy_results11/inputs/roles')
    p.add_argument('--data',type=Path,default=ROOT/'audit_denice/appliance_transfer_results9/inputs/dataset')
    p.add_argument('--store',type=Path,default=ROOT/'audit_denice/appliance_current_runtime_data_local/full100_v1/calibration_store')
    p.add_argument('--device',default='cuda')
    a=p.parse_args();a.task=4;a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    started=time.monotonic();write(a.out/'protocol_before_data.json',dict(rules=RULES,script_sha256=file_sha256(__file__)))
    from eval_checkpoint import _make_denice_client_model
    ckpt,source=load_input(a.inputs/'checkpoint_task_4_all_rounds.zip',4,19)
    roles=CleanRoleData(a.roles,source_data_dir=a.data);role_sha=file_sha256(a.roles/'role_manifest.json')
    if ckpt['config']['denice_data_roles_sha256']!=role_sha:raise ValueError('Role identity mismatch')
    classes=read(a.store/'calibration_store_manifest.json')['task_classes']['4']
    seen=sorted(map(int,ckpt['seen_classes']));debug=read(a.inputs/'training_artifacts/denice_debug_history.json')
    all_cal_hashes=set();prepared=[]
    # Finish and hash every threshold/packet BEFORE opening evaluation validation.
    for case in RULES['cases']:
        cid=case['receiver'];targets=sorted(case['targets'])
        model,router=_make_denice_client_model(ckpt,cid,a.device);model.eval();before=complete_hash(model,router)
        if any(c in values for c in targets for values in router.episode_classes.values()):raise ValueError('Expected actual unregistered gaps')
        receiver=partition_pool(a,None,role_sha,cid,classes)['construction']
        negative=~np.isin(receiver['y'],targets)
        if negative.sum()<16:raise ValueError('Insufficient current receiver negatives')
        nx=receiver['x'][negative];all_cal_hashes.update(receiver['hashes'])
        nh,_=features(model,router,nx,4,a.device)
        packs=[]
        for c in targets:
            donor_id=case['targets'][c]
            pool=partition_pool(a,None,role_sha,donor_id,classes)['construction']
            ids=np.flatnonzero(pool['y']==c)
            if len(ids)<16:raise ValueError('Insufficient donor current positives')
            px=pool['x'][ids];all_cal_hashes.update(pool['hashes'])
            donor,dr=_make_denice_client_model(ckpt,donor_id,a.device);donor.eval();dbefore=complete_hash(donor,dr)
            dh,_=features(donor,dr,px,4,a.device);proto=dh.mean(0)
            donor_cos=normalized_dot(dh,proto);negative_cos=normalized_dot(nh,proto)
            tau_d=float(donor_cos.mean()-2*donor_cos.std())
            tau_n=float(np.nextafter(np.float32(negative_cos.max()),np.float32(np.inf)))
            rh,_=features(model,router,px,4,a.device)
            w=(donor.fc2.weight[c]*donor.weight_masks['fc2'][c].to(donor.fc2.weight)).detach().cpu().numpy()
            b=float((donor.fc2.bias[c]*donor.bias_masks['fc2'][c].to(donor.fc2.bias)).detach().cpu())
            scale=np.clip(dh.std(0)/np.maximum(rh.std(0),1e-3),.25,4.)
            adjusted=w*scale;bias=b+float(w@(dh.mean(0)-scale*rh.mean(0)))
            packs.append(dict(class_id=c,donor=donor_id,prototype=proto.tolist(),direct_w=w.tolist(),
                direct_b=b,adjusted_w=adjusted.tolist(),adjusted_b=bias,
                thresholds=dict(primary=max(tau_d,tau_n),donor_only=tau_d,receiver_negative_only=tau_n),
                donor_self_cosine=dict(mean=float(donor_cos.mean()),std=float(donor_cos.std())),
                donor_positive_FIT_rows=len(px),receiver_negative_FIT_rows=len(nx),
                receiver_target_features_for_threshold=False,
                receiver_reference_required_for_adjusted_head=True))
            if dbefore!=complete_hash(donor,dr):raise ValueError('Donor changed')
            del donor,dr
        if before!=complete_hash(model,router):raise ValueError('Receiver changed')
        lock=dict(case=case,packets=packs,source=source,receiver_complete_hash=before)
        path=a.out/f'receiver_{cid}_lock.json';write(path,lock)
        prepared.append((lock,file_sha256(path)));print('LOCKED',cid,[(r['class_id'],r['thresholds']) for r in packs],flush=True)
        del model,router;gc.collect();torch.cuda.empty_cache()
    panel=build_panel(roles,debug,a.out)
    if set(panel['hashes'])&all_cal_hashes:raise ValueError('Validation/calibration overlap')
    results=[]
    for lock,lock_sha in prepared:
        cid=lock['case']['receiver'];packs=lock['packets'];targets=[p['class_id'] for p in packs]
        model,router=_make_denice_client_model(ckpt,cid,a.device);model.eval();before=complete_hash(model,router)
        data=cache(model,router,panel['x'],seen,a.device);h=data['h'][4];y=panel['y'];fallback=data['pred']
        proto_scores=np.stack([normalized_dot(h,np.asarray(p['prototype'],np.float32)) for p in packs],axis=1)
        score_methods=dict(prototype_cosine=proto_scores,
            direct_head_cosine=np.stack([normalized_dot(h,np.asarray(p['direct_w'],np.float32)) for p in packs],axis=1),
            adjusted_head_cosine=np.stack([normalized_dot(h,np.asarray(p['adjusted_w'],np.float32)) for p in packs],axis=1),
            adjusted_head_raw=np.stack([h@np.asarray(p['adjusted_w'],np.float32)+p['adjusted_b'] for p in packs],axis=1))
        rows=[]
        for policy in ['primary','donor_only','receiver_negative_only']:
            gated=proto_scores>=np.asarray([p['thresholds'][policy] for p in packs])
            collision=gated.sum(1)>=2
            prototype_pred=choose(proto_scores,gated,targets,fallback)
            for method,score in score_methods.items():
                pred=choose(score,gated,targets,fallback)
                rows.append(dict(policy=policy,method=method,metrics=metrics(pred,y,fallback,targets),
                    total_activated=int(gated.any(1).sum()),multiple_gate_matches=int(collision.sum()),
                    imported_target_collisions=int((collision&np.isin(y,targets)).sum()),
                    different_predictions_from_prototype=int((pred!=prototype_pred).sum()),
                    collision_accuracy=float((pred[collision]==y[collision]).mean()) if collision.any() else None))
                print('EVAL',cid,policy,method,'target',rows[-1]['metrics']['per_target'],
                      'FP',rows[-1]['metrics']['false_activation'],'break',rows[-1]['metrics']['breaks'],
                      'collisions',int(collision.sum()),flush=True)
        fixed=[]
        if cid in [7,65]:
            directory=ROOT/'audit_denice/candle_v1_head_unlock_replay_20261022_v1'
            for q in ['0p05','0p01']:
                packet=read(directory/f'receiver_{cid}_q_{q}.json')
                pred,_=predict_packet(model,router,panel['x'],packet,seen,a.device)
                proto=np.asarray(packet['prototypes']['donor' if packet['spec']['gate']=='donor_proto' else 'receiver'],np.float32)
                simple=fallback.copy();simple[normalized_dot(h,proto)>=packet['spec']['tau']]=24
                fixed.append(dict(quantile=q,head_plus_gate=metrics(pred,y,fallback,[24]),
                    prototype_only=metrics(simple,y,fallback,[24]),prediction_disagreements=int((pred!=simple).sum())))
        if before!=complete_hash(model,router) or before!=lock['receiver_complete_hash']:raise ValueError('Source state changed')
        if lock_sha!=file_sha256(a.out/f'receiver_{cid}_lock.json'):raise ValueError('Threshold lock changed')
        result=dict(receiver=cid,targets=targets,variants=rows,prior_fixed_packets_broad_panel=fixed,
                    source_unchanged=True,lock_sha256=lock_sha,
                    negative_scope='34-class validation stress; T5 future classes separately reported')
        write(a.out/f'receiver_{cid}_result.json',result);results.append(result)
        del model,router,data;gc.collect();torch.cuda.empty_cache()
    write(a.out/'completion.json',dict(completed=True,seconds=time.monotonic()-started,source=source,
        results=results,no_final_TEST=True,no_optimizer=True,all_thresholds_locked_before_validation=True))


if __name__=='__main__':main()
