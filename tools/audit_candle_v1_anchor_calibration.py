"""Target-free current-anchor alignment, frozen thresholds and multiclass scoring."""
import argparse
from collections import Counter
import gc
import hashlib
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from appliance.current_base_data import CurrentBaseData
from appliance.current_calibration_data import CurrentCalibrationData
from appliance.runner import load_input
from appliance.state import complete_hash
from fed_learning.data.denice_clean_roles import CleanRoleData,file_sha256
from tools.audit_candle_v1_two_tier import read,write,digest,features
from tools.audit_candle_v1_head_unlock import cache,normalized_dot
from tools.audit_candle_v1_multiclass_prototype import content_pool,learned_classes,metrics,choose

RULES=dict(version='candle_anchor_calibration_v1',task=4,round=19,
    cases=[dict(receiver=7,donor=11,targets=[24]),dict(receiver=65,donor=11,targets=[24]),
           dict(receiver=23,donor=13,targets=[24,25,26])],
    common_donor_selection='Receiver23 donor13 has two current anchors with >=32 BASE rows and maximum minimum target BASE count among qualifying common mature in-graph donors',
    anchor_scope='Actually locally trained common Task4 classes, excluding every imported target',
    anchor_min_BASE_rows=32,anchor_min_unique_per_half=8,BASE_cap_per_class_half=256,
    anchor_split='SHA256 anchor-fit-v1 plus input content parity, shared across owners',
    target_prototype='Donor-only current BASE FIT; no target input is run through receiver until evaluation',
    target_moments='Donor current CAL, minimum8 unique positives; empirical fitting only, not acceptance certification',
    background='Receiver current CAL negatives only, quantile0.999 linear, std floor1e-6',
    geometries=['raw','mean_shift','centered'],policies=['attenuation_and_background','background_only','donor_unadjusted'],
    scorers=['prototype_cosine','prototype_z','affine_head_z','centered_subhead_z'],
    threshold='max(donor target mean minus mean anchor attenuation minus2 target std, receiver negative q.999)',
    primary_A='mean_shift + attenuation_and_background + prototype_cosine',
    primary_B='centered + attenuation_and_background + prototype_z; compare centered_subhead_z with identical gates',
    headline_gate=dict(recall_each=.95,FAR=.001,breaks=0),secondary_user_FAR=.005,
    evaluation='Reused34-class panel for paired diagnostics plus content-disjoint validation tail where available; never fit on either',
    no_optimizer=True,no_historical_raw_BASE_or_CAL=True,no_final_TEST=True,
    limitation='Anchor mean/variance transfer is an empirical assumption, not guaranteed alignment or safety')


def normalize(v):
    return v/max(float(np.linalg.norm(v)),1e-8)


def base_halves(pool,classes):
    unique=content_pool(pool['X'],pool['y'],set(classes))
    side=np.asarray([int(hashlib.sha256(('anchor-fit-v1:'+h).encode()).hexdigest()[:8],16)%2 for h in unique['hashes']])
    halves=[]
    for part in [0,1]:
        ids=[]
        for c in classes:ids.extend(np.flatnonzero((unique['y']==c)&(side==part))[:256].tolist())
        ids=np.asarray(ids,np.int64)
        halves.append(dict(x=unique['x'][ids],y=unique['y'][ids]))
    return halves


def calibration_pool(root,cid,classes,role_sha):
    view=CurrentCalibrationData(root,cid,4,role_sha)
    pool=view.current_pool(cid,'calibration',classes)
    return content_pool(pool['X'],pool['y'])


def prepare(a,ckpt,roles,classes,case):
    from eval_checkpoint import _make_denice_client_model
    r,d,targets=case['receiver'],case['donor'],case['targets']
    rb=CurrentBaseData(a.base_store,r,4,a.role_sha).current_pool(r,'base',classes)
    db=CurrentBaseData(a.base_store,d,4,a.role_sha).current_pool(d,'base',classes)
    common=sorted(c for c in classes if c not in targets and (rb['y']==c).sum()>=32 and (db['y']==c).sum()>=32)
    if not common:return dict(case=case,status='UNKNOWN_NO_CURRENT_ANCHORS')
    rfit,rhold=base_halves(rb,common);dfit,dhold=base_halves(db,common+targets)
    common=[c for c in common if min((rfit['y']==c).sum(),(rhold['y']==c).sum(),(dfit['y']==c).sum(),(dhold['y']==c).sum())>=8]
    if not common:return dict(case=case,status='UNKNOWN_INSUFFICIENT_UNIQUE_ANCHORS')
    model,router=_make_denice_client_model(ckpt,r,a.device)
    donor,dr=_make_denice_client_model(ckpt,d,a.device);model.eval();donor.eval()
    before=(complete_hash(model,router),complete_hash(donor,dr))
    rf,_=features(model,router,rfit['x'],4,a.device);rh,_=features(model,router,rhold['x'],4,a.device)
    df,_=features(donor,dr,dfit['x'],4,a.device);dh,_=features(donor,dr,dhold['x'],4,a.device)
    pr=np.stack([rf[rfit['y']==c].mean(0) for c in common])
    pd=np.stack([df[dfit['y']==c].mean(0) for c in common])
    center_r=pr.mean(0);center_d=pd.mean(0);shift=center_r-center_d
    rp=calibration_pool(a.cal_store,r,classes,a.role_sha)
    rp={k:(v[~np.isin(rp['y'],targets)] if k in ('x','y') else v) for k,v in rp.items()}
    if len(rp['y'])<16:raise ValueError('Insufficient current receiver background')
    dp=calibration_pool(a.cal_store,d,classes,a.role_sha)
    bg,_=features(model,router,rp['x'],4,a.device)
    packets=[]
    for c in targets:
        target_ids=np.flatnonzero(dp['y']==c)
        if len(target_ids)<8:return dict(case=case,status='UNKNOWN_DONOR_TARGET_CAL_LT8',class_id=c)
        tc,_=features(donor,dr,dp['x'][target_ids],4,a.device)
        proto=df[dfit['y']==c].mean(0)
        if not np.isfinite(proto).all():raise ValueError('Nonfinite donor prototype')
        weight=(donor.fc2.weight[c]*donor.weight_masks['fc2'][c].to(donor.fc2.weight)).detach().cpu().numpy()
        bias=float((donor.fc2.bias[c]*donor.bias_masks['fc2'][c].to(donor.fc2.bias)).detach().cpu())
        variants={}
        for geometry in RULES['geometries']:
            if geometry=='raw':cr=np.zeros(256,np.float32);cd=cr;translated=np.zeros(256,np.float32)
            elif geometry=='mean_shift':cr=np.zeros(256,np.float32);cd=cr;translated=shift
            else:cr=center_r;cd=center_d;translated=shift
            effective=proto+translated-cr
            bg_score=normalized_dot(bg-cr,effective)
            target_score=normalized_dot(tc-cd,proto-cd)
            attenuation=[];anchor_details=[]
            for k,y in enumerate(common):
                donor_score=normalized_dot(dh[dhold['y']==y]-cd,pd[k]-cd)
                receiver_score=normalized_dot(rh[rhold['y']==y]-cr,pd[k]+translated-cr)
                delta=float(donor_score.mean()-receiver_score.mean());attenuation.append(delta)
                anchor_details.append(dict(class_id=y,donor_mean=float(donor_score.mean()),receiver_cross_mean=float(receiver_score.mean()),delta=delta))
            delta=float(np.mean(attenuation));td=float(target_score.mean()-2*target_score.std())
            tn=float(np.quantile(bg_score,.999,method='linear'))
            # h_R approximately h_D+shift: w unchanged, bias_D-dot(w,shift).
            affine_bias=bias-float(weight@translated)+float(weight@cr)
            head_background=(bg-cr)@weight+affine_bias
            variants[geometry]=dict(prototype_effective=effective.tolist(),receiver_center=cr.tolist(),
                attenuation=delta,anchor_details=anchor_details,
                donor_target_mean=float(target_score.mean()),donor_target_std=float(target_score.std()),
                background_mean=float(bg_score.mean()),background_std=max(float(bg_score.std()),1e-6),
                thresholds=dict(attenuation_and_background=max(td-delta,tn),background_only=tn,donor_unadjusted=td),
                affine_bias=affine_bias,head_bg_mean=float(head_background.mean()),head_bg_std=max(float(head_background.std()),1e-6))
        packets.append(dict(class_id=c,prototype_donor=proto.tolist(),weight=weight.tolist(),bias=bias,
            target_CAL_rows=len(target_ids),geometries=variants))
    # Column centering subtracts the same linear function from every class.
    # It preserves raw pairwise margins, but per-class background z normalization can change ranking.
    mean_w=np.mean([p['weight'] for p in packets],axis=0)
    for p in packets:
        w=np.asarray(p['weight'])-mean_w
        for geom,g in p['geometries'].items():
            cr=np.asarray(g['receiver_center']);v=(bg-cr)@w
            g['subhead_weight']=w.tolist();g['subhead_bg_mean']=float(v.mean());g['subhead_bg_std']=max(float(v.std()),1e-6)
    if before!=(complete_hash(model,router),complete_hash(donor,dr)):raise ValueError('Source mutation')
    result=dict(case=case,status='READY',anchors=common,receiver_hash=before[0],donor_hash=before[1],
        center_r=center_r.tolist(),center_d=center_d.tolist(),shift_norm=float(np.linalg.norm(shift)),
        anchor_counts={str(c):dict(receiver_fit=int((rfit['y']==c).sum()),receiver_hold=int((rhold['y']==c).sum()),
            donor_fit=int((dfit['y']==c).sum()),donor_hold=int((dhold['y']==c).sum())) for c in common},
        receiver_background_CAL_rows=len(bg),packets=packets,
        target_inputs_processed_by_receiver_before_evaluation=0,
        raw_samples_transmitted=False,offline_single_process_not_network_protocol=True)
    del model,router,donor,dr;gc.collect();torch.cuda.empty_cache()
    return result


def panels(roles,debug,reference,out):
    previous=read(reference/'panel_receipt.json');wanted=previous['content_hashes'];wanted_set=set(wanted)
    loaded={};reference_rows={};excluded=set(wanted);access=[]
    def load(cid):
        if cid not in loaded:loaded[cid]=roles.client_role(cid,'validation')[:2];access.append(cid)
        return loaded[cid]
    for cid,cs in [(7,learned_classes(debug,7)-{24}),(65,learned_classes(debug,65)-{24}),(11,{24})]:
        x,y=load(cid)
        for c in cs:excluded.update(digest(v) for v in x[np.flatnonzero(y==c)[:256]])
    for c,owners in previous['owners'].items():
        for record in owners:
            x,y=load(record['owner'])
            for ix in np.flatnonzero(y==int(c)):
                h=digest(x[ix])
                if h in wanted_set:reference_rows[h]=(x[ix],int(c))
    if set(reference_rows)!=wanted_set:raise ValueError('Cannot restore exact prior panel')
    paired=dict(x=np.stack([reference_rows[h][0] for h in wanted]),y=np.asarray([reference_rows[h][1] for h in wanted]),hashes=wanted)
    chunks=[];used=set(excluded);counts={}
    for c in range(34):
        n=0
        owners=sorted(map(int,roles.manifest['clients']),key=lambda cid:(-roles.manifest['clients'][str(cid)]['role_class_counts']['validation'].get(str(c),0),cid))
        for cid in owners:
            if not roles.manifest['clients'][str(cid)]['role_class_counts']['validation'].get(str(c),0):continue
            x,y=load(cid);part=content_pool(x,y,{c},used,128-n)
            if len(part['y']):chunks.append(part);used.update(part['hashes']);n+=len(part['y'])
            if n>=128:break
        counts[str(c)]=n
    fresh=dict(x=np.concatenate([p['x'] for p in chunks]),y=np.concatenate([p['y'] for p in chunks]),hashes=[h for p in chunks for h in p['hashes']])
    write(out/'evaluation_panels.json',dict(paired_rows=len(paired['y']),paired_content_hashes=paired['hashes'],
        fresh_counts=counts,fresh_rows=len(fresh['y']),fresh_content_hashes=fresh['hashes'],
        fresh_overlap_previous_panels=0,missing_fresh_classes=[c for c,n in counts.items() if not n],validation_shards_read=access,
        limitations='Paired panel was already exposed; fresh tail cannot restore exhausted class coverage or pristine confirmation'))
    return dict(paired=paired,fresh_tail=fresh)


def evaluate(lock,panel,data,label,reference):
    targets=lock['case']['targets'];h=data['h'][4];fallback=data['pred'];truth=panel['y'];rows=[]
    old_collision=None
    if lock['case']['receiver']==23 and label=='paired':
        old=read(reference/'receiver_23_lock.json')['packets']
        old_gates=np.stack([normalized_dot(h,np.asarray(p['prototype'],np.float32))>=p['thresholds']['receiver_negative_only'] for p in old],axis=1)
        old_collision=old_gates.sum(1)>=2
        if old_collision.sum()!=206:raise ValueError('Prior206 collision diagnostic not restored')
    for geometry in RULES['geometries']:
        proto=[];pz=[];hz=[];sz=[];diagnostics=[]
        for p in lock['packets']:
            g=p['geometries'][geometry];center=np.asarray(g['receiver_center'],np.float32)
            score=normalized_dot(h-center,np.asarray(g['prototype_effective'],np.float32));proto.append(score)
            pz.append((score-g['background_mean'])/g['background_std'])
            head=(h-center)@np.asarray(p['weight'],np.float32)+g['affine_bias']
            hz.append((head-g['head_bg_mean'])/g['head_bg_std'])
            sub=(h-center)@np.asarray(g['subhead_weight'],np.float32)
            sz.append((sub-g['subhead_bg_mean'])/g['subhead_bg_std'])
            diagnostics.append(dict(class_id=p['class_id'],AUROC=float(roc_auc_score(truth==p['class_id'],score)),
                target_quantiles=np.quantile(score[truth==p['class_id']],[0,.05,.5,.95,1]).tolist(),
                negative_quantiles=np.quantile(score[truth!=p['class_id']],[0,.05,.5,.95,1]).tolist()))
        proto=np.stack(proto,axis=1)
        scores=dict(prototype_cosine=proto,prototype_z=np.stack(pz,axis=1),affine_head_z=np.stack(hz,axis=1),centered_subhead_z=np.stack(sz,axis=1))
        for policy in RULES['policies']:
            gated=proto>=np.asarray([p['geometries'][geometry]['thresholds'][policy] for p in lock['packets']])
            collision=gated.sum(1)>=2
            pp=choose(proto,gated,targets,fallback)
            for scorer,values in scores.items():
                pred=choose(values,gated,targets,fallback);m=metrics(pred,truth,fallback,targets)
                row=dict(geometry=geometry,policy=policy,scorer=scorer,metrics=m,
                    primary=(geometry==('centered' if len(targets)>1 else 'mean_shift') and policy=='attenuation_and_background' and scorer==('prototype_z' if len(targets)>1 else 'prototype_cosine')),
                    secondary_user_FAR_gate=m['FAR']<=.005,activated=int(gated.any(1).sum()),multiple_gate_matches=int(collision.sum()),
                    collision_accuracy=float((pred[collision]==truth[collision]).mean()) if collision.any() else None,
                    prediction_difference_from_prototype=int((pred!=pp).sum()))
                if old_collision is not None:row.update(old206_still_multiple=int((old_collision&collision).sum()),old206_accuracy=float((pred[old_collision]==truth[old_collision]).mean()))
                rows.append(row)
        write_diagnostics=diagnostics
        # Keep distributions once per geometry rather than once per scorer.
        for r in rows[-len(RULES['policies'])*len(RULES['scorers']):]:
            if r['primary']:r['score_diagnostics']=write_diagnostics
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);start=time.monotonic()
    inputs=ROOT/'audit_denice/appliance_legacy_results11/inputs'
    a.base_store=ROOT/'audit_denice/appliance_current_runtime_data_local/full100_v1/base_store'
    a.cal_store=ROOT/'audit_denice/appliance_current_runtime_data_local/full100_v1/calibration_store'
    reference=ROOT/'audit_denice/candle_v1_multiclass_prototype_20261022_v1'
    write(a.out/'protocol_before_data.json',dict(rules=RULES,script_sha256=file_sha256(__file__)))
    ckpt,source=load_input(inputs/'checkpoint_task_4_all_rounds.zip',4,19)
    roles=CleanRoleData(inputs/'roles',source_data_dir=ROOT/'audit_denice/appliance_transfer_results9/inputs/dataset')
    a.role_sha=file_sha256(inputs/'roles/role_manifest.json')
    if ckpt['config']['denice_data_roles_sha256']!=a.role_sha:raise ValueError('Role mismatch')
    classes=read(a.base_store/'base_store_manifest.json')['task_classes']['4'];locks=[]
    for case in RULES['cases']:
        lock=prepare(a,ckpt,roles,classes,case);lock['source']=source
        path=a.out/f"receiver_{case['receiver']}_lock.json";write(path,lock);locks.append((lock,file_sha256(path)))
        print('LOCK',case['receiver'],lock['status'],'anchors',lock.get('anchors'),flush=True)
    debug=read(inputs/'training_artifacts/denice_debug_history.json')
    evaluation=panels(roles,debug,reference,a.out)
    from eval_checkpoint import _make_denice_client_model
    results=[];seen=sorted(map(int,ckpt['seen_classes']))
    for lock,lock_sha in locks:
        cid=lock['case']['receiver']
        if lock['status']!='READY':results.append(lock);continue
        model,router=_make_denice_client_model(ckpt,cid,a.device);model.eval()
        before=complete_hash(model,router);all_rows={}
        for label,panel in evaluation.items():
            data=cache(model,router,panel['x'],seen,a.device)
            rows=evaluate(lock,panel,data,label,reference);all_rows[label]=rows
            primary=next(r for r in rows if r['primary'])
            print('PRIMARY',cid,label,primary['metrics']['per_target'],'FP',primary['metrics']['false_activation'],'break',primary['metrics']['breaks'],'collisions',primary['multiple_gate_matches'],flush=True)
        if before!=complete_hash(model,router) or before!=lock['receiver_hash']:raise ValueError('Receiver state changed')
        if lock_sha!=file_sha256(a.out/f'receiver_{cid}_lock.json'):raise ValueError('Threshold lock changed')
        result=dict(receiver=cid,case=lock['case'],anchors=lock['anchors'],panels=all_rows,source_unchanged=True,lock_sha256=lock_sha)
        write(a.out/f'receiver_{cid}_result.json',result);results.append(result)
        del model,router;gc.collect();torch.cuda.empty_cache()
    write(a.out/'completion.json',dict(completed=True,seconds=time.monotonic()-start,results=results,source=source,
        all_locks_before_evaluation=True,no_receiver_target_features_before_evaluation=True,no_optimizer=True,no_final_TEST=True))


if __name__=='__main__':main()
