"""Bounded, optimizer-free head/unmasking search; lock on CAL before validation."""
import argparse
from collections import Counter
import copy
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from appliance.runner import load_input
from appliance.state import complete_hash
from fed_learning.data.denice_clean_roles import CleanRoleData, file_sha256
from tools.audit_candle_v1_two_tier import partition_pool, digest, features, save_rows, write, read

RULES = dict(version='candle_head_unlock_development_v1', seed=20261022, task=4, target=24,
    cases=[dict(receiver=7, donor=11, tier='intra'), dict(receiver=65, donor=11, tier='extra')],
    CAL_source='current task4 CAL only; old construction parity, then content-hash FIT/selection split',
    validation_source='locked validation role; donor target and receiver actually trained classes through T4',
    cap_validation_per_class=256, target_recall=.95, FAR=.001, negative_break=0,
    threshold_grid='FIT positive quantiles [0,.05,.25,.5] and negative [.5,.9,1], plus abstention',
    margin_grid='FIT positive [0,.05,.5] and negative [.5,.9,1], plus zero and no cross-task margin',
    top_k=[2,3], top_gap=[0.,.005,.02,.1,2.], top_scores=['raw','weight_norm','cosine'],
    head_eta=[.25,.5,1.], no_optimizer=True, no_memory_import=True,
    no_final_TEST=True, current_CAL_selection_cannot_certify_old_risk=True,
    prior_exposure='Previously audited development source/checkpoints, not untouched confirmation')


def take(pool, ids):
    ids = np.asarray(ids, np.int64)
    return dict(x=pool['x'][ids], y=pool['y'][ids], hashes=[pool['hashes'][i] for i in ids])


def combine(pools):
    records = {}
    for pool in pools:
        for x, y, h in zip(pool['x'], pool['y'], pool['hashes']):
            records.setdefault(h, (x, int(y)))
    return dict(x=np.stack([r[0] for r in records.values()]),
                y=np.asarray([r[1] for r in records.values()]), hashes=list(records))


def development(a, cid, donor, role_sha, classes):
    r = partition_pool(a, None, role_sha, cid, classes)['construction']
    d = partition_pool(a, None, role_sha, donor, classes)['construction']
    target = RULES['target']
    pool = combine([take(r, np.flatnonzero(r['y'] != target)), take(d, np.flatnonzero(d['y'] == target))])
    sides = np.asarray([int(hashlib.sha256(('head-unlock:' + h).encode()).hexdigest()[:8], 16) % 2 for h in pool['hashes']])
    fit, selection = take(pool, np.flatnonzero(sides == 0)), take(pool, np.flatnonzero(sides == 1))
    for name, p in [('FIT', fit), ('selection', selection)]:
        if min(int((p['y'] == target).sum()), int((p['y'] != target).sum())) < 16:
            raise ValueError(f'{name} lacks diagnostic positive/negative support')
    if set(fit['hashes']) & set(selection['hashes']):
        raise ValueError('FIT/selection content overlap')
    return fit, selection


def validation(roles, cid, donor, learned, used):
    # Access only after the selection lock is written. Never read global_test_data.npz.
    pools = []
    for owner, permitted in [(cid, set(learned) - {RULES['target']}), (donor, {RULES['target']})]:
        x, y, _ = roles.client_role(owner, 'validation')
        indexes = []
        for label in sorted(permitted):
            indexes.extend(np.flatnonzero(y == label)[:RULES['cap_validation_per_class']].tolist())
        indexes = np.asarray(indexes, np.int64)
        pools.append(dict(x=x[indexes], y=y[indexes], hashes=[digest(v) for v in x[indexes]]))
    pool = combine(pools)
    if set(pool['hashes']) & used:
        raise ValueError('CAL/validation content overlap')
    return pool


@torch.no_grad()
def cache(model, router, x, seen, device):
    from fed_learning.training.denice_eval import _denice_routed_logits_with_episodes
    tasks = sorted(int(t) for t, bank in router.activation_memory.items() if len(bank))
    if RULES['task'] not in tasks:
        raise ValueError('Imported task has no native routing memory')
    h, bits, predictions, episodes, raw_best = {t: [] for t in tasks}, [], [], [], []
    active = copy.deepcopy(model.active_adapters)
    model.eval()
    try:
        for start in range(0, len(x), 256):
            batch = torch.from_numpy(x[start:start+256]).to(device)
            z, ep = _denice_routed_logits_with_episodes(model, batch, router, seen, device, inference_policy='pred_hard')
            predictions.extend(z.argmax(1).cpu().tolist()); episodes.extend(ep.tolist())
            raw_best.extend(z.max(1).values.cpu().tolist())
            bits.append(router._binarize_per_sample(model, batch))
            for t in tasks:
                model.set_active_context(t)
                h[t].append(model.penultimate_features(batch).cpu().numpy())
        h = {t: np.concatenate(values) for t, values in h.items()}
        binary = np.concatenate(bits)
        prototypes = np.stack([np.asarray(router.activation_memory[t], np.float32).mean(0) for t in tasks])
        raw_cos = np.stack([binary @ proto / np.maximum(
            np.linalg.norm(binary,axis=1)*np.linalg.norm(proto),1e-8) for proto in prototypes],axis=1)
        pred_ep = np.asarray(tasks)[raw_cos.argmax(1)]
        if not np.array_equal(pred_ep, episodes):
            raise ValueError('Cached cosine routing differs from native router')
        w = (model.fc2.weight * model.weight_masks['fc2'].to(model.fc2.weight)).detach().cpu().numpy().copy()
        b = (model.fc2.bias * model.bias_masks['fc2'].to(model.fc2.bias)).detach().cpu().numpy().copy()
        logits = {t: values @ w.T + b for t, values in h.items()}
        pred = np.asarray(predictions); ep = np.asarray(episodes)
        local_dot = np.asarray([h[int(t)][i] @ w[int(c)] for i,(t,c) in enumerate(zip(ep,pred))])
        local_h_norm = np.asarray([np.linalg.norm(h[int(t)][i]) for i,t in enumerate(ep)])
        local_w_norm = np.linalg.norm(w[pred], axis=1)
        return dict(h=h, w=w, b=b, logits=logits, tasks=tasks, raw_cos=raw_cos, bits=binary,
            pred=pred, episode=ep, local_raw=np.asarray(raw_best),
            local_weight_norm=np.asarray(raw_best)/np.maximum(local_w_norm,1e-8),
            local_cosine=local_dot/np.maximum(local_h_norm*local_w_norm,1e-8),
            episode_classes={int(t):sorted(set(map(int,cs)) & set(seen)) for t,cs in router.episode_classes.items()})
    finally:
        model.active_adapters = active


def normalized_dot(h, p):
    return h @ p / np.maximum(np.linalg.norm(h,axis=1)*np.linalg.norm(p),1e-8)


def jaccard(h, prototype, threshold):
    bits, p = h > threshold, prototype > threshold
    return np.logical_and(bits,p).sum(1)/np.maximum(np.logical_or(bits,p).sum(1),1)


def scores(data, head, mode):
    t, c = RULES['task'], RULES['target']
    h = data['h'][t]
    dot = h @ head['w']; z = dot + head['b']
    matrix = data['logits'][t].copy(); matrix[:,c] = z
    if mode == 'raw':
        target, local = z, data['local_raw']
    elif mode == 'weight_norm':
        norms = np.linalg.norm(data['w'],axis=1); norms[c] = np.linalg.norm(head['w'])
        matrix /= np.maximum(norms,1e-8)[None,:]
        target, local = z/max(np.linalg.norm(head['w']),1e-8), data['local_weight_norm']
    else:
        norms = np.linalg.norm(data['w'],axis=1); norms[c] = np.linalg.norm(head['w'])
        matrix = h @ data['w'].T
        matrix[:,c] = dot
        matrix /= np.maximum(np.linalg.norm(h,axis=1)[:,None]*norms[None,:],1e-8)
        target, local = normalized_dot(h,head['w']), data['local_cosine']
    allowed = sorted(set(data['episode_classes'].get(t,[])) | {c})
    within = np.asarray(allowed)[matrix[:,allowed].argmax(1)] == c
    return target-local, within, matrix


def stats(pred, truth, baseline):
    target = RULES['target']; pos = truth == target; neg = ~pos
    rescued = (baseline != truth) & (pred == truth); broken = (baseline == truth) & (pred != truth)
    return dict(rows=len(truth), positives=int(pos.sum()), negatives=int(neg.sum()),
        recall=float((pred[pos]==target).mean()) if pos.any() else None,
        accuracy=float((pred==truth).mean()), FAR=float((pred[neg]==target).mean()) if neg.any() else None,
        rescue=int(rescued.sum()), breaks=int(broken.sum()), negative_break=int((broken & neg).sum()),
        net=int(rescued.sum()-broken.sum()), false_activation=int((pred[neg]==target).sum()),
        per_negative_class={str(int(c)):dict(rows=int((truth==c).sum()),
            false_activation=int(((truth==c)&(pred==target)).sum()), breaks=int(((truth==c)&broken).sum()))
            for c in np.unique(truth[neg])})


def head_recipes(receiver, donor, fit, donor_h):
    c,t = RULES['target'], RULES['task']
    wr,br = fit['w'][c],float(fit['b'][c])
    wd = (donor.fc2.weight[c]*donor.weight_masks['fc2'][c].to(donor.fc2.weight)).detach().cpu().numpy().copy()
    bd = float((donor.fc2.bias[c]*donor.bias_masks['fc2'][c].to(donor.fc2.bias)).detach().cpu())
    heads = {'availability':dict(w=wr.copy(),b=br,kind='control')}
    for eta in RULES['head_eta']:
        heads[f'direct_eta_{eta}']=dict(w=(1-eta)*wr+eta*wd,b=(1-eta)*br+eta*bd,kind='donor_weights')
    current = [v for v in fit['episode_classes'].get(t,[]) if v != c]
    ratio = float(np.median(np.linalg.norm(fit['w'][current],axis=1))/max(np.linalg.norm(wd),1e-8))
    heads['norm_matched'] = dict(w=wd*ratio,b=bd*ratio,kind='donor_weights_norm')
    heads['norm_and_bias_matched'] = dict(w=wd*ratio,b=float(np.median(fit['b'][current])),kind='donor_weights_norm_bias')
    # Paired moments only on donor-positive FIT, never validation.
    hr, hd = fit['h'][t][fit['truth']==c], donor_h
    scale = np.clip(hd.std(0)/np.maximum(hr.std(0),1e-3),.25,4.)
    offset = hd.mean(0)-scale*hr.mean(0)
    heads['diagonal_moment_transport']=dict(w=wd*scale,b=bd+float(wd@offset),kind='receiver_aligned_donor_weights')
    return heads


def thresholds(values, labels, kind):
    pos,neg = values[labels==RULES['target']],values[labels!=RULES['target']]
    qs_p = [0,.05,.25,.5] if kind=='gate' else [0,.05,.5]
    return sorted(set([float(v) for v in np.quantile(pos,qs_p)] +
                      [float(v) for v in np.quantile(neg,[.5,.9,1.])] + ([0.] if kind=='margin' else [])))


def gate_values(data, head, prototypes):
    h=data['h'][RULES['task']]
    return dict(donor_proto=normalized_dot(h,prototypes['donor']),
        receiver_proto=normalized_dot(h,prototypes['receiver']), weight_cos=normalized_dot(h,head['w']),
        donor_FC1_binary=jaccard(h,prototypes['donor'],prototypes['threshold']),
        receiver_FC1_binary=jaccard(h,prototypes['receiver'],prototypes['threshold']))


def candidate_prediction(data, head, prototypes, spec):
    c,t=RULES['target'],RULES['task']
    if spec['family']=='baseline':return data['pred'].copy()
    mode=spec.get('score','raw')
    margin,within,matrix=scores(data,head,mode)
    if spec['family']=='native_head':
        result=data['pred'].copy(); ix=data['episode']==t
        allowed=sorted(set(data['episode_classes'].get(t,[]))|{c})
        result[ix]=np.asarray(allowed)[matrix[ix][:,allowed].argmax(1)]
        return result
    if spec['family'].startswith('top'):
        result=candidate_prediction(data,head,prototypes,dict(family='native_head',score=mode))
        order=np.argsort(-data['raw_cos'],axis=1,kind='stable')
        gap=np.take_along_axis(data['raw_cos'],order[:,:1],1)[:,0]-np.take_along_axis(data['raw_cos'],order[:,1:2],1)[:,0]
        for n in np.flatnonzero(gap<=spec['gap']):
            choices=[]
            for index in order[n,:spec['k']]:
                task=data['tasks'][int(index)]; allowed=sorted(set(data['episode_classes'].get(task,[]))|({c} if task==t else set()))
                if not allowed:continue
                if task==t:z=matrix[n]
                elif mode=='raw':z=data['logits'][task][n]
                elif mode=='weight_norm':z=data['logits'][task][n]/np.maximum(np.linalg.norm(data['w'],axis=1),1e-8)
                else:z=(data['h'][task][n]@data['w'].T)/np.maximum(np.linalg.norm(data['h'][task][n])*np.linalg.norm(data['w'],axis=1),1e-8)
                label=allowed[int(np.argmax(z[allowed]))]; choices.append((float(z[label]),-int(task),label))
            if choices:result[n]=max(choices)[2]
        return result
    gate=gate_values(data,head,prototypes)[spec['gate']]
    mask=(gate>=spec['tau']) & within
    if spec['gamma'] is not None:mask &= margin>=spec['gamma']
    result=data['pred'].copy();result[mask]=c
    return result


def eligible(m):
    return m['FAR'] is not None and m['FAR']<=RULES['FAR'] and m['negative_break']==0


def search(fit, selection, heads, prototypes):
    trials=[]
    def add(head_name,spec):
        pred=candidate_prediction(selection,heads[head_name],prototypes,spec)
        m=stats(pred,selection['truth'],selection['pred'])
        trial=dict(id=len(trials),head=head_name,spec=spec,selection=m,
                   negative_gate=eligible(m),full_selection_gate=eligible(m) and m['recall']>=RULES['target_recall'] and m['net']>0)
        trials.append(trial)
    add('availability',dict(family='baseline'))
    for name,head in heads.items():
        for mode in RULES['top_scores']:
            add(name,dict(family='native_head',score=mode))
            for k in RULES['top_k']:
                for gap in RULES['top_gap']:
                    add(name,dict(family=f'top{k}_energy',k=k,gap=gap,score=mode))
            margin,_,_=scores(fit,head,mode)
            gammas=[None]+thresholds(margin,fit['truth'],'margin')
            for gate,values in gate_values(fit,head,prototypes).items():
                for tau in thresholds(values,fit['truth'],'gate'):
                    for gamma in gammas:
                        add(name,dict(family='targeted_'+gate,gate=gate,tau=tau,gamma=gamma,score=mode))
    def key(row):
        m=row['selection']
        return (row['negative_gate'],row['full_selection_gate'],m['net'],m['recall'],-m['FAR'],-m['breaks'],-row['id'])
    families={}
    for trial in trials:
        family=trial['spec']['family']
        if family not in families or key(trial)>key(families[family]):families[family]=trial
    winner=max(trials,key=key)
    return trials,families,winner


def distributions(cache_data, labels, head, prototypes):
    c=RULES['target']; results={}
    for name,values in gate_values(cache_data,head,prototypes).items():
        results[name]=dict(AUROC=float(roc_auc_score(labels==c,values)),
            target_quantiles=np.quantile(values[labels==c],[0,.05,.5,.95,1]).tolist(),
            negative_quantiles=np.quantile(values[labels!=c],[0,.05,.5,.95,1]).tolist())
    order=np.argsort(-cache_data['raw_cos'],axis=1,kind='stable')
    target_index=cache_data['tasks'].index(RULES['task'])
    ranks=np.argmax(order==target_index,axis=1)+1
    positive=ranks[labels==c]
    results['task_rank']=dict(target_rows=len(positive),counts=dict(Counter(map(str,positive))),
        top1=float((positive<=1).mean()),top2=float((positive<=2).mean()),top3=float((positive<=3).mean()),
        ranking='stable task-ID tie break; raw cosine, not softmax confidence')
    return results


def export_value(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,dict):return {k:export_value(v) for k,v in value.items()}
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--inputs',type=Path,default=ROOT/'audit_denice/appliance_legacy_results11/inputs')
    p.add_argument('--roles',type=Path,default=ROOT/'audit_denice/appliance_legacy_results11/inputs/roles')
    p.add_argument('--data',type=Path,default=ROOT/'audit_denice/appliance_transfer_results9/inputs/dataset')
    p.add_argument('--store',type=Path,default=ROOT/'audit_denice/appliance_current_runtime_data_local/full100_v1/calibration_store')
    p.add_argument('--device',default='cuda')
    a=p.parse_args();a.task=RULES['task']
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(RULES['seed'])
    write(a.out/'protocol_before_data.json',dict(rules=RULES,script_sha256=file_sha256(__file__)))
    write(a.out/'completion.json',dict(completed=False))
    started=time.monotonic()
    from eval_checkpoint import _make_denice_client_model
    checkpoint=a.inputs/'checkpoint_task_4_all_rounds.zip'
    print('Loading original V1 terminal Task4',flush=True)
    ckpt,hashes=load_input(checkpoint,4,19)
    roles=CleanRoleData(a.roles,source_data_dir=a.data);role_sha=file_sha256(a.roles/'role_manifest.json')
    if ckpt['config']['denice_data_roles_sha256']!=role_sha or ckpt['config'].get('denice_cl_method')!='legacy':raise ValueError('Wrong baseline/roles')
    store=read(a.store/'calibration_store_manifest.json');classes=store['task_classes']['4'];seen=sorted(map(int,ckpt['seen_classes']))
    summary=[]
    debug=read(a.inputs/'training_artifacts/denice_debug_history.json')
    for case in RULES['cases']:
        i,j,c=case['receiver'],case['donor'],RULES['target']
        out=a.out/f'receiver_{i}';out.mkdir()
        print(f'Case receiver={i}, donor={j}: CAL FIT/selection only',flush=True)
        fit,sel=development(a,i,j,role_sha,classes)
        receiver,router=_make_denice_client_model(ckpt,i,a.device)
        donor,dr=_make_denice_client_model(ckpt,j,a.device)
        receiver.eval();donor.eval();original=(complete_hash(receiver,router),complete_hash(donor,dr))
        fc=cache(receiver,router,fit['x'],seen,a.device);fc['truth']=fit['y']
        sc=cache(receiver,router,sel['x'],seen,a.device);sc['truth']=sel['y']
        dh,_=features(donor,dr,fit['x'][fit['y']==c],4,a.device)
        proto=dict(donor=dh.mean(0),receiver=fc['h'][4][fit['y']==c].mean(0),
            threshold=fc['h'][4][fit['y']!=c].mean(0)+fc['h'][4][fit['y']!=c].std(0))
        heads=head_recipes(receiver,donor,fc,dh)
        trials,families,winner=search(fc,sc,heads,proto)
        lock=dict(case=case,source_hashes=hashes,FIT_counts=dict(Counter(map(str,fit['y']))),
            selection_counts=dict(Counter(map(str,sel['y']))),winner=winner,family_winners=families,
            validation_opened=False,selection_trials=len(trials),rules=RULES)
        write(out/'selection_lock.json',lock);locked_sha=file_sha256(out/'selection_lock.json')
        save_rows(out/'selection_trials.csv',trials)
        write(out/'selection_diagnostics.json',distributions(sc,sel['y'],heads['direct_eta_1.0'],proto))
        print('Selection locked',i,'trials',len(trials),'winner',winner['head'],winner['spec'],'metrics',winner['selection'],flush=True)
        learned=set()
        for event in debug:
            if event.get('type')=='task_start' and int(event['task'])<=4:
                record=event['client_data'].get(str(i))
                if record:learned.update(map(int,record['labels']))
        used=set(fit['hashes'])|set(sel['hashes'])
        val=validation(roles,i,j,learned,used)
        vc=cache(receiver,router,val['x'],seen,a.device);vc['truth']=val['y']
        results=[]
        for family,chosen in families.items():
            head=heads[chosen['head']]
            prediction=candidate_prediction(vc,head,proto,chosen['spec'])
            results.append(dict(family=family,head=chosen['head'],spec=chosen['spec'],
                primary=chosen['id']==winner['id'],selection=chosen['selection'],
                validation=stats(prediction,val['y'],vc['pred'])))
        primary_prediction=candidate_prediction(vc,heads[winner['head']],proto,winner['spec'])
        primary=stats(primary_prediction,val['y'],vc['pred'])
        # Mechanism control: identical chosen gate, but no donor weight changes.
        original_row_prediction=candidate_prediction(vc,heads['availability'],proto,winner['spec'])
        mechanism=stats(original_row_prediction,val['y'],vc['pred'])
        score_mode=winner['spec'].get('score','raw')
        _,_,class_scores=scores(vc,heads[winner['head']],score_mode)
        allowed=sorted(set(vc['episode_classes'].get(4,[]))|{c})
        forced=np.asarray(allowed)[class_scores[:,allowed].argmax(1)]
        before=vc['pred']==val['y'];after=primary_prediction==val['y']
        if int(after.sum()-before.sum())!=primary['rescue']-primary['breaks']:raise ValueError('Accounting mismatch')
        unchanged=original==(complete_hash(receiver,router),complete_hash(donor,dr))
        if not unchanged or file_sha256(out/'selection_lock.json')!=locked_sha:raise ValueError('Reference/lock changed')
        write(out/'selected_packet.json',export_value(dict(class_id=c,context_id=4,head=heads[winner['head']],
            prototypes=proto,inference_policy=winner['spec'],scope='development current CAL only; no installation authorization')))
        result=dict(case=case,selected=winner,validation=primary,original_weights_same_gate=mechanism,
            variants=results,validation_diagnostics=distributions(vc,val['y'],heads['direct_eta_1.0'],proto),
            selected_forced_task_recall_DIAGNOSTIC_ONLY=float((forced[val['y']==c]==c).mean()),
            validation_counts=dict(Counter(map(str,val['y']))),old_class_negative_rows=int((val['y']<24).sum()),
            source_reference_unchanged=unchanged,selection_lock_sha256=locked_sha,
            validation_selection_content_overlap=0,production_accepted=False,no_optimizer=True,
            no_donor_memory_import=True,raw_samples_transmitted=False,
            limitations=['Offline single-process diagnostic; network protocol not implemented',
                'Donor-origin target positives, not receiver target domain',
                'Receiver validation includes historical classes for evaluation only, never fitting',
                'Previously exposed development source; not untouched confirmation',
                'Receiver-space prototype requires receiver function at donor; setup bytes not zero',
                'Moment alignment modifies head only and uses current paired FIT features'])
        write(out/'result.json',result)
        summary.append(result)
        print('VALIDATION',i,json.dumps(dict(primary=primary,original_weights_same_gate=mechanism,
            rank=result['validation_diagnostics']['task_rank'])),flush=True)
        del receiver,router,donor,dr,fc,sc,vc
        gc.collect();torch.cuda.empty_cache()
    write(a.out/'completion.json',dict(completed=True,seconds=time.monotonic()-started,rules=RULES,
        source_hashes=hashes,results=summary,backbone_training=False,head_training=False,
        final_TEST_read=False,source_checkpoint_unchanged=True))
    print('Completed:',a.out/'completion.json',flush=True)


if __name__=='__main__':main()
