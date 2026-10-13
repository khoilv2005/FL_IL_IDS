"""One locked linear-guard development experiment: 65 <- 88 / class 20.

Chronological BASE-summary reconstruction is a simulation, not proof of old
production persistence. Current CAL selects thresholds. Untouched validation
tails evaluate protection once; no test, install, retrain or historical CAL.
"""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from appliance.base_sketch_shield import CurrentBaseSketchShield
from appliance.closure import effective_linear
from appliance.config import Protocol, Rejected
from appliance.current_base_data import CurrentBaseData
from appliance.current_calibration_data import CurrentCalibrationData
from appliance.discriminative_guard import (CurrentBaseSketchMoments, LinearGuard, RIDGE,
    fit_linear_guard, read_negative_packet)
from appliance.distributed_calibration import (CalibrationSession, GuardGrid, LocalCalibrationEndpoint,
    select_distributed_guard, current_acceptance)
from appliance.imported_route import ROUTE_RULES, stratified_roles
from appliance.portable_route import SharedSketch, ProtectedRoute, fit_support_cosine_floor, transitions
from appliance.receiver_aware_discovery import maturity_precheck, head_offer
from appliance.selector import lookup
from appliance.stable_head import guard_function, stable_signals
from appliance.state import complete_hash, digest, write_json
from appliance.transport import Transport
from eval_checkpoint import _make_denice_client_model
from fed_learning.data.denice_clean_roles import CleanRoleData, file_sha256
from fed_learning.training.checkpoint_state import snapshot_denice_state
from tools.audit_appliance_provenance_acceptance import original_model

PAIR = (65, 88, 20)


def content_hashes(x):
    if not len(x):return []
    values=np.array(x,dtype='<f4',copy=True).reshape(len(x),-1)
    values[values==0]=0  # Canonicalize signed zero, no rounding.
    return [hashlib.sha256(row.tobytes()).hexdigest() for row in values]


def run(a):
    a.out.mkdir(parents=True, exist_ok=False)
    write_json(a.out/'completion.json', dict(completed=False))
    write_json(a.out/'protocol_before_data.json', dict(receiver=65, donor=88, class_id=20,
        dimension=16, ridge=RIDGE, model='balanced linear ridge; no MLP or parameter sweep',
        negative_fit='peer-owned BASE moments only, all supported classes except target',
        negative_class_weight='equal class mass, within class weighted by usable row counts',
        linear_score_floor=0., current_CAL_recall_gate=.95, FAR_budget=.001, max_break=0,
        validation_rule='skip first 256 rows/class previously inspected; take next 512',
        content_overlap='report exact BASE/FIT overlap; no post-score filtering or independent-test claim',
        validation_used_for_threshold_selection=False, production_enabled=False,
        retrospective_summary_simulation=True, historical_raw_CAL_opened=False,
        historical_summary_not_present_in_original_checkpoint=True, test_opened=False))
    ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    cid, donor, c = PAIR
    task, round_id = int(ckpt['task']), int(ckpt['final_round_id'])
    if task != 3 or ckpt['config']['denice_cl_method'] != 'legacy':
        raise Rejected('EXPECTED_LEGACY_TASK3_FIXTURE')
    graph = next(g for g in json.loads(a.graphs.read_text()) if g['task']==task and g['round']==round_id)
    edge = lookup(graph['alpha_debug'], cid)
    peers = {int(i):float(w) for i,w in zip(edge['group_ids'], edge['alphas']) if i!=cid and w>0}
    if donor not in peers:
        raise Rejected('DONOR_NOT_LIVE_PEER')
    state = lookup(ckpt['client_algorithm_states'], cid); state = state.get('denice', state)
    entry = lookup(state['appliance_guarded_head_entries'], c)
    if entry is None or entry['task'] != task:
        raise Rejected('EXPECTED_ORIGINAL_PAIR_PATCH')
    route = ProtectedRoute.from_packet(entry['packet'])
    shadow_route = ProtectedRoute.from_packet(entry['packet'])
    if route.metadata['donor'] != donor:
        raise Rejected('FIXTURE_DONOR_CHANGED')
    m = route.metadata['signature']
    sketch = SharedSketch(tuple(m['input_shape']), 16, m['preprocessing_sha256'], m['seed'])
    role_sha = ckpt['config']['denice_data_roles_sha256']
    model, router = original_model(ckpt, cid)
    original = complete_hash(model, router)
    dm, _ = original_model(ckpt, donor)
    pre = maturity_precheck(model, router, head_offer(dm, c, digest(dm.state_dict())), task, ckpt['seen_classes'])
    if not pre['eligible']:
        raise Rejected(pre['reason'])
    refs = pre['reference_classes']
    own = CurrentBaseSketchShield.restore(entry['shield_at_install'])
    required = entry['required_old_classes']
    w, b = effective_linear(dm, 'fc2')
    if not np.array_equal(w[c].numpy(), route.head_weight) or float(b[c]) != route.head_bias:
        raise Rejected('DONOR_HEAD_CHANGED_FROM_FIXTURE')
    controls = {}
    wire = Transport(a.out/'application_wire.jsonl', [(cid,donor),(donor,cid)] + [(p,cid) for p in peers],
        Protocol(max_incoming_bytes=64*1024*1024, max_outgoing_bytes=64*1024*1024))
    # Measure and reconstruct the receiver capsule, including router/masks/state.
    alg = snapshot_denice_state(model, router)
    if alg['context_detector'].get('reference_input_memory'):
        raise Rejected('RAW_REFERENCE_IN_CAPSULE')
    capsule = dict(config=copy.deepcopy(ckpt['config']), task=task, round=round_id,
        client_model_states={cid:{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}},
        client_algorithm_states={cid:{'denice':alg}})
    buffer = io.BytesIO(); torch.save(capsule, buffer)
    capsule_packet = wire.send(cid, donor, 'receiver_function_capsule', buffer.getvalue())
    replica, replica_router = _make_denice_client_model(torch.load(io.BytesIO(capsule_packet), map_location='cpu', weights_only=False), cid, 'cpu')
    controls['capsule_restores_identical_receiver'] = complete_hash(replica, replica_router) == original
    del replica, replica_router, capsule, buffer
    summaries, logs, packets, sources = [], {}, [], []
    training_content=set()  # Audit-only metadata, never guard state or wire.
    # Explicit chronological reconstruction while each task is current. Only
    # checkpoint-owned support is exported; inherited binary memory is ignored.
    for p in sorted(peers):
        ps = lookup(ckpt['client_algorithm_states'], p); ps = ps.get('denice', ps)
        owned = CurrentBaseSketchShield.restore(ps['appliance_base_sketch_shield_state'])
        negative_classes = sorted(int(k) for k in owned.memory.entries if int(k)!=c)
        if not negative_classes:
            continue
        view = CurrentBaseData(a.base_store, p, 0, role_sha)
        summary = CurrentBaseSketchMoments(view, sketch)
        for t in range(task+1):
            if t:
                view.advance(t)
            summary.observe_current(view)
            # Fingerprint only while the BASE partition is CURRENT. Do not
            # reopen a historical partition after advancing the runtime view.
            pool=view.current_pool(p,'base',view.store['task_classes'][str(t)])
            selected_rows=np.isin(pool['y'],negative_classes)
            training_content.update(content_hashes(pool['X'][selected_rows]))
            del pool,selected_rows
        try:
            view.current_pool(p, 'base', view.store['task_classes']['0'])
            controls[f'peer_{p}_historical_BASE_runtime_rejected'] = False
        except Rejected:
            controls[f'peer_{p}_historical_BASE_runtime_rejected'] = True
        version = digest(ckpt['client_model_states'][p])
        packet = summary.packet(negative_classes, cid, version)
        raw = wire.send(p, cid, 'peer_BASE_sketch_moments_and_provenance', packet)
        checked = read_negative_packet(raw, p, cid, peers, version, sketch, role_sha, task)
        # Fixed source snapshot/class authority, not a classifier ownership receipt.
        receipt = dict(sender=p, receiver=cid, task=task, round=round_id, kind='protection_only',
            alpha=peers[p], classes=negative_classes, source_version=version,
            packet_sha256=hashlib.sha256(raw).hexdigest(), receiver_version=original,
            claims_CAL=False, claims_classifier_ownership=False, simulated=True)
        wire.send(p, cid, 'negative_support_receipt', receipt)
        write_json(a.out/f'peer_{p}_BASE_moments.json', checked)
        (a.out/f'peer_{p}_BASE_moments.bin.gz').write_bytes(raw)
        summaries.append(checked); packets.append((raw,p,version)); sources.append((p,owned))
        logs[str(p)] = view.access_log
        print(f'Linear guard: peer={p}, summarized classes={negative_classes}, packet={len(raw)} bytes', flush=True)
    write_json(a.out/'chronological_BASE_access.json', logs)
    views = {i:CurrentCalibrationData(a.calibration_store, i, task, role_sha) for i in (cid,donor)}
    classes = views[cid].store['task_classes'][str(task)]
    pools = {i:v.current_pool(i, 'calibration', classes) for i,v in views.items()}
    splits = {i:stratified_roles(p, ROUTE_RULES['seed']+i) for i,p in pools.items()}
    pos = splits[donor]['fit']; pos = pos[pools[donor]['y'][pos]==c]
    z, valid = sketch.features(pools[donor]['X'][pos])
    training_content.update(content_hashes(pools[donor]['X'][pos]))
    binding = dict(receiver=cid, donor=donor, class_id=c, task=task, round=round_id,
        authorized_sources=sorted(p['owner'] for p in summaries),
        sketch=sketch.manifest(), role_sha=role_sha, receiver_function=original,
        patch_head=digest(dict(weight=route.head_weight, bias=route.head_bias)),
        fixed_margin_reference=refs, owned_veto=digest(own.state()), required_old_classes=required,
        positive_FIT_rows_sha256=hashlib.sha256(pools[donor]['rows'][pos].astype('<i8').tobytes()).hexdigest(),
        code_sha256=file_sha256(Path('appliance/discriminative_guard.py')))
    guard = fit_linear_guard(z, valid, summaries, binding)
    write_json(a.out/'linear_guard_before_SELECTION.json', guard.state)
    guard_packet = json.dumps(guard.state, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    # Receiver-fitted detector travels to donor for private CAL execution, then
    # the complete committed-capability representation returns to receiver.
    wire.send(cid, donor, 'linear_detector_and_function_binding', guard_packet)
    floor = fit_support_cosine_floor(route.metadata['support'])
    def signals(x):
        sig = stable_signals(model, router, x, ckpt['seen_classes'], route, refs, a.batch_size, 'cpu')
        score, okay = guard.score(x, sketch, binding)
        sig['signature_valid'] &= okay & (sig['signature_score'] > floor) & ~own.veto(x, required)
        sig['signature_score'] = score
        sig['local_confidence'] = np.zeros(len(x), np.float64)
        return sig
    def session(stage):
        return CalibrationSession(cid, donor, c, task, round_id, tuple(classes), original,
            digest(dict(linear=guard.state, base_guard=guard_function(model,route,refs))), role_sha,
            f'linear-only-{stage}')
    def endpoints(s, holdout=False):
        result={}
        for i,pool in pools.items():
            private={}
            for role in ('fit','selection','holdout'):
                ix = splits[i][role] if role!='holdout' or holdout else np.zeros(0,np.int64)
                sig = signals(pool['X'][ix]) if len(ix) else dict(signature_score=np.zeros(0),
                    signature_valid=np.zeros(0,bool), margin=np.zeros(0), local_confidence=np.zeros(0), local_pred=np.zeros(0,np.int64))
                private[role]=dict(y=pool['y'][ix], row_id=pool['rows'][ix], signals=sig)
            result[i]=LocalCalibrationEndpoint(i,s,private)
        return result
    s=session('selection'); ep=endpoints(s)
    quantiles=[ep[i].quantiles() for i in (cid,donor)]
    for q in quantiles:
        if q['client']==donor:
            wire.send(donor,cid,'CAL_SELECTION_quantiles',q)
    proposed=GuardGrid.from_quantiles(s,quantiles,0.)
    grid=GuardGrid(s,proposed.taus,proposed.gammas,[1.])
    rpacket, dpacket=ep[cid].count_packet(grid), ep[donor].count_packet(grid)
    wire.send(donor,cid,'CAL_SELECTION_counts',dpacket)
    selected=select_distributed_guard(s,grid,rpacket,dpacket)
    # tau below is the linear score threshold, separate from signature cosine.
    s=session('holdout'); final=GuardGrid(s,[selected['tau']],[selected['gamma']],[1.])
    declaration=dict(binding=binding, linear_guard=guard.state['state_digest'], selected=selected,
        policy='experimental_discriminative_linear_guard_v1',
        activation='valid sketch AND cosine>FIT floor AND no owned veto AND tanh(linear)>tau AND patch_margin>gamma',
        cosine_floor=floor, session=s.manifest(), production_enabled=False,
        old_CUDA_certificate_revalidated=False, peer_negative_budget=.001, max_break=0,
        thresholds_locked_before_HOLDOUT_and_validation=True)
    write_json(a.out/'threshold_lock_before_HOLDOUT.json',declaration)
    wire.send(cid,donor,'locked_threshold_declaration',declaration)
    ep=endpoints(s,True)
    for e in ep.values():e.lock_guard(final)
    dp=ep[donor].count_packet(final,'holdout')
    wire.send(donor,cid,'CAL_HOLDOUT_counts',dp)
    acceptance=current_acceptance(s,final,ep[cid].count_packet(final,'holdout'),dp)
    # These bytes include head, signature, detector, threshold/version/metadata.
    capability=dict(head_and_signature_packet=route.packet().hex(), detector=json.loads(guard_packet), declaration=declaration)
    wire.send(donor,cid,'complete_experimental_capability',capability)
    write_json(a.out/'current_CAL_result.json',dict(selection=selected,acceptance=acceptance))
    print(f'Linear guard CAL: {acceptance}',flush=True)
    check_x=pools[donor]['X'][pos][:32]
    score=guard.score(check_x,sketch,binding)[0]
    restored=LinearGuard(json.loads(guard_packet))
    controls['save_restore_exact']=np.array_equal(score,restored.score(check_x,sketch,binding)[0])
    restored_head=ProtectedRoute.from_packet(bytes.fromhex(capability['head_and_signature_packet']))
    controls['capability_head_restores_exact']=np.array_equal(restored_head.head_weight,route.head_weight) and restored_head.head_bias==route.head_bias
    bad=copy.deepcopy(guard.state); bad['weight'][0]+=1
    try: LinearGuard(bad); controls['coefficient_tamper_rejected']=False
    except Rejected: controls['coefficient_tamper_rejected']=True
    changed=dict(binding,receiver_function='wrong')
    try: guard.score(check_x,sketch,changed); controls['function_version_change_rejected']=False
    except Rejected: controls['function_version_change_rejected']=True
    invalid=check_x.copy();invalid[0]=np.nan
    try: guard.score(invalid,sketch,binding); controls['nonfinite_input_rejected']=False
    except Rejected: controls['nonfinite_input_rejected']=True
    raw,p,version=packets[0]
    try:
        read_negative_packet(raw,p,cid,{},version,sketch,role_sha,task)
        controls['outside_graph_source_rejected']=False
    except Rejected: controls['outside_graph_source_rejected']=True
    try:
        read_negative_packet(raw,p,cid,peers,'wrong',sketch,role_sha,task)
        controls['negative_source_version_change_rejected']=False
    except Rejected: controls['negative_source_version_change_rejected']=True
    try:
        read_negative_packet(raw,p,cid,peers,version,sketch,'wrong',task)
        controls['negative_role_change_rejected']=False
    except Rejected: controls['negative_role_change_rejected']=True
    try:
        guard.score(check_x,SharedSketch(sketch.input_shape,32,sketch.preprocessing_sha256),binding)
        controls['feature_space_change_rejected']=False
    except Rejected: controls['feature_space_change_rejected']=True
    controls['label_blind_inputs']=np.array_equal(score,guard.score(check_x.copy(),sketch,binding)[0])
    controls['CAL_roles_disjoint']=all(not (set(pools[i]['rows'][splits[i]['fit']]) & set(pools[i]['rows'][splits[i]['holdout']]))
        and not (set(pools[i]['rows'][splits[i]['selection']]) & set(pools[i]['rows'][splits[i]['holdout']])) for i in pools)
    roles=CleanRoleData(a.roles,source_data_dir=a.data)
    panels=[('receiver_validation_tail',cid,set(ckpt['seen_classes'])),('donor_positive_validation_tail',donor,{c})]
    panels += [(f'peer_{p}_negative_validation_tail',p,{int(k) for k,e in owned.memory.entries.items() if e['task']<task}) for p,owned in sources]
    results=[]; positive_scores=[]; negative_scores=[]
    for name,owner,scope in panels:
        x,y,ids=roles.client_role(owner,'validation')
        ix=np.concatenate([np.flatnonzero(y==k)[256:768] for k in sorted(scope)])
        x,y,ids=x[ix],y[ix],ids[ix]
        overlap=sum(h in training_content for h in content_hashes(x))
        write_json(a.out/f'{name}_panel_lock.json',dict(owner=owner,rows=len(y), row_ids=ids,
            rows_sha256=hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest(),role='validation',
            excluded_first_256_per_class=True,guard_digest=guard.state['state_digest'],threshold_digest=digest(declaration),
            predictions_opened=False))
        if not len(y):
            results.append(dict(pool=name,owner=owner,rows=0,passed=False,reason='no unused validation rows'));continue
        sig=signals(x)
        hit=sig['signature_valid'] & (sig['signature_score']>selected['tau']) & (sig['margin']>selected['gamma'])
        old=stable_signals(model,router,x,ckpt['seen_classes'],shadow_route,refs,a.batch_size,'cpu')
        old_hit=(old['signature_valid'] & (old['signature_score']>shadow_route.metadata['tau'])
            & (old['margin']>shadow_route.metadata['gamma']) & ~own.veto(x,required))
        if 'positive_validation' in name:positive_scores.extend(sig['signature_score'].tolist())
        if 'negative_validation' in name:negative_scores.extend(sig['signature_score'].tolist())
        pred=np.where(hit,c,sig['local_pred'])
        trans=transitions(y,sig['local_pred'],pred)
        per_class={str(k):dict(rows=int((y==k).sum()),false_activation=int((hit&(y==k)&(y!=c)).sum()),
            activations=int((hit&(y==k)).sum()), original_activations=int((old_hit&(y==k)).sum()),
            linear_score_quantiles=np.quantile(sig['signature_score'][y==k],[0,.05,.5,.95,1]).tolist()) for k in sorted(set(y))}
        false=int((hit&(y!=c)).sum()); neg=int((y!=c).sum())
        passed=(false/max(neg,1)<=.001 and trans['break_count']==0 and
            all(v['false_activation']/v['rows']<=.001 for v in per_class.values()))
        results.append(dict(pool=name,owner=owner,rows=len(y),target_rows=int((y==c).sum()),
            original_route_activations=int(old_hit.sum()),original_route_false_activations=int((old_hit&(y!=c)).sum()),
            activations=int(hit.sum()),false_activations=false,negative_rows=neg,far=false/max(neg,1),
            target_recall=float(hit[y==c].mean()) if np.any(y==c) else None,
            per_class=per_class,transitions=trans,passed_negative_gate=passed,
            declared_classes=sorted(scope),missing_tail_classes=sorted(scope-set(map(int,y))),
            exact_content_overlap_with_guard_training=overlap,
            role_row_disjoint_from_guard_training=True,independent_content_holdout_claim=False,
            validation_not_CAL_certificate=True))
        print(f'Linear guard {name}: rows={len(y)}, activations={hit.sum()}, false={false}, break={trans["break_count"]}',flush=True)
    controls['receiver_unchanged']=complete_hash(model,router)==original
    controls['no_individual_negative_rows_in_packets']=all(p['retained_examples']==p['retained_individual_features']==0 for p in summaries)
    if not all(controls.values()):raise AssertionError(controls)
    negatives=[r for r in results if 'negative_validation' in r['pool']]
    negative_pass=bool(negatives) and all(r.get('passed_negative_gate',False) for r in negatives)
    from sklearn.metrics import roc_auc_score,average_precision_score
    auc = dict(available=bool(positive_scores and negative_scores))
    if auc['available']:
        labels=np.r_[np.ones(len(positive_scores)),np.zeros(len(negative_scores))]
        scores=np.r_[positive_scores,negative_scores]
        auc.update(auroc=float(roc_auc_score(labels,scores)),
            average_precision=float(average_precision_score(labels,scores)),
            positive_rows=len(positive_scores),negative_rows=len(negative_scores),
            scores='linear-only, before cosine/margin/veto; descriptive, not threshold tuning')
    completion=dict(completed=True,controls=controls,acceptance=acceptance,
        peer_negatives_passed=negative_pass,development_gate_passed=bool(acceptance['passed_current_scope'] and negative_pass),
        results=results,linear_score_discrimination=auc,wire=wire.summary(),wire_categories={k:sum(r['bytes'] for r in wire.records if r['kind']==k) for k in sorted({r['kind'] for r in wire.records})},
        classifier_numeric_FP32_bytes=17*4,measurement='actual serialized simulator application messages; not network/TLS',
        raw_historical_CAL_opened=False,BASE_chronological_simulation=True,original_checkpoint_persisted_moments=False,
        old_CAL_holdout_previously_observed=True,validation_tail_is_development_not_final_test=True,
        installation_authorized=False,production_enabled=False,backbone_trained=False,
        next_step='native smoke only after development protection gate passes; otherwise reconsider sketch/donor, no automatic expansion')
    write_json(a.out/'completion.json',completion)
    print(json.dumps({k:v for k,v in completion.items() if k not in ('results','controls')},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('checkpoint','graphs','calibration-store','base-store','roles','data','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=512)
    args=parser.parse_args();torch.set_num_threads(4)
    with threadpool_limits(limits=1): run(args)
