"""Reproduce a frozen smoke trial and trace its head objective without tuning or TEST."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO/'candle_v3_runtime'
sys.path.insert(0,str(RUNTIME))
import torch
import torch.nn.functional as F
from fed_learning.data.denice_clean_roles import CleanRoleIncrementalDataLoader
from fed_learning.training.decentralized_denice_il import (
    _load_denice_continuation_state,_make_model,_stratified_limit_client_task_data,_split_local_validation)
from fed_learning.training.checkpoint_state import restore_denice_state
from fed_learning.training.denice_provenance import source_identity
from fed_learning.utils.seed import set_seed
from fed_learning.servers.nice_server import ContextDetector
from fed_learning.strategies.decentralized.candle_v3 import service
from fed_learning.strategies.decentralized.candle_v3.contracts import TeachingConfig
from fed_learning.strategies.decentralized.candle_v3.discovery import TeacherSelection
from fed_learning.strategies.decentralized.candle_v3.data import CurrentTaskRoles
from fed_learning.strategies.decentralized.candle_v3.snapshot import freeze_receiver,encode_panel,digest,reference_payload
from fed_learning.strategies.decentralized.candle_v3.knowledge import announce
from fed_learning.strategies.decentralized.candle_v3.head_learning import initial_head,participant_delta
from fed_learning.strategies.decentralized.candle_v3.head_aggregation import aggregate_head
from fed_learning.strategies.decentralized.candle_v3.communication import WireLedger
from fed_learning.strategies.decentralized.candle_v3.verification import predict


def describe(values):
    v = values.detach().cpu().numpy().reshape(-1)
    return dict(n=len(v),mean=float(v.mean()),minimum=float(v.min()),maximum=float(v.max()),
                median=float(np.median(v))) if len(v) else dict(n=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    args = parser.parse_args(); args.out.mkdir(parents=True,exist_ok=False)
    started = time.monotonic(); torch.set_num_threads(2); set_seed(42)
    previous = json.loads(args.receipt.read_text(encoding='utf-8'))
    trial = next(r for s in previous['teaching_sessions'] for r in s['receipts'] if r.get('head_training'))
    task,receiver,donors,targets = trial['task'],trial['receiver'],trial['donors'],trial['targets']
    saved = _load_denice_continuation_state(str(args.checkpoint.resolve()))
    if int(saved['meta']['completed_task'])!=task:
        raise ValueError('Use the checkpoint at the trial task, not a later task')
    cfg = saved['config']; settings = TeachingConfig.from_config(cfg)
    loader = CleanRoleIncrementalDataLoader(previous['roles'],evaluation_role='validation',source_data_dir=previous['data_dir'])
    ids = [receiver]+donors; models,detectors,base = {},{},{}
    for owner in ids:
        model = _make_model(cfg,torch.device(args.device))
        detector = ContextDetector(router_mode='binary_cosine')
        algorithm = saved['client_algorithm_states'][owner]
        restore_denice_state(model,detector,algorithm.get('denice',algorithm))
        model.load_state_dict(saved['client_model_states'][owner]); model.eval()
        models[owner],detectors[owner] = model,detector
        x,y = loader.get_client_data(owner,task)
        x,y = _stratified_limit_client_task_data(x,y,cfg['denice_max_train_samples_per_client'],
            seed=cfg.get('random_seed',42)+10000*task+owner)
        x,y,_,_ = _split_local_validation(x,y,float(cfg.get('denice_validation_fraction',0)),
                                        cfg.get('seed',42)+10000*task+owner)
        base[owner] = x,y
    view = CurrentTaskRoles(loader.roles,task,loader.get_task_classes(task),base,cfg.get('random_seed',42))
    preprocessing = trial['snapshot']['preprocessing_sha256']
    reference = freeze_receiver(models[receiver],detectors[receiver],receiver,task,preprocessing,args.device)
    if reference.identity.reference_sha256 != trial['snapshot']['reference_sha256']:
        raise ValueError('Checkpoint predictor differs from the original frozen trial')
    before = digest(reference_payload(models[receiver],detectors[receiver]))
    announcements = {o:announce(models[o],detectors[o],o,task,base[o][1],preprocessing) for o in ids}
    selection = TeacherSelection(receiver,tuple(donors),tuple(targets),
        tuple((c,tuple(d for d in donors if c in announcements[d].current_classes)) for c in targets))
    captured = []; original = service.learn_head
    def observe(*a,**kw):
        fitted = original(*a,**kw)
        captured.append(dict(reference=a[0],panels=a[1],targets=a[2],settings=kw['config'],fitted=fitted))
        return fitted
    with patch.object(service,'learn_head',observe):
        replay = service._session(receiver,selection,models,detectors,view,task,
            [c for t in range(task+1) for c in loader.get_task_classes(t)],
            preprocessing,settings,WireLedger(int(settings.communication_budget_mib*1024**2)),
            args.device,announcements)
    match = json.loads(json.dumps(replay['controls'])) == trial['controls']
    primary_match = all(json.loads(json.dumps(replay['controls'][name])) == trial['controls'][name]
                        for name in ('availability_only','router_only','mask_only','full'))
    if not match:
        (args.out/'mismatch.json').write_text(json.dumps(dict(original=trial,replayed=replay),indent=2))
    if not match:
        raise AssertionError('Replayed controls differ from the original smoke receipt')
    if digest(reference_payload(models[receiver],detectors[receiver]))!=before:
        raise AssertionError('Rejected trial modified the model predictor')
    full = captured[0]; fitted,panels = full['fitted'],full['panels']
    allowed = fitted['allowed']; permission = fitted['permissions']
    counts = Counter(int(y) for p in panels for y in p['labels'] if int(y) in allowed)
    weights = {c:1/n for c,n in counts.items()}
    phi = initial_head(reference,targets,permission); phi_initial = phi.clone()
    all_z = torch.cat([p['features'] for p in panels]); all_y = torch.cat([p['labels'] for p in panels])
    logits = torch.cat([p['logits'] for p in panels]); target_index = [allowed.index(c) for c in targets]
    class_y = torch.as_tensor([allowed.index(int(c)) for c in all_y])
    sample_weights = torch.as_tensor([weights[int(c)] for c in all_y])
    role_panels = {}
    for role in ('base','fit','calibration'):
        role_panels[role] = [encode_panel(reference,*view.get(o,role,task=task,per_class=getattr(settings,
            {'base':'base_per_class','fit':'fit_per_class','calibration':'cal_per_class'}[role])),o,settings.batch_size) for o in ids]
    def metric(current,items):
        rows = []
        for p in items:
            scores = p['logits'][:,allowed].clone()
            scores[:,target_index] = F.linear(p['features'],current[:,:-1]*permission,current[:,-1])
            pred = torch.as_tensor(allowed)[scores.argmax(1)]
            pos = torch.isin(p['labels'],torch.as_tensor(targets))
            margin = scores[:,target_index].max(1).values-scores[:,[i for i,c in enumerate(allowed) if c not in targets]].max(1).values
            rows.append(dict(owner=p['owner'],positive=int(pos.sum()),target_correct=int(((pred==p['labels'])&pos).sum()),
                negative=int((~pos).sum()),false_activation=int((torch.isin(pred,torch.as_tensor(targets))&~pos).sum()),
                positive_margin=describe(margin[pos]),negative_margin=describe(margin[~pos])))
        return rows
    trace = []
    for step in range(settings.head_rounds+1):
        scores = logits[:,allowed].clone()
        scores[:,target_index] = F.linear(all_z,phi[:,:-1]*permission,phi[:,-1])
        loss = (F.cross_entropy(scores,class_y,reduction='none')*sample_weights).sum()/sample_weights.sum()
        trace.append(dict(step=step,weighted_base_loss=float(loss),head_change_l2=float((phi-phi_initial).norm()),
            # CAL is not used for trace selection or tuning; terminal CAL remains in replay receipt only.
            base=metric(phi,role_panels['base']),fit=metric(phi,role_panels['fit'])))
        if step==settings.head_rounds: break
        packets = [participant_delta(p,phi,targets=targets,allowed=allowed,class_weights=weights,
            permissions=permission,rate=settings.learning_rate,round_id=step,batch_size=settings.batch_size) for p in panels]
        phi = aggregate_head(phi,[p for p in packets if p is not None])
    torch.testing.assert_close(phi,fitted['head'],rtol=0,atol=0)
    panels_info = {role:[dict(owner=p['owner'],counts=dict(Counter(p['labels'].tolist()))) for p in items]
                   for role,items in role_panels.items()}
    model_metrics = {}
    for owner in ids:
        x,y = view.get(owner,'fit',task=task,per_class=settings.fit_per_class)
        donor_pred,_ = predict(models[owner],detectors[owner],x,seen_classes=allowed,batch_size=settings.batch_size)
        model_metrics[owner] = dict(n=len(y),correct=int((donor_pred==y.numpy()).sum()))
    report = dict(schema='candle_v3_receiver_head_debug_v1',completed=True,
        source_sha256=source_identity(RUNTIME)['source_sha256'],checkpoint=str(args.checkpoint.resolve()),
        frozen_reference_matches_original=True,all_controls_match_original=match,
        primary_controls_match_original=primary_match,rejected_live_predictor_unchanged=True,
        primary_head_history_matches_original=(replay['head_training']==trial['head_training']),
        donor_only_head_history_matches_original=(replay['donor_only_training']==trial['donor_only_training']),
        task=task,receiver=receiver,donors=donors,targets=targets,settings=asdict(settings),
        evidence=panels_info,allowed_classes=allowed,mature_fc1=int(permission.sum()),total_fc1=len(permission),
        training_feature_norm=describe(all_z.norm(dim=1)),initial_weight_norm=float(phi_initial[:,:-1].norm()),
        final_weight_norm=float(phi[:,:-1].norm()),head_change_l2=float((phi-phi_initial).norm()),
        head_trace=trace,terminal_trial=replay,participant_native_fit=model_metrics,
        test_accessed=False,hyperparameters_changed=False,production_source_changed=False,
        full_training_executed=False,elapsed_seconds=time.monotonic()-started,
        limitation='One early two-client smoke checkpoint; quality is not representative of a fully trained encoder.')
    view.close()
    (args.out/'report.json').write_bytes((json.dumps(report,indent=2)+'\n').encode('utf-8'))
    print(json.dumps({k:v for k,v in report.items() if k not in ('head_trace','terminal_trial')},indent=2))


if __name__=='__main__': main()
