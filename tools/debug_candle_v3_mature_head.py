"""Bounded current-task head trial on a mature legacy checkpoint; no backbone training."""
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict,replace
import hashlib
import io
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile

REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO/'candle_v3_runtime'
sys.path.insert(0,str(RUNTIME))
import torch
from fed_learning.data.denice_clean_roles import CleanRoleIncrementalDataLoader,file_sha256
from fed_learning.training.decentralized_denice_il import _make_model,_split_local_validation
from fed_learning.training.checkpoint_state import restore_denice_state
from fed_learning.training.denice_provenance import source_identity
from fed_learning.utils.seed import set_seed
from fed_learning.strategies.decentralized.candle_v3 import service
from fed_learning.strategies.decentralized.candle_v3.contracts import TeachingConfig
from fed_learning.strategies.decentralized.candle_v3.discovery import TeacherSelection
from fed_learning.strategies.decentralized.candle_v3.data import CurrentTaskRoles
from fed_learning.strategies.decentralized.candle_v3.knowledge import announce
from fed_learning.strategies.decentralized.candle_v3.snapshot import freeze_receiver,reference_payload,digest
from fed_learning.strategies.decentralized.candle_v3.verification import predict
from fed_learning.strategies.decentralized.candle_v3.communication import WireLedger
from fed_learning.servers.nice_server import ContextDetector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--roles',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--discovery-policy',choices=['availability_only_v1','control_fit_preflight_v1'])
    args = parser.parse_args(); args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2); set_seed(42); started = time.monotonic()
    loader = CleanRoleIncrementalDataLoader(args.roles,source_data_dir=args.data_dir,evaluation_role='validation')
    task = 2; participants = [9,30]
    classes = loader.get_task_classes(task)
    settings = TeachingConfig.from_config(json.loads((REPO/'configs/candle_v3_peer_teaching.json').read_text()))
    if args.discovery_policy:
        settings = replace(settings,discovery_policy=args.discovery_policy)
    cases = []
    # Metadata only, locked before opening the trained checkpoint or scoring any examples.
    for receiver,donor in ((9,30),(30,9)):
        own = loader.roles.manifest['clients'][str(receiver)]['role_class_counts']
        peer = loader.roles.manifest['clients'][str(donor)]['role_class_counts']
        eligible = [c for c in classes if not own['base'].get(str(c),0)
                    and peer['base'].get(str(c),0) and peer['fit'].get(str(c),0)>=settings.min_fit_positive
                    and peer['calibration'].get(str(c),0)>=settings.min_cal_positive
                    and sum(v for c1,v in own['calibration'].items() if int(c1) in classes and int(c1)!=c)>=settings.min_cal_negative]
        if eligible:
            c = min(eligible)
            cases.append(dict(receiver=receiver,donor=donor,target=c,
                positive_fit=peer['fit'][str(c)],positive_cal=peer['calibration'][str(c)]))
    lock = dict(task=task,participants=participants,cases=cases,settings=asdict(settings),
        source_sha256=source_identity(RUNTIME)['source_sha256'],
        selection='fixed two cached clients, smallest current class with metadata evidence per direction; no score selection',
        role_manifest_sha256=file_sha256(args.roles/'role_manifest.json'),test_forbidden=True,
        optimizer='unchanged original eight rounds',production_installation=False)
    (args.out/'protocol_lock.json').write_bytes((json.dumps(lock,indent=2)+'\n').encode())
    print('LOCKED CASES',cases,flush=True)
    with zipfile.ZipFile(args.checkpoint) as archive:
        manifest = json.loads(archive.read('checkpoint_archive_manifest.json'))
        if manifest['task_id']!=task: raise ValueError('Wrong current task checkpoint')
        name = manifest['full_terminal_checkpoint']; content = archive.read(name)
        if hashlib.sha256(content).hexdigest()!=manifest['checksums'][name]: raise ValueError('Checkpoint corrupt')
        saved = torch.load(io.BytesIO(content),map_location='cpu',weights_only=False)
        del content
    cfg = saved['config']
    if (cfg.get('denice_cl_method','legacy')!='legacy' or cfg.get('rounds_per_task')!=20
            or cfg.get('denice_data_roles_sha256')!=lock['role_manifest_sha256']
            or cfg.get('denice_max_train_samples_per_client')
            or cfg.get('appliance_enabled') or cfg.get('denice_appliance_enabled')):
        raise ValueError('Expected full-budget legacy checkpoint with matching roles')
    print('MATURE CHECKPOINT LOADED',flush=True)
    models,detectors,base = {},{},{}
    for owner in participants:
        model = _make_model(cfg,torch.device('cuda'))
        detector = ContextDetector(router_mode='binary_cosine')
        algorithm = saved['client_algorithm_states'][owner]
        restore_denice_state(model,detector,algorithm.get('denice',algorithm))
        model.load_state_dict(saved['client_model_states'][owner]); model.eval()
        expected = saved['client_model_states'][owner]
        if any(not torch.equal(v.detach().cpu(),expected[k]) for k,v in model.state_dict().items()):
            raise ValueError('Model restore mismatch')
        models[owner],detectors[owner] = model,detector
        x,y = loader.get_client_data(owner,task)
        x,y,_,_ = _split_local_validation(x,y,float(cfg.get('denice_validation_fraction',0)),
                                        cfg.get('seed',42)+task*10000+owner)
        base[owner] = x,y
    del saved
    preprocessing = digest(dict(metadata=loader.roles.manifest['metadata_sha256'],input_shape=cfg['input_shape'],
                                feature_contract='original_preprocessed_input_v1'))
    results = []
    for case in cases:
        receiver,donor,target = case['receiver'],case['donor'],case['target']
        if int(models[receiver].unit_ranks['fc2'][target])!=0:
            results.append(dict(case=case,status='INELIGIBLE',reason='receiver row is already allocated'))
            continue
        view = CurrentTaskRoles(loader.roles,task,classes,base,42)
        local_models,local_routes = deepcopy(models),deepcopy(detectors)
        before = digest(reference_payload(local_models[receiver],local_routes[receiver]))
        announcements = {o:announce(local_models[o],local_routes[o],o,task,base[o][1],preprocessing) for o in participants}
        selection = TeacherSelection(receiver,(donor,),(target,),((target,(donor,)),))
        fitted_heads = []; original = service.learn_head
        captured = {}; verify = service._verify_and_commit
        def observe_verify(*a,**kw):
            captured['conditions'] = a[2]
            return verify(*a,**kw)
        def observe(*a,**kw):
            fitted = original(*a,**kw)
            fitted_heads.append(dict(include_receiver=kw['include_receiver'],
                mature_fc1=int(fitted['permissions'].sum()),head_change_l2=float((fitted['head']-
                    service.initial_head(a[0],a[2],fitted['permissions'])).norm()),history=fitted['history']))
            return fitted
        ledger = WireLedger(int(settings.communication_budget_mib*1024**2))
        with patch.object(service,'learn_head',observe), patch.object(service,'_verify_and_commit',observe_verify):
            receipt = service._session(receiver,selection,local_models,local_routes,view,task,
                [c for t in range(task+1) for c in loader.get_task_classes(t)],
                preprocessing,settings,ledger,'cuda',announcements)
        fit_diagnostics = {}
        for condition,(candidate,route) in captured.get('conditions',{}).items():
            rows = []
            forced = deepcopy(candidate); forced.set_active_context(task); forced.eval()
            allowed = sorted(route.episode_classes[task])
            for owner in participants:
                x,y = view.get(owner,'fit',task=task,per_class=settings.fit_per_class)
                prediction,episodes = predict(candidate,route,x,seen_classes=[c for t in range(task+1)
                    for c in loader.get_task_classes(t)],batch_size=settings.batch_size)
                forced_predictions = []
                with torch.no_grad():
                    for start in range(0,len(y),settings.batch_size):
                        out = forced(x[start:start+settings.batch_size].cuda())[:,allowed]
                        forced_predictions.extend(torch.as_tensor(allowed)[out.argmax(1).cpu()].tolist())
                labels = y.numpy(); positive = labels==target
                rows.append(dict(owner=owner,positive=int(positive.sum()),
                    target_correct=int(((prediction==target)&positive).sum()),
                    target_routed_correctly=int(((episodes==task)&positive).sum()),
                    forced_target_correct_DIAGNOSTIC_ONLY=int(sum(c==target for c,p in zip(forced_predictions,positive) if p)),
                    negative=int((~positive).sum()),false_activation=int(((prediction==target)&~positive).sum())))
            fit_diagnostics[condition] = rows
        result = dict(case=case,receipt=receipt,head_diagnostics=fitted_heads,fit_diagnostics=fit_diagnostics,wire=ledger.summary(),
                      live_checkpoint_untouched=True,
                      rejected_shadow_predictor_unchanged=(receipt['status']=='PASS' or
                        before==digest(reference_payload(local_models[receiver],local_routes[receiver]))))
        results.append(result); view.close()
        print('MATURE TRIAL',case,'STATUS',receipt['status'],receipt.get('totals'),flush=True)
        (args.out/f'receiver_{receiver}_trial.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
    report = dict(completed=True,protocol=lock,results=results,elapsed_seconds=time.monotonic()-started,
        checkpoint_terminal_sha256=manifest['checksums'][name],backbone_training=False,full_training=False,
        test_read=False,historical_task_data_read=False,
        limitation='Offline current T2 experiment on an existing V1 encoder, not a fresh V3 survival run.')
    (args.out/'report.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())


if __name__=='__main__': main()
