"""Two-client real-BASE native smoke plus task-boundary resume equivalence.

No TEST/validation evaluation, no historical CAL, no parameter/route mocks.
Participants are an explicit induced subgraph of an existing locked dataset.
"""
import argparse
from collections import Counter
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile

REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO/'candle_v3_runtime'
sys.path[:] = [str(RUNTIME)]+[p for p in sys.path if Path(p or '.').resolve() not in (REPO,REPO/'tools')]

import numpy as np
import torch
from fed_learning.data.denice_clean_roles import CleanRoleIncrementalDataLoader, file_sha256
from fed_learning.data.incremental_loader import IncrementalDataLoader
from fed_learning.training.decentralized_denice_il import run_decentralized_denice_il, _load_denice_continuation_state
from fed_learning.training.checkpoint_state import restore_denice_state
from fed_learning.models.denice_model import DeNICEModel
from fed_learning.servers.nice_server import ContextDetector
from fed_learning.strategies.decentralized.candle_v3.data import CurrentTaskRoles
from fed_learning.strategies.decentralized.candle_v3.snapshot import digest
from fed_learning.strategies.decentralized.candle_v3.verification import predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--roles',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--clients',type=int,nargs=2,default=[9,30])
    parser.add_argument('--variant',choices=['direct_fc2','receiver_head_learning'],default='direct_fc2')
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    started = time.monotonic()
    profile = 'candle_v3_direct_fc2.json' if args.variant == 'direct_fc2' else 'candle_v3_peer_teaching.json'
    template = json.loads((REPO/'configs'/profile).read_text(encoding='utf-8'))
    template.update(data_dir=str(args.data_dir.resolve()),denice_clean_roles_dir=str(args.roles.resolve()),
        candle_v3_integration_smoke=True,task_start=0,task_end=1,rounds_per_task=1,
        denice_max_clients=2,denice_max_train_samples_per_client=64,
        denice_amp_enabled=False,batch_size=32,denice_fisher_samples=2,
        denice_evaluation_data_role='validation',denice_post_task_eval=False,
        denice_eval_final_round=False,denice_eval_local_validation=False,
        denice_eval_terminal_state_only=True,denice_log_loss_semantics=False,
        denice_debug=False,round_checkpoint_every=1,denice_archive_checkpoints=True,
        save_continuation_every_task=True)
    template['candle_v3_teaching'].update(base_per_class=64,fit_per_class=64,cal_per_class=64)
    access, step_events, results = [],[],{}
    original_role = CurrentTaskRoles.get
    original_train = CleanRoleIncrementalDataLoader.get_client_data
    def role_access(view,owner,role,*,task,per_class):
        x,y = original_role(view,owner,role,task=task,per_class=per_class)
        assert task == view.task
        assert set(y.tolist()) <= set(view.classes)
        access.append(dict(owner=owner,role=role,requested_task=task,current_task=view.task,
                           classes=sorted(set(y.tolist())),n=len(y)))
        return x,y
    def train_access(loader,owner,task):
        x,y = original_train(loader,owner,task)
        assert set(y.tolist()) <= set(loader.get_task_classes(task))
        step_events.append(dict(owner=owner,task=task,n=len(y),classes=sorted(set(y.tolist()))))
        return x,y
    def no_evaluation(*_args,**_kwargs):
        raise AssertionError('Smoke must not open TEST or run cumulative evaluation')
    def run(name,end,resume=None):
        output = args.out/name
        cfg = deepcopy(template)
        cfg.update(output_dir=str(output.resolve()),resume_output_dir=str(output.resolve()),task_end=end)
        if resume is not None:
            cfg.update(task_start=1,resume_state_path=str(resume.resolve()))
        else:
            cfg['resume_state_path'] = None
        with patch.object(CleanRoleIncrementalDataLoader,'get_all_client_ids',lambda _loader:list(args.clients)), \
             patch.object(CleanRoleIncrementalDataLoader,'get_client_data',train_access), \
             patch.object(CleanRoleIncrementalDataLoader,'get_test_data',no_evaluation), \
             patch.object(CleanRoleIncrementalDataLoader,'get_full_test_data',no_evaluation), \
             patch.object(IncrementalDataLoader,'get_test_data',no_evaluation), \
             patch.object(IncrementalDataLoader,'get_full_test_data',no_evaluation), \
             patch.object(CurrentTaskRoles,'get',role_access):
            result = run_decentralized_denice_il(cfg)
        results[name] = result
        return output
    full = run('uninterrupted',1)
    split = run('split_task0',0)
    resumed = run('resumed_task1',1,split/'checkpoint_task_0_all_rounds.zip')
    left = _load_denice_continuation_state(str(full/'checkpoint_task_1_all_rounds.zip'))
    right = _load_denice_continuation_state(str(resumed/'checkpoint_task_1_all_rounds.zip'))
    checks, max_difference = {},0.
    checks['both_clients_restored'] = left['client_ids'] == right['client_ids'] == sorted(args.clients)
    for cid in left['client_ids']:
        for key,value in left['client_model_states'][cid].items():
            other = right['client_model_states'][cid][key]
            if torch.is_floating_point(value):
                max_difference = max(max_difference,float((value-other).abs().max()))
            checks[f'weights/{cid}/{key}'] = torch.equal(value,other)
        a,b = left['client_algorithm_states'][cid],right['client_algorithm_states'][cid]
        checks[f'algorithm_state/{cid}'] = digest(a) == digest(b)
        record = a['candle_v3_state']
        checks[f'knowledge_owner/{cid}'] = record['owner'] == cid
        checks[f'no_double_task_receipt/{cid}'] = len(record['completed_tasks']) == len(set(record['completed_tasks']))
        model = DeNICEModel((39,1),34)
        detector = ContextDetector(router_mode='binary_cosine')
        restore_denice_state(model,detector,a); model.load_state_dict(left['client_model_states'][cid]); model.eval()
        restored = DeNICEModel((39,1),34)
        rd = ContextDetector(router_mode='binary_cosine')
        restore_denice_state(restored,rd,b); restored.load_state_dict(right['client_model_states'][cid]); restored.eval()
        torch.manual_seed(123)
        panel = torch.randn(8,39,1)
        p = predict(model,detector,panel,seen_classes=list(range(12)),batch_size=4)
        q = predict(restored,rd,panel,seen_classes=list(range(12)),batch_size=4)
        checks[f'predictions_restored/{cid}'] = all(np.array_equal(x,y) for x,y in zip(p,q))
    optimizer = [json.loads(line) for line in (full/'optimizer_health.jsonl').read_text().splitlines()]
    checks['real_optimizer_updates'] = all(row['optimizer_steps'] > 0 for row in optimizer)
    checks['two_task_history'] = [row['task'] for row in results['uninterrupted']['history']['task_accuracies']] == [0,1]
    checks['current_role_access_only'] = all(e['current_task']==e['requested_task'] for e in access)
    checks['finalized_archive_self_contained'] = all((output/f'checkpoint_task_{task}_all_rounds.zip').is_file()
        for output,task in ((full,0),(full,1),(split,0),(resumed,1)))
    sessions = results['uninterrupted']['history'].get('candle_v3_teaching',[])
    checks['native_task_end_service_executed_twice'] = len(sessions) == 2
    statuses = Counter(row['status'] for report in sessions for row in report['receipts'])
    trained = [row for report in sessions for row in report['receipts'] if row.get('head_training')]
    if args.variant == 'receiver_head_learning':
        stopped = [row for report in sessions for row in report['receipts'] if row.get('stage')=='evidence_preflight']
        checks['head_attempt_or_explicit_evidence_preflight'] = bool(trained or stopped)
        checks['executed_head_rounds_match_profile'] = all(len(row['head_training']) == 8 for row in trained)
        checks['executed_head_participants_match_profile'] = all(
            set(p['owner'] for p in round_row['participants']) == set([row['receiver']]+row['donors'])
            for row in trained for round_row in row['head_training'])
        checks['evidence_stops_have_no_head_updates'] = all(row['optimizer_steps']==0 and
            row['status']=='UNKNOWN' and not row['evidence_preflight']['possible'] for row in stopped)
    report = dict(schema='candle_v3_native_smoke_v2',variant=args.variant,completed=all(checks.values()),
        checks=checks,checks_passed=sum(checks.values()),checks_total=len(checks),max_parameter_difference=max_difference,
        dataset='existing locked real BASE; explicit two-client induced subgraph',
        data_dir=str(args.data_dir.resolve()),roles=str(args.roles.resolve()),
        role_manifest_sha256=file_sha256(args.roles/'role_manifest.json'),participants=args.clients,
        tasks=[0,1],rounds_per_task=1,max_train_samples_per_client_task=64,
        real_optimizer_steps=sum(r['optimizer_steps'] for r in optimizer),session_statuses=dict(statuses),
        head_sessions=len(trained),head_aggregation_rounds=sum(len(row['head_training']) for row in trained),
        teaching_sessions=sessions,
        source_sha256=left['config']['source_sha256'],
        current_access=access,base_access=step_events,elapsed_seconds=time.monotonic()-started,
        torch=str(torch.__version__),gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        test_dataset_accessed=False,accuracy_benchmark=False,
        limitation='Natural installs are counted, not forced. Controlled commit smoke is a separate receipt.')
    (args.out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('checks','current_access','base_access')},indent=2))
    if not report['completed']:
        print('FAILED CHECKS:',[k for k,v in checks.items() if not v])
        raise SystemExit(1)


if __name__ == '__main__':
    main()
