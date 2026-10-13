"""Issue a fresh empirical production lock only after actual bounded checks.

The lock covers source bytes and tested integration, not population safety,
fresh six-task accuracy, CUDA guard portability or the Task4 loss regression.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from appliance.empirical_deployment import VERSION, deployment_scope
from appliance.patch_lifecycle import route_authorized
from appliance.stable_head import StableHeadRegistry
from appliance.state import write_json
from eval_checkpoint import _make_denice_client_model


def run(a):
    controlled=json.loads((a.smoke/'controlled/completion.json').read_text())
    activation=json.loads((a.smoke/'controlled/empirical_activation.json').read_text())
    native=json.loads((a.smoke/'completion.json').read_text())
    unit=json.loads(a.protocol_checks.read_text())
    if not all(r['completed'] for r in (controlled,native,unit)):
        raise ValueError('All empirical protocol/automatic/native checks must complete first')
    final=torch.load(a.smoke/'native/continuation_state_task_4.pt',map_location='cpu',weights_only=False)
    service=final['appliance_service_state']
    if service['contract']['scope_mode']!=VERSION:
        raise ValueError('Wrong native service contract')
    checks={**{f'protocol/{k}':v for k,v in unit['checks'].items()},
            **{f'automatic/{k}':v for k,v in controlled['checks'].items()},
            **{f'native/{k}':v for k,v in native['checks'].items()}}
    initial=service['rounds'][0]
    pairs=[r['pair'] for r in initial['transactions'] if r['transaction']['applied']]
    checks['multiple_receivers_same_class']=len({p['class_id'] for p in pairs})<len(pairs)
    checks['actual_imported_activation']=bool(activation['records']) and all(
        r['authorized'] and r['activated']>0 and r['seal_unchanged'] for r in activation['records'])
    checks['every_observed_head_function_survives']=all(r['head_exact_after'] and r['certificate_current']
        for row in service['rounds'] for r in row['lifecycle'])
    # A genuinely trained receiver must remain active at the transition;
    # inactive endpoint carry alone is insufficient survival evidence.
    active4=set(map(int,final['cluster_history'][-1]['groups']))
    status=[]
    for pair in pairs:
        cid,c=pair['receiver'],pair['class_id']
        model,router=_make_denice_client_model(final,cid,'cpu')
        registry=StableHeadRegistry();registry.entries=model.appliance_guarded_head_entries
        entry=registry.entries[c]
        scope=dict(domain_id=service['contract']['application_domain'],classes=list(range(34)))
        status.append(dict(receiver=cid,class_id=c,trained_at_task4=cid in active4,
            authorized_at_full_application_scope=route_authorized(entry,scope),
            function_current=registry.certificate_current(model,router,c),
            state=entry['lifecycle_state'],reason=entry['lifecycle_reason'],
            deployment_scope=deployment_scope(entry),initial_CAL_scope=entry['lifecycle_certificate']['scope']))
    checks['active_trained_receiver_survival']=any(r['trained_at_task4'] and r['function_current'] and
        r['authorized_at_full_application_scope'] for r in status)
    paths=list(Path('appliance').glob('*.py'))+[Path(p) for p in (
        'fed_learning/training/decentralized_denice_il.py','fed_learning/training/checkpoint_state.py',
        'fed_learning/training/denice_checkpoint_archive.py','fed_learning/training/denice_eval.py',
        'fed_learning/data/denice_clean_roles.py','fed_learning/clients/nice_client.py',
        'tools/eval_denice_legacy_self.py')]
    result=dict(version='appliance_production_integration_smoke_v2',scope_mode=VERSION,
        completed=all(checks.values()),checks=checks,comparisons=len(checks),
        mismatches=sum(not value for value in checks.values()),
        imported_route_activation_verified=checks['actual_imported_activation'],
        automatic_commits=len(pairs),automatic_rejections=controlled['rejections'],
        lifecycle=status,guard_unchanged=True,xi=.8,training_method='legacy',
        native_backend='CPU FP32',native_task=4,native_rounds=2,native_max_samples_per_client=256,
        historical_CAL_runtime_reads=0,final_test_opened=False,
        full_campaign_started=False,population_FAR_claim=False,
        frozen_guard_CUDA_portability_verified=False,Task4_loss_regression_resolved=False,
        fixture='legacy terminal Task3 and actual graph; owned BASE summaries retrospectively seeded for late-start smoke',
        activation_source=activation['source'],independent_accuracy_test=False,
        communication=controlled['communication'],
        limitation='Empirical authorization outside finite CAL evidence; native smoke is bounded CPU integration, not full campaign performance',
        source_sha256={p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))})
    write_json(a.report,result)
    print(json.dumps({k:result[k] for k in ('completed','comparisons','mismatches','automatic_commits','lifecycle')},indent=2))
    if not result['completed']:raise AssertionError(checks)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke',type=Path,required=True)
    p.add_argument('--protocol-checks',type=Path,default=Path('audit_denice/appliance_empirical_integration/protocol_checks.json'))
    p.add_argument('--report',type=Path,default=Path('artifacts/appliance_production_integration_smoke.json'))
    torch.set_num_threads(4)
    run(p.parse_args())
