"""Summarize the one pre-registered graph-aware smoke without reopening historical CAL."""
import argparse
import json
from pathlib import Path
import pickle

from run_candle_v4_graph_transition import ROOT,bank_spec
from pilot_candle_v4_standalone import write,file_sha256
from fed_learning.training.candle_v4_bank import ExpertBank
from fed_learning.training.candle_v4_shadow import digest,make_expert
from fed_learning.training.candle_v4_preflight import positive_neighbors,select_fixture
import torch


def main(out):
    lock=json.loads((out/'protocol_lock.json').read_text());fixture=json.loads((out/'candidate_preflight.json').read_text())
    states={t:{name:torch.load(out/name/f'continuation_state_task_{t}.pt',map_location='cpu',weights_only=False)
               for name in ('bank','baseline')} for t in (2,3)}
    a=states[2]['bank']['candle_v4_shadow_state'];b=states[3]['bank']['candle_v4_shadow_state']
    classes=sorted(b['engines']);receiver=lock['receiver'];native_keys=('client_model_states','client_algorithm_states','rng_state')
    checks={f'native_on_off_task_{t}':digest({k:s['bank'][k] for k in native_keys})==
                                         digest({k:s['baseline'][k] for k in native_keys}) for t,s in states.items()}
    restored=ExpertBank(bank_spec(out),b)
    checks['bank_restore_exact']=digest(b)==digest(restored.state_dict())
    checks['both_experts_trained']=all(e['steps']>0 for e in a['engines'].values())
    initial=digest(make_expert(42).state_dict())
    checks['both_experts_changed_from_init']=all(digest(e['branches']['joint_class_balanced']['model'])!=initial for e in a['engines'].values())
    checks['learned_expert_weights_retained']=all(digest(a['engines'][c]['branches']['joint_class_balanced']['model'])==
                                              digest(b['engines'][c]['branches']['joint_class_balanced']['model']) for c in classes)
    checks['no_T2_expert_updates_in_T3']=all(a['engines'][c]['steps']==b['engines'][c]['steps'] for c in classes)
    checks['unique_round_step_ids']=len({(r['task'],r['round']) for r in b['history']})==len(b['history'])==16
    checks['native_parameters_finite']=all(torch.isfinite(v).all().item() for s in states.values() for state in s.values()
        for model in state['client_model_states'].values() for v in model.values() if isinstance(v,torch.Tensor))
    checks['closed_historical_CAL_rejected']=False
    try:restored.role(lock['experts'][0]['donor'],2,'calibration')
    except ValueError:checks['closed_historical_CAL_rejected']=True
    native_trace=states[3]['bank']['cluster_history'];trace=[]
    for r in native_trace:
        peers=positive_neighbors(r,receiver)
        trace.append(dict(task=r['task'],round=r['round'],current_peers=peers,
            donors_eligible={e['target']:e['donor'] in peers for e in lock['experts']}))
    checks['all_T2_updates_have_current_edges']=all(all(r['donors_eligible'].values()) for r in trace if r['task']==2)
    checks['T2_gate_owners_reachable']=all(o==receiver or o in [r for r in trace if r['task']==2][-1]['current_peers'] for o in lock['owners'])
    graph=json.loads((out/'native_preflight/prepared_graph.json').read_text())['graph']
    metadata={int(o):{k:{int(c):n for c,n in row.items()} for k,row in entry.items()}
              for o,entry in fixture['metadata'].items()}
    first=select_fixture(graph,metadata,list(range(12,18)))
    repeat=select_fixture(graph,dict(reversed(list(metadata.items()))),list(range(12,18)))
    checks['deterministic_metadata_selection']=first==repeat and first['experts']==lock['experts']
    # Exact serialized metadata advertisement model, additional to native graph descriptors.
    metadata_bytes=sum(len(pickle.dumps(dict(sender=j,receiver=i,metadata=metadata[j],
        authority_sha256=lock['roles_sha256']),protocol=5)) for i in metadata for j in metadata if i!=j)
    setup=fixture['preflight_cost']['application_descriptor_bytes']+metadata_bytes
    setup_by_class={c:setup//len(classes)+(i<setup%len(classes)) for i,c in enumerate(classes)}
    per_class={c:b['per_class_bytes'][c]+setup_by_class[c] for c in classes}
    native_bytes=sum(sum(r['v4_native_wire'].values()) for r in native_trace)
    rules=lock['rules'];bank_bytes=b['bytes']+setup;total=bank_bytes+native_bytes
    checks['bank_budget_including_setup']=bank_bytes<=rules['budgets']['bank']
    checks['per_class_budget_including_setup']=all(v<=rules['budgets']['receiver_class'] for v in per_class.values())
    checks['combined_budget_including_setup']=total<=rules['budgets']['total']
    active=all(b['status'][c]=='ACTIVE' for c in classes)
    accepted=all(g['outcome']=='PASS' for g in b['acceptances'])
    result=dict(experiment='CANDLE V4 graph-aware preflight plus one locked transition smoke',
        completed_execution=True,engineering_pass=all(checks.values()),checks=checks,
        decision='GO_FULL_SEED42' if all(checks.values()) and active and accepted else 'NO_GO_FULL',
        full_training_started=False,registered_protocol=lock,
        preflight_candidate={k:v for k,v in fixture.items() if k not in ('metadata','candidate_rows')},
        preflight_artifact='artifacts/candle_v4_graph_preflight_20261023.json',
        preflight_artifact_sha256=file_sha256(ROOT/'artifacts/candle_v4_graph_preflight_20261023.json'),
        source_graph_clock='prepared T2 before any current-task optimizer; not a future round snapshot',
        expert_steps={c:e['steps'] for c,e in b['engines'].items()},status=b['status'],
        two_trained_experts=checks['both_experts_trained'],two_ACTIVE=active,
        ACTIVE_inference_verified=any(g.get('ACTIVE_inference_verified',False) for g in b['acceptances']),
        gates=b['acceptances'],native_graph_trace=trace,expert_history=b['history'],
        communication=dict(preflight_native_descriptor_bytes=fixture['preflight_cost']['application_descriptor_bytes'],
            preflight_metadata_advertisement_bytes=metadata_bytes,preflight_setup_total=setup,
            bank_training_and_gate_bytes=b['bytes'],bank_by_kind=b['by_kind'],
            V4_incremental_total_including_setup=bank_bytes,per_receiver_class_including_setup=per_class,
            native_T2_T3_modeled_bytes=native_bytes,combined_modeled_bytes=total,
            mode='Application serialization models, not actual network benchmark; metadata setup fully charged to this one fixture'),
        model_device_resume_fix=json.loads((out/'resume_device_fix/receipt.json').read_text()),
        execution_source_lock=json.loads((out/'execution_source_lock.json').read_text()),
        graph_implementation_lock=json.loads((out/'graph_implementation_lock.json').read_text()),
        metadata_negative_coverage={e['target']:dict(donor=e['donor'],receiver=receiver,
            donor_current_BASE=fixture['metadata'][str(e['donor'])]['current_train_BASE'],
            receiver_current_BASE=fixture['metadata'][str(receiver)]['current_train_BASE']) for e in lock['experts']},
        CAL_confirmation_independent=False,TEST_opened=False,no_post_CAL_reselection=True,
        no_threshold_or_optimizer_tuning=True,
        limits=['Metadata feasibility is not proof of functional gap or safety',
            'CAL reused development lineage; no independent confirmation or population FAR guarantee',
            'CAL current-role exclusion checks selected owners BASE/FIT; other preflight BASE influences graph selection; no stronger independence claim',
            'No historical raw CAL/FIT reopened for this summary; only checkpoint receipts and role metadata',
            'Full-run projection and ACTIVE inference latency not established when bank is rejected'])
    write(out/'graph_smoke_report.json',result)
    write(ROOT/'artifacts/candle_v4_graph_transition_20261023.json',result)
    print(json.dumps(dict(decision=result['decision'],checks=checks,expert_steps=result['expert_steps'],
        communication=result['communication']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out.resolve())
