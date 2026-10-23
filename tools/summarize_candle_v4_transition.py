"""Read-only audit of the single transition smoke; no candidate/threshold search."""
import argparse
import json
from pathlib import Path

from pilot_candle_v4_standalone import write,file_sha256
import torch


def main(out):
    root=Path(__file__).resolve().parents[1]
    report=json.loads((out/'report.json').read_text())
    states={t:torch.load(out/'bank'/f'continuation_state_task_{t}.pt',map_location='cpu',weights_only=False) for t in (2,3)}
    roles=json.loads((root/'audit_denice/appliance_legacy_results11/inputs/roles/role_manifest.json').read_text())
    rows=[]
    for graph in states[3]['cluster_history']:
        if graph['task']!=2:continue
        r=graph['alpha_debug'].get(18,graph['alpha_debug'].get('18'))
        peers=[int(i) for i,a in zip(r['group_ids'],r['alphas']) if int(i)!=18 and a>0]
        reachable={}
        for c in (13,17):
            sources=[p for p in peers if roles['clients'][str(p)]['role_class_counts']['base'].get(str(c),0)]
            reachable[c]=dict(current_sources=sources,count_eligible_sources=[p for p in sources
                if roles['clients'][str(p)]['role_class_counts']['calibration'].get(str(c),0)>=32])
        rows.append(dict(round=graph['round'],receiver=18,current_peers=peers,configured_donor_16_eligible=16 in peers,
                         target_metadata=reachable))
    bank=states[3]['candle_v4_shadow_state']
    checks=dict(report['checks'])
    checks['native_parameters_finite']=all(torch.isfinite(v).all().item() for state in states.values()
        for model in state['client_model_states'].values() for v in model.values() if isinstance(v,torch.Tensor))
    learned=all(e['steps']>0 for e in bank['engines'].values())
    # Final gate attempted all owners; actual graph has no link to owner16.
    scope_valid=all(o==18 or o in rows[-1]['current_peers'] for o in bank['spec']['owners'])
    result=dict(experiment='CANDLE V4 final transition smoke, fixed development fixture',
        decision='NO_GO_FULL',full_training_started=False,registered_protocol=json.loads((out/'protocol_lock.json').read_text()),
        mechanical_checks=checks,mechanical_checks_pass=all(checks.values()),
        qualifying_smoke_pass=False,two_trained_experts=learned,two_ACTIVE=False,
        learned_expert_retention_proven=False,ACTIVE_inference_tested=False,
        controlled_collision_handler_only=True,
        candidate_graph_trace=rows,configured_gate_owner_scope_valid=scope_valid,
        primary_blocker='Fixed donor16 was outside receiver18 live neighborhood in all eight T2 rounds: both drafts had zero updates.',
        implementation_findings=['Fixture locked ownership/count metadata before checking live donor reachability',
            'Acceptance should defer before FIT/CAL if an expert has no authorized training updates',
            'Gate reference traffic must also enforce current graph; owner16 was outside the final gate neighborhood'],
        scientific_interpretation='CAL results describe untrained drafts and an invalid gate-owner path; do not interpret as trained V4 quality, learned retention or proof of method failure.',
        smoke_receipt=report,execution_source_lock=json.loads((out/'execution_source_lock_before_output_fix.json').read_text()),
        output_only_correction=json.loads((out/'output_fix_receipt.json').read_text()),
        raw_receipt_sha256={p:file_sha256(out/p) for p in ('report.json','protocol_lock.json','execution_source_lock_before_output_fix.json')},
        CAL_confirmation_independent=False,TEST_opened=False,
        inference_latency=None,full_training_communication_projection=None,
        projection_reason='No learned/ACTIVE bank: not a valid basis to project full V4 training cost or inference latency',
        no_post_CAL_candidate_switch=True,no_gate_lowering=True,no_additional_full_smoke=True)
    write(root/'artifacts/candle_v4_transition_20261023.json',result)
    print(json.dumps(dict(decision=result['decision'],steps=report['expert_steps'],mechanical_checks=checks,
                         gate_scope_valid=scope_valid,bytes=report['combined_modeled_bytes']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)
