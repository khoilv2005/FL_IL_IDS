"""Lock the scoped-install decision from existing evidence metadata.

No models, raw datasets, historical CAL or new guard/classifier experiments.
The existing cumulative application has no independently registered input
source restricting its semantic class inventory. Do not invent that source.
"""
import argparse
import json
from pathlib import Path

from appliance.scope_preflight import VERSION, scope_plan, declaration_sha
from appliance.state import write_json


def run(a):
    evidence = json.loads(a.evidence.read_text(encoding='utf-8'))
    if not evidence['summary']['completed']:
        raise ValueError('Incomplete evidence feasibility authority')
    source_sha = declaration_sha(evidence)
    candidates = evidence['deferred_shortlist']
    decisions = []
    task_classes = {0:list(range(6)), 1:list(range(6,12)), 2:list(range(12,18)),
                    3:list(range(18,24)), 4:list(range(24,30)), 5:list(range(30,34))}
    for pair in candidates:
        task = pair['birth_task']
        # Most optimistic current CAL coverage. Real endpoint coverage could
        # be smaller; even this optimistic full-task grant cannot qualify the
        # cumulative source. This is not an issued CAL certificate.
        observed = task_classes[task]
        application = [c for t in range(task+1) for c in task_classes[t]]
        planned = scope_plan(application, observed)
        decisions.append(dict(candidate_id=pair['candidate_id'],
            receiver=pair['receiver'], donor=pair['donor'], class_id=pair['class_id'],
            task=task, experiment_isolated=True,
            current_CAL_capacity=pair['donor_positive_CAL_capacity'],
            optimistic_current_CAL_coverage_not_a_certificate=observed,
            scope_preflight=planned,
            unqualified_prebirth_classes=planned['missing_CAL_classes'],
            known_missing_protected_pools=pair['missing_protected_pools'],
            known_prebirth_missing_pools=pair['pre_birth_missing_pools'],
            compiled=False, CAL_opened=False, smoke_authorized=False,
            decision='DEFERRED_NO_OBSERVABLE_SAFE_CUMULATIVE_SCOPE'))
    result = dict(version=VERSION, completed=True, source_evidence_sha256=source_sha,
        scope_preflight_implementation_sha256=declaration_sha(
            (Path(__file__).parents[1]/'appliance'/'scope_preflight.py').read_text(encoding='utf-8')),
        planner_implementation_sha256=declaration_sha(Path(__file__).read_text(encoding='utf-8')),
        decision='REDESIGN_REQUIRED_FOR_STRICT_CUMULATIVE_HEAD_PATCH_DEPLOYMENT',
        candidates=decisions, production_gate_changed=False,
        current_CAL_negative_minimum=32, current_CAL_positive_minimum=32,
        target_recall_minimum=.95, negative_FAR_maximum=.001, break_maximum=0,
        protected_owner_class_32_rule_is_audit_only=True,
        application_domain='cumulative_dataset', trusted_input_source_contract_present=False,
        retrospective_Task5_install_authorized=False,
        raw_dataset_reads=0, classifier_fits=0, model_forwards=0,
        production_runner_modified=False, existing_empirical_mode_silently_enabled=False,
        prospective_receipts_expand_evidence_only=True,
        summary_role='finite protected-support veto only, not CAL',
        empirical_guards_can_be_useful_but_safe_unknown_class_exclusion_not_proven=True,
        population_FAR_claim=False,
        next_step='Stop additional guard/eligibility audits for the current strict head-patch design. Redesign transfer/protection or explicitly establish a legitimate observable domain authority before enabling prospective smoke.')
    write_json(a.out, result)
    print(json.dumps(dict(completed=True, decision=result['decision'],
        shortlist=[dict(candidate_id=v['candidate_id'], blockers=v['scope_preflight']['blockers'],
            uncovered_classes=v['unqualified_prebirth_classes'], smoke_authorized=False)
                   for v in decisions]), indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    run(p.parse_args())
