"""Metadata-only evidence feasibility for the 23 frozen transfer candidates.

Never opens any NPZ, historical CAL, validation/test inputs or predictors.
Counts are optimistic row capacities, not independent/unique sample evidence.
No summary is promoted to CAL acceptance and no prospective smoke is launched.
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path
import torch

from appliance.active_pair_calibration import split_counts
from appliance.selector import lookup
from appliance.state import write_json
from fed_learning.data.denice_clean_roles import file_sha256


def zero_error_upper(n, alpha=.05):
    return -math.expm1(math.log(alpha)/n) if n else 1.


def source_counts(role_manifest, cal_manifest, owner, c, tasks):
    counts=role_manifest['clients'][str(owner)]['role_class_counts']
    source={r:int(v.get(str(c),0)) for r,v in counts.items()}
    task=tasks[c]
    partition=cal_manifest['clients'][str(owner)][str(task)]
    n=int(partition['class_counts'][str(c)])
    if source['calibration']!=n:raise ValueError('CAL partition/role count mismatch')
    f,s,h=map(int,split_counts(n))
    return source,dict(total=n,fit=f,selection=s,holdout=h,
        task=task,partition_sha256=partition['sha256'],
        counts_are_upper_bounds_for_unique_independent_contents=True)


def run(a):
    a.out.mkdir(parents=True,exist_ok=False)
    write_json(a.out/'completion.json',dict(completed=False))
    prior=json.loads(a.artifact.read_text(encoding='utf-8'))
    candidates=prior['partial_candidates']
    expected=prior['summary']
    role_path=a.cal_store/'role_manifest.json'
    cal_path=a.cal_store/'calibration_store_manifest.json'
    lock=json.loads((a.cal_store/'calibration_store_lock.json').read_text())
    if (file_sha256(role_path)!=expected['role_manifest_sha256'] or
        file_sha256(cal_path)!=lock['manifest_sha256'] or
        file_sha256(a.checkpoint)!=expected['checkpoint_sha256']):
        raise ValueError('Evidence authority changed')
    roles=json.loads(role_path.read_text(encoding='utf-8'))
    cal=json.loads(cal_path.read_text(encoding='utf-8'))
    if (not cal['completed'] or cal['role_manifest_sha256']!=expected['role_manifest_sha256'] or
        file_sha256(a.cal_store/'metadata.json')!=cal['metadata_sha256'] or
        cal['metadata_sha256']!=roles['metadata_sha256']):
        raise ValueError('CAL metadata source changed')
    tasks={c:int(t) for t,cs in cal['task_classes'].items() for c in cs}
    if sorted(tasks)!=list(range(34)):raise ValueError('Task coverage changed')
    protocol=dict(kind='Evidence feasibility, metadata only',source_artifact_sha256=file_sha256(a.artifact),
        audit_script_sha256=file_sha256(Path(__file__)),
        checkpoint_sha256=expected['checkpoint_sha256'],role_manifest_sha256=expected['role_manifest_sha256'],
        calibration_store_manifest_sha256=file_sha256(cal_path),graph_file_sha256=file_sha256(a.graphs),
        candidates=len(candidates),current_task=5,minimum_audit_owner_class_rows=32,
        production_negative_rule='receiver CAL HOLDOUT aggregate >=32; different from audit per-owner/class >=32',
        gate_recall=.95,gate_far=.001,gate_break=0,
        raw_calibration_reads=0,raw_FIT_reads=0,raw_BASE_reads=0,raw_validation_reads=0,raw_test_reads=0,
        classifier_fits=0,model_forwards=0,smoke_authorized=False,statistical_certificates_issued=0,
        limits='Metadata counts are row upper bounds; distinct content and independence are not assumed. Earlier selected 23 are development, not untouched confirmation.')
    write_json(a.out/'protocol_before_analysis.json',protocol)
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if ckpt['task']!=5:raise ValueError('Terminal task changed')
    states={int(i):s.get('denice',s) for i,s in ckpt['client_algorithm_states'].items()}
    graphs=json.loads(a.graphs.read_text(encoding='utf-8'))
    matrix={};incidences=[];options=[]
    for v in candidates:
        i,j,c,t=(v[k] for k in ('receiver','donor','class_id','task'))
        candidate_id=f'{i}_{j}_{c}'
        if tasks[c]!=t:raise ValueError('Candidate birth task changed')
        rows=v['evaluation']['missing_pools']
        for p in rows:
            o,k=p['owner'],p['class_id'];key=o,k
            if key not in matrix:
                counts,capacity=source_counts(roles,cal,o,k,tasks)
                shield=states[o]['appliance_base_sketch_shield_state']
                e=shield['memory']['entries'].get(str(k))
                if e is None or e['count']!=counts['base']:
                    raise ValueError('Protected owner BASE provenance changed')
                matrix[key]=dict(owner=o,class_id=k,source_task=tasks[k],
                    locked_FIT_evaluation_unique_rows=p['rows'],source_role_rows=counts,
                    all_original_train_role_rows=sum(counts.values()),CAL_capacity=capacity,
                    VAL_rows_ge32=counts['validation']>=32,
                    VAL_use='qualification/development only; unseen unique content not verified; never CAL',
                    old_CAL_runtime_read_authorized=tasks[k]==5,
                    prospective_current_CAL_HOLDOUT_ge32=capacity['holdout']>=32,
                    existing_summary_role='own BASE support/provenance, not CAL acceptance',
                    summary_acceptance_authorized=False,candidate_ids=[])
            elif matrix[key]['locked_FIT_evaluation_unique_rows']!=p['rows']:
                raise ValueError('Shared protected pool row count changed across candidates')
            if candidate_id in matrix[key]['candidate_ids']:
                raise ValueError('Duplicate candidate/protected-pool incidence')
            matrix[key]['candidate_ids'].append(candidate_id)
            when='pre_birth' if tasks[k]<t else 'at_birth' if tasks[k]==t else 'post_birth'
            incidences.append(dict(candidate_id=candidate_id,receiver=i,donor=j,target_class=c,
                target_task=t,owner=o,protected_class=k,negative_task=tasks[k],temporal_position=when,
                feasible_receipt_window=('No exact future guard existed; a prior generic summary cannot grant acceptance'
                    if when=='pre_birth' else 'Only when candidate guard exists and is locked/current in this task'),
                currently_readable_CAL=tasks[k]==5,
                current_CAL_HOLDOUT_capacity=matrix[key]['CAL_capacity']['holdout'],
                adequate_audit_pool_capacity=matrix[key]['prospective_current_CAL_HOLDOUT_ge32']))
        _,donor_cal=source_counts(roles,cal,j,c,tasks)
        rc=cal['clients'][str(i)][str(t)]['class_counts']
        neg=sum(int(split_counts(n)[2]) for k,n in rc.items() if int(k)!=c)
        neg_sel=sum(int(split_counts(n)[1]) for k,n in rc.items() if int(k)!=c)
        graph_rounds=[]
        for g in graphs:
            if g['task']!=t:continue
            alpha=lookup(g['alpha_debug'],i,None)
            if alpha and any(int(d)==j and w>0 for d,w in zip(alpha['group_ids'],alpha['alphas'])):
                graph_rounds.append(g['round'])
        registered=states[i].get('appliance_guarded_head_entries',{}).get(str(c))
        if registered is None:registered=states[i].get('appliance_guarded_head_entries',{}).get(c)
        target_absent=(roles['clients'][str(i)]['role_class_counts']['base'].get(str(c),0)==0
                       and roles['clients'][str(i)]['role_class_counts']['calibration'].get(str(c),0)==0)
        # Potential metadata window only, never a pass on an earlier model.
        potential=(donor_cal['fit']>=32 and donor_cal['selection']>=8 and
            donor_cal['holdout']>=32 and neg>=32 and neg_sel>=32 and bool(graph_rounds) and target_absent)
        related=[p for p in incidences if p['candidate_id']==candidate_id]
        old_required=[k for k in ckpt['seen_classes'] if k not in cal['task_classes']['5']
                      and int(states[i]['neuron_ages']['fc2'][k])>=2]
        own_classes=set(map(int,states[i]['appliance_base_sketch_shield_state']['memory']['entries']))
        options.append(dict(candidate_id=candidate_id,receiver=i,donor=j,class_id=c,birth_task=t,
            donor_positive_CAL_capacity=donor_cal,receiver_birth_CAL_negative_HOLDOUT_rows=neg,
            receiver_birth_CAL_selection_rows=neg_sel,receiver_target_absent_BASE_and_CAL=target_absent,
            birth_task_live_positive_alpha_rounds=graph_rounds,
            prospective_current_scope_metadata_window=potential,
            current_Task5_positive_CAL_available=False,
            exact_registered_candidate_certificate_present=registered is not None,
            missing_protected_pools=len(rows),pre_birth_missing_pools=sum(p['temporal_position']=='pre_birth' for p in related),
            same_task_missing_pools=sum(p['temporal_position']=='at_birth' for p in related),
            post_birth_missing_pools=sum(p['temporal_position']=='post_birth' for p in related),
            terminal_installer_old_BASE_missing_classes=sorted(set(old_required)-own_classes),
            maturity_and_compatibility_at_birth='UNKNOWN; terminal Task5 evidence cannot be backdated',
            guard_at_birth='must compile from current birth-task snapshots; never import the later Task5 head/thresholds',
            full_protected_evidence_capacity_feasible=all(p['adequate_audit_pool_capacity'] for p in related),
            smoke_authorized=False,qualification='UNKNOWN, not a production CAL pass'))
    entries=[matrix[k] for k in sorted(matrix)]
    if len(entries)!=prior['selected_partial_missing_evidence']['unique_owner_class_pools']:
        raise AssertionError('408 matrix denominator changed')
    if len({v['candidate_id'] for v in options})!=len(candidates):
        raise AssertionError('Duplicate transfer candidate')
    if sum(len(v['candidate_ids']) for v in entries)!=len(incidences):
        raise AssertionError('Evidence matrix/timeline incidence accounting changed')
    # Ranking uses metadata capacities/deficits and ID, not outcomes. Keep the
    # two smallest-deficit candidates as isolated future experiments, not two
    # installs into a receiver (current installer permits one capability).
    ranking=sorted((v for v in options if v['prospective_current_scope_metadata_window']),
        key=lambda v:(v['missing_protected_pools'],v['pre_birth_missing_pools'],
                      v['receiver'],v['donor'],v['class_id']))
    shortlist=[dict(v,rank=n+1,status='DEFERRED_NO_FULL_PROTECTED_EVIDENCE',
        isolated_receiver_clone_required=True) for n,v in enumerate(ranking[:2])]
    n95=math.ceil(math.log(.05)/math.log1p(-.001))
    if not zero_error_upper(n95)<=.001<zero_error_upper(n95-1):
        raise AssertionError('Binomial upper-bound boundary failed')
    if any(v['current_Task5_positive_CAL_available'] or v['smoke_authorized'] for v in options):
        raise AssertionError('Invalid late install authorization')
    summary=dict(completed=True,candidates=len(options),unique_missing_pools=len(entries),
        candidate_pool_incidences=len(incidences),temporal_incidences=dict(Counter(p['temporal_position'] for p in incidences)),
        pools_CAL_HOLDOUT_row_capacity_ge32=sum(v['prospective_current_CAL_HOLDOUT_ge32'] for v in entries),
        pools_total_CAL_row_capacity_ge32=sum(v['CAL_capacity']['total']>=32 for v in entries),
        pools_original_train_all_roles_below32=sum(v['all_original_train_role_rows']<32 for v in entries),
        pools_VAL_row_capacity_ge32=sum(v['VAL_rows_ge32'] for v in entries),
        prospective_current_scope_metadata_windows=sum(v['prospective_current_scope_metadata_window'] for v in options),
        candidates_with_full_audit_protected_evidence_capacity=sum(v['full_protected_evidence_capacity_feasible'] for v in options),
        candidates_with_current_Task5_positive_CAL=0,
        candidates_with_exact_persisted_certificate=sum(v['exact_registered_candidate_certificate_present'] for v in options),
        authorized_smokes=0,old_raw_CAL_reads=0,all_raw_dataset_reads=0,model_forwards=0,classifier_fits=0,
        production_negative_gate_changed=False,independent_sample_claim=False,
        conclusion='Broad owner-class qualification is infeasible with unchanged clean CAL allocations for these 23; current-only prospective acceptance may be possible but does not certify cumulative inference.')
    retention_plan=[dict(kind='protection summary',capture='owner current BASE/FIT or CAL FIT, before advancing task',
        fields=['owner','class','task','role','partition SHA','coordinate digest','encoder/sketch version','count','summary bounds/statistics'],
        raw_examples_retained=0,acceptance_authorized=False,
        limitation='No exact pass/fail for an arbitrary new guard; moment statistics do not reconstruct tail activations'),
        dict(kind='exact frozen-guard negative receipt',capture='owner current CAL HOLDOUT after guard lock',
        fields=['patch ID','guard/declaration SHA','owner','task','class','rows','activated','break count',
                'role/partition/coordinate SHA','precision/device','parent certificate SHA'],
        raw_examples_retained=0,can_extend_existing_scope_only_if_function_unchanged=True,
        absent_before_patch_birth=True,positive_install_acceptance_substitution=False),
        dict(kind='positive acceptance',capture='donor current target-task CAL HOLDOUT after FIT/selection lock',
        fields=['exact receiver function','head/signature/threshold versions','target positive rows/hits',
                'current negative receipts','role/partition/coordinate SHA'],
        no_historical_reopen=True,no_native_donor_certificate_transfer_to_new_receiver=True)]
    result=dict(protocol=protocol,summary=summary,protected_pool_matrix=entries,
        candidate_pool_timeline=incidences,candidate_feasibility=options,deferred_shortlist=shortlist,
        prospective_retention_plan=retention_plan,
        statistics=dict(one_sided_confidence=.95,far_target=.001,zero_error_independent_rows_required=n95,
            upper_with_zero_errors_n32=zero_error_upper(32),upper_with_zero_errors_n64=zero_error_upper(64),
            multiple_pool_adjustment_applied=False,metadata_rows_not_unique_or_independent=True))
    write_json(a.out/'evidence_feasibility.json',result)
    write_json(a.out/'completion.json',summary)
    write_json(a.publish,result)
    table='\n'.join(f'| {v["receiver"]} ← {v["donor"]} / {v["class_id"]} | T{v["birth_task"]} | '
        f'{v["donor_positive_CAL_capacity"]["fit"]}/{v["donor_positive_CAL_capacity"]["selection"]}/{v["donor_positive_CAL_capacity"]["holdout"]} | '
        f'{v["receiver_birth_CAL_negative_HOLDOUT_rows"]} | {v["missing_protected_pools"]} | DEFERRED |' for v in shortlist)
    report=f'''# APPLIANCE — Evidence Feasibility Audit

## Kết luận

**Không thể chứng nhận đủ broad protected scope cho 23 candidate bằng clean CAL allocation hiện có và quy tắc audit ≥32/owner–class.** Cả 408 missing pools đều có CAL HOLDOUT row capacity <32; **133 pool có tổng toàn bộ original train roles <32 rows**. Không thể giải quyết bằng cap, batch size hoặc capture summary nhiều lần trên cùng samples.

Đây không phải kết luận mọi head transfer thất bại. Nó xác định giới hạn dữ liệu và thời gian của **qualification đang dùng**. Production yêu cầu receiver CAL HOLDOUT aggregate ≥32, khác với quy tắc conservative audit per-owner/class ≥32; audit này **không đổi** một trong hai gate và không áp requirement 3.000 mẫu vào production.

## Phạm vi và nguồn

23 candidate của actual-transfer development audit `fdaf47a`; checkpoint Task5/round19 SHA `{expected['checkpoint_sha256']}`. Chỉ đọc artifact, role/store manifests, graph history và checkpoint state. **Không mở NPZ hoặc raw CAL/FIT/BASE/VAL/test; không forward, classifier fit, install, smoke hoặc sửa production.** Counts là số row tối đa từ metadata, chưa kiểm chứng unique/independent content.

## Ma trận evidence

| Kiểm tra 408 pool | Số |
|---|---:|
| CAL HOLDOUT row capacity ≥32 | {summary['pools_CAL_HOLDOUT_row_capacity_ge32']} |
| Toàn bộ CAL rows ≥32 (không phải reserved HOLDOUT) | {summary['pools_total_CAL_row_capacity_ge32']} |
| Tổng original train BASE+FIT+CAL+VAL <32 | {summary['pools_original_train_all_roles_below32']} |
| VAL row capacity ≥32 | {summary['pools_VAL_row_capacity_ge32']} |

VAL có thể là nguồn qualification/development khi có authority/content split hợp lệ; ở đây chưa đọc, chưa kiểm chứng unseen/unique và **không phải CAL acceptance**. BASE/provenance chỉ giúp bảo vệ/kiểm kê, không bù số mẫu CAL. Ma trận đầy đủ gồm owner, class, task, role counts, CAL FIT/SEL/H counts, partition version, nguồn summary và các candidate liên quan trong artifact JSON.

408 pool được dùng ở **{len(incidences)} candidate–pool incidences**. Theo task của negative so với birth task của target:

- Pre-birth: {summary['temporal_incidences'].get('pre_birth',0)}.
- Cùng task: {summary['temporal_incidences'].get('at_birth',0)}.
- Sau birth: {summary['temporal_incidences'].get('post_birth',0)}.

Không cộng các incidence trùng owner–class thành số mẫu độc lập.

## Production CAL authorization

Target của cả 23 candidate thuộc T1/T2/T4; current T5 chỉ có classes 30–33. **0/23 có current positive CAL cho target**, và **0/23 có certificate đúng candidate persist trong checkpoint**. Không thể cấp phép cài ở T5 bằng cách mở lại CAL cũ, đổi tên FIT thành CAL hoặc tái sử dụng native donor quality.

Metadata cho thấy **{summary['prospective_current_scope_metadata_windows']}/23** cặp có cửa sổ prospective ở đúng birth task: donor FIT≥32/SEL≥8/H≥32, receiver current SEL/H≥32, target absent và live positive-alpha edge. Đây chỉ là **khả năng về count/graph**. Maturity, head compatibility, routing, FAR/break, BASE coverage và phiên bản certificate ở birth chưa được kiểm tra; Task5 weights/guard không được backdate vào T1/T2/T4. Ba cặp bị loại ngay theo metadata: `63 ← 71 / 13` thiếu edge ở birth; `16 ← 44 / 24` có receiver negative SEL/H = 26/27; `84 ← 4 / 6` có H = 32 nhưng SEL chỉ 30.

## Evidence có thể giữ mà không lưu raw CAL

1. **Generic protection summary**: tạo khi owner/class còn current; lưu role/partition/coordinate hashes, sketch version, count và moments/enclosure. Giúp bảo vệ/ranking, không tự trở thành CAL pass cho guard mới.
2. **Exact guard receipt**: chỉ khi patch/guard đã tồn tại, đã khóa và endpoint CAL vẫn current. Lưu per-class rows/activated/break cùng patch, function/declaration, precision/device và provenance. Scope chỉ mở theo evidence thực sự đã đo; receipt không dùng lại nếu head, route, threshold, dependency hoặc shield khác phiên bản. Code `current_scope_evidence.py` / `cumulative_certificate.py` đã có binding cho installed guards.
3. **Positive acceptance**: cần đúng receiver function trên donor current target CAL HOLDOUT. Certificate của donor-native hoặc old moments không chứng minh positive recall của receiver mới.

**Pre-birth gap**: guard của class mới chưa tồn tại lúc các class cũ còn current. Không thể tạo receipt cho một function tương lai. Summary cũ chỉ là protection evidence; muốn derive một bound/receipt mới từ summary phải có một protocol kiểm chứng riêng, hiện chưa được authorize/implement. Cumulative receipt chỉ giải quyết negative classes đến **sau** install, không tự giải quyết các class trước install.

## Shortlist cho prospective smoke

Chọn theo metadata: ít missing pools, ít pre-birth gaps, tie theo ID; không dùng evaluation accuracy để xếp hạng. Hai cặp dưới đây là **đề xuất có điều kiện**, chưa được chạy:

| Receiver ← donor / class | Task phải compile | Donor CAL F/S/H | Receiver current negative H | Missing pools | Status |
|---|---|---:|---:|---:|---|
{table}

Hai cặp cùng receiver 2 phải chạy **hai clone/experiments tách biệt**, không cài đồng thời: installer hiện chưa chứng minh multi-capability acceptance. Compile guard mới bằng snapshot và CAL thực sự current của birth task; không chuyển Task5 head/threshold trở ngược. Chỉ khởi chạy khi có legitimate positive acceptance và protected scope được xác định đủ. Khi scope chưa đủ, giữ DEFERRED/UNKNOWN, không coi pass current-only là safe cumulative deployment.

## Statistical FAR và empirical gate

Với 0 lỗi và n mẫu độc lập, upper bound một phía 95% là `1 - 0.05^(1/n)`. n=64 cho **{100*zero_error_upper(64):.3f}%**, không phải ≤0,1%. Cần **{n95}** mẫu âm độc lập để bound một pool ≤0,1%; chưa hiệu chỉnh nhiều pool/candidate. Metadata row counts và content dedup không tự chứng minh independence. Gate FAR empirical 0,1% hiện tại giữ nguyên, không được viết thành population guarantee.

## Mốc quyết định

- **Không train full hoặc chạy smoke ở T5 cho 23 class cũ này.** Chỉ bổ sung negative evidence không khôi phục positive CAL đã đóng cửa.
- Muốn thử tiếp bằng cơ chế hiện tại: chốt prospective install **khi target còn current**, coverage contract/scope rõ và evidence capture từ thời điểm thích hợp. Pre-birth protection vẫn phải giải quyết; không lấy summary thay CAL. Metadata-only shortlist chưa cho phép cài.
- Với dataset/roles và broad ≥32/owner–class rule giữ nguyên, mở thêm audit guard/classifier sẽ không khắc phục được 408 deficit. Cần nguồn CAL mới hợp lệ hoặc thiết kế evidence/transfer và phạm vi chứng nhận khác được nêu rõ; không âm thầm đổi allocation/gate, pooling owner hay cho missing=PASS.
- Sau khi thực sự có đủ legitimate evidence, nếu acceptance vẫn FAIL thì reject cặp/design; không kéo dài tuning guard. Hiện chưa đo được bước đó, nên chưa kết luận head-patch toàn bộ thất bại.

Artifact: `{a.publish.as_posix()}`. Script: `tools/audit_appliance_evidence_feasibility.py`.
'''
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(report,encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('artifact','cal-store','checkpoint','graphs','out','publish','report'):
        p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
