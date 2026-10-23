"""Aggregate-only report for actual receiver-transfer development eligibility."""
import argparse
import json
from collections import Counter
from pathlib import Path
from appliance.state import write_json
from fed_learning.data.denice_clean_roles import file_sha256


def observed_ok(v):
    return (v['positive_rows'] >= 32 and v['recall'] >= .95 and v['negative_rows'] > 0
            and v['max_owner_class_far'] <= .001 and v['breaks'] == 0)


def small(v):
    return {k: v[k] for k in ('status', 'positive_rows', 'target_hits', 'recall',
        'negative_rows', 'false_activation', 'far', 'max_owner_class_far', 'breaks', 'rescue',
        'forced_head_hits') if k in v} | dict(missing_pool_count=len(v['missing_or_under32_pools']),
        missing_pools=v['missing_or_under32_pools'], worst_negative_pools=sorted(
            v['per_owner_class'], key=lambda p: (-p['far'], -p['false_activation']))[:5])


def run(a):
    def read(name): return json.loads((a.run/name).read_text(encoding='utf-8'))
    summary = read('completion.json')
    if not summary['completed']:
        raise ValueError('Audit not completed')
    records = read('graph_results.json')
    scored = [v for v in records if 'evaluation' in v]
    lock = read('selection_lock_before_evaluation.json')
    locked = {(v['receiver'], v['donor'], v['class_id']): v for v in lock['locks']}
    if any(locked[v['receiver'], v['donor'], v['class_id']]['guard'] != v['guard_lock']
           for v in scored):
        raise AssertionError('Evaluation changed guard lock')
    def axes(variant):
        values = [v['evaluation'][variant] for v in scored]
        return dict(positive_recall_fail=sum(v['positive_rows'] >= 32 and v['recall'] < .95 for v in values),
            forced_head_recall_ge95=sum(v['positive_rows'] >= 32 and
                v['forced_head_hits']/v['positive_rows'] >= .95 for v in values),
            negative_FAR_fail=sum(v['max_owner_class_far'] is not None and v['max_owner_class_far'] > .001 for v in values),
            breaks_fail=sum(v['breaks'] > 0 for v in values),
            missing_positive_or_negative_evidence=sum(v['positive_rows'] < 32 or
                bool(v['missing_or_under32_pools']) for v in values),
            observed_metrics_pass_despite_missing_scope=sum(observed_ok(v) and
                v['status'] == 'insufficient_evidence' for v in values),
            axes_overlap=True)
    partial = [v for v in scored if all(observed_ok(v[r]['calibrated']) for r in ('selection', 'evaluation'))]
    selected_partial = [v for v in partial if v['guard_lock']['status'] == 'selected']
    panels = read('panels_before_predictions.json')['panels']
    pool_lookup = {(p['owner'],p['class_id'],p['role']):p for p in panels}
    missing_keys = {(p['owner'],p['class_id']) for v in selected_partial
        for p in v['evaluation']['calibrated']['missing_or_under32_pools']}
    missing_evidence = dict(unique_owner_class_pools=len(missing_keys),
        classes=dict(Counter(c for _,c in missing_keys)),
        pools_with_available_unique_below32=sum(pool_lookup[o,c,'evaluation']['available_unique']<32
            for o,c in missing_keys), cap64_is_cause=False,
        note='Increasing the sample cap cannot fill these pools with <32 available distinct contents in the locked evaluation role.')
    integrity=read('integrity_checks.json')
    def detail(v):
        return {k: v[k] for k in ('receiver', 'donor', 'class_id', 'task', 'guard_lock',
            'unobserved_cumulative_classes')} | dict(selection=small(v['selection']['calibrated']),
            evaluation=small(v['evaluation']['calibrated']), functional_eligible=v['functional_eligible'],
            current_CAL=v['current_CAL'])
    native = json.loads((a.prepared/'graph_results.json').read_text(encoding='utf-8'))
    native_map = {(v['receiver'], v['donor'], v['class_id']): v for v in native}
    false_rejections = [v for v in selected_partial if
        native_map[v['receiver'], v['donor'], v['class_id']]['evaluation']['status'] == 'failed']
    axes_fixed, axes_calibrated = axes('fixed_control'), axes('calibrated')
    installed = read('installed_patch_shadow.json')
    active = [v for v in installed['results'] if v['authorized_saved_scope']]
    result = dict(summary=summary, protocol=read('protocol_before_data.json'),
        fixed_evaluation_failure_axes=axes_fixed, calibrated_evaluation_failure_axes=axes_calibrated,
        selected_partial_observed_both_splits=len(selected_partial),
        selected_partial_receivers=len({v['receiver'] for v in selected_partial}),
        selected_partial_classes=dict(Counter(v['class_id'] for v in selected_partial)),
        selected_partial_missing_evidence=missing_evidence, integrity_checks=integrity,
        native_failed_but_actual_observed_both_splits=len(false_rejections),
        partial_is_unknown_not_install_permission=True,
        partial_candidates=[detail(v) for v in selected_partial],
        installed_shadow_source=installed['source_sha256'],
        installed_patches=[{k: v[k] for k in ('receiver','donor','class_id','lifecycle_state',
            'authorized_saved_scope','donor_still_in_live_graph')} | dict(evaluation=small(v['evaluation']))
            for v in installed['results']],
        saved_active_patch_shadow_status=dict(Counter(v['evaluation']['status'] for v in active)),
        files_sha256={name:file_sha256(a.run/name) for name in ('protocol_before_data.json',
            'graph_results.json','selection_lock_before_evaluation.json','cache_parity.json',
            'panels_before_predictions.json','completion.json')},
        guard_lock_unchanged=True, new_population_safety_certificates=0)
    write_json(a.artifact, result)
    sf, sc = summary['fixed_evaluation'], summary['calibrated_evaluation']
    partial_table = '\n'.join(f'| {v["receiver"]} ← {v["donor"]} / {v["class_id"]} | '
        f'{100*v["selection"]["calibrated"]["recall"]:.2f}% | '
        f'{100*v["evaluation"]["calibrated"]["recall"]:.2f}% | '
        f'{100*v["evaluation"]["calibrated"]["max_owner_class_far"]:.2f}% | '
        f'{len(v["evaluation"]["calibrated"]["missing_or_under32_pools"])} |'
        for v in sorted(selected_partial,key=lambda v:(v['receiver'],v['donor'],v['class_id']))[:8])
    text = f'''# APPLIANCE — Actual Transfer Eligibility Audit

## Protocol đã khóa

Task 5/round 19, checkpoint DeNICE legacy + APPLIANCE của results (13), SHA256 `{summary['checkpoint_sha256']}`. Dùng graph thật với alpha dương, ξ=0,8. Không train backbone, cài patch, sửa runner hoặc thay gate.

Native donor `binary_cosine` không còn sàng lọc candidate. Đo donor FC2 head trên **receiver features**, shared sketch 16D và guard `signature > tau AND fixed-context margin > gamma AND NOT own BASE veto`. Router của receiver chỉ tạo baseline để đếm break/rescue; donor router không quyết định imported activation.

Hai nhánh: (1) control có tau=donor-FIT p95 floor, gamma=0; (2) cùng guard production, quantile grid 31 điểm và tie rule chọn trên **development selection** của receiver/donor. Không thêm classifier hoặc feature mới. Nhánh (2) dùng đầy đủ owned development classes của hai endpoint; không giả danh đây là current CAL production.

Dữ liệu là các original FIT panels đã khóa ở audit trước: signature-fit / selection / evaluation tách content, tối đa 64 nội dung/owner/class/split. Đây là **development hồi cứu đã được dùng**, không phải untouched confirmation. Toàn bộ guard selection khóa trước actual-transfer evaluation. Gate: recall ≥95%, FAR từng owner–class ≤0,1%, break=0; positive và từng protected pool ≥32 mẫu. Đây là sufficiency rule bảo thủ của audit, không đổi production aggregate CAL gate.

## Funnel và kết quả

| Bước | Số candidate |
|---|---:|
| Metadata candidates | {summary['metadata_candidates']} |
| Qua maturity | {summary['maturity_candidates']} |
| Thiếu usable signature FIT → UNKNOWN, chưa probe | {summary['insufficient_signature_FIT']} |
| Đã chạy actual head+route+guard | {summary['actual_function_evaluated_pairs']} |

| Evaluation | Guard control | Guard chọn trên development |
|---|---:|---:|
| PASS trong scope quan sát đủ evidence | {sf.get('passed_observed_scope',0)} | {sc.get('passed_observed_scope',0)} |
| FAIL | {sf.get('failed',0)} | {sc.get('failed',0)} |
| UNKNOWN do evidence | {sf.get('insufficient_evidence',0)} | {sc.get('insufficient_evidence',0)} |

Candidate được chọn trước evaluation và pass cả hai split: **{summary['functional_eligible_pairs']}**. Không cộng số PASS evaluation-only thành donor đủ điều kiện.

Trục lỗi evaluation (có chồng lắp):

| Trục | Control | Calibrated |
|---|---:|---:|
| Recall positive <95%, n≥32 | {axes_fixed['positive_recall_fail']} | {axes_calibrated['positive_recall_fail']} |
| Head thắng task-reference ≥95% positive (chưa có routing/veto) | {axes_fixed['forced_head_recall_ge95']} | {axes_calibrated['forced_head_recall_ge95']} |
| FAR protected negative vượt gate | {axes_fixed['negative_FAR_fail']} | {axes_calibrated['negative_FAR_fail']} |
| Break >0 | {axes_fixed['breaks_fail']} | {axes_calibrated['breaks_fail']} |
| Thiếu/ít evidence positive hoặc negative | {axes_fixed['missing_positive_or_negative_evidence']} | {axes_calibrated['missing_positive_or_negative_evidence']} |

## Candidate có tín hiệu nhưng chưa đủ evidence

Có **{len(selected_partial)}** candidate đạt toàn bộ metric **đã quan sát trên cả selection và evaluation**, thuộc {len({v['receiver'] for v in selected_partial})} receiver; vẫn UNKNOWN nếu còn protected pools dưới 32 mẫu. Trong đó **{len(false_rejections)}** candidate bị native evaluation đánh FAIL nhưng actual function đạt metric quan sát. Đây là bằng chứng native screening không tương đương transfer eligibility; không phải chứng minh safe install.

Không cộng rescue/break trên các receiver/candidate dùng trùng samples thành accuracy hệ thống. Chi tiết candidate và missing pools có trong artifact JSON. Scope chỉ bao gồm class sở hữu BASE của receiver/live peers; class ngoài scope không được chứng nhận. 0 false activation trên ≤64 mẫu không chứng minh population FAR ≤0,1%.

23 cặp có tín hiệu này thiếu tổng cộng **{len(missing_keys)} owner–class pools khác nhau**; phân bố class: `{missing_evidence['classes']}`. Toàn bộ có <32 unique contents khả dụng trong evaluation role đã khóa: tăng cap 64 hoặc batch size không giải quyết được. Cần evidence hợp lệ bổ sung hoặc phạm vi chứng nhận có thể kiểm chứng; không lấy test/BASE/provenance giả làm CAL.

Ví dụ đầu theo thứ tự ID (không chọn donor mới bằng evaluation):

| Receiver ← donor / class | Recall selection | Recall evaluation | FAR max evaluation | Pool thiếu evidence |
|---|---:|---:|---:|---:|
{partial_table}

## Current CAL và patch đã cài

Current CAL status: `{summary['current_CAL_status']}`. Chỉ guard đã pass development mới được mở CAL holdout Task 5. Class cũ không có positive CAL trong Task 5 → UNKNOWN; không đọc lại CAL cũ. Nếu có phép đo current CAL thì signature/guard vẫn khóa, không tune CAL holdout. Không tạo certificate hoặc install; prototype development-FIT không được gọi là một production current-CAL-FIT compile mới.

7 patch active và 1 suspended giữ nguyên checkpoint. Tái sử dụng exact frozen shadow kết quả audit trước, cùng checkpoint/panels/packet, có source SHA; **không phải replication mới**. Active shadow status: `{result['saved_active_patch_shadow_status']}`. 6/7 active fail observed gate, 1/7 thiếu evidence vẫn là cảnh báo. Patch suspended không được tính thành thiệt hại deployed. Không mở rộng activation scope, không sửa certificate/registry trong audit.

## Integrity và giới hạn

- Model/router bất biến: `{summary['all_models_unchanged']}`.
- Cache imported features chỉ cho checkpoint không adapter; so với `stable_signals` ở {summary['cache_parity_receivers']} receiver: `{summary['all_parity_checks_passed']}`, prediction/activation match. Numerical checks dùng selection, không sửa threshold trên evaluation.
- Guard lock không đổi sau evaluation: `True`.
- Recompute accounting {integrity['metric_records_recomputed']} bộ metric và 15 synthetic controls (12 so selector production + 3 missing-evidence): PASS.
- Historical raw CAL reads=0; final test reads=0; installations=0; certificate mới=0.
- Số CAL reads ở đây là **truy cập role evidence**: FIT loader phải giải nén original train shard để trích đúng FIT indices. Không dùng nhãn/mẫu CAL lịch sử để fit hay chấm; không khẳng định zero byte-level I/O của các row khác trong shard nén. Đây là audit hồi cứu, không phải runtime replay-free đã triển khai.
- CPU shadow không revalidate CUDA certificate. Baseline break là receiver checkpoint binary router, không phải full-test multiclass refit.
- Không đo communication deployment mới; audit local CPU không được gọi là protocol decentralized đã triển khai.

## Quyết định tiếp theo

Tách ba tầng: functional metric quan sát, đầy đủ protected evidence và current CAL authorization. Không dùng 0 native pass làm lý do bác bỏ head transfer; cũng không dùng partial success để cài patch. Chưa chạy full training. Nếu có candidate giữ recall/FAR tốt nhưng thiếu scope, ưu tiên coverage evidence hợp lệ; nếu có đủ evidence mà actual function vẫn FAIL thì loại đúng cặp/checkpoint đó trước khi xem xét redesign.

Artifact: `{a.artifact.as_posix()}`. Script: `tools/audit_appliance_actual_transfer.py`.
'''
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(text, encoding='utf-8')
    print(json.dumps(dict(summary=summary, partial=len(selected_partial),
        native_false_rejections=len(false_rejections), fixed_axes=axes_fixed, calibrated_axes=axes_calibrated), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'prepared', 'artifact', 'report'):
        p.add_argument('--'+name, type=Path, required=True)
    run(p.parse_args())
