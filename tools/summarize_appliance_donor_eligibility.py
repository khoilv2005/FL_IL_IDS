"""Publish aggregate-only evidence from the frozen donor eligibility audit."""
import argparse
import json
from collections import Counter
from pathlib import Path
from appliance.state import write_json
from fed_learning.data.denice_clean_roles import file_sha256


def installed_controls(checkpoint,expected_sha):
    import hashlib
    import numpy as np
    import torch
    from appliance.portable_route import ProtectedRoute
    from appliance.guarded_head import head_snapshot
    from appliance.stable_head import guard_function
    from appliance.state import complete_hash
    from eval_checkpoint import _make_denice_client_model
    if file_sha256(checkpoint)!=expected_sha:raise ValueError('Checkpoint identity changed')
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
    torch.set_num_threads(4)
    results=[]
    for i,s in ckpt['client_algorithm_states'].items():
        entries=s.get('denice',s).get('appliance_guarded_head_entries',{})
        if not entries:continue
        model,router=_make_denice_client_model(ckpt,int(i),'cpu')
        before=complete_hash(model,router)
        for c,e in entries.items():
            route=ProtectedRoute.from_packet(e['packet']);live=head_snapshot(model,int(c))
            exact=bool(np.array_equal((live['weight']*live['weight_mask']).numpy(),route.head_weight)
                       and float(live['bias']*live['bias_mask'])==route.head_bias)
            function=guard_function(model,route,e['reference_classes'])
            results.append(dict(receiver=int(i),class_id=int(c),live_head_matches_packet=exact,
                all_query_dependencies_mature=function['all_dependencies_mature'],
                packet_sha256=hashlib.sha256(e['packet']).hexdigest(),
                CPU_guard_fingerprint=function['fingerprint'],old_CUDA_certificate_revalidated=False))
        if complete_hash(model,router)!=before:raise AssertionError('Integrity inspection mutated model')
    if not all(v['live_head_matches_packet'] and v['all_query_dependencies_mature'] for v in results):
        raise AssertionError('Installed head/dependency controls failed')
    return results


def small(result):
    keys = ('status','positive_rows','target_hits','recall','negative_rows','false_activation',
            'far','max_owner_class_far','breaks','rescue','missing_or_under32_pools','negative_screen_opened','early_stop')
    return {k:result[k] for k in keys if k in result} | {
        'worst_negative_pools': sorted(result['per_owner_class'],
            key=lambda v:(-v['far'],-v['false_activation'],v['owner'],v['class_id']))[:10]}


def native_small(result):
    value=small(result)
    value.pop('breaks',None)
    value['breaks_measured']=False
    value['breaks_not_applicable']='No receiver override in native donor screen; only target confusion measured'
    for pool in value['worst_negative_pools']:
        pool.pop('breaks',None)
    return value


def run(a):
    def read(name):return json.loads((a.run/name).read_text(encoding='utf-8'))
    summary = read('completion.json')
    if not summary['completed']:raise ValueError('Incomplete audit')
    candidates = read('graph_results.json')
    patches = read('installed_patch_results.json')
    summary['transfer_evaluated_pairs']=sum('transfer' in v for v in candidates)
    head_controls=installed_controls(a.checkpoint,summary['checkpoint_sha256'])
    coverage=json.loads(a.coverage_metadata.read_text(encoding='utf-8')) if a.coverage_metadata else read('coverage_metadata.json')
    if coverage['metadata_eligible']!=len(candidates):raise ValueError('Coverage denominator changed')
    def failures(role):
        rows = [v[role] for v in candidates if role in v]
        return dict(evaluated=len(rows), positive_recall_below95_with_at_least32=sum(
            v['positive_rows']>=32 and v['recall']<.95 for v in rows),
            negative_FAR_exceeded=sum(any(p['far']>.001 for p in v['per_owner_class']) for v in rows),
            missing_or_undersized_scope=sum(bool(v['missing_or_under32_pools']) or v['positive_rows']<32 for v in rows),
            negative_screen_opened=sum(v.get('negative_screen_opened',False) for v in rows),
            note='Failure counts overlap; unknown scope is not proof of safe transfer or impossibility')
    selected = {(v['receiver'],v['donor'],v['class_id']) for v in candidates
                if v.get('selection',{}).get('status')=='passed_observed_scope'}
    def observed(v):
        return (v.get('positive_rows',0)>=32 and v.get('recall',0)>=.95
                and v.get('negative_rows',0)>0 and v.get('max_owner_class_far',1)<=.001)
    partial=[dict(receiver=v['receiver'],donor=v['donor'],class_id=v['class_id'],
        selection=native_small(v['selection']),evaluation=native_small(v['evaluation']),
        evaluation_observed_metrics_met=observed(v['evaluation']),installation_authorized=False)
        for v in candidates if 'selection' in v and observed(v['selection'])
        and v['selection']['status']=='insufficient_evidence']
    compact = dict(summary=summary, coverage_metadata=coverage, installed_head_controls=head_controls,
        protocol=read('protocol_before_data.json'),
        source_checksums={n:file_sha256(a.run/n) for n in ('completion.json','graph_results.json',
            'installed_patch_results.json','panels_before_predictions.json','selection_lock_before_evaluation.json')},
        failure_axes={r:failures(r) for r in ('selection','evaluation')},
        precheck_reasons=dict(Counter(v.get('reason',v.get('precheck',{}).get('reason'))
            for v in candidates if v['status']=='precheck_rejected')),
        unique_donor_class_pairs=len({(v['donor'],v['class_id']) for v in candidates}),
        selected_receiver_class_requests=len({(i,c) for i,j,c in selected}),
        selection_observed_metrics_met_but_scope_insufficient=partial,
        installed_patches=[{k:v[k] for k in ('receiver','donor','class_id','installed_task',
            'lifecycle_state','authorized_saved_scope','donor_still_in_live_graph','packet_sha256','tau','gamma')} |
            {r:small(v[r]) | {'donor_native':native_small(v[r]['donor_native'])} for r in ('selection','evaluation')}
            for v in patches],
        empirical_transfer_passes=[{k:v[k] for k in ('receiver','donor','class_id','task','transfer')}
            for v in candidates if v.get('transfer',{}).get('status')=='passed_observed_scope'],
        quarantined_fixture=dict(receiver=65,donor=88,class_id=20,task=3,
            checkpoint_sha256='cac0d0e08c7c60495952fbf3dda3a22088e85acba06f0317fe5f8b80d4d021bf',
            decision='Rejected for this audited fixture; no production guard changed',
            evidence_commit='3bda3f5', universal_donor_or_class_blacklist=False),
        raw_inputs_or_row_hashes_published=False)
    write_json(a.artifact,compact)
    lines=['# APPLIANCE — Donor eligibility và kiểm tra patch active', '',
        '## Phạm vi', '',
        'Checkpoint DeNICE legacy + APPLIANCE Task 5/round 19 của results (13). '
        'Graph thật với alpha > 0; không CGoFed, CME, retrain hay cài patch mới.', '',
        'Dữ liệu là role FIT 8% đã tách khỏi BASE/CAL/VAL, chia theo content hash thành '
        'signature-FIT / selection / evaluation; tối đa 64 nội dung khác nhau mỗi owner–class–split. '
        'Phần cũ trong audit separability được loại theo content. Đây là development hồi cứu, '
        'không thay thế CAL acceptance và không phải final confirmation.', '',
        'Gate giữ nguyên: recall ≥95%, FAR từng owner–class ≤0,1%, break=0, '
        'positive và mỗi negative pool ≥32 mẫu. Thiếu evidence không được tính là pass. '
        '32 mẫu mỗi owner–class là quy tắc đủ evidence bảo thủ của audit này; '
        'không thay gate production CAL vốn đếm negative aggregate. '
        '0 FP trên panel nhỏ chưa chứng nhận FAR population ≤0,1%. '
        'Native donor screen chỉ đo recall/confusion, không có receiver override nên không đo break; '
        'break=0 được kiểm tra ở shadow patch và transfer probe.', '',
        '## Toàn graph', '',
        f'- Clients: **{summary["clients"]}**.',
        f'- Missing-owned receiver–class requests: **{coverage["missing_owned_requests"]}**; '
        f'có donor sở hữu class trong graph: **{coverage["requests_with_owned_live_donor"]}**; '
        f'không có: **{coverage["requests_without_owned_live_donor"]}**.',
        f'- Owned donor edges: **{coverage["owned_donor_edges"]}**; '
        f'loại vì receiver output không free: **{coverage["receiver_output_not_free"]}**. '
        'Class có output mature không tự bảo đảm competence, nhưng installer hiện không thay các row này.',
        f'- Metadata candidates: **{summary["graph_metadata_candidates"]}**; unique donor–class: **{compact["unique_donor_class_pairs"]}**.',
        f'- Maturity funnel: `{json.dumps(summary["maturity_funnel"])}`.',
        f'- Selection: `{json.dumps(summary["donor_selection_status"])}`.',
        f'- Evaluation: `{json.dumps(summary["donor_evaluation_status"])}`.',
        f'- Donor pairs qua cả hai screen: **{summary["donor_eligible_pairs"]}**.',
        f'- Transfer probes thực sự chạy: **{summary["transfer_evaluated_pairs"]}**; '
        f'pass trong scope quan sát: **{summary["empirical_transfer_pass_pairs"]}**. '
        'Không có certificate mới hoặc quyền install.', '',
        'Native screen dùng **patch-free self view + binary_cosine lưu trong checkpoint**, '
        'không chạy APPLIANCE của donor và không refit multiclass. '
        'Đây là điều kiện eligibility bảo thủ, không phải upper bound của head + imported route. '
        'Donor native router sai vẫn có thể chứa head transfer được. Vì vậy zero pass của '
        'screen này không chứng minh tất cả parameter transfer bất khả thi.', '',
        f'Có **{len(partial)}** cặp đạt recall/FAR trên phần selection đã quan sát nhưng thiếu '
        'evidence cho toàn scope. Chúng vẫn là unknown, không được nâng lên safe donor '
        'và không được chọn bổ sung sau khi xem evaluation.', '',
        '| Trục lỗi trên evaluation | Số cặp |', '|---|---:|']
    for label,k in [('Recall positive <95% (n≥32)','positive_recall_below95_with_at_least32'),
                    ('FAR negative vượt ngân sách','negative_FAR_exceeded'),
                    ('Phạm vi thiếu/ít mẫu','missing_or_undersized_scope')]:
        lines.append(f'| {label} | {compact["failure_axes"]["evaluation"][k]} |')
    lines += ['', 'Các trục lỗi chồng lắp; không cộng để suy ra số cặp reject. '
        'Dừng sớm nếu positive <32 hoặc recall <95%; FAR chưa được đo cho các cặp đó. '
        'Negative screen vẫn chạy đầy đủ cho cặp có positive đạt gate và tất cả patch đã cài.', '',
        '## 7 patch active và 1 patch suspended', '',
        '| Receiver ← donor / class | State | Recall evaluation | FAR max owner–class | Break | Kết luận shadow |',
        '|---|---|---:|---:|---:|---|']
    for v in patches:
        r=v['evaluation']
        rec='N/A' if r['recall'] is None else f'{r["target_hits"]}/{r["positive_rows"]} ({100*r["recall"]:.2f}%)'
        far='N/A' if r['max_owner_class_far'] is None else f'{100*r["max_owner_class_far"]:.2f}%'
        lines.append(f'| {v["receiver"]} ← {v["donor"]} / {v["class_id"]} | {v["lifecycle_state"]} | {rec} | {far} | {r["breaks"]} | {r["status"]} |')
    lines += ['', 'Packet, threshold, head, reference classes và shield-at-install giữ nguyên. '
        'Shadow chạy CPU FP32 trên checkpoint hiện tại, không revalidate certificate CUDA. '
        'Baseline để tính break/rescue ở đây là binary router của receiver; không so trực tiếp '
        'với break của full-test multiclass. Negative scope gồm receiver, live peers và donor gốc. '
        'Các receiver khác nhau có thể query cùng nội dung; không cộng rescue/break giữa patch '
        'để suy ra accuracy toàn hệ thống.', '',
        '### Diễn giải', '',
        '- Trong 7 patch active: **6 không đạt gate quan sát, 1 thiếu evidence**. '
        'Patch 64 ← 4/class 9 đạt 61/64 positive và không false activation trên phần quan sát, '
        'nhưng không đủ số mẫu ở các protected pools để được tính safe.',
        '- Patch 65 ← 88/class 20 có 63/64 target hits, nhưng kích hoạt trên '
        '**64/64 class 13 của owner 89 và 63/64 của owner 88**. Đây là '
        'lỗi phân biệt positive/protected negative tái xuất hiện trên development khác role.',
        '- **Break=0 không có nghĩa FAR=0**: nếu baseline đã sai trên negative, '
        'patch nhận nhầm không tạo thêm break nhưng vẫn là false activation.',
        '- Patch suspended 9 ← 89/class 12 có 163 break trong shadow; '
        'route này đang tắt, các break đó không phải thiệt hại đã xảy ra trong full-test deployed.',
        '- Các patch class 6 dùng cùng donor 30 và cùng positive pool, '
        'không phải ba replication độc lập. 59/64 là point estimate trên panel nhỏ; '
        'không được suy ra population recall chắc chắn <95% hoặc toàn head-patch thất bại.',
        '- Scope ở đây gồm các class có owned BASE support trong live neighborhood và donor gốc. '
        'Không chứng nhận mọi class ngoài scope hoặc toàn provenance lịch sử của aggregation.', '',
        '## Quyết định', '',
        '- Reject fixture **65 ← 88/class 20 tại Task 3** theo audit `3bda3f5`; '
        'không biến thành blacklist mọi checkpoint hoặc mọi donor cho class 20.',
        '- Chưa chạy full training. Chưa tăng transaction budget, giảm gate, đổi guard hoặc production runner.',
        '- Tách **donor native competence**, **receiver transfer compatibility**, '
        '**protected-negative evidence coverage** và **current CAL acceptance**. '
        'Không coi metadata ownership/maturity hoặc zero FP do thiếu mẫu là safe transfer.',
        '- Nếu thiếu scope hoặc native router làm screen quá bảo thủ, cần evidence/qualification '
        'phù hợp với imported route trước khi kết luận phải thay toàn bộ transfer design.', '',
        '## Integrity', '',
        f'- Models/routers bất biến: `{summary["all_models_unchanged"]}`.',
        f'- Content splits disjoint: `{summary["content_splits_disjoint"]}`.',
        f'- Installed head exact + query dependencies mature: **{len(head_controls)}/{len(head_controls)}**.',
        '- Raw historical CAL reads=0; test reads=0; installations=0; new certificates=0.',
        '- Không chọn lại donor/threshold trên evaluation.',
        '- Communication mới chưa đo; CPU/local simulation không phải triển khai decentralized mới.', '',
        f'Artifact tổng hợp: `{a.artifact.as_posix()}`.', '']
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','artifact','report'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--coverage-metadata',type=Path)
    p.add_argument('--checkpoint',type=Path,required=True)
    run(p.parse_args())
