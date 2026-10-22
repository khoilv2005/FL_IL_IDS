"""Create a compact receipt/report for low-accuracy client diagnostics."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.hashsalt'] = 'candle_v2_client32_T3_20261022'
import matplotlib.pyplot as plt
from fed_learning.data.denice_clean_roles import file_sha256
from tools.debug_candle_v2_checkpoints import write


def compact_metric(value):
    return {key: val for key, val in value.items() if key not in ('confusion', 'source_rows', 'row_state')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--low', type=Path, required=True)
    p.add_argument('--rounds', type=Path, required=True)
    p.add_argument('--components', type=Path, required=True)
    p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    low = json.loads((a.low/'completion.json').read_text())
    trajectory = json.loads((a.rounds/'completion.json').read_text())
    components = json.loads((a.components/'completion.json').read_text())
    v1 = json.loads((ROOT/'audit_denice/appliance_legacy_results11/inputs/legacy_self_task_5/client_metrics.json').read_text())
    v1 = {r['receiver']: r for r in v1}
    with zipfile.ZipFile('C:/Users/khoak/Downloads/results (14).zip') as z:
        debug = json.loads(z.read(next(n for n in z.namelist() if n.endswith('/denice_debug_history.json'))))
        train = [json.loads(v) for v in z.read(next(n for n in z.namelist() if n.endswith('/candle_v2_training.jsonl'))).decode().splitlines()]
    train = {(r['task'], r['round'], r['client_id']): r for r in train}
    result = dict(completed=True, selected_client_ids=low['selected_client_ids'], sources=low['sources'],
        protocols={key: json.loads((path/'protocol_before_data.json').read_text())
                   for key, path in [('low', a.low), ('rounds', a.rounds), ('components', a.components)]},
        no_optimizer=True, no_new_TEST_inference=True, clients=[], preservation=[])
    for client in low['clients']:
        cid = client['client_id']; final = client['checkpoint_diagnostics']['5']
        r = dict(client_id=cid, full_TEST=client['full_TEST'], training=client['training'],
                 historical_v1_full_TEST={k: dict(accuracy=v['accuracy'], macro_f1=v['macro_f1_34'])
                    for k, v in v1[cid]['metrics'].items()},
                 owner_validation_T5={k: compact_metric(v) for k, v in final['methods'].items()},
                 owner_validation_birth={}, graph=[],
                 global_tail_development={k: compact_metric(v) for k, v in client['prior_global_development']['tail'].items()})
        end = np.load(a.low/f'T5_client_{cid}_owner_validation.npz')
        for task, diagnostic in client['checkpoint_diagnostics'].items():
            if task == '5':
                continue
            birth = np.load(a.low/f'T{task}_client_{cid}_owner_validation.npz')
            ix = end['y']//6 == int(task)
            if not np.array_equal(birth['y'], end['y'][ix]):
                raise ValueError('Owner validation order changed')
            result['preservation'].append(dict(client_id=cid, task=int(task), rows=int(ix.sum()),
                oracle_prediction_disagreements=int((birth['oracle_task'] != end['oracle_task'][ix]).sum())))
            r['owner_validation_birth'][task] = {k: compact_metric(v) for k, v in diagnostic['methods'].items()}
        for task in range(6):
            rows = []
            for d in debug:
                if d.get('task') != task or 'cluster' not in d or 'round' not in d:
                    continue
                alpha = d['cluster']['alpha_debug'].get(str(cid))
                if alpha is None:
                    continue
                peers = [int(peer) for peer, weight in zip(alpha['group_ids'], alpha['alphas']) if int(peer) != cid and weight > 0]
                rows.append(dict(round=d['round'], positive_peer_ids=peers,
                    self_alpha=float(sum(weight for peer, weight in zip(alpha['group_ids'], alpha['alphas']) if int(peer) == cid)),
                    class21_peer_owners=[peer for peer in peers if 21 in train[(task, d['round'], peer)]['local_classes']]))
            if rows:
                r['graph'].append(dict(task=task, rounds=rows,
                    self_only_rounds=sum(not row['positive_peer_ids'] for row in rows),
                    self_alpha_min=min(row['self_alpha'] for row in rows),
                    self_alpha_max=max(row['self_alpha'] for row in rows)))
        result['clients'].append(r)
    result['client32_round_trajectory'] = [dict(round=r['round'],
        oracle_accuracy=r['oracle']['accuracy'], class21_recall=r['class21_recall'], class21_to23=r['class21_to23'],
        class21_margin_vs23_mean=r['class21_margin_vs23_mean'], row21_rank=r['row21_rank'],
        real_CE=r['training_real_CE'], self_alpha=r['self_alpha'], positive_peer_count=r['positive_peer_count'])
        for r in trajectory['rows']]
    result['round_precision_control'] = trajectory['fp16_R19_vs_exact_FP32']
    result['component_counterfactuals'] = {k: {name: value for name, value in r.items()
        if name not in ('per_class_correct', 'class_counts')} for k, r in components['controls'].items()}
    result['component_compatibility'] = {k: components[k] for k in ('active_layout_matches', 'row_mask_equal',
        'feature_cosine_mean', 'feature_cosine_class21_mean', 'head_row_cosine')}
    n = 13505771
    result['full_TEST_improvement_upper_bounds_not_predictions'] = dict(
        fixed_availability_delta_accuracy=sum(r['full_TEST']['covered_errors'] for r in result['clients'])/n,
        unlimited_perfect_three_models_delta_accuracy=sum(r['full_TEST']['samples']-r['full_TEST']['correct'] for r in result['clients'])/n)
    result['limitations'] = [
        'Selected clients using already-observed TEST; new diagnostics read validation only.',
        'Owner validation and global balanced development have different distributions; no direct accuracy equivalence.',
        'High pooled owner accuracy can hide rare-class failures.',
        'Round archives use FP16 reconstruction; R19 checked against exact FP32 terminal.',
        'Both head and encoder change; component mixing is not a clean local-training versus aggregation ablation.',
        'Neighborhood alpha is not the effective per-coordinate contribution after label/age/connection masks.',
        'No production behavior changed; no core bug proven solely from missing task participation.',
    ]
    trace = result['client32_round_trajectory']
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True, constrained_layout=True)
    rounds = [r['round'] for r in trace]
    axes[0].plot(rounds, [100*r['oracle_accuracy'] for r in trace], label='Owner validation accuracy (oracle task)', color='#1769aa', marker='o', markersize=3)
    axes[0].plot(rounds, [100*r['class21_recall'] for r in trace], label='Class 21 recall (oracle task)', color='#c45c00', marker='o', markersize=3)
    axes[0].set(ylabel='Accuracy / recall (%)', ylim=(-3, 103), title='Client 32, Task 3: collapse within the same task')
    axes[0].legend(loc='lower left', fontsize=8)
    axes[1].plot(rounds, [r['class21_margin_vs23_mean'] for r in trace], color='#6b4697', marker='o', markersize=3)
    axes[1].axhline(0, color='#555555', linewidth=.8, linestyle='--')
    axes[1].set_ylabel('Mean score21 - score23')
    axes[1].set_title('Margins on owner-validation class 21 inputs', fontsize=9)
    axes[2].plot(rounds, [r['real_CE'] for r in trace], color='#b13c4b', marker='o', markersize=3)
    axes[2].set(xlabel='Task 3 round (post-aggregation checkpoint)', ylabel='Local training real CE', xticks=[0, 4, 7, 11, 15, 19])
    for axis in axes:
        axis.grid(alpha=.2)
    stem = a.artifact.parent/'candle_v2_client32_T3_collapse_20261022'
    fig.savefig(stem.with_suffix('.png'), dpi=160)
    fig.savefig(stem.with_suffix('.svg'), metadata={'Date': None})
    plt.close(fig)
    svg = stem.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')
    result['figures'] = {ext: dict(path=str(stem.with_suffix('.'+ext)), sha256=file_sha256(stem.with_suffix('.'+ext))) for ext in ('png', 'svg')}
    result['summary_source_sha256'] = file_sha256(__file__)
    write(a.artifact, result)
    pct = lambda v: f'{100*v:.2f}%'
    clients = {r['client_id']: r for r in result['clients']}
    lines = ['# CANDLE V2 — debug clients có FULL TEST accuracy dưới 10%', '',
        'Ngày 22/10/2026. Policy hiện tại `binary_cosine_registered_v1`, checkpoint T5/R19.',
        'Đã kiểm tra đúng 3/98 client dưới 10%. Không optimizer, không chạy thêm TEST, không sửa training.', '',
        '## 1. Danh sách và trần do coverage', '',
        '| Client | V2 TEST Accuracy | TEST Macro-F1 | TEST route accuracy | TEST coverage / trần accuracy khi giữ availability | V1 Multiclass TEST |',
        '|---|---:|---:|---:|---:|---:|']
    for cid in low['selected_client_ids']:
        r = clients[cid]; t = r['full_TEST']
        lines.append(f'| {cid} | {pct(t["accuracy"])} | {pct(t["macro_f1"])} | {pct(t["route_accuracy"])} | {pct(t["coverage"])} | {pct(r["historical_v1_full_TEST"]["MulticlassSelf"]["accuracy"])} |')
    lines += ['', 'Coverage được tính bằng số TEST rows có true class nằm trong saved available classes; ',
        'empty-scope fallback bằng 0 ở full run. Các số TEST là kết quả đã lưu của results (15).',
        'Không thể quy toàn bộ sai số của ba client cho cùng một lỗi router.', '',
        '## 2. Client 38 — thiếu coverage là giới hạn lớn nhất', '',
        '- Chỉ train local T0, T4, T5; không có current BASE ở T1, T2, T3. Saved availability ',
        '  gồm 0–5, 24–33, khớp những task đã tham gia và imports trong task đó.',
        '- Các class đã available chỉ phủ 7,07% TEST. Vì vậy **không thể đạt 10%** trên ',
        '  shard này nếu giữ nguyên availability, kể cả classifier/router hoàn hảo trên phần đã phủ.',
        '- Owner validation T5: Binary 62,87%, oracle-task 72,36%. Global balanced tail: ',
        '  Binary 10,04%, Multiclass 12,90%, oracle-task 18,67%. Những pool này khác phân bố.',
        '- Global tail class 24: oracle recall 100%, Binary recall 0%, route-to-T4 0%. ',
        '  Router chặn một capability có sẵn, nhưng sửa router không tự tạo được T1–T3 coverage.',
        '- T5 current BASE trước internal split chỉ có 19 rows: class 31 = 1, class 33 = 18. ',
        '  Final real CE 5,56, không có AMP skip ở T4/T5. Không thể gọi đây là training khỏe ',
        '  chỉ vì một số class imported đã được đánh dấu available.', '',
        '**Diễn giải:** Non-IID + vắng task và giới hạn temporal transfer, chưa phải bug mất đăng ký. ',
        'Không tự mở class chưa học hay dùng lại dữ liệu train cũ để lấp coverage.', '',
        '## 3. Client 32 — regression thật xảy ra ngay trong Task 3', '',
        '- V1 Multiclass TEST 20,56% → V2 Binary TEST 8,68%.',
        '- Vắng T1/T2, nên coverage chỉ 27,99%; ngoài ra model đã phân loại sai cả dữ liệu riêng.',
        '- Exact FP32 terminal T3: owner validation 5.382 rows, oracle-task accuracy **6,93%**.',
        '- Cả **3.797/3.797** owner-validation class 21 bị dự đoán thành class 23, dù biết đúng task.', '',
        '### 3.1. Trace 20 round đã lưu', '',
        '| T3 round | Owner oracle accuracy | Class 21 recall | Mean score21 − score23 trên class 21 | Local training real CE |',
        '|---|---:|---:|---:|---:|']
    for idx in (0, 4, 7, 8, 11, 19):
        r = result['client32_round_trajectory'][idx]
        lines.append(f'| {idx} | {pct(r["oracle_accuracy"])} | {pct(r["class21_recall"])} | {r["class21_margin_vs23_mean"]:+.3f} | {r["real_CE"]:.3f} |')
    lines += ['', '![Client 32 Task 3 trajectory](../artifacts/candle_v2_client32_T3_collapse_20261022.png)', '',
        'Đây là **collapse trong cùng task**, không phải chỉ forgetting sau khi chuyển task. ',
        'Row 21 vẫn rank 1 qua R0–R19, optimizer vẫn có steps, T3 không có AMP skip và còn ',
        'learner/free capacity. Vì vậy chưa thấy dấu hiệu row bị freeze sớm hoặc backbone hết capacity.', '',
        'Trên 8 cặp client/task được so bằng cùng owner-validation rows và cùng oracle task mask, ',
        'prediction ở checkpoint cuối task và tại T5 **khớp 100%**. Client 32 đã học hỏng T3 ',
        'trước khi consolidation; các task tiếp theo giữ nguyên những prediction sai đó.', '',
        'Round checkpoints là FP16 base/deltas. Ở R19, oracle prediction chỉ lệch 1/5.382 rows ',
        'so với exact FP32 terminal; cả hai đều class-21 recall 0%. Collapse lớn không được giải ',
        'thích bởi sai số lưu checkpoint. Không gọi R4 là best checkpoint hợp lệ cho TEST: R4 ',
        'được chọn sau khi xem trajectory để chẩn đoán.', '',
        '### 3.2. Tách head, encoder và BN', '',
        'Cùng owner validation T3 và cùng 6 class được phép cạnh tranh:', '',
        '| Features / head | Oracle class accuracy | Class 21 recall |', '|---|---:|---:|']
    labels = [('h4_W4','R4 features + R4 head'),('h4_W19','R4 features + R19 head'),
              ('h19_W4','R19 features + R4 head'),('h19_W19','R19 features + R19 head'),
              ('h19_R4_BN_buffers_W19','R19 model + R4 BN running statistics'),
              ('h19_R4_BN_affine_and_buffers_W19','R19 model + R4 BN affine/statistics')]
    for key, label in labels:
        cell = result['component_counterfactuals'][key]
        lines.append(f'| {label} | {pct(cell["accuracy"])} | {pct(cell["class21_recall"])} |')
    lines += ['', 'FC1 active layout và FC2 connection masks khớp giữa hai trạng thái. Cả head lẫn ',
        'encoder đã lệch; thay riêng BN cũ không phục hồi. Source models giữ nguyên hash.',
        'R4 là reconstructed FP16, R19 là exact FP32; đây là functional counterfactual, ',
        'không phải một training ablation sạch.', '',
        'Self-alpha của Client 32 chỉ khoảng 5–8% trong neighborhood ở các round được nêu. ',
        'Đó là group weights, **không phải** tỷ lệ tác động thực tế lên từng row sau masks. ',
        'Combined local-update/aggregation dynamics là vùng cần kiểm tra. Không có snapshot ',
        'pre-aggregation local model, nên chưa thể kết luận peer aggregation hay local objective ',
        'là nguyên nhân duy nhất. Không được tự đổi alpha, xi hoặc loss dựa trên kết quả này.', '',
        '## 4. Client 47 — local fit tốt, global coverage và imported-class discrimination kém', '',
        '- Owner validation T5: Binary **90,29%**, oracle-task **99,12%** trên 5.909 rows.',
        '- Global balanced tail: Binary 15,29%, Multiclass 17,31%, oracle-task 24,02%.',
        '- Vắng T1/T4; class 21 cũng chưa available. TEST coverage 48,43%.',
        '- Task 3 chỉ có peer 89 ở R0/R1; **18/20 rounds self-only**, không có eligible ',
        '  class-21 donor trong các neighborhoods đã ghi. Vì vậy class 21 không được import.',
        '- Đã đăng ký imported classes 12/13/14, nhưng global-tail oracle recall của cả ba ',
        '  là **0%**: tất cả 64 mẫu mỗi class bị dự đoán thành class 15. Class 17 chỉ 9,38%. ',
        '  Mở mask hay đổi router không sửa được các lỗi phân biệt trong đúng task này.',
        '- Class 19 chỉ nhận virtual training ở 2 rounds; sau đó vẫn available nhưng global-tail ',
        '  oracle recall 0%. Availability đang chứng minh có optimizer step, không chứng minh competence.',
        '- Owner accuracy 99,12% bị chi phối bởi các class phổ biến. Nó không chứng minh mọi ',
        '  class local hoặc imported đều tốt; các class hiếm vẫn có failure.', '',
        '**Diễn giải:** coverage, graph isolation, Non-IID generalization và prototype-to-input gap. ',
        'Không có bằng chứng model quên toàn bộ kiến thức local.', '',
        '## 5. Cần làm gì tiếp theo?', '',
        '1. Ưu tiên Client 32/T3, khoảng R5–R11: lần diagnostic native tiếp theo cần ghi ',
        '   cùng current validation trước local, sau local, sau native aggregation, sau ',
        '   prototype-row restore và sau BN handling. Giữ nguyên config để định vị thao tác gây hại.',
        '2. So sánh có kiểm soát self-only/native aggregation và V2 loss components sau khi ',
        '   có boundary receipts; hiện chưa đủ căn cứ chọn một bản sửa thuật toán.',
        '3. Với 38/47, báo cáo riêng coverage và class competence; không tự unmask class ',
        '   thiếu bằng chứng hoặc gọi đặc điểm Non-IID là lỗi implementation.',
        '4. Chưa cần full retrain. Audit này không thay production, không dùng TEST để tune.', '',
        'Chỉ sửa classifier/router trong availability hiện tại của ba client có trần gain toàn ',
        f'hệ thống **+{100*result["full_TEST_improvement_upper_bounds_not_predictions"]["fixed_availability_delta_accuracy"]:.3f} pp accuracy**. ',
        'Nếu giả sử cả ba model dự đoán hoàn hảo mọi class, trần gain là ',
        f'**+{100*result["full_TEST_improvement_upper_bounds_not_predictions"]["unlimited_perfect_three_models_delta_accuracy"]:.3f} pp**. ',
        'Đây là upper bounds, không phải dự báo; ba client xấu nhất không giải thích toàn bộ ',
        'accuracy 35,70% của hệ thống.', '',
        '## 6. Receipts', '',
        f'- Owner/birth diagnostics: `{a.low.as_posix()}`.',
        f'- Round trace: `{a.rounds.as_posix()}`.',
        f'- Component controls: `{a.components.as_posix()}`.',
        '- Tools: `debug_candle_v2_low_accuracy_clients.py`, `debug_candle_v2_client32_rounds.py`, ',
        '  `debug_candle_v2_client32_components.py`, `summarize_candle_v2_low_accuracy_clients.py`.',
        '- Member SHA checks, data-role identity và model hashes trước/sau đã kiểm tra.',
        '- Historical validation chỉ vào offline evaluator; không raw replay trong training.',
        '- Một seed, selected-worst analysis trên development đã quan sát; không có untouched confirmation.', '']
    a.report.write_text('\n'.join(line.rstrip() for line in lines), encoding='utf-8')
    print(json.dumps(dict(artifact=str(a.artifact), report=str(a.report),
                         upper_bounds=result['full_TEST_improvement_upper_bounds_not_predictions']), indent=2))


if __name__ == '__main__':
    main()
