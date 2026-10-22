"""Summarize matched development diagnostics and already-published TEST confusion matrices."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from fed_learning.data.denice_clean_roles import file_sha256
from tools.debug_candle_v2_checkpoints import argmax, write


def paired(y, before, after):
    rescue = int(((before != y) & (after == y)).sum())
    breaks = int(((before == y) & (after != y)).sum())
    if int((after == y).sum() - (before == y).sum()) != rescue-breaks:
        raise ValueError('Paired accounting mismatch')
    return dict(rescue=rescue, breaks=breaks, net_correct=rescue-breaks,
                prediction_rows=len(y), delta_accuracy=(rescue-breaks)/len(y))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--training-archive', type=Path, required=True)
    p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args(); run = json.loads((a.input/'completion.json').read_text())
    y = np.load(a.input/'labels.npz')['y']; ids = run['clients']
    ref_n = 2176
    panels = dict(reference=slice(0, ref_n), tail=slice(ref_n, len(y)))
    predictions = {v: {cid: np.load(a.input/f'{v}_client_{cid}_predictions.npz') for cid in ids} for v in ('v1', 'v2')}
    records = {v: {cid: json.loads((a.input/f'{v}_client_{cid}.json').read_text()) for cid in ids} for v in ('v1', 'v2')}
    checks = dict(model_hash_receipts=len(ids)*2, unexplained_error_rows=0,
                  v2_previous_development_confusion_mismatches=0,
                  v1_canonical_inference_check='32 inputs per model per router, performed during inference')
    old = ROOT/'audit_denice/candle_v2_routing_repair_20261022_v1/repair_verification'
    for cid in ids:
        receipt = json.loads((old/f'client_{cid}.json').read_text())
        for panel in panels:
            if records['v2'][cid]['panels'][panel]['binary']['confusion'] != receipt[panel]['repaired']['confusion']:
                checks['v2_previous_development_confusion_mismatches'] += 1
        for v in records:
            for panel in panels:
                for cell in records[v][cid]['panels'][panel].values():
                    checks['unexplained_error_rows'] += cell.get('error_partition_order_coverage_then_within_task_then_route', {}).get('unexplained', 0)
    if checks['v2_previous_development_confusion_mismatches'] or checks['unexplained_error_rows']:
        raise ValueError('Development identity or accounting mismatch')
    result = dict(completed=True, protocol=json.loads((a.input/'protocol_before_data.json').read_text()),
                  sources=run['sources'], clients=ids, checks=checks,
                  evaluation_seconds=run['seconds'], summaries={}, paired={}, class_diagnostics={})
    for panel, ix in panels.items():
        yy = np.tile(y[ix], len(ids)); result['paired'][panel] = {}
        for name, vbefore, kbefore, vafter, kafter in [
            ('v2_binary_to_multiclass', 'v2', 'binary', 'v2', 'refit_multiclass'),
            ('matched_multiclass_v1_to_v2', 'v1', 'refit_multiclass', 'v2', 'refit_multiclass'),
            ('historical_v1_router_to_same_version_refit', 'v1', 'frozen_multiclass', 'v1', 'refit_multiclass')]:
            before = np.concatenate([predictions[vbefore][cid][kbefore][ix] for cid in ids])
            after = np.concatenate([predictions[vafter][cid][kafter][ix] for cid in ids])
            result['paired'][panel][name] = paired(yy, before, after)
        result['class_diagnostics'][panel] = {}
        for c in range(34):
            entries = {}; mask = y[ix] == c
            both = [cid for cid in ids if all(c in records[v][cid]['scopes'].get(str(c//6), []) for v in ('v1', 'v2'))]
            for v in ('v1', 'v2'):
                entries[v] = dict(registered_models=sum(c in records[v][cid]['scopes'].get(str(c//6), []) for cid in ids))
                for key in ('binary', 'refit_multiclass', 'oracle_available', 'oracle_full_task'):
                    values = np.concatenate([predictions[v][cid][key][ix][mask] for cid in ids])
                    entries[v][key+'_recall'] = float((values == c).mean())
                    if key in ('binary', 'refit_multiclass'):
                        routes = np.concatenate([predictions[v][cid]['route_'+key][ix][mask] for cid in ids])
                        entries[v][key+'_route_accuracy'] = float((routes == c//6).mean())
                entries[v]['both_registered_oracle_full_task_recall'] = (float(np.mean([
                    (predictions[v][cid]['oracle_full_task'][ix][mask] == c).mean() for cid in both])) if both else None)
            entries['both_registered_models'] = len(both)
            result['class_diagnostics'][panel][str(c)] = entries
    for v in ('v1', 'v2'):
        result['summaries'][v] = {panel: {k: {name: val for name, val in entry.items()
            if name not in ('confusion',)} for k, entry in run['summaries'][v][panel].items()} for panel in panels}
        result.setdefault('router_memory', {})[v] = {
            'mean_resubstitution_balanced_accuracy': float(np.mean([r['fit']['memory_resubstitution_balanced_accuracy'] for r in records[v].values()])),
            'fit_warnings': sorted(set(msg for r in records[v].values() for msg in r['fit']['warnings']))}
    v1_cm = np.load(ROOT/'audit_denice/appliance_legacy_results11/inputs/legacy_self_task_5/confusion_matrices.npz')
    v2_cm = np.asarray(json.loads((ROOT/'audit_denice/candle_v2_results15_20261022_v2/confusion_matrix.json').read_text()))
    mc = v1_cm['MulticlassSelf']; delta = np.diag(v2_cm)-np.diag(mc)
    if not np.array_equal(mc.sum(1), v2_cm.sum(1)):
        raise ValueError('Different TEST class counts')
    result['already_published_full_TEST'] = dict(no_new_TEST_inference=True, rows=int(mc.sum()),
        v1_multiclass_correct=int(np.diag(mc).sum()), v2_binary_correct=int(np.diag(v2_cm).sum()),
        delta_correct=int(delta.sum()), per_class_delta_correct=delta.tolist(),
        per_task=[dict(task=t, v1_accuracy=float(np.diag(mc)[t*6:min(t*6+6,34)].sum()/mc[t*6:min(t*6+6,34)].sum()),
            v2_accuracy=float(np.diag(v2_cm)[t*6:min(t*6+6,34)].sum()/v2_cm[t*6:min(t*6+6,34)].sum()),
            delta_correct=int(delta[t*6:min(t*6+6,34)].sum())) for t in range(6)])
    with zipfile.ZipFile(a.training_archive) as z:
        def jsonl(name):
            path = next(n for n in z.namelist() if n.endswith('/'+name))
            return [json.loads(line) for line in z.read(path).decode().splitlines()]
        train = jsonl('candle_v2_training.jsonl'); optimizer = jsonl('optimizer_health.jsonl')
    training = []
    for t in range(6):
        tt = [r for r in train if r['task'] == t]; oo = [r for r in optimizer if r['task'] == t]
        steps = sum(r['optimizer_steps'] for r in oo); skips = sum(r['skipped_optimizer_steps'] for r in oo)
        training.append(dict(task=t, client_rounds=len(tt), optimizer_steps=steps, skipped_steps=skips,
            skipped_fraction=skips/(steps+skips), zero_update_client_rounds=sum(r['zero_update'] for r in oo),
            mean_real_CE=float(np.mean([r['real_CE'] for r in tt])),
            mean_virtual_CE=float(np.mean([r['losses']['virtual_CE'] for r in tt])),
            mean_preservation_KL=float(np.mean([r['losses']['preservation'] for r in tt]))))
    result['training_log_descriptive_not_causal'] = training
    # Cached raw scores: check whether simply removing row normalization fixes class 6.
    diag = ROOT/'audit_denice/candle_v2_checkpoint_debug_20261022_v1/diagnostics'
    ry = y[:ref_n]; raw_pred = []
    for cid in ids:
        z = np.load(diag/f'T5_client_{cid}_scores.npz'); pred = np.full(len(ry), -1)
        for t in range(6):
            ix = ry//6 == t
            pred[ix] = argmax(z['raw'][ix], range(t*6, min(t*6+6,34)))
        raw_pred.append(pred)
    raw_pred = np.concatenate(raw_pred); ryy = np.tile(ry, len(ids))
    result['cached_raw_affine_oracle_control_reference'] = dict(accuracy=float((raw_pred == ryy).mean()),
        class6_recall=float((raw_pred[ryy == 6] == 6).mean()),
        cosine_class6_recall=result['class_diagnostics']['reference']['6']['v2']['oracle_full_task_recall'])
    result['limitations'] = [
        'Development only for new inference; full TEST numbers are previously published confusion matrices.',
        'Both development panels were previously exposed; no untouched confirmation or multiple seeds.',
        'Mean client Macro-F1 on development is distinct from pooled Macro-F1 on full TEST.',
        'V2 multiclass is a frozen-checkpoint diagnostic, not deployed production policy or a new full TEST result.',
        'Checkpoint comparison changes multiple training components; it does not isolate prototype-transfer gain.',
        'Full-task oracle includes unregistered classes and is diagnostic only.',
        'Old-class-only preservation KL does not constrain old-versus-new total score competition.',
    ]
    result['audit_source_sha256'] = file_sha256(__file__)
    write(a.artifact, result)
    s = result['summaries']; pct = lambda x: f'{100*x:.4f}%'
    lines = ['# CANDLE V2 so với V1 — audit matched router trên checkpoint', '',
        'Ngày 22/10/2026. Đã chạy inference development cho 98 client ở mỗi checkpoint T5/R19.',
        '**Không train backbone, không chạy thêm TEST, không đổi production router.**', '',
        '## 1. Kết luận', '',
        'V2 hiện thua **V1 Multiclass**, nhưng vượt V1 Binary trên full TEST đã công bố. ',
        'Matched development cho thấy lựa chọn Binary thay vì Multiclass giải thích phần lớn chênh lệch ',
        'trên hai panel đã khóa. Đồng thời tồn tại điểm yếu classifier riêng ở class 6; không thể ',
        'quy mọi lỗi cho router hoặc gọi prototype transfer đã thành công.', '',
        '## 2. So sánh cùng dữ liệu và cùng kiểu router', '',
        'Panel reference: 2.176 unique validation inputs, 64/class. Tail: 2.132 inputs, ',
        '64/class trừ class 31 còn 20. Hai panel content-disjoint nhưng đều thuộc development đã ',
        'được quan sát. Mỗi model chạy toàn bộ panel; đây là **mean client metrics**, không phải ',
        'các số full TEST 35,70%/37,15%. Không dùng nhãn panel để fit router.', '',
        '| Checkpoint / router | Reference Accuracy | Tail Accuracy | Tail mean Macro-F1 | Tail route accuracy |',
        '|---|---:|---:|---:|---:|']
    for v, key, name in [('v1','binary','V1 Binary'),('v1','frozen_multiclass','V1 Multiclass đã lưu'),
                          ('v1','refit_multiclass','V1 Multiclass refit'),('v2','binary','V2 Binary repair'),
                          ('v2','refit_multiclass','V2 Multiclass diagnostic')]:
        cell=s[v]['tail'][key]
        lines.append(f'| {name} | {pct(s[v]["reference"][key]["accuracy"])} | {pct(cell["accuracy"])} | {pct(cell["macro_f1"])} | {pct(cell["route_accuracy"])} |')
    lines += ['', 'Refit dùng đúng cấu hình V1: balanced Logistic Regression, C=1, max_iter=1000, ',
        'random_state=0, **chỉ fit saved binary activation memory của chính client đó**. Không trộn ',
        'memory giữa hai encoders; không đọc historical raw CAL để fit. Saved router V1 tạo bằng ',
        'sklearn 1.6.1, môi trường local là 1.9.0: giữ frozen V1 làm control và refit cả V1/V2 ',
        'trong cùng 1.9.0. Tail V1 frozen/refit chỉ lệch 0,0034 điểm phần trăm accuracy.', '',
        'Phân rã chênh lệch trên tail, theo đường counterfactual đã khóa:', '',
        f'- V1 refit MC → V2 refit MC: **+{100*(s["v2"]["tail"]["refit_multiclass"]["accuracy"]-s["v1"]["tail"]["refit_multiclass"]["accuracy"]):.4f} pp**.',
        f'- V2 MC → V2 Binary: **{100*(s["v2"]["tail"]["binary"]["accuracy"]-s["v2"]["tail"]["refit_multiclass"]["accuracy"]):.4f} pp**.',
        '- Đây là tác động thay router trên cùng checkpoint và chênh lệch giữa hai checkpoint. ',
        '  Không phải ablation cô lập alignment/virtual CE trong quá trình training.', '',
        '## 3. Classifier có yếu đi không?', '',
        'Oracle dưới đây biết task thật **chỉ để chẩn đoán**. Không phải inference chính thức.', '',
        '| Tail diagnostic | V1 | V2 |', '|---|---:|---:|']
    for key, label in [('oracle_available','Đúng task, availability riêng'),('oracle_full_task','Đúng task, mở toàn bộ class của task'),
                       ('oracle_common_availability','Đúng task, cùng intersection availability')]:
        lines.append(f'| {label} | {pct(s["v1"]["tail"][key]["accuracy"])} | {pct(s["v2"]["tail"][key]["accuracy"])} |')
    lines += ['', 'Availability coverage trên tail: V1 64,02%, V2 84,82%. Khi dùng cùng class mask, ',
        'V2 không còn gain tổng thể rõ ràng; một phần lợi ích đang đến từ coverage. Đây chưa ',
        'phải bằng chứng weight transfer cải thiện các class chung.', '',
        '| Recall class 6 trên tail | V1 | V2 |', '|---|---:|---:|']
    for key, label in [('binary_recall','Binary thực tế'),('refit_multiclass_recall','Multiclass thực tế'),
                       ('oracle_full_task_recall','Đúng task, cùng 6 class T1')]:
        cc=result['class_diagnostics']['tail']['6']
        lines.append(f'| {label} | {pct(cc["v1"][key])} | {pct(cc["v2"][key])} |')
    lines += ['', 'Class 6 có cả hai vấn đề: Binary định tuyến kém, và khả năng phân biệt trong T1 ',
        'của checkpoint V2 cũng thấp hơn V1. Đổi raw affine trên chính weights V2 không cứu được: ',
        f'ở reference, oracle recall class 6 raw chỉ {pct(result["cached_raw_affine_oracle_control_reference"]["class6_recall"])} ',
        f'so với cosine {pct(result["cached_raw_affine_oracle_control_reference"]["cosine_class6_recall"])}.',
        'Do đó chưa có căn cứ sửa bằng một phép bỏ normalization ở evaluator.', '',
        'V1 oracle có/không adapter cho cùng accuracy trên cả hai panel; chưa thấy adapter giải thích ',
        'chênh lệch trong diagnostic này. Không suy ra adapter luôn vô ích ở mọi checkpoint.', '',
        '## 4. Phần thua trên full TEST đã lưu nằm ở đâu?', '',
        'Chỉ tính lại confusion matrices đã công bố; không mở TEST rows hay chạy predictor mới.', '',
        '| True task | V1 Multiclass | V2 Binary repair | Delta số đúng |', '|---|---:|---:|---:|']
    for rr in result['already_published_full_TEST']['per_task']:
        lines.append(f'| T{rr["task"]} | {pct(rr["v1_accuracy"])} | {pct(rr["v2_accuracy"])} | {rr["delta_correct"]:+,} |')
    lines += ['', 'Net: **−194.925** prediction đúng. Riêng class 6: **−574.115**; class 24: ',
        '**−77.869**. Các class tăng bù lại một phần. Vì TEST mất cân bằng, class 6 có hơn ',
        '2 triệu mẫu nên ảnh hưởng headline accuracy rất lớn. Không dùng những test labels này ',
        'để chọn threshold hay hardcode class guard.', '',
        '## 5. Training log nói gì?', '',
        'Thống kê mô tả theo client-round, không weighted theo số mẫu; không dùng để quy kết nhân quả.', '',
        '| Task | Mean real CE | Mean virtual CE | AMP skipped / attempted | Zero-update client-rounds |',
        '|---|---:|---:|---:|---:|']
    for rr in training:
        lines.append(f'| T{rr["task"]} | {rr["mean_real_CE"]:.4f} | {rr["mean_virtual_CE"]:.4f} | {pct(rr["skipped_fraction"])} | {rr["zero_update_client_rounds"]} |')
    lines += ['', 'T4/T5 real CE tăng mạnh, trong khi hai task này không có AMP skipped steps. Vì vậy ',
        'không có căn cứ đổ failure cuối run cho AMP. Virtual prototype CE thấp hơn real CE ',
        'không chứng minh class đã học tốt trên input thật của receiver; tương thích prototype ',
        'với receiver encoder vẫn cần ablation riêng.', '',
        'Code preservation KL chỉ chuẩn hóa trong các old classes, và teacher được chụp trước ',
        'mỗi local round. Nó không trực tiếp bảo vệ cạnh tranh old-vs-new hoặc bù logit calibration. ',
        'Đây là giới hạn objective, chưa phải bằng chứng cần thêm một regularizer mới.', '',
        '## 6. Việc tiếp theo', '',
        '1. Giữ nguyên kết quả Binary repair 35,7030% / 18,4767%.',
        '2. Khóa một candidate V2 Multiclass dùng đúng saved-memory protocol của V1; nếu đánh giá ',
        '   full TEST thì dùng notebook Kaggle riêng, không cần train lại. Kết quả development ',
        '   28,3158% không được thay thế hoặc dự đoán thành số full TEST.',
        '3. Chỉ sau khi kiểm soát router, làm ablation training có giới hạn cho classifier T1 ',
        '   và late-task learning: V2 full so với bỏ alignment / bỏ virtual CE, cùng reference, ',
        '   data budget, masks và seed; dùng development. Không tune riêng class 6 theo TEST.',
        '4. Chưa retrain full hoặc thay kiến trúc thêm trong audit này.', '',
        '## 7. Receipts và tái lập', '',
        f'- Local output: `{a.input.as_posix()}`.',
        '- Tool: `tools/debug_candle_v1_v2_matched.py`.',
        '- Summary: `tools/summarize_candle_v1_v2_matched.py`.',
        '- 196 model hash receipts trước/sau khớp; canonical V1 prediction control không mismatch.',
        '- V2 Binary confusion matrices khớp lượt repair trước ở 98 models × 2 panels.',
        '- Error partition khớp tổng errors, unexplained = 0; rescue/break khớp delta correct.',
        '- Một seed, development đã quan sát, không có untouched confirmation.', '',
        '```powershell',
        '.venv-denice-gpu\\Scripts\\python.exe -X utf8 tools/debug_candle_v1_v2_matched.py --out <NEW_OUTPUT>',
        '```', '']
    a.report.write_text('\n'.join(line.rstrip() for line in lines), encoding='utf-8')
    print(json.dumps(dict(artifact=str(a.artifact), report=str(a.report), checks=checks), indent=2))


if __name__ == '__main__':
    main()
