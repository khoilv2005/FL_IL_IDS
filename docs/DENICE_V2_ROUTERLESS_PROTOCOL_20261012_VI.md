# DeNICE V2 routerless_v1 — pilot protocol trước khi chạy

## Kiến trúc và phạm vi

Opt-in variant: cluster model dự đoán trực tiếp từ CNN/GRU/FC1/FC2, argmax trên
cumulative union classes có receipt local training hợp lệ. Không task router,
không activation memory, không routing dependencies/freeze. NICE maturity
protection, P0/P1 và ξ=0,8 giữ nguyên. Không thêm distillation/regularization.

Đây là **class-incremental training có task boundaries, task-free inference**;
không tuyên bố online training không biết task boundaries. Binary-cosine vẫn
giữ trong repo để ablation. Không migrate run router cũ bằng cách xóa protection.

## Budget và loss

- Fresh legacy model seed 42, members 9/30 của cluster T0 thực tế.
- Membership giữ qua T1 cho fixture bounded, không phải full reclustering runner.
- T0 → T1: một native round/task, Adam FP32 LR 0,001, batch16, một epoch;
  current BASE cap32/class/client. Client9 T1 không có BASE: zero-step receipt.
- Class union dùng trong CE = classes đã commit trước đó + current classes có
  authorized BASE trong cluster, không chỉ classes trong minibatch/client.
- `model.forward` giống inference; không dùng LetLearner zero-logit path cho
  variant này. Cumulative CE slice cả logits và target mapping. Unlearned/future
  classes không có probability mass hoặc gradient; mature weights vẫn protected.
- Registration sau aggregation chỉ lấy successful optimizer receipts. Class
  nhận qua broadcast không biến thành ownership local của receiver.

## Development và forgetting

Cap64 VALIDATION/class/client giống shallow pilot. Roles content-disjoint với
BASE, đã khóa trước training. Offline preparation tách source NPZ thành file
theo task trước model initialization; online training chỉ đọc current BASE.

Evaluator được đọc **historical VALIDATION** để đo forgetting/cumulative CIL,
chỉ inference, không đưa vào optimizer, BN calibration, registration, memory,
threshold/hyperparameter selection. Không đọc historical raw BASE/CAL.
Đây là development panel đã dùng trong pilot trước, không untouched test.

- T0: evaluate VALIDATION T0 trước transition.
- T1: evaluate cùng VALIDATION T0 và current T1; argmax trên cùng cumulative
  class mask, không dùng nhãn thật/task ID để chọn model hoặc mask.
- Báo accuracy; Macro-F1 với fixed label scope0–5/T0,0–11/T1, zero_division0;
  class recall và confusion; class thiếu validation ghi rõ.
- Old/new macro recall báo riêng present-class mean và fixed-scope mean.
- Forgetting = accuracy_T0_after_T0 − accuracy_T0_after_T1, pp trên cùng rows.
  Báo thêm old per-class recall drop; negative delta biểu thị improvement.
- Logit diagnostic: old/new group thắng bao nhiêu, margin, native finite
  logits, excluded classes probability=0. True labels chỉ dùng thống kê sau predict.

## Gates đã đặt trước kết quả

Technical PASS cần optimizer thật; mask/loss đúng; không routing memory/deps;
không overwrite mature weights/buffers; registration hợp lệ; broadcast và
model/class registry checkpoint restore prediction giống hệt.

Quality feasibility cần old accuracy loss ≤10pp, present-new-class macro recall
≥25%, cumulative Macro-F1 vượt uniform-on-registered-classes expected-confusion
baseline trên cùng panel/scope. Baseline dùng expected confusion, không tune seed.
Gate này chỉ quyết định có nên mở rộng pilot; **không cấp GO full benchmark**.
Nếu FAIL: báo thẳng, không tune LR/epochs/distillation để cứu cùng panel.

## Chạy local

```powershell
.\.venv-audit\Scripts\python.exe tools/run_denice_v2_routerless_smoke.py `
  --cache audit_denice/denice_v2_native_p1 `
  --out artifacts/denice_v2_routerless_smoke_20261012.json
```
