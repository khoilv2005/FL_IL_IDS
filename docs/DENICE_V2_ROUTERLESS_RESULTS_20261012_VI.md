# DeNICE V2 routerless_v1 — native pilot hai task

## 1. Kết luận

**Technical PASS, quality feasibility NO-GO ở pilot đã khóa.** Routerless giữ
capacity và học class mới tốt, nhưng sau Task 1 tất cả 374 validation samples
Task 0 đều bị dự đoán sang class mới. Bỏ router chưa đủ để có cumulative CIL
predictor tốt với loss/protection hiện tại.

- [Protocol trước pilot](DENICE_V2_ROUTERLESS_PROTOCOL_20261012_VI.md)
- [Native receipt](../artifacts/denice_v2_routerless_smoke_20261012.json)
- [Unit/regression receipt](../artifacts/denice_v2_routerless_test_receipt_20261012.json)

Không thêm distillation, không chỉnh LR/epochs/gates, không dùng TEST.
NO-GO giới hạn cho cấu hình/budget này, không bác bỏ mọi phương pháp routerless.

## 2. Kiến trúc đã triển khai

```text
current Non-IID local BASE
→ shared-reference native optimizer
→ class-wise masked aggregation + canonical BN
→ đăng ký union class từ successful training receipts
→ một model CNN/GRU/FC1/FC2
→ cumulative learned-class mask
→ argmax
```

Không ContextDetector, thresholds, activation sketches, routing dependencies
hoặc routing freeze trong `routerless_v1`. State giữ learned-class provenance
theo task để kiểm tra training; inference không dùng task ID để chọn class mask.
Task boundaries vẫn biết khi training; đây là **task-free inference**, không phải
online training không biết task boundaries. V1 và binary-cosine ablation giữ nguyên.

## 3. Loss và protection

Native NICE `forward_output` đặt non-learner logits bằng zero; zero vẫn tham gia
softmax. Routerless dùng `model.forward` cùng kiểu với inference, rồi CE trên
cumulative registered classes + current cluster BASE class union. Các class chưa
học không có probability mass/gradient. Old registered logits vẫn cạnh tranh với
new logits; mature parameter gradients/optimizer drift vẫn bị chặn bằng P1 masks.

Đây là thay đổi objective opt-in cần thiết cho variant, **không phải** ablation
chỉ xóa router rồi giữ nguyên mọi loss. Không cập nhật loss mặc định V1/router cũ.
Unit tests kiểm tra extreme future logits không đổi loss, old logits tham gia CE,
one-class minibatch vẫn có nhiều cluster classes trong denominator, và zero
gradient ở class không được phép.

## 4. Data và budget

Fresh model seed42, cluster T0 thực tế 9/30, một native round/task, Adam FP32
LR0,001, batch16, một epoch; cap32 current BASE/class/client, cap64 VALIDATION.
Giữ membership qua T1 để kiểm tra transition, chưa có full reclustering runner.

| Task | Client9 BASE/steps | Client30 BASE/steps | Class union đã đăng ký |
|---|---|---|---|
| 0 | 160 / 10 | 128 / 8 | 1,2,3,4,5 |
| 1 | 0 / 0 | 128 / 8 | 1,2,3,4,5,6,7,8,10 |

Source roles đã content-disjoint/locked trước backbone training. Offline data
preparation tách current files trước model initialization. Online train/BN chỉ
đọc current BASE. Historical VALIDATION chỉ được đọc ở evaluator sau training
để đo forgetting; không raw BASE/CAL replay, không fit trên validation.

Panel này đã được dùng làm development trong pilot trước; không untouched test.
Không sửa final full TEST protocol (mỗi cluster chạy toàn cumulative TEST).

## 5. Capacity và correctness

- Routing dependency parameters: **0**; activation memory rows: **0**.
- T0: protected backbone 0%, current trainable **76.993 / 512.654** parameters.
- T1 sau consolidation/allocation: maturity protection khóa **14,91% backbone**;
  current trainable **59.483 / 512.654**, không thêm closure freeze.
- Optimizer thật, loss finite, cumulative class mask hợp lệ.
- Mature weights và protected BN channels giữ byte exact.
- Class cũ không mất registration; class absent trong BASE không tự đăng ký.
- Broadcast state/logits nhất quán; file checkpoint restore model/class registry
  và predictions/logits giống hệt. Chưa phải full optimizer/scheduler resume.

Protection trên parameters không chứng minh function/logits của class cũ giữ
nguyên: upstream mutable parameters vẫn có thể đổi, và new logits có thể thắng
old logits dù old FC2 rows không bị ghi đè.

## 6. Development classification

| Checkpoint / panel | Rows | Accuracy |
|---|---:|---:|
| Sau T0, VALIDATION T0 | 374 | **55,0802%** |
| Sau T1, cùng VALIDATION T0 | 374 | **0%** |
| Sau T1, VALIDATION T1 | 213 | **94,3662%** |
| Sau T1, cumulative T0+T1 | 587 | **34,2419%** |

- T0 Macro-F1, fixed scope0–5: **40,2869%**.
- T1 cumulative Macro-F1, fixed scope0–11: **21,4184%**.
- Old present-class macro recall: **0%**.
- New present-class macro recall: **94,5126%**.
- New fixed-scope macro recall6–11: **63,0084%**; absent validation classes9/11
  có zero recall theo metric convention, không phải observed recall failure.
- Cumulative validation thiếu classes **0,9,11**. Không loại chúng khỏi fixed-scope
  Macro-F1; báo missing support rõ ràng, không tự đăng ký class để tăng coverage.

Forgetting trên cùng old panel = **55,0802 điểm phần trăm**. Đây là development
class accuracy, khác balanced task accuracy41,58% của shallow routing pilot.
Không so trực tiếp với headline full TEST V1 hoặc các CME/APPLIANCE diagnostics.

## 7. Logit competition

- Old samples bị chọn class mới: **374/374**.
- New samples bị chọn class cũ: **0/213**.
- Median `max(new_logits) − max(old_logits)`:
  - Old panel: **+0,98067**.
  - New panel: **+0,82729**.

Những kết quả này xác nhận prediction lệch mạnh về new classes. Chưa tách riêng
được đóng góp của representation drift và logit calibration; không suy ra rằng
old function hoàn toàn mất hoặc old head weights bị hỏng. Không dùng oracle
task/class trong prediction hoặc chọn hyperparameters theo panel này.

## 8. Gates và tests

| Gate khóa trước | Kết quả |
|---|---|
| Technical integrity | PASS |
| Old accuracy drop ≤10pp | **FAIL: 55,08pp** |
| Present-new macro recall ≥25% | PASS:94,51% |
| Cumulative Macro-F1 > uniform registered-class expected-confusion baseline | PASS:21,42% >7,64% |

**146/146 regression tests PASS**, gồm23 tests routerless mới; không failure,
error hoặc skipped. Các structural tests không thay quality gate. Theo protocol,
decision là **NO_GO_BOUNDED_PILOT**, chưa có GO full training.

## 9. Hướng tiếp theo

Giữ routerless làm variant nghiên cứu, không mở coordinate-level router protection.
Bottleneck cần xem xét tiếp là cumulative old/new decision và preservation qua
task transition. Current-BASE distillation từ previous checkpoint có thể là một
pilot objective riêng sau khi khóa protocol; không coi nó tự bảo đảm old knowledge
hoặc thêm vào cùng experiment để cứu con số hiện tại.

P0/P1, graph ξ0,8, V1, binary-cosine ablation, final TEST protocol giữ nguyên.
Không cần full retrain lúc này; một cluster/two-task cap-budget chưa đủ để chốt
routerless làm main architecture benchmark.
