# DeNICE V2 — CNN coordinate router pilot

## 1. Kết luận

**Conv2 và Conv3: technical PASS, routing/quality NO-GO trong bounded pilot.**
Coordinate protection giải quyết capacity và giữ selected features ổn định.
Binary-cosine trên những features đó vẫn gần như luôn chọn Task0; không khai
thác được headroom oracle. Chưa chuyển P3 hoặc full training.

- [Protocol khóa trước](DENICE_V2_CNN_COORDINATE_PROTOCOL_20261012_VI.md)
- [Conv2 receipt](../artifacts/denice_v2_coordinate_conv2_smoke_20261012.json)
- [Conv3 receipt](../artifacts/denice_v2_coordinate_conv3_smoke_20261012.json)
- [179/179 tests receipt](../artifacts/denice_v2_coordinate_test_receipt_20261012.json)

Theo decision tree đã chốt, dừng mở rộng family này và chuyển trọng tâm về
routerless calibration/preservation. Không suy rộng rằng mọi learned task router
hoặc mọi CNN representation đều không thể phân biệt task; chỉ scorer/policy/budget
đã thử có kết quả này.

## 2. Implementation

- Policy `cnn_coordinate_router_v1`, feature target Conv2 hoặc Conv3 version-bound.
- Chọn toàn bộ allocated target features tại T0; giữ mask qua các task.
- Khóa selected kernel rows, bias và toàn bộ connection-mask bits của row đó,
  kể cả inactive edges, để không có reactivation làm đổi historical function.
- Active incoming mask edges quyết định upstream channel closure, recursively
  tới Conv1. Không dựa vào architecture shape để bỏ upstream thật.
- BN affine/running moments bảo vệ theo channel; remaining channels được train
  và recalibrate. Canonical counters giữ nguyên, codec hỗ trợ scalar bool masks.
- P1 permissions áp dụng coordinate masks, không freeze cả tensor chỉ vì tên
  tensor có trong dependency list. Maturity/existing protections vẫn union.
- Fingerprint bind đúng parameter/mask coordinates, thresholds, input shape,
  feature layer và version. Unrelated coordinates/GRU không làm đổi CNN hash.
- Committed feature mask không được mở rộng; missing upstream/mask protection
  bị reject. Root disconnected không tạo upstream dependency giả.
- GRU không góp routing features; vẫn là một nhánh classifier được train theo
  maturity masks. Không tạo donor ensemble hoặc thêm neural router trong pilot.

## 3. So sánh công bằng trong pilot

Fresh seed42, cluster9/30; current BASE cap32/class/client, VALIDATION cap64.
Mỗi task một native round, một epoch, Adam FP32 LR0,001, batch16. **Cumulative CE
và full forward giống routerless**, không dùng native LetLearner zero-logit loss.
P0/P1 và ξ0,8 giữ nguyên; membership qua T1 là controlled transition fixture,
chưa phải full reclustering scheduler.

| Task | Client9 BASE/steps | Client30 BASE/steps |
|---|---|---|
| 0 | 160 /10 | 128 /8 |
| 1 | 0 /0 | 128 /8 |

T0 class accuracy của cả hai fresh runs là **55,0802%**, khớp routerless baseline
trên cùng development rows. Khi gating ở T1, mọi inference variant đều dùng
**chính checkpoint của run đó**. Không gắn shallow task metrics hoặc routerless
oracle80,75% vào checkpoint coordinate.

Router memory/thresholds chỉ đọc current BASE; không historical BASE/CAL replay.
Read-only evaluator dùng historical VALIDATION để chấm classification và kiểm
tra selected feature drift. VALIDATION không tham gia fit/threshold selection.
Đây là development đã quan sát, không untouched/final TEST. Không tune LR,
feature subset, thresholds, scorer hoặc thêm distillation sau khi thấy outcome.

## 4. Capacity: PASS

Backbone có215.916 neural parameters; whole model512.654. BN running buffers và
counters được bind/protect nhưng không tính vào neural parameter denominator.

| Target | Selected features | Upstream required | Closure T0 | Locked backbone T0 |
|---|---:|---|---:|---:|
| Conv2 | 19 | Conv1:64 channels | 4.089 | **1,8938%** |
| Conv3 | 38 | Conv2:128 + Conv1:64 | 40.050 | **18,5489%** |

Dense fan-in làm Conv2 target cần toàn Conv1, và Conv3 target cần toàn Conv2/Conv1.
Đây là closure đúng; vẫn không kéo theo whole Conv3 hoặc GRU.

| Target | Current trainable T0 trước→sau closure | Actual protected backbone T1 | Current trainable T1 |
|---|---:|---:|---:|
| Conv2 | 76.993→73.120 | 32.520 =15,0614% | 59.347 |
| Conv3 | 76.993→58.196 | 53.775 =24,9055% | 56.234 |

T1 protection includes maturity T0 và routing closure; rank0 reserve không được
gọi là currently trainable. Các unprotected/current rows và GRU có optimizer
updates thật. Existing protections không bị xóa để tạo capacity PASS.

## 5. Stability/restore: PASS

Cho cả hai targets:

- Selected activation max absolute drift trên374 historical VALIDATION rows: **0**.
- Selected activation drift trên current T1 BASE witness: **0**.
- Historical binary memory, validation bits, thresholds và routing fingerprint
  giữ byte exact; không recalibrate old thresholds bằng raw historical inputs.
- Protected mature/coordinate weights và BN channels không đổi.
- Empty client zero-step receipt; không đóng góp BN/memory giả.
- Model/router checkpoint restore cho cùng class prediction, task, scores và bits.
- T0 và T1 checkpoints đều persist; đây là model/router state, không full
  optimizer/scheduler resume hoặc production ZIP campaign.

Vì representation đã ổn định trên evidence kiểm tra, routing FAIL không thể giải
thích bằng selected activation drift trong pilot này.

## 6. Task routing: FAIL

| Target | T0 correct/374 | T1 correct/213 | T0 recall | T1 recall | Balanced task accuracy |
|---|---:|---:|---:|---:|---:|
| Conv2 | 348 | 5 | 93,0481% | **2,3474%** | **47,6978%** |
| Conv3 | 373 | 2 | 99,7326% | **0,9390%** | **50,3358%** |

Conv2 confusion: `[[348,26],[208,5]]`.
Conv3 confusion: `[[373,1],[211,2]]`.
Pooled task accuracy60,14%/63,88% che khuất collapse về T0 vì old panel lớn hơn.
Cả hai FAIL minimum feasibility60% balanced/25% each recall, và FAIL stricter
P3 discrimination75% balanced/60% each recall.

## 7. Class prediction trên cùng checkpoint

Cumulative panel587 rows, fixed Macro-F1 scope0–11, zero_division0. Missing
validation classes0/9/11 báo trong receipts, không loại khỏi fixed-scope metrics.

| Target | Inference | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Conv2 | Cumulative, no task gate | 34,0716% | **22,3133%** |
| Conv2 | Actual predicted-task gate | **40,3748%** | **16,4931%** |
| Conv2 | Oracle true-task gate, diagnostic | 77,3424% | 51,3442% |
| Conv3 | Cumulative, no task gate | 34,4123% | **21,5149%** |
| Conv3 | Actual predicted-task gate | **45,1448%** | **18,0112%** |
| Conv3 | Oracle true-task gate, diagnostic | 79,2164% | 52,7860% |

Actual gating tăng sample-weighted accuracy nhưng **giảm Macro-F1** so own
cumulative predictor; gần như mất toàn bộ new-class recall. Không coi45,14% là
thành công CIL hoặc dùng77–79% oracle làm achieved metric.

| Target | Actual old accuracy | Actual new accuracy | Rescue /break vs own cumulative |
|---|---:|---:|---:|
| Conv2 | 62,0321% | **2,3474%** | 232 /195 |
| Conv3 | 70,3209% | **0,9390%** | 263 /200 |

Old accuracy tăng so T055,08%, nhưng router đổi nhiều new correct predictions
thành sai: net accuracy gain37/587 và63/587 không phản ánh retention toàn class.
Oracle-minus-actual gap là36,9676pp và34,0716pp trên đúng checkpoint tương ứng.

## 8. Class2 và giới hạn classifier

Trên20 class2 VALIDATION rows, T0 recall90%; sau T1 cả hai runs:

- Actual gated recall: **0%**.
- Oracle old-only recall: **0%**.

Vì vậy task gating không sửa được lỗi class2. Routing stability không chứng minh
toàn classifier function được bảo toàn: Conv3/GRU/FC1 inputs và old/new scoring
ngoài selected routing subspace vẫn có thể thay đổi.

## 9. Tests và bước tiếp theo

**179/179 unit/regression tests PASS**, không failures/errors/skipped, gồm26
coordinate tests mới và7 inference/logit audit tests. Có native optimizer, partial
BN survival, actual sparse-edge upstream proof, scalar checkpoint restore,
unrelated-coordinate mutation và missing-closure rejection. Structural PASS
không thay empirical routing/class-quality gates.

Dừng family static CNN binary-cosine này theo decision tree. Giữ implementation
và receipts làm ablation/diagnostic. Chưa P3/full run, chưa thêm alignment hoặc
distillation vào same panel. Nhánh tiếp theo là một protocol riêng cho routerless
logit calibration/preservation, gồm class-wise retention; không dựa vào oracle
để tuyên bố router deployed đã khai thác collective knowledge.
