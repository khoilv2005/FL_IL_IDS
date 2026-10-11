# DeNICE V2 — P2-Shallow: capacity PASS, discrimination NO-GO

## 1. Quyết định

**NO-GO cho pilot `conv1_only_v1` theo protocol đã khóa.** Giảm closure giải
quyết được capacity, nhưng 10 binary Conv1 features chưa phân biệt tốt Task 1
trong fixture này. Không bypass gate, không tune lại threshold, không chạy full.
Đây không phải kết luận shared cluster router hoặc mọi Conv1 router đều thất bại.

- [Protocol khóa trước pilot](DENICE_V2_P2_SHALLOW_PROTOCOL_20261012_VI.md)
- [Native receipt](../artifacts/denice_v2_p2_shallow_smoke_20261012.json)
- [Unit/regression receipt](../artifacts/denice_v2_p2_shallow_test_receipt_20261012.json)

## 2. Implementation

- Feature policy version riêng, chỉ chọn Conv1 units đã allocate; builder thực
  sự dùng policy, không chỉ sửa capacity probe. Default whole-native giữ nguyên.
- Closure Conv1/BN1; union mọi dependency cũ, giữ protection đã commit. Regression
  test giữ GRU witness và tính thêm 30.000 weights khi witness này tồn tại.
- Routing fingerprint bind Conv1 weights, masks, BN buffers, input shape,
  thresholds và feature mask. Deep masks không thuộc dependency không làm đổi
  shallow routing hash. Dependency cũ vẫn được bind khi hiện diện.
- Capacity tính cả protection/maturity có sẵn, không chỉ closure mới.
- Client không có current BASE tạo zero-step receipt, không đóng góp BN/sketch.
- Task-end consolidation tăng neuron age một lần, bảo vệ mature weights, không
  đổi weights/masks/router memory. Handoff dùng P0 state contract.
- Checkpoint model/router được ghi file rồi load lại; không gọi đó là full
  optimizer/scheduler resume. Production runner/P3 chưa hoàn thành.

## 3. Native data và optimizer

Seed 42; fresh legacy model; cluster Task 0 thực tế gồm client 9 và 30. Giữ
membership qua Task 1 để kiểm tra transition, không thay P0 partition policy.
Một round/task, một epoch, Adam FP32 LR 0,001, batch 16, cap 32 BASE/class/client.
Cluster T0 có classes 1–5; T1 có 6, 7, 8, 10. Không mở classes chưa học.

| Task | Client | Current BASE rows | Optimizer steps |
|---|---:|---:|---:|
| 0 | 9 | 160 | 10 |
| 0 | 30 | 128 | 8 |
| 1 | 9 | 0 | 0 |
| 1 | 30 | 128 | 8 |

Source clean roles locked trước backbone training. Source NPZ chứa nhiều task;
bước chuẩn bị dataset tách current-role files trước khởi tạo model. Sau đó online
Task 1 chỉ đọc file Task 1. Không dùng historical raw BASE/CAL/VALIDATION để train,
recalibrate, refresh threshold hoặc re-encode historical memory. Không đọc TEST.

## 4. Capacity

Chọn **10/548 features**, tất cả ở Conv1. Fresh fixture không thêm GRU witness
của smoke P1; contract bảo vệ BN1 counter ban đầu, sau đó bảo vệ toàn Conv1/BN1.

| Thời điểm | Protected backbone | Tỷ lệ | Current trainable / toàn model |
|---|---:|---:|---:|
| Closure sau round Task 0 | 384 / 215.916 | 0,178% | 76.825 / 512.654 = 14,99% |
| Allocate Task 1, gồm maturity T0 | 32.520 / 215.916 | 15,06% | 59.347 / 512.654 = 11,58% |

Conv2, Conv3 và GRU còn lần lượt 2.895, 11.165, 10.980 learner parameters tại T1.
Reserve rank 0 không được gọi là current trainable. BN buffers/counters được
protect và hash nhưng không tính vào số neural parameters ở bảng này.

## 5. Stability và restore

Các check đều PASS:

- Native optimizer thật thay đổi parameters được phép qua hai task.
- Không ghi đè mature/routing parameters; Conv1 current T1 witness giữ byte exact.
- Routing dependency hash, thresholds và memory Task 0 giữ nguyên qua Task 1.
- Broadcast model state/logits nhất quán; empty client không thành contributor.
- Model/router sau file checkpoint restore có class prediction, task, scores,
  binary bits và full state hash giống hệt.

Historical consistency gồm structural dependency protection và frozen old bits.
Không đo lại activation accuracy trên raw T0 tại T1, không tuyên bố có empirical
historical raw re-encoding. Đây là bounded native smoke, không full CL benchmark.

## 6. Discrimination — gate FAIL

Validation cap 64/class/client, tách BASE theo clean roles. Task 0 có 374 rows,
encode lúc T0 còn current rồi chỉ giữ bits và task IDs. T1 có 213 current rows.
Task ID thật chỉ dùng chấm confusion, không đưa vào router inference.

| Task thật | Router chọn T0 | Router chọn T1 | Task recall |
|---|---:|---:|---:|
| T0 | 297 | 77 | 79,41% |
| T1 | 205 | 8 | **3,76%** |

Balanced task accuracy **41,58%**, dưới gate khóa trước **60%**. T1 recall dưới
gate **25%**. Router không chọn một task tuyệt đối, nhưng lệch mạnh về T0:
502/587 predictions chọn T0. Những số này là **task-routing metrics**, không phải
class accuracy, final TEST accuracy hoặc kết quả benchmark DeNICE V2.

## 7. Tests và giới hạn

**123/123 unit/regression tests PASS**, không failure/error/skipped; 20 tests
shallow mới dùng native optimizer thật cho controlled fixtures. Structural PASS
không thay thế native discrimination gate: kết quả pilot vẫn **NO-GO**.

Đã kiểm tra một cluster, hai task, mỗi task một round với BASE cap. Không đủ để
suy rộng rằng mọi shallow router đều kém; cũng không đủ để khóa router này cho
full benchmark. Wire bytes trong router receipt chỉ là moments/sketch uploads,
không phải tổng communication của V2.

## 8. Bước kế tiếp

Giữ P0/P1 và full TEST protocol. Chưa P3/full training. Theo decision tree đã
chốt, nhánh tiếp theo có thể là coordinate-level dependency protection để dùng
features sâu hơn mà không khóa toàn tensor. Cần contract xử lý upstream, BN và
GRU recurrent dependencies đầy đủ trước pilot mới. Không âm thầm bỏ protection
hoặc thay binary-cosine để cứu kết quả này.

## 9. Chạy lại pilot

```powershell
.\.venv-audit\Scripts\python.exe tools/run_denice_v2_shallow_smoke.py `
  --cache audit_denice/denice_v2_native_p1 `
  --out artifacts/denice_v2_p2_shallow_repeat.json
```

Cache phải có `roles/` và `dataset/` đã xác minh bằng native P1 preparation.
Output receipt phải là path mới. Không cần Kaggle/full training cho pilot này.
