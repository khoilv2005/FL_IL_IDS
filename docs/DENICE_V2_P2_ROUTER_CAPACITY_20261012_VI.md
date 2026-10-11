# DeNICE V2 — P2 Shared Router: capacity NO-GO

## Quyết định

**Chưa PASS P2. NO-GO cho thiết kế bảo vệ router theo toàn tensor hiện tại.**
Preflight chạy sau một round native trên current BASE thật. Closure của các
features Conv1/Conv2/Conv3/GRU khóa toàn bộ backbone CNN/GRU/BN. Đã dừng trước
native smoke Task 0 → Task 1; không commit production router, không chạy TEST.

P2 primitives và shadow contracts đã triển khai. **103/103 regression/unit tests
PASS**: 89 tests P0/P1 cũ + 14 tests P2. Unit PASS không thay thế capacity gate.

- [Preflight BASE thật và capacity receipt](../artifacts/denice_v2_p2_capacity_preflight_20261012.json).
- [Test receipt và source hashes](../artifacts/denice_v2_p2_test_receipt_20261012.json).

## 1. Phần đã triển khai

`fed_learning/strategies/decentralized/denice_v2/router.py`:

- `dependency_closure`: closure bảo thủ theo **toàn named tensor**, đúng với
  granularity của P0/P1. Conv features bao gồm upstream Conv + BN affine và
  running buffers; GRU features bao gồm cả hai recurrent layers, weights/biases.
- `probe_router_capacity`: đo closure và permissions còn trainable trước khi
  fit memory hoặc thực hiện native task transition.
- Routing moments từ current BASE trên cùng post-aggregation model. Header
  bind task/round/member, plan/model/mask/source authority.
- Threshold chung lần đầu = global mean + unbiased std của selected activation
  entries, tổng hợp bằng counts/sums/squared sums. Không trộn local thresholds.
- Binary sketches per sample, giới hạn per class/member, cùng encoder,
  thresholds và stable mask; registration từ actual training receipts.
- Historical binary banks/thresholds/mask giữ nguyên. Refresh current bank chỉ
  đọc current BASE; không refresh lịch sử bằng raw inputs.
- `RouterProposal`: shadow state không mutate reference; capacity NO-GO hoặc
  proposal bị sửa thì không được commit.
- `ClusterRouter.predict(x)`: lấy native activations, binary cosine chọn task,
  classifier chọn class trong registration của task đó. Không nhận `y_true`
  hoặc true task ID. Import `ContextDetector` chỉ để reuse thuật toán; không
  tạo một central aggregation server.

P2 routing fingerprint bổ sung binding cho GRU connection masks,
`structural_protection` và input shape. Contract P0/P1 cũ giữ semantics; chỉ
state có `whole_native_closure_v1` dùng binding mới.

Upload bytes của moment/sketch packets đo bằng typed JSON UTF-8 per message.
Đây chưa là tổng communication: chưa tính model/broadcast/native traffic;
full ledger vẫn thuộc runner integration P3.

## 2. Preflight dữ liệu thật

Fresh Task 0, seed 42, cùng cluster baseline **9/30**, ξ=0,8. BASE roles và
dataset checksums đã khóa từ `results (11).zip` và `archive (1).zip`; không
dùng weights T5. Cap 32 BASE rows/class/member, tổng 288 mẫu.

Đã chạy lại native P1, optimizer Adam thật, rồi probe trên model sau aggregation
và canonical BN. P1 checks tiếp tục PASS. Không đọc raw BASE/CAL Task 0 ở Task 1
vì native Task 1 chưa bắt đầu. Không đọc TEST hoặc chọn cấu hình bằng TEST.

Feature selection baseline: `rank > 0` ở cả bốn native context layers, khóa tại
lần tạo memory. Chọn 82/548 features: Conv1=10, Conv2=19, Conv3=38, GRU=15.

Contract hiện freeze **toàn tensor khi tensor đó thuộc routing dependencies**.
Vì vậy, chọn một số Conv3 features vẫn khóa Conv3 và các Conv/BN upstream;
chọn GRU features khóa cả hai GRU layers. Chưa có compiler dependency masks
theo từng tọa độ trong P2 này.

| Metric | Kết quả |
|---|---:|
| Tổng parameters model | 512.654 |
| Backbone Conv/GRU/BN parameters | 215.916 |
| Backbone bị khóa bởi closure | **215.916 — 100%** |
| Backbone parameters ngoài closure | **0** |
| Hiện trainable trước closure | 72.493 |
| Hiện trainable sau closure | 44.035 |
| Hiện trainable sau closure / toàn model | **8,59%** |
| Toàn bộ FC1/FC2 parameters | 296.738 |
| Production router commit | 0 |
| Native two-task smoke | Chưa chạy |

Các numbers trainable dùng **canonical learner permissions hiện tại**, không
coi rank0 reserve là đang trainable. FC1/FC2 vẫn có current learners và reserve
cho allocation tương lai. **Không kết luận model không học được gì**; kết luận
là backbone hoàn toàn bị khóa dưới cơ chế bảo vệ này.

Capacity policy engineering được định nghĩa trước preflight:
`whole_dependency_max_backbone_lock_0.90_v1` — reject nếu closure khóa ≥90%
backbone hoặc không còn parameter hiện trainable. Ngưỡng là formalization của
gate "closure gần toàn backbone", không là hyperparameter chọn bằng test.
Kết quả 100% → `BACKBONE_CLOSURE_TOO_LARGE`.

## 3. Phạm vi chứng minh của tests

14 tests P2 kiểm tra thresholds chung với phép mean/std trực tiếp, compatible
sketches/registrations, stale/duplicate/empty/degenerate packets, dependency
closure Conv/BN và hai GRU layers, protection metadata, proposal reject,
broadcast và snapshot/restore.

Controlled shadow fixtures có hai task banks. Handoff giữ memory T0, builder
T1 chỉ nhận current BASE T1; test cấm gọi threshold recalibration. Router nhận
input x để sinh task và class, các bản broadcast/restore có outputs giống nhau.

**Các fixtures này dùng synthetic delta receipts, không phải native training
hai task.** State dùng cho shadow unit tests không được production commit:
`RouterProposal.commit()` từ chối capacity NO-GO. Không dùng chúng làm bằng
chứng empirical historical activation accuracy hoặc survival PASS.

## 4. Tái lập

```powershell
.\.venv-audit\Scripts\python.exe -m unittest discover -s tests -p 'test_denice_v2_p*.py' -v

.\.venv-audit\Scripts\python.exe tools/probe_denice_v2_router_capacity.py `
  --cache audit_denice/denice_v2_native_p1 `
  --out audit_denice/denice_v2_p2/new_capacity_receipt.json
```

Cache được chuẩn bị bởi `tools/run_denice_v2_native_smoke.py`. Output dùng path
mới. Exit code 0 của probe có nghĩa hoàn tất diagnostic; phải đọc `decision`
và `p2_pass`, không suy thành PASS training.

## 5. Bước tiếp theo

Giữ ξ=0,8, clustering và protection. Chưa P3/full training.

Cần một quyết định thiết kế có version riêng trước native two-task smoke:

1. Thu hẹp routing subspace để closure nhỏ hơn; vẫn phải đo tính phân biệt task.
2. Hoặc mở rộng dependency contract sang coordinate masks chính xác. Đây là
   thay đổi contract P0/P1 cần tests và proof mới. Dense GRU recurrence vẫn có
   thể tạo closure lớn dù giảm granularity của CNN.

Không tự thêm frozen auxiliary encoder, replay, mapper hoặc bỏ protection để
đánh dấu P2 PASS. Preflight này chứng minh **giới hạn của whole-tensor closure
đang triển khai**, chưa chứng minh mọi shared binary router đều thất bại.
