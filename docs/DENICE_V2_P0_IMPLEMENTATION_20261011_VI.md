# DeNICE V2 — P0 Cluster State

## Kết quả

P0 đã triển khai trong namespace riêng `denice_v2`. **44/44 unit tests PASS**
trên CPU ngày 11/10/2026. Không thay runner, model hoặc cấu hình mặc định V1.
P1 aggregation, P2 router và notebook full training chưa triển khai trong mốc này.

Receipt: [denice_v2_p0_test_receipt_20261011.json](../artifacts/denice_v2_p0_test_receipt_20261011.json).
Receipt ghi môi trường, command, số tests và SHA-256 của code với newline LF.

## Module đã có

| Module | Chức năng |
|---|---|
| `partition.py` | Complete-linkage deterministic, ξ mặc định 0,8, K tự động; membership khóa theo task; log rejected merge và hash nguồn similarity/reference |
| `knowledge.py` | Ledger local learning theo thời gian, Class Knowledge Matrix; successful optimizer steps, current BASE và class counts; không suy ownership từ broadcast |
| `election.py` | Số unique class thực sự học local; tie-break ID nhỏ nhất; giữ incumbent trong task; failover khi unavailable |
| `contracts.py` | Schema model/router/masks/ranks/BN/protection/versions; routing dependency hash; codec lossless cho tensor, NumPy, tuple và integer keys |
| `state.py` | Snapshot/restore/broadcast độc lập; khởi tạo chung Task 0; split/merge/handoff; registry active/dormant; atomic checkpoint có integrity hash |

Source nằm tại `fed_learning/strategies/decentralized/denice_v2/`.
Tests nằm tại `tests/test_denice_v2_p0.py`.

## Task 0: quyết định bootstrap

Đầu Task 0 chưa có activation memory đã học. P0 định nghĩa fallback bằng Jaccard
của tập class có số row >0 trong **current BASE metadata**:

\[
s_{ij}^{bootstrap}=\frac{|B_i^0\cap B_j^0|}{|B_i^0\cup B_j^0|}.
\]

Không dùng future tasks, labels TEST hoặc prototypes giả. Count này chỉ cho biết
dữ liệu sẵn có, **không** đánh dấu client đã học class. Active client không có
usable BASE support bị từ chối. Task khác không được gọi Task0 bootstrap.

Đây là fallback engineering mới cho V2, chưa phải similarity đã benchmark hay
công thức context của V1. P0 nhận ma trận có provenance cho các task sau;
việc tạo current context capsules trên shared encoder sẽ được nối ở P1/P2.

`initialize_clusters()` yêu cầu chung initialization Task0/round=-1, ranks chưa
học, không có router history/ownership, và reference hash khớp template.
Leader lúc chưa có local learning được chọn bằng ID nhỏ nhất.

## Partition và handoff

- Score merge bằng minimum pair similarity; chọn score cao nhất đạt ξ.
  Tie-break bằng sorted client IDs; không ép K=5 hoặc minimum group size.
- `merge_gate()` lấy snapshot riêng của parent states, không đọc live state đang
  bị mutate giữa lúc quyết định partition.
- Split sao chép toàn bộ parent state, giữ lịch sử router, protection và metadata.
- P0 chỉ cho merge khi model, schema, versions, routing space, algorithm state,
  BN, protection và router history khớp chính xác.
- Không average encoder, không concatenate historical sketches, không reset model.
  Merge khác state bị từ chối có reason; parents giữ nguyên hash.
- Trùng parent UID nhưng state khác nhau cũng bị từ chối.
- Parent mapping khi chuyển task phải explicit; client không được lấy history của
  state ngoài mapping. Rejoining/dormant authority sẽ được nối với runner sau.
- Registry validate toàn bộ active partition trước khi cập nhật. Parent states
  không còn active vẫn được giữ trong dormant registry khi dùng UID mới.
- Cluster không đổi members/UID trong task; leader có thể failover mà giữ
  model/router/masks nguyên vẹn.

**Giới hạn chủ ý:** gate exact-state khá bảo thủ. Hai cluster đã học khác nhau có
thể bị từ chối merge dù similarity cao; K thực tế có thể lớn. P0 không chứng minh
rằng mọi merge bị từ chối đều bất khả thi về chức năng. Không tự nới gate để giảm K.

## State và checkpoint

Full state hash bao gồm model, algorithm metadata, BN, router, protection,
ownership, lineage và RNG state do caller cung cấp. Routing hash chỉ bind các
dependency đã khai báo, context-layer order, masks, thresholds và feature version;
FC2 không được đưa vào nếu không ảnh hưởng routing activations.

Contract kiểm tra shape/dtype, binary masks, mature-neuron freezing/protection,
GRU gate dimensions, BN consistency, router width và episode time. Router schema
không nhận raw reference inputs/CAL. Ledger chỉ lưu counts/receipts, không lưu
raw samples. Broadcast không thêm local-learning events.

Codec giữ nguyên tensor bytes/dtype, kể cả scalar integer, BF16 và tensor rỗng;
restore lưu tensor trên CPU. Broadcast clones không chia sẻ buffer với canonical
state hoặc với nhau. Device placement và native training restore thuộc integration.

`P0Checkpoint` lưu registry, knowledge ledger và partition book. Load kiểm tra
checksum, schema, class mapping, clocks, active membership và locked partition.
Write dùng file tạm rồi atomic replace. Không dùng pickle và không truncate lịch sử.

Đây là primitive checkpoint P0 dạng JSON lossless, **chưa phải** cơ chế delta/ZIP
quota Kaggle hoặc resume optimizer trong training runner. State có thể lưu thêm
allocation/adapter/optimizer policy trong `algorithm_state`; P1/P2 phải nối đầy đủ
native attributes vào canonical schema trước khi chạy training.

## Các tests đã chạy

| Nhóm | Kiểm tra |
|---|---|
| Partition | Singleton, K=1, mọi client singleton, complete-linkage không gộp transitive chain, threshold inclusive, deterministic replay/shuffled IDs, ma trận invalid, task0 bootstrap, locked membership |
| Ownership/election | Local counts, skipped/incomplete steps, broadcast/replay bị từ chối, future evidence không được dùng, event idempotency/collision, restore, leader score/tie/incumbent/failover |
| Contract | Model/mask/version invalid, mature protection, BN mismatch, router width/history/raw fields, dependency hash, FC2 không làm đổi routing hash |
| State | Broadcast không alias, split, compatible merge, rejected merge không mutate parent, failover, registry atomic reject và dormant retention |
| Restore/integration | Codec lossless, checkpoint checksum/replay, checkpoint sau split, partition/ledger mismatch, controlled transition hai task, shape contract với `DeNICEModel` native |

Lệnh tái lập từ repo root, dùng Python có NumPy và PyTorch:

```powershell
python -m unittest discover -s tests -p test_denice_v2_p0.py -v
```

Môi trường local: Python 3.14.7, NumPy 2.5.2, PyTorch 2.14.0+cpu.
Lần cuối: 44 tests, 0 failures, 0 errors, 0 skipped.

Controlled transition là **structural test bằng fixture**, không phải smoke native
hai task có local training. P0 hash/shape validation không chứng minh activation
drift bằng 0 hoặc dependency closure đã đầy đủ. Hai điều đó là gate P2.

## Mốc tiếp theo

P0 structural gate PASS. Bước tiếp theo trong kế hoạch là P1 common-reference
class-wise masked aggregation. Trước full run vẫn cần P1, P2 shared router,
P3 evaluator/checkpoint integration và native smoke. Không dùng kết quả 44 tests
để tuyên bố V2 đã training thành công hoặc accuracy được cải thiện.
