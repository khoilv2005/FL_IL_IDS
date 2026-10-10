# DeNICE coverage: nguyên nhân từ allocation, participation và catch-up

## Kết luận

Đã truy ngược toàn bộ **98 receiver × 6 tasks × 34 classes** của `results (11)`.
Audit dùng đúng checkpoint DeNICE legacy, ξ = 0,8, seed 42, Task 5 / Round 19,
training commit `4a45a7176ac4f8cb8671dba83f3e52071e44a3ca`.

**Coverage thiếu không phải do checkpoint làm mất một class đã học local.**
Registration cuối cùng được tái dựng từ lịch sử tham gia và bootstrap, khớp
**98/98 receiver**. Classifier ranks trong checkpoint cũng khớp registration:
mọi class reachable có row trưởng thành; mọi class unreachable có `fc2 rank=0`.

Có hai nguyên nhân riêng:

| Nguyên nhân | Receiver–task / receiver–class pairs | Test rows | % toàn test |
|---|---:|---:|---:|
| Không học task, không có profile của task | 74 task pairs / 444 class pairs | 2.902.000 | 21,4871% |
| Có profile nhưng native class mask chỉ chứa class local/source đã học | 756 class pairs | 2.807.415 | 20,7868% |
| Tổng unreachable | 1.200 class pairs | 5.709.415 | 42,2739% |

Những rows này chiếm **67,2576% lỗi MulticlassSelf**, như full audit trước.
Đây là nguyên nhân coverage trong **protocol hiện tại**, không phải bằng chứng
mọi class ngoài coverage đều có thể được mở an toàn.

## 1. Participation: tăng số client không có nghĩa giữ nguyên client cũ

Metadata dataset chọn tập client theo từng task, không phải các tập lồng nhau.
Training chỉ tạo/update task profile khi client có BASE rows của task đó.

| Task | Dataset allocated | Thực sự train BASE | Client allocated ở task trước nhưng không allocated ở task này |
|---|---:|---:|---:|
| T0 | 50 | 50 | — |
| T1 | 60 | 60 | 26 |
| T2 | 70 | 70 | 14 |
| T3 | 80 | 80 | 14 |
| T4 | 90 | 89 | 7 |
| T5 | 100 | 98 | 0 |

T5 có 98 receiver thực tế. Không đổi tập receiver hoặc phân shard test để tránh lỗi.

Trong 74 missing profile pairs của các receiver này:

- **73 pairs / 2.897.559 test rows:** client không được dataset phân vào task.
- **1 pair / 4.441 test rows:** client 47 được allocated vào T4 nhưng không có
  usable BASE rows. Role summaries của cả BASE/CAL/FIT/VALIDATION cho T4 đều rỗng.
  Audit metadata không xác định thêm rằng raw shard ban đầu rỗng hay rows đã bị
  loại khi tạo clean roles; không đọc raw shard để suy diễn.
- **0 pair:** đã có local BASE task data nhưng profile bị mất.

### Option dễ gây hiểu nhầm

Dataset config ghi `old_client_data_prob=1.0`. Trong `ClientAllocator` ở source
tại training commit, option này được lưu vào attribute nhưng **không được đọc
trong allocation**. `_allocate()` gọi random choice độc lập cho từng task.

Do đó option này không bảo đảm old clients tiếp tục có dữ liệu ở task mới.
Metadata thực tế cũng xác nhận các tập không lồng nhau.

Không có commit của lần tạo dataset trong artifact. Vì vậy source tại training
commit chỉ corroborate allocation quan sát được; không tuyên bố đó chắc chắn là
đúng phiên bản producer. Không thay generator hay tái tạo dataset trong audit.

Nếu một experiment tương lai yêu cầu participation lồng nhau, cần định nghĩa
đó là **data protocol mới**, tái tạo split và chạy lại các baseline tương ứng.
Không được gọi việc thay allocation là sửa checkpoint hiện tại.

## 2. Bootstrap và rejoining có behavior khác nhau

### Client mới: exact representative clone

Client mới clone model và detector của representative trước khi prepare current
task. Binary activation sketches và class registration đi cùng clone; raw
reference inputs được xóa. Đây là cơ chế tạo **72 inherited task profiles**
trong 514 profiles cuối cùng của 98 receiver.

| Task | Local profile | Profile từ exact clone | Missing profile |
|---|---:|---:|---:|
| T0 | 48 | 50 | 0 |
| T1 | 59 | 14 | 25 |
| T2 | 70 | 5 | 23 |
| T3 | 80 | 2 | 16 |
| T4 | 87 | 1 | 10 |
| T5 | 98 | 0 | 0 |
| Tổng | 442 | 72 | 74 |

### Client quay lại: plastic-weight catch-up

`_catch_up_rejoining_model()` cập nhật những parameters target còn plastic từ
representative, bảo vệ mature parameters. Caller không clone detector, không
thêm task profile hoặc episode-class entry cho task bỏ lỡ.

**Cả 74 missing task pairs** đều xuất hiện trong `missed_tasks` của một lần
catch-up đã ghi nhận. Nguồn catch-up của cả 74 pairs có profile tương ứng.
Trong nhóm terminal receivers có 60 catch-up events, tổng missed tasks = 74.

Điều này giải thích vì sao model có thể đã thay đổi weights nhưng route bank
vẫn thiếu task. Không được suy ra catch-up đã chuyển đủ chức năng class đó.

**Không sửa bằng cách copy detector source một cách mù quáng.** Model receiver
chỉ nhận một phần weights; feature space của nó không được chứng minh giống
source. Binary sketches của source có thể không tương thích. Exact clone ở
client mới không có cùng giả định với partial catch-up ở client cũ.

### Ví dụ receiver 2

| Task | Nguồn profile / classes |
|---|---|
| T0 | Inherited từ clone: 0–5 |
| T1 | Local: 7, 9, 10, 11; không có 6, 8 |
| T2 | Local: 12, 13, 14, 16, 17; không có 15 |
| T3 | Không tham gia, thiếu profile |
| T4 | Không tham gia, thiếu profile |
| T5 | Local: 30–33 |

Khi quay lại T5, receiver 2 catch-up từ client 85. Client 85 có profile T3/T4;
receiver 2 vẫn không có chúng sau catch-up. Terminal registration đúng với
behavior của runner, không có lost-state anomaly.

## 3. Có graph edge chưa đồng nghĩa nhận được class mới

Ở `paper` CANC, `_prepare_client_task()` đăng ký `episode_classes[task]` bằng
**local BASE labels của current task**, không phải toàn bộ global task classes.

Trong native aggregation:

1. Capsule `label_set` được tạo từ current local training labels.
2. `fc2` pair mask chỉ cho phép class thuộc giao của receiver và sender labels.
3. Router class registration / activation memory không được aggregate ở đây.

Vì vậy peer có class c và có positive alpha vẫn không làm c tự xuất hiện trong
class mask của receiver chưa có c.

| Thiếu class mask dù có profile | Pairs | Test rows |
|---|---:|---:|
| Local profile nhưng thiếu class trong local subset | 754 | 2.795.524 |
| Inherited profile nhưng source subset thiếu class | 2 | 11.891 |
| Tổng | 756 | 2.807.415 |

**686 missing class pairs / 2.665.681 test rows** có receiver tham gia ở task
khai sinh và có ít nhất một peer owner với positive alpha ở round 19 của task
đó. Đây là cơ hội cho một cơ chế admission được kiểm chứng **lúc class còn
current**, không phải bằng chứng các rows của peer hoạt động trên receiver.

Tại graph T5/R19, 686 unreachable pairs / 3.102.338 test rows có peer owner đã
đăng ký class. Hai con số 686 ở hai graph không được hiểu là cùng một tập pairs.
Không thể dùng ownership lịch sử này như current CAL authorization.

## 4. Kiểm tra classifier ranks và connection masks

Đọc trực tiếp terminal FP32 state, không forward hoặc update model:

| Regime | Classifier rows | fc2 rank | Mean nonzero connection-mask fraction |
|---|---:|---|---:|
| Reachable | 2.132 | Tất cả ≥2 | 42,7804% |
| Thiếu task profile | 444 | Tất cả 0 | 100% |
| Thiếu class mask | 756 | Tất cả 0 | 100% |

Không có row mature bị bỏ khỏi task/class registry trong fixture này. Các
unreachable rows chưa được DeNICE đăng ký/consolidate như class đã học của
receiver. Rank 0 cũng không chứng minh weights luôn là random initialization:
partial catch-up có thể đã thay chúng. Maturity không thay thế competence.

## 5. Model có chức năng tiềm ẩn ngoài coverage không?

Dùng lại confusion matrices của full audit, không chạy/tune thêm trên test:

| Nhóm ngoài coverage | Samples | TP khi ép true task + global task mask | Conditional recall |
|---|---:|---:|---:|
| Thiếu profile | 2.902.000 | 2.091.272 | 72,0631% |
| Có profile, thiếu class mask | 2.807.415 | 1.456.009 | 51,8630% |

Đây là tín hiệu có chức năng tiềm ẩn ở một phần models, nên không thể đồng nhất
ownership/maturity/availability với năng lực thực tế. Nhưng phép đo dùng
**true task**, kể cả task ngoài legal bank. Không có route deployable được tạo
ra từ các con số này và chúng không chứng nhận an toàn cho class cũ.

Variant global-task mask đã kiểm tra trước chỉ thay class mask của **predicted
task trong bank hiện có**: accuracy 37,1463% → 38,7700%. Nó không tạo profile
cho 74 missing pairs; cũng không đổi `fc2 rank=0` thành knowledge đã chứng nhận.

## 6. Root cause nào nên sửa trước?

**Điểm sửa có phạm vi hẹp nhất là class admission lúc task còn current**, cho
receiver đã có profile của task đó:

```
Local training + graph aggregation hiện tại
    ↓
Class thiếu availability nhưng peer current sở hữu class
    ↓
Shadow Availability-only, không đổi weights/ranks
    ↓
Current evidence hợp lệ đánh giá gain và negative interference
    ↓
PASS → đăng ký class với provenance và phạm vi evidence rõ ràng
UNKNOWN/FAIL → giữ nguyên registration
```

Đây là availability repair, không donor update hoặc Head-Agg. Không tự mở toàn
bộ classes và không chọn ngưỡng bằng test. Cần so với local-mask baseline và
global-task-mask diagnostic, xác nhận weights/ranks không đổi, báo empirical
scope; không gọi là strict old-class safety nếu thiếu evidence.

Một class pass Availability-only vẫn có thể có `fc2 rank=0`. Trước production
phải định nghĩa protection/version của row đã được admission và kiểm tra nó qua
round tiếp theo. Không tự promote maturity hoặc ownership chỉ từ peer metadata.

**Missing task profiles là bài toán riêng.** Rejoining phải có contract cho
coverage/function transfer và feature compatibility. Chỉ gửi thêm source
profile hoặc class mask không đủ. Không sửa hai nhóm cùng lúc rồi gán toàn bộ
gain cho routing. Không khởi động lại APPLIANCE head-patch để chữa nhóm này.

Ưu tiên current-task admission trước vì có cơ hội tiếp cận evidence đúng thời
điểm. T5 không thể hồi tố CAL của T1/T2 theo protocol no historical replay.
Nếu triển khai sau này, đây phải là một variant được đánh giá với main baseline
và data/test protocol giữ nguyên; chưa có production admission trong commit này.

## 7. Integrity và giới hạn

- Checkpoint container trong original results ZIP được hash và bind với full audit.
- Role manifest và dataset metadata hashes khớp; historical BASE chỉ đọc counts.
- Tái dựng task/class registration: **98/98 receiver khớp**.
- **13/13 provenance checks PASS**, **5/5 checkpoint rank checks PASS**.
- Không train, inference mới, raw historical train/CAL, router fit hoặc test reassignment.
- Không đổi graph, ξ, backbone, optimizer, data allocation hoặc production defaults.
- Không tìm ngưỡng hay fit classifier để cứu một class cụ thể.
- Full test đã dùng làm development; không tuyên bố independent final confirmation.
- Metadata eligibility không thay thế kiểm tra functional compatibility / acceptance.

## Scripts và artifacts

- `tools/audit_denice_coverage_provenance.py`: source hashes, 588 task rows,
  3.332 receiver-class rows, chronology và graph opportunity audit.
- `tools/audit_denice_coverage_ranks.py`: đối chiếu 3.332 classifier rows trong
  trusted terminal checkpoint với provenance.
- `artifacts/denice_coverage_provenance_20261011.json`: summary máy đọc được.
- `artifacts/denice_coverage_provenance_20261011.zip`: CSV, receipts, historical
  source excerpts và báo cáo; không chứa dataset/model weights.

Ví dụ chạy metadata audit:

```powershell
python tools/audit_denice_coverage_provenance.py `
  --results 'C:\Users\khoak\Downloads\results (11).zip' `
  --audit-dir audit_denice/denice_bottleneck_20261010/full_cuda `
  --roles audit_denice/appliance_legacy_results11/inputs/roles/role_manifest.json `
  --metadata audit_denice/denice_bottleneck_20261010/inputs/metadata.json `
  --out audit_denice/coverage_provenance_new
```

Chạy rank audit bằng môi trường có Torch và dependencies của checkpoint:

```powershell
python tools/audit_denice_coverage_ranks.py `
  --checkpoint audit_denice/appliance_legacy_results11/inputs/checkpoint_task_5_all_rounds.zip `
  --provenance-dir audit_denice/coverage_provenance_new `
  --out audit_denice/coverage_provenance_new/ranks
```
