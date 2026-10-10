# APPLIANCE Feature Separability & Donor Compatibility Audit

## Quyết định

**Không cho phép deployment acceptance cặp 65 ← 88 / class 20 theo cơ chế hiện
tại.** Mọi linear probe được chốt trước đều fail gate recall ≥95% và FAR ≤0,1%.
Tăng sketch dimension không giải quyết được. Receiver input/FC1 cũng chưa cung
cấp bằng chứng tách được bằng các linear probe này. Chưa sửa guard, chưa chạy
native smoke/full training và chưa thay runner production.

Đây **không phải chứng minh toán học rằng mọi classifier đều bất khả thi**.
Kết quả chỉ áp dụng cho các representation, fixed ridge probes, donor và split
development trong audit. Không mở thêm MLP hay classifier prototype.

## Fixture và donor hợp lệ

- DeNICE legacy Task 3 / round 19 từ results (13), cùng checkpoint đã audit.
- SHA256 checkpoint:
  `cac0d0e08c7c60495952fbf3dda3a22088e85acba06f0317fe5f8b80d4d021bf`.
- Receiver 65 thiếu class 20. Graph snapshot có peers **52, 88, 89**, positive α.

| Donor | Own BASE class 20 | FC2 rank | Maturity/dependency | Eligible |
|---|---:|---:|---|---|
| 52 | 0 | 0 | Không có head trưởng thành | Không |
| 88 | 24.451 | 2 | PASS | Có |
| 89 | 0 | 0 | Không có head trưởng thành | Không |

**Không có donor khác hợp lệ để so sánh trên graph này.** Không lấy donor ngoài
graph hoặc suy ra ownership từ inherited router memory. Không thay α/ξ/graph.

## Protocol đã khóa

### FIT

- Positive: **306 class-20 samples thuộc current CAL-FIT của donor 88**.
- Negative: current BASE của peers, chỉ class có checkpoint-owned support, loại
  class 20. Mỗi owner/class dùng tối đa **2.048 rows**, chọn theo hash row ID với
  seed cố định **20261010**. Tổng **41.863 negative BASE rows**.
- Tái dựng BASE Task 0→1→2→3 theo thời gian qua `CurrentBaseData`. Không đọc raw
  CAL task cũ. Sau `advance` không mở ngược partition BASE.
- Receiver FC1 của checkpoint Task 3 được áp vào BASE task trước **chỉ như
  retrospective diagnostic**. Không claim statistics đó đã có ở run cũ, hoặc
  có thể dùng cùng receiver feature coordinates trong một streaming run.

### Representation và objective

Tám representation chốt trước: unit shared sketch **16/32/64/128D**, preprocessed
input **39D**, unit preprocessed input **39D**, receiver FC1 **256D**, unit FC1.

Mỗi representation có hai **diagnostic linear ridge probes**, không phải guard
deployment mới:

1. `all_protected`: positive/negative mỗi bên mass 1/2; negative classes mass
   bằng nhau; trong class, nguồn weight theo usable sampled BASE rows.
2. `20_vs_13`: giữ thuật toán/ridge giống hệt, chỉ dùng hard negative class 13
   để kiểm tra ảnh hưởng của objective/weighting.

Ridge **0,001**, standardization từ FIT moments, intercept không penalize.
Probe tính ở FP64 standardized coordinates; không phải codec 68-byte FP32 của
guard trước. Không sweep C, weighting hay model family. Không train backbone.

Class 13 là development stratum đã lộ ở các audit trước; **không hardcode class
13 vào production method**. Các probe này không sửa head/mask/router.

### SELECTION / EVALUATION

1. Từ validation role, loại **768 rows đầu/owner/class** đã thuộc các panel cũ.
2. Loại thêm những row có exact canonical FP32 content hash trùng panel cũ ở
   owner khác hoặc trùng positive FIT/negative BASE dùng fit probe.
3. Có **1.202 row-content matches với panel cũ** bị loại thêm; 0 FIT matches.
4. Hash content + seed chia **50/50 SELECTION/EVALUATION**. Exact duplicate
   content cùng nhóm, không thể đi qua hai split. Cap **2.048 rows/owner/class/split**.
5. Ngưỡng mỗi probe lấy từ positive **SELECTION** bằng order statistic để đạt
   observed recall ≥95%, rồi khóa. Không thay ngưỡng bằng EVALUATION labels.
6. Chọn primary representation trong `20_vs_13` bằng SELECTION: joint gate trước,
   rồi lowest maximum owner/class FAR, dimension nhỏ hơn và tên khi tie.
   Không cấu hình nào pass SELECTION gate; primary là cấu hình để chẩn đoán.

| Split | Positive class 20 | Class 13 negatives | Toàn bộ negatives |
|---|---:|---:|---:|
| SELECTION | 1.121 | 3.208 | 13.590 |
| EVALUATION | 1.131 | 3.226 | 13.397 |

Không dùng lại validation tail đã xem để chọn ngưỡng. Đây vẫn là development
cùng source dataset, không phải final test, không claim population independence.
Thiếu nhiều rare-class strata sau exclusion; danh sách đầy đủ nằm trong artifact.

## Kết quả representation

**Các ngưỡng đều được chọn để đạt recall ≥95% trên SELECTION, rồi giữ nguyên.**
Recall EVALUATION được báo thật; không viết FAR@95 nếu recall EVALUATION dưới 95%.
Các con số dưới đây là **probe-only activation**, chưa kết hợp cosine/margin/veto
của installed route, không phải final system accuracy hay CAL certificate.

| Representation, objective `20_vs_13` | Eval recall | Eval FAR class 13 | AUROC 20-vs-13 |
|---|---:|---:|---:|
| Sketch 16D | 95,58% | 80,29% | 0,6821 |
| Sketch 32D | 95,23% | 78,33% | 0,6885 |
| Sketch 64D | 94,52% | 76,19% | 0,6894 |
| **Sketch 128D — selection-locked primary** | **94,61%** | **75,57%** | **0,6913** |
| Preprocessed input 39D | 95,40% | 81,53% | 0,6832 |
| Unit preprocessed input 39D | 94,52% | 75,57% | 0,6903 |
| Receiver FC1 256D | 93,81% | 77,31% | 0,6847 |
| Unit receiver FC1 256D | 93,28% | 76,91% | 0,6846 |

Primary có **2.438/3.226 class-13 false activations**; all-negative FAR **23,94%**.
Joint gate FAIL cả recall lẫn FAR.

`all_protected` cũng fail ở mọi representation. Ví dụ receiver FC1 đạt recall
**95,23%** nhưng FAR class 13 vẫn **83,63%**; raw input đạt recall **95,14%** nhưng
FAR class 13 **99,19%**. Chi tiết cả 16 probes lưu trong JSON.

Chuyển sang objective hard-negative cải thiện ranking nhưng không tới operating
point yêu cầu. Bằng chứng hiện tại không ủng hộ cách sửa đơn giản “tăng từ 16 lên
128D” hoặc “chỉ đổi weighting”. Cũng chưa có positive raw-input linear result để
quy toàn bộ lỗi cho compression.

Input chỉ có 39 dimensions: projection matrix ranks **16/32/39/39** tương ứng
sketch widths **16/32/64/128**. 128D không tạo thêm thông tin input. Unit sketch
còn bỏ amplitude; vì vậy width, normalization và learned-feature probes phải
được diễn giải riêng, không gọi đây là một dimension-only causal experiment.

## Donor compatibility: head hoạt động, cross-context rejection thất bại

Trên **cùng 1.131 EVALUATION class-20 positives**:

| Phép đo | Kết quả |
|---|---:|
| Donor 88 head, forced Task-3 context | 99,91% |
| Donor head trên receiver 65 features, forced Task-3 context | **100%** |
| Donor native router + classifier | **1.045/1.131 = 92,40%** |
| Target donor-vs-receiver head logit MAE | 6,7363 |
| Target feature cosine trung bình | 0,89483 |

Forced-context success cho thấy target function còn transferable. Nó không
chứng nhận rejection với class ngoài context hoặc toàn hệ thống label space.

Với head-margin threshold chọn ở SELECTION để giữ ≥95% target recall:

| Head score | Eval target recall | Eval FAR class 13 |
|---|---:|---:|
| Head trên donor features | 96,02% | **97,52%** |
| Head trên receiver features | 96,37% | **98,76%** |

Không có cosine/veto trong hai dòng này; đây là ablation của head compatibility,
không phải APPLIANCE final route.

Đáng chú ý, **ngay donor native inference**, class 13 đã bị nhầm nặng:

| Class-13 source | Rows | Donor đoán đúng 13 | Donor route Task 3 / đoán 20 |
|---|---:|---:|---:|
| Peer 52 | 393 | 15 | 369 |
| Donor 88 | 1.278 | 41 | 1.206 |
| Peer 89 | 1.555 | 63 | 1.468 |
| **Tổng** | **3.226** | **119 (3,69%)** | **3.043 (94,33%)** |

Donor có owned support và mature head cho class 13 **không đồng nghĩa chức năng
phân loại class 13 hiện tại còn tốt**. Support provenance vẫn cần bảo vệ class,
nhưng không được tự coi là competence quality hoặc CAL evidence.

## Integrity và giới hạn

- Content split disjoint; evaluation không trùng FIT hoặc các nội dung panel cũ.
- Receiver và donor complete state hash giữ nguyên.
- Chỉ donor thuộc live graph; không query donor khác để có con số đẹp hơn.
- Current CAL runtime access chỉ Task 3 / owner 88; **0 historical CAL reads**.
- Thresholds, chosen representation và probe parameters khớp lock sau evaluation.
- BASE cap/sampling, role SHA, source/lock hashes, row coordinates và content hashes
  được lưu trước predictions. Không có raw sample trong artifact đẩy GitHub.
- `_01/_02/_03` là rerun để thêm donor/native histogram và integrity instrumentation,
  không phải independent replications; không thay fit/selection policy giữa các rerun.

Fixed linear failures không loại trừ nonlinear structure, temporal/source shift
hoặc phương pháp representation khác. FAR là empirical finite-panel value,
không phải population guarantee. Một diagnostic probe pass cũng chưa thay được
CAL acceptance/native lifecycle của APPLIANCE.

## Communication

Audit này không transfer patch mới và không đo lại wire traffic. Giữ đúng kết
quả audit trước: experimental capability **12.835 bytes (~12,8 KB)**, tổng simulated
messages **6.926.708 bytes (~6,61 MiB)**, chủ yếu receiver function capsule
**6.829.408 bytes**. Không gọi coefficient/sketch payload là total cost. Sau khi
có protection hợp lệ vẫn còn việc giảm setup communication.

## Artifacts và bước tiếp theo

- Repo summary: `artifacts/appliance_feature_separability_development_20261010.json`.
- Local locks/results/code snapshot:
  `audit_denice/appliance_provenance_v1/feature_separability65_03/`.
- Driver: `tools/audit_appliance_feature_separability.py`.

Không cần chạy lại Kaggle cho audit này. Giữ patch pair này ở trạng thái **chưa
được phép deploy**; không mở rộng neighbor pool hay giảm gate để ép install.
Hướng tiếp theo phải có bằng chứng target-vs-protected-class competence từ
development trước khi donor/patch được coi là an toàn. Với graph hiện tại, không
có fallback donor cho class 20. Không full retrain chỉ để thử tăng sketch width.
