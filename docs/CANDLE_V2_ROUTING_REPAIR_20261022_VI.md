# CANDLE V2 — sửa inference bằng saved router và class registry

Ngày 22/10/2026. Nối tiếp debug checkpoint tại `eb9b281`.

Cập nhật: full TEST của repair đã hoàn tất trong results (15), Accuracy
**35,7030%**, pooled Macro-F1 **18,4767%**, route accuracy **38,8590%**.
Xem `CANDLE_V2_RESULTS15_FULL_TEST_20261022_VI.md`. Các mục development và trạng
thái chưa chạy TEST bên dưới ghi lại thời điểm triển khai trước lượt eval đó.

## 1. Đã sửa gì

Policy mới: **`binary_cosine_registered_v1`**, primary **BinaryCosineRegisteredSelf**.

```
Input
  → masked CNN/GRU: một lần forward
      ├─ FC1 → cosine FC2
      └─ context activations → saved binary_cosine router → predicted task
  → chỉ cho cạnh tranh những class available có class_tasks == predicted task
  → predicted class
```

- Không truyền true task/class vào inference.
- Dùng đúng router, thresholds và activation memory đã lưu trong checkpoint.
- Class được phép dự đoán lấy từ **CANDLE `available + class_tasks`**, bao gồm
  imported classes đã train. Không chỉ dùng native `episode_classes` vốn phản
  ánh local registration và có thể thiếu imported classes.
- Một model local, không donor inference, không server/global model mới.
- Không sửa weights, connection masks, neuron ranks hoặc router memory khi eval.
- Task được router chọn không có authorized class thì giữ task-free prediction
  và ghi rõ fallback count. Không tự mở class chưa available.

Đây là thay đổi inference policy có chủ đích, không phải chỉ chỉnh một threshold.
Nó xử lý một phần cross-task competition đã thấy ở results (14). Training
objective, learning rate, data budget và loss coefficients giữ nguyên.

## 2. Đối chứng router đã lưu

Trên 2.176 validation inputs cũ, đủ 64 unique inputs/class, cùng 98 model:

| Policy | Development Accuracy |
|---|---:|
| Task-free cosine của run cũ | 5,1635% |
| Saved binary router + native episode_classes | 19,6025% |
| Saved binary router + CANDLE available/class_tasks | **21,2405%** |

Route accuracy **34,6807%**. Registered variant có **40.137 rescue / 5.853 break**
trên 213.248 prediction rows. Empty-scope fallback: **0**.

Việc thêm imported classes vào task scope cải thiện net accuracy trong phép đo
này, nhưng không có nghĩa mọi imported class đã học tốt hoặc mọi override an toàn.

## 3. Khóa policy rồi đánh giá validation tail khác nội dung

Protocol ghi trước khi đọc tail: chỉ policy trên, không tune threshold hoặc fit
classifier. Loại toàn bộ content hashes của 2.176 input cũ. Giữ tối đa 64 unique
inputs/class, owner order deterministic như panel trước.

Tail có **2.132 inputs**: 33 class có 64, class 31 chỉ còn 20. Với 98 model có
**208.936 prediction rows**, không phải 208.936 input độc lập. Không trùng input
với panel trước, nhưng vẫn là development source đã từng được dùng trong dự án;
không gọi là independent scientific confirmation.

| Metric trên tail | Task-free baseline | Registered-router repair |
|---|---:|---:|
| Accuracy | 5,2011% | **21,7540%** |
| Pooled Macro-F1, 34 class | 3,6010% | **19,8784%** |
| Old T0–T4 accuracy | 3,2313% | **22,3331%** |
| New T5 accuracy | 23,0410% | **16,5094%** |

- Route accuracy: **34,9198%**.
- Rescue **40.584**, break **5.999**, net correct **+34.585**.
- Accounting: `delta_accuracy = (40584 - 5999) / 208936`.
- Empty-scope fallback: **0**.

**Repair cải thiện tổng thể nhưng có tradeoff rõ ràng với class mới.** Đây không
phải PASS theo yêu cầu break=0, không phải chứng nhận an toàn trên mọi class và
chưa phải bằng chứng CANDLE V2 vượt V1.

## 4. Kiểm tra correctness và integration đã thực hiện

### One-forward implementation

`features_and_context()` thực hiện cùng các phép masked Conv/BN/pool/GRU/FC1
với model hiện có. Với probe 16 inputs/client trên 98 model:

- Max absolute error của FC1 features so với canonical path: **0**.
- Max absolute error context activations: **0**.

Trên toàn bộ panel cũ, baseline prediction trùng cached task-free prediction
và repair prediction trùng saved-router control: **0 disagreements**.
Fingerprint model và router trước/sau inference khớp.

### Evaluator thực tế, không chỉ helper

Chạy `tools/eval_candle_v2.py` với policy mới và explicit validation panel,
qua cả 98 model, dùng chính disjoint shard allocator của full evaluator:

- **2.132/2.132 inputs**, mỗi input được dự đoán một lần.
- Baseline correct **108**, repair correct **468**.
- Rescue **417**, break **57**; `468 - 108 = 417 - 57`.
- Accuracy **5,0657% → 21,9512%**.
- Route accuracy **35,9756%**, fallback **0**.
- `evaluation_role=validation_diagnostic`, `test_rows=null`.

Đây là allocation khác phép đo 98 model × toàn panel ở mục 3, vì vậy không trộn
hai denominator. Lượt integration cuối dùng checkpoint/roles trích trực tiếp
từ outer `results (14).zip`.

### Input packaging

Đã chạy preparation cho outer ZIP và mounted extracted results. Role-manifest
SHA vẫn là `b89cd6868f4185d3307f1bbed56dafd170a10c73a1752496aeb944b037d98a15`.
Terminal checkpoint SHA vẫn là
`23ce5c1ff80c75cb8ca7c1f89c339579c86e795967796b9bea8333ba347ea8bf`.
Original confusion có **13.505.771 rows / 426.077 correct**.

Resolver hỗ trợ cả checkpoint ZIP và checkpoint folder có manifest, roles ZIP
hoặc roles folder đã giải nén. Nếu có nhiều runs/roles thì yêu cầu chỉ rõ path,
không tự chọn tùy ý. Không yêu cầu tạo hoặc upload runtime ZIP mới.

## 5. File chạy ngay trên Kaggle

**`eval_candle_v2_repair_kaggle.ipynb`** — đánh giá lại weights hiện có, không train.

Attach dataset gốc **`100-clients`**, bật GPU + Internet và chạy notebook.
Notebook tải `results (14)` trực tiếp từ link Google Drive đã cấu hình, vào
`/kaggle/temp/candle_v2_results14.zip`, rồi dùng checkpoint/roles trong ZIP đó.
Source clone từ GitHub với full commit SHA, không nhúng source.

Nếu muốn dùng results đã mount thay vì Drive, xóa `RESULTS_DRIVE_URL` và đặt
`RESULTS_ROOT` hoặc `CHECKPOINT` trong cell cấu hình. Cơ chế tải dùng cùng mẫu
gdown như `tools/build_denice_class_meta_notebook.py` trong repository.

Full evaluator giữ đúng protocol cũ: 13.505.771 TEST rows chia disjoint cho các
receiver, mỗi row tính một lần. Cùng một forward lấy cả baseline task-free và
repair; báo Accuracy, pooled Macro-F1, route accuracy, per-client metrics và
rescue/break trên cùng pool. Khi có original confusion, baseline phải khớp
chính xác; mismatch sẽ dừng và xuất diagnostic thay vì ghi completed result.

Files cần xem:

- `repair_comparison.json`: baseline/repair và delta, rescue/break.
- `completion.json`: full metrics, route accuracy và fallback count.
- `paired_taskfree_confusion_matrix.json`, `confusion_matrix.json`.
- `candle_v2_routed_repair_task5.zip`: kết quả gọn; không chứa checkpoint copies.

Source, task5 input và roles chuẩn bị trong `/kaggle/temp`, không đưa vào output.

## 6. Training runner và backward compatibility

- Config/notebook training mới chọn rõ `binary_cosine_registered_v1`; training
  objective không đổi. Launcher ghi primary/inference policy và module hashes.
- Checkpoint cũ không có policy key vẫn mặc định task-free. Evaluator repair
  dùng override có tên và ghi cả policy gốc lẫn override trong pipeline lock.
- `taskfree_cosine_v1` vẫn có thể chọn làm đối chứng; V1 không bật V2 hooks.
- Chưa chạy fresh training lại hoặc full TEST repair trong lượt debug này.
- Full TEST đã công bố của run gốc vẫn là **3,1548% Accuracy / 2,4369% Macro-F1**.
  Không thay hai số đó bằng development results ở trên.

Run full TEST sau khi chọn policy bằng development là engineering evaluation
sau điều chỉnh; không tự biến thành untouched final confirmation cho paper.

## Receipts

Local: `audit_denice/candle_v2_routing_repair_20261022_v1/`.
Public compact receipt: `artifacts/candle_v2_routing_repair_20261022.json`.
Không commit raw validation panel hoặc checkpoint vào repository.
