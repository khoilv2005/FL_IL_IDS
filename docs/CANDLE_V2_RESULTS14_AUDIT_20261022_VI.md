# CANDLE V2 — audit full run results (14)

Ngày: 22/10/2026. File gốc `C:\Users\khoak\Downloads\results (14).zip`,
4.822.736.054 bytes. Training source `a7830e2`, notebook phát hành `a0429ef`.
Đây là audit đọc outputs, không train lại, không chạy thêm model inference,
không điều chỉnh hyperparameters hoặc sửa production code theo TEST.

## 1. Kết luận chính

**Run đã hoàn tất kỹ thuật nhưng CANDLE V2 hiện tại không đạt mục tiêu cải thiện.**
Accuracy và Macro-F1 thấp hơn baseline V1 rất nhiều; không thể dùng loss thấp
hoặc virtual prototype CE thấp làm bằng chứng transfer thành công.

| Metric, cùng full TEST | Historical V1 BinarySelf | CANDLE V2 TaskFreeCosineSelf | Delta |
|---|---:|---:|---:|
| Accuracy | 31,2446% | **3,1548%** | −28,0898 pp |
| Pooled Macro-F1, 34 class | 15,4274% | **2,4369%** | −12,9905 pp |
| Correct predictions | 4.219.822 | **426.077** | −3.793.745 |

V2 mean client Macro-F1 là1,1321%, khác pooled Macro-F1 ở bảng trên. Route
accuracy=null vì inference V2 không dự đoán task; không ghi nhầm thành0%.

V1 comparator là legacy results(11) đã dùng làm nguồn config CANDLE V1, không
phải một baseline vừa train lại cùng lượt. Global role-manifest SHA khớp;
class supports và toàn bộ TEST partition receipt khớp (client membership,
seed, per-client sample/class counts). Đây là so sánh mô tả phù hợp protocol,
chưa phải controlled ablation hoặc multi-seed claim.

## 2. Run có hoàn tất và metric có đọc đúng không?

- Sáu task, mỗi task đủ20 rounds:120 communication records,8.940 client-round
  training records. Active clients theo task:50,60,70,80,89,98.
- Có đủ sáu `checkpoint_task_*_all_rounds.zip`, exact roles ZIP và evaluation ZIP.
- Completion ở root và evaluator giống nhau, `completed=true`.
- Evaluation đủ **13.505.771/13.505.771** rows, full34 class.
- Tính lại Accuracy/Macro-F1 từ saved confusion matrix khớp completion JSON.
- Eval khoảng151 giây trên GPU T4. Kiểm tra này chưa độc lập rerun inference
  hoặc đọc lại toàn bộ terminal tensors/checkpoint checksums.

Do đó không có bằng chứng metric bị lệch dấu phần trăm hoặc cộng sai confusion
matrix. Điều này không loại trừ mọi implementation bug trong training/inference.

## 3. Failure mode quan sát được: lệch rất mạnh về task cuối

Task5/classes30–33 chỉ chiếm **110.252/13.505.771 =0,8163%** true TEST rows,
nhưng nhận **9.551.425 predictions =70,7211%**.
Riêng class32 nhận **7.613.940 predictions =56,3755%** toàn TEST.

| True task | Rows | Correct | Accuracy cuối run |
|---|---:|---:|---:|
| T0 | 409.691 | 1.209 | 0,2951% |
| T1 | 5.576.297 | 13.295 | 0,2384% |
| T2 | 4.149.643 | 94.232 | 2,2708% |
| T3 | 2.824.677 | 155.110 | 5,4912% |
| T4 | 435.211 | 95.793 | 22,0107% |
| T5 | 110.252 | 66.438 | 60,2601% |

Gộp T0–T4: accuracy chỉ **2,6848%**. Đây là bằng chứng cumulative inference
thất bại và last-task prediction bias, chưa tách được logit competition khỏi
representation forgetting. Không có eval ở thời điểm vừa hoàn tất từng task,
nên không tính được average forgetting chuẩn từ bảng này.

## 4. Không chỉ là thiếu availability

- Label coverage: **78,1264%** (10.551.570 rows có true class trong availability).
- Accuracy trong phần covered chỉ **4,0380%**.
- Có10.125.493 errors ngay trong covered rows; tương đương77,41% mọi errors.
- Số class available/client có quantiles min/25%/median/75%/max:
  **16 /27 /28 /34 /34**.
- Median client accuracy chỉ khoảng0,7945%; best client khoảng25,26%.

Mở thêm class mask không tự khôi phục accuracy: phần lớn lỗi nằm ở phân loại
giữa các class đã có quyền tham gia, không phải chỉ class bị cấm dự đoán.

## 5. Prototype learning có chạy, nhưng chưa chứng minh functional transfer

| Task | CE round0 → round19 | Virtual CE round19 | Preservation round19 | Unique receiver-class pairs đã virtual-train |
|---|---|---:|---:|---:|
| T0 | 1,3069 →0,7842 | 0,3972 | 0 | 90 |
| T1 | 0,5879 →0,2328 | 0,0156 | 0,05280 | 108 |
| T2 | 1,6194 →0,4852 | 0,0229 | 0,04491 | 126 |
| T3 | 1,9667 →0,6049 | 0,0516 | 0,02603 | 138 |
| T4 | 3,4287 →1,2844 | 0,2044 | 0,00102 | 149 |
| T5 | 4,4457 →1,9852 | 0,3057 | 0,00032 | 93 |

CE là mean theo client, trên current real training data; không phải cumulative
validation loss. Virtual CE đo khả năng phân loại **prototype vectors**, không
đo target recall trên actual receiver inputs. Các pair counts là đã có optimizer
steps trên objective chứa virtual prototype, không phải transfers qua acceptance.

Tổng12.415 row-import events trong logs; không có unsupported-prototype skip.
Các dấu hiệu này chứng minh đường training được kích hoạt, chưa chứng minh nó
đã tạo knowledge có ích khi inference trên raw samples.

## 6. Điểm yếu xác định được từ source preservation

Trong `candle_v2.py`, teacher và student được slice xuống **old classes** rồi
mới softmax/KL, trên current inputs:

```
KL(softmax(teacher_old / T), softmax(student_old / T))
```

Term này giữ tương quan giữa old logits, **không giữ probability mass của nhóm
old so với new**. Ví dụ teacher/student old logits cùng[1,0] thì KL=0; student
vẫn có thể thêm new logit9 và dự đoán new class cho input đó. Cosine scale10
không ngăn được loại cạnh tranh này.

Thêm hai giới hạn đã có trong implementation/config:

- Teacher được clone trước mỗi local round, nên là reference di động; không
  phải fixed checkpoint của task trước.
- Không có old raw positive inputs trong preservation. KL thấp trên current
  inputs không chứng minh old-class behavior trên historical distribution còn giữ.

Preservation T5≈0,00032 đồng thời70,72% predictions vào T5 phù hợp với giới hạn
trên. Đây là underconstraint thực sự của objective, nhưng chưa là controlled
causal attribution cho toàn bộ28,09pp accuracy loss.

V2 đã đổi nhiều thành phần cùng lúc: bỏ task routing, tắt context adapters,
đổi cosine classifier, thêm alignment/injection/preservation. Tôi chưa có cơ sở
quy failure chỉ cho prototype alignment hoặc khẳng định raw head weights đã mất
hoàn toàn old-class information.

## 7. AMP và communication

-158.953 optimizer steps thành công,527 skipped steps (khoảng0,33% attempts).
-82 client-round records không có optimizer step, chủ yếu T0 (81 records).
-377 optimizer health records báo nonfinite gradients sau clipping; AMP có skip
  và training tiếp tục. Đây là numeric issue cần theo dõi, chưa đủ evidence coi
  nó là nguyên nhân chính của last-task collapse.
- Prototype JSON payload cộng dồn **1.229.867.346 bytes ≈1,23GB** trong120 rounds.
  Đây là bytes đã encode ở application harness, chưa cộng native model/capsule
  traffic hoặc wire headers. Không được gọi communication toàn hệ thống là nhẹ
  chỉ vì mỗi prototype nhỏ.

## 8. Quyết định sau run

1. Không gọi V2 hiện tại là bản cải thiện V1 hoặc dùng nó làm phương pháp final.
2. Giữ nguyên run/TEST result; không tăng lambda, chỉnh threshold hoặc đổi loss
   rồi chọn bằng chính TEST này.
3. Bước chẩn đoán phù hợp là dùng checkpoint đã có: trong-task/old-only oracle
   diagnostics, old-vs-new cosine margins và representation retention trên
   development. Oracle chỉ là diagnostic, không được thay headline3,1548%.
4. Tách task-free/cosine-only control khỏi prototype alignment/injection để biết
   thành phần nào thực sự giúp hoặc làm hại, trước khi full training lần nữa.

Audit lượt này chỉ đọc saved outputs và source, chưa thực hiện các inference
diagnostics ở bước3. Chưa sửa production runner hoặc tạo notebook retrain mới.

## Receipts

Tool: `tools/audit_candle_v2_results.py`.
Local output: `audit_denice/candle_v2_results14_20261022_v2`.
Public compact receipt: `artifacts/candle_v2_results14_audit_20261022.json`.
Raw archive/checkpoints không commit vào Git.
