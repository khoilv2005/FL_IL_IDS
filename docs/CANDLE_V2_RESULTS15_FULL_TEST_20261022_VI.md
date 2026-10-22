# CANDLE V2 — kết quả full TEST của routing repair, results (15)

Audit tiếp theo so V1/V2 với cùng loại router trên development:
[Matched-router debug](CANDLE_V1_V2_MATCHED_ROUTER_DEBUG_20261022_VI.md).
Không thay thế các số full TEST trong báo cáo này.

Ngày 22/10/2026. Archive `C:\Users\khoak\Downloads\results (15).zip`, 179.989 bytes.
Audit chỉ đọc outputs đã lưu, không train hoặc chạy thêm model inference.

## 1. Kết quả đã hoàn tất

**Accuracy 35,7030%, pooled Macro-F1 18,4767%, route accuracy 38,8590%.**

Đây là kết quả full TEST của checkpoint T5/R19 với policy
`binary_cosine_registered_v1`, không phải số development 21,75% trước đó.

| Metric | V2 task-free cũ, paired control | V2 registered-router repair | Delta |
|---|---:|---:|---:|
| Accuracy | 3,1548% | **35,7030%** | +32,5482 pp |
| Pooled Macro-F1, 34 class | 2,4369% | **18,4767%** | +16,0398 pp |
| Correct | 426.077 | **4.821.969** | +4.395.892 |
| TEST rows | 13.505.771 | 13.505.771 | 0 |

98 local predictors chia cùng full TEST thành các disjoint receiver shards;
mỗi input tính một lần. Đây không phải mỗi model chạy toàn bộ N TEST rows,
không phải cluster-model mean và không phải oracle task inference.

Mean client Macro-F1 là **14,0633%**, khác pooled Macro-F1 18,4767% ở trên.
Mean client accuracy xấp xỉ pooled accuracy do shards gần bằng nhau. Runtime
evaluator **261,66 giây**, chưa tính tải Drive, setup hoặc checkpoint preparation.

## 2. Những kiểm tra đã khớp

- `completed=true`, `evaluation_role=full_TEST`, đủ 13.505.771 rows và 34 class.
- Tính lại Accuracy/Macro-F1 từ confusion matrix khớp completion.
- Paired task-free confusion matrix **khớp chính xác từng ô** với results (14).
- Partition receipt khớp results (14): cùng receiver IDs, seed 523687, sample
  counts/class counts và mỗi global row được assign một lần.
- Terminal và graph SHA khớp checkpoint gốc. Không có training mới.
- Tổng samples/correct/rescue/break của 98 per-client records khớp global.
- Weighted route accuracy từ per-client records khớp global.
- Nested output ZIP khớp các JSON bên ngoài.
- Empty-scope fallback: **0**. Label coverage giữ **78,1264%**.
- Pipeline lock ghi `oracle=false`, `donor_inference=false`, dùng saved router
  dự đoán task từ input, authorization theo `available + class_tasks`.

Do đó không có bằng chứng gain đến từ đổi checkpoint, giảm TEST scope, thay
allocation hoặc cộng sai metric. Audit outputs này không độc lập replay từng
sample/model; inference source đã được kiểm tra trong lượt development trước.

## 3. Accounting rescue/break

```
rescue       = 4.618.108
break        =   222.216
net correct  = 4.395.892
N            = 13.505.771
```

\[
35,7030\%-3,1548\%
=\frac{4.618.108-222.216}{13.505.771}
=32,5482\text{ điểm phần trăm}.
\]

Có gain ròng lớn nhưng không phải zero-break transfer. Hơn nửa số prediction
đúng ít ỏi của baseline task-free bị đổi thành sai; phải giữ số break trong
báo cáo thay vì chỉ nêu rescue.

## 4. Tác động khác nhau giữa các task

| True task | TEST rows | Task-free accuracy | Repair accuracy |
|---|---:|---:|---:|
| T0 | 409.691 | 0,30% | **34,27%** |
| T1 | 5.576.297 | 0,24% | **37,29%** |
| T2 | 4.149.643 | 2,27% | **33,16%** |
| T3 | 2.824.677 | 5,49% | **40,31%** |
| T4 | 435.211 | 22,01% | **8,93%** |
| T5 | 110.252 | 60,26% | **44,03%** |

Phần gain chủ yếu đến từ khôi phục predictions T0–T3; T4/T5 bị giảm. Ví dụ
class 24 recall 32,35% → 10,42%; class 32 recall 61,77% → 45,10%.

Tỷ lệ predictions rơi vào T5 giảm **70,72% → 3,91%**. Điều này xác nhận failure
mode last-task dominance đã được giảm mạnh; chưa chứng minh phân phối prediction
mới đã tối ưu hoặc mọi class hiếm được xử lý tốt.

## 5. So với V1: ghi đủ các variant

Cùng TEST class counts và partition receipt với historical results (11):

| Cấu hình | Accuracy | Pooled Macro-F1 |
|---|---:|---:|
| V1 BinarySelf | 31,2446% | 15,4274% |
| V1 MulticlassSelf | **37,1463%** | **19,7131%** |
| V2 registered-router repair | 35,7030% | 18,4767% |

Repair cao hơn V1 BinarySelf **4,4584 pp accuracy / 3,0493 pp Macro-F1**, nhưng
thấp hơn V1 MulticlassSelf **1,4433 pp accuracy / 1,2364 pp Macro-F1**.

Đây là so sánh một historical run/seed với các cấu hình khác nhau, không phải
controlled training ablation. Không được chỉ chọn V1 Binary rồi tuyên bố V2
vượt mọi V1, hoặc quy gain riêng cho prototype alignment/virtual CE. Gain do
thay inference policy trong cùng V2 checkpoint đã được đo trực tiếp bằng paired
control; đóng góp riêng của training-time transfer vẫn chưa được tách.

## 6. Trạng thái hiện tại

- Full TEST của **cấu hình repair hiện tại đã có**, có thể ghi 35,7030% / 18,4767%
  làm kết quả run này; giữ riêng baseline cũ 3,1548% / 2,4369%.
- Không cần gọi số development 21,75% là kết quả cuối hoặc trộn hai pool.
- Đây là post-development repair evaluation. TEST của run trước đã được dùng để
  phân tích lỗi; chưa có untouched confirmation hoặc multi-seed benchmark.
- Router vẫn chỉ đúng 38,86%; đây là hạn chế lớn còn lại. Chưa đổi classifier,
  guard, thresholds hoặc train full thêm trong lượt audit này.
- Không tự chuyển sang một router khác dựa trên chính TEST mới để chọn kết quả.
  Cần giữ baseline repair này và dùng development nếu tiếp tục cải tiến.

## Receipts

Tool: `tools/audit_candle_v2_repair_results.py`.
Local output: `audit_denice/candle_v2_results15_20261022_v2`.
Public compact receipt: `artifacts/candle_v2_results15_audit_20261022.json`.

```powershell
.venv-denice-gpu\Scripts\python.exe -X utf8 tools/audit_candle_v2_repair_results.py --archive "C:/Users/khoak/Downloads/results (15).zip" --original "C:/Users/khoak/Downloads/results (14).zip" --out <NEW_OUTPUT>
```
