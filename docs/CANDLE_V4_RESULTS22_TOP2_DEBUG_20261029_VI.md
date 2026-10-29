# Results22: actual Top2 union thất bại, không phải oracle Top2

## Xác minh

ZIP có đủ completion, sáu task reports và summary.csv. Cumulative TEST T0–T5,
checkpoint Round19 sau consolidation. Đã tính lại Accuracy/Macro-F1 từ CMs,
kiểm tra sum per-client CMs và net correct = rescue - break.
T5 Top1 khớp confusion matrix results21 trên toàn bộ client theo evaluator receipt.
T0 chỉ có một task, hai nhánh hoàn toàn giống nhau.

| Task | Top1 Accuracy | Top2 Accuracy | Top1 Macro-F1 | Top2 Macro-F1 |
|---|---:|---:|---:|---:|
| 0 | 80,20% | 80,20% | 8,11% | 8,11% |
| 1 | 63,86% | 56,85% | 13,53% | 12,27% |
| 2 | 57,53% | 45,09% | 18,60% | 14,39% |
| 3 | 49,01% | 36,40% | 24,09% | 17,45% |
| 4 | 48,22% | 36,25% | 27,06% | 19,89% |
| 5 | 48,08% | 36,40% | 27,61% | 20,34% |

Macro-F1 trên fixed34 labels như native. T0–T4 còn có seen-class F1 trong reports;
không trộn hai định nghĩa để so sánh.

## Full T5

- 13.505.771 rows; actual Top2 cứu1.239.602, làm sai thêm2.817.813.
- Net -1.578.211 correct, tương đương -11,6855 điểm Accuracy.
- 8 clients cải thiện, 92 giảm.
- Client62 giảm65,22% ->26,17%; client20 giảm68,68% ->37,74%.
- True task có trong Top2 trên65,22% rows. Nhưng final class prediction thuộc đúng
  task chỉ38,50%, so với51,09% ở Top1.

65,22% là candidate task recall, không phải routing accuracy của quyết định cuối.
Oracle trước đó dùng true task để phân xử; actual union không có thông tin đó.

## Attribution local có kiểm soát

Giữ nguyên checkpoint/weights/experts/scaler/LR. Cùng panel204.800 TEST rows của
audit trước:2048/client, seed20261028+client ID. Không train, refit hoặc tune.
Results20/21 continuation có cùng CRC và byte size; dùng local results20 checkpoint.
Panel baseline CM khớp tuyệt đối baseline audit Top2 oracle trước.

| Native mask | Expert scope | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Top1 | Top1 | 48,07% | 27,56% |
| Top2 union | Top1 | 42,86% | 23,35% |
| Top1 | Top2 union | 40,51% | 23,92% |
| Top2 union | Top2 union | 36,34% | 20,25% |

Trong42.750 panel breaks của actual Top2:

- 25.226 (~59,01%): expert task thứ hai ghi đè dự đoán đúng.
- 17.524 (~40,99%): không expert nào kích hoạt; native expanded mask chọn sai.
- Không có break ngoài hai nhóm trên.

Hai tác động không cộng tuyến tính vì expert có thể override native fallback.
Đây là attribution panel, không ngoại suy tỷ lệ chính xác cho full TEST.

## Kết luận

Không thấy lỗi checkpoint/metric qua các invariant đã kiểm tra. Actual union Top2
không đạt: mở rộng tập ứng viên không giải quyết được việc chọn giữa hai task.
Không nên đưa union Top2 vào nhánh inference chính hoặc train full lại để cứu
readout này. Giữ Top1 chính thức48,08% /27,61%.

Cross-task expert specificity và native logit competition đều cần xử lý. Experts
được học từ current-task positive/negative samples, nên không được mặc định score0,5
đủ phân biệt toàn bộ task cũ/mới khi bỏ task scope. Chưa có bằng chứng xác định toàn
bộ lỗi xuất phát riêng từ thiếu historical negatives.

Nếu tiếp tục hướng Top2, cần một cơ chế input-only chọn task trước khi dùng nguyên
pipeline scoped; không tuyên bố oracle61,30% là mức actual union có thể đạt.
Phát triển cơ chế mới phải có evaluation độc lập phù hợp, không chỉnh threshold trên
results22 rồi gọi kết quả là confirmation.

Artifact: `artifacts/candle_v4_results22_debug_20261029.json`.
Script local: `tools/debug_candle_v4_results22_top2.py`.
Không sửa V1, V4 runtime, notebook training hoặc checkpoint.
