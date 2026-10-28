# Debug chọn đúng task trong Top-2

## Kết quả

Hai thử nghiệm bổ sung chưa cải thiện so với Top-1. Không đổi production,
notebook, V1, training, checkpoint hoặc expert threshold.

Panel development cố định: 100 clients × 2.048 = 204.800 mẫu, cùng checkpoint
results20/21 và seed chọn mẫu 20261028 + client ID như audit trước.

| Phương án | Accuracy | Macro-F1 | Rescue | Break |
|---|---:|---:|---:|---:|
| Top-1 hiện tại | 48,0718% | 27,5621% | — | — |
| Chọn task bằng LR trực tiếp | 47,2378% | 26,7511% | 7.440 | 9.148 |
| Chọn task bằng SVM RBF | 45,4873% | 26,0618% | 8.851 | 14.144 |

LR đổi 31.945 task: route rescue 7.965 nhưng route break 9.785.
RBF đổi 37.607 task: route rescue 9.501 nhưng route break 14.838.
Như vậy vấn đề không chỉ là chưa thử ranh giới phi tuyến. Hai cơ chế đều có
khả năng cứu một phần Top-2, nhưng chưa phân biệt đủ tốt khi nào phải giữ Top-1.
Đây không phải chứng minh mọi bộ chọn Top-2 đều bất khả thi.

## Cách thử

Giữ nguyên hai ứng viên từ router gốc. Fit một bộ phân xử task riêng từ
activation TRAIN đã lưu trong receiver checkpoint; dùng scaler có sẵn.
Không dùng nhãn/mẫu TEST để fit, không đọc historical raw TRAIN.

- LR: C=1, balanced, lbfgs, max_iter=1000, random_state=0.
- Follow-up RBF: C=1, gamma=scale, balanced, OVR decision scores; không phải
  xác suất task được hiệu chuẩn. Không sweep hyperparameter.
- Chỉ chọn một trong hai task; tie giữ Top-1. Adapter, class mask, experts đều
  thuộc task được chọn. Không so raw logits xuyên task, không mở union.
- Đây là fitting auxiliary router từ saved features, không phải zero-fit rule.

Nhãn thật chỉ dùng tính metric sau prediction. Tuy nhiên panel TEST đã được
dùng development nhiều lần, và RBF được thử sau kết quả LR: không gọi đây là
confirmation độc lập hoặc kết quả benchmark mới.

## Kiểm tra

- Top-1 confusion matrix khớp chính xác panel trước cho cả hai lần chạy.
- Native weights và toàn bộ service state (expert/Adam/router) không bị mutate.
- 8 protected file hashes V1/native khớp manifest.
- 3 tests mới PASS: fit chỉ saved memory, query không cập nhật bộ phân xử,
  singleton/absent runner-up, hướng binary SVM decision.
- 7 selector contract tests trước đó PASS.

## Kết luận và hướng tiếp theo

Giữ Top-1 cho kết quả chính thức: full TEST 48,08% / Macro-F1 27,61%.
61,30% vẫn là oracle dùng task thật, không phải kết quả bộ phân xử đã đạt.

Saved router features chỉ đại diện các class đã được lưu; expert có thể biết
class mà router không có mẫu đại diện. Refit cùng memory không tự tạo thêm
coverage, cũng không sửa được historical feature drift. Hai hạn chế này cần
đo riêng trước khi quy nguyên nhân chắc chắn.

Bước có cơ sở hơn việc tiếp tục đổi công thức trên TEST là thiết kế thí nghiệm
training-time: cho donor chạy input current-task trên routing encoder reference
của receiver và trả thống kê/features tương thích cho missing classes, hoặc
học bộ phân xử từ dữ liệu TRAIN hợp lệ có coverage tốt hơn. Cần tính bytes,
graph/provenance và drift; đây là đề xuất, chưa implement hoặc chứng minh gain.
Không tự đưa vào V1 hay yêu cầu train full lại.

Chạy tái lập (local bounded diagnostics):

```powershell
.venv-denice-gpu/Scripts/python.exe tools/debug_candle_v4_top2_direct_task.py
.venv-denice-gpu/Scripts/python.exe tools/debug_candle_v4_top2_direct_task.py --rbf
```

Receipts đầy đủ:
`audit_denice/candle_v4_top2_direct_task_20261029/audit.json`,
`audit_denice/candle_v4_top2_direct_task_rbf_20261029/audit.json`.
