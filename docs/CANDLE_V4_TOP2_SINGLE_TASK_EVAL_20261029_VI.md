# Sửa eval Top2: chọn một task trước, không mở union

Notebook mới: `eval_candle_v4_top2_task_select_kaggle.ipynb`.
Notebook union và training cũ được giữ nguyên để tái lập results22.
Không sửa V1/runtime/training/checkpoints. Đây là experimental readout, chưa phải
một phương pháp đã tăng Accuracy so với Top1.

## Thay đổi cơ chế

1. Giữ router gốc để lấy hai task ứng viên có memory từ input.
2. Primary rule khóa trước: cộng xác suất từng class từ frozen LR theo task,
   chọn tổng lớn hơn trong hai ứng viên (mass). Giữ Top1 khi tie hoặc thiếu task2.
3. Chạy adapter, native class mask và expert scope của đúng một task vừa chọn.
   Chỉ re-evaluate những rows đổi task; không so raw logits/experts giữa hai task.
4. Threshold expert0,5 và tất cả weights/optimizer giữ nguyên.
5. Không truyền true task/label vào predictor, không refit LR hay tune threshold.

Mass là tổng xác suất class ước lượng bởi LR, không được coi là posterior task
đã được hiệu chuẩn. Số class/task khác nhau có thể ảnh hưởng quyết định.

Đối chứng bổ sung giới hạn: receiver-space router class prototypes. Tính centroid
từ saved TRAIN feature memory, dùng cosine sau scaler có sẵn, chọn task có prototype
tương đồng nhất trong hai ứng viên. Không dùng prototype donor, không raw replay.
Không tiếp tục sweep threshold hoặc chọn winner trên TEST.

## Local development panel đã chạy

Cùng204.800 rows của audit trước, seed20261028+client ID,2048/client. Checkpoint
results20/21 có cùng state CRC/size theo audit trước. Top1 CM khớp tuyệt đối audit
trước. Hash checks native weights, experts/Adam/router state giữ nguyên.

| Readout | Accuracy | Macro-F1 | Rescue | Break |
|---|---:|---:|---:|---:|
| Top1 | 48,0718% | 27,5621% | — | — |
| Union Top2 cũ | 36,3369% | 20,2474% | 18.717 | 42.750 |
| Single-task mass Top2 | 48,0654% | 27,6294% | 1.093 | 1.106 |
| Single-task prototype Top2 | 46,6797% | 26,3485% | 5.885 | 8.736 |

Mass đổi4.632/204.800 rows; task-route rescue1.241, route break1.277.
Prototype đổi26.796 rows; route rescue6.420, route break9.373.

Kết luận: đường single-task đã loại bỏ phần lớn sụt giảm do union; mass chưa
cải thiện Accuracy so với Top1 (net -13 rows). Prototype cũng không vượt baseline.
Chưa đạt mục tiêu vượt CGoFed. Không khẳng định đây là quality PASS hoặc lấy
oracle61,30% làm dự báo. Panel TEST đã dùng development nhiều lần, không phải
confirmation độc lập. Dừng thêm thử nghiệm công thức trên panel này.

## Notebook

- Eval T0–T5 Round19 sau consolidation; cumulative TEST tương ứng.
- results19 và21 có Drive URLs tự tải, chỉ cần dataset100-clients + GPU/Internet.
- Top1 khớp production; full T5 CMs bắt buộc khớp results21.
- Báo thêm selected-task routing accuracy, không nhầm với Top2 candidate recall.
- T0 chỉ có một task thì hai nhánh phải giống nhau.
- Output mới tách biệt: `eval_candle_v4_round19_top2_task_select`.
- CLI cũ vẫn default `--top2-policy union`; notebook mới khóa `mass`.
- Notebook là lựa chọn tái lập/full readout, không phải khuyến nghị train lại.

## Verification

- 7/7 legacy union/archive regressions PASS.
- 7/7 selector contracts PASS.
- Real T0/T5 checkpoints + synthetic inputs: Top1 parity, forced runner-up route
  đúng adapter/mask path, task0 equality, frozen state. Không dùng dữ liệu thật
  trong regression này.
- Full cumulative TEST mới chưa chạy local/Kaggle trong thay đổi này.

Artifacts:
`artifacts/candle_v4_top2_task_select_development_20261029.json`,
`artifacts/candle_v4_top2_task_select_preflight_20261029.json`.
