# Eval CANDLE V4 T0–T5 Round19: Top-1 và Top-2 thực tế

Notebook: `eval_candle_v4_tasks0_5_round19_top1_top2_kaggle.ipynb`.
Chỉ eval, không train/refit hoặc sửa V1.

## Input và checkpoints

- Dataset100-clients nguyên bản.
- results19 (Drive đã điền) chứa task archives T0–T4.
- results21 (gắn ZIP vào Kaggle Input hoặc điền Drive URL) chứa Task5 archive và
  full Top-1 TEST dùng làm đối chứng correctness.
- Hai ZIP phải có run lock giống nhau, training source3731fe1.
- Chỉ trích continuation biên cuối mỗi task; verify manifest đủ rounds0–19 và checksum.
- Đánh giá state sau consolidation, không dùng delta weights thô của Round19.

## Inference

- Top-1 giữ nguyên V4 continuous router, native task mask và experts scoped.
- Top-2 chỉ xét hai task có memory do router chọn từ input.
- Native classifier cạnh tranh trên hợp class của hai task; adapter vẫn theo Top-1.
- Experts của cả hai task được tham gia, threshold0,5; highest normalized excess,
  tie-break theo class ID như production. Không expert nào kích hoạt thì dùng native.
- Không dùng nhãn thật để lựa task, mở mask hoặc fusion. Đây không phải oracle
  Top-2 có Accuracy61,30% trên panel trước đây. Không cam kết Top-2 thực tế sẽ tốt hơn.
- T0 chỉ có một task thì hai nhánh phải tương đương. Task thiếu memory không được
  thêm làm candidate chỉ vì có zero-score trong vector router.

## Evaluation

- Cumulative TEST chỉ chứa class T0 đến t; mỗi row giao một lần qua client shards.
- Seed partition42+104729×task; T5 seed523687 giống results21.
- Accuracy/Macro-F1 pooled từ confusion matrix, không average client F1.
- Macro-F1 chính giữ fixed34 labels như native; báo thêm Macro-F1 seen classes.
- Báo riêng Top-1 route accuracy và Top-2 task recall; không gọi Top-2 recall là
  accuracy của một router đã chọn task cuối cùng.
- Báo rescue, break, net rescue, per-class và per-client.
- Mỗi client kiểm tra batch đầu Top-1 khớp production; T5 kiểm tra toàn bộ
  confusion matrix từng client khớp results21. Lệch thì dừng.
- Hash checks xác nhận native weights, expert/Adam/router state không đổi.

Output gồm `summary.csv`, sáu `task_{t}_round19_topk.json`, `completion.json`.
Không thay kết quả training gốc. Full TEST readout là development evaluation,
không phải confirmation độc lập sau khi chọn thuật toán trên TEST.

## Verification đã chạy local

- 7/7 synthetic contract tests PASS.
- Real checkpoint T0/T5, clients0/99, mỗi trường hợp32 input tổng hợp: Top-1
  prediction exact match với production, T0 Top-2=Top-1, weights/bank giữ nguyên.
- Không mở raw TRAIN/TEST hoặc chạy full evaluation trong bước chuẩn bị notebook.
- Local sklearn1.9 chỉ để regression; notebook yêu cầu pinned sklearn1.6.1.

Artifact: `artifacts/candle_v4_round19_topk_preflight_20261028.json`.
Script: `tools/eval_candle_v4_task_round19_topk.py`.
