# Một notebook chỉ TRAIN + EVAL CANDLE V4

File: `train_eval_candle_v4_only_kaggle.ipynb`, companion cùng tên `.py`. Kaggle GPU + Internet, mount `100-clients`, Run All. Đây là **fresh training từ Task0**, không phải sửa/eval results16/17.

## Cấu hình giữ nguyên

100 clients, seed42, T0–T5×20 rounds; native batch2048, LR0,001, AMP; 100% current-task TRAIN, validation0%; không BASE/FIT/CAL, raw replay hoặc shadow predictor. Coverage experts, donor handoff, scoped threshold0,5 và continuous router với exact-bootstrap memory fix giữ nguyên.

V4 có config và requirements riêng (`configs/candle_v4.json`, `requirements-candle-v4-kaggle.txt`), giá trị training/dependency khởi đầu giống bản bootstrap-fixed. Không sửa V1 code/config/notebook. Launcher V4 tự resolve dataset; không gọi launcher train V1.

## Chỉ một predictor V4 được đánh giá

Flag `--v4-only` được khóa trong service spec/run lock. Terminal evaluator dispatch sang `candle_v4_only_eval.py`, chỉ tính continuous-router native fallback rồi expert fusion thành **V4 prediction**. Fallback/backbone là thành phần nội bộ của V4, không phải một lần train/eval phương pháp đối chứng riêng.

Không forward native multiclass baseline; không chạy continuous-native ablation readout; không có Accuracy/F1 đối chứng trong report. Không chỉ ẩn cột: các lượt inference đối chứng được bỏ thực sự. Không tính rescue/break relative baseline.

Giữ nguyên inference V4 và tập TEST: chỉ sau T5/R19/consolidation, 34 classes, disjoint shards seed523687 trên các client, mỗi row tính một lần. Pooled Accuracy/Macro-F1, per-class recall/F1/confusion matrix, overall/per-client router accuracy, expert trigger/override counts, timing và thông tin bank được lưu. Không đổi sang mỗi client chạy full TEST.

## Output

`/kaggle/working/results_candle_v4_only_seed42`:

- Round continuation và task checkpoint ZIP theo cơ chế có sẵn.
- `candle_v4_development_report.json`: expert/training/communication; per-task quality metrics NOT_MEASURED, không validation.
- `candle_v4_final_test.json`: `metrics.V4`, per-client V4 metrics và router metrics.
- `candle_v4_completion.json`: completion cùng final TEST V4 metrics.

Không cần results16/17 hoặc link Drive. Save Version để giữ kết quả. Notebook repair và các notebook train/resume cũ vẫn giữ nguyên.

## Kiểm tra

18/18 technical regression PASS: V4-only4, legacy terminal3, bootstrap11. Synthetic fixture xác nhận `metrics.V4` khớp đúng `coverage` của evaluator cũ trên cùng inputs; chỉ một V4 routing pass, baseline mock phải không được gọi. Kiểm tra terminal boundary, overwrite protection, report không baseline keys, config/requirements cùng budget.

Protected V1/shared helper SHA256 không đổi. Chưa chạy full training hoặc real TEST với notebook mới trong bước tạo file. Thay đổi này bỏ comparator computation, không phải thuật toán tăng Accuracy mới.
