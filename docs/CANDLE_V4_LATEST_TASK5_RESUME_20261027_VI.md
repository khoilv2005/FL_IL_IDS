# CANDLE V4 mới nhất: resume Task5 từ results16

Notebook: `resume_candle_v4_latest_task5_kaggle.ipynb`.
GPU + Internet, dataset `100-clients`, Run All. Tự tải results16 từ Drive;
có thể thay bằng ZIP gắn vào Kaggle Input.

## Phạm vi

- Chọn đúng continuation cuối Task4, SHA256
  `1336e1d91f4301bac73b78005c9ecd195bbd3ec36b9d4832e7b3b63967bed075`.
- Trích riêng continuation T0–T4, xác minh manifest/checksum; bỏ qua outer latest
  và mọi round Task5 đã chạy dở trong results16.
- Khôi phục continuous feature memory thiếu theo exact native clone lineage.
  Refit router LR trên saved TRAIN feature memory, không đọc raw data task cũ.
- Chuyển protocol v7 sang bootstrap-fixed v8 và V4-only evaluation bằng bước import
  tường minh. Không gọi đây là exact legacy resume hay fresh full training.
- Giữ backbone, native masks/router, experts/Adam, lifecycle, history và RNG.
- Train Task5 Round0–19; consolidation; chỉ eval V4 trên Full TEST 34 classes.
  Không train/eval baseline V1 riêng.
- Giữ seed42, batch2048, AMP, threshold0,5, coverage-aware expert updates,
  100% current-task TRAIN và validation0. Không quality acceptance gate.
- Không sửa V1 hoặc các notebook cũ.

Output mới: `/kaggle/working/results_candle_v4_latest_resume_task5_seed42`.
Xuất `task4_migration_receipt.json`, migrated Task4 continuation, run lock,
round checkpoints, Task5 archive và final TEST/completion.
Historical migration serialized bytes được ghi riêng; không giả thành traffic
đã truyền trong run T0–T4 gốc. Future bootstrap bytes bắt đầu từ 0.

## Kiểm tra đã thực hiện

- 6/6 migration regressions: archive boundary/checksum, compatibility rejection,
  native/RNG preservation, expert certificate identity, latest service restore,
  checkpoint roundtrip và notebook command.
- 4/4 V4-only evaluator regressions PASS.
- Preflight checkpoint Task4 thật: restore 99 clients, repair 49 routers thiếu
  inherited task memory. Native state/RNG và expert/Adam state giữ nguyên.
- Local sklearn1.9 chỉ dùng kiểm tra cấu trúc/restore; không dùng để báo cáo accuracy.
  Notebook bắt buộc pinned sklearn1.6.1 trước khi migration và training.
- Không mở raw TRAIN hoặc TEST, không chạy full training local.

Artifact: `artifacts/candle_v4_task4_migration_preflight_20261027.json`.
Chưa có kết quả train Task5/eval cho notebook mới; không cam kết accuracy vượt 50%.
