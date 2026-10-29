# CANDLE V4 — sửa thiếu continuous state khi exact bootstrap

## Thay đổi

Giữ CANDLE V4, thêm protocol nội bộ `candle_v4_continuous_bootstrap_v8` (không đổi tên phương pháp thành CANDLE V8). Legacy v7 giữ nguyên để tái lập results16/17 và notebook resume đã phát hành.

Trong private V4 runner, ngay sau khi clone nguyên model và restore native context detector từ representative, hook V8 copy thêm continuous feature memory, scaler, class LR, selected indices và class/task mapping của **chính source tại thời điểm đó**. Copy qua serialized state để receiver/source không alias. Không dùng thêm samples, encoder forward hoặc optimizer step.

Hook chỉ nằm ở `representative_clone`, không gọi từ `rejoining_plastic_catch_up`, local training hay native aggregation. Source/target model weights, connection masks và native detector phải khớp; routing indices cũng khớp. Không copy memory giữa các encoder chỉ vì shape giống nhau. Nếu support của continuous không khớp native memory trong V8, báo lỗi correctness thay vì âm thầm tạo bank rỗng.

Mỗi clone ghi receiver/source/task, memory digest, classes/tasks và serialized bytes. V4 service checkpoint lưu provenance và byte counters. Bootstrap bytes được cộng vào application-wire budget/total của coverage arm; không giả định state transfer là miễn phí. Không log thêm mỗi bước train.

## Những phần giữ nguyên

- Không sửa root `fed_learning/**`, V1 config/notebook hoặc private shared ContextDetector/sampler/routed evaluator/checkpoint helpers; SHA256 kiểm chứng trước/sau.
- 100% current-task TRAIN, validation0%, không BASE/FIT/CAL, không historical raw replay, không shadow model.
- Native loss/optimizer/batch2048/số steps, graphξ0,8, expert architecture và positive/negative learning, threshold0,5/scoped fusion không đổi.
- 6 tasks×20 rounds, seed42. Cùng disjoint TEST partition seed523687, pooled metrics, chỉ terminal T5/R19 sau consolidation. Không thêm quality gate hay dùng TEST chọn cấu hình.

## Kiểm tra

**41/41 PASS** trên synthetic fixtures; không full training hoặc real TEST inference:

| Nhóm | Tests |
|---|---:|
| V8 exact clone / chain / no alias / bytes / resume / real initialized CNN-GRU |11|
| Continuous router v7 regression / native isolation |9|
| Donor handoff |7|
| Original TRAIN / scoped expert |11|
| Terminal evaluator |3|

V8 test dùng chính `_bootstrap_denice_model()` với CNN-GRU thật đã khởi tạo; thu memory T0, clone sang late join T5, thêm memory T5 và xác nhận T0 vẫn tồn tại. Kiểm tra routed inference trước/sau restore khớp. Đây không phải smoke full native training hoặc kiểm chứng chất lượng.

Replay **support sets** theo actual bootstrap/local-task metadata của results17: sau bổ sung exact-clone memory,100/100 client có task support khớp native. Không fit lại classifier hoặc chạy TEST trong replay; không suy ra accuracy sẽ phục hồi toàn bộ.

Receipts: `artifacts/candle_v4_bootstrap_fix_regression_20261026.json`, `artifacts/candle_v4_bootstrap_isolation_20261026.json`. Legacy regression receipts giữ nguyên.

## Notebook và checkpoint cũ

Notebook riêng `train_candle_v4_bootstrap_fixed_kaggle.ipynb`; launcher thêm `--continuous-bootstrap`, chọn protocol V8. Notebook V1, V4 multiclass, V4 continuous v7 và resume results16 không bị ghi đè.

Fresh run dùng output `results_candle_v4_continuous_bootstrap_seed42`, RESUME_STATE rỗng. Checkpoint V7 không được đổi nhãn thành V8. Nếu cần dùng lại checkpoint cũ, phải có migration riêng theo provenance đúng thời điểm và đầy đủ bootstrap chain; chưa triển khai migration trong bản sửa này.

Đã sửa nguyên nhân state bị bỏ sót cho run mới. **Chưa có accuracy sau fix**, không gọi48,26% là kết quả hoặc dự báo. Không chạy full local hay tự bắt đầu Kaggle job.
