# CANDLE V4 — sửa donor handoff theo actual graph

Đã triển khai protocol `candle_v4_current_graph_handoff_v6` để expert không mắc kẹt ở primary donor đã rời graph. Giữ donor khi còn hợp lệ; nếu không, chọn donor ID nhỏ nhất đang có positive-alpha edge, active trong current task, ≥32 positive TRAIN và negative pool hợp lệ. Không có donor thì tạm dừng. Không dùng graph tương lai hoặc replay dữ liệu task cũ.

Handoff giữ nguyên expert model và Adam object/state, steps, quota, byte counters. Giữ44 updates/round và catch-up tối đa88. Lịch sử lưu initial donor, lineage đổi donor và authorization/counts theo round; training proof chấp nhận nhiều donor khi từng lượt học có provenance hợp lệ. Restore dùng latest donor, không ghi lại lịch sử thành donor cuối cùng.

Giữ100% current-task TRAIN, validation0%, không BASE/FIT/CAL, không shadow native model và không acceptance gate. Scoped fusion vẫn threshold0,5 và native fallback. Expert task cũ frozen. Notebook V4 mới chọn `multiclass_balanced`; binary vẫn là option explicit. Notebook V1 hiện tại không thay đổi.

Logging: summary V4 ở round0 và mỗi5 rounds, gồm tổng handoffs của task; không in từng expert/mẫu/gradient. Chi tiết provenance ở checkpoint và training-history JSON. Lựa chọn donor dùng cached class counts, không scan lại toàn bộ tensors cho từng candidate.

## Regression

**21/21 PASS**, toàn bộ dùng dữ liệu synthetic:

- 7 donor-handoff tests: weights/Adam không reset; giữ donor không churn; từ chối donor ngoài graph/31 positives; resume trước và sau handoff khớp model/optimizer/quota/bytes/provenance (không so wall-clock runtime); catch-up88 và historical freeze; từ chối graph/lineage/count bị sửa; streaming multiclass fit/restore và router/config binding.
- 11 full-TRAIN/provisional/scoped regression tests.
- 3 terminal evaluator tests, chỉ synthetic TEST.

Receipts: `artifacts/candle_v4_handoff_regression_20261024.json`, `artifacts/candle_v4_handoff_original_regression_20261024.json`, `artifacts/candle_v4_handoff_terminal_regression_20261024.json`.

Không chạy full training hoặc mở real TEST/CAL trong regression này. Không hạ threshold hoặc áp thêm ngưỡng số updates để chọn expert. Chưa chứng minh handoff tăng accuracy. **42,91% /23,04% là kết quả checkpoint readout trước khi sửa handoff**, không phải benchmark mới.

Notebook `train_candle_v4_kaggle.ipynb` dùng output mới `results_candle_v4_handoff_multiclass_seed42`, seed42,6tasks×20rounds; fresh run, không resume checkpoint v4/v5. Full TEST vẫn chỉ ở T5/R19 sau consolidation. Protocol mới và router được khóa trong source/config/run lock.

Routing/coverage vẫn là nút thắt riêng: checkpoint trước có trần45,52% theo task routing. Sửa handoff giải quyết learning continuity, không tự giải quyết giới hạn đó.
