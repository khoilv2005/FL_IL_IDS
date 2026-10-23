# Bổ sung full TEST cuối CANDLE V4

Theo yêu cầu mới, notebook tự động đánh giá toàn bộ TEST sau Task5/Round19,
sau consolidation, gate và lưu task archive. Không chạy TEST ở task/round trước.
Điều này thay thế phần “không tự động TEST” trong protocol/notebook trước.

Giữ cumulative VALIDATION sau mỗi task. Full TEST chỉ là terminal readout:
không chọn donor, threshold, round, ACTIVE hay hyperparameters từ TEST.

Toàn bộ `global_test_data.npz` phải có đủ34class, không giới hạn số mẫu, không
loại class ngoài ownership/availability của receiver. Giữ cách chia shard TEST
disjoint và seed của V1; toàn bộ dataset được dùng đúng một lần cho mỗi nhánh.
V1, coverage-aware, no-aux và donor-only nhận đúng cùng các sample–receiver
assignments. Headline là pooled metrics, không phải mean-cluster accuracy hay
mỗi receiver chạy lại toàn bộ TEST.

Evaluator kiểm tra đủ6task, boundary[5,19], gate Task5 và không cho ghi đè kết quả.
Native model/router và toàn bộ expert state phải có cùng fingerprint trước/sau.
Chỉ ACTIVE expert được tham gia; DRAFT/REJECTED/SUSPENDED không override.

Kết quả: `results_candle_v4_seed42/candle_v4_final_test.json`, gồm Accuracy,
Macro-F1 đủ34class, recall/F1/confusion matrix, từng receiver, rescue/break và
inference time. `candle_v4_completion.json` chứa `final_validation`, `final_test`
và `TEST_opened: true`. Báo cáo development riêng vẫn mô tả phần VALIDATION.

Đây vẫn là readout của development seed42, chưa phải benchmark nhiều seed hay
bằng chứng confirmation độc lập của cả quá trình phát triển. Không chạy full
training hoặc đọc TEST thật local khi triển khai thay đổi này.
