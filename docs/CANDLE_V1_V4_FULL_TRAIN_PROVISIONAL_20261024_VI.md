# V1/V4: 100% TRAIN, không local validation, provisional activation

Đã khóa protocol `candle_v4_full_train_provisional_v3` cho run mới.
`candle_v4_runtime/configs/candle_v1.json` đặt validation fraction0.0 và clean
roles null. Native V1 và V4 cùng dùng toàn bộ TRAIN gốc thuộc task hiện tại;
không chia/đọc BASE/FIT/CAL/validation. Expert lấy mẫu từ đúng view TRAIN đó,
không cap4096/class và không dùng raw data task cũ để học tiếp.

Không có bản V1 shadow. Notebook V1 chạy native-only, không tạo expert;
notebook V4 có một backbone và bank coverage-aware/client. Cả hai cùng source,
data protocol, router, xi=.8, seed42,6tasks x20rounds,batch2048/AMP.

Expert có update thực, provenance/graph hợp lệ, steps khớp và weights hữu hạn
thì ACTIVE cuối task. Receipt ghi `training_only_unverified`,
`quality_status=UNVERIFIED`. Threshold0.5. Không còn gate recall95%/FAR0.1%/
break0/net rescue và không tính những số đó trên TRAIN để thay validation.
Các metrics trước TEST ghi NOT_MEASURED/null. Không update hoặc trạng thái
huấn luyện không hợp lệ thì DEFERRED. Expert cũ giữ nguyên weights và tiếp tục
ACTIVE qua task mới; native fallback dùng khi không expert nào vượt threshold.

ACTIVE này không khẳng định an toàn hoặc chất lượng. False activation, break
và lợi ích thực tế chỉ được đo sau huấn luyện; TEST không quyết định activation.
Full TEST34class giữ sau Task5/Round19 và consolidation; không TEST trung gian.

Hai file chạy:

- `train_candle_v1_kaggle.ipynb` → `results_candle_v1_full_train_seed42`.
- `train_candle_v4_kaggle.ipynb` → `results_candle_v4_full_train_seed42`.

Phải dùng notebook/source mới và chạy fresh. Không đổi code giữa phiên,
không resume checkpoint có validation/roles/protocol cũ. Lịch sử V1 cũ vẫn
có trong các commit trước; không so trực tiếp với số V1 dùng80/2/8/10.

Regression dùng NPZ tổng hợp để kiểm tra đủ số mẫu TRAIN, không truy cập
validation, update thật, activation UNVERIFIED, native/RNG isolation, checkpoint
resume, graph causality, frozen retention, inference/fallback và final TEST.
Không chạy full training hoặc mở TEST thật local khi cập nhật.
