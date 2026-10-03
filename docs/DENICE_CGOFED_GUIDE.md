# Chạy biến thể DeNICE + CGoFed

Launcher mới: `train_denice_cgofed_kaggle.py`. Cấu hình được tạo tại `configs/denice_cgofed.json`; nguồn cấu hình chuẩn vẫn là `variant_preset("cgofed")` trong `fed_learning/strategies/incremental/denice_variants.py`.

Chạy với preset mặc định:

```powershell
$env:DENICE_VARIANT = 'cgofed'
python train_denice_cgofed_kaggle.py
```

Preset giữ `algorithm=denice`, `mode=decentralized`, tắt replay/EWC, train trên toàn bộ lớp đã thấy, mở mature `fc2.weight` cho local training và dùng projection riêng của từng client. Mặc định dùng Adam với `mu=0.5`, decay `0.8`, giữ tối đa 64 chiều basis mỗi task, tối đa 512 activation train mỗi client/task. Mature bias và mature encoder vẫn frozen. Có thể đổi optimizer tham chiếu sang `sgd_gradient` qua `DENICE_CONFIG_OVERRIDES`.

Ví dụ ghi đè hệ số projection:

```powershell
$env:DENICE_CONFIG_OVERRIDES = '{"denice_cgofed_mu": 0.75, "denice_cgofed_decay": 1.0}'
python train_denice_cgofed_kaggle.py
```

Log `training_history.json` có `cgofed_local_projection`, `cgofed_peer_projection` trong summary mỗi round và `cgofed_projection` ở task boundary. Các trường chính gồm số hàng được project, tỷ lệ update bị loại, hệ số mu, rank và năng lượng basis giữ lại. Kiểm tra `source_audit.json` và `config.json` để xác nhận đúng source và preset.

Preset Kaggle chỉ chạy test evaluation ở round cuối của task cuối. Test set hiện có là `global_test_data.npz`, không có test split gốc theo client; evaluation tạo các shard global disjoint, deterministic và phân tầng theo lớp cho những client được chọn. Mỗi mẫu test được dùng đúng một lần qua toàn bộ các client eval. Output giữ metrics từng client, sample count và class counts của từng shard. Đây là ước lượng client-wise từ global test, không phải phép đo trên test distribution riêng của từng client. Nomask, ensemble và báo cáo local-validation bị tắt trong preset này để giảm thời gian eval.

Bank activation được lấy từ train partition riêng của client sau consolidation ở cuối task, không chứa input thô. Full continuation checkpoint giữ bank FP32 cùng model state. Khi tiếp tục CGoFed từ task sau, runner yêu cầu bank hợp lệ trong checkpoint; checkpoint DeNICE cũ không thể chuyển thành continuation CGoFed vì không có lịch sử subspace.

Đây là bản đầu chỉ project classifier `fc2`. Chạy pilot qua ít nhất hai task để xác nhận bank được tạo và projection thực sự có hàng mature để tác động trước khi dùng full benchmark. Chưa có kết quả accuracy cho biến thể mới; cần so sánh `fc2` mở với `mu=0` và cùng cấu hình có `mu>0`, trên cùng seed, client cohort và router.
