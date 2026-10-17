# Chạy bằng một file trên Kaggle

Bản cập nhật launcher: 17/10/2026.

1. Upload bản mới `train_incremental_kaggle.py` vào notebook Kaggle.
2. Bật Internet và GPU. Chạy trong session mới để tránh biến môi trường/import cũ.
3. Chạy file như trước. Không cần upload JSON hoặc gọi `tools/run_denice_recovery.py`.

Nếu không có cấu hình môi trường riêng, launcher mặc định dùng `row_overlay`, tải supporting source từ commit `d802d46944b2628a079b6238395c915d1637fbcb`, rồi nạp JSON từ checkout đã tải. File launcher được upload vẫn điều khiển cấu hình; nó không chạy lại launcher cũ trong checkout. Git LFS checkpoint không được tải.

Log đúng: `DENICE launch profile: row_overlay`, `effective_method: row_overlay`, `denice_appliance_enabled: True`, `appliance_enabled: False`, `denice_router_mode: multiclass_balanced`, `denice_amp_enabled: False`, `denice_clustering_mode: adaptive_ap`. Khi train có `APPLIANCE overlay: leaders=..., packets=..., accepted=...`.

Đường dẫn dataset mặc định: `/kaggle/input/datasets/khoilv2005/100-clients/100-clients`. Output mặc định bắt đầu bằng `/kaggle/working/results_denice_row_overlay_seed_42`; runner có thể thêm hậu tố cho fresh run. Chạy task 0–5 từ đầu; validation cục bộ bật, test cuối giới hạn 50.000 mẫu, mọi client active được đánh giá. Đây là mô phỏng FL, không phải mạng nhiều máy thật.

Các launcher đối chứng cũ vẫn có quyền cấu hình qua `DENICE_CONFIG_FILE`/`DENICE_CONFIG_OVERRIDES`; khi các biến này có giá trị, profile mặc định là legacy để không trộn thí nghiệm. Có thể chọn baseline rõ ràng bằng `DENICE_RUN_PROFILE=legacy`. Vì vậy khi muốn chỉ upload và chạy mặc định mới, hãy dùng session sạch, không chạy cell thiết lập cấu hình cũ.

Kiểm tra cấu hình mà chưa train: đặt `DENICE_CONFIG_ONLY=1` rồi chạy file; bỏ biến này trước khi train. Đã kiểm tra ba trường hợp: mặc định mới, legacy tường minh và override đối chứng. Chưa có kết quả train 100 client cho cấu hình mới; đổi phương pháp không bảo đảm tăng điểm.
