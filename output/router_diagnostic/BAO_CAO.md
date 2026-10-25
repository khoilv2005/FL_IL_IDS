# Chẩn đoán DENICE — 25/10/2026

Đã chạy thực tế trên checkpoint cuối Task 5, đủ 51 server, sử dụng cùng 2.176 mẫu TEST
(64 mẫu/lớp, 34 lớp, seed 42). Đây là tập con cân bằng lớp phục vụ chẩn đoán, không
phải kết quả full test và không dùng để chọn tham số. Không sửa checkpoint hay train lại.

## Kết quả

| Chỉ số trung bình theo server | Giá trị |
|---|---:|
| Accuracy dùng router hiện tại | 5,83% |
| Accuracy cung cấp đúng task (oracle) | 24,71% |
| Chênh lệch | +18,88 điểm phần trăm |
| Router chọn đúng task | 20,77% |
| Truy vấn có vector nhị phân toàn 0 | 63,11% |
| Hòa điểm cao nhất giữa nhiều task | 61,38% |

50/51 server cải thiện khi biết đúng task. Oracle tốt nhất đạt 39,94%, thấp nhất
2,94%; việc sửa router không tự động khắc phục khả năng phân loại còn hạn chế.

## Bằng chứng về hạn chế

- Cả 51 server đều thiếu lớp đăng ký: chỉ có 3–30/34 lớp, trung bình 19,82 lớp.
  Lớp không nằm trong episode_classes không thể được dự đoán bởi mask hiện tại,
  kể cả khi biết đúng task. Server 35 chỉ có 3 lớp; server 66 có 13 lớp.
- 38/51 server thiếu ít nhất một task trong activation_memory.
- 16 server có ít nhất một bank task toàn vector 0 (17 bank tổng cộng).
- Router dùng cosine với prototype trung bình. Truy vấn toàn 0 cho điểm cosine
  bằng nhau giữa các task hiện có; argmax chọn chỉ số đầu tiên trong các điểm cực đại.
  Đây là cơ chế cụ thể có thể tạo thiên lệch về task sớm. Chưa kết luận nguyên nhân
  tạo vector 0 là threshold, phân phối dữ liệu hay đặc trưng; cần audit tiếp trên TRAIN/validation.

## Môi trường và giới hạn

Mã nguồn gốc đã đối chiếu SHA-256 theo provenance. Chạy trên RTX 4050 Laptop,
PyTorch 2.5.1+cu121, khác bản train 2.10.0+cu128. Vì thế kết quả này là chẩn đoán sơ bộ.
Thông tin thiếu task/lớp/bank là từ checkpoint; các tỷ lệ suy luận cần xác nhận trên
runtime gốc. Oracle sử dụng nhãn task thật, tuyệt đối không báo là accuracy triển khai.
Không so trực tiếp 5,83% của tập cân bằng này với 5,61% của full test mất cân bằng.

## Việc tiếp theo

1. Xác nhận chẩn đoán bằng eval_selective.py trên Kaggle: DIAGNOSTIC=True,
   DIAGNOSTIC_PER_CLASS=64 để tái hiện cùng mẫu. Giữ link ZIP train đã hoàn tất.
   Launcher kiểm tra mã nguồn và khôi phục đúng torch gốc, ghi vào thư mục diagnostics riêng.
2. Audit threshold/binarization trên TRAIN và validation: tỷ lệ bit bật, bank 0,
   độ phân biệt prototype giữa task. Không chữa bằng đổi tie-break đơn thuần.
3. Xác định yêu cầu độ bao phủ: nếu mỗi server phải nhận diện đủ 34 lớp, cần cơ chế
   học/chuyển giao kiến thức lớp còn thiếu giữa cluster. Không chỉ mở mask các lớp
   chưa học vì trọng số có thể chưa được huấn luyện.
4. Sau khi xác định nguyên nhân, thử sửa trên validation trước khi train lại toàn bộ.

## Tệp

- results/diagnostic_summary.json: toàn bộ số liệu và runtime.
- results/diagnostic_<cluster_uid>.json: từng server, ma trận router, metrics oracle/routed,
  lớp đăng ký, thống kê bank, fingerprint checkpoint và binding mẫu.
- ../../eval_selective.py (tại thư mục gốc repository): file một cell dùng trên Kaggle.
- tools/diagnose_cluster_servers.py (tại thư mục gốc repository): worker chẩn đoán.

Các kiểm thử launcher và diagnostic: 10 test đạt, gồm xử lý tham số notebook,
worker nhúng, mask task đúng, task thiếu được tính sai và output diagnostic riêng.
