# Bản sửa hierarchical/routerless — train mới

## Thay đổi giao thức

- Không gửi dữ liệu TRAIN thô hoặc replay mẫu giữa client/cluster.
- Mỗi client là worker tạm, tải reference của round rồi gửi delta cho temporary server.
- Trong cluster, cộng tử số delta có trọng số và mẫu số theo tọa độ. Giữa các
  temporary server, cộng các thống kê này rồi chia một lần; không trung bình đều các cluster.
  Lớp fc2 dùng số mẫu theo lớp, backbone dùng số mẫu đóng góp. Không ưu tiên trọng số leader.
- Một temporary server trong số các leader được chọn theo số lớp đã học cục bộ,
  hòa thì ID nhỏ nhất, để điều phối phép tổng hợp round. Không thêm server cố định.
- BN dùng count/sum/squared-sum từ TRAIN, không gửi activation/mẫu thô.
- Reference tổng hợp được đồng bộ cho tất cả cluster, kể cả nhóm dormant để khi
  quay lại không mất kiến thức. Model giữa các cluster giống nhau sau round.
  Cluster lúc này tổ chức truyền/tổng hợp, không duy trì các chuyên gia độc lập.
- Phân loại trực tiếp các lớp đã học (cumulative CE); không dùng router, không có
  binary memory toàn 0 và không cần task label lúc inference.
- Kiểm tra coverage mỗi round: thiếu lớp trong các task đã đến sẽ dừng và ghi lỗi,
  không giả vờ đăng ký các lớp chưa nhận cập nhật học thành công.

Đây là biến thể mới, không phải kết quả của giao thức DENICE cũ. Chi phí trao đổi
giữa server tăng thêm; không khẳng định tiết kiệm tổng communication hoặc tăng accuracy
trước khi đo. communication.json là chi phí payload mô phỏng, không đo TCP/TLS.
Không sửa/test-tune trên tập TEST. Chất lượng phân loại/forgetting vẫn cần đánh giá sau train.

## File cần chạy trên Kaggle

1. Mở `output/kaggle_hierarchical/train_incremental_kaggle.py` tại repo.
   Copy toàn bộ file vào một cell Kaggle, hoặc upload file rồi chạy bằng Python.
   Đây là file đã nhúng source mới; không dùng launcher cũ clone revision 78a07de.
2. Lần đầu để `RESUME_DRIVE_URL = ""`. Chọn GPU T4 x2, Internet, dataset 100-clients.
   Kiểm tra DATA_DIR. Output mới: `/kaggle/working/results_denice_hierarchical_seed_42`.
3. Save & Run All. Train Task 0–5, 20 round/task, một epoch/client/round, batch 2048.
   TRAIN không đọc TEST. Không được resume ZIP từ lần cluster_server cũ.
4. Nếu hết thời gian Kaggle: tải output, ZIP và upload Drive; dùng cùng file launcher,
   điền link mới vào RESUME_DRIVE_URL rồi chạy tiếp. Giữ nguyên file launcher/phiên bản.
5. Chỉ khi completion.json có completed=true, tasks_completed=6 mới eval.

## Eval sau train

ZIP toàn bộ output train (phải giữ `denice_cluster_server_source.zip`, provenance,
completion.json, checkpoint Task 5). Upload Drive.
Copy file `output/kaggle_hierarchical/eval_selective.py` vào một cell Kaggle mới,
điền link ZIP mới vào EVAL_DRIVE_URL và giữ `DIAGNOSTIC = False`.
Gắn dataset 100-clients, GPU, Internet rồi chạy.
Launcher lấy đúng source từ ZIP output, kiểm tra hash và torch trước khi eval.
Mỗi temporary server được eval trên toàn bộ TEST. Vì model đã đồng bộ nên các
cluster dự kiến có metrics giống nhau; đây là hành vi của giao thức mới.

## Nhận biết đang chạy đúng

Log train có `prediction=routerless_v1`, dòng `Hierarchy T...` (CPU) hoặc GPU worker,
receipt aggregation có `hierarchical_class_sample_count_masked_delta_v1`.
File `coverage_task_*_round_*.json` ghi lớp đã học và missing_classes=[];
sau Task 5 phải có đủ lớp 0–33. Không có kết quả TEST trong output train.

Kết quả cũ vẫn giữ để so sánh; không cần xóa hay ghi đè thư mục cũ.

## Kiểm thử đã thực hiện

- 68 kiểm thử liên quan (tổng hợp, routerless, launcher, chẩn đoán và hierarchy) đạt.
- Resume giữa round cho cùng fingerprint model và cùng communication ledger.
- Tổng hợp hai tầng với cluster có số mẫu khác nhau khớp phép tổng hợp phẳng.
- Launcher nhúng đã kiểm tra giải nén, SHA-256 và biên dịch 166 file Python.
- Đây chưa phải kết quả accuracy trên dữ liệu thực 100 client; cần train và eval mới.
- Smoke test đủ 6 task, mỗi task 1 round trên dữ liệu tổng hợp: coverage 34 lớp, dormant/rejoin và eval riêng 102/102 mẫu đều đạt.
