# DENICE overlay: launcher nhỏ, tiếp tục task 4–5

## Cách chạy hiện tại

Upload `train_incremental_kaggle.py` ở thư mục gốc. Bản sao trong
`output/single_file/train_incremental_kaggle.py` giống hệt. File khoảng 49 KB,
không còn mã nguồn nhúng hoặc chuỗi base85 lớn. Không cần cell cấu hình mới.

Giữ dataset 100-clients đúng phiên bản của lần chạy trước và bật Internet trên
Kaggle. Launcher dùng `resume_drive_url` có sẵn trong file để tải ZIP chứa đúng
một `continuation_state_task_3.pt`, kể cả checkpoint nằm trong thư mục con.
Không lấy checkpoint tồn dư từ lần tải trước. Link Drive cần cho phép tải bằng
link. Việc tải link Drive thực tế chưa được kiểm chứng tại local.

Mặc định:

- Profile `row_overlay`, seed 42, phase 4.
- `task_start=4`, `task_end=5`: chạy task 4 rồi task 5.
- Phục hồi private validation bank cho checkpoint cũ từ nguyên dataset ban đầu.
- Giữ 20 rounds/task và AMP tắt theo cấu hình phương pháp đang dùng.
- Sau task 4 lưu `continuation_state_task_4.pt`; sau task 5 lưu checkpoint và
  kết quả đánh giá cuối theo cấu hình hiện có.

Không cần upload riêng thư viện Python. File tải source từ GitHub và checkout
chính xác commit `d074f0e8cf4870e26150975b9e9df988cbe425f9`. Đã kiểm tra bằng
`git ls-remote`: commit này có trên origin/main. Commit chứa module
`fed_learning/training/denice_overlay_resume.py`, hỗ trợ khôi phục validation
banks, knowledge và class counts. Bản launcher lỗi trước đó vẫn ghim vào
`d802d46…`, nơi còn chặn resume overlay.

Log mong đợi:

```text
Training source: /tmp/denice-source-...
Overlay resume support: True
Restoring DeNICE continuation: completed task 3, resume at task 4
Row overlay restored: reconstructed_from_identical_dataset_required
```

Đường dẫn `/tmp/denice-source-...` là bình thường với bản nhỏ tải GitHub này.
Điều quan trọng là checkout đúng commit mới và `Overlay resume support: True`.
Không xóa điều kiện kiểm tra cấu hình của phương pháp.

Nếu notebook có biến môi trường DENICE_* cũ, chúng vẫn có quyền ghi đè cấu hình
trong file. Dùng phiên kernel mới không có override cũ để áp dụng mặc định trên.
`DENICE_CODE_DIR` hoặc `DENICE_CODE_REF` chỉ định source cũ sẽ bị từ chối nếu thiếu
module resume. Dataset/metadata không được chia lại hoặc đổi thứ tự dòng: audit
sample counts/histogram không chứng minh được features và row order bất biến.

## Giới hạn và checkpoint

Checkpoint gốc xác nhận hoàn thành task 3 round 19. Log task 4 đến overlay round 7
và optimizer round 8, nhưng không có continuation đủ trạng thái ở task 4. Vì thế
phải chạy task 4 lại từ đầu, không chạy lại task 0–3.

Accuracy/F1 của output cũ là null vì cấu hình chỉ đánh giá test ở task cuối,
không phải điểm 0. 6.001 optimizer records không ghi nhận zero-update hoặc
skipped optimizer steps. Chưa có bằng chứng bảo đảm tổng task 4–5 sẽ dưới giới
hạn 12 giờ. Bản sửa này chữa lỗi source/resume, không tăng tốc training. Nếu phiên
mới bị ngắt trong task 5, cần giữ checkpoint task 4 để tiếp tục từ đó. Resume hiện
ở ranh giới task, chưa hỗ trợ tiếp tục round đang chạy dở của overlay.

Checkpoint chứa replay/validation data, cần giữ riêng tư. Mô phỏng một tiến trình
không cung cấp secure aggregation hay differential privacy. Estimator trong
checkpoint cũ được lưu bằng scikit-learn 1.6.1; nên giữ môi trường tương thích.

## Kiểm thử

11 tests launcher đã qua: cấu hình mặc định chạy 4–5, tải/checkout đúng revision
(giả lập lệnh Git), giữ checkpoint khi áp dụng profile, tải ZIP Drive giả lập,
từ chối ZIP sai/không rõ ràng, từ chối source thiếu resume và file dưới 100 KB.
Source runtime resume không đổi so với bản đã qua smoke test hai đường resume
trên dữ liệu tổng hợp. Chưa chạy lại toàn bộ 100 client trên Kaggle.

Cập nhật bản sao launcher sau khi chỉnh file gốc:

```powershell
python tools/package_denice_single_file.py
```

Công cụ chỉ sao chép file nhỏ, không nhúng source trở lại. Nếu sửa thư viện hỗ
trợ, cần đưa commit đó lên GitHub và cập nhật DEFAULT_CODE_REF trước khi chạy
standalone. Các thay đổi launcher local không yêu cầu push vì người dùng upload
trực tiếp file launcher; revision hỗ trợ nói trên đã có trên GitHub.
