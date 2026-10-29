# Upload một file và tiếp tục sau timeout

## File cần upload

Upload **`train_incremental_kaggle.py` ở thư mục gốc**. Bản sao tại
`output/single_file/train_incremental_kaggle.py` có nội dung giống hệt.
Source Python cần thiết đã được nhúng trong file, không cần upload riêng thư mục
`fed_learning` và không phụ thuộc revision GitHub cũ cho lần train mới.
Dataset, PyTorch và các thư viện trong `requirements-denice-v2-kaggle.txt` vẫn
phải có trong môi trường Kaggle. File này không nhúng thư viện hoặc dữ liệu.

**Link Drive hiện để rỗng theo chỉnh sửa của người dùng.** Chạy nguyên trạng sẽ
train mới nếu không có biến môi trường resume. Muốn continue, khai báo nguồn
checkpoint như hướng dẫn dưới đây.

Có thể chạy bằng `%run` hoặc dán **toàn bộ file** vào một cell Kaggle. Source nhúng
nằm trong hàm `_denice_support_payload`, được định nghĩa trước bootstrap; không
cần `__file__` hoặc lịch sử cell. Thay toàn bộ launcher cũ bằng bản mới, đừng chỉ
chép riêng hàm giải nén. Đã có kiểm thử chạy cell không có `__file__` và kiểm tra
source giải nén khớp source gốc; tổng 17 kiểm thử launcher/resume đạt sau bản sửa.

## Train mới với nén upload

Trong notebook Kaggle, dùng một kernel mới:

```python
import os
os.environ["DENICE_RESUME_DRIVE_URL"] = ""
os.environ["DENICE_RUN_PROFILE"] = "cluster_server"
os.environ["DENICE_DATA_DIR"] = "/kaggle/input/<dataset>/100-clients"
os.environ["DENICE_OUTPUT_DIR"] = "/kaggle/working/denice_compact_run1"
os.environ["DENICE_SESSION_SECONDS"] = "36000"  # ví dụ: 10 giờ cho training runner
os.environ["DENICE_STOP_MARGIN_SECONDS"] = "900"  # dự phòng 15 phút

# Thay bằng đường dẫn file đã upload của bạn:
%run /kaggle/input/<uploaded-code>/train_incremental_kaggle.py
```

Lần train mới mặc định dùng temporary server theo cụm, trao đổi hierarchical,
`routerless_v1` và nén upload không mất thông tin. Đây là nhánh đã triển khai ở
lượt trước, **không phải toàn bộ thiết kế DENICE-ACE**. Nhánh này không có route
F1 để so sánh; đừng hiểu nén là nâng cấp router. Muốn giữ nhánh V2 có router,
đặt `DENICE_SERVER_EXCHANGE=0`, `DENICE_COMPACT_UPLOADS=0`,
`DENICE_PREDICTION_MODE=cnn_coordinate_router_v1` trước khi chạy mới.

Các biến `DENICE_CODE_DIR`, `DENICE_CODE_REF`, `DENICE_SOURCE_ZIP` chỉ nên dùng
khi chủ động chọn source khác. Không dùng chúng để thay source giữa một lần train.

## Timeout và checkpoint

- Sau mỗi round hoàn tất, runner ghi checkpoint qua file `.partial`, rồi đổi
  tên thành `.zip`. File `.partial` không được chọn để resume.
- Ngân sách thời gian tính từ khi vào runner, **không gồm thời gian cài thư viện,
  tải Drive, dựng dữ liệu trước runner hoặc khôi phục checkpoint**. Chọn ngân sách
  thấp hơn thời gian phiên còn lại. Ví dụ 10 giờ không phải khẳng định giới hạn
  thời gian Kaggle của tài khoản bạn.
- Runner dự phòng thời lượng round dài nhất đã thấy cùng khoảng dự phòng, rồi
  dừng tại ranh giới đã lưu. Không ngắt optimizer giữa minibatch. Round đầu tiên
  của lần train mới phải hoàn tất mới có checkpoint; round đột ngột dài hơn dự
  kiến vẫn có thể bị Kaggle ngắt. Đây là dừng chủ động, không bảo đảm chống hard kill.
- Dừng chủ động ghi `session_status.json` với `status="paused"` và
  `completed=false`. Chỉ hoàn tất toàn bộ mới ghi `completion.json` thành công.
- Nếu bị cắt giữa round R, tiếp tục từ round đã lưu gần nhất và chạy lại R.
  Không mất các round trước đó nếu toàn bộ checkpoint phụ thuộc còn được giữ.

**Lưu toàn bộ thư mục output**, đặc biệt:

```text
denice_cluster_server_source.zip
execution_provenance.json
protocol_lock.json
checkpoint_task_*_round_*.zip
checkpoint_task_*_all_rounds.zip
```

Không chỉ lấy ZIP round cuối: checkpoint round có thể dùng blob của các round
trước trong cùng task. ZIP `all_rounds` của task đã seal là tự chứa.
Có thể bỏ `current_data_cache`/`test_cache` khi tạo ZIP output. Cần chủ động lưu
output qua cơ chế của Kaggle hoặc tải về trước khi mất phiên; code không thể
khôi phục checkpoint nếu nền tảng đã xóa toàn bộ file.

## Continue trong phiên mới

Gắn output lần trước thành Kaggle Dataset, dùng cùng dữ liệu gốc và kernel mới:

```python
import os
os.environ["DENICE_CLUSTER_RESUME_ROOT"] = "/kaggle/input/<saved-output>/<run-folder>"
os.environ["DENICE_DATA_DIR"] = "/kaggle/input/<dataset>/100-clients"
os.environ["DENICE_OUTPUT_DIR"] = "/kaggle/working/denice_compact_run2"
os.environ["DENICE_SESSION_SECONDS"] = "36000"
%run /kaggle/input/<uploaded-code>/train_incremental_kaggle.py
```

Hoặc thay `DENICE_CLUSTER_RESUME_ROOT` bằng `DENICE_RESUME_ZIP` trỏ tới ZIP
output đã gắn vào Kaggle. Hoặc dùng `DENICE_RESUME_DRIVE_URL` trỏ tới ZIP trên
Drive. Chỉ chọn một nguồn. Nguồn local được khai báo sẽ thay link mặc định trong
file; nếu đã đặt biến môi trường Drive khác thì phải xóa biến đó trước.

Launcher tự chọn task/round đã lưu mới nhất, phục hồi cấu hình nén, hierarchy,
router, batch size và worker devices. Source ZIP trong output được ưu tiên.
Runner kiểm tra hash source, runtime PyTorch, protocol và dữ liệu theo cơ chế
hiện có. RNG và trạng thái thuật toán được khôi phục từ checkpoint; đổi ngân sách
thời gian phiên không đổi protocol học. Output phiên mới phải khác thư mục
dataset chỉ đọc. Giữ cả output các phiên trước để có đủ checkpoint đánh giá lịch sử.

## Checkpoint cũ

Không bật nén/routerless vào checkpoint cũ đang dùng nhánh khác. Launcher phục
hồi cấu hình cũ và từ chối override xung đột. Nếu output cũ có source ZIP, dùng
đúng ZIP đó; nếu không có, launcher dùng revision được provenance ghi lại.
Một source cũ không có tính năng dừng theo ngân sách vẫn chỉ dùng checkpoint
round vốn có; cần bắt đầu run mới với bản này để dùng đầy đủ tính năng mới.

Sau khi sửa production source, chạy `rtk proxy python tools/package_denice_single_file.py`
để cập nhật source nhúng trước khi upload. Không chỉnh phần payload cuối file bằng tay.

## Kiểm chứng ngày 29/10/2026

15 kiểm thử launcher/bootstrap đạt: source nhúng khớp code hiện tại, bản upload
khớp bản gốc, truyền cấu hình nén/ngân sách, khôi phục cấu hình và source cũ,
từ chối override xung đột và bỏ qua file `.partial`.

Một kiểm thử native CPU (2 task × 2 round) đạt trong 348,43 giây: dừng chủ động
sau round đầu, resume từ checkpoint, rồi so sánh fingerprint trạng thái cuối và
sổ communication với chạy liền mạch — trùng khớp, không đọc TEST khi train.
Đây là dữ liệu fixture; chưa kiểm thử phiên Kaggle thật hoặc hard kill của nền tảng.
Môi trường Windows phát diagnostic ở import pyarrow có sẵn nhưng kiểm thử hoàn
tất; không sửa dependency để bỏ qua kiểm tra. Kiểm tra cú pháp và diff đều đạt.
