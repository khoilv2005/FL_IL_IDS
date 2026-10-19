# Tiếp tục row overlay sau timeout tại task 4

## Bằng chứng từ lần chạy 20261017_082002

Đã đọc trực tiếp `continuation_state_task_3.pt` trong thư mục kết quả người dùng cung cấp:

- `meta.completed_task=3`, `resume_from_task=4`, `boundary=task`; 98 model client tích lũy.
- Task 0–3 đã hoàn thành round 19. Task 3 có 80 client hoạt động.
- `appliance_rounds.jsonl` ghi đến task 4 / round 7; `optimizer_health.jsonl` đến task 4 / round 8.
- 6.001 bản ghi optimizer: 0 `zero_update`, 0 skipped optimizer steps.
- Chỉ có `checkpoint_task_4_base.pt`, chưa có continuation task 4. File base không đủ để tiếp tục giữa task.
- Test Accuracy/F1 là `null`: `denice_eval_final_task_only=true` và `eval_reason=not_final_task`. Không thể kết luận chất lượng test từ train loss 0,09035 ở task 3.

Giải pháp hiện tại: giữ task 0–3, chạy lại task 4 từ đầu trong phiên riêng, sau đó task 5 trong phiên khác. Không giảm rounds, clients, hay bật AMP để chữa timeout. Chưa có số đo đủ để bảo đảm riêng mỗi task dưới 12 giờ. Đây là hỗ trợ resume tại ranh giới task, chưa phải resume mỗi round hoặc tự dừng trước hạn giờ.

## Mã đã thay đổi

- Launcher nhận `DENICE_RESUME_STATE_PATH`, `DENICE_TASK_END`, `DENICE_DATA_DIR` và cho phép overlay resume.
- Không dùng ngầm checkpoint Google Drive mặc định của legacy cho overlay.
- Runner giữ checkpoint được chọn thay vì profile ghi đè thành fresh run.
- Checkpoint mới lưu `denice_appliance_state`: validation banks riêng, tập lớp đã học và số mẫu theo lớp của từng client. Ghi qua file tạm rồi rename.
- Checkpoint cũ thiếu state này: phải bật `DENICE_OVERLAY_REBUILD_PRIVATE_STATE=1`, đọc lại các train NPZ lịch sử, áp dụng cùng cap, thứ tự client, seed và chia validation để dựng lại state. Không huấn luyện lại model và không đọc test trong bước dựng bank.
- Kiểm tra số mẫu, histogram lớp, tập client mỗi task với audit đã lưu; từ chối sai seed/partition hoặc thiếu audit. RNG được bảo toàn quanh recovery.
- Histogram không chứng minh features/thứ tự dòng giống nhau. Bắt buộc dùng nguyên phiên bản dataset và metadata của lần chạy gốc; không shuffle, chia lại hay tiền xử lý lại dữ liệu.
- `overlay_recovery.json` ghi phương thức phục hồi. Không tự vô hiệu hóa overlay khi tiếp tục checkpoint overlay.

## Một file code để upload

Chạy ở repo nếu cần đóng gói lại sau sửa mã:

```powershell
python tools/package_denice_single_file.py
```

Upload **`output/single_file/train_incremental_kaggle.py`**, không nhầm với launcher ở thư mục gốc. File sinh ra chứa ZIP source được mã hóa, có kiểm tra SHA256, giải nén vào thư mục tạm và dùng mã này. Không cần clone GitHub, không cần upload riêng `fed_learning`. Không chứa dataset hoặc checkpoint.

Thêm vào notebook:

1. Dataset **100-clients đúng phiên bản cũ**.
2. `continuation_state_task_3.pt` từ output cũ, qua một input dataset riêng hoặc output notebook đã lưu. Không cần upload toàn bộ checkpoint task 0–3.
3. File Python đóng gói ở trên.

Checkpoint chứa estimator scikit-learn được lưu với **1.6.1**. Dùng cùng phiên bản, cài trước khi import; nếu đã import phiên bản khác, khởi động lại kernel. Cần torch, numpy, pandas, scikit-learn, matplotlib, seaborn, tqdm theo môi trường training hiện có; phiên bản torch/numpy gốc chưa được xác minh từ checkpoint.

```python
%pip install -q scikit-learn==1.6.1
```

Phiên chạy **task 4**:

```python
import os
from pathlib import Path

# Bỏ cấu hình thí nghiệm cũ có thể ghi đè profile.
for key in ('DENICE_CONFIG_FILE', 'DENICE_CONFIG_OVERRIDES', 'DENICE_VARIANT', 'DENICE_CONFIG_ONLY'):
    os.environ.pop(key, None)

paths = list(Path('/kaggle/input').rglob('continuation_state_task_3.pt'))
assert len(paths) == 1, f'Cần chọn đúng một checkpoint task 3: {paths}'
os.environ.update(
    DENICE_RUN_PROFILE='row_overlay',
    DENICE_TRAIN_PHASE='4',
    DENICE_TASK_END='4',
    DENICE_SEED='42',
    DENICE_RESUME_STATE_PATH=str(paths[0]),
    DENICE_OVERLAY_REBUILD_PRIVATE_STATE='1',
    DENICE_OUTPUT_DIR='/kaggle/working/overlay_resume_task4',
    DENICE_DATA_DIR='/kaggle/input/datasets/khoilv2005/100-clients/100-clients',
)
assert Path(os.environ['DENICE_DATA_DIR'], 'metadata.json').is_file()
# Thay đường dẫn dưới bằng vị trí file đóng gói vừa upload.
%run /kaggle/input/denice-resume-code/train_incremental_kaggle.py
```

Kiểm tra log `completed task 3, resume at task 4`, `Restored private overlay banks: task 0...3`, `Row overlay restored`. Phiên này chỉ chạy task 4. Sau khi hoàn tất, giữ toàn bộ Output, đặc biệt `continuation_state_task_4.pt` và `overlay_recovery.json`. Accuracy/F1 test vẫn có thể null vì chưa tới task cuối.

Phiên **task 5**: thêm output mới làm input. Giữ cùng file code/dataset/seed và cell trên, thay:

```python
paths = list(Path('/kaggle/input').rglob('continuation_state_task_4.pt'))
assert len(paths) == 1, paths
os.environ.update(
    DENICE_TRAIN_PHASE='4',  # Nhóm tiếp tục cuối; checkpoint xác định task bắt đầu thực tế.
    DENICE_TASK_END='5',
    DENICE_RESUME_STATE_PATH=str(paths[0]),
    DENICE_OVERLAY_REBUILD_PRIVATE_STATE='0',  # Checkpoint mới đã có bank.
    DENICE_OUTPUT_DIR='/kaggle/working/overlay_resume_task5',
)
%run /kaggle/input/denice-resume-code/train_incremental_kaggle.py
```

Log phải báo `completed task 4, resume at task 5`, `Row overlay restored: checkpoint`. Task 5 thực hiện đánh giá cuối theo cấu hình đang có (tối đa 50.000 mẫu/client; không được gọi là full-test nếu giới hạn này có hiệu lực). Báo cáo: `final_report.json`, `task_metrics.json`, `training_history.json`; log protocol/optimizer mới chỉ gồm phiên mới, cần giữ cả output cũ để phân tích toàn bộ thí nghiệm.

## Kiểm thử và giới hạn

Unit tests kiểm tra reconstruction bằng state tạo liên tục, lớp xuất hiện lặp lại, cap mẫu/client, client rỗng, clone không alias, bảo toàn RNG và từ chối các trường hợp resume không hợp lệ. Test launcher kiểm tra profile không xóa đường dẫn checkpoint hoặc khoảng task yêu cầu.

`tools/smoke_denice_overlay_resume.py` chạy runner trên dữ liệu tổng hợp, lưu checkpoint, tiếp tục bằng state mới và bằng state cũ thiếu bank. Đây là kiểm thử tích hợp, không phải kết quả IDS. Chưa chạy lại đầy đủ 100 client trên GPU Kaggle hoặc xác nhận thời lượng mỗi task. Không khẳng định training sau resume giống từng bit với một phiên liên tục trên môi trường khác.

Kết quả kiểm tra tại máy local ngày 19/10/2026: **23 tests passed** cho ba file `test_denice_overlay_resume.py`, `test_denice_appliance.py`, `test_denice_single_file_launcher.py`. Smoke test hai đường resume **passed**, log tại `output/overlay_resume_bv598thf/training.log`, tóm tắt tại `summary.json` trong cùng thư mục. File đóng gói đã chạy đến in effective configuration với `DENICE_CONFIG_ONLY=1`, profile row_overlay; không tải GitHub.

Checkpoint này chứa dữ liệu cục bộ (replay và validation), không chỉ trọng số. Giữ input/output chứa checkpoint ở chế độ riêng tư; không công bố như một model artifact đã khử dữ liệu. Đây là mô phỏng một tiến trình, không cung cấp bảo mật hạ tầng FL, secure aggregation hay differential privacy.
